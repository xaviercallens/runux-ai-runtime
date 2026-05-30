"""
SymBrain v4 — Benchmark Evaluation Harness
═══════════════════════════════════════════
Runs curated test problems against a live /v4/solve endpoint,
computes accuracy with Wilson-score 95% confidence intervals,
and performs McNemar's test comparing against v3 baseline results.

Usage:
    python -m v4.eval.benchmark_runner \
        --server-url http://localhost:8080 \
        --tier all \
        --baseline-json results_v3.json \
        --output-dir ./eval_results
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import httpx

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════
#  Data models
# ══════════════════════════════════════════════════════════════════════


@dataclass
class BenchmarkResult:
    """Result of a single benchmark problem evaluation."""

    problem_id: str
    category: str
    query: str
    expected_answer: str
    actual_answer: str
    correct: bool
    latency_ms: float
    routing_decision: str  # e.g. "7B", "14B", "32B", "72B"


@dataclass
class BenchmarkSummary:
    """Aggregate summary for one benchmark run."""

    timestamp: str
    server_url: str
    tier: str
    total: int
    correct: int
    accuracy: float
    ci_lower: float
    ci_upper: float
    results: list[BenchmarkResult]
    mcnemar_p_value: float | None = None
    mcnemar_statistic: float | None = None
    category_breakdown: dict[str, dict[str, float]] = field(default_factory=dict)


# ══════════════════════════════════════════════════════════════════════
#  Statistical utilities
# ══════════════════════════════════════════════════════════════════════


def wilson_score_ci(
    successes: int,
    total: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """
    Compute the Wilson score confidence interval for a binomial proportion.

    Returns (lower, upper) bounds.  Handles edge cases (total=0, p=0, p=1).

    Reference:
        Wilson, E.B. (1927). "Probable inference, the law of succession,
        and statistical inference". JASA 22(158): 209–212.
    """
    if total == 0:
        return (0.0, 1.0)

    # z value for the given confidence level (two-tailed)
    # Using the approximation for common values; fall back to scipy if available
    z_map: dict[float, float] = {
        0.90: 1.6449,
        0.95: 1.9600,
        0.99: 2.5758,
    }
    z = z_map.get(confidence)
    if z is None:
        try:
            from scipy.stats import norm

            z = norm.ppf(1 - (1 - confidence) / 2)
        except ImportError:
            # Rational approximation (Abramowitz & Stegun 26.2.23)
            alpha = (1 - confidence) / 2
            t = math.sqrt(-2.0 * math.log(alpha))
            z = t - (2.515517 + 0.802853 * t + 0.010328 * t**2) / (
                1 + 1.432788 * t + 0.189269 * t**2 + 0.001308 * t**3
            )

    p_hat = successes / total
    z2 = z * z
    denominator = 1 + z2 / total
    centre = p_hat + z2 / (2 * total)
    spread = z * math.sqrt((p_hat * (1 - p_hat) + z2 / (4 * total)) / total)

    lower = max(0.0, (centre - spread) / denominator)
    upper = min(1.0, (centre + spread) / denominator)
    return (lower, upper)


def mcnemar_test(
    baseline_results: Sequence[BenchmarkResult],
    experiment_results: Sequence[BenchmarkResult],
) -> tuple[float, float]:
    """
    Perform McNemar's test on paired binary outcomes from baseline vs experiment.

    The test examines whether the *disagreements* between two classifiers
    are symmetric.  Uses the exact binomial test when the number of
    discordant pairs is < 25, else the chi-squared approximation with
    Edwards' continuity correction.

    Returns (chi2_or_binomial_stat, p_value).
    """
    # Build concordance table
    #            experiment correct | experiment wrong
    # base corr       a                  b
    # base wrong       c                  d
    if len(baseline_results) != len(experiment_results):
        raise ValueError(
            "Baseline and experiment must have the same number of results "
            f"({len(baseline_results)} vs {len(experiment_results)})"
        )

    b = 0  # baseline correct, experiment wrong
    c = 0  # baseline wrong, experiment correct

    for br, er in zip(baseline_results, experiment_results):
        if br.problem_id != er.problem_id:
            raise ValueError(
                f"Mismatched problem IDs: {br.problem_id} vs {er.problem_id}"
            )
        if br.correct and not er.correct:
            b += 1
        elif not br.correct and er.correct:
            c += 1

    n_discordant = b + c

    if n_discordant == 0:
        # No disagreements → no evidence of difference
        return (0.0, 1.0)

    if n_discordant < 25:
        # Exact binomial test: P(X ≥ max(b,c)) under H₀: p=0.5
        from math import comb

        k = max(b, c)
        p_value = 0.0
        for i in range(k, n_discordant + 1):
            p_value += comb(n_discordant, i) * 0.5**n_discordant
        p_value *= 2  # two-sided
        p_value = min(p_value, 1.0)
        return (float(n_discordant), p_value)
    else:
        # Chi-squared with Edwards' continuity correction
        chi2 = (abs(b - c) - 1) ** 2 / (b + c)

        # p-value from chi2 distribution with 1 df
        # Using the survival function approximation
        try:
            from scipy.stats import chi2 as chi2_dist

            p_value = chi2_dist.sf(chi2, df=1)
        except ImportError:
            # Approximation via complementary error function
            p_value = math.erfc(math.sqrt(chi2 / 2))

        return (chi2, p_value)


# ══════════════════════════════════════════════════════════════════════
#  Curated test problems (20+ across 4 categories)
# ══════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class TestProblem:
    """A curated test problem with known correct answer."""

    id: str
    category: str  # GSM8K | MATH | MMLU_STEM | FRENCH
    query: str
    expected_answer: str
    difficulty: int  # 1-5


# ── GSM8K-style arithmetic word problems ─────────────────────────────

_GSM8K_PROBLEMS = [
    TestProblem(
        id="GSM-001",
        category="GSM8K",
        query=(
            "Natalia sold clips to 48 of her friends in April, and then "
            "sold half as many clips in May. How many clips did Natalia "
            "sell altogether in April and May?"
        ),
        expected_answer="72",
        difficulty=1,
    ),
    TestProblem(
        id="GSM-002",
        category="GSM8K",
        query=(
            "A baker bakes 200 loaves of bread. He sells 93 on Monday "
            "and 39 on Tuesday. He donates a quarter of the remaining "
            "loaves to charity. How many loaves does he have left?"
        ),
        expected_answer="51",
        difficulty=1,
    ),
    TestProblem(
        id="GSM-003",
        category="GSM8K",
        query=(
            "Tom has 5 times as many marbles as Jerry. Together they have "
            "72 marbles. How many marbles does Tom have?"
        ),
        expected_answer="60",
        difficulty=1,
    ),
    TestProblem(
        id="GSM-004",
        category="GSM8K",
        query=(
            "A train travels 120 km in 2 hours, then 180 km in 3 hours. "
            "What is the average speed of the entire journey in km/h?"
        ),
        expected_answer="60",
        difficulty=1,
    ),
    TestProblem(
        id="GSM-005",
        category="GSM8K",
        query=(
            "Emily saves $15 per week. After 8 weeks, she spends $45 on "
            "a book and $30 on supplies. How much money does she have left?"
        ),
        expected_answer="45",
        difficulty=1,
    ),
]

# ── MATH competition problems ────────────────────────────────────────

_MATH_PROBLEMS = [
    TestProblem(
        id="MATH-001",
        category="MATH",
        query="What is the remainder when 2^100 is divided by 7?",
        expected_answer="2",
        difficulty=2,
    ),
    TestProblem(
        id="MATH-002",
        category="MATH",
        query=(
            "Find the sum of the infinite geometric series "
            "1 + 1/3 + 1/9 + 1/27 + ..."
        ),
        expected_answer="3/2",
        difficulty=2,
    ),
    TestProblem(
        id="MATH-003",
        category="MATH",
        query=(
            "How many positive integer divisors does 360 have?"
        ),
        expected_answer="24",
        difficulty=2,
    ),
    TestProblem(
        id="MATH-004",
        category="MATH",
        query=(
            "Let f(x) = x^3 - 6x^2 + 11x - 6. What is the sum of the "
            "roots of f?"
        ),
        expected_answer="6",
        difficulty=2,
    ),
    TestProblem(
        id="MATH-005",
        category="MATH",
        query=(
            "In how many ways can 5 distinct books be arranged on a shelf?"
        ),
        expected_answer="120",
        difficulty=1,
    ),
    TestProblem(
        id="MATH-006",
        category="MATH",
        query=(
            "Compute the integral ∫₀^π sin²(x) dx."
        ),
        expected_answer="π/2",
        difficulty=3,
    ),
]

# ── MMLU-STEM physics questions ──────────────────────────────────────

_MMLU_STEM_PROBLEMS = [
    TestProblem(
        id="MMLU-001",
        category="MMLU_STEM",
        query=(
            "A 2 kg object is dropped from rest at a height of 20 m. "
            "Ignoring air resistance, what is its speed just before "
            "hitting the ground? Use g = 10 m/s²."
        ),
        expected_answer="20",
        difficulty=1,
    ),
    TestProblem(
        id="MMLU-002",
        category="MMLU_STEM",
        query=(
            "What is the de Broglie wavelength of an electron "
            "(mass 9.11 × 10⁻³¹ kg) moving at 10⁶ m/s? "
            "Give your answer in nanometers, rounded to 2 decimal places. "
            "Use h = 6.626 × 10⁻³⁴ J·s."
        ),
        expected_answer="0.73",
        difficulty=2,
    ),
    TestProblem(
        id="MMLU-003",
        category="MMLU_STEM",
        query=(
            "A parallel plate capacitor has plates of area 0.01 m² "
            "separated by 0.001 m. What is its capacitance in pF? "
            "Use ε₀ = 8.854 × 10⁻¹² F/m."
        ),
        expected_answer="88.54",
        difficulty=2,
    ),
    TestProblem(
        id="MMLU-004",
        category="MMLU_STEM",
        query=(
            "A 0.5 kg ball is thrown upward with an initial velocity "
            "of 15 m/s. What is its kinetic energy at the highest point? "
            "Ignore air resistance."
        ),
        expected_answer="0",
        difficulty=1,
    ),
    TestProblem(
        id="MMLU-005",
        category="MMLU_STEM",
        query=(
            "Two resistors of 6 Ω and 3 Ω are connected in parallel. "
            "What is the equivalent resistance in Ohms?"
        ),
        expected_answer="2",
        difficulty=1,
    ),
]

# ── Custom French exam problems ──────────────────────────────────────

_FRENCH_PROBLEMS = [
    TestProblem(
        id="FR-001",
        category="FRENCH",
        query=(
            "Calculer les valeurs propres de la matrice A = [[2, 1], [1, 2]]. "
            "Donner les valeurs séparées par une virgule, en ordre croissant."
        ),
        expected_answer="1, 3",
        difficulty=2,
    ),
    TestProblem(
        id="FR-002",
        category="FRENCH",
        query=(
            "Déterminer la pulsation de résonance ω₀ d'un circuit RLC série "
            "avec L = 0.1 H et C = 10 μF. Réponse en rad/s."
        ),
        expected_answer="1000",
        difficulty=1,
    ),
    TestProblem(
        id="FR-003",
        category="FRENCH",
        query=(
            "Calculer ∫₀^{2π} dθ / (2 + cos θ) en utilisant le théorème "
            "des résidus. Donner la réponse exacte."
        ),
        expected_answer="2π/√3",
        difficulty=3,
    ),
    TestProblem(
        id="FR-004",
        category="FRENCH",
        query=(
            "Donner l'épaisseur de peau δ dans un bon conducteur en fonction "
            "de ω, μ₀ et σ."
        ),
        expected_answer="√(2/(ωμ₀σ))",
        difficulty=2,
    ),
]


ALL_TEST_PROBLEMS: tuple[TestProblem, ...] = tuple(
    _GSM8K_PROBLEMS
    + _MATH_PROBLEMS
    + _MMLU_STEM_PROBLEMS
    + _FRENCH_PROBLEMS
)


def get_problems_for_tier(tier: str) -> list[TestProblem]:
    """Filter problems by tier (category) or return all."""
    tier_upper = tier.upper()
    if tier_upper == "ALL":
        return list(ALL_TEST_PROBLEMS)
    return [p for p in ALL_TEST_PROBLEMS if p.category == tier_upper]


# ══════════════════════════════════════════════════════════════════════
#  Server interaction
# ══════════════════════════════════════════════════════════════════════


def _query_server(
    client: httpx.Client,
    server_url: str,
    query: str,
    timeout: float = 120.0,
) -> tuple[str, str, float]:
    """
    Send a query to the /v4/solve endpoint and return
    (answer, routing_decision, latency_ms).
    """
    url = f"{server_url.rstrip('/')}/v4/solve"
    payload = {"query": query}

    t0 = time.perf_counter()
    try:
        response = client.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        data = response.json()
    except (httpx.HTTPStatusError, httpx.RequestError, json.JSONDecodeError) as exc:
        logger.error("Request failed for query %.60s...: %s", query, exc)
        latency = (time.perf_counter() - t0) * 1000
        return ("ERROR", "UNKNOWN", latency)

    latency = (time.perf_counter() - t0) * 1000
    answer = str(data.get("answer", "")).strip()
    routing = str(data.get("routing_decision", data.get("model_tier", "UNKNOWN")))
    return (answer, routing, latency)


def _normalize_answer(raw: str) -> str:
    """Normalize an answer string for comparison (strip, lowercase, collapse whitespace)."""
    import re

    s = raw.strip().lower()
    s = re.sub(r"\s+", " ", s)
    # Remove trailing periods / units for numerical answers
    s = re.sub(r"[.\s]+$", "", s)
    return s


def _check_correct(expected: str, actual: str) -> bool:
    """Compare expected and actual answers with fuzzy matching."""
    e = _normalize_answer(expected)
    a = _normalize_answer(actual)

    if e == a:
        return True

    # Try numeric comparison
    try:
        e_val = float(eval(e.replace("π", str(math.pi)).replace("√", "math.sqrt")))  # noqa: S307
        a_val = float(eval(a.replace("π", str(math.pi)).replace("√", "math.sqrt")))  # noqa: S307
        return abs(e_val - a_val) < 1e-6 * max(1, abs(e_val))
    except Exception:
        pass

    # Check if actual answer *contains* the expected answer
    return e in a


# ══════════════════════════════════════════════════════════════════════
#  Runner
# ══════════════════════════════════════════════════════════════════════


def run_benchmark(
    server_url: str,
    tier: str = "all",
    baseline_path: Path | None = None,
    timeout_per_problem: float = 120.0,
) -> BenchmarkSummary:
    """Execute the full benchmark suite and return a summary."""
    problems = get_problems_for_tier(tier)
    if not problems:
        raise ValueError(f"No problems found for tier '{tier}'")

    logger.info("Running %d problems against %s (tier=%s)", len(problems), server_url, tier)

    results: list[BenchmarkResult] = []

    with httpx.Client(verify=False) as client:
        for i, prob in enumerate(problems, 1):
            logger.info("[%d/%d] %s — %s", i, len(problems), prob.id, prob.query[:60])
            answer, routing, latency = _query_server(
                client, server_url, prob.query, timeout=timeout_per_problem
            )
            correct = _check_correct(prob.expected_answer, answer)
            results.append(
                BenchmarkResult(
                    problem_id=prob.id,
                    category=prob.category,
                    query=prob.query,
                    expected_answer=prob.expected_answer,
                    actual_answer=answer,
                    correct=correct,
                    latency_ms=latency,
                    routing_decision=routing,
                )
            )

    # ── Aggregate metrics ─────────────────────────────────────────────
    n_correct = sum(r.correct for r in results)
    total = len(results)
    accuracy = n_correct / total if total > 0 else 0.0
    ci_lo, ci_hi = wilson_score_ci(n_correct, total)

    # ── Category breakdown ────────────────────────────────────────────
    from collections import defaultdict

    cat_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        cat_counts[r.category]["total"] += 1
        if r.correct:
            cat_counts[r.category]["correct"] += 1
    cat_breakdown = {
        cat: {
            "correct": v["correct"],
            "total": v["total"],
            "accuracy": v["correct"] / v["total"] if v["total"] else 0.0,
        }
        for cat, v in cat_counts.items()
    }

    # ── McNemar's test against baseline ───────────────────────────────
    mcnemar_stat: float | None = None
    mcnemar_p: float | None = None

    if baseline_path and baseline_path.exists():
        try:
            baseline_data = json.loads(baseline_path.read_text())
            baseline_results_raw = baseline_data.get("results", [])
            baseline_results = [
                BenchmarkResult(**r) for r in baseline_results_raw
            ]
            # Align by problem_id
            baseline_map = {r.problem_id: r for r in baseline_results}
            paired_baseline = []
            paired_experiment = []
            for r in results:
                if r.problem_id in baseline_map:
                    paired_baseline.append(baseline_map[r.problem_id])
                    paired_experiment.append(r)
            if paired_baseline:
                mcnemar_stat, mcnemar_p = mcnemar_test(paired_baseline, paired_experiment)
                logger.info(
                    "McNemar test: statistic=%.4f, p-value=%.6f", mcnemar_stat, mcnemar_p
                )
        except Exception as exc:
            logger.warning("Could not run McNemar test: %s", exc)

    summary = BenchmarkSummary(
        timestamp=datetime.now(timezone.utc).isoformat(),
        server_url=server_url,
        tier=tier,
        total=total,
        correct=n_correct,
        accuracy=accuracy,
        ci_lower=ci_lo,
        ci_upper=ci_hi,
        results=results,
        mcnemar_p_value=mcnemar_p,
        mcnemar_statistic=mcnemar_stat,
        category_breakdown=cat_breakdown,
    )
    return summary


# ══════════════════════════════════════════════════════════════════════
#  Reporting
# ══════════════════════════════════════════════════════════════════════


def save_json(summary: BenchmarkSummary, output_dir: Path) -> Path:
    """Write the full results to a JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"benchmark_{ts}.json"
    path.write_text(json.dumps(asdict(summary), indent=2, default=str))
    logger.info("JSON results saved to %s", path)
    return path


def generate_markdown_report(summary: BenchmarkSummary) -> str:
    """Generate a formatted Markdown report from benchmark results."""
    lines: list[str] = []
    lines.append("# SymBrain v4 — Benchmark Report")
    lines.append("")
    lines.append(f"**Timestamp:** {summary.timestamp}  ")
    lines.append(f"**Server:** `{summary.server_url}`  ")
    lines.append(f"**Tier:** {summary.tier}  ")
    lines.append("")

    # Overall metrics
    lines.append("## Overall Results")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Total problems | {summary.total} |")
    lines.append(f"| Correct | {summary.correct} |")
    lines.append(f"| Accuracy | {summary.accuracy:.1%} |")
    lines.append(
        f"| 95% Wilson CI | [{summary.ci_lower:.3f}, {summary.ci_upper:.3f}] |"
    )
    if summary.mcnemar_p_value is not None:
        sig = "✅ Yes" if summary.mcnemar_p_value < 0.05 else "❌ No"
        lines.append(
            f"| McNemar χ² | {summary.mcnemar_statistic:.4f} (p={summary.mcnemar_p_value:.4f}) |"
        )
        lines.append(f"| Significant vs baseline? | {sig} |")
    lines.append("")

    # Category breakdown
    lines.append("## Category Breakdown")
    lines.append("")
    lines.append("| Category | Correct | Total | Accuracy |")
    lines.append("|----------|---------|-------|----------|")
    for cat, info in sorted(summary.category_breakdown.items()):
        lines.append(
            f"| {cat} | {info['correct']} | {info['total']} | {info['accuracy']:.1%} |"
        )
    lines.append("")

    # Per-problem detail
    lines.append("## Detailed Results")
    lines.append("")
    lines.append("| ID | Category | Correct | Latency (ms) | Routing |")
    lines.append("|-----|----------|---------|-------------|---------|")
    for r in summary.results:
        icon = "✅" if r.correct else "❌"
        lines.append(
            f"| {r.problem_id} | {r.category} | {icon} | {r.latency_ms:.0f} | {r.routing_decision} |"
        )
    lines.append("")

    # Latency statistics
    latencies = [r.latency_ms for r in summary.results]
    if latencies:
        lines.append("## Latency Statistics")
        lines.append("")
        lines.append(f"| Metric | Value (ms) |")
        lines.append(f"|--------|-----------|")
        lines.append(f"| Mean | {sum(latencies)/len(latencies):.0f} |")
        lines.append(f"| Median | {sorted(latencies)[len(latencies)//2]:.0f} |")
        lines.append(f"| Min | {min(latencies):.0f} |")
        lines.append(f"| Max | {max(latencies):.0f} |")
        lines.append(f"| P95 | {sorted(latencies)[int(0.95*len(latencies))]:.0f} |")
    lines.append("")

    return "\n".join(lines)


def save_markdown(summary: BenchmarkSummary, output_dir: Path) -> Path:
    """Write the Markdown report to a file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"benchmark_{ts}.md"
    path.write_text(generate_markdown_report(summary))
    logger.info("Markdown report saved to %s", path)
    return path


# ══════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SymBrain v4 Benchmark Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--server-url",
        required=True,
        help="Base URL of the SymBrain v4 server (e.g. http://localhost:8080)",
    )
    parser.add_argument(
        "--tier",
        default="all",
        choices=["all", "GSM8K", "MATH", "MMLU_STEM", "FRENCH"],
        help="Category of problems to run (default: all)",
    )
    parser.add_argument(
        "--baseline-json",
        type=Path,
        default=None,
        help="Path to v3 baseline results JSON for McNemar comparison",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./eval_results"),
        help="Directory for output files (default: ./eval_results)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Timeout per problem in seconds (default: 120)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    summary = run_benchmark(
        server_url=args.server_url,
        tier=args.tier,
        baseline_path=args.baseline_json,
        timeout_per_problem=args.timeout,
    )

    json_path = save_json(summary, args.output_dir)
    md_path = save_markdown(summary, args.output_dir)

    # Print summary to stdout
    print(generate_markdown_report(summary))
    print(f"\n📄 JSON: {json_path}")
    print(f"📝 Report: {md_path}")

    sys.exit(0 if summary.accuracy >= 0.5 else 1)


if __name__ == "__main__":
    main()
