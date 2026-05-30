"""
SymBrain v4 — Peer Review Agent
════════════════════════════════
Evaluates experiment results for mathematical soundness and statistical
rigor.  Can optionally call the local SymBrain v3 server (port 8085)
for cross-validation / peer evaluation.

Usage:
    python -m v4.eval.peer_review \
        --results-json eval_results/benchmark_20260528.json \
        --output review_report.md \
        --use-v3-peer
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════
#  Data models
# ══════════════════════════════════════════════════════════════════════


@dataclass
class PeerReviewReport:
    """Structured peer-review report of an experiment run."""

    timestamp: str
    source_file: str

    # Scores (1-10 scale)
    mathematical_soundness_score: int
    statistical_rigor_score: int

    # Structured feedback
    mathematical_issues: list[str]
    statistical_issues: list[str]
    refinement_suggestions: list[str]

    # Overall
    overall_verdict: str  # "PASS" | "MARGINAL" | "FAIL"
    summary: str

    # Optional v3 peer evaluation
    v3_peer_verdict: str | None = None
    v3_peer_details: str | None = None


# ══════════════════════════════════════════════════════════════════════
#  Review logic
# ══════════════════════════════════════════════════════════════════════


def _evaluate_mathematical_soundness(
    results: list[dict[str, Any]],
) -> tuple[int, list[str]]:
    """
    Evaluate mathematical soundness of the experiment results.

    Checks:
    - Are answers numerically plausible?
    - Are there systematic errors (e.g. all wrong in one category)?
    - Is the answer format consistent?

    Returns (score, list_of_issues).
    """
    issues: list[str] = []
    score = 10  # start at perfect, deduct

    if not results:
        return (1, ["No results to evaluate."])

    total = len(results)
    n_correct = sum(1 for r in results if r.get("correct", False))
    n_error = sum(1 for r in results if r.get("actual_answer") == "ERROR")

    # Check for high error rate
    error_rate = n_error / total if total else 0
    if error_rate > 0.3:
        issues.append(
            f"High error rate: {n_error}/{total} ({error_rate:.0%}) queries returned ERROR. "
            "This suggests server instability rather than mathematical errors."
        )
        score -= 3

    # Check for suspiciously perfect or zero accuracy
    accuracy = n_correct / total if total else 0
    if accuracy == 1.0 and total > 5:
        issues.append(
            "Perfect accuracy (100%) on all problems — verify that answer checking "
            "is not trivially matching (e.g. substring match on common tokens)."
        )
        score -= 1
    elif accuracy == 0.0 and total > 5:
        issues.append(
            "Zero accuracy (0%) — indicates fundamental solver failure or "
            "answer format mismatch."
        )
        score -= 4

    # Check category-level performance
    from collections import defaultdict

    cat_results: dict[str, list[bool]] = defaultdict(list)
    for r in results:
        cat_results[r.get("category", "UNKNOWN")].append(r.get("correct", False))

    for cat, corrections in cat_results.items():
        cat_acc = sum(corrections) / len(corrections) if corrections else 0
        if cat_acc == 0.0 and len(corrections) >= 3:
            issues.append(
                f"Category '{cat}': 0% accuracy on {len(corrections)} problems — "
                f"systematic failure in this domain."
            )
            score -= 2

    # Check for answer diversity (are all answers identical?)
    answers = [r.get("actual_answer", "") for r in results]
    unique_answers = set(answers)
    if len(unique_answers) == 1 and total > 5:
        issues.append(
            f"All {total} answers are identical ('{answers[0]}'). "
            "The solver may be returning a default response."
        )
        score -= 3

    # Check latency anomalies
    latencies = [r.get("latency_ms", 0) for r in results]
    if latencies:
        mean_lat = sum(latencies) / len(latencies)
        if mean_lat < 1.0 and total > 5:
            issues.append(
                f"Mean latency is suspiciously low ({mean_lat:.1f} ms). "
                "Are cached / mocked responses being returned?"
            )
            score -= 2

    return (max(1, min(10, score)), issues)


def _evaluate_statistical_rigor(
    experiment_data: dict[str, Any],
) -> tuple[int, list[str]]:
    """
    Evaluate statistical rigor of the reported results.

    Checks:
    - Are confidence intervals reported?
    - Is a baseline comparison present?
    - Is the sample size adequate?
    - Are p-values interpreted correctly?

    Returns (score, list_of_issues).
    """
    issues: list[str] = []
    score = 10

    total = experiment_data.get("total", 0)

    # ── Sample size ───────────────────────────────────────────────────
    if total < 10:
        issues.append(
            f"Sample size is very small (n={total}). Results are unreliable; "
            "consider running at least 30 problems for meaningful statistics."
        )
        score -= 3
    elif total < 30:
        issues.append(
            f"Sample size (n={total}) is below 30. Wilson CI will be wide; "
            "interpret with caution."
        )
        score -= 1

    # ── Confidence intervals ──────────────────────────────────────────
    ci_lower = experiment_data.get("ci_lower")
    ci_upper = experiment_data.get("ci_upper")
    if ci_lower is None or ci_upper is None:
        issues.append(
            "No confidence interval reported. Cannot assess precision of "
            "the accuracy estimate."
        )
        score -= 3
    else:
        ci_width = ci_upper - ci_lower
        if ci_width > 0.4:
            issues.append(
                f"CI width is very large ({ci_width:.2f}). The accuracy estimate "
                "is imprecise — increase sample size."
            )
            score -= 1

        # Sanity check CI bounds
        accuracy = experiment_data.get("accuracy", 0)
        if accuracy < ci_lower or accuracy > ci_upper:
            issues.append(
                f"Accuracy ({accuracy:.3f}) falls outside its own CI "
                f"[{ci_lower:.3f}, {ci_upper:.3f}]. This is a bug."
            )
            score -= 3

    # ── Baseline comparison ───────────────────────────────────────────
    mcnemar_p = experiment_data.get("mcnemar_p_value")
    mcnemar_stat = experiment_data.get("mcnemar_statistic")

    if mcnemar_p is None:
        issues.append(
            "No McNemar test / baseline comparison provided. "
            "Without a baseline, it is impossible to assess whether v4 "
            "represents an improvement."
        )
        score -= 2
    else:
        if mcnemar_p < 0.001:
            # Suspiciously small p-value with small sample
            if total < 20:
                issues.append(
                    f"McNemar p-value ({mcnemar_p:.6f}) is very small for "
                    f"n={total}. Verify the test is correctly implemented."
                )
                score -= 1
        elif mcnemar_p > 0.05:
            issues.append(
                f"McNemar test is not significant (p={mcnemar_p:.4f}). "
                "There is no evidence that v4 differs from the baseline."
            )
            # This is informational, not a rigor issue
        else:
            pass  # p ∈ [0.001, 0.05] — reasonable

    # ── Category balance ──────────────────────────────────────────────
    breakdown = experiment_data.get("category_breakdown", {})
    if breakdown:
        sizes = [v.get("total", 0) for v in breakdown.values()]
        if sizes and max(sizes) > 3 * min(sizes):
            issues.append(
                "Category sizes are highly imbalanced. Overall accuracy may be "
                "dominated by the largest category."
            )
            score -= 1

    return (max(1, min(10, score)), issues)


def _generate_refinement_suggestions(
    math_issues: list[str],
    stat_issues: list[str],
    experiment_data: dict[str, Any],
) -> list[str]:
    """Generate actionable refinement suggestions based on identified issues."""
    suggestions: list[str] = []

    total = experiment_data.get("total", 0)
    accuracy = experiment_data.get("accuracy", 0)

    if total < 30:
        suggestions.append(
            "Increase the benchmark suite to at least 50 problems per category "
            "for statistically meaningful comparisons."
        )

    if accuracy < 0.5:
        suggestions.append(
            "Accuracy below 50% suggests the solver needs fundamental improvements. "
            "Consider a detailed error analysis by category and difficulty level."
        )

    if experiment_data.get("mcnemar_p_value") is None:
        suggestions.append(
            "Run the v3 baseline on the same problem set and provide the results "
            "JSON for McNemar comparison (--baseline-json flag)."
        )

    breakdown = experiment_data.get("category_breakdown", {})
    for cat, info in breakdown.items():
        cat_acc = info.get("accuracy", 0)
        if cat_acc < 0.3:
            suggestions.append(
                f"Category '{cat}' has very low accuracy ({cat_acc:.0%}). "
                f"Investigate prompt engineering or model selection for this domain."
            )

    if any("latency" in i.lower() for i in math_issues):
        suggestions.append(
            "Review the inference pipeline for caching or mocking issues "
            "that could produce artificially low latencies."
        )

    if not suggestions:
        suggestions.append(
            "Results look solid. Consider expanding to harder problems "
            "(difficulty 4-5) and testing edge cases."
        )

    return suggestions


def _compute_overall_verdict(math_score: int, stat_score: int) -> str:
    """Determine overall verdict from component scores."""
    avg = (math_score + stat_score) / 2
    if avg >= 7:
        return "PASS"
    elif avg >= 4:
        return "MARGINAL"
    else:
        return "FAIL"


# ══════════════════════════════════════════════════════════════════════
#  v3 peer evaluation
# ══════════════════════════════════════════════════════════════════════


def _query_v3_peer(
    experiment_summary: str,
    v3_url: str = "http://localhost:8085",
    timeout: float = 60.0,
) -> tuple[str, str]:
    """
    Send the experiment summary to the SymBrain v3 server for peer
    evaluation.

    Returns (verdict, details).
    """
    try:
        with httpx.Client() as client:
            response = client.post(
                f"{v3_url}/v3/solve",
                json={
                    "query": (
                        "You are acting as a peer reviewer. Evaluate the following "
                        "benchmark results for mathematical correctness and statistical "
                        "rigor. Provide a verdict (PASS/MARGINAL/FAIL) and brief "
                        f"justification.\n\n{experiment_summary}"
                    ),
                },
                timeout=timeout,
            )
            response.raise_for_status()
            data = response.json()
            answer = data.get("answer", "No response from v3 peer.")
            # Try to extract verdict
            answer_upper = answer.upper()
            if "PASS" in answer_upper:
                verdict = "PASS"
            elif "FAIL" in answer_upper:
                verdict = "FAIL"
            else:
                verdict = "MARGINAL"
            return (verdict, answer)
    except Exception as exc:
        logger.warning("v3 peer evaluation failed: %s", exc)
        return ("UNAVAILABLE", f"Could not reach v3 server: {exc}")


# ══════════════════════════════════════════════════════════════════════
#  Main review pipeline
# ══════════════════════════════════════════════════════════════════════


def review_experiment(
    results_path: Path,
    use_v3_peer: bool = False,
    v3_url: str = "http://localhost:8085",
) -> PeerReviewReport:
    """
    Run the full peer review pipeline on an experiment results JSON file.
    """
    data = json.loads(results_path.read_text())
    results = data.get("results", [])

    logger.info(
        "Reviewing experiment: %d results from %s",
        len(results),
        data.get("server_url", "unknown"),
    )

    # ── Mathematical soundness ────────────────────────────────────────
    math_score, math_issues = _evaluate_mathematical_soundness(results)
    logger.info("Mathematical soundness: %d/10", math_score)

    # ── Statistical rigor ─────────────────────────────────────────────
    stat_score, stat_issues = _evaluate_statistical_rigor(data)
    logger.info("Statistical rigor: %d/10", stat_score)

    # ── Refinement suggestions ────────────────────────────────────────
    suggestions = _generate_refinement_suggestions(math_issues, stat_issues, data)

    # ── Overall verdict ───────────────────────────────────────────────
    verdict = _compute_overall_verdict(math_score, stat_score)

    # ── Summary ───────────────────────────────────────────────────────
    accuracy = data.get("accuracy", 0)
    ci_lo = data.get("ci_lower", 0)
    ci_hi = data.get("ci_upper", 1)
    summary = (
        f"Experiment evaluated {data.get('total', 0)} problems with "
        f"{accuracy:.1%} accuracy (95% CI: [{ci_lo:.3f}, {ci_hi:.3f}]). "
        f"Mathematical soundness: {math_score}/10, Statistical rigor: {stat_score}/10. "
        f"Verdict: {verdict}."
    )

    # ── v3 peer evaluation ────────────────────────────────────────────
    v3_verdict = None
    v3_details = None
    if use_v3_peer:
        logger.info("Requesting v3 peer evaluation from %s...", v3_url)
        v3_verdict, v3_details = _query_v3_peer(summary, v3_url=v3_url)
        logger.info("v3 peer verdict: %s", v3_verdict)

    report = PeerReviewReport(
        timestamp=datetime.now(timezone.utc).isoformat(),
        source_file=str(results_path),
        mathematical_soundness_score=math_score,
        statistical_rigor_score=stat_score,
        mathematical_issues=math_issues,
        statistical_issues=stat_issues,
        refinement_suggestions=suggestions,
        overall_verdict=verdict,
        summary=summary,
        v3_peer_verdict=v3_verdict,
        v3_peer_details=v3_details,
    )
    return report


# ══════════════════════════════════════════════════════════════════════
#  Markdown report
# ══════════════════════════════════════════════════════════════════════


def generate_review_markdown(report: PeerReviewReport) -> str:
    """Generate a structured markdown peer review report."""
    verdict_emoji = {
        "PASS": "✅",
        "MARGINAL": "⚠️",
        "FAIL": "❌",
    }

    lines: list[str] = []
    lines.append("# SymBrain v4 — Peer Review Report")
    lines.append("")
    lines.append(f"**Timestamp:** {report.timestamp}  ")
    lines.append(f"**Source:** `{report.source_file}`  ")
    lines.append(
        f"**Verdict:** {verdict_emoji.get(report.overall_verdict, '❓')} "
        f"**{report.overall_verdict}**"
    )
    lines.append("")
    lines.append(f"> {report.summary}")
    lines.append("")

    # Scores
    lines.append("## Scores")
    lines.append("")
    lines.append("| Dimension | Score |")
    lines.append("|-----------|-------|")
    lines.append(f"| Mathematical Soundness | {report.mathematical_soundness_score}/10 |")
    lines.append(f"| Statistical Rigor | {report.statistical_rigor_score}/10 |")
    lines.append("")

    # Mathematical issues
    if report.mathematical_issues:
        lines.append("## Mathematical Issues")
        lines.append("")
        for issue in report.mathematical_issues:
            lines.append(f"- ⚠️ {issue}")
        lines.append("")

    # Statistical issues
    if report.statistical_issues:
        lines.append("## Statistical Issues")
        lines.append("")
        for issue in report.statistical_issues:
            lines.append(f"- 📊 {issue}")
        lines.append("")

    # Refinement suggestions
    lines.append("## Refinement Suggestions")
    lines.append("")
    for i, suggestion in enumerate(report.refinement_suggestions, 1):
        lines.append(f"{i}. {suggestion}")
    lines.append("")

    # v3 peer evaluation
    if report.v3_peer_verdict:
        lines.append("## v3 Peer Evaluation")
        lines.append("")
        lines.append(
            f"**v3 Verdict:** {verdict_emoji.get(report.v3_peer_verdict, '❓')} "
            f"{report.v3_peer_verdict}"
        )
        if report.v3_peer_details:
            lines.append("")
            lines.append(f"> {report.v3_peer_details}")
        lines.append("")

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SymBrain v4 Peer Review Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--results-json",
        type=Path,
        required=True,
        help="Path to the experiment results JSON file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path for the output markdown review (default: stdout)",
    )
    parser.add_argument(
        "--use-v3-peer",
        action="store_true",
        help="Query the local v3 server for peer evaluation",
    )
    parser.add_argument(
        "--v3-url",
        default="http://localhost:8085",
        help="URL of the v3 server (default: http://localhost:8085)",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optional path to save structured review as JSON",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    if not args.results_json.exists():
        logger.error("Results file not found: %s", args.results_json)
        sys.exit(1)

    report = review_experiment(
        results_path=args.results_json,
        use_v3_peer=args.use_v3_peer,
        v3_url=args.v3_url,
    )

    md = generate_review_markdown(report)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(md)
        logger.info("Review saved to %s", args.output)
    else:
        print(md)

    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(asdict(report), indent=2, default=str)
        )
        logger.info("JSON review saved to %s", args.json_output)

    # Exit code based on verdict
    exit_codes = {"PASS": 0, "MARGINAL": 0, "FAIL": 1}
    sys.exit(exit_codes.get(report.overall_verdict, 1))


if __name__ == "__main__":
    main()
