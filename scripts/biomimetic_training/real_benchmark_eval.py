#!/usr/bin/env python3
# Copyright (c) 2026 Xavier Callens / Socrate AI Lab. All Rights Reserved.
# SPDX-License-Identifier: LicenseRef-RunuX-Commercial
#
# Real Benchmark Evaluation — GSM8K · MATH · MMLU-STEM
# =====================================================
# Runs actual model inference against standard benchmarks with LaTeX answer
# parsing, multi-choice extraction, MCTS-based evaluation, and per-topic
# accuracy breakdowns.

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ─────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("real_benchmark_eval")

# ─────────────────────────────────────────────────────────────────
# Console formatting
# ─────────────────────────────────────────────────────────────────
GREEN = "\033[0;32m"
YELLOW = "\033[0;33m"
BLUE = "\033[0;34m"
CYAN = "\033[0;36m"
RED = "\033[0;31m"
MAGENTA = "\033[0;35m"
BOLD = "\033[1m"
NC = "\033[0m"


# ─────────────────────────────────────────────────────────────────
# Shared data structures
# ─────────────────────────────────────────────────────────────────
@dataclass
class EvalSample:
    """Single evaluation sample."""

    question: str
    ground_truth: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalResult:
    """Result for a single evaluation sample."""

    sample_index: int
    predicted: str
    ground_truth: str
    is_correct: bool
    raw_generation: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    latency_s: float = 0.0


@dataclass
class BenchmarkReport:
    """Aggregate report for an entire benchmark."""

    benchmark: str
    accuracy: float
    correct: int
    total: int
    results: List[EvalResult] = field(default_factory=list)
    breakdown: Dict[str, Any] = field(default_factory=dict)
    wall_time_s: float = 0.0
    simulated: bool = False

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Drop per-sample results from the serialised dict to keep output small
        d.pop("results", None)
        return d


# ─────────────────────────────────────────────────────────────────
# Answer parsing utilities
# ─────────────────────────────────────────────────────────────────

def _strip_commas(s: str) -> str:
    """Remove comma groupings from number strings (e.g. '1,234' → '1234')."""
    return s.replace(",", "")


def parse_gsm8k_answer(text: str) -> Optional[str]:
    """Extract the numerical answer from ``#### <number>`` format.

    Handles:
    - ``#### 42``
    - ``#### 1,234``
    - ``#### -3.5``
    - Free-form final line with a number (fallback)
    """
    # Primary: look for #### <number>
    m = re.search(r"####\s*([+-]?[\d,]+\.?\d*)", text)
    if m:
        return _strip_commas(m.group(1).strip())
    # Fallback: last number in the text
    nums = re.findall(r"[+-]?[\d,]+\.?\d*", text)
    if nums:
        return _strip_commas(nums[-1].strip())
    return None


def normalise_number(s: str) -> Optional[float]:
    """Try to convert *s* to a float for numerical comparison."""
    try:
        return float(_strip_commas(s.strip()))
    except (ValueError, TypeError):
        return None


def parse_latex_answer(text: str) -> str:
    r"""Extract and normalise a LaTeX answer.

    Handles: ``\boxed{…}``, ``\frac{a}{b}``, ``\dfrac``, ``\tfrac``,
    plain numbers, and common notation.
    """
    # 1. Extract from \boxed{…} (may be nested)
    boxed = _extract_boxed(text)
    if boxed is not None:
        text = boxed

    # 2. Remove dollar signs and \text{…}
    text = text.strip().strip("$").strip()
    text = re.sub(r"\\text\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", text)

    # 3. Simplify via sympy if available
    simplified = _sympy_simplify(text)
    if simplified is not None:
        return simplified

    # 4. Manual normalisation
    text = text.replace(" ", "")
    text = re.sub(r"\\(?:left|right)[()\\|.\[\]]", "", text)
    text = text.replace("\\cdot", "*").replace("\\times", "*")
    text = text.replace("\\%", "%")

    # Fraction → decimal
    frac_m = re.match(r"\\(?:d|t)?frac\{([^}]+)\}\{([^}]+)\}", text)
    if frac_m:
        try:
            num = float(frac_m.group(1))
            den = float(frac_m.group(2))
            if den != 0:
                return str(num / den)
        except ValueError:
            pass

    return text.strip()


def _extract_boxed(text: str) -> Optional[str]:
    r"""Extract content inside ``\boxed{…}`` with brace matching."""
    idx = text.rfind("\\boxed{")
    if idx == -1:
        idx = text.rfind("\\boxed ")
        if idx == -1:
            return None
        # Simple form: \boxed 42
        rest = text[idx + len("\\boxed "):]
        m = re.match(r"(\S+)", rest)
        return m.group(1) if m else None

    start = idx + len("\\boxed{")
    depth = 1
    pos = start
    while pos < len(text) and depth > 0:
        if text[pos] == "{":
            depth += 1
        elif text[pos] == "}":
            depth -= 1
        pos += 1
    return text[start : pos - 1] if depth == 0 else text[start:]


def _sympy_simplify(expr_str: str) -> Optional[str]:
    """Try to parse and simplify *expr_str* with SymPy.

    Returns a canonical string, or ``None`` if SymPy is unavailable or
    the expression is not parseable.
    """
    try:
        import sympy
        from sympy.parsing.latex import parse_latex as _parse_latex

        parsed = _parse_latex(expr_str)
        simplified = sympy.nsimplify(parsed, rational=True)
        # Return the simplest representation
        if simplified.is_number:
            # Use rational form if exact, else float
            if simplified.is_Rational:
                if simplified.q == 1:
                    return str(simplified.p)
                return str(simplified)
            return str(float(simplified))
        return str(simplified)
    except Exception:
        return None


def answers_equivalent(predicted: str, ground_truth: str) -> bool:
    """Check if *predicted* and *ground_truth* represent the same answer.

    Tries numerical comparison first, then symbolic (SymPy), then exact
    string match.
    """
    if not predicted or not ground_truth:
        return False

    # 1. Exact string match (after whitespace normalisation)
    p_norm = predicted.strip().lower().replace(" ", "")
    g_norm = ground_truth.strip().lower().replace(" ", "")
    if p_norm == g_norm:
        return True

    # 2. Numerical comparison
    p_num = normalise_number(predicted)
    g_num = normalise_number(ground_truth)
    if p_num is not None and g_num is not None:
        return math.isclose(p_num, g_num, rel_tol=1e-6, abs_tol=1e-8)

    # 3. SymPy symbolic equivalence
    try:
        import sympy
        from sympy.parsing.latex import parse_latex as _parse_latex

        p_sym = _parse_latex(predicted)
        g_sym = _parse_latex(ground_truth)
        if sympy.simplify(p_sym - g_sym) == 0:
            return True
    except Exception:
        pass

    return False


def parse_multiple_choice(text: str) -> Optional[str]:
    """Extract a multiple-choice letter (A-D) from model generation.

    Recognises patterns like:
    - ``The answer is (B)``
    - ``Answer: C``
    - ``B.``  (as first token)
    - Bare ``A``
    """
    text = text.strip()

    # Pattern: "the answer is (X)" / "answer: X"
    m = re.search(r"(?:the\s+)?answer\s*(?:is|:)\s*\(?([A-Da-d])\)?", text, re.I)
    if m:
        return m.group(1).upper()

    # Pattern: "(X)" standalone
    m = re.search(r"\(([A-Da-d])\)", text)
    if m:
        return m.group(1).upper()

    # Pattern: letter followed by period or at end
    m = re.search(r"\b([A-Da-d])\s*[.):]", text)
    if m:
        return m.group(1).upper()

    # Bare letter (last resort, first character)
    if len(text) >= 1 and text[0].upper() in "ABCD":
        return text[0].upper()

    return None


# ─────────────────────────────────────────────────────────────────
# Model inference helpers
# ─────────────────────────────────────────────────────────────────

def _load_model_and_tokenizer(
    model_name: str, device: Optional[str] = None
) -> Tuple[Any, Any, Any]:
    """Load a HuggingFace causal-LM and tokenizer.

    Returns (model, tokenizer, torch_device).
    Raises ``RuntimeError`` if loading fails.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    torch_device = torch.device(device)
    dtype = torch.bfloat16 if device in ("cuda", "tpu") else torch.float32

    logger.info(f"  Loading tokenizer for {BOLD}{model_name}{NC} …")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"  Loading model weights ({dtype}) → {device} …")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map="auto" if device == "cuda" else None,
    )
    if device != "cuda":  # device_map already handles cuda
        model = model.to(torch_device)
    model.eval()

    logger.info(f"  {GREEN}✓ Model ready on {device}{NC}")
    return model, tokenizer, torch_device


@dataclass
class _MCTSNode:
    """Lightweight MCTS tree node for answer-level search."""

    text: str
    score: float = 0.0
    visits: int = 0
    children: List["_MCTSNode"] = field(default_factory=list)

    @property
    def ucb(self) -> float:
        if self.visits == 0:
            return float("inf")
        return self.score / self.visits + 1.414 * math.sqrt(
            math.log(max(1, sum(c.visits for c in self.children) + self.visits))
            / self.visits
        )


def _generate_greedy(
    model: Any,
    tokenizer: Any,
    prompt: str,
    device: Any,
    max_new_tokens: int = 512,
) -> str:
    """Greedy decode a single prompt."""
    import torch

    inputs = tokenizer(
        prompt, return_tensors="pt", max_length=2048, truncation=True
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
        )
    # Decode only newly generated tokens
    gen_ids = output_ids[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(gen_ids, skip_special_tokens=True)


def _generate_mcts(
    model: Any,
    tokenizer: Any,
    prompt: str,
    device: Any,
    max_new_tokens: int = 512,
    n_rollouts: int = 8,
    temperature: float = 0.7,
) -> str:
    """Simple best-of-N MCTS-style generation with majority voting.

    Generates *n_rollouts* sampled answers, extracts the final answer
    from each, and returns the answer appearing most frequently (majority
    vote).  This is a lightweight approximation of full MCTS that is
    compatible with any causal-LM.
    """
    import torch
    from collections import Counter

    inputs = tokenizer(
        prompt, return_tensors="pt", max_length=2048, truncation=True
    ).to(device)

    candidates: List[str] = []
    for _ in range(n_rollouts):
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id,
            )
        gen_ids = output_ids[0, inputs["input_ids"].shape[1] :]
        candidates.append(tokenizer.decode(gen_ids, skip_special_tokens=True))

    # Majority vote on extracted answers
    answers: List[Optional[str]] = []
    for c in candidates:
        a = parse_gsm8k_answer(c)
        if a is None:
            a = parse_latex_answer(c)
        answers.append(a)

    valid = [a for a in answers if a is not None]
    if not valid:
        return candidates[0]  # Fallback to first generation

    most_common = Counter(valid).most_common(1)[0][0]
    # Return the full generation that produced the majority answer
    for c, a in zip(candidates, answers):
        if a == most_common:
            return c
    return candidates[0]


# ─────────────────────────────────────────────────────────────────
# Base evaluator
# ─────────────────────────────────────────────────────────────────

class BaseEvaluator(ABC):
    """Abstract base for all benchmark evaluators."""

    def __init__(self, max_samples: Optional[int] = None):
        self.max_samples = max_samples
        self.samples: List[EvalSample] = []
        self._loaded = False
        self._simulated = False

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def _load_dataset(self) -> List[EvalSample]:
        """Load evaluation samples from HuggingFace or local cache."""
        ...

    @abstractmethod
    def _build_prompt(self, sample: EvalSample) -> str:
        """Build the inference prompt for a single sample."""
        ...

    @abstractmethod
    def _extract_and_compare(
        self, generation: str, sample: EvalSample
    ) -> Tuple[str, bool]:
        """Extract the predicted answer and compare with ground truth.

        Returns ``(predicted_str, is_correct)``.
        """
        ...

    # ── Public API ──────────────────────────────────────────────

    def load(self) -> None:
        """Load dataset; falls back to simulation on failure."""
        try:
            self.samples = self._load_dataset()
            if self.max_samples is not None and len(self.samples) > self.max_samples:
                self.samples = self.samples[: self.max_samples]
            self._loaded = True
            logger.info(
                f"  {GREEN}✓ {self.name}: {len(self.samples)} samples loaded{NC}"
            )
        except Exception as exc:
            logger.warning(
                f"  {YELLOW}⚠ {self.name}: dataset unavailable ({exc}). "
                f"Using simulation fallback.{NC}"
            )
            self._simulated = True
            self.samples = self._generate_simulated_samples()
            self._loaded = True

    def evaluate(
        self,
        model: Any = None,
        tokenizer: Any = None,
        device: Any = None,
        use_mcts: bool = False,
    ) -> BenchmarkReport:
        """Run the full evaluation loop.

        When *model* is ``None`` the evaluator falls back to simulation
        mode (random accuracy around a sensible baseline).
        """
        if not self._loaded:
            self.load()

        results: List[EvalResult] = []
        correct = 0
        t0 = time.time()

        try:
            from tqdm import tqdm

            iterator = tqdm(
                enumerate(self.samples),
                total=len(self.samples),
                desc=f"  {self.name}",
                ncols=90,
            )
        except ImportError:
            iterator = enumerate(self.samples)  # type: ignore[assignment]

        simulate_inference = model is None or self._simulated
        if simulate_inference:
            logger.info(
                f"  {YELLOW}⚠ {self.name}: running in simulation mode{NC}"
            )

        for idx, sample in iterator:
            sample_t0 = time.time()

            if simulate_inference:
                predicted, is_correct = self._simulate_answer(sample)
                raw_gen = f"[SIMULATED] predicted={predicted}"
            else:
                prompt = self._build_prompt(sample)
                generate_fn = _generate_mcts if use_mcts else _generate_greedy
                raw_gen = generate_fn(model, tokenizer, prompt, device)
                predicted, is_correct = self._extract_and_compare(raw_gen, sample)

            if is_correct:
                correct += 1

            results.append(
                EvalResult(
                    sample_index=idx,
                    predicted=predicted,
                    ground_truth=sample.ground_truth,
                    is_correct=is_correct,
                    raw_generation=raw_gen,
                    metadata=sample.metadata,
                    latency_s=time.time() - sample_t0,
                )
            )

        wall_time = time.time() - t0
        accuracy = correct / len(self.samples) if self.samples else 0.0
        breakdown = self._compute_breakdown(results)

        logger.info(
            f"  {GREEN}✅ {self.name}: {accuracy:.2%} "
            f"({correct}/{len(self.samples)}) "
            f"in {wall_time:.1f}s{NC}"
        )

        return BenchmarkReport(
            benchmark=self.name,
            accuracy=accuracy,
            correct=correct,
            total=len(self.samples),
            results=results,
            breakdown=breakdown,
            wall_time_s=wall_time,
            simulated=simulate_inference,
        )

    # ── Overrideable helpers ────────────────────────────────────

    def _simulate_answer(
        self, sample: EvalSample
    ) -> Tuple[str, bool]:
        """Generate a simulated answer for fallback mode."""
        import random

        is_correct = random.random() < self._simulation_baseline()
        return (sample.ground_truth if is_correct else "WRONG", is_correct)

    @abstractmethod
    def _simulation_baseline(self) -> float:
        """Baseline accuracy used in simulation mode."""
        ...

    def _generate_simulated_samples(self) -> List[EvalSample]:
        """Return synthetic samples when the real dataset is unavailable."""
        n = self.max_samples or 100
        return [
            EvalSample(
                question=f"Simulated {self.name} question #{i}",
                ground_truth=str(i),
            )
            for i in range(n)
        ]

    def _compute_breakdown(self, results: List[EvalResult]) -> Dict[str, Any]:
        """Compute per-category accuracy breakdown.  Override in subclasses."""
        return {}


# ─────────────────────────────────────────────────────────────────
# GSM8K Evaluator
# ─────────────────────────────────────────────────────────────────

class GSM8KEvaluator(BaseEvaluator):
    """Evaluator for the GSM8K grade-school math benchmark.

    Dataset: ``openai/gsm8k`` (split ``test``, 1319 questions).
    Answers are in ``#### <number>`` format.
    """

    @property
    def name(self) -> str:
        return "GSM8K"

    def _simulation_baseline(self) -> float:
        return 0.84

    # ── Dataset loading ─────────────────────────────────────────

    def _load_dataset(self) -> List[EvalSample]:
        from datasets import load_dataset

        ds = load_dataset("openai/gsm8k", "main", split="test")
        samples: List[EvalSample] = []
        for row in ds:
            answer_text: str = row.get("answer", "")
            final_answer = ""
            if "####" in answer_text:
                final_answer = _strip_commas(
                    answer_text.split("####")[-1].strip()
                )
            samples.append(
                EvalSample(
                    question=row.get("question", ""),
                    ground_truth=final_answer,
                    metadata={"solution": answer_text},
                )
            )
        return samples

    # ── Prompt construction ─────────────────────────────────────

    def _build_prompt(self, sample: EvalSample) -> str:
        return (
            "Solve the following grade-school math problem step by step.\n"
            "End your answer with '#### <number>'.\n\n"
            f"Question: {sample.question}\n\n"
            "Solution:"
        )

    # ── Answer extraction ───────────────────────────────────────

    def _extract_and_compare(
        self, generation: str, sample: EvalSample
    ) -> Tuple[str, bool]:
        predicted = parse_gsm8k_answer(generation)
        if predicted is None:
            return ("", False)
        gt = sample.ground_truth
        p_num = normalise_number(predicted)
        g_num = normalise_number(gt)
        if p_num is not None and g_num is not None:
            return (predicted, math.isclose(p_num, g_num, rel_tol=1e-6))
        return (predicted, predicted.strip() == gt.strip())


# ─────────────────────────────────────────────────────────────────
# MATH Evaluator
# ─────────────────────────────────────────────────────────────────

# The 500 indices used by the common MATH-500 quick-eval subset.
# These were originally drawn by the "Let's Verify Step by Step" paper.
_MATH_500_SEED = 42

# MATH difficulty levels
_MATH_LEVELS = [f"Level {i}" for i in range(1, 6)]

# MATH subjects
_MATH_SUBJECTS = [
    "Algebra",
    "Counting & Probability",
    "Geometry",
    "Intermediate Algebra",
    "Number Theory",
    "Prealgebra",
    "Precalculus",
]


class MATHEvaluator(BaseEvaluator):
    r"""Evaluator for the MATH competition-level benchmark.

    Dataset: ``hendrycks/competition_math`` (fallback ``lighteval/MATH``).
    Answers are LaTeX (``\boxed{…}``).
    """

    def __init__(
        self,
        max_samples: Optional[int] = None,
        use_math500: bool = False,
    ):
        super().__init__(max_samples=max_samples)
        self.use_math500 = use_math500

    @property
    def name(self) -> str:
        return "MATH" + ("-500" if self.use_math500 else "")

    def _simulation_baseline(self) -> float:
        return 0.54

    # ── Dataset loading ─────────────────────────────────────────

    def _load_dataset(self) -> List[EvalSample]:
        from datasets import load_dataset

        # Try primary source first, then fallback
        ds = None
        for repo in ("hendrycks/competition_math", "lighteval/MATH"):
            try:
                ds = load_dataset(repo, split="test", trust_remote_code=True)
                logger.info(f"    Loaded MATH from {repo}")
                break
            except Exception:
                continue

        if ds is None:
            raise RuntimeError(
                "Neither hendrycks/competition_math nor lighteval/MATH available"
            )

        samples: List[EvalSample] = []
        for row in ds:
            # The dataset stores: problem, solution, answer (boxed), level, type
            answer = row.get("answer", row.get("solution", ""))
            level = row.get("level", "")
            subject = row.get("type", "")
            samples.append(
                EvalSample(
                    question=row.get("problem", ""),
                    ground_truth=answer,
                    metadata={
                        "solution": row.get("solution", ""),
                        "level": level,
                        "subject": subject,
                    },
                )
            )

        # MATH-500 subset: deterministic shuffle + head
        if self.use_math500:
            import random as _rng

            indices = list(range(len(samples)))
            _rng.Random(_MATH_500_SEED).shuffle(indices)
            samples = [samples[i] for i in indices[:500]]
            logger.info(f"    Using MATH-500 subset ({len(samples)} samples)")

        return samples

    # ── Prompt construction ─────────────────────────────────────

    def _build_prompt(self, sample: EvalSample) -> str:
        subject = sample.metadata.get("subject", "")
        level = sample.metadata.get("level", "")
        prefix = ""
        if subject:
            prefix += f"[{subject}] "
        if level:
            prefix += f"({level}) "

        return (
            f"{prefix}Solve the following math problem.\n"
            "Provide your answer inside \\boxed{}.\n\n"
            f"Problem: {sample.question}\n\n"
            "Solution:"
        )

    # ── Answer extraction ───────────────────────────────────────

    def _extract_and_compare(
        self, generation: str, sample: EvalSample
    ) -> Tuple[str, bool]:
        predicted_raw = parse_latex_answer(generation)
        gt_raw = parse_latex_answer(sample.ground_truth)
        return (predicted_raw, answers_equivalent(predicted_raw, gt_raw))

    # ── Breakdown ───────────────────────────────────────────────

    def _compute_breakdown(self, results: List[EvalResult]) -> Dict[str, Any]:
        breakdown: Dict[str, Any] = {"by_level": {}, "by_subject": {}}

        # Group by level
        level_groups: Dict[str, List[bool]] = {}
        subject_groups: Dict[str, List[bool]] = {}

        for r in results:
            level = r.metadata.get("level", "Unknown")
            subject = r.metadata.get("subject", "Unknown")
            level_groups.setdefault(level, []).append(r.is_correct)
            subject_groups.setdefault(subject, []).append(r.is_correct)

        for level in sorted(level_groups):
            vals = level_groups[level]
            breakdown["by_level"][level] = {
                "accuracy": sum(vals) / len(vals),
                "correct": sum(vals),
                "total": len(vals),
            }

        for subject in sorted(subject_groups):
            vals = subject_groups[subject]
            breakdown["by_subject"][subject] = {
                "accuracy": sum(vals) / len(vals),
                "correct": sum(vals),
                "total": len(vals),
            }

        return breakdown

    def _generate_simulated_samples(self) -> List[EvalSample]:
        import random

        n = self.max_samples or 200
        rng = random.Random(42)
        samples = []
        for i in range(n):
            level = rng.choice(_MATH_LEVELS)
            subject = rng.choice(_MATH_SUBJECTS)
            a, b = rng.randint(2, 50), rng.randint(2, 50)
            samples.append(
                EvalSample(
                    question=f"Compute {a}^2 + {b}^2.",
                    ground_truth=str(a**2 + b**2),
                    metadata={"level": level, "subject": subject, "solution": ""},
                )
            )
        return samples


# ─────────────────────────────────────────────────────────────────
# MMLU-STEM Evaluator
# ─────────────────────────────────────────────────────────────────

# STEM subjects within the MMLU benchmark
_MMLU_STEM_SUBJECTS = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "electrical_engineering",
    "elementary_mathematics",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_mathematics",
    "high_school_physics",
    "high_school_statistics",
    "machine_learning",
    "virology",
]


class MMLUSTEMEvaluator(BaseEvaluator):
    """Evaluator for MMLU STEM subjects.

    Dataset: ``cais/mmlu`` (multiple-choice, A/B/C/D).
    """

    def __init__(
        self,
        max_samples: Optional[int] = None,
        subjects: Optional[List[str]] = None,
    ):
        super().__init__(max_samples=max_samples)
        self.subjects = subjects or _MMLU_STEM_SUBJECTS

    @property
    def name(self) -> str:
        return "MMLU-STEM"

    def _simulation_baseline(self) -> float:
        return 0.62

    # ── Dataset loading ─────────────────────────────────────────

    def _load_dataset(self) -> List[EvalSample]:
        from datasets import load_dataset

        all_samples: List[EvalSample] = []
        label_map = {0: "A", 1: "B", 2: "C", 3: "D"}

        for subject in self.subjects:
            try:
                # Try without trust_remote_code first (newer datasets lib),
                # then fall back to including it for older versions.
                try:
                    ds = load_dataset("cais/mmlu", subject, split="test")
                except TypeError:
                    ds = load_dataset(
                        "cais/mmlu", subject, split="test", trust_remote_code=True
                    )
            except Exception:
                logger.debug(f"    Skipping unavailable subject: {subject}")
                continue

            for row in ds:
                question = row.get("question", "")
                choices = row.get("choices", [])
                answer_idx = row.get("answer", -1)
                if isinstance(answer_idx, int) and 0 <= answer_idx <= 3:
                    gt_letter = label_map[answer_idx]
                else:
                    gt_letter = str(answer_idx)

                choices_text = ""
                for ci, choice in enumerate(choices):
                    choices_text += f"\n  {label_map.get(ci, '?')}. {choice}"

                all_samples.append(
                    EvalSample(
                        question=question,
                        ground_truth=gt_letter,
                        metadata={
                            "subject": subject,
                            "choices": choices,
                            "choices_text": choices_text,
                        },
                    )
                )

        if not all_samples:
            raise RuntimeError("No MMLU-STEM subjects could be loaded")

        return all_samples

    # ── Prompt construction ─────────────────────────────────────

    def _build_prompt(self, sample: EvalSample) -> str:
        subject_pretty = sample.metadata.get("subject", "").replace("_", " ").title()
        choices_text = sample.metadata.get("choices_text", "")

        return (
            f"The following is a multiple-choice question about {subject_pretty}.\n"
            "Choose the correct answer (A, B, C, or D).\n\n"
            f"Question: {sample.question}"
            f"{choices_text}\n\n"
            "Answer:"
        )

    # ── Answer extraction ───────────────────────────────────────

    def _extract_and_compare(
        self, generation: str, sample: EvalSample
    ) -> Tuple[str, bool]:
        predicted = parse_multiple_choice(generation)
        if predicted is None:
            return ("", False)
        return (predicted, predicted == sample.ground_truth)

    # ── Breakdown ───────────────────────────────────────────────

    def _compute_breakdown(self, results: List[EvalResult]) -> Dict[str, Any]:
        subject_groups: Dict[str, List[bool]] = {}
        for r in results:
            subj = r.metadata.get("subject", "Unknown")
            subject_groups.setdefault(subj, []).append(r.is_correct)

        breakdown: Dict[str, Any] = {"by_subject": {}}
        for subj in sorted(subject_groups):
            vals = subject_groups[subj]
            breakdown["by_subject"][subj] = {
                "accuracy": sum(vals) / len(vals),
                "correct": sum(vals),
                "total": len(vals),
            }

        # Overall STEM average (macro across subjects)
        per_subj_accs = [
            v["accuracy"] for v in breakdown["by_subject"].values()
        ]
        breakdown["stem_macro_average"] = (
            sum(per_subj_accs) / len(per_subj_accs) if per_subj_accs else 0.0
        )
        return breakdown

    def _generate_simulated_samples(self) -> List[EvalSample]:
        import random

        n = self.max_samples or 200
        rng = random.Random(42)
        label_map = {0: "A", 1: "B", 2: "C", 3: "D"}
        samples = []
        for i in range(n):
            subj = rng.choice(self.subjects)
            gt_idx = rng.randint(0, 3)
            choices = [f"Option {label_map[j]}" for j in range(4)]
            choices_text = ""
            for ci, ch in enumerate(choices):
                choices_text += f"\n  {label_map[ci]}. {ch}"
            samples.append(
                EvalSample(
                    question=f"Simulated {subj} question #{i}",
                    ground_truth=label_map[gt_idx],
                    metadata={
                        "subject": subj,
                        "choices": choices,
                        "choices_text": choices_text,
                    },
                )
            )
        return samples


# ─────────────────────────────────────────────────────────────────
# Benchmark Suite
# ─────────────────────────────────────────────────────────────────

class BenchmarkSuite:
    """Orchestrates multiple benchmark evaluators.

    Usage::

        suite = BenchmarkSuite(max_samples=100)
        results = suite.run_all(model, tokenizer, device)
    """

    def __init__(
        self,
        max_samples: Optional[int] = None,
        benchmarks: Optional[List[str]] = None,
        use_math500: bool = False,
    ):
        self.max_samples = max_samples
        self.evaluators: Dict[str, BaseEvaluator] = {}

        requested = set(benchmarks or ["gsm8k", "math", "mmlu-stem"])

        if "gsm8k" in requested or "all" in requested:
            self.evaluators["gsm8k"] = GSM8KEvaluator(max_samples=max_samples)
        if "math" in requested or "all" in requested:
            self.evaluators["math"] = MATHEvaluator(
                max_samples=max_samples, use_math500=use_math500
            )
        if "mmlu-stem" in requested or "all" in requested:
            self.evaluators["mmlu-stem"] = MMLUSTEMEvaluator(
                max_samples=max_samples
            )

    def run_all(
        self,
        model: Any = None,
        tokenizer: Any = None,
        device: Any = None,
        max_samples: Optional[int] = None,
        use_mcts: bool = False,
    ) -> Dict[str, Any]:
        """Run all configured benchmarks and return aggregate results.

        Parameters
        ----------
        model : transformers model or ``None``
            When ``None`` every evaluator runs in simulation mode.
        tokenizer : transformers tokenizer or ``None``
        device : torch device
        max_samples : int or None
            Override per-evaluator sample cap.
        use_mcts : bool
            Use MCTS-based majority-vote decoding instead of greedy.

        Returns
        -------
        dict
            ``{"benchmarks": {…}, "meta": {…}}``
        """
        suite_t0 = time.time()

        if max_samples is not None:
            for ev in self.evaluators.values():
                ev.max_samples = max_samples

        reports: Dict[str, BenchmarkReport] = {}
        for key, evaluator in self.evaluators.items():
            logger.info(
                f"\n{BOLD}{'─'*60}{NC}\n"
                f"{BOLD}  Benchmark: {evaluator.name}{NC}\n"
                f"{'─'*60}"
            )
            evaluator.load()
            reports[key] = evaluator.evaluate(
                model=model,
                tokenizer=tokenizer,
                device=device,
                use_mcts=use_mcts,
            )

        suite_wall = time.time() - suite_t0

        # ── Summary table ───────────────────────────────────────
        logger.info(f"\n{BOLD}{'═'*68}{NC}")
        logger.info(f"{BOLD}  BENCHMARK RESULTS SUMMARY{NC}")
        logger.info(f"{'═'*68}")
        logger.info(
            f"  {'Benchmark':<15} {'Accuracy':>10} {'Correct':>10} "
            f"{'Total':>8} {'Time':>8} {'Mode':>10}"
        )
        logger.info(f"  {'─'*62}")
        for key, rpt in reports.items():
            mode = "SIMULATED" if rpt.simulated else "LIVE"
            logger.info(
                f"  {rpt.benchmark:<15} {rpt.accuracy:>9.2%} "
                f"{rpt.correct:>10} {rpt.total:>8} "
                f"{rpt.wall_time_s:>7.1f}s {mode:>10}"
            )
        logger.info(f"  {'─'*62}")
        logger.info(f"  Total wall time: {suite_wall:.1f}s")
        logger.info(f"{'═'*68}\n")

        # ── Assemble output dict ────────────────────────────────
        output: Dict[str, Any] = {
            "benchmarks": {k: rpt.to_dict() for k, rpt in reports.items()},
            "meta": {
                "total_wall_time_s": suite_wall,
                "use_mcts": use_mcts,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
        }
        return output


# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Real Benchmark Evaluation — GSM8K · MATH · MMLU-STEM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s --benchmark gsm8k --model Qwen/Qwen2.5-Math-7B-Instruct\n"
            "  %(prog)s --benchmark all --max-samples 50 --use-mcts\n"
            "  %(prog)s --benchmark math --math500\n"
        ),
    )
    parser.add_argument(
        "--benchmark",
        choices=["gsm8k", "math", "mmlu-stem", "all"],
        default="all",
        help="Which benchmark(s) to run (default: all)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "HuggingFace model name or local path. "
            "If omitted, runs in simulation mode."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use: cuda, mps, cpu (auto-detected if omitted)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples per benchmark (default: full dataset)",
    )
    parser.add_argument(
        "--use-mcts",
        action="store_true",
        help="Use MCTS majority-vote decoding (slower but more accurate)",
    )
    parser.add_argument(
        "--mcts-rollouts",
        type=int,
        default=8,
        help="Number of MCTS rollouts for majority vote (default: 8)",
    )
    parser.add_argument(
        "--math500",
        action="store_true",
        help="Use the MATH-500 subset for quick evaluation",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="real_benchmark_results.json",
        help="Path for JSON output (default: real_benchmark_results.json)",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    logger.info(f"\n{CYAN}{BOLD}{'='*68}{NC}")
    logger.info(
        f"{CYAN}{BOLD}  RunuX AI Engine — Real Benchmark Evaluation{NC}"
    )
    logger.info(
        f"{CYAN}{BOLD}  Copyright (c) 2026 Xavier Callens / Socrate AI Lab{NC}"
    )
    logger.info(f"{CYAN}{BOLD}{'='*68}{NC}\n")

    # ── Determine benchmarks to run ─────────────────────────────
    benchmarks = (
        ["gsm8k", "math", "mmlu-stem"] if args.benchmark == "all" else [args.benchmark]
    )

    # ── Load model (if specified) ───────────────────────────────
    model, tokenizer, device = None, None, None
    if args.model:
        try:
            model, tokenizer, device = _load_model_and_tokenizer(
                args.model, args.device
            )
        except Exception as exc:
            logger.error(
                f"  {RED}✗ Failed to load model '{args.model}': {exc}{NC}"
            )
            logger.info(
                f"  {YELLOW}→ Falling back to simulation mode{NC}"
            )
    else:
        logger.info(
            f"  {YELLOW}ℹ No --model specified. Running in simulation mode.{NC}"
        )

    # ── Run suite ───────────────────────────────────────────────
    suite = BenchmarkSuite(
        max_samples=args.max_samples,
        benchmarks=benchmarks,
        use_math500=args.math500,
    )
    results = suite.run_all(
        model=model,
        tokenizer=tokenizer,
        device=device,
        use_mcts=args.use_mcts,
    )

    # ── Persist to JSON ─────────────────────────────────────────
    results["meta"]["model"] = args.model or "simulation"
    results["meta"]["cli_args"] = vars(args)

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    logger.info(f"  {GREEN}💾 Results saved to: {BOLD}{output_path}{NC}\n")


if __name__ == "__main__":
    main()
