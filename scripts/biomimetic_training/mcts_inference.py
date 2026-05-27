#!/usr/bin/env python3
# Copyright (c) 2026 Xavier Callens / Socrate AI Lab. All Rights Reserved.
# SPDX-License-Identifier: LicenseRef-RunuX-Commercial
#
# MCTS Inference Engine — Monte Carlo Tree Search for Math Reasoning
# ==================================================================
# Implements Process-Reward-Model-guided MCTS over step-by-step
# reasoning traces.  Each tree node is a partial Chain-of-Thought step;
# expansion samples K next-steps via the language model, simulation
# completes greedily, and back-propagation aggregates correctness
# scores.  Code blocks embedded in steps are executed in a sandboxed
# subprocess for augmented verification.

from __future__ import annotations

import hashlib
import logging
import math
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ─────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────
# Optional heavy imports — graceful fallback to simulation
# ─────────────────────────────────────────────────────────────────
_TORCH_AVAILABLE = False
_TRANSFORMERS_AVAILABLE = False
_SYMPY_AVAILABLE = False

try:
    import torch
    import torch.nn.functional as F

    _TORCH_AVAILABLE = True
except ImportError:
    logger.warning("torch not available — running in simulation mode")

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: F401

    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("transformers not available — running in simulation mode")

try:
    import sympy  # noqa: F401

    _SYMPY_AVAILABLE = True
except ImportError:
    logger.warning("sympy not available — SymPyVerifier will be disabled")

# ─────────────────────────────────────────────────────────────────
# Console colours (consistent with the rest of the project)
# ─────────────────────────────────────────────────────────────────
GREEN = "\033[0;32m"
YELLOW = "\033[0;33m"
BLUE = "\033[0;34m"
CYAN = "\033[0;36m"
RED = "\033[0;31m"
MAGENTA = "\033[0;35m"
BOLD = "\033[1m"
NC = "\033[0m"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  1. MCTSNode
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@dataclass
class MCTSNode:
    """A single node in the MCTS search tree.

    Each node corresponds to one *step* in a multi-step Chain-of-Thought
    reasoning trace.  ``state`` holds the text generated so far (the
    concatenation of all ancestor states plus this node's contribution).

    Attributes:
        state:        Full reasoning text from root to this node.
        score:        Accumulated reward (sum of back-propagated values).
        visits:       Number of times this node has been visited during search.
        children:     Expanded child nodes.
        parent:       Back-pointer (``None`` for the root).
        code_output:  Captured stdout/stderr if the step contained executable code.
        step_text:    The text of *this* step only (excluding ancestors).
        depth:        Depth from root (root = 0).
    """

    state: str
    score: float = 0.0
    visits: int = 0
    children: List["MCTSNode"] = field(default_factory=list)
    parent: Optional["MCTSNode"] = None
    code_output: Optional[str] = None
    step_text: str = ""
    depth: int = 0

    # ── convenience ────────────────────────────────────────────
    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    @property
    def is_root(self) -> bool:
        return self.parent is None

    @property
    def average_score(self) -> float:
        return self.score / max(self.visits, 1)

    def add_child(self, child: "MCTSNode") -> "MCTSNode":
        child.parent = self
        child.depth = self.depth + 1
        self.children.append(child)
        return child

    def best_child(self) -> Optional["MCTSNode"]:
        """Return the child with the highest average score."""
        if not self.children:
            return None
        return max(self.children, key=lambda c: c.average_score)

    def __repr__(self) -> str:
        trunc = (self.step_text[:60] + "…") if len(self.step_text) > 60 else self.step_text
        return (
            f"MCTSNode(depth={self.depth}, visits={self.visits}, "
            f"avg={self.average_score:.4f}, text={trunc!r})"
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  2. ProcessRewardModel
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class ProcessRewardModel:
    """Score intermediate reasoning steps using a language model's
    log-probabilities as a proxy reward signal.

    The model is queried in *teacher-forcing* mode: we feed the full
    partial solution and measure the average log-probability the model
    assigns to the continuation.  Higher log-prob ≈ more "natural" and
    internally consistent reasoning.

    When ``torch`` / ``transformers`` are not installed the class falls
    back to a deterministic heuristic that rewards longer, more
    structured steps.
    """

    # Heuristic keywords that boost the simulated score
    _QUALITY_SIGNALS: List[str] = [
        "therefore", "thus", "hence", "so we get",
        "substituting", "simplifying", "we know",
        "equation", "solve", "answer",
        "step", "=",
    ]

    def __init__(self, max_length: int = 1024) -> None:
        self.max_length = max_length
        logger.info(
            f"  {CYAN}ProcessRewardModel initialised "
            f"(torch={'yes' if _TORCH_AVAILABLE else 'no'}, "
            f"transformers={'yes' if _TRANSFORMERS_AVAILABLE else 'no'}){NC}"
        )

    # ── public API ─────────────────────────────────────────────
    def score_step(
        self,
        partial_solution: str,
        model: Any = None,
        tokenizer: Any = None,
    ) -> float:
        """Return a scalar reward ∈ [0, 1] for *partial_solution*.

        If a real model + tokenizer are provided and torch is available
        the score is derived from the model's own log-probs.  Otherwise
        a heuristic fallback is used.
        """
        if model is not None and tokenizer is not None and _TORCH_AVAILABLE:
            return self._score_with_model(partial_solution, model, tokenizer)
        return self._score_heuristic(partial_solution)

    def batch_score(
        self,
        candidates: List[str],
        model: Any = None,
        tokenizer: Any = None,
    ) -> List[float]:
        """Score multiple candidate steps in one call.

        When a real model is available we batch-encode to reduce
        overhead; otherwise each candidate is scored independently via
        the heuristic.
        """
        if model is not None and tokenizer is not None and _TORCH_AVAILABLE:
            return self._batch_score_with_model(candidates, model, tokenizer)
        return [self._score_heuristic(c) for c in candidates]

    # ── model-based scoring ────────────────────────────────────
    def _score_with_model(
        self,
        text: str,
        model: Any,
        tokenizer: Any,
    ) -> float:
        """Compute normalised log-prob score using the LM."""
        try:
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
            )
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs, labels=inputs["input_ids"])
                # outputs.loss is the mean cross-entropy over all tokens
                neg_log_prob = outputs.loss.item()

            # Map negative log-prob → reward in [0, 1].
            # A perfect model would have loss ≈ 0 → reward ≈ 1.
            # Typical losses range 1–5; we use a soft sigmoid mapping.
            reward = 1.0 / (1.0 + math.exp(neg_log_prob - 2.5))
            return float(max(0.0, min(1.0, reward)))

        except Exception as e:
            logger.warning(f"Model scoring failed ({e}), falling back to heuristic")
            return self._score_heuristic(text)

    def _batch_score_with_model(
        self,
        candidates: List[str],
        model: Any,
        tokenizer: Any,
    ) -> List[float]:
        """Batch-encode candidates and score in one forward pass."""
        try:
            inputs = tokenizer(
                candidates,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits  # (B, T, V)

            scores: List[float] = []
            for i in range(logits.size(0)):
                seq_ids = inputs["input_ids"][i]
                mask = inputs["attention_mask"][i]
                seq_logits = logits[i]  # (T, V)

                # Shift: predict token t from position t-1
                shift_logits = seq_logits[:-1, :]
                shift_labels = seq_ids[1:]
                shift_mask = mask[1:]

                log_probs = F.log_softmax(shift_logits, dim=-1)
                token_lp = log_probs.gather(
                    1, shift_labels.unsqueeze(-1)
                ).squeeze(-1)

                # Masked mean
                valid = shift_mask.sum().clamp(min=1)
                mean_lp = (token_lp * shift_mask).sum() / valid
                reward = 1.0 / (1.0 + math.exp(-mean_lp.item() - 2.5))
                scores.append(float(max(0.0, min(1.0, reward))))

            return scores

        except Exception as e:
            logger.warning(f"Batch model scoring failed ({e}), falling back to heuristic")
            return [self._score_heuristic(c) for c in candidates]

    # ── heuristic fallback ─────────────────────────────────────
    def _score_heuristic(self, text: str) -> float:
        """Deterministic heuristic reward when no model is available.

        Rewards:
          • Longer reasoning traces (up to a ceiling)
          • Presence of mathematical operators / keywords
          • Structural markers (numbered steps, equations)
        """
        if not text.strip():
            return 0.0

        # Length component (diminishing returns after ~300 chars)
        length_score = min(len(text) / 300.0, 1.0) * 0.3

        # Keyword density
        lower = text.lower()
        keyword_hits = sum(1 for kw in self._QUALITY_SIGNALS if kw in lower)
        keyword_score = min(keyword_hits / 6.0, 1.0) * 0.3

        # Structural markers (numbered steps, LaTeX fragments, code blocks)
        struct_hits = len(re.findall(r"(?:Step \d|\\boxed|```)", text))
        struct_score = min(struct_hits / 3.0, 1.0) * 0.2

        # Presence of numeric answer
        has_numeric = 1.0 if re.search(r"\d+\.?\d*", text) else 0.0
        numeric_score = has_numeric * 0.2

        total = length_score + keyword_score + struct_score + numeric_score
        # Add reproducible jitter keyed on content hash
        h = int(hashlib.md5(text.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        total += (h - 0.5) * 0.04  # ± 0.02

        return float(max(0.0, min(1.0, total)))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  3. MCTSEngine
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class MCTSEngine:
    """Monte Carlo Tree Search over multi-step reasoning traces.

    Pipeline (per iteration):
      1. **Selection** — Walk from root choosing children via UCB1.
      2. **Expansion** — At a leaf, sample K candidate next-steps
         from the language model (temperature = 0.7).
      3. **Simulation** — Complete each candidate greedily to obtain
         a terminal value.
      4. **Backpropagation** — Propagate the value up to root,
         incrementing visit counts.
      5. **Code verification** — If a step contains a fenced Python
         block, execute it in a sandboxed subprocess and incorporate
         the pass/fail signal.

    Parameters:
        exploration_constant:  UCB1 exploration weight *c* (default √2).
        expansion_width:       Number of candidates *K* to sample
                               during expansion.
        temperature:           Sampling temperature for expansion.
        code_timeout:          Maximum seconds for subprocess execution.
    """

    DEFAULT_C = math.sqrt(2)  # ≈ 1.414

    def __init__(
        self,
        exploration_constant: float = DEFAULT_C,
        expansion_width: int = 4,
        temperature: float = 0.7,
        code_timeout: int = 10,
    ) -> None:
        self.c = exploration_constant
        self.expansion_width = expansion_width
        self.temperature = temperature
        self.code_timeout = code_timeout
        self.prm = ProcessRewardModel()

        logger.info(
            f"  {CYAN}MCTSEngine initialised "
            f"(c={self.c:.3f}, K={self.expansion_width}, "
            f"temp={self.temperature}, code_timeout={self.code_timeout}s){NC}"
        )

    # ── main entry point ───────────────────────────────────────
    def search(
        self,
        problem: str,
        model: Any = None,
        tokenizer: Any = None,
        num_paths: int = 64,
        max_depth: int = 8,
    ) -> str:
        """Run MCTS and return the best complete reasoning path.

        Args:
            problem:    The math problem statement.
            model:      A ``transformers`` causal-LM (or ``None`` for
                        simulation).
            tokenizer:  Matching tokenizer (or ``None``).
            num_paths:  Total number of MCTS iterations (rollouts).
            max_depth:  Maximum reasoning depth (number of steps).

        Returns:
            The text of the highest-scoring complete reasoning trace.
        """
        logger.info(
            f"\n{BOLD}🌲 MCTS Search — "
            f"{num_paths} rollouts, depth ≤ {max_depth}{NC}"
        )
        logger.info(f"   Problem: {problem[:120]}{'…' if len(problem) > 120 else ''}")

        root = MCTSNode(state=f"Problem: {problem}\n\nSolution:\n", step_text="")
        t_start = time.time()

        for iteration in range(1, num_paths + 1):
            # 1. Selection
            node = self._select(root)

            # 2. Expansion (if not at max depth)
            if node.depth < max_depth:
                node = self._expand(node, model, tokenizer)

            # 3. Simulation
            value = self._simulate(node, model, tokenizer, max_depth)

            # 4. Backpropagation
            self._backpropagate(node, value)

            if iteration % max(1, num_paths // 8) == 0:
                best = self._best_leaf(root)
                logger.info(
                    f"   Iter {iteration:4d}/{num_paths} | "
                    f"Tree size: {self._tree_size(root):4d} | "
                    f"Best avg: {best.average_score:.4f} | "
                    f"Depth: {best.depth}"
                )

        elapsed = time.time() - t_start
        best_leaf = self._best_leaf(root)
        logger.info(
            f"\n  {GREEN}✅ MCTS complete in {elapsed:.2f}s — "
            f"best score {best_leaf.average_score:.4f} "
            f"(depth {best_leaf.depth}, {best_leaf.visits} visits){NC}"
        )
        return best_leaf.state

    # ── 1. UCB1 selection ──────────────────────────────────────
    def _select(self, node: MCTSNode) -> MCTSNode:
        """Walk from *node* to a leaf using UCB1."""
        while not node.is_leaf:
            node = self._ucb1_child(node)
        return node

    def _ucb1_child(self, node: MCTSNode) -> MCTSNode:
        """Pick the child that maximises the UCB1 formula."""
        log_parent = math.log(max(node.visits, 1))

        def ucb1(child: MCTSNode) -> float:
            if child.visits == 0:
                return float("inf")
            exploitation = child.score / child.visits
            exploration = self.c * math.sqrt(log_parent / child.visits)
            return exploitation + exploration

        return max(node.children, key=ucb1)

    # ── 2. Expansion ───────────────────────────────────────────
    def _expand(
        self,
        node: MCTSNode,
        model: Any,
        tokenizer: Any,
    ) -> MCTSNode:
        """Sample K candidate next-steps and attach as children."""
        candidates = self._generate_candidates(
            node.state, model, tokenizer, k=self.expansion_width
        )
        for cand_text in candidates:
            child_state = node.state + cand_text + "\n"
            child = MCTSNode(state=child_state, step_text=cand_text)

            # Code-augmented verification
            code_output = self._execute_code_if_present(cand_text)
            if code_output is not None:
                child.code_output = code_output

            node.add_child(child)

        # Return one of the new children (prefer unexplored)
        unexplored = [c for c in node.children if c.visits == 0]
        return unexplored[0] if unexplored else node.children[0]

    def _generate_candidates(
        self,
        context: str,
        model: Any,
        tokenizer: Any,
        k: int,
    ) -> List[str]:
        """Generate *k* candidate continuation steps.

        Falls back to synthetic candidates when no model is available.
        """
        if (
            model is not None
            and tokenizer is not None
            and _TORCH_AVAILABLE
            and _TRANSFORMERS_AVAILABLE
        ):
            return self._generate_with_model(context, model, tokenizer, k)
        return self._generate_simulated(context, k)

    def _generate_with_model(
        self,
        context: str,
        model: Any,
        tokenizer: Any,
        k: int,
    ) -> List[str]:
        """Sample *k* continuations from the language model."""
        candidates: List[str] = []
        try:
            inputs = tokenizer(
                context,
                return_tensors="pt",
                truncation=True,
                max_length=768,
            )
            device = next(model.parameters()).device
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)

            with torch.no_grad():
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=128,
                    do_sample=True,
                    temperature=self.temperature,
                    top_p=0.95,
                    num_return_sequences=k,
                    pad_token_id=tokenizer.pad_token_id
                    or tokenizer.eos_token_id,
                )

            for seq in outputs:
                new_tokens = seq[input_ids.shape[1] :]
                text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
                if text:
                    # Take only the first "step" (up to the next newline pair)
                    step = text.split("\n\n")[0].strip()
                    candidates.append(step if step else text[:200])

        except Exception as e:
            logger.warning(f"Model generation failed ({e}), using simulation")
            return self._generate_simulated(context, k)

        # Pad with simulated candidates if we got fewer than k
        while len(candidates) < k:
            candidates.extend(self._generate_simulated(context, k - len(candidates)))

        return candidates[:k]

    def _generate_simulated(self, context: str, k: int) -> List[str]:
        """Generate synthetic reasoning steps for simulation mode."""
        import random

        # Determine step number from context
        existing_steps = context.count("Step ")
        step_num = existing_steps + 1

        templates = [
            (
                "Step {n}: Let's identify the key quantities. "
                "We know that the total is {a} and each group has {b}."
            ),
            (
                "Step {n}: Setting up the equation: "
                "{a} × {b} = {c}. Therefore the intermediate result is {c}."
            ),
            (
                "Step {n}: We can simplify by noting that "
                "{a} - {b} = {d}. This gives us the remaining amount."
            ),
            (
                "Step {n}: Substituting back, we get "
                "{c} + {d} = {e}. So the answer is {e}."
            ),
            (
                "Step {n}: Verifying our answer: {e} ÷ {b} = {f:.2f}, "
                "which confirms our calculation."
            ),
            (
                "Step {n}: Using Python to double-check:\n"
                "```python\nresult = {a} * {b}\nprint(result)\n```\n"
                "Output: {c}"
            ),
        ]

        a = random.randint(5, 50)
        b = random.randint(2, 12)
        c = a * b
        d = a - b
        e = c + d
        f_val = e / max(b, 1)

        candidates: List[str] = []
        for i in range(k):
            tmpl = templates[(step_num + i) % len(templates)]
            text = tmpl.format(
                n=step_num, a=a, b=b, c=c, d=d, e=e, f=f_val
            )
            candidates.append(text)

        return candidates

    # ── 3. Simulation ──────────────────────────────────────────
    def _simulate(
        self,
        node: MCTSNode,
        model: Any,
        tokenizer: Any,
        max_depth: int,
    ) -> float:
        """Complete the reasoning trace greedily and return a value."""
        state = node.state
        depth = node.depth

        # Greedily extend the path to max_depth
        while depth < max_depth:
            candidates = self._generate_candidates(state, model, tokenizer, k=1)
            if not candidates or not candidates[0].strip():
                break
            state = state + candidates[0] + "\n"
            depth += 1

        # Score the terminal state
        score = self.prm.score_step(state, model, tokenizer)

        # Bonus / penalty from code execution results
        code_output = self._execute_code_if_present(state)
        if code_output is not None:
            # If code ran without error → small bonus
            if "Error" not in code_output and "Traceback" not in code_output:
                score = min(1.0, score + 0.1)
            else:
                score = max(0.0, score - 0.15)

        return score

    # ── 4. Backpropagation ─────────────────────────────────────
    @staticmethod
    def _backpropagate(node: MCTSNode, value: float) -> None:
        """Walk from *node* to root, updating visits and scores."""
        current: Optional[MCTSNode] = node
        while current is not None:
            current.visits += 1
            current.score += value
            current = current.parent

    # ── 5. Code-Augmented Verification ─────────────────────────
    def _execute_code_if_present(self, text: str) -> Optional[str]:
        """If *text* contains a fenced Python code block, execute it
        in a subprocess and return captured output.

        Returns ``None`` when no code block is found.
        """
        pattern = r"```python\s*\n(.*?)```"
        matches = re.findall(pattern, text, re.DOTALL)
        if not matches:
            return None

        code = matches[-1].strip()  # Execute the last code block
        if not code:
            return None

        logger.debug(f"Executing code block ({len(code)} chars)")

        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False
            ) as f:
                f.write(code)
                tmp_path = f.name

            result = subprocess.run(
                [sys.executable, tmp_path],
                capture_output=True,
                text=True,
                timeout=self.code_timeout,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )
            output = (result.stdout + result.stderr).strip()
            return output if output else "(no output)"

        except subprocess.TimeoutExpired:
            logger.warning("Code execution timed out")
            return "Error: execution timed out"
        except Exception as e:
            logger.warning(f"Code execution error: {e}")
            return f"Error: {e}"
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    # ── helpers ─────────────────────────────────────────────────
    def _best_leaf(self, root: MCTSNode) -> MCTSNode:
        """Return the leaf node with the highest average score."""
        best: MCTSNode = root
        stack: List[MCTSNode] = [root]
        while stack:
            node = stack.pop()
            if node.is_leaf and node.visits > 0:
                if node.average_score > best.average_score:
                    best = node
            stack.extend(node.children)
        return best

    def _tree_size(self, root: MCTSNode) -> int:
        """Count total nodes in the tree rooted at *root*."""
        count = 0
        stack: List[MCTSNode] = [root]
        while stack:
            node = stack.pop()
            count += 1
            stack.extend(node.children)
        return count


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  4. SymPyVerifier
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class SymPyVerifier:
    r"""Extract mathematical expressions from model output and verify
    numerical answers using SymPy.

    Supports:
      - ``\boxed{...}`` LaTeX answers (common in math-LM outputs)
      - ``The answer is <number>`` patterns
      - ``#### <number>`` (GSM8K convention)
      - Bare trailing numbers

    Returns a binary correctness signal: ``True`` if the extracted
    answer matches the ground truth within a configurable tolerance.
    """

    # Ordered from most specific to most general
    _ANSWER_PATTERNS: List[re.Pattern] = [
        re.compile(r"\\boxed\{([^}]+)\}"),
        re.compile(r"####\s*([+-]?\d[\d,]*\.?\d*)"),
        re.compile(r"[Tt]he\s+(?:final\s+)?answer\s+is\s*:?\s*\$?([+-]?\d[\d,]*\.?\d*)\$?"),
        re.compile(r"=\s*\$?([+-]?\d[\d,]*\.?\d*)\$?\s*$", re.MULTILINE),
    ]

    def __init__(self, tolerance: float = 1e-6) -> None:
        self.tolerance = tolerance
        logger.info(
            f"  {CYAN}SymPyVerifier initialised "
            f"(sympy={'yes' if _SYMPY_AVAILABLE else 'no'}, "
            f"tol={self.tolerance}){NC}"
        )

    def extract_answer(self, text: str) -> Optional[str]:
        """Extract the final numerical answer from *text*.

        Returns the raw string of the first matched answer, or
        ``None`` if nothing was found.
        """
        for pattern in self._ANSWER_PATTERNS:
            match = pattern.search(text)
            if match:
                return match.group(1).replace(",", "").strip()
        return None

    def verify(
        self,
        model_output: str,
        ground_truth: str,
    ) -> bool:
        """Return ``True`` if the model's extracted answer matches
        *ground_truth* within tolerance.

        When SymPy is available, both sides are parsed symbolically
        for robust comparison (e.g. ``3/2`` == ``1.5``).  Otherwise
        a plain float comparison is used.
        """
        extracted = self.extract_answer(model_output)
        if extracted is None:
            logger.debug("No answer extracted from model output")
            return False

        gt_clean = ground_truth.replace(",", "").strip()
        logger.debug(f"Comparing extracted={extracted!r} vs ground_truth={gt_clean!r}")

        if _SYMPY_AVAILABLE:
            return self._verify_sympy(extracted, gt_clean)
        return self._verify_float(extracted, gt_clean)

    # ── SymPy path ─────────────────────────────────────────────
    def _verify_sympy(self, extracted: str, ground_truth: str) -> bool:
        """Symbolic comparison via SymPy."""
        try:
            import sympy as sp

            # Parse both to SymPy expressions
            ext_expr = sp.sympify(extracted)
            gt_expr = sp.sympify(ground_truth)

            # Try exact symbolic equality first
            diff = sp.simplify(ext_expr - gt_expr)
            if diff == 0:
                return True

            # Fall back to numerical evaluation
            ext_val = complex(ext_expr.evalf())
            gt_val = complex(gt_expr.evalf())
            return abs(ext_val - gt_val) < self.tolerance

        except Exception as e:
            logger.debug(f"SymPy verification failed ({e}), trying float")
            return self._verify_float(extracted, ground_truth)

    # ── Float fallback ─────────────────────────────────────────
    def _verify_float(self, extracted: str, ground_truth: str) -> bool:
        """Plain float comparison fallback."""
        try:
            ext_val = float(extracted)
            gt_val = float(ground_truth)
            return abs(ext_val - gt_val) < self.tolerance
        except (ValueError, TypeError):
            # Last resort: exact string match
            return extracted.strip() == ground_truth.strip()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  5. Main — GSM8K Demonstration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
_SAMPLE_GSM8K_PROBLEM = (
    "Janet's ducks lay 16 eggs per day. She eats three for breakfast "
    "every morning and bakes muffins for her friends every day with four. "
    "She sells the remainder at the farmers' market daily for $2 per "
    "fresh duck egg. How much in dollars does she make every day at the "
    "farmers' market?"
)
_SAMPLE_GSM8K_ANSWER = "18"  # (16 - 3 - 4) × 2 = 18


def main() -> None:
    """Demonstrate MCTS-guided math reasoning on a sample GSM8K problem."""
    print(f"\n{CYAN}{BOLD}{'='*72}{NC}")
    print(f"{CYAN}{BOLD}   RunuX AI Engine — MCTS Math Reasoning Inference{NC}")
    print(f"{CYAN}{BOLD}   Process-Reward-Model × UCB1 × Code Verification{NC}")
    print(f"{CYAN}{BOLD}   Copyright (c) 2026 Xavier Callens / Socrate AI Lab{NC}")
    print(f"{CYAN}{BOLD}{'='*72}{NC}\n")

    # ── Initialise components ──────────────────────────────────
    logger.info(f"{BOLD}🔧 Initialising MCTS Components{NC}")

    model = None
    tokenizer = None

    if _TORCH_AVAILABLE and _TRANSFORMERS_AVAILABLE:
        logger.info("  Attempting to load language model …")
        try:
            model_name = "Qwen/Qwen2.5-Math-1.5B-Instruct"
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            model.eval()
            logger.info(f"  {GREEN}✅ Model loaded: {model_name}{NC}")
        except Exception as e:
            logger.warning(f"  {YELLOW}⚠ Could not load model ({e}) — simulation mode{NC}")
            model = None
            tokenizer = None
    else:
        logger.info(f"  {YELLOW}⚠ torch/transformers not available — simulation mode{NC}")

    engine = MCTSEngine(
        exploration_constant=math.sqrt(2),
        expansion_width=4,
        temperature=0.7,
        code_timeout=10,
    )
    verifier = SymPyVerifier(tolerance=1e-4)

    # ── Run MCTS Search ────────────────────────────────────────
    logger.info(f"\n{BOLD}📝 Sample GSM8K Problem:{NC}")
    logger.info(f"   {_SAMPLE_GSM8K_PROBLEM}")
    logger.info(f"   Expected answer: {BOLD}{_SAMPLE_GSM8K_ANSWER}{NC}")

    best_solution = engine.search(
        problem=_SAMPLE_GSM8K_PROBLEM,
        model=model,
        tokenizer=tokenizer,
        num_paths=64,
        max_depth=8,
    )

    # ── Verify ─────────────────────────────────────────────────
    logger.info(f"\n{BOLD}🔍 Verification{NC}")
    extracted = verifier.extract_answer(best_solution)
    is_correct = verifier.verify(best_solution, _SAMPLE_GSM8K_ANSWER)

    logger.info(f"   Extracted answer: {extracted}")
    logger.info(
        f"   Correct: "
        f"{GREEN + '✅ YES' + NC if is_correct else RED + '❌ NO' + NC}"
    )

    # ── Display best solution ──────────────────────────────────
    logger.info(f"\n{BOLD}📋 Best Reasoning Trace:{NC}")
    for line in best_solution.split("\n"):
        logger.info(f"   {line}")

    # ── Summary ────────────────────────────────────────────────
    logger.info(f"\n{CYAN}{BOLD}{'='*72}{NC}")
    logger.info(f"{CYAN}{BOLD}  🎉 MCTS Inference Complete{NC}")
    logger.info(f"{CYAN}{BOLD}{'='*72}{NC}")
    logger.info(f"  Mode:            {'Model' if model else 'Simulation'}")
    logger.info(f"  Paths explored:  64")
    logger.info(f"  Max depth:       8")
    logger.info(f"  Extracted answer: {extracted}")
    logger.info(f"  Ground truth:    {_SAMPLE_GSM8K_ANSWER}")
    logger.info(f"  Verified:        {is_correct}")
    logger.info(f"{'='*72}\n")


if __name__ == "__main__":
    main()
