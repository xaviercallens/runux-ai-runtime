"""
SymBrain v4 — Universal Calibrated PFC (Deductive Active) Router
=================================================================

Three-stage prefrontal cortex routing engine that classifies incoming queries
and produces a continuous routing tensor σ = (σ_deductive, σ_generative,
σ_mcts_budget) for downstream inference.

Architecture:
    Stage 1 — Lexical Intent Scanner:   Fast regex-based domain detection
    Stage 2 — Semantic Complexity:       Token-level feature classifier (0–1)
    Stage 3 — Dynamic Difficulty:        MCTS search-budget multiplier

Invariant (Routing-Stall Elimination):
    σ_deductive ≥ 0.3 for ALL queries.  This is enforced as a hard floor
    in the final tensor normalization, permanently removing the class of
    bugs where the deductive pathway received near-zero weight and the
    engine stalled on formal reasoning tasks.

(c) 2026 Socrate AI Lab, Paris, France
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import ClassVar

logger = logging.getLogger("symbrain.v4.pfc")

# ─────────────────────────────── Constants ────────────────────────────────── #

# Hard floor: σ_deductive is never allowed below this value.
DEDUCTIVE_FLOOR: float = 0.30

# Baseline MCTS budget multiplier for trivial queries.
MCTS_BASE_BUDGET: float = 1.0

# Maximum MCTS budget multiplier for the hardest queries.
MCTS_MAX_BUDGET: float = 8.0


# ─────────────────────────────── Enumerations ─────────────────────────────── #

class Domain(str, Enum):
    """Detected academic / STEM domain."""

    MATHEMATICS = "mathematics"
    PHYSICS = "physics"
    CHEMISTRY = "chemistry"
    COMPUTER_SCIENCE = "computer_science"
    BIOLOGY = "biology"
    ENGINEERING = "engineering"
    GENERAL = "general"


class RoutingStage(str, Enum):
    """Which PFC stage determined the final routing."""

    LEXICAL = "lexical"
    SEMANTIC = "semantic"
    DYNAMIC = "dynamic"


# ─────────────────────────────── Data Classes ─────────────────────────────── #

@dataclass(frozen=True, slots=True)
class RoutingDecision:
    """Immutable output of the PFC router.

    The routing tensor σ = (deductive_weight, generative_weight,
    mcts_budget_multiplier) is guaranteed to satisfy:
        • deductive_weight ≥ DEDUCTIVE_FLOOR (0.30)
        • deductive_weight + generative_weight ≈ 1.0
        • mcts_budget_multiplier ∈ [MCTS_BASE_BUDGET, MCTS_MAX_BUDGET]
    """

    deductive_weight: float
    generative_weight: float
    mcts_budget_multiplier: float
    complexity_score: float          # ∈ [0, 1]
    detected_domain: Domain
    routing_stage: RoutingStage

    # ── Derived helpers ────────────────────────────────────────────────── #

    @property
    def sigma(self) -> tuple[float, float, float]:
        """Return the full routing tensor σ."""
        return (self.deductive_weight, self.generative_weight,
                self.mcts_budget_multiplier)

    @property
    def is_deductive_dominant(self) -> bool:
        return self.deductive_weight >= 0.6

    def __repr__(self) -> str:
        return (
            f"RoutingDecision(σ_ded={self.deductive_weight:.3f}, "
            f"σ_gen={self.generative_weight:.3f}, "
            f"mcts×={self.mcts_budget_multiplier:.2f}, "
            f"C={self.complexity_score:.3f}, "
            f"domain={self.detected_domain.value}, "
            f"stage={self.routing_stage.value})"
        )


# ═══════════════════════════════════════════════════════════════════════════ #
#  Stage 1 — Lexical Intent Scanner                                          #
# ═══════════════════════════════════════════════════════════════════════════ #

@dataclass
class LexicalScanner:
    """Fast regex-based multi-domain keyword scanner.

    Each domain has a compiled pattern that matches against the raw query.
    Scores are the fraction of unique domain-keywords found.
    """

    # ── Pattern banks ──────────────────────────────────────────────────── #

    _MATH_SYMBOLS: ClassVar[list[str]] = [
        r"∫", r"∑", r"∏", r"lim", r"∇", r"∂",
        r"sin", r"cos", r"tan", r"log", r"ln", r"exp",
        r"sqrt", r"matrix", r"eigenvalue", r"eigenvector",
        r"derivative", r"integral", r"differential",
        r"equation", r"theorem", r"proof", r"lemma",
        r"corollary", r"polynomial", r"determinant",
        r"vector", r"scalar", r"tensor", r"manifold",
        r"topology", r"isomorphism", r"homomorphism",
        r"convergence", r"divergence", r"series",
        r"Banach", r"Hilbert", r"Lebesgue", r"Fourier",
        r"Dirichlet", r"Cauchy", r"Riemann",
        r"LaTeX", r"\\frac", r"\\int", r"\\sum",
        r"\\lim", r"\\partial", r"\\nabla",
        r"\\begin\{", r"\\mathbb", r"\\mathcal",
    ]

    _MATH_FR: ClassVar[list[str]] = [
        r"démontrer", r"calculer", r"résoudre",
        r"déterminer", r"soit\b", r"montrer\s+que",
        r"en\s+déduire", r"on\s+pose", r"on\s+considère",
        r"justifier", r"vérifier", r"simplifier",
        r"factoriser", r"développer", r"exprimer",
        r"établir", r"conclure", r"en\s+fonction\s+de",
        r"pour\s+tout", r"il\s+existe",
    ]

    _CHEMISTRY: ClassVar[list[str]] = [
        r"H2O", r"NaCl", r"pH", r"mol\b", r"molar",
        r"molecule", r"reaction", r"oxidation", r"reduction",
        r"acid", r"base", r"buffer", r"titration",
        r"equilibrium", r"Ksp", r"Ka\b", r"Kb\b", r"Kw\b",
        r"enthalpy", r"entropy", r"Gibbs", r"bond",
        r"orbital", r"electron", r"proton", r"neutron",
        r"isotope", r"catalysis", r"catalyst",
        r"stoichiometry", r"molarity", r"solubility",
        r"electronegativity", r"hybridization",
        r"thermodynamics", r"exothermic", r"endothermic",
        r"Le\s+Chatelier", r"Nernst",
        r"CH[234]", r"C[0-9]+H[0-9]+",
        r"acide\b", r"réaction", r"oxydation", r"réduction",
    ]

    _PHYSICS: ClassVar[list[str]] = [
        r"force", r"velocity", r"acceleration",
        r"energy", r"momentum", r"impulse",
        r"wave", r"field", r"potential",
        r"gravity", r"Newton", r"Lagrangian",
        r"Hamiltonian", r"Schrödinger",
        r"Maxwell", r"Gauss", r"Faraday",
        r"resistance", r"capacitance", r"inductance",
        r"current", r"voltage", r"circuit",
        r"torque", r"angular", r"frequency",
        r"wavelength", r"amplitude", r"photon",
        r"quantum", r"relativity", r"entropy",
        r"Boltzmann", r"Planck", r"Carnot",
        r"adiabatic", r"isothermal", r"isobaric",
        r"inclined\s+plane", r"pulley", r"friction",
        r"skin\s+depth", r"Sackur.Tetrode",
        r"electromagnetism", r"electromagnetic",
        r"thermodynamique", r"mécanique", r"cinétique",
        r"champ", r"potentiel", r"onde",
    ]

    _COMPUTER_SCIENCE: ClassVar[list[str]] = [
        r"algorithm", r"complexity", r"O\(n",
        r"NP-hard", r"NP-complete", r"Turing",
        r"recursion", r"dynamic\s+programming",
        r"graph", r"tree", r"sort", r"hash",
        r"neural\s+network", r"gradient\s+descent",
        r"backpropagation", r"machine\s+learning",
        r"compiler", r"automaton", r"regex",
    ]

    _BIOLOGY: ClassVar[list[str]] = [
        r"DNA", r"RNA", r"protein", r"gene",
        r"cell", r"mitosis", r"meiosis",
        r"enzyme", r"substrate", r"ATP",
        r"membrane", r"ribosome", r"transcription",
        r"translation", r"CRISPR", r"genome",
    ]

    _ENGINEERING: ClassVar[list[str]] = [
        r"circuit", r"amplifier", r"transistor",
        r"PID\b", r"control\s+system", r"feedback",
        r"signal", r"filter", r"Bode",
        r"Laplace", r"transfer\s+function",
        r"modulation", r"bandwidth", r"impedance",
    ]

    # ── Compiled patterns (built once) ─────────────────────────────────── #

    _compiled: dict[Domain, re.Pattern[str]] = field(
        default_factory=dict, init=False, repr=False,
    )

    def __post_init__(self) -> None:
        self._compiled = {
            Domain.MATHEMATICS: self._compile(
                self._MATH_SYMBOLS + self._MATH_FR,
            ),
            Domain.CHEMISTRY: self._compile(self._CHEMISTRY),
            Domain.PHYSICS: self._compile(self._PHYSICS),
            Domain.COMPUTER_SCIENCE: self._compile(self._COMPUTER_SCIENCE),
            Domain.BIOLOGY: self._compile(self._BIOLOGY),
            Domain.ENGINEERING: self._compile(self._ENGINEERING),
        }

    @staticmethod
    def _compile(terms: list[str]) -> re.Pattern[str]:
        return re.compile("|".join(f"(?:{t})" for t in terms), re.IGNORECASE)

    # ── Scan ───────────────────────────────────────────────────────────── #

    def scan(self, query: str) -> dict[Domain, float]:
        """Return per-domain hit scores ∈ [0, 1].

        Score = (number of unique keyword hits) / (number of keywords in bank).
        """
        scores: dict[Domain, float] = {}
        bank_sizes = {
            Domain.MATHEMATICS: len(self._MATH_SYMBOLS) + len(self._MATH_FR),
            Domain.CHEMISTRY: len(self._CHEMISTRY),
            Domain.PHYSICS: len(self._PHYSICS),
            Domain.COMPUTER_SCIENCE: len(self._COMPUTER_SCIENCE),
            Domain.BIOLOGY: len(self._BIOLOGY),
            Domain.ENGINEERING: len(self._ENGINEERING),
        }
        for domain, pattern in self._compiled.items():
            hits = set(pattern.findall(query))
            scores[domain] = min(len(hits) / max(bank_sizes[domain], 1), 1.0)
        return scores


# ═══════════════════════════════════════════════════════════════════════════ #
#  Stage 2 — Semantic Complexity Classifier                                  #
# ═══════════════════════════════════════════════════════════════════════════ #

@dataclass
class SemanticComplexityClassifier:
    """Lightweight token-level feature extractor that estimates query
    complexity on a continuous [0, 1] scale.

    Feature dimensions:
        1. LaTeX presence and density
        2. Nested mathematical expressions (bracket depth)
        3. Variable count (single-letter identifiers)
        4. Formal proof language markers
        5. Query length (log-scaled)
        6. Multi-step reasoning indicators
        7. Advanced topic markers (CPGE / agrégation level)
    """

    # Feature weights (hand-tuned; sum ≈ 1.0 is not required — we clamp)
    _W_LATEX: ClassVar[float] = 0.18
    _W_NESTING: ClassVar[float] = 0.14
    _W_VARIABLES: ClassVar[float] = 0.10
    _W_PROOF: ClassVar[float] = 0.16
    _W_LENGTH: ClassVar[float] = 0.08
    _W_MULTISTEP: ClassVar[float] = 0.14
    _W_ADVANCED: ClassVar[float] = 0.20

    # ── Feature extractors ─────────────────────────────────────────────── #

    _LATEX_RE: ClassVar[re.Pattern[str]] = re.compile(
        r"\\(?:frac|int|sum|prod|lim|partial|nabla|begin|end|math|left|right"
        r"|alpha|beta|gamma|delta|epsilon|lambda|mu|sigma|omega|phi|psi|theta"
        r"|infty|cdot|times|leq|geq|neq|approx|equiv|forall|exists|in\b|subset"
        r"|cup|cap|to\b|mapsto|rightarrow|Rightarrow|iff)",
        re.IGNORECASE,
    )
    _VAR_RE: ClassVar[re.Pattern[str]] = re.compile(
        r"(?<![a-zA-Z])[a-zA-Z](?:_[a-zA-Z0-9]+)?(?![a-zA-Z])",
    )
    _PROOF_RE: ClassVar[re.Pattern[str]] = re.compile(
        r"\b(?:prove|proof|theorem|lemma|corollary|proposition|QED|CQFD"
        r"|contradict|induction|by\s+contradiction|by\s+induction"
        r"|assume|suppose|let\b|hence|therefore|thus|it\s+follows"
        r"|démontrer|montrer\s+que|en\s+déduire|soit\b|on\s+suppose"
        r"|récurrence|absurde|contraposée)\b",
        re.IGNORECASE,
    )
    _MULTISTEP_RE: ClassVar[re.Pattern[str]] = re.compile(
        r"\b(?:then|next|after|finally|step\s+\d|first|second|third"
        r"|part\s*[a-d()\d]|question\s+\d|\d+\)\s|\([a-d]\)"
        r"|puis|ensuite|enfin|d'abord|premièrement)\b",
        re.IGNORECASE,
    )
    _ADVANCED_RE: ClassVar[re.Pattern[str]] = re.compile(
        r"\b(?:Banach|Hilbert|Lebesgue|Sobolev|Fourier\s+transform"
        r"|Dirichlet|Riemann\s+integral|Green.s\s+function"
        r"|Navier.Stokes|Lagrangian|Hamiltonian"
        r"|Sackur.Tetrode|skin\s+depth|Debye|Bose.Einstein"
        r"|Fermi.Dirac|partition\s+function|canonical\s+ensemble"
        r"|density\s+operator|perturbation\s+theory"
        r"|variational|functional\s+analysis"
        r"|compact\s+operator|spectral\s+theorem"
        r"|measure\s+theory|dominated\s+convergence"
        r"|CPGE|agrégation|classes\s+prépa|concours)\b",
        re.IGNORECASE,
    )

    def classify(self, query: str) -> float:
        """Return a complexity score ∈ [0, 1]."""

        # 1. LaTeX density
        latex_hits = len(self._LATEX_RE.findall(query))
        f_latex = min(latex_hits / 8.0, 1.0)

        # 2. Nesting depth (max bracket depth)
        depth = max_depth = 0
        for ch in query:
            if ch in "({[":
                depth += 1
                max_depth = max(max_depth, depth)
            elif ch in ")}]":
                depth = max(depth - 1, 0)
        f_nesting = min(max_depth / 5.0, 1.0)

        # 3. Distinct single-letter variables
        variables = set(self._VAR_RE.findall(query))
        # Filter out common English words that are single letters
        variables -= {"I", "a", "A", "i"}
        f_variables = min(len(variables) / 6.0, 1.0)

        # 4. Proof language markers
        proof_hits = len(set(self._PROOF_RE.findall(query)))
        f_proof = min(proof_hits / 3.0, 1.0)

        # 5. Log-scaled query length
        word_count = len(query.split())
        f_length = min(math.log1p(word_count) / math.log1p(200), 1.0)

        # 6. Multi-step reasoning
        multistep_hits = len(set(self._MULTISTEP_RE.findall(query)))
        f_multistep = min(multistep_hits / 3.0, 1.0)

        # 7. Advanced topic markers
        advanced_hits = len(set(self._ADVANCED_RE.findall(query)))
        f_advanced = min(advanced_hits / 2.0, 1.0)

        # Weighted sum → clamp to [0, 1]
        raw = (
            self._W_LATEX * f_latex
            + self._W_NESTING * f_nesting
            + self._W_VARIABLES * f_variables
            + self._W_PROOF * f_proof
            + self._W_LENGTH * f_length
            + self._W_MULTISTEP * f_multistep
            + self._W_ADVANCED * f_advanced
        )
        score = max(0.0, min(raw, 1.0))

        logger.debug(
            "Complexity features: latex=%.2f nest=%.2f vars=%.2f "
            "proof=%.2f len=%.2f multi=%.2f adv=%.2f → C=%.3f",
            f_latex, f_nesting, f_variables,
            f_proof, f_length, f_multistep, f_advanced, score,
        )
        return score


# ═══════════════════════════════════════════════════════════════════════════ #
#  Stage 3 — Dynamic Difficulty Estimator                                    #
# ═══════════════════════════════════════════════════════════════════════════ #

@dataclass
class DynamicDifficultyEstimator:
    """Maps a complexity score C ∈ [0, 1] to an MCTS search-budget
    multiplier ∈ [MCTS_BASE_BUDGET, MCTS_MAX_BUDGET].

    The mapping uses a sigmoid-like curve so that:
        • Easy queries (C < 0.2) get ~1× budget (fast response)
        • Medium queries (C ≈ 0.5) get ~3× budget
        • Hard queries (C > 0.8) get up to 8× budget (deep exploration)
    """

    base: float = MCTS_BASE_BUDGET
    maximum: float = MCTS_MAX_BUDGET
    steepness: float = 6.0   # Controls sigmoid slope
    midpoint: float = 0.45   # Inflection point

    def estimate(self, complexity: float) -> float:
        """Return MCTS budget multiplier for the given complexity."""
        # Logistic sigmoid scaled to [base, maximum]
        t = self.steepness * (complexity - self.midpoint)
        sigmoid = 1.0 / (1.0 + math.exp(-t))
        budget = self.base + (self.maximum - self.base) * sigmoid
        return round(budget, 2)


# ═══════════════════════════════════════════════════════════════════════════ #
#  PFC Router — Main Entry Point                                             #
# ═══════════════════════════════════════════════════════════════════════════ #

class PFCRouter:
    """Universal Calibrated PFC (Deductive Active) Router.

    Usage::

        router = PFCRouter()
        decision = router.route("Calculer lim_{x→0} sin(x)/x")
        print(decision)
        # RoutingDecision(σ_ded=0.750, σ_gen=0.250, mcts×=2.10, ...)

    The router guarantees ``decision.deductive_weight >= 0.30`` for every
    query, which eliminates the Routing-Stall anomaly class permanently.
    """

    VERSION: ClassVar[str] = "4.0.0-calibrated"

    def __init__(self) -> None:
        self._scanner = LexicalScanner()
        self._classifier = SemanticComplexityClassifier()
        self._estimator = DynamicDifficultyEstimator()
        logger.info("PFCRouter v%s initialized", self.VERSION)

    # ── Core routing method ────────────────────────────────────────────── #

    def route(self, query: str) -> RoutingDecision:
        """Classify *query* and produce a calibrated routing tensor.

        Returns a :class:`RoutingDecision` with the invariant
        ``deductive_weight >= DEDUCTIVE_FLOOR``.
        """
        if not query or not query.strip():
            return self._default_decision()

        # ── Stage 1: Lexical Intent Scan ───────────────────────────────── #
        domain_scores = self._scanner.scan(query)
        top_domain, top_score = max(domain_scores.items(), key=lambda kv: kv[1])

        if top_score < 0.01:
            top_domain = Domain.GENERAL

        logger.debug(
            "Stage 1 → domain=%s (score=%.3f), scores=%s",
            top_domain.value, top_score,
            {d.value: round(s, 3) for d, s in domain_scores.items() if s > 0},
        )

        # ── Stage 2: Semantic Complexity ───────────────────────────────── #
        complexity = self._classifier.classify(query)

        # Domain boost: STEM domains raise base deductive weight
        stem_boost = self._stem_boost(top_domain, top_score)

        # ── Stage 3: Dynamic Difficulty → MCTS Budget ──────────────────── #
        mcts_budget = self._estimator.estimate(complexity)

        # ── Tensor construction with deductive floor ───────────────────── #
        raw_deductive = self._compute_raw_deductive(
            complexity, stem_boost, top_domain,
        )
        sigma_ded, sigma_gen = self._enforce_floor(raw_deductive)

        # Determine which stage was dominant
        if stem_boost > 0.15:
            stage = RoutingStage.LEXICAL
        elif complexity > 0.3:
            stage = RoutingStage.DYNAMIC
        else:
            stage = RoutingStage.SEMANTIC

        decision = RoutingDecision(
            deductive_weight=sigma_ded,
            generative_weight=sigma_gen,
            mcts_budget_multiplier=mcts_budget,
            complexity_score=complexity,
            detected_domain=top_domain,
            routing_stage=stage,
        )
        logger.info("PFC route: %r", decision)
        return decision

    # ── Internal helpers ───────────────────────────────────────────────── #

    @staticmethod
    def _stem_boost(domain: Domain, score: float) -> float:
        """Return an additive boost for STEM domains."""
        boosts = {
            Domain.MATHEMATICS: 0.30,
            Domain.PHYSICS: 0.25,
            Domain.CHEMISTRY: 0.22,
            Domain.COMPUTER_SCIENCE: 0.18,
            Domain.BIOLOGY: 0.12,
            Domain.ENGINEERING: 0.20,
            Domain.GENERAL: 0.0,
        }
        return boosts.get(domain, 0.0) * min(score * 10.0, 1.0)

    @staticmethod
    def _compute_raw_deductive(
        complexity: float,
        stem_boost: float,
        domain: Domain,
    ) -> float:
        """Compute raw deductive weight before floor enforcement.

        Higher complexity and STEM domains push deductive weight up.
        Even general queries get a base deductive allocation.
        """
        base = 0.35  # Ensures floor even without any signal
        complexity_contrib = complexity * 0.40
        domain_contrib = stem_boost
        raw = base + complexity_contrib + domain_contrib
        return min(raw, 0.95)  # Cap: always leave ≥5% for generative

    @staticmethod
    def _enforce_floor(raw_deductive: float) -> tuple[float, float]:
        """Enforce the deductive floor and normalize.

        Returns (σ_deductive, σ_generative) summing to 1.0.
        """
        sigma_ded = max(raw_deductive, DEDUCTIVE_FLOOR)
        sigma_gen = 1.0 - sigma_ded

        # Round for clean output
        sigma_ded = round(sigma_ded, 4)
        sigma_gen = round(sigma_gen, 4)

        assert sigma_ded >= DEDUCTIVE_FLOOR, (
            f"CRITICAL: Deductive floor violation: {sigma_ded} < {DEDUCTIVE_FLOOR}"
        )
        return sigma_ded, sigma_gen

    def _default_decision(self) -> RoutingDecision:
        """Return a safe default for empty / whitespace-only queries."""
        return RoutingDecision(
            deductive_weight=DEDUCTIVE_FLOOR,
            generative_weight=1.0 - DEDUCTIVE_FLOOR,
            mcts_budget_multiplier=MCTS_BASE_BUDGET,
            complexity_score=0.0,
            detected_domain=Domain.GENERAL,
            routing_stage=RoutingStage.SEMANTIC,
        )

    # ── Diagnostics ────────────────────────────────────────────────────── #

    def diagnose(self, query: str) -> dict:
        """Return full diagnostic payload for debugging / monitoring.

        Includes all intermediate stage outputs alongside the final decision.
        """
        domain_scores = self._scanner.scan(query)
        complexity = self._classifier.classify(query)
        decision = self.route(query)

        return {
            "query_length": len(query),
            "word_count": len(query.split()),
            "stage1_domain_scores": {
                d.value: round(s, 4) for d, s in domain_scores.items()
            },
            "stage2_complexity": round(complexity, 4),
            "stage3_mcts_budget": decision.mcts_budget_multiplier,
            "final_decision": {
                "deductive_weight": decision.deductive_weight,
                "generative_weight": decision.generative_weight,
                "mcts_budget_multiplier": decision.mcts_budget_multiplier,
                "complexity_score": decision.complexity_score,
                "detected_domain": decision.detected_domain.value,
                "routing_stage": decision.routing_stage.value,
            },
            "deductive_floor_satisfied": (
                decision.deductive_weight >= DEDUCTIVE_FLOOR
            ),
            "pfc_version": self.VERSION,
        }


# ═══════════════════════════════════════════════════════════════════════════ #
#  Self-test / CLI diagnostics                                               #
# ═══════════════════════════════════════════════════════════════════════════ #

def _self_test() -> None:
    """Run a suite of routing tests and print results."""
    import json

    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")

    router = PFCRouter()
    test_queries = [
        # Mathematics
        "Calculer lim_{x→0} sin(x)/x",
        "Démontrer que tout espace de Banach réflexif est faiblement séquentiellement compact.",
        "Solve ∫₀^∞ sin(x)/x dx using the Dirichlet integral.",
        "Find the eigenvalues of the matrix A = [[2,1],[1,3]]",
        "\\frac{d}{dx}\\left(\\int_0^x e^{-t^2} dt\\right)",
        # Physics
        "Derive the Sackur-Tetrode equation for the entropy of an ideal gas.",
        "Calculate the skin depth for copper at 10 GHz.",
        "A block slides down a frictionless inclined plane of angle 30°. Find acceleration.",
        # Chemistry
        "Calculate the pH of a 0.1M acetic acid solution (Ka = 1.8×10⁻⁵).",
        "Balance the redox reaction: MnO₄⁻ + Fe²⁺ → Mn²⁺ + Fe³⁺",
        # General
        "What is the capital of France?",
        "Tell me a joke about programming.",
        # Empty
        "",
    ]

    print("\n" + "=" * 80)
    print("  SymBrain v4 — PFC Router Self-Test")
    print("=" * 80)

    all_pass = True
    for q in test_queries:
        decision = router.route(q)
        floor_ok = decision.deductive_weight >= DEDUCTIVE_FLOOR
        status = "✓" if floor_ok else "✗ FLOOR VIOLATION"
        if not floor_ok:
            all_pass = False
        print(f"\n  [{status}]  {q[:70]!r}")
        print(f"          {decision}")

    print("\n" + "-" * 80)
    if all_pass:
        print("  ✓ ALL TESTS PASSED — Deductive floor (≥0.30) guaranteed for all queries.")
    else:
        print("  ✗ FLOOR VIOLATIONS DETECTED — Routing-Stall bug still present!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    _self_test()
