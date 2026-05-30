"""
SymBrain v4 — Multi-Scale Model Registry
==========================================

Defines the four inference tiers (EDGE_7B → CLOUD_122B) and their
associated model configurations.  Each tier specifies:

    • Deductive model  — formal reasoning / proof generation
    • Generative model — creative / fluency-oriented generation
    • VRAM footprint   — minimum GPU memory required
    • Quantization     — weight precision (fp16, int8, int4, etc.)
    • Deployment target — where this tier is expected to run
    • Max context length — supported context window

The registry provides two main entry points:

    get_tier_config(tier)
        → ModelConfig for the requested tier

    select_optimal_tier(available_vram_gb)
        → Best ModelConfig that fits in the given VRAM budget

(c) 2026 Socrate AI Lab, Paris, France
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

logger = logging.getLogger("symbrain.v4.registry")


# ─────────────────────────────── Enumerations ─────────────────────────────── #

class ModelTier(str, Enum):
    """Inference scale tiers, smallest to largest."""

    EDGE_7B = "7B"
    CLOUD_32B = "32B"
    CLOUD_70B = "70B"
    CLOUD_122B = "122B"


class DeploymentTarget(str, Enum):
    """Where a given tier is expected to run."""

    EDGE_DEVICE = "edge_device"           # Jetson / Apple Silicon / small GPU
    CLOUD_SINGLE_GPU = "cloud_single_gpu"  # 1× A100-80 GB or equivalent
    CLOUD_MULTI_GPU = "cloud_multi_gpu"    # 2–4× A100-80 GB, tensor parallel
    CLOUD_POD = "cloud_pod"               # 8× H100 or TPU v5e pod slice


class Quantization(str, Enum):
    """Weight quantization schemes."""

    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"
    INT4_GPTQ = "int4_gptq"
    INT4_AWQ = "int4_awq"
    FP8_E4M3 = "fp8_e4m3"


# ─────────────────────────────── Data Classes ─────────────────────────────── #

@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Immutable configuration for a single inference tier.

    Attributes:
        tier: The model-scale tier this config represents.
        deductive_model: HuggingFace / model identifier for the deductive
            (formal reasoning) pathway.
        generative_model: Model identifier for the generative (creative)
            pathway.
        vram_footprint_gb: Minimum VRAM required to load this tier (both
            models simultaneously).
        quantization: Weight precision.
        deployment_target: Recommended deployment environment.
        max_context_length: Maximum context window in tokens.
    """

    tier: ModelTier
    deductive_model: str
    generative_model: str
    vram_footprint_gb: float
    quantization: Quantization
    deployment_target: DeploymentTarget
    max_context_length: int

    def __repr__(self) -> str:
        return (
            f"ModelConfig(tier={self.tier.value}, "
            f"ded={self.deductive_model!r}, "
            f"gen={self.generative_model!r}, "
            f"vram={self.vram_footprint_gb}GB, "
            f"quant={self.quantization.value}, "
            f"ctx={self.max_context_length:,})"
        )

    @property
    def total_param_billions(self) -> float:
        """Extract numeric parameter count from the tier label."""
        return float(self.tier.value.rstrip("B"))


# ═══════════════════════════════════════════════════════════════════════════ #
#  Global Registry                                                           #
# ═══════════════════════════════════════════════════════════════════════════ #

REGISTRY: dict[ModelTier, ModelConfig] = {
    # ── Edge / Mobile tier ────────────────────────────────────────────── #
    ModelTier.EDGE_7B: ModelConfig(
        tier=ModelTier.EDGE_7B,
        deductive_model="symbrain/deductive-qwen2.5-math-7b-awq",
        generative_model="symbrain/generative-mistral-7b-v0.4-awq",
        vram_footprint_gb=8.0,
        quantization=Quantization.INT4_AWQ,
        deployment_target=DeploymentTarget.EDGE_DEVICE,
        max_context_length=32_768,
    ),

    # ── Cloud single-GPU tier ─────────────────────────────────────────── #
    ModelTier.CLOUD_32B: ModelConfig(
        tier=ModelTier.CLOUD_32B,
        deductive_model="symbrain/deductive-qwen2.5-math-32b-int8",
        generative_model="symbrain/generative-mixtral-8x7b-int8",
        vram_footprint_gb=40.0,
        quantization=Quantization.INT8,
        deployment_target=DeploymentTarget.CLOUD_SINGLE_GPU,
        max_context_length=65_536,
    ),

    # ── Cloud multi-GPU tier ──────────────────────────────────────────── #
    ModelTier.CLOUD_70B: ModelConfig(
        tier=ModelTier.CLOUD_70B,
        deductive_model="symbrain/deductive-deepseek-math-70b-bf16",
        generative_model="symbrain/generative-llama3.1-70b-bf16",
        vram_footprint_gb=160.0,
        quantization=Quantization.BF16,
        deployment_target=DeploymentTarget.CLOUD_MULTI_GPU,
        max_context_length=131_072,
    ),

    # ── Cloud pod tier (maximum capability) ───────────────────────────── #
    ModelTier.CLOUD_122B: ModelConfig(
        tier=ModelTier.CLOUD_122B,
        deductive_model="symbrain/deductive-mathstral-122b-fp8",
        generative_model="symbrain/generative-command-r-plus-122b-fp8",
        vram_footprint_gb=320.0,
        quantization=Quantization.FP8_E4M3,
        deployment_target=DeploymentTarget.CLOUD_POD,
        max_context_length=131_072,
    ),
}

# Sorted by VRAM footprint ascending for tier selection
_TIERS_BY_VRAM = sorted(REGISTRY.values(), key=lambda c: c.vram_footprint_gb)


# ═══════════════════════════════════════════════════════════════════════════ #
#  Public API                                                                #
# ═══════════════════════════════════════════════════════════════════════════ #

def get_tier_config(tier: ModelTier | str) -> ModelConfig:
    """Retrieve the :class:`ModelConfig` for a specific tier.

    Args:
        tier: A :class:`ModelTier` enum member or its string value
              (e.g. ``"7B"``, ``"70B"``).

    Returns:
        The corresponding :class:`ModelConfig`.

    Raises:
        KeyError: If the tier is not found in the registry.
    """
    if isinstance(tier, str):
        # Normalize: accept "7B", "EDGE_7B", "edge_7b", etc.
        tier_upper = tier.upper()
        for member in ModelTier:
            if member.value == tier_upper or member.name == tier_upper:
                tier = member
                break
        else:
            raise KeyError(
                f"Unknown tier {tier!r}. "
                f"Available: {[t.value for t in ModelTier]}"
            )

    config = REGISTRY[tier]
    logger.debug("get_tier_config(%s) → %r", tier.value, config)
    return config


def select_optimal_tier(
    available_vram_gb: float,
    *,
    prefer_quality: bool = True,
) -> ModelConfig:
    """Select the best model tier that fits within the VRAM budget.

    Args:
        available_vram_gb: Available GPU VRAM in gigabytes.
        prefer_quality: If ``True`` (default), pick the *largest* tier
            that fits.  If ``False``, pick the *smallest* that fits.

    Returns:
        The optimal :class:`ModelConfig`.

    Raises:
        ValueError: If no tier fits within the given VRAM budget.
    """
    candidates = [
        cfg for cfg in _TIERS_BY_VRAM
        if cfg.vram_footprint_gb <= available_vram_gb
    ]

    if not candidates:
        raise ValueError(
            f"No tier fits within {available_vram_gb:.1f} GB VRAM. "
            f"Minimum required: {_TIERS_BY_VRAM[0].vram_footprint_gb:.1f} GB "
            f"(tier {_TIERS_BY_VRAM[0].tier.value})."
        )

    selected = candidates[-1] if prefer_quality else candidates[0]
    logger.info(
        "select_optimal_tier(%.1f GB) → %s (needs %.1f GB)",
        available_vram_gb, selected.tier.value, selected.vram_footprint_gb,
    )
    return selected


def list_tiers() -> list[dict]:
    """Return a summary of all available tiers for API / UI display."""
    return [
        {
            "tier": cfg.tier.value,
            "deductive_model": cfg.deductive_model,
            "generative_model": cfg.generative_model,
            "vram_gb": cfg.vram_footprint_gb,
            "quantization": cfg.quantization.value,
            "deployment": cfg.deployment_target.value,
            "max_context": cfg.max_context_length,
        }
        for cfg in _TIERS_BY_VRAM
    ]


# ═══════════════════════════════════════════════════════════════════════════ #
#  CLI / diagnostics                                                         #
# ═══════════════════════════════════════════════════════════════════════════ #

def _print_registry() -> None:
    """Print the full registry in a human-readable table."""
    import json

    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")

    print("\n" + "=" * 90)
    print("  SymBrain v4 — Model Registry")
    print("=" * 90)

    header = f"{'Tier':>6}  {'VRAM':>8}  {'Quant':>10}  {'Context':>9}  {'Deductive Model'}"
    print(f"\n  {header}")
    print(f"  {'-' * len(header)}")

    for cfg in _TIERS_BY_VRAM:
        print(
            f"  {cfg.tier.value:>6}  "
            f"{cfg.vram_footprint_gb:>6.0f}GB  "
            f"{cfg.quantization.value:>10}  "
            f"{cfg.max_context_length:>7,}  "
            f"{cfg.deductive_model}"
        )

    print(f"\n  Total tiers: {len(REGISTRY)}")

    # VRAM selection demo
    print("\n  ── Tier Selection Examples ──")
    for vram in [10.0, 48.0, 200.0, 500.0]:
        try:
            cfg = select_optimal_tier(vram)
            print(f"    {vram:6.0f} GB → {cfg.tier.value} ({cfg.deployment_target.value})")
        except ValueError as e:
            print(f"    {vram:6.0f} GB → {e}")

    print("=" * 90 + "\n")


if __name__ == "__main__":
    _print_registry()
