# ==============================================================================
# RunuX AI Runtime — Core Python Acceleration & Optimization Engine
#
# Copyright (c) 2026 Xavier Callens / Socrate AI Lab. All Rights Reserved.
# Licensed under LicenseRef-RunuX-Commercial.
# ==============================================================================

from .deterministic_attn import Int64DeterministicAttention, build_int64_softmax_lut
from .polarquant import PolarQuantKVCache, PolarQuantConfig
from .signsgd import SignSGDOptimizer
from .carbon_scheduler import CarbonAwareSpeculativeScheduler, GridCarbonProfile
from .systolic_advisor import SystolicTilingAdvisor, TpuProfile

__version__ = "1.0.0-phase1"
__all__ = [
    "Int64DeterministicAttention",
    "build_int64_softmax_lut",
    "PolarQuantKVCache",
    "PolarQuantConfig",
    "SignSGDOptimizer",
    "CarbonAwareSpeculativeScheduler",
    "GridCarbonProfile",
    "SystolicTilingAdvisor",
    "TpuProfile",
]
