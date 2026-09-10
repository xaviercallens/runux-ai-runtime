# ==============================================================================
# RunuX AI Runtime — Core Python Acceleration & Optimization Engine
#
# Copyright (c) 2026 Xavier Callens / Socrate AI Lab. All Rights Reserved.
# Licensed under LicenseRef-RunuX-Commercial.
#
# Patent Pending: INPI Demande Provisoire de Brevet RunuX (2026).
# ==============================================================================

from .deterministic_attn import Int64DeterministicAttention
from .polarquant import PolarQuantKVCache, PolarQuantConfig
from .signsgd import SignSGDOptimizer
from .carbon_scheduler import CarbonAwareSpeculativeScheduler, GridCarbonProfile
from .systolic_advisor import SystolicTilingAdvisor
from .gqa_kernel import GroupedQueryAttention
from .paged_cache import PagedKVCache

__all__ = [
    "Int64DeterministicAttention",
    "PolarQuantKVCache",
    "PolarQuantConfig",
    "SignSGDOptimizer",
    "CarbonAwareSpeculativeScheduler",
    "GridCarbonProfile",
    "SystolicTilingAdvisor",
    "GroupedQueryAttention",
    "PagedKVCache",
]
