# ==============================================================================
# RunuX AI Runtime — RTE Carbon-Aware Speculative Decoding Scheduler
#
# Copyright (c) 2026 Xavier Callens / Socrate AI Lab. All Rights Reserved.
# Licensed under LicenseRef-RunuX-Commercial.
#
# Patent Pending: INPI Dossier Demande Provisoire de Brevet RunuX (2026).
# ==============================================================================

import math
from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class GridCarbonProfile:
    region_name: str
    carbon_intensity_gco2_kwh: float  # gCO2 per kWh
    base_power_watts: float           # Server base power
    peak_power_watts: float           # Server peak load power

class CarbonAwareSpeculativeScheduler:
    """
    Dynamically adjusts speculative decoding draft length K in response to
    real-time electrical grid carbon intensity signals (e.g. RTE Eco2Mix API).
    """
    def __init__(
        self,
        grid_profile: Optional[GridCarbonProfile] = None,
        k_min: int = 2,
        k_max: int = 7,
        c_ref: float = 56.0,  # France nuclear mix reference (56 gCO2/kWh)
        gamma: float = 0.65   # Carbon sensitivity damping factor
    ):
        self.profile = grid_profile or GridCarbonProfile("France (RTE)", 56.0, 180.0, 320.0)
        self.k_min = k_min
        self.k_max = k_max
        self.c_ref = c_ref
        self.gamma = gamma

    def compute_optimal_draft_k(self, current_carbon_gco2_kwh: float, acceptance_rate: float = 0.70) -> int:
        """
        Calculates optimal draft length K*(t) using closed-loop carbon regulation formula:
          K*(t) = clamp(round(K_base * (C_ref / C_grid)^gamma * alpha_accept), K_min, K_max)
        """
        k_base = 5.0
        ratio = max(0.1, self.c_ref / max(1.0, current_carbon_gco2_kwh))
        scaled_k = k_base * math.pow(ratio, self.gamma) * (acceptance_rate / 0.70)
        k_opt = int(round(scaled_k))
        return max(self.k_min, min(self.k_max, k_opt))

    def evaluate_inference_emissions(
        self,
        num_tokens: int,
        effective_tps: float,
        power_watts: float,
        carbon_gco2_kwh: float
    ) -> Dict[str, float]:
        """
        Computes the physical energy and carbon emissions for a generation session.
        """
        elapsed_seconds = num_tokens / max(1e-5, effective_tps)
        joules_total = power_watts * elapsed_seconds
        joules_per_token = joules_total / max(1, num_tokens)

        kwh_total = joules_total / 3_600_000.0
        gco2_total = kwh_total * carbon_gco2_kwh
        gco2_per_1k_tokens = (gco2_total / max(1, num_tokens)) * 1000.0

        return {
            "elapsed_seconds": elapsed_seconds,
            "joules_per_token": joules_per_token,
            "total_joules": joules_total,
            "total_gco2": gco2_total,
            "gco2_per_1k_tokens": gco2_per_1k_tokens,
        }
