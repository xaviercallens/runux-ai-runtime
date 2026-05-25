# Copyright (c) 2026 Xavier Callens / Socrate AI Lab. All Rights Reserved.
# SPDX-License-Identifier: LicenseRef-RunuX-Commercial
#
# WARS-Quantum-LTN: Logic Tensor Network Fuzzy Constraints Module
# ===============================================================

import numpy as np

class FuzzyLogicGatekeeper:
    """Implements first-order fuzzy Logic Tensor Network constraints for quantum state boundaries."""
    def __init__(self, beta: float = 10.0, energy_tolerance: float = 1e-12):
        self.beta = beta
        self.energy_tolerance = energy_tolerance

    def preserves_unitary(self, norm_sq: float) -> float:
        """
        Fuzzy predicate: preserves_unitary(v)
        Evaluates the truth value in [0.0, 1.0] that the state vector norm is preserved.
        $$I(\\text{preserves\\_unitary}(v)) = e^{-\\beta |\\text{norm}\\_sq - 1.0|}$$
        """
        truth = np.exp(-self.beta * abs(norm_sq - 1.0))
        return float(np.clip(truth, 0.0, 1.0))

    def energy_drift_bounded(self, initial_energy: float, current_energy: float) -> float:
        """
        Fuzzy predicate: energy_drift_bounded(v)
        Evaluates truth that energy drift is zero.
        $$I(\\text{energy\\_drift\\_bounded}(v)) = e^{-\\beta |E_{\\text{curr}} - E_{\\text{init}}|}$$
        """
        drift = abs(current_energy - initial_energy)
        truth = np.exp(-self.beta * drift)
        return float(np.clip(truth, 0.0, 1.0))

    def gauge_invariant(self, discrepancy: float) -> float:
        """
        Fuzzy predicate: gauge_invariant(v)
        Evaluates the truth value in [0.0, 1.0] that local gauge symmetry is perfectly preserved.
        $$I(\\text{gauge\\_invariant}(v)) = e^{-\\beta \\cdot \\text{discrepancy}}$$
        """
        truth = np.exp(-self.beta * discrepancy)
        return float(np.clip(truth, 0.0, 1.0))

    def evaluate_fuzzy_satisfaction(self, p_unitary: float, p_energy: float, p_gauge: float = 1.0) -> float:
        """
        Evaluates the global fuzzy logic satisfaction using Product t-norm:
        $$I(\\phi \\land \\psi \\land \\chi) = I(\\phi) \\times I(\\psi) \\times I(\\chi)$$
        """
        return float(p_unitary * p_energy * p_gauge)

