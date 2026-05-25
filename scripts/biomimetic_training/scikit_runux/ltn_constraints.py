# Copyright (c) 2026 Xavier Callens / Socrate AI Lab. All Rights Reserved.
# SPDX-License-Identifier: LicenseRef-RunuX-Commercial
#
# Biomimetic Co-Inference Training: Logic Tensor Network Fuzzy Constraints
# ======================================================================

import numpy as np
from typing import List

class BiomimeticFuzzyLogicGatekeeper:
    """
    Implements Logic Tensor Network (LTN) fuzzy logic predicates
    to audit biologically-inspired training safety boundaries.
    """
    def __init__(self, beta: float = 5.0):
        self.beta = beta

    def weights_bounded(self, W_list: List[np.ndarray], limit: float = 100.0) -> float:
        """
        Fuzzy predicate: weights_bounded(W)
        Evaluates the truth value in [0.0, 1.0] that the Frobenius norm of all weight matrices is bounded.
        $$I(\\text{weights\\_bounded}(W)) = \\prod_i e^{-\\beta \\max(0, \\|W_i\\|_F - \\text{limit})}$$
        """
        satisfaction = 1.0
        for W in W_list:
            norm = np.linalg.norm(W)
            excess = max(0.0, norm - limit)
            truth = np.exp(-self.beta * excess)
            satisfaction *= truth
        return float(np.clip(satisfaction, 0.0, 1.0))

    def error_converging(self, loss_history: List[float], convergence_threshold: float = 0.5) -> float:
        """
        Fuzzy predicate: error_converging(e)
        Evaluates the truth value in [0.0, 1.0] that the training loss converges towards zero.
        $$I(\\text{error\\_converging}(e)) = e^{-\\beta \\max(0, L_{\\text{final}} - L_{\\text{initial}} \\cdot \\text{threshold})}$$
        """
        if not loss_history or len(loss_history) < 2:
            return 0.0
            
        initial_loss = loss_history[0]
        final_loss = loss_history[-1]
        
        # Loss must decrease below a fraction of initial loss
        target_max_loss = initial_loss * convergence_threshold
        excess_loss = max(0.0, final_loss - target_max_loss)
        
        # Also check for NaNs/Infs
        if np.isnan(final_loss) or np.isinf(final_loss):
            return 0.0
            
        truth = np.exp(-self.beta * excess_loss)
        return float(np.clip(truth, 0.0, 1.0))

    def evaluate_global_satisfaction(self, p_weights: float, p_error: float) -> float:
        """
        Evaluates the global fuzzy logic satisfaction using the Product t-norm:
        $$I(\\phi \\land \\psi) = I(\\phi) \\times I(\\psi)$$
        """
        return float(p_weights * p_error)
