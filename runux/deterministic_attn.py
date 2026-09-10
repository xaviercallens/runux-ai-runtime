# ==============================================================================
# RunuX AI Runtime — INT64 Deterministic Attention with Fixed-Point LUT Softmax
#
# Copyright (c) 2026 Xavier Callens / Socrate AI Lab. All Rights Reserved.
# Licensed under LicenseRef-RunuX-Commercial.
#
# Patent Pending: INPI Dossier Demande Provisoire de Brevet RunuX (2026).
# ==============================================================================

import torch
import torch.nn as nn
from typing import Optional, Tuple

def build_int64_softmax_lut(
    lut_half: int = 128,
    exp_div: float = 16.0,
    fixed_scale: int = 1 << 16,
    device: str = "cuda"
) -> torch.Tensor:
    """
    Precompute a fixed-point exponential lookup table (ROM emulation).
    Maps clamped integer scores in [-lut_half, lut_half - 1] to exact
    fixed-point weights. Pure integer indexing at runtime.
    """
    lut_size = 2 * lut_half
    idx = torch.arange(lut_size, dtype=torch.float64, device=device)
    x = (idx - lut_half) / exp_div
    values = torch.round(torch.exp(x) * fixed_scale).to(torch.int64)
    return values

class Int64DeterministicAttention(nn.Module):
    """
    Deterministic Scaled Dot-Product Attention using pure integer arithmetic
    and fixed-point LUT Softmax. Eliminates IEEE-754 non-associative drift.
    """
    def __init__(
        self,
        lut_half: int = 128,
        scale_shift: int = 10,
        fixed_scale: int = 1 << 16,
        chunk_size: int = 64,
        device: str = "cuda"
    ):
        super().__init__()
        self.lut_half = lut_half
        self.scale_shift = scale_shift
        self.fixed_scale = fixed_scale
        self.chunk_size = chunk_size
        self.register_buffer("lut", build_int64_softmax_lut(lut_half, 16.0, fixed_scale, device))

    def forward(
        self,
        q_int: torch.Tensor,
        k_int: torch.Tensor,
        v_int: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes deterministic attention for integer quantized Q, K, V.
        Shapes: [B, H, S, D] -> returns [B, H, S, D].
        """
        B, H, S, D = q_int.shape
        out = torch.empty((B, H, S, D), dtype=torch.int64, device=q_int.device)
        chunk = self.chunk_size

        for i in range(0, S, chunk):
            c_end = min(i + chunk, S)
            qc = q_int[:, :, i:c_end, :]  # [B, H, c, D]

            # Exact integer dot product: sum over head_dim D
            # qc: [B, H, c, 1, D], k: [B, H, 1, S, D] -> [B, H, c, S]
            scores = (qc.unsqueeze(3) * k_int.unsqueeze(2)).sum(dim=-1)
            scores = torch.bitwise_right_shift(scores, self.scale_shift)

            # Integer clamp to LUT domain
            idx = torch.clamp(scores, -self.lut_half, self.lut_half - 1) + self.lut_half
            w = self.lut[idx]  # [B, H, c, S]

            # Weighted sum over V: w: [B, H, c, S, 1], v: [B, H, 1, S, D]
            num = (w.unsqueeze(-1) * v_int.unsqueeze(2)).sum(dim=-2)  # [B, H, c, D]
            den = w.sum(dim=-1, keepdim=True)  # [B, H, c, 1]

            # Integer division
            out[:, :, i:c_end, :] = num // torch.clamp(den, min=1)

        return out

    def verify_determinism(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_runs: int = 5) -> bool:
        """Runs the attention pass multiple times and checks for bit-exact equality."""
        first_pass = self.forward(q, k, v)
        for _ in range(num_runs - 1):
            next_pass = self.forward(q, k, v)
            if not torch.equal(first_pass, next_pass):
                return False
        return True
