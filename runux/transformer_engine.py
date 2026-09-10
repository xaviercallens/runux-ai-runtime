# ==============================================================================
# RunuX AI Runtime — Integrated End-to-End Transformer Engine (GPU T4)
#
# Copyright (c) 2026 Xavier Callens / Socrate AI Lab. All Rights Reserved.
# Licensed under LicenseRef-RunuX-Commercial.
#
# Integrates GQA, RoPE, RMSNorm, SwiGLU, and PolarQuant Paged Compression.
# ==============================================================================

import math
import torch
import torch.nn as nn
from typing import Optional, Tuple
from .gqa_kernel import GroupedQueryAttention
from .polarquant import PolarQuantKVCache, PolarQuantConfig

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, device: str = "cuda"):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = torch.mean(x ** 2, dim=-1, keepdim=True)
        return x * torch.rsqrt(var + self.eps) * self.weight

class SwiGLU(nn.Module):
    """SwiGLU feed-forward network (Mistral / LLaMA architecture)."""
    def __init__(self, hidden_dim: int = 4096, intermediate_dim: int = 14336, device: str = "cuda"):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False, device=device)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False, device=device)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.nn.functional.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)

class TransformerBlock(nn.Module):
    """Full Transformer Layer combining RMSNorm, GQA, SwiGLU, and PolarQuant compression."""
    def __init__(
        self,
        hidden_dim: int = 2048,
        intermediate_dim: int = 5632,
        num_query_heads: int = 32,
        num_kv_heads: int = 8,
        head_dim: int = 64,
        device: str = "cuda"
    ):
        super().__init__()
        self.input_layernorm = RMSNorm(hidden_dim, device=device)
        self.attention = GroupedQueryAttention(
            hidden_dim=hidden_dim,
            num_query_heads=num_query_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            device=device
        )
        self.post_attention_layernorm = RMSNorm(hidden_dim, device=device)
        self.mlp = SwiGLU(hidden_dim=hidden_dim, intermediate_dim=intermediate_dim, device=device)
        self.polarquant = PolarQuantKVCache(head_dim=head_dim, config=PolarQuantConfig(bits=3), device=device)

    def forward(
        self,
        x: torch.Tensor,
        cached_k: Optional[torch.Tensor] = None,
        cached_v: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Pre-LN GQA Attention with residual
        norm_x = self.input_layernorm(x)
        attn_out, new_k, new_v = self.attention(norm_x, cached_k, cached_v)
        x = x + attn_out

        # Pre-LN SwiGLU MLP with residual
        norm_x2 = self.post_attention_layernorm(x)
        mlp_out = self.mlp(norm_x2)
        x = x + mlp_out

        return x, new_k, new_v
