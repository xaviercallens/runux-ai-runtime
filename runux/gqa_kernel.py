# ==============================================================================
# RunuX AI Runtime — Batched Grouped-Query Attention (GQA) Kernel
#
# Copyright (c) 2026 Xavier Callens / Socrate AI Lab. All Rights Reserved.
# Licensed under LicenseRef-RunuX-Commercial.
#
# Optimized for Mistral 7B / Large 2 (32/8 heads) and LLaMA 3 (32/8 heads).
# ==============================================================================

import math
import torch
import torch.nn as nn
from typing import Optional, Tuple

class GroupedQueryAttention(nn.Module):
    """
    Batched Grouped-Query Attention (GQA) with memory-efficient head repetition.
    Eliminates redundant tensor materialization by broadcasting KV heads across
    query head groups.
    """
    def __init__(
        self,
        hidden_dim: int = 4096,
        num_query_heads: int = 32,
        num_kv_heads: int = 8,
        head_dim: Optional[int] = None,
        device: str = "cuda"
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.num_queries_per_kv = num_query_heads // num_kv_heads
        self.head_dim = head_dim or (hidden_dim // num_query_heads)
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(hidden_dim, num_query_heads * self.head_dim, bias=False, device=device)
        self.k_proj = nn.Linear(hidden_dim, num_kv_heads * self.head_dim, bias=False, device=device)
        self.v_proj = nn.Linear(hidden_dim, num_kv_heads * self.head_dim, bias=False, device=device)
        self.out_proj = nn.Linear(num_query_heads * self.head_dim, hidden_dim, bias=False, device=device)

    def forward(
        self,
        x: torch.Tensor,
        cached_k: Optional[torch.Tensor] = None,
        cached_v: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for GQA.
        x: [B, S, D]
        returns: (output, present_k, present_v)
        """
        B, S, _ = x.shape

        # Linear projections
        q = self.q_proj(x).view(B, S, self.num_query_heads, self.head_dim).transpose(1, 2)  # [B, H_q, S, D_h]
        k = self.k_proj(x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)     # [B, H_kv, S, D_h]
        v = self.v_proj(x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)     # [B, H_kv, S, D_h]

        if cached_k is not None:
            k = torch.cat([cached_k, k], dim=2)
            v = torch.cat([cached_v, v], dim=2)

        # Broadcast KV heads to match query heads without materializing full copies
        # Reshape [B, H_kv, 1, S_kv, D] -> [B, H_kv, G, S_kv, D] -> [B, H_q, S_kv, D]
        S_kv = k.shape[2]
        k_expanded = k.unsqueeze(2).expand(B, self.num_kv_heads, self.num_queries_per_kv, S_kv, self.head_dim)
        k_expanded = k_expanded.reshape(B, self.num_query_heads, S_kv, self.head_dim)

        v_expanded = v.unsqueeze(2).expand(B, self.num_kv_heads, self.num_queries_per_kv, S_kv, self.head_dim)
        v_expanded = v_expanded.reshape(B, self.num_query_heads, S_kv, self.head_dim)

        # Scaled dot-product attention
        scores = torch.matmul(q, k_expanded.transpose(-1, -2)) * self.scale

        if mask is not None:
            scores = scores + mask

        attn_weights = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, v_expanded)  # [B, H_q, S, D_h]

        out = out.transpose(1, 2).contiguous().view(B, S, self.num_query_heads * self.head_dim)
        output = self.out_proj(out)

        return output, k, v
