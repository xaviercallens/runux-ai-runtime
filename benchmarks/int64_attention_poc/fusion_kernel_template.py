#!/usr/bin/env python3
"""Template for PoC 2 Triton/CUDA fusion kernel implementation.

This template provides a starting point for implementing the fused attention kernel.
Key requirements:
  1. Compute QK^T -> rescale -> softmax -> AV all in one pass
  2. Use INT64 LUT softmax (from build_int64_softmax_lut)
  3. Keep intermediates in SRAM/registers, avoid HBM round-trips
  4. Output must be bit-exact match to reference (determinism critical)

Implementation strategies:
  A. Triton @triton.jit kernel (use tl.static_range for T4 compatibility)
  B. Native CUDA C++ (fused_attention_kernel.cu)
  C. Pure PyTorch with careful memory management (fallback)
"""

import torch
import triton
import triton.language as tl


# Strategy A: Simple Triton kernel (starting point)
# NOTE: This is a template - adjust block sizes and optimize as needed

@triton.jit
def fused_int64_attention_simple(
    q_ptr, k_ptr, v_ptr, out_ptr, lut_ptr, den_ptr,
    stride_bh, stride_s, stride_d,
    SEQ_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SCALE_SHIFT: tl.constexpr,
    LUT_HALF: tl.constexpr,
):
    """
    Fused INT64 attention: QK^T -> rescale -> LUT softmax -> AV

    Grid: (num_query_blocks, batch*heads)
    Each block processes BLOCK_M query positions, iterates over key/value.
    """
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    # Query range for this block
    m_start = pid_m * BLOCK_M
    m_offs = m_start + tl.arange(0, BLOCK_M)
    d_offs = tl.arange(0, HEAD_DIM)

    # Base pointers for this (batch, head)
    q_base = q_ptr + pid_bh * stride_bh
    k_base = k_ptr + pid_bh * stride_bh
    v_base = v_ptr + pid_bh * stride_bh
    out_base = out_ptr + pid_bh * stride_bh
    den_base = den_ptr + pid_bh * stride_bh

    # Load Q block: [BLOCK_M, HEAD_DIM]
    # Note: This stays in registers for the whole loop - the "fusion" part
    q_ptrs = q_base + m_offs[:, None] * stride_s + d_offs[None, :] * stride_d
    q_block = tl.load(q_ptrs)

    # Accumulators for output and denominator
    acc_out = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.int64)
    acc_den = tl.zeros((BLOCK_M,), dtype=tl.int64)

    # Iterate over key/value blocks (Flash Attention style)
    # CRITICAL: Use explicit loop (not Python range) to avoid huge unrolling
    for kv_block_idx in tl.static_range((SEQ_LEN + BLOCK_N - 1) // BLOCK_N):
        n_start = kv_block_idx * BLOCK_N
        n_offs = n_start + tl.arange(0, BLOCK_N)
        n_mask = n_offs < SEQ_LEN

        # Load K: [BLOCK_N, HEAD_DIM]
        k_ptrs = k_base + n_offs[:, None] * stride_s + d_offs[None, :] * stride_d
        k_block = tl.load(k_ptrs, mask=n_mask[:, None], other=0)

        # Compute QK^T: [BLOCK_M, BLOCK_N]
        # Manual reduction over HEAD_DIM (no int64 GEMM available)
        qk = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int64)
        for d in tl.static_range(HEAD_DIM):
            q_d = tl.expand_dims(q_block[:, d], axis=1)  # [BLOCK_M, 1]
            k_d = tl.expand_dims(k_block[:, d], axis=0)  # [1, BLOCK_N]
            qk = qk + q_d * k_d

        # Rescale
        qk = qk >> SCALE_SHIFT

        # LUT softmax: clamp and gather from LUT
        idx = tl.maximum(tl.minimum(qk, LUT_HALF - 1), -LUT_HALF) + LUT_HALF
        weights = tl.load(lut_ptr + idx)

        # Accumulate denominator
        acc_den = acc_den + tl.sum(weights, axis=1)

        # Load V: [BLOCK_N, HEAD_DIM]
        v_ptrs = v_base + n_offs[:, None] * stride_s + d_offs[None, :] * stride_d
        v_block = tl.load(v_ptrs, mask=n_mask[:, None], other=0)

        # Weighted sum: weights @ values
        for d in tl.static_range(HEAD_DIM):
            v_d = tl.expand_dims(v_block[:, d], axis=0)  # [1, BLOCK_N]
            acc_out[:, d] = acc_out[:, d] + tl.sum(weights * v_d, axis=1)

    # Store output and denominator (division on host)
    out_ptrs = out_base + m_offs[:, None] * stride_s + d_offs[None, :] * stride_d
    tl.store(out_ptrs, acc_out)

    den_ptrs = den_base + m_offs * stride_s
    tl.store(den_ptrs, acc_den)


def fused_int64_attention_triton(q_int, k_int, v_int, lut, scale_shift, lut_half,
                                  block_m=32, block_n=32):
    """Wrapper to launch the fused kernel."""
    B, H, S, D = q_int.shape
    assert q_int.is_contiguous() and k_int.is_contiguous() and v_int.is_contiguous()

    # Adjust block sizes if needed
    block_m = min(block_m, S)
    block_n = min(block_n, S)

    # Output tensors
    out = torch.zeros_like(q_int)
    den = torch.zeros((B, H, S), dtype=torch.int64, device=q_int.device)

    q_v = q_int.view(B * H, S, D)
    k_v = k_int.view(B * H, S, D)
    v_v = v_int.view(B * H, S, D)
    out_v = out.view(B * H, S, D)
    den_v = den.view(B * H, S)

    # Grid: (num_m_blocks, batch*heads)
    grid = ((S + block_m - 1) // block_m, B * H)

    # Launch kernel
    fused_int64_attention_simple[grid](
        q_v, k_v, v_v, out_v, lut, den_v,
        q_v.stride(0), q_v.stride(1), q_v.stride(2),
        SEQ_LEN=S, HEAD_DIM=D,
        BLOCK_M=block_m, BLOCK_N=block_n,
        SCALE_SHIFT=scale_shift, LUT_HALF=lut_half,
    )

    # Division on host (avoid int64 div in Triton on T4)
    out_v[:] = out_v // den_v.unsqueeze(-1)
    return out


# ============================================================
# Helper: Pure PyTorch fusion (fallback, no actual fusion yet)
# ============================================================

def fused_int64_attention_pytorch(q_int, k_int, v_int, lut, scale_shift, lut_half, chunk=64):
    """Pure PyTorch version with explicit memory management (reference for correctness)."""
    B, H, S, D = q_int.shape
    out = torch.empty(B, H, S, D, dtype=torch.int64, device=q_int.device)

    for i in range(0, S, chunk):
        qc = q_int[:, :, i:i + chunk, :]
        scores = (qc.unsqueeze(3) * k_int.unsqueeze(2)).sum(dim=-1)
        scores = torch.bitwise_right_shift(scores, scale_shift)

        # LUT softmax
        idx = torch.clamp(scores, -lut_half, lut_half - 1) + lut_half
        weights = lut[idx]
        den = weights.sum(dim=-1, keepdim=True)

        # Weighted sum
        av = (weights.unsqueeze(-1) * v_int.unsqueeze(2)).sum(dim=-2)
        out[:, :, i:i + chunk, :] = av // den

    return out


if __name__ == "__main__":
    print("Fusion kernel template loaded")
    print("Usage: python3 fusion_harness.py --kernel fusion_kernel_template.py --test --benchmark")
