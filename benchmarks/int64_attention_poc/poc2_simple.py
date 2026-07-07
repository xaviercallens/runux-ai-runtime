#!/usr/bin/env python3
"""Simplified PoC 2: single-pass Triton kernel (no fancy tiling yet).

Instead of Flash-Attention style tiling, this is a straightforward
single-pass kernel that computes QK^T, LUT softmax, and AV all in one go.
Easier to compile, validates correctness before optimizing.
"""
import torch
import triton
import triton.language as tl


def build_int64_softmax_lut(lut_half=128, exp_div=16, fixed_scale=1 << 16, device="cuda"):
    """Precompute fixed-point exp() LUT, indexed by clamped score."""
    lut_size = 2 * lut_half
    idx = torch.arange(lut_size, dtype=torch.float64)
    x = (idx - lut_half) / exp_div
    values = torch.round(torch.exp(x) * fixed_scale).to(torch.int64)
    return values.to(device)


@triton.jit
def _simple_int64_attn_kernel(
    q_ptr, k_ptr, v_ptr, out_ptr, lut_ptr, den_ptr,
    stride_bh, stride_s, stride_d,
    SEQ_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SCALE_SHIFT: tl.constexpr,
    LUT_HALF: tl.constexpr,
):
    """Single-pass kernel: compute QK^T -> softmax -> AV in one go."""
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    m_start = pid_m * BLOCK_M
    m_offs = m_start + tl.arange(0, BLOCK_M)
    d_offs = tl.arange(0, HEAD_DIM)

    q_base = q_ptr + pid_bh * stride_bh
    k_base = k_ptr + pid_bh * stride_bh
    v_base = v_ptr + pid_bh * stride_bh
    out_base = out_ptr + pid_bh * stride_bh
    den_base = den_ptr + pid_bh * stride_bh

    # Load Q: [BLOCK_M, HEAD_DIM], compute manually instead of fancy reshape
    q_ptrs = q_base + tl.zeros((), dtype=tl.int64)
    q_row = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.int64)
    for i in tl.static_range(BLOCK_M):
        q_row = tl.where(i < BLOCK_M,
                          tl.load(q_base + (m_start + i) * stride_s + d_offs * stride_d),
                          q_row)

    # Actually, that's too complicated. Let me just load it the simple way.
    out_acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.int64)
    den_acc = tl.zeros((BLOCK_M,), dtype=tl.int64)

    for start_n in tl.static_range(0, SEQ_LEN, BLOCK_N):
        for i in tl.static_range(BLOCK_M):
            for j in tl.static_range(BLOCK_N):
                if start_n + j < SEQ_LEN and m_start + i < SEQ_LEN:
                    # QK^T[i, j] = sum_d Q[i, d] * K[start_n + j, d]
                    score = tl.zeros((), dtype=tl.int64)
                    for d in tl.static_range(HEAD_DIM):
                        q_val = tl.load(q_base + (m_start + i) * stride_s + d * stride_d)
                        k_val = tl.load(k_base + (start_n + j) * stride_s + d * stride_d)
                        score += q_val * k_val

                    score = score >> SCALE_SHIFT
                    # Manual clamp for int64
                    score_clamped = tl.maximum(tl.minimum(score, LUT_HALF - 1), -LUT_HALF)
                    idx = score_clamped + LUT_HALF
                    w = tl.load(lut_ptr + idx)
                    den_acc[i] += w

                    for d in tl.static_range(HEAD_DIM):
                        v_val = tl.load(v_base + (start_n + j) * stride_s + d * stride_d)
                        out_acc[i, d] += w * v_val

    # Store
    for i in tl.static_range(BLOCK_M):
        if m_start + i < SEQ_LEN:
            out_ptr_row = out_base + (m_start + i) * stride_s
            den_ptr_row = den_base + (m_start + i)
            for d in tl.static_range(HEAD_DIM):
                tl.store(out_ptr_row + d * stride_d, out_acc[i, d])
            tl.store(den_ptr_row, den_acc[i])


def simple_int64_attention_triton(q_int, k_int, v_int, lut, scale_shift, lut_half, block_m=32, block_n=32):
    """Wrapper: launch simple kernel."""
    B, H, S, D = q_int.shape
    assert q_int.is_contiguous() and k_int.is_contiguous() and v_int.is_contiguous()

    out = torch.zeros_like(q_int)
    den = torch.zeros((B, H, S), dtype=torch.int64, device=q_int.device)

    q_v = q_int.view(B * H, S, D)
    k_v = k_int.view(B * H, S, D)
    v_v = v_int.view(B * H, S, D)
    out_v = out.view(B * H, S, D)
    den_v = den.view(B * H, S)

    grid = ((S + block_m - 1) // block_m, B * H)

    _simple_int64_attn_kernel[grid](
        q_v, k_v, v_v, out_v, lut, den_v,
        q_v.stride(0), q_v.stride(1), q_v.stride(2),
        SEQ_LEN=S, HEAD_DIM=D, BLOCK_M=block_m, BLOCK_N=block_n,
        SCALE_SHIFT=scale_shift, LUT_HALF=lut_half,
    )

    # Division on host
    out_v[:] = out_v // den_v.unsqueeze(-1)
    return out
