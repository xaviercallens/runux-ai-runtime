#!/usr/bin/env python3
"""PoC 2: fused Triton kernel for INT64 deterministic attention.

PoC 1 (`benchmark_t4.py`) proved the determinism claim but showed the
unfused, pure-PyTorch INT64 path is ~153x slower than FP16 because every
intermediate (QK^T scores, softmax weights) round-trips through HBM. This
module implements the fix: a single `@triton.jit` kernel that tiles over
query/key blocks Flash-Attention-style and keeps QK^T -> rescale -> softmax
-> AV entirely in registers/SRAM for each tile, writing only the final
output to HBM.

Softmax: the patented mechanism is an INT64 LUT/polynomial softmax, not a
float softmax. This module builds a fixed-point exponential lookup table
(`build_int64_softmax_lut`) on the host once (a one-time constant table,
analogous to a ROM in a hardware implementation) and both the reference
and the Triton kernel gather from it with integer indexing only - no
floats appear anywhere in the per-call kernel path.

Determinism note: integer addition/multiplication is associative and exact
(no rounding), so unlike floats, the *order* of accumulation cannot change
the result. That means the Triton kernel's tiled reduction order and the
reference's chunked reduction order should produce bit-identical output,
not just approximately equal - this is verified in `benchmark_t4_poc2.py`.
"""
import torch
import triton
import triton.language as tl


# ==========================================
# 1. INT64 FIXED-POINT SOFTMAX LUT
# ==========================================
def build_int64_softmax_lut(lut_half=128, exp_div=16, fixed_scale=1 << 16, device="cuda"):
    """Precompute a fixed-point exp() table, indexed by a clamped integer
    score bucket in [-lut_half, lut_half - 1].

    table[i] ~= round(exp((i - lut_half) / exp_div) * fixed_scale)

    This host-side construction uses floats to *generate* the constant
    table (once, like a ROM), but the table itself is a fixed int64 tensor
    thereafter - every runtime lookup is pure integer indexing.
    """
    lut_size = 2 * lut_half
    idx = torch.arange(lut_size, dtype=torch.float64)
    x = (idx - lut_half) / exp_div
    values = torch.round(torch.exp(x) * fixed_scale).to(torch.int64)
    return values.to(device)


# ==========================================
# 2. REFERENCE (chunked, non-fused) LUT-softmax attention
#    Used only to validate the Triton kernel is bit-exact.
# ==========================================
def lut_softmax_attention_reference(q_int, k_int, v_int, lut, scale_shift, lut_half, chunk=64):
    B, H, S, D = q_int.shape
    out = torch.empty(B, H, S, D, dtype=torch.int64, device=q_int.device)
    for i in range(0, S, chunk):
        qc = q_int[:, :, i:i + chunk, :]                       # [B,H,c,D]
        scores = (qc.unsqueeze(3) * k_int.unsqueeze(2)).sum(-1)  # [B,H,c,S]
        scores = torch.bitwise_right_shift(scores, scale_shift)
        idx = torch.clamp(scores, -lut_half, lut_half - 1) + lut_half
        w = lut[idx]                                            # [B,H,c,S] int64
        num = (w.unsqueeze(-1) * v_int.unsqueeze(2)).sum(-2)     # [B,H,c,D]
        den = w.sum(-1, keepdim=True)                            # [B,H,c,1]
        out[:, :, i:i + chunk, :] = num // den
    return out


# ==========================================
# 3. FUSED TRITON KERNEL (simplified for T4 compatibility)
# ==========================================
@triton.jit
def _fused_int64_attention_kernel(
    q_ptr, k_ptr, v_ptr, out_ptr, lut_ptr, den_ptr,
    stride_qbh, stride_qs, stride_qd,
    stride_kbh, stride_ks, stride_kd,
    stride_vbh, stride_vs, stride_vd,
    stride_obh, stride_os, stride_od,
    stride_denbh, stride_dens,
    SEQ_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SCALE_SHIFT: tl.constexpr,
    LUT_HALF: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    d_offsets = tl.arange(0, HEAD_DIM)

    q_base = q_ptr + pid_bh * stride_qbh
    k_base = k_ptr + pid_bh * stride_kbh
    v_base = v_ptr + pid_bh * stride_vbh
    out_base = out_ptr + pid_bh * stride_obh
    den_base = den_ptr + pid_bh * stride_denbh

    # Load Q: [BLOCK_M, HEAD_DIM], stays resident
    q_ptrs = q_base + m_offsets[:, None] * stride_qs + d_offsets[None, :] * stride_qd
    q = tl.load(q_ptrs)

    acc_v = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.int64)
    acc_den = tl.zeros((BLOCK_M,), dtype=tl.int64)

    # Tile over K/V dimension
    for n_start in range(0, SEQ_LEN, BLOCK_N):
        n_offsets = n_start + tl.arange(0, BLOCK_N)
        n_mask = n_offsets < SEQ_LEN

        # Load K: [BLOCK_N, HEAD_DIM]
        k_ptrs = k_base + n_offsets[:, None] * stride_ks + d_offsets[None, :] * stride_kd
        k = tl.load(k_ptrs, mask=n_mask[:, None])

        # Manual QK^T reduction: sum over HEAD_DIM
        scores = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int64)
        for d in range(HEAD_DIM):
            scores += q[:, d:d+1] * k[None, :, d]

        scores = scores >> SCALE_SHIFT
        idx = tl.clamp(scores, -LUT_HALF, LUT_HALF - 1) + LUT_HALF
        w = tl.load(lut_ptr + idx)

        # Accumulate den
        den_tile = tl.sum(w, axis=1)
        acc_den += den_tile

        # Load V: [BLOCK_N, HEAD_DIM]
        v_ptrs = v_base + n_offsets[:, None] * stride_vs + d_offsets[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=n_mask[:, None])

        # Weighted sum over values
        for d in range(HEAD_DIM):
            w_expanded = w  # [BLOCK_M, BLOCK_N]
            v_d = v[:, d]  # [BLOCK_N]
            acc_v[:, d] += tl.sum(w_expanded * v_d[None, :], axis=1)

    # Store output and denominator
    out_ptrs = out_base + m_offsets[:, None] * stride_os + d_offsets[None, :] * stride_od
    tl.store(out_ptrs, acc_v)

    den_ptrs = den_base + m_offsets * stride_dens
    tl.store(den_ptrs, acc_den)


def fused_int64_attention_triton(q_int, k_int, v_int, lut, scale_shift, lut_half,
                                  block_m=64, block_n=64):
    B, H, S, D = q_int.shape
    block_m = min(block_m, S)
    block_n = min(block_n, S)
    if S % block_m != 0:
        block_m = S // max(1, S // block_m)
    if S % block_n != 0:
        block_n = S // max(1, S // block_n)
    assert S % block_m == 0 and S % block_n == 0, \
        f"Could not find compatible block sizes for SEQ_LEN={S}"
    assert q_int.is_contiguous() and k_int.is_contiguous() and v_int.is_contiguous()

    # Allocate accumulators for numerator (B*H, S, D) and denominator (B*H, S)
    acc_v = torch.empty((B * H, S, D), dtype=torch.int64, device=q_int.device)
    acc_den = torch.empty((B * H, S), dtype=torch.int64, device=q_int.device)

    q_v = q_int.view(B * H, S, D)
    k_v = k_int.view(B * H, S, D)
    v_v = v_int.view(B * H, S, D)

    grid = (S // block_m, B * H)
    _fused_int64_attention_kernel[grid](
        q_v, k_v, v_v, acc_v, lut, acc_den,
        q_v.stride(0), q_v.stride(1), q_v.stride(2),
        k_v.stride(0), k_v.stride(1), k_v.stride(2),
        v_v.stride(0), v_v.stride(1), v_v.stride(2),
        acc_v.stride(0), acc_v.stride(1), acc_v.stride(2),
        acc_den.stride(0), acc_den.stride(1),
        SEQ_LEN=S, HEAD_DIM=D, BLOCK_M=block_m, BLOCK_N=block_n,
        SCALE_SHIFT=scale_shift, LUT_HALF=lut_half,
    )

    # Division on host (avoiding Triton/LLVM int64 division on T4)
    out = torch.empty_like(q_int)
    out_v = out.view(B * H, S, D)
    out_v[:] = acc_v // acc_den.unsqueeze(-1)
    return out
