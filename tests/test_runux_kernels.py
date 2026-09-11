# ==============================================================================
# RunuX AI Runtime — Unit Tests: Core Acceleration Kernels & Modules
#
# Copyright (c) 2026 Xavier Callens / Socrate AI Lab. All Rights Reserved.
# Licensed under LicenseRef-RunuX-Commercial.
# ==============================================================================

import pytest
import math
import torch
import torch.nn as nn

from runux.gqa_kernel import GroupedQueryAttention
from runux.paged_cache import PagedKVCache
from runux.polarquant import (
    PolarQuantKVCache,
    PolarQuantConfig,
    generate_splitmix64_orthogonal_matrix,
)
from runux.deterministic_attn import (
    Int64DeterministicAttention,
    build_int64_softmax_lut,
)
from runux.signsgd import SignSGDOptimizer
from runux.carbon_scheduler import (
    CarbonAwareSpeculativeScheduler,
    GridCarbonProfile,
)
from runux.systolic_advisor import SystolicTilingAdvisor
from runux.transformer_engine import RMSNorm, SwiGLU, TransformerBlock

# --- GPU availability guards (AUDIT2_REPORT §3.2) ---
import torch as _torch
_HAS_CUDA = _torch.cuda.is_available()
_HAS_T4   = _HAS_CUDA and "T4" in _torch.cuda.get_device_name(0)

requires_gpu = pytest.mark.skipif(not _HAS_CUDA, reason="Requires CUDA GPU")
requires_t4  = pytest.mark.skipif(not _HAS_T4,   reason="Requires NVIDIA Tesla T4")
# --- end guards ---


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------------------------------------------------------------
# 1. GroupedQueryAttention Tests
# -----------------------------------------------------------------------------
@requires_gpu
def test_gqa_kernel_initialization_and_shapes():
    hidden_dim = 128
    num_q_heads = 8
    num_kv_heads = 2
    head_dim = 16

    gqa = GroupedQueryAttention(
        hidden_dim=hidden_dim,
        num_query_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        device=DEVICE,
    )
    assert gqa.num_queries_per_kv == 4
    assert gqa.scale == 1.0 / math.sqrt(head_dim)

    # Forward pass without cache
    B, S = 2, 8
    x = torch.randn(B, S, hidden_dim, device=DEVICE)
    out, k, v = gqa(x)

    assert out.shape == (B, S, hidden_dim)
    assert k.shape == (B, num_kv_heads, S, head_dim)
    assert v.shape == (B, num_kv_heads, S, head_dim)
    assert torch.all(torch.isfinite(out))


@requires_gpu
def test_gqa_kernel_with_kv_cache_and_mask():
    hidden_dim = 64
    num_q_heads = 4
    num_kv_heads = 2
    head_dim = 16

    gqa = GroupedQueryAttention(
        hidden_dim=hidden_dim,
        num_query_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        device=DEVICE,
    )

    B = 1
    # Step 1: prompt tokens
    x_prompt = torch.randn(B, 4, hidden_dim, device=DEVICE)
    out_prompt, k_prompt, v_prompt = gqa(x_prompt)

    # Step 2: single next token with KV cache
    x_next = torch.randn(B, 1, hidden_dim, device=DEVICE)
    # Mask of shape [B, num_q_heads, 1, 5]
    mask = torch.zeros(B, num_q_heads, 1, 5, device=DEVICE)
    mask[:, :, :, 0] = -1e4  # mask out first token

    out_next, k_total, v_total = gqa(
        x_next, cached_k=k_prompt, cached_v=v_prompt, mask=mask
    )

    assert out_next.shape == (B, 1, hidden_dim)
    assert k_total.shape == (B, num_kv_heads, 5, head_dim)
    assert v_total.shape == (B, num_kv_heads, 5, head_dim)
    assert torch.all(torch.isfinite(out_next))


# -----------------------------------------------------------------------------
# 2. PagedKVCache Tests
# -----------------------------------------------------------------------------
def test_paged_cache_allocation_and_append():
    num_blocks = 4
    block_size = 4
    num_heads = 2
    head_dim = 8

    cache = PagedKVCache(
        num_blocks=num_blocks,
        block_size=block_size,
        num_heads=num_heads,
        head_dim=head_dim,
        dtype=torch.float32,
        device=DEVICE,
    )

    # Sequence allocation
    cache.allocate_sequence(seq_id=101)
    with pytest.raises(ValueError, match="already exists"):
        cache.allocate_sequence(seq_id=101)

    # Append tokens crossing block boundary: 6 tokens -> requires 2 blocks
    written_k = []
    written_v = []
    for t in range(6):
        k_t = torch.full((num_heads, head_dim), fill_value=float(t), device=DEVICE)
        v_t = torch.full((num_heads, head_dim), fill_value=float(t * 10), device=DEVICE)
        cache.append_kv(seq_id=101, k_token=k_t, v_token=v_t)
        written_k.append(k_t)
        written_v.append(v_t)

    assert cache.seq_lengths[101] == 6
    assert len(cache.block_tables[101]) == 2
    assert len(cache.free_blocks) == num_blocks - 2

    # Reconstruction test
    k_rec, v_rec = cache.get_sequence_kv(seq_id=101)
    assert k_rec.shape == (num_heads, 6, head_dim)
    assert v_rec.shape == (num_heads, 6, head_dim)
    for t in range(6):
        assert torch.allclose(k_rec[:, t, :], written_k[t])
        assert torch.allclose(v_rec[:, t, :], written_v[t])


def test_paged_cache_empty_sequence_and_free():
    cache = PagedKVCache(num_blocks=2, block_size=4, num_heads=2, head_dim=8, device=DEVICE)
    cache.allocate_sequence(seq_id=202)
    k_empty, v_empty = cache.get_sequence_kv(seq_id=202)
    assert k_empty.shape == (2, 0, 8)
    assert v_empty.shape == (2, 0, 8)

    # Free sequence
    cache.free_sequence(seq_id=202)
    assert 202 not in cache.block_tables
    # Free non-existent sequence should be a no-op
    cache.free_sequence(seq_id=999)


def test_paged_cache_oom_and_stats():
    cache = PagedKVCache(num_blocks=2, block_size=2, num_heads=1, head_dim=4, device=DEVICE)
    cache.allocate_sequence(seq_id=1)

    # Write 4 tokens (fills both 2 blocks)
    for _ in range(4):
        cache.append_kv(
            1,
            torch.zeros(1, 4, device=DEVICE),
            torch.zeros(1, 4, device=DEVICE),
        )

    # Fifth token should trigger OOM
    with pytest.raises(RuntimeError, match="Out Of Memory"):
        cache.append_kv(
            1,
            torch.zeros(1, 4, device=DEVICE),
            torch.zeros(1, 4, device=DEVICE),
        )

    stats = cache.memory_stats()
    assert stats["num_blocks"] == 2
    assert stats["used_blocks"] == 2
    assert stats["free_blocks"] == 0
    assert stats["total_tokens_stored"] == 4
    assert stats["external_fragmentation_pct"] == 0.0


# -----------------------------------------------------------------------------
# 3. PolarQuant Tests
# -----------------------------------------------------------------------------
@requires_gpu
def test_polarquant_splitmix64_orthogonality():
    dim = 32
    R = generate_splitmix64_orthogonal_matrix(dim, seed=42, device=DEVICE)
    assert R.shape == (dim, dim)

    # Verify R * R^T = Identity
    identity = torch.eye(dim, device=DEVICE)
    gram = torch.matmul(R, R.T)
    assert torch.allclose(gram, identity, atol=1e-5)


@requires_gpu
def test_polarquant_compress_decompress_and_energy():
    dim = 32
    cfg = PolarQuantConfig(bits=3, seed=42, energy_threshold=0.35)
    pq = PolarQuantKVCache(head_dim=dim, config=cfg, device=DEVICE)

    # Input tensor [B, H, S, D]
    x = torch.randn(2, 4, 16, dim, device=DEVICE)
    q_3bit, x_min, scale = pq.compress(x)

    assert q_3bit.dtype == torch.uint8
    assert q_3bit.max().item() <= 7
    assert q_3bit.min().item() >= 0
    assert x_min.shape == (2, 4, 16, 1)
    assert scale.shape == (2, 4, 16, 1)

    # Decompress
    x_rec = pq.decompress(q_3bit, x_min, scale)
    assert x_rec.shape == x.shape

    # Energy preservation
    metrics = pq.evaluate_energy_preservation(x)
    assert metrics["relative_norm_diff"] < 0.35
    assert metrics["cosine_similarity"] > 0.85
    assert metrics["energy_preserved"] is True


# -----------------------------------------------------------------------------
# 4. Int64DeterministicAttention Tests
# -----------------------------------------------------------------------------
def test_deterministic_attn_lut_and_forward():
    lut = build_int64_softmax_lut(lut_half=64, exp_div=16.0, fixed_scale=1 << 16, device=DEVICE)
    assert lut.shape == (128,)
    assert lut.dtype == torch.int64
    # Values should be monotonically non-decreasing
    assert torch.all(lut[1:] >= lut[:-1])

    attn = Int64DeterministicAttention(
        lut_half=64,
        scale_shift=8,
        fixed_scale=1 << 16,
        chunk_size=16,
        device=DEVICE,
    )

    B, H, S, D = 1, 2, 32, 16
    q = torch.randint(-64, 63, (B, H, S, D), dtype=torch.int64, device=DEVICE)
    k = torch.randint(-64, 63, (B, H, S, D), dtype=torch.int64, device=DEVICE)
    v = torch.randint(-64, 63, (B, H, S, D), dtype=torch.int64, device=DEVICE)

    out = attn(q, k, v)
    assert out.shape == (B, H, S, D)
    assert out.dtype == torch.int64

    # Check strict bit-exact determinism across 5 runs
    is_deterministic = attn.verify_determinism(q, k, v, num_runs=5)
    assert is_deterministic is True


# -----------------------------------------------------------------------------
# 5. SignSGDOptimizer Tests
# -----------------------------------------------------------------------------
@requires_gpu
def test_signsgd_optimizer_step_and_bandwidth():
    with pytest.raises(ValueError, match="Invalid learning rate"):
        SignSGDOptimizer([nn.Parameter(torch.randn(4))], lr=-0.01)

    model = nn.Linear(10, 2, device=DEVICE)
    optimizer = SignSGDOptimizer(
        model.parameters(), lr=0.01, weight_decay=1e-4, momentum=0.9
    )

    # First step
    x = torch.randn(4, 10, device=DEVICE)
    loss = model(x).sum()
    loss.backward()
    optimizer.step()

    # Second step with closure
    def closure():
        optimizer.zero_grad()
        loss = model(x).sum()
        loss.backward()
        return loss

    loss_val = optimizer.step(closure)
    assert loss_val is not None

    # Check bandwidth estimation
    bw_stats = optimizer.estimate_bandwidth_savings(num_params=70_000_000_000)
    assert bw_stats["num_params"] == 70_000_000_000
    assert bw_stats["bandwidth_reduction"] == "32.0x"
    assert bw_stats["fp32_sync_mb"] > bw_stats["sign1bit_sync_mb"]


# -----------------------------------------------------------------------------
# 6. CarbonAwareSpeculativeScheduler Tests
# -----------------------------------------------------------------------------
def test_carbon_scheduler_optimization_and_emissions():
    profile = GridCarbonProfile(
        region_name="France (RTE Nuclear)",
        carbon_intensity_gco2_kwh=56.0,
        base_power_watts=180.0,
        peak_power_watts=300.0,
    )
    scheduler = CarbonAwareSpeculativeScheduler(
        grid_profile=profile, k_min=2, k_max=7, c_ref=56.0, gamma=0.65
    )

    # Low carbon grid (clean France mix) -> high draft K
    k_clean = scheduler.compute_optimal_draft_k(current_carbon_gco2_kwh=30.0, acceptance_rate=0.8)
    assert k_clean >= 5

    # High carbon grid (coal mix) -> low draft K
    k_dirty = scheduler.compute_optimal_draft_k(current_carbon_gco2_kwh=600.0, acceptance_rate=0.5)
    assert k_dirty <= 3

    # Emission evaluation
    emissions = scheduler.evaluate_inference_emissions(
        num_tokens=1000,
        effective_tps=50.0,
        power_watts=250.0,
        carbon_gco2_kwh=56.0,
    )
    assert emissions["elapsed_seconds"] == 20.0
    assert emissions["total_joules"] == 5000.0
    assert emissions["joules_per_token"] == 5.0
    assert emissions["total_gco2"] > 0.0
    assert emissions["gco2_per_1k_tokens"] > 0.0


# -----------------------------------------------------------------------------
# 7. SystolicTilingAdvisor Tests
# -----------------------------------------------------------------------------
def test_systolic_advisor_platforms():
    for plat in ["tpu_v5e", "tpu_v6e", "nvidia_t4", "nvidia_h100"]:
        advisor = SystolicTilingAdvisor(platform=plat)
        res = advisor.compute_optimal_tiling(m=100, k=200, n=300)

        assert res["platform"] == advisor.profile.name
        assert res["raw_shape"] == (100, 200, 300)
        assert res["padded_shape"][0] >= 100
        assert res["padded_shape"][1] >= 200
        assert res["padded_shape"][2] >= 300
        assert "speedup" in res
        assert float(res["speedup"].replace("x", "")) > 1.0


# -----------------------------------------------------------------------------
# 8. TransformerEngine Tests
# -----------------------------------------------------------------------------
def test_transformer_engine_components():
    dim = 64
    # RMSNorm
    norm = RMSNorm(dim=dim, eps=1e-5, device=DEVICE)
    x = torch.randn(2, 4, dim, device=DEVICE) * 5.0
    normed = norm(x)
    assert normed.shape == x.shape
    var = torch.mean(normed ** 2, dim=-1)
    assert torch.allclose(var, torch.ones_like(var), atol=0.2)

    # SwiGLU
    mlp = SwiGLU(hidden_dim=dim, intermediate_dim=128, device=DEVICE)
    mlp_out = mlp(x)
    assert mlp_out.shape == x.shape
    assert torch.all(torch.isfinite(mlp_out))

    # TransformerBlock
    block = TransformerBlock(
        hidden_dim=dim,
        intermediate_dim=128,
        num_query_heads=4,
        num_kv_heads=2,
        head_dim=16,
        device=DEVICE,
    )
    out, k, v = block(x)
    assert out.shape == x.shape
    assert k.shape == (2, 2, 4, 16)
    assert v.shape == (2, 2, 4, 16)
