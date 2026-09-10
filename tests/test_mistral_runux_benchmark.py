# ==============================================================================
# RunuX AI Runtime — Integration Tests: Mistral 7B Benchmark & Gains
#
# Copyright (c) 2026 Xavier Callens / Socrate AI Lab. All Rights Reserved.
# Licensed under LicenseRef-RunuX-Commercial.
# ==============================================================================

import os
import pytest
from scripts.benchmark_mistral_runux_gains import (
    parse_gguf_metadata,
    evaluate_t4_vram_and_oom_scaling,
    benchmark_physical_t4_kernels,
    evaluate_tpu_architectural_gains,
    evaluate_grid_carbon_shifting,
    DEFAULT_MODEL_PATH,
)


def test_mistral_gguf_parsing():
    if not os.path.isfile(DEFAULT_MODEL_PATH):
        pytest.skip(f"Model file {DEFAULT_MODEL_PATH} not present")

    meta = parse_gguf_metadata(DEFAULT_MODEL_PATH)
    assert meta["architecture"] == "llama"
    assert meta["block_count"] == 32
    assert meta["embedding_length"] == 4096
    assert meta["head_count"] == 32
    assert meta["head_count_kv"] == 8
    assert meta["context_length"] == 32768
    assert meta["file_size_gb"] > 3.5 and meta["file_size_gb"] < 5.0


def test_mistral_vram_and_oom_scaling():
    meta = {
        "block_count": 32,
        "head_count_kv": 8,
        "head_dim": 128,
        "file_size_gb": 4.07,
    }
    rows = evaluate_t4_vram_and_oom_scaling(meta)
    assert len(rows) == 7

    # Check 32k context behavior
    row_32k = [r for r in rows if r["context_length"] == 32768][0]
    assert row_32k["baseline_status"] == "CUDA OOM (FAIL)"
    assert row_32k["runux_status"] == "OK (Headroom)"
    assert row_32k["kv_compression_ratio"] > 4.5
    assert row_32k["runux_headroom_gb"] > 8.0


def test_mistral_tpu_gains():
    meta = {"head_count": 32, "head_count_kv": 8, "head_dim": 128}
    tpu_gains = evaluate_tpu_architectural_gains(meta)
    assert "TPU v5e" in tpu_gains
    assert "TPU v6e Trillium" in tpu_gains

    v5e = tpu_gains["TPU v5e"]
    assert v5e["runux_mlgo_occupancy_pct"] == 88.0
    assert float(v5e["throughput_speedup"].replace("x", "")) >= 2.0
    assert v5e["tokens_lost_on_preemption"] == 0


def test_mistral_carbon_shifting():
    res = evaluate_grid_carbon_shifting()
    assert "France (RTE Nuclear & Hydro)" in res["regions"]
    assert "China (Coal Dominant)" in res["regions"]
    fr_em = res["regions"]["France (RTE Nuclear & Hydro)"]["emissions_gco2_per_1k_tokens"]
    cn_em = res["regions"]["China (Coal Dominant)"]["emissions_gco2_per_1k_tokens"]
    assert fr_em < cn_em
