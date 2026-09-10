# ==============================================================================
# RunuX AI Runtime — Integration Tests: Partner Evaluation Package
#
# Copyright (c) 2026 Xavier Callens / Socrate AI Lab. All Rights Reserved.
# Licensed under LicenseRef-RunuX-Commercial.
# ==============================================================================

import pytest
import numpy as np

from evaluation_package.harness_mistral import (
    ModelProfile,
    evaluate_kv_cache_compression,
    verify_polarquant_energy_preservation,
    evaluate_carbon_aware_speculative,
    main as mistral_main,
)
from evaluation_package.harness_nvidia import (
    build_int64_softmax_lut,
    test_deterministic_attention_runs as harness_test_det_attn,
    evaluate_signsgd_compression,
    main as nvidia_main,
)
from evaluation_package.harness_google import (
    TpuSystolicModel,
    main as google_main,
)


# -----------------------------------------------------------------------------
# 1. Mistral AI Evaluation Harness Tests
# -----------------------------------------------------------------------------
def test_mistral_harness_kv_compression():
    model = ModelProfile(
        name="Mistral Large 2 (123B)",
        params_b=123.0,
        hidden_dim=12288,
        num_layers=88,
        num_heads=96,
        num_kv_heads=8,
        head_dim=128,
        context_window=131072,
    )
    res = evaluate_kv_cache_compression(model)
    assert res["model"] == "Mistral Large 2 (123B)"
    assert res["fp16_vram_gb"] > res["polarquant_vram_gb"]
    assert res["compression_vs_fp16"] > 4.5
    assert res["compression_vs_fp8"] > 2.0


def test_mistral_harness_energy_and_carbon():
    # Orthogonal rotation energy preservation simulation
    ok = verify_polarquant_energy_preservation(64)
    assert ok is True

    # Speculative scheduling
    res = evaluate_carbon_aware_speculative(base_tps=40.0)
    assert "France (RTE Nuclear Mix)" in res
    assert "USA (Avg Electric Mix)" in res
    assert "China (Coal Dominant)" in res

    fr = res["France (RTE Nuclear Mix)"]
    cn = res["China (Coal Dominant)"]
    assert fr["draft_k"] > cn["draft_k"]
    assert fr["gco2_per_1k_tokens"] < cn["gco2_per_1k_tokens"]


def test_mistral_harness_main(capsys):
    mistral_main()
    captured = capsys.readouterr()
    assert "RunuX Evaluation Harness: Mistral AI" in captured.out
    assert "Completed Successfully" in captured.out


# -----------------------------------------------------------------------------
# 2. NVIDIA Evaluation Harness Tests
# -----------------------------------------------------------------------------
def test_nvidia_harness_deterministic_attention():
    lut = build_int64_softmax_lut(lut_half=128, exp_div=16.0, fixed_scale=1 << 16)
    assert len(lut) == 256
    assert lut[0] >= 0

    det_res = harness_test_det_attn(num_runs=3)
    assert det_res["bit_exact"] is True
    assert det_res["max_abs_drift"] == 0.0


def test_nvidia_harness_signsgd_compression():
    res = evaluate_signsgd_compression(num_params=70_000_000_000)
    assert res["params"] == "70.0B"
    assert res["compression_ratio"] == "32.0x"
    assert res["comm_speedup"] > 10.0


def test_nvidia_harness_main(capsys):
    nvidia_main()
    captured = capsys.readouterr()
    assert "RunuX Evaluation Harness: NVIDIA" in captured.out
    assert "Completed Successfully" in captured.out


# -----------------------------------------------------------------------------
# 3. Google Cloud Evaluation Harness Tests
# -----------------------------------------------------------------------------
def test_google_harness_systolic_model():
    v5e = TpuSystolicModel("v5e")
    assert v5e.platform == "TPU v5e"
    assert v5e.mxu_dim == 128

    v6e = TpuSystolicModel("v6e")
    assert v6e.platform == "TPU v6e (Trillium)"
    assert v6e.mxu_dim == 256

    with pytest.raises(ValueError, match="Unknown TPU platform"):
        TpuSystolicModel("v4_unknown")

    gemm_res = v5e.evaluate_gemm("Gemma 9B", 1, 3584, 3584)
    assert gemm_res["runux_tflops"] > gemm_res["base_tflops"]
    assert "88.0%" in gemm_res["runux_occupancy"]


def test_google_harness_main(capsys):
    google_main()
    captured = capsys.readouterr()
    assert "RunuX Evaluation Harness: Google Cloud" in captured.out
    assert "Completed Successfully" in captured.out
