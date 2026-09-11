"""
Tests for 1-Hour Soak Benchmark Protocol and Zenodo Open Science Packaging.
Verifies hardware measurement integrity, Spot TPU preemption resilience,
and strict IP/trade-secret sanitation (no engine binaries/code in public bundles).
"""

import os
import json
from pathlib import Path
import pytest

from workflow_soak_benchmark import (
    query_gpu_telemetry,
    run_t4_sustained_soak_benchmark,
    run_tpu_spot_serverless_soak_protocol,
    generate_zenodo_bundle,
    retrofit_soak_results_to_datasets,
)

# --- GPU availability guards (AUDIT2_REPORT §3.2) ---
import torch as _torch
_HAS_CUDA = _torch.cuda.is_available()
_HAS_T4   = _HAS_CUDA and "T4" in _torch.cuda.get_device_name(0)

requires_gpu = pytest.mark.skipif(not _HAS_CUDA, reason="Requires CUDA GPU")
requires_t4  = pytest.mark.skipif(not _HAS_T4,   reason="Requires NVIDIA Tesla T4")
# --- end guards ---


@requires_t4
def test_query_gpu_telemetry():
    """Telemetry should return a dictionary with all required keys."""
    metrics = query_gpu_telemetry()
    assert isinstance(metrics, dict)
    assert "temperature_c" in metrics
    assert "power_draw_w" in metrics
    assert "gpu_util_pct" in metrics
    assert "mem_used_mb" in metrics
    assert "mem_total_mb" in metrics
    assert metrics["temperature_c"] > 0
    assert metrics["mem_total_mb"] > 0


@requires_t4
def test_t4_sustained_soak_short_run():
    """Verify that soak benchmark executes on hardware and records time series."""
    res = run_t4_sustained_soak_benchmark(
        total_minutes=2,
        iterations_per_window=10,
        accelerated=True,
        real_window_sleep_s=0.01,
    )
    assert "summary" in res
    assert "time_series" in res
    assert len(res["time_series"]) == 2

    summary = res["summary"]
    assert summary["duration_minutes"] == 2
    assert summary["throughput_mean_tflops"] > 0.0
    assert summary["thermal_equilibrium_achieved"] is True
    assert summary["memory_leak_detected"] is False
    assert summary["int64_bit_exact_zero_drift"] is True
    assert "Tesla T4" in summary["device"]

    first_window = res["time_series"][0]
    assert first_window["minute"] == 1
    assert first_window["avg_latency_ms"] > 0.0
    assert first_window["p99_latency_ms"] >= first_window["p50_latency_ms"]


@requires_t4
def test_tpu_spot_serverless_soak_protocol():
    """Verify that Spot Serverless TPU protocol handles preemptions with 0 token loss."""
    res = run_tpu_spot_serverless_soak_protocol(total_minutes=60)
    assert "summary" in res
    assert "time_series" in res
    assert len(res["time_series"]) == 60

    summary = res["summary"]
    assert summary["preemption_events_handled"] == 2
    assert summary["mean_checkpoint_latency_ms"] < 12.0
    assert summary["token_loss_percentage"] == 0.0
    assert summary["status"] == "PASS (Zero Token Loss Under Spot Preemption)"


def test_zenodo_bundle_sanitation(tmp_path):
    """
    STRICT TRADE SECRET SANITATION TEST:
    Ensures that the Zenodo bundle NEVER contains proprietary Rust source code,
    Cargo.lock, target binaries, or private credentials.
    """
    public_rel = Path("public_release")
    bundle_dir = tmp_path / "zenodo_test_bundle"

    dummy_t4 = {"summary": {"test": True}, "time_series": []}
    dummy_tpu = {"summary": {"test": True}, "time_series": []}

    zenodo_json, tar_path = generate_zenodo_bundle(
        public_release_dir=public_rel,
        bundle_dir=bundle_dir,
        t4_data=dummy_t4,
        tpu_data=dummy_tpu,
    )

    assert zenodo_json.exists()
    assert tar_path.exists()

    with open(zenodo_json, "r") as f:
        meta = json.load(f)
    assert "title" in meta
    assert "creators" in meta
    assert meta["license"] == "CC-BY-4.0"
    assert meta["access_right"] == "open"

    # Scan all files in bundle
    forbidden_extensions = {".rs", ".rlib", ".so", ".dylib", ".dll", ".a"}
    for p in bundle_dir.rglob("*"):
        if p.is_file():
            assert p.suffix not in forbidden_extensions, f"Proprietary file found in bundle: {p}"
            assert "target" not in p.parts, f"Build target found in bundle: {p}"
            assert "crates" not in p.parts, f"Internal crate found in bundle: {p}"

    # Clean up test tar
    if tar_path.exists():
        tar_path.unlink()


@requires_t4
def test_t4_soak_monitor_short_run(tmp_path):
    """Test T4SoakMonitor initialization, live status updating, and dataset flushing."""
    from monitor_t4_1hour_benchmark import T4SoakMonitor

    monitor = T4SoakMonitor(
        duration_seconds=2,
        window_seconds=1,
        output_dir=str(tmp_path),
    )
    assert monitor.duration_seconds == 2
    assert monitor.window_seconds == 1
    assert monitor.total_windows == 2

    monitor.run()

    # Verify output datasets
    status_file = tmp_path / "datasets" / "t4_soak_live_status.json"
    json_file = tmp_path / "datasets" / "t4_long_duration_soak_1hour.json"
    csv_file = tmp_path / "datasets" / "t4_long_duration_soak_1hour.csv"

    assert status_file.exists()
    assert json_file.exists()
    assert csv_file.exists()

    with open(status_file, "r") as f:
        status_data = json.load(f)
    assert status_data["status"] == "COMPLETED"
    assert status_data["progress_percent"] == 100.0
    assert status_data["int64_deterministic_drift"] == 0.0

    with open(json_file, "r") as f:
        soak_data = json.load(f)
    assert "summary" in soak_data
    assert "time_series" in soak_data
    assert len(soak_data["time_series"]) == 2
    assert soak_data["summary"]["memory_leak_detected"] is False
    assert soak_data["summary"]["int64_bit_exact_zero_drift"] is True
