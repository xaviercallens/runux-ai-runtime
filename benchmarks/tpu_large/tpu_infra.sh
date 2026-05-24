#!/bin/bash
# RunuX-AI — TPU v5e Benchmark Infrastructure
# Copyright (c) 2026 Xavier Callens / Socrate AI Lab
# 
# Usage:
#   ./tpu_infra.sh create    — Provision TPU v5e-1 (spot/preemptible)
#   ./tpu_infra.sh run       — Run benchmark suite
#   ./tpu_infra.sh results   — Download results
#   ./tpu_infra.sh destroy   — Teardown TPU + cleanup
#   ./tpu_infra.sh cost      — Check billing

set -euo pipefail

# ============================================================================
# Configuration
# ============================================================================
PROJECT="gen-lang-client-0625573011"
TPU_NAME="runux-bench-v5e"
ZONE="us-west4-a"  # TPU v5e (v5litepod) confirmed available
TPU_TYPE="v5litepod-1"  # Single TPU v5e chip
RUNTIME_VERSION="tpu-ubuntu2204-base"
BUDGET_MAX=80  # USD

# ============================================================================
# Functions
# ============================================================================

create_tpu() {
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║  Creating TPU v5e-1 (spot) — Budget: \$${BUDGET_MAX}        ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    
    # Try spot first (cheapest), fall back to preemptible, then on-demand
    echo "  Attempting spot TPU..."
    gcloud compute tpus tpu-vm create "${TPU_NAME}" \
        --zone="${ZONE}" \
        --accelerator-type="${TPU_TYPE}" \
        --version="${RUNTIME_VERSION}" \
        --spot \
        --project="${PROJECT}" \
        2>&1 || {
            echo "  Spot not available, trying preemptible..."
            gcloud compute tpus tpu-vm create "${TPU_NAME}" \
                --zone="${ZONE}" \
                --accelerator-type="${TPU_TYPE}" \
                --version="${RUNTIME_VERSION}" \
                --preemptible \
                --project="${PROJECT}" \
                2>&1 || {
                    echo "  Preemptible not available, trying on-demand..."
                    gcloud compute tpus tpu-vm create "${TPU_NAME}" \
                        --zone="${ZONE}" \
                        --accelerator-type="${TPU_TYPE}" \
                        --version="${RUNTIME_VERSION}" \
                        --project="${PROJECT}" \
                        2>&1
                }
        }
    
    echo "  TPU created. Waiting for SSH..."
    sleep 30
    
    echo "  Installing dependencies..."
    gcloud compute tpus tpu-vm ssh "${TPU_NAME}" \
        --zone="${ZONE}" \
        --project="${PROJECT}" \
        --command="
            set -e
            echo '=== Installing Python packages ==='
            pip install --upgrade pip
            pip install torch torch_xla[tpu] \
                -f https://storage.googleapis.com/libtpu-releases/index.html \
                -f https://storage.googleapis.com/libtpu-wheels/index.html
            pip install transformers accelerate sentencepiece protobuf
            pip install jax[tpu] flax \
                -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
            echo '=== Installation complete ==='
            python3 -c 'import torch; import torch_xla; print(f\"PyTorch: {torch.__version__}, XLA OK\")'
            python3 -c 'import jax; print(f\"JAX: {jax.__version__}, devices: {jax.devices()}\")'
        "
    
    echo "  ✓ TPU ready for benchmarks"
}

run_benchmark() {
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║  Running Benchmark Suite on TPU v5e-1                       ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    
    # Upload benchmark script
    gcloud compute tpus tpu-vm scp \
        "${SCRIPT_DIR}/benchmark_suite.py" \
        "${TPU_NAME}:~/benchmark_suite.py" \
        --zone="${ZONE}" \
        --project="${PROJECT}"
    
    # Run benchmark (with timeout — 2 hours max)
    gcloud compute tpus tpu-vm ssh "${TPU_NAME}" \
        --zone="${ZONE}" \
        --project="${PROJECT}" \
        --command="
            cd ~ && \
            timeout 7200 python3 benchmark_suite.py 2>&1 | tee benchmark_output.log && \
            echo '=== BENCHMARK COMPLETE ==='
        "
}

download_results() {
    echo "  Downloading results..."
    mkdir -p results/
    
    gcloud compute tpus tpu-vm scp \
        "${TPU_NAME}:~/runux_benchmark_all.json" \
        "results/runux_benchmark_$(date +%Y%m%d_%H%M%S).json" \
        --zone="${ZONE}" \
        --project="${PROJECT}" 2>/dev/null || echo "  No results file found"
    
    gcloud compute tpus tpu-vm scp \
        "${TPU_NAME}:~/benchmark_output.log" \
        "results/benchmark_output_$(date +%Y%m%d_%H%M%S).log" \
        --zone="${ZONE}" \
        --project="${PROJECT}" 2>/dev/null || echo "  No log file found"
    
    echo "  ✓ Results downloaded to results/"
    ls -la results/
}

destroy_tpu() {
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║  TEARING DOWN TPU v5e-1 — Stopping billing                  ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    
    gcloud compute tpus tpu-vm delete "${TPU_NAME}" \
        --zone="${ZONE}" \
        --project="${PROJECT}" \
        --quiet \
        2>&1 || echo "  TPU already deleted or not found"
    
    echo "  ✓ TPU destroyed. No further charges."
}

check_cost() {
    echo "  Checking recent billing..."
    gcloud billing accounts list --project="${PROJECT}" 2>&1 || echo "  Cannot check billing"
    echo ""
    echo "  Estimated cost for this benchmark session:"
    echo "  - TPU v5e-1 spot:   ~\$0.40/hr × 2hr = \$0.80"
    echo "  - TPU v5e-1 preempt: ~\$0.60/hr × 2hr = \$1.20"
    echo "  - TPU v5e-1 on-demand: ~\$1.20/hr × 2hr = \$2.40"
    echo "  - Network egress:    ~\$0.10"
    echo "  - Builder VM (used):  ~\$0.10"
    echo "  ────────────────────────────────"
    echo "  TOTAL ESTIMATE:      \$1.00 — \$2.60"
    echo ""
    echo "  Budget remaining: ~\$${BUDGET_MAX} - \$2.60 = \$$(( BUDGET_MAX - 3 ))"
}

# ============================================================================
# Main
# ============================================================================
case "${1:-help}" in
    create)  create_tpu ;;
    run)     run_benchmark ;;
    results) download_results ;;
    destroy) destroy_tpu ;;
    cost)    check_cost ;;
    all)
        create_tpu
        run_benchmark
        download_results
        destroy_tpu
        check_cost
        ;;
    *)
        echo "Usage: $0 {create|run|results|destroy|cost|all}"
        exit 1
        ;;
esac
