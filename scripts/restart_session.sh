#!/usr/bin/env bash
# ==============================================================================
# RunuX AI Runtime — Quickstart & Session Restart Verification Script
#
# Copyright (c) 2026 Xavier Callens / Socrate AI Lab. All rights reserved.
# ==============================================================================

set -e

# Resolve repository root robustly
if git rev-parse --show-toplevel &>/dev/null; then
    REPO_ROOT="$(git rev-parse --show-toplevel)"
else
    SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
    REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi

echo "================================================================================"
echo "  🚀 RunuX AI Runtime — Automated Session Recovery & Health Verification"
echo "  Timestamp: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "  Repo Root: ${REPO_ROOT}"
echo "================================================================================"

# 1. Environment & Paths
echo ""
echo "[1/6] Checking Python Environment & Virtualenv..."
VENV_PYTHON="/home/callensxavier_gmail_com/venv/bin/python"
if [ -f "${VENV_PYTHON}" ]; then
    PY_VER=$("${VENV_PYTHON}" --version)
    echo "  ✓ Python executable found: ${VENV_PYTHON} (${PY_VER})"
else
    echo "  ⚠️ Virtualenv python not found at ${VENV_PYTHON}, falling back to system python3"
    VENV_PYTHON="$(which python3)"
fi

# 2. Local GPU Telemetry & Health
echo ""
echo "[2/6] Checking NVIDIA Tesla T4 GPU Health & Thermals..."
if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -n1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | head -n1)
    GPU_TEMP=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | head -n1)
    GPU_POWER=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits | head -n1)
    GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -n1)
    echo "  ✓ Accelerator: ${GPU_NAME}"
    echo "  ✓ VRAM Usage : ${GPU_MEM} MiB"
    echo "  ✓ Temperature: ${GPU_TEMP}°C | Power: ${GPU_POWER}W | Utilization: ${GPU_UTIL}%"
    if [ "${GPU_TEMP}" -lt 80 ]; then
        echo "  ✓ Thermal State: Healthy (well below 85°C throttle limit)"
    else
        echo "  ⚠️ Warning: GPU temperature elevated (${GPU_TEMP}°C)"
    fi
else
    echo "  ℹ️ nvidia-smi not available."
fi

# 3. Cloud Resources & Spend Safeguard
echo ""
echo "[3/6] Checking Google Cloud TPU / Compute Resource Spend Safeguard..."
if command -v gcloud &>/dev/null; then
    ACTIVE_TPUS=$(gcloud compute tpus tpu-vm list --zone=us-central1-a --format="value(name)" 2>/dev/null | wc -l || echo "0")
    if [ "${ACTIVE_TPUS}" -eq "0" ]; then
        echo "  ✓ Cloud TPUs: 0 active instances (Clean, 0 spend leak)"
    else
        echo "  ⚠️ Active Cloud TPUs detected: ${ACTIVE_TPUS}"
    fi
else
    echo "  ℹ️ gcloud CLI not in path, skipping TPU cloud query."
fi

# 4. Check Tokens & Secrets Configuration
echo ""
echo "[4/6] Checking Configured API Tokens & Secrets..."
# Source .env if present
if [ -f "${REPO_ROOT}/.env" ]; then
    set -a
    source "${REPO_ROOT}/.env"
    set +a
fi

if [ -n "${ZENODO_TOKEN:-${ZENODO_ACCESS_TOKEN}}" ]; then
    echo "  ✓ ZENODO_TOKEN: Configured (ready for publication)"
else
    echo "  ℹ️ ZENODO_TOKEN not detected in environment."
fi

if command -v gh &>/dev/null; then
    GH_STATUS=$(gh auth status 2>&1 | head -n2 | tr '\n' ' ')
    echo "  ✓ GitHub Auth : ${GH_STATUS}"
fi

# 5. Core Test Suite Verification
echo ""
echo "[5/6] Executing Fast Sanity Pytest Suite (29 Tests)..."
cd "${REPO_ROOT}"
PYTHONPATH=. "${VENV_PYTHON}" -m pytest tests/ -q --disable-warnings

# 6. Key Deliverables & Next Steps
echo ""
echo "[6/6] RunuX Assets & Quick Commands:"
echo "  • Open-Weights Mistral 7B Model : /home/callensxavier_gmail_com/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
echo "  • Compiled Scientific Paper PDF: ${REPO_ROOT}/papers/runux_scientific_proof_paper.pdf"
echo "  • INPI Patent Dossier (French) : ${REPO_ROOT}/legal/patents/INPI_DEMANDE_BREVET_PROVISOIRE_RUNUX.md"
echo "  • Zenodo Open Science Archive  : ${REPO_ROOT}/public_release/zenodo_bundle/zenodo_open_science_bundle.tar.gz"
echo "  • Run Mistral Benchmark Script : ${VENV_PYTHON} ${REPO_ROOT}/scripts/benchmark_mistral_runux_gains.py"
echo ""
echo "================================================================================"
echo "  ✅ RunuX Environment is 100% Ready for Work!"
echo "================================================================================"
