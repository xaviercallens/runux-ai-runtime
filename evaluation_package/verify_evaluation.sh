#!/usr/bin/env bash
# ==============================================================================
# RunuX AI Runtime — Master Partnership Evaluation Verification Script
# Copyright (c) 2026 Xavier Callens / Socrate AI Lab. All Rights Reserved.
# ==============================================================================

set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

echo "========================================================================"
echo "          RUNUX AI RUNTIME — PARTNERSHIP EVALUATION SUITE               "
echo "========================================================================"

echo ""
echo ">>> [1/3] Executing Mistral AI Evaluation Harness..."
python3 "${DIR}/harness_mistral.py"

echo ""
echo ">>> [2/3] Executing NVIDIA Evaluation Harness..."
python3 "${DIR}/harness_nvidia.py"

echo ""
echo ">>> [3/3] Executing Google Cloud Evaluation Harness..."
python3 "${DIR}/harness_google.py"

echo "========================================================================"
echo "      ALL PARTNER EVALUATION HARNESSES VERIFIED SUCCESSFULLY!           "
echo "========================================================================"
