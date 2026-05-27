#!/bin/bash
# Copyright (c) 2026 Xavier Callens / Socrate AI Lab. All Rights Reserved.
# SPDX-License-Identifier: LicenseRef-RunuX-Commercial
#
# Neuro-Symbolic Brain — GCP TPU v5e-8 Deployment Script
# =======================================================

set -euo pipefail

# ─── Configuration ───
TPU_NAME="wars-neurosym-brain"
ZONE="us-central1-a"
ACCEL_TYPE="v5e-8"
TPU_VERSION="tpu-ubuntu2204-base"
PROJECT=$(gcloud config get-value project 2>/dev/null || echo "")
USE_SPOT="--spot"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
BOLD='\033[1m'
NC='\033[0m'

echo -e "${CYAN}${BOLD}======================================================================${NC}"
echo -e "${CYAN}${BOLD}   RunuX AI — Neuro-Symbolic Brain TPU Deployment${NC}"
echo -e "${CYAN}${BOLD}   Target: Cloud TPU ${ACCEL_TYPE} in ${ZONE}${NC}"
echo -e "${CYAN}${BOLD}======================================================================${NC}"
echo ""

# ─── Step 1: Create TPU VM ───
echo -e "${BOLD}[1/7] Creating TPU VM: ${TPU_NAME} (${ACCEL_TYPE}, spot)...${NC}"
gcloud compute tpus tpu-vm create "${TPU_NAME}" \
    --zone="${ZONE}" \
    --accelerator-type="${ACCEL_TYPE}" \
    --version="${TPU_VERSION}" \
    ${USE_SPOT} \
    --quiet 2>&1 || {
        echo -e "${YELLOW}⚠ TPU creation failed. Checking if it already exists...${NC}"
        gcloud compute tpus tpu-vm describe "${TPU_NAME}" --zone="${ZONE}" 2>/dev/null || {
            echo -e "${RED}❌ Cannot create or find TPU VM. Exiting.${NC}"
            exit 1
        }
    }
echo -e "${GREEN}✅ TPU VM created/found successfully.${NC}\n"

# ─── Step 2: Install Dependencies ───
echo -e "${BOLD}[2/7] Installing Python dependencies on TPU VM...${NC}"
gcloud compute tpus tpu-vm ssh "${TPU_NAME}" --zone="${ZONE}" --command="
    pip install --quiet --upgrade pip
    pip install --quiet torch torch_xla transformers datasets accelerate peft trl
    pip install --quiet numpy scipy scikit-learn matplotlib
    echo '✅ Dependencies installed.'
" 2>&1
echo -e "${GREEN}✅ Dependencies installed.${NC}\n"

# ─── Step 3: Upload Training Scripts ───
echo -e "${BOLD}[3/7] Uploading training scripts to TPU VM...${NC}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

gcloud compute tpus tpu-vm scp \
    "${SCRIPT_DIR}/wars_ci_dfa_bridge.py" \
    "${SCRIPT_DIR}/neuro_symbolic_brain.py" \
    "${SCRIPT_DIR}/benchmark_math.py" \
    "${TPU_NAME}":~/ \
    --zone="${ZONE}" 2>&1
echo -e "${GREEN}✅ Scripts uploaded.${NC}\n"

# ─── Step 4: Configure TPU Environment ───
echo -e "${BOLD}[4/7] Configuring TPU environment variables...${NC}"
gcloud compute tpus tpu-vm ssh "${TPU_NAME}" --zone="${ZONE}" --command="
    export PJRT_DEVICE=TPU
    export XLA_USE_BF16=1
    export XLA_TENSOR_ALLOCATOR_MAXSIZE=100000000
    echo 'export PJRT_DEVICE=TPU' >> ~/.bashrc
    echo 'export XLA_USE_BF16=1' >> ~/.bashrc
    echo '✅ TPU environment configured.'
" 2>&1
echo -e "${GREEN}✅ Environment configured.${NC}\n"

# ─── Step 5: Execute Training ───
echo -e "${BOLD}[5/7] Launching Neuro-Symbolic Brain training (>10 min)...${NC}"
echo -e "  Left Hemisphere:  Qwen/Qwen2.5-Math-7B-Instruct"
echo -e "  Right Hemisphere: mistralai/Ministral-8B-Instruct-2410"
echo -e "  PFC Bridge:       WARS-CI-DFA v2 (proj_rank=256)"
echo -e "  Training Time:    ~37 min (Phase 1: 5min + Phase 2: 10min + Phase 3: eval)"
echo ""

TRAINING_START=$(date +%s)
gcloud compute tpus tpu-vm ssh "${TPU_NAME}" --zone="${ZONE}" --command="
    cd ~
    export PJRT_DEVICE=TPU
    export XLA_USE_BF16=1
    python3 neuro_symbolic_brain.py \
        --left-model Qwen/Qwen2.5-Math-7B-Instruct \
        --right-model mistralai/Ministral-8B-Instruct-2410 \
        --projection-rank 256 \
        --batch-size 4 \
        --max-seq-len 512 \
        --learning-rate 2e-5 \
        --warmup-duration 300 \
        --co-inference-duration 600 \
        --output-file neurosymbolic_results.json
" 2>&1
TRAINING_END=$(date +%s)
TRAINING_ELAPSED=$((TRAINING_END - TRAINING_START))
echo -e "${GREEN}✅ Training complete in ${TRAINING_ELAPSED}s.${NC}\n"

# ─── Step 6: Download Results ───
echo -e "${BOLD}[6/7] Downloading results from TPU VM...${NC}"
gcloud compute tpus tpu-vm scp \
    "${TPU_NAME}":~/neurosymbolic_results.json \
    "${SCRIPT_DIR}/neurosymbolic_results.json" \
    --zone="${ZONE}" 2>&1
echo -e "${GREEN}✅ Results downloaded to ${SCRIPT_DIR}/neurosymbolic_results.json${NC}\n"

# ─── Step 7: Teardown TPU VM ───
echo -e "${BOLD}[7/7] Deleting TPU VM to prevent cost leakage...${NC}"
gcloud compute tpus tpu-vm delete "${TPU_NAME}" \
    --zone="${ZONE}" \
    --quiet 2>&1
echo -e "${GREEN}✅ TPU VM deleted. Zero active billing.${NC}\n"

# ─── Summary ───
echo -e "${CYAN}${BOLD}======================================================================${NC}"
echo -e "${CYAN}${BOLD}  🎉 Neuro-Symbolic Brain Deployment Complete!${NC}"
echo -e "${CYAN}${BOLD}======================================================================${NC}"
echo -e "  Training Duration: ${TRAINING_ELAPSED}s"
echo -e "  Results File:      ${SCRIPT_DIR}/neurosymbolic_results.json"
echo -e "  TPU Status:        DELETED (no cost leakage)"
echo -e "  Estimated Cost:    ~\$$(echo "scale=2; ${TRAINING_ELAPSED} * 9.60 / 3600" | bc) (on-demand)"
echo -e "${CYAN}${BOLD}======================================================================${NC}"
