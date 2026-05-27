#!/bin/bash
# Copyright (c) 2026 Xavier Callens / Socrate AI Lab. All Rights Reserved.
# SPDX-License-Identifier: LicenseRef-RunuX-Commercial
#
# TPU v5e Deployment Script — SymBrain v2 Phase 3
# =================================================
# Provisions, trains, uploads, and tears down.
# Budget-guarded with auto-teardown on exit.
#
# Usage:
#   ./deploy_tpu_training.sh              # default $100 budget
#   ./deploy_tpu_training.sh --spot       # use preemptible
#   ./deploy_tpu_training.sh --budget 50  # $50 budget

set -euo pipefail

# ── Colors ──
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# ═══════════════════════════════════════════════════════════════
# §1  CONFIGURATION (from env vars or defaults)
# ═══════════════════════════════════════════════════════════════

PROJECT_ID="${GCP_PROJECT:-runux-ai}"
ZONE="${TPU_ZONE:-us-central2-b}"
TPU_NAME="${TPU_NAME:-symbrain-v2-training}"
TPU_TYPE="${TPU_TYPE:-v5litepod-4}"
RUNTIME_VERSION="v2-alpha-tpuv5-lite"
GCS_BUCKET="${GCS_BUCKET:-gs://runux-ai-models}"
BUDGET=100
SPOT=false
TIMESTAMP=$(date +%Y%m%dT%H%M%S)
LOG_FILE="tpu_training_${TIMESTAMP}.log"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --spot) SPOT=true; shift ;;
        --budget) BUDGET="$2"; shift 2 ;;
        --tpu-type) TPU_TYPE="$2"; shift 2 ;;
        --zone) ZONE="$2"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

# SECURITY: Verify API keys are set (never hardcode!)
if [ -z "${GEMINI_API_KEY:-}" ]; then
    echo -e "${RED}⛔ GEMINI_API_KEY not set. Export it first.${NC}"
    exit 1
fi

echo -e "\n${CYAN}${BOLD}════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}${BOLD}  SymBrain v2 — TPU Training Deployment${NC}"
echo -e "${CYAN}${BOLD}  Project: ${PROJECT_ID} | Zone: ${ZONE}${NC}"
echo -e "${CYAN}${BOLD}  TPU: ${TPU_TYPE} | Budget: \$${BUDGET}${NC}"
echo -e "${CYAN}${BOLD}  Spot: ${SPOT} | Log: ${LOG_FILE}${NC}"
echo -e "${CYAN}${BOLD}════════════════════════════════════════════════════════${NC}\n"

# ═══════════════════════════════════════════════════════════════
# §2  AUTO-TEARDOWN ON EXIT
# ═══════════════════════════════════════════════════════════════

cleanup() {
    echo -e "\n${YELLOW}${BOLD}🧹 Auto-teardown: Deleting TPU ${TPU_NAME}...${NC}"
    gcloud compute tpus tpu-vm delete "${TPU_NAME}" \
        --project="${PROJECT_ID}" --zone="${ZONE}" --quiet 2>/dev/null || true
    echo -e "${GREEN}✓ TPU deleted. Final log: ${LOG_FILE}${NC}"
}
trap cleanup EXIT

# ═══════════════════════════════════════════════════════════════
# §3  TPU PROVISIONING
# ═══════════════════════════════════════════════════════════════

echo -e "${BLUE}${BOLD}[1/5] Provisioning TPU ${TPU_TYPE}...${NC}" | tee -a "${LOG_FILE}"

SPOT_FLAG=""
if [ "$SPOT" = true ]; then
    SPOT_FLAG="--preemptible"
    echo -e "  ${YELLOW}Using preemptible/spot instance${NC}"
fi

gcloud compute tpus tpu-vm create "${TPU_NAME}" \
    --project="${PROJECT_ID}" \
    --zone="${ZONE}" \
    --accelerator-type="${TPU_TYPE}" \
    --version="${RUNTIME_VERSION}" \
    ${SPOT_FLAG} \
    2>&1 | tee -a "${LOG_FILE}"

echo -e "  ${GREEN}✓ TPU created${NC}" | tee -a "${LOG_FILE}"

# ═══════════════════════════════════════════════════════════════
# §4  INSTALL DEPENDENCIES
# ═══════════════════════════════════════════════════════════════

echo -e "\n${BLUE}${BOLD}[2/5] Installing dependencies...${NC}" | tee -a "${LOG_FILE}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

gcloud compute tpus tpu-vm ssh "${TPU_NAME}" \
    --project="${PROJECT_ID}" --zone="${ZONE}" \
    --command="pip install -r /dev/stdin < <(cat)" \
    < "${SCRIPT_DIR}/tpu_requirements.txt" \
    2>&1 | tee -a "${LOG_FILE}"

# Copy training scripts
gcloud compute tpus tpu-vm scp \
    "${SCRIPT_DIR}"/*.py "${TPU_NAME}":~/training/ \
    --project="${PROJECT_ID}" --zone="${ZONE}" \
    2>&1 | tee -a "${LOG_FILE}"

echo -e "  ${GREEN}✓ Dependencies installed, scripts deployed${NC}" | tee -a "${LOG_FILE}"

# ═══════════════════════════════════════════════════════════════
# §5  RUN TRAINING
# ═══════════════════════════════════════════════════════════════

echo -e "\n${BLUE}${BOLD}[3/5] Running baseline evaluation...${NC}" | tee -a "${LOG_FILE}"

gcloud compute tpus tpu-vm ssh "${TPU_NAME}" \
    --project="${PROJECT_ID}" --zone="${ZONE}" \
    --command="cd ~/training && \
        export GEMINI_API_KEY='${GEMINI_API_KEY}' && \
        export MISTRAL_API_KEY='${MISTRAL_API_KEY:-}' && \
        python3 baseline_eval.py --num-seeds 3" \
    2>&1 | tee -a "${LOG_FILE}"

echo -e "\n${BLUE}${BOLD}[4/5] Running Phase 3 full training...${NC}" | tee -a "${LOG_FILE}"

START_EPOCH=$(date +%s)

gcloud compute tpus tpu-vm ssh "${TPU_NAME}" \
    --project="${PROJECT_ID}" --zone="${ZONE}" \
    --command="cd ~/training && \
        export GEMINI_API_KEY='${GEMINI_API_KEY}' && \
        export MISTRAL_API_KEY='${MISTRAL_API_KEY:-}' && \
        python3 phase3_full_training.py \
            --budget ${BUDGET} \
            --gcs-bucket ${GCS_BUCKET} \
            --output-dir ./phase3_output" \
    2>&1 | tee -a "${LOG_FILE}"

END_EPOCH=$(date +%s)
WALL_HOURS=$(echo "scale=2; ($END_EPOCH - $START_EPOCH) / 3600" | bc)
EST_COST=$(echo "scale=2; $WALL_HOURS * 1.20 * 4" | bc)

echo -e "  ${GREEN}✓ Training complete: ${WALL_HOURS}h, est. \$${EST_COST}${NC}" | tee -a "${LOG_FILE}"

# ═══════════════════════════════════════════════════════════════
# §6  MODEL UPLOAD
# ═══════════════════════════════════════════════════════════════

echo -e "\n${BLUE}${BOLD}[5/5] Uploading models...${NC}" | tee -a "${LOG_FILE}"

# Copy results back
gcloud compute tpus tpu-vm scp --recurse \
    "${TPU_NAME}":~/training/phase3_output/ ./phase3_results/ \
    --project="${PROJECT_ID}" --zone="${ZONE}" \
    2>&1 | tee -a "${LOG_FILE}"

# Upload to GCS
gsutil -m cp -r ./phase3_results/ "${GCS_BUCKET}/symbrain-v2/${TIMESTAMP}/" \
    2>&1 | tee -a "${LOG_FILE}"

# Push to HuggingFace if token available
if [ -n "${HF_TOKEN:-}" ]; then
    echo -e "  Pushing to HuggingFace..."
    python3 -c "
from huggingface_hub import HfApi
api = HfApi(token='${HF_TOKEN}')
api.upload_folder(
    folder_path='./phase3_results/final_model',
    repo_id='xaviercallens/symbrain-v2-math',
    repo_type='model',
)
print('  ✓ Pushed to HuggingFace')
" 2>&1 | tee -a "${LOG_FILE}"
fi

echo -e "  ${GREEN}✓ Models uploaded${NC}" | tee -a "${LOG_FILE}"

# ═══════════════════════════════════════════════════════════════
# §7  FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════

echo -e "\n${CYAN}${BOLD}════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}${BOLD}  Training Complete!${NC}"
echo -e "${CYAN}${BOLD}════════════════════════════════════════════════════════${NC}"
echo -e "  Wall time:    ${WALL_HOURS}h"
echo -e "  Est. cost:    \$${EST_COST} / \$${BUDGET}"
echo -e "  GCS:          ${GCS_BUCKET}/symbrain-v2/${TIMESTAMP}/"
echo -e "  Log:          ${LOG_FILE}"
echo -e "  TPU:          ${TPU_NAME} (will be deleted on exit)"
echo -e "${CYAN}${BOLD}════════════════════════════════════════════════════════${NC}\n"

# Teardown happens via trap EXIT
