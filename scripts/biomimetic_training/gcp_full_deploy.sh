#!/bin/bash
# Copyright (c) 2026 Xavier Callens / Socrate AI Lab. All Rights Reserved.
# SPDX-License-Identifier: LicenseRef-RunuX-Commercial
#
# SymBrain v2 — GCP Spot Training + Inference Deployment
# ========================================================
# Provisions spot GPU, trains, serializes, archives, deploys inference.
# Total budget: <$150 (training ~$50-80, inference ~$5-10)
#
# Usage:
#   ./gcp_full_deploy.sh                  # Full pipeline
#   ./gcp_full_deploy.sh --train-only     # Train + serialize only
#   ./gcp_full_deploy.sh --infer-only     # Deploy inference only

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# ═══════════════════════════════════════════════════════════════
# §1  CONFIGURATION — ALL FROM ENVIRONMENT
# ═══════════════════════════════════════════════════════════════

PROJECT="${GCLOUD_PROJECT:-gen-lang-client-0625573011}"
ZONE="${TRAIN_ZONE:-us-central1-a}"
INFER_ZONE="${INFER_ZONE:-us-central1-a}"
TRAIN_INSTANCE="symbrain-v2-train-$(date +%s)"
INFER_INSTANCE="symbrain-v2-infer-$(date +%s)"
GCS_BUCKET="${GCS_BUCKET:-gs://symbrain-v2-models}"
TIMESTAMP=$(date +%Y%m%dT%H%M%S)
BUDGET=150
TRAIN_ONLY=false
INFER_ONLY=false
SKIP_CLEANUP=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --train-only) TRAIN_ONLY=true; shift ;;
        --infer-only) INFER_ONLY=true; shift ;;
        --budget) BUDGET="$2"; shift 2 ;;
        --zone) ZONE="$2"; shift 2 ;;
        --skip-cleanup) SKIP_CLEANUP=true; shift ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

# Validate environment
for key in GEMINI_API_KEY HF_TOKEN; do
    val="${!key:-}"
    if [ -z "$val" ]; then
        echo -e "${RED}⛔ $key not set. Export it before running.${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ $key loaded from environment${NC}"
done

MISTRAL_KEY="${MISTRAL_API_KEY:-}"

echo -e "\n${CYAN}${BOLD}════════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}${BOLD}  SymBrain v2 — GCP Full Deployment Pipeline${NC}"
echo -e "${CYAN}${BOLD}  Project: ${PROJECT} | Budget: \$${BUDGET}${NC}"
echo -e "${CYAN}${BOLD}  Training: Spot A100/L4 | Inference: L4 Spot${NC}"
echo -e "${CYAN}${BOLD}════════════════════════════════════════════════════════════════${NC}\n"

# ═══════════════════════════════════════════════════════════════
# §2  GCS BUCKET SETUP
# ═══════════════════════════════════════════════════════════════

echo -e "${BLUE}${BOLD}[0/6] Setting up GCS bucket...${NC}"
gsutil ls "${GCS_BUCKET}" 2>/dev/null || \
    gsutil mb -p "${PROJECT}" -l us-central1 -b on "${GCS_BUCKET}" 2>/dev/null || true

# Make bucket private (uniform bucket-level access)
gsutil uniformbucketlevelaccess set on "${GCS_BUCKET}" 2>/dev/null || true
echo -e "  ${GREEN}✓ GCS bucket ready: ${GCS_BUCKET}${NC}"

# ═══════════════════════════════════════════════════════════════
# §3  TRAINING INSTANCE PROVISIONING
# ═══════════════════════════════════════════════════════════════

STARTUP_SCRIPT=$(cat <<'STARTUP_EOF'
#!/bin/bash
set -euo pipefail

# Install dependencies
pip install --upgrade pip
pip install torch transformers peft datasets accelerate trl \
    sympy safetensors sentencepiece huggingface-hub google-genai \
    requests numpy scipy scikit-learn protobuf tqdm vllm

# Install NVIDIA drivers if not present
if ! nvidia-smi > /dev/null 2>&1; then
    apt-get update && apt-get install -y nvidia-driver-550
fi

echo "STARTUP_COMPLETE" > /tmp/startup_done
STARTUP_EOF
)

if [ "$INFER_ONLY" = false ]; then

echo -e "\n${BLUE}${BOLD}[1/6] Provisioning SPOT training instance...${NC}"

# Try A100 first, fall back to L4, then T4
GPU_TYPE=""
MACHINE_TYPE=""
for gpu in nvidia-tesla-a100 nvidia-l4; do
    echo -e "  Trying ${gpu}..."
    if [ "$gpu" = "nvidia-tesla-a100" ]; then
        MACHINE_TYPE="a2-highgpu-1g"
    else
        MACHINE_TYPE="g2-standard-8"
    fi

    if gcloud compute instances create "${TRAIN_INSTANCE}" \
        --project="${PROJECT}" \
        --zone="${ZONE}" \
        --machine-type="${MACHINE_TYPE}" \
        --accelerator="type=${gpu},count=1" \
        --provisioning-model=SPOT \
        --instance-termination-action=STOP \
        --boot-disk-size=200GB \
        --boot-disk-type=pd-ssd \
        --image-family=pytorch-latest-gpu \
        --image-project=deeplearning-platform-release \
        --maintenance-policy=TERMINATE \
        --scopes=cloud-platform \
        --metadata="startup-script=${STARTUP_SCRIPT}" \
        2>&1; then
        GPU_TYPE="$gpu"
        echo -e "  ${GREEN}✓ Created ${TRAIN_INSTANCE} with ${gpu} (SPOT)${NC}"
        break
    else
        echo -e "  ${YELLOW}⚠ ${gpu} not available, trying next...${NC}"
    fi
done

if [ -z "$GPU_TYPE" ]; then
    echo -e "${RED}⛔ No GPU available. Request quota increase.${NC}"
    exit 1
fi

# Auto-cleanup on exit
cleanup_train() {
    if [ "$SKIP_CLEANUP" = false ]; then
        echo -e "\n${YELLOW}🧹 Deleting training instance ${TRAIN_INSTANCE}...${NC}"
        gcloud compute instances delete "${TRAIN_INSTANCE}" \
            --project="${PROJECT}" --zone="${ZONE}" --quiet 2>/dev/null || true
    fi
}
trap cleanup_train EXIT

# ═══════════════════════════════════════════════════════════════
# §4  WAIT FOR STARTUP + UPLOAD TRAINING CODE
# ═══════════════════════════════════════════════════════════════

echo -e "\n${BLUE}${BOLD}[2/6] Waiting for instance startup...${NC}"
for i in $(seq 1 60); do
    if gcloud compute ssh "${TRAIN_INSTANCE}" \
        --project="${PROJECT}" --zone="${ZONE}" \
        --command="test -f /tmp/startup_done && echo READY" 2>/dev/null | grep -q READY; then
        echo -e "  ${GREEN}✓ Instance ready after ~${i}0 seconds${NC}"
        break
    fi
    echo -e "  Waiting... (${i}/60)"
    sleep 10
done

echo -e "\n${BLUE}${BOLD}[3/6] Uploading training scripts...${NC}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

gcloud compute scp --recurse \
    "${SCRIPT_DIR}/real_training_pipeline.py" \
    "${SCRIPT_DIR}/baseline_eval.py" \
    "${SCRIPT_DIR}/phase3_full_training.py" \
    "${SCRIPT_DIR}/neuro_symbolic_brain.py" \
    "${SCRIPT_DIR}/wars_ci_dfa_bridge.py" \
    "${SCRIPT_DIR}/deepprolog_verifier.py" \
    "${SCRIPT_DIR}/mcts_inference.py" \
    "${SCRIPT_DIR}/real_benchmark_eval.py" \
    "${SCRIPT_DIR}/pretrained_models/" \
    "${TRAIN_INSTANCE}":~/training/ \
    --project="${PROJECT}" --zone="${ZONE}" 2>&1

echo -e "  ${GREEN}✓ Scripts uploaded${NC}"

# ═══════════════════════════════════════════════════════════════
# §5  RUN REAL TRAINING
# ═══════════════════════════════════════════════════════════════

echo -e "\n${BLUE}${BOLD}[4/6] Running REAL training...${NC}"
TRAIN_START=$(date +%s)

gcloud compute ssh "${TRAIN_INSTANCE}" \
    --project="${PROJECT}" --zone="${ZONE}" \
    --command="cd ~/training && \
        export GEMINI_API_KEY='${GEMINI_API_KEY}' && \
        export MISTRAL_API_KEY='${MISTRAL_KEY}' && \
        export HF_TOKEN='${HF_TOKEN}' && \
        nvidia-smi && \
        echo '--- Baseline Evaluation ---' && \
        python3 baseline_eval.py --num-seeds 3 2>&1 | tail -30 && \
        echo '--- Full Training Pipeline ---' && \
        python3 real_training_pipeline.py \
            --model Qwen/Qwen2.5-Math-7B-Instruct \
            --lora-r 128 \
            --budget ${BUDGET} \
            --sft-duration 36000 \
            --hf-repo xaviercallens/symbrain-v2-math \
            --hf-dataset-repo xaviercallens/symbrain-v2-results \
            --gcs-bucket ${GCS_BUCKET} \
            --output-dir ./output 2>&1" | tee "train_log_${TIMESTAMP}.log"

TRAIN_END=$(date +%s)
TRAIN_HOURS=$(echo "scale=2; ($TRAIN_END - $TRAIN_START) / 3600" | bc)
echo -e "  ${GREEN}✓ Training: ${TRAIN_HOURS} hours${NC}"

# ═══════════════════════════════════════════════════════════════
# §6  DOWNLOAD + ARCHIVE MODELS
# ═══════════════════════════════════════════════════════════════

echo -e "\n${BLUE}${BOLD}[5/6] Downloading & archiving models...${NC}"

# Download from instance
mkdir -p "./trained_models_${TIMESTAMP}"
gcloud compute scp --recurse \
    "${TRAIN_INSTANCE}":~/training/output/ \
    "./trained_models_${TIMESTAMP}/" \
    --project="${PROJECT}" --zone="${ZONE}" 2>&1

# Upload to GCS (private)
gsutil -m cp -r "./trained_models_${TIMESTAMP}/" \
    "${GCS_BUCKET}/training_runs/${TIMESTAMP}/" 2>&1

# Archive to Remote SSD
SSD_DEST="/Volumes/MacCleanerStorage/symbrain_v2_models/${TIMESTAMP}"
if [ -d "/Volumes/MacCleanerStorage" ]; then
    mkdir -p "${SSD_DEST}"
    cp -r "./trained_models_${TIMESTAMP}/"* "${SSD_DEST}/"
    echo -e "  ${GREEN}✓ Archived to SSD: ${SSD_DEST}${NC}"
fi

echo -e "  ${GREEN}✓ Models archived to GCS + SSD${NC}"

fi  # end TRAIN_ONLY check

# ═══════════════════════════════════════════════════════════════
# §7  INFERENCE SERVER DEPLOYMENT (L4 Spot)
# ═══════════════════════════════════════════════════════════════

if [ "$TRAIN_ONLY" = false ]; then

echo -e "\n${BLUE}${BOLD}[6/6] Deploying inference server (L4 Spot)...${NC}"

INFER_STARTUP=$(cat <<'INFER_EOF'
#!/bin/bash
set -euo pipefail
pip install --upgrade pip
pip install vllm transformers peft safetensors sentencepiece \
    uvicorn fastapi huggingface-hub
echo "INFER_READY" > /tmp/startup_done
INFER_EOF
)

gcloud compute instances create "${INFER_INSTANCE}" \
    --project="${PROJECT}" \
    --zone="${INFER_ZONE}" \
    --machine-type="g2-standard-4" \
    --accelerator="type=nvidia-l4,count=1" \
    --provisioning-model=SPOT \
    --instance-termination-action=STOP \
    --boot-disk-size=100GB \
    --boot-disk-type=pd-ssd \
    --image-family=pytorch-latest-gpu \
    --image-project=deeplearning-platform-release \
    --maintenance-policy=TERMINATE \
    --scopes=cloud-platform \
    --metadata="startup-script=${INFER_STARTUP}" \
    2>&1

echo -e "  ${GREEN}✓ Inference instance created: ${INFER_INSTANCE}${NC}"

# Wait for startup
for i in $(seq 1 30); do
    if gcloud compute ssh "${INFER_INSTANCE}" \
        --project="${PROJECT}" --zone="${INFER_ZONE}" \
        --command="test -f /tmp/startup_done && echo READY" 2>/dev/null | grep -q READY; then
        break
    fi
    sleep 10
done

# Upload inference server script
gcloud compute scp \
    "${SCRIPT_DIR}/inference_server.py" \
    "${INFER_INSTANCE}":~/inference/ \
    --project="${PROJECT}" --zone="${INFER_ZONE}" 2>&1

# Download trained model from GCS to inference instance
gcloud compute ssh "${INFER_INSTANCE}" \
    --project="${PROJECT}" --zone="${INFER_ZONE}" \
    --command="gsutil -m cp -r ${GCS_BUCKET}/training_runs/${TIMESTAMP}/lora_adapters/ ~/inference/model/" 2>&1

# Start inference server
gcloud compute ssh "${INFER_INSTANCE}" \
    --project="${PROJECT}" --zone="${INFER_ZONE}" \
    --command="cd ~/inference && \
        export HF_TOKEN='${HF_TOKEN}' && \
        nohup python3 inference_server.py --port 8080 > server.log 2>&1 &" 2>&1

# Get external IP
INFER_IP=$(gcloud compute instances describe "${INFER_INSTANCE}" \
    --project="${PROJECT}" --zone="${INFER_ZONE}" \
    --format='get(networkInterfaces[0].accessConfigs[0].natIP)' 2>/dev/null)

echo -e "  ${GREEN}✓ Inference server: http://${INFER_IP}:8080${NC}"
echo -e "  ${GREEN}  API: http://${INFER_IP}:8080/v1/solve${NC}"
echo -e "  ${GREEN}  Health: http://${INFER_IP}:8080/health${NC}"

fi

# ═══════════════════════════════════════════════════════════════
# §8  FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════

echo -e "\n${CYAN}${BOLD}════════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}${BOLD}  SymBrain v2 Deployment Complete!${NC}"
echo -e "${CYAN}${BOLD}════════════════════════════════════════════════════════════════${NC}"
echo -e "  GCS:          ${GCS_BUCKET}/training_runs/${TIMESTAMP}/"
if [ -d "/Volumes/MacCleanerStorage" ]; then
    echo -e "  SSD:          ${SSD_DEST}"
fi
if [ "$TRAIN_ONLY" = false ]; then
    echo -e "  Inference:    http://${INFER_IP:-N/A}:8080"
fi
echo -e "  Log:          train_log_${TIMESTAMP}.log"
echo -e "${CYAN}${BOLD}════════════════════════════════════════════════════════════════${NC}"
echo -e ""
echo -e "  ${YELLOW}To tear down inference: gcloud compute instances delete ${INFER_INSTANCE} --zone=${INFER_ZONE} --quiet${NC}"
echo -e ""
