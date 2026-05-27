#!/bin/bash
# SymBrain v2 — Vertex AI Training Job
# Uses pre-built PyTorch container with L4/A100 GPU
# Doesn't require GPUS_ALL_REGIONS quota
set -euo pipefail

PROJECT="gen-lang-client-0625573011"
REGION="us-central1"
GCS_BUCKET="gs://symbrain-v2-models"
TIMESTAMP=$(date +%Y%m%dT%H%M%S)
JOB_NAME="symbrain-v2-train-${TIMESTAMP}"

echo "=== Uploading training code to GCS ==="
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Create a package directory
PACKAGE_DIR="/tmp/symbrain_training_pkg"
rm -rf "$PACKAGE_DIR"
mkdir -p "$PACKAGE_DIR"

# Copy training scripts
cp "$SCRIPT_DIR/real_training_pipeline.py" "$PACKAGE_DIR/"
cp "$SCRIPT_DIR/baseline_eval.py" "$PACKAGE_DIR/"
cp "$SCRIPT_DIR/neuro_symbolic_brain.py" "$PACKAGE_DIR/"
cp "$SCRIPT_DIR/wars_ci_dfa_bridge.py" "$PACKAGE_DIR/"
cp "$SCRIPT_DIR/deepprolog_verifier.py" "$PACKAGE_DIR/"
cp "$SCRIPT_DIR/mcts_inference.py" "$PACKAGE_DIR/"
cp "$SCRIPT_DIR/real_benchmark_eval.py" "$PACKAGE_DIR/"
cp -r "$SCRIPT_DIR/pretrained_models/" "$PACKAGE_DIR/" 2>/dev/null || true

# Create entrypoint
cat > "$PACKAGE_DIR/entrypoint.sh" << 'EOF'
#!/bin/bash
set -euo pipefail
echo "=== SymBrain v2 Training Entrypoint ==="
echo "Device: $(python3 -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")')"
echo "VRAM: $(python3 -c 'import torch; print(f"{torch.cuda.get_device_properties(0).total_mem/1e9:.1f}GB") if torch.cuda.is_available() else print("N/A")')"

pip install -q peft datasets accelerate trl sympy safetensors sentencepiece huggingface-hub google-genai

# Run baseline first
echo "--- Baseline Evaluation ---"
python3 /app/baseline_eval.py --num-seeds 3 2>&1 | tail -20

# Run real training
echo "--- Real Training ---"
python3 /app/real_training_pipeline.py \
    --model Qwen/Qwen2.5-Math-7B-Instruct \
    --lora-r 128 \
    --budget 150 \
    --sft-duration 36000 \
    --hf-repo xaviercallens/symbrain-v2-math \
    --hf-dataset-repo xaviercallens/symbrain-v2-results \
    --gcs-bucket "${GCS_STAGING_BUCKET}" \
    --output-dir /app/output

# Copy results to GCS
gsutil -m cp -r /app/output/ "${GCS_STAGING_BUCKET}/training_output/"
echo "=== Training Complete ==="
EOF
chmod +x "$PACKAGE_DIR/entrypoint.sh"

# Upload to GCS
gsutil -m cp -r "$PACKAGE_DIR/"* "${GCS_BUCKET}/training_code/${TIMESTAMP}/"
echo "✓ Code uploaded to ${GCS_BUCKET}/training_code/${TIMESTAMP}/"

echo ""
echo "=== Submitting Vertex AI Custom Training Job ==="

gcloud ai custom-jobs create \
    --project="${PROJECT}" \
    --region="${REGION}" \
    --display-name="${JOB_NAME}" \
    --worker-pool-spec="\
machine-type=g2-standard-12,\
accelerator-type=NVIDIA_L4,\
accelerator-count=1,\
replica-count=1,\
container-image-uri=us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-4:latest" \
    --args="bash,-c,gsutil -m cp -r ${GCS_BUCKET}/training_code/${TIMESTAMP}/* /app/ && chmod +x /app/entrypoint.sh && /app/entrypoint.sh" \
    --env-vars="GEMINI_API_KEY=${GEMINI_API_KEY},HF_TOKEN=${HF_TOKEN},GCS_STAGING_BUCKET=${GCS_BUCKET}" \
    2>&1

echo ""
echo "=== Monitor job ==="
echo "  gcloud ai custom-jobs list --project=${PROJECT} --region=${REGION}"
echo "  gcloud ai custom-jobs describe ${JOB_NAME} --project=${PROJECT} --region=${REGION}"
