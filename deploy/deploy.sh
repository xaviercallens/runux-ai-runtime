#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
# SymBrain v4 — GCP Deployment Script
# ═══════════════════════════════════════════════════════════════════
#
# Deploys SymBrain v4 inference server to GCP Cloud Run.
#
# Usage:
#   ./deploy.sh                  # Deploy Edge-7B (simulation, CPU-only)
#   ./deploy.sh --tier 32B       # Deploy Cloud-32B (L4 GPU, production)
#   ./deploy.sh --tier 32B --production  # Full production with model weights
#   ./deploy.sh --status         # Check deployment status
#
# Prerequisites:
#   - gcloud CLI authenticated with project gen-lang-client-0625573011
#   - Artifact Registry repository 'symbrain-v4' exists
#   - Docker installed for local builds
#
# (c) 2026 Socrate AI Lab, Paris, France
# ═══════════════════════════════════════════════════════════════════

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────

PROJECT_ID="${GCP_PROJECT:-gen-lang-client-0625573011}"
REGION="${GCP_REGION:-europe-west1}"
REGISTRY="${REGION}-docker.pkg.dev/${PROJECT_ID}/symbrain-v4"
SERVICE_BASE="symbrain-v4"
IMAGE_TAG="${IMAGE_TAG:-$(date +%Y%m%d-%H%M%S)}"
TIER="7B"
PRODUCTION=false
SIMULATION=true

# ── Parse Arguments ───────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
    case $1 in
        --tier)
            TIER="$2"
            shift 2
            ;;
        --production)
            PRODUCTION=true
            SIMULATION=false
            shift
            ;;
        --simulation)
            SIMULATION=true
            shift
            ;;
        --tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        --region)
            REGION="$2"
            REGISTRY="${REGION}-docker.pkg.dev/${PROJECT_ID}/symbrain-v4"
            shift 2
            ;;
        --status)
            echo "═══ SymBrain v4 Deployment Status ═══"
            gcloud run services list --project="${PROJECT_ID}" \
                --filter="metadata.labels.component=symbrain-v4" \
                --format="table(name,region,status.url,status.conditions[0].type,metadata.labels.tier)" 2>/dev/null || \
            gcloud run services list --project="${PROJECT_ID}" \
                --filter="name~symbrain-v4" 2>/dev/null
            exit 0
            ;;
        --help|-h)
            head -18 "$0" | tail -14
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ── Tier Configuration ───────────────────────────────────────────

case $TIER in
    7B|edge)
        SERVICE_NAME="${SERVICE_BASE}-edge"
        CPU=2
        MEMORY="4Gi"
        GPU_FLAG=""
        MAX_INSTANCES=4
        MIN_INSTANCES=0
        TIMEOUT=300
        DOCKERFILE="Dockerfile.edge"
        IMAGE_NAME="inference-edge"
        ;;
    32B|cloud32)
        SERVICE_NAME="${SERVICE_BASE}-cloud32"
        CPU=8
        MEMORY="32Gi"
        GPU_FLAG="--gpu=1 --gpu-type=nvidia-l4 --no-gpu-zonal-redundancy"
        MAX_INSTANCES=2
        MIN_INSTANCES=0
        TIMEOUT=900
        DOCKERFILE="Dockerfile.v4"
        IMAGE_NAME="inference"
        ;;
    70B|cloud70)
        echo "⚠️  70B tier requires GCE with 2×A100 GPUs."
        echo "    Use Terraform or manual GCE deployment."
        echo "    Run: terraform apply -var='project_id=${PROJECT_ID}'"
        exit 1
        ;;
    122B|cloud122)
        echo "⚠️  122B tier requires GCE with 4×A100 GPUs."
        echo "    Use Terraform or manual GCE deployment."
        exit 1
        ;;
    *)
        echo "❌ Unknown tier: $TIER (valid: 7B, 32B, 70B, 122B)"
        exit 1
        ;;
esac

echo "═══════════════════════════════════════════════════════════"
echo "  SYMBRAIN v4 — GCP DEPLOYMENT"
echo "═══════════════════════════════════════════════════════════"
echo "  Project:    ${PROJECT_ID}"
echo "  Region:     ${REGION}"
echo "  Service:    ${SERVICE_NAME}"
echo "  Tier:       ${TIER}"
echo "  Mode:       $([ "$PRODUCTION" = true ] && echo "PRODUCTION" || echo "SIMULATION")"
echo "  Image Tag:  ${IMAGE_TAG}"
echo "  CPU/Memory: ${CPU} / ${MEMORY}"
echo "  GPU:        ${GPU_FLAG:-none}"
echo "═══════════════════════════════════════════════════════════"
echo ""

# ── Step 1: Ensure Artifact Registry exists ──────────────────────

echo "▸ Step 1/4: Checking Artifact Registry..."
if ! gcloud artifacts repositories describe symbrain-v4 \
    --project="${PROJECT_ID}" --location="${REGION}" &>/dev/null; then
    echo "  Creating repository 'symbrain-v4'..."
    gcloud artifacts repositories create symbrain-v4 \
        --project="${PROJECT_ID}" \
        --location="${REGION}" \
        --repository-format=docker \
        --description="SymBrain v4 inference server images"
fi
echo "  ✅ Artifact Registry: ${REGISTRY}"

# ── Step 2: Build Docker image ───────────────────────────────────

echo ""
echo "▸ Step 2/4: Building Docker image..."
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

docker build \
    --file="${SCRIPT_DIR}/${DOCKERFILE}" \
    --build-arg="MODEL_TIER=${TIER}" \
    --tag="${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}" \
    --tag="${REGISTRY}/${IMAGE_NAME}:latest" \
    --platform=linux/amd64 \
    "${PROJECT_ROOT}"

echo "  ✅ Image built: ${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"

# ── Step 3: Push to Artifact Registry ────────────────────────────

echo ""
echo "▸ Step 3/4: Pushing to Artifact Registry..."
gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet

docker push "${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"
docker push "${REGISTRY}/${IMAGE_NAME}:latest"

echo "  ✅ Image pushed"

# ── Step 4: Deploy to Cloud Run ──────────────────────────────────

echo ""
echo "▸ Step 4/4: Deploying to Cloud Run..."

# GCP labels must be lowercase. Convert TIER to lowercase for labels.
TIER_LOWER=$(echo "${TIER}" | tr '[:upper:]' '[:lower:]')

# Build the deploy command
DEPLOY_CMD=(
    gcloud run deploy "${SERVICE_NAME}"
    --project="${PROJECT_ID}"
    --region="${REGION}"
    --image="${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"
    --platform=managed
    --port=8080
    --cpu="${CPU}"
    --memory="${MEMORY}"
    --max-instances="${MAX_INSTANCES}"
    --min-instances="${MIN_INSTANCES}"
    --timeout="${TIMEOUT}"
    --concurrency=1
    --no-cpu-throttling
    --execution-environment=gen2
    --allow-unauthenticated
    --set-env-vars="MODEL_TIER=${TIER},SIMULATION_MODE=${SIMULATION},PYTHONUNBUFFERED=1,HF_HOME=/app/model_cache"
    --labels="environment=prod,tier=${TIER_LOWER},component=symbrain-v4"
)

# Add GPU flags if applicable
if [[ -n "$GPU_FLAG" ]]; then
    DEPLOY_CMD+=($GPU_FLAG)
fi

# Execute deployment
"${DEPLOY_CMD[@]}"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  ✅ DEPLOYMENT COMPLETE"
echo "═══════════════════════════════════════════════════════════"

# Get the service URL
URL=$(gcloud run services describe "${SERVICE_NAME}" \
    --project="${PROJECT_ID}" \
    --region="${REGION}" \
    --format="value(status.url)" 2>/dev/null)

echo ""
echo "  Service URL: ${URL}"
echo ""
echo "  Quick test:"
echo "    curl -s ${URL}/v4/health | python3 -m json.tool"
echo ""
echo "    curl -s -X POST ${URL}/v4/solve \\"
echo "      -H 'Content-Type: application/json' \\"
echo "      -d '{\"query\": \"Calculate lim sin(x)/x as x→0\"}' | python3 -m json.tool"
echo ""
echo "═══════════════════════════════════════════════════════════"
