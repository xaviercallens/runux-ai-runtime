FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl && \
    rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir \
    transformers peft datasets accelerate trl \
    sympy safetensors sentencepiece huggingface-hub \
    google-genai google-cloud-storage bitsandbytes \
    numpy scipy requests tqdm

# Copy training code
WORKDIR /app
COPY real_training_pipeline.py baseline_eval.py neuro_symbolic_brain.py \
     wars_ci_dfa_bridge.py deepprolog_verifier.py mcts_inference.py \
     real_benchmark_eval.py inference_server.py ./
COPY pretrained_models/ ./pretrained_models/

# Default: run training
CMD ["python3", "real_training_pipeline.py", \
     "--model", "Qwen/Qwen2.5-Math-7B-Instruct", \
     "--lora-r", "64", "--budget", "150", \
     "--sft-duration", "2400", \
     "--max-seq-len", "1024", "--batch-size", "2", "--grad-accum", "16", \
     "--hf-repo", "xaviercallens/symbrain-v2-math", \
     "--hf-dataset-repo", "xaviercallens/symbrain-v2-results", \
     "--output-dir", "/app/output"]
