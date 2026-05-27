FROM nvidia/cuda:12.1.1-devel-ubuntu22.04 as builder

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/root/.elan/bin:${PATH}"

# Install core runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl python3-pip python3-dev libgmp-dev && \
    rm -rf /var/lib/apt/lists/*

# Install Elan and Lean 4 toolchain (Standard for formal verification)
RUN curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh -s -- -y --default-toolchain leanprover/lean4:v4.8.0

# Install Python packages
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cu121 && \
    pip3 install --no-cache-dir \
    transformers peft datasets accelerate trl fastapi uvicorn pydantic \
    sympy safetensors sentencepiece huggingface-hub google-genai \
    google-cloud-storage bitsandbytes numpy scipy requests tqdm python-multipart

WORKDIR /app

# Copy SymBrain codebase
COPY real_training_pipeline.py neuro_symbolic_brain.py wars_ci_dfa_bridge.py \
     deepprolog_verifier.py mcts_inference.py inference_server.py dashboard.html ./
RUN mkdir -p pretrained_models

# Expose port
EXPOSE 8080

# Environment variables for PyTorch VRAM optimization
ENV PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
ENV PYTHONUNBUFFERED=1

# Pre-warm models on container spinup then launch server
CMD ["python3", "inference_server.py", "--host", "0.0.0.0", "--port", "8080", "--simulation"]
