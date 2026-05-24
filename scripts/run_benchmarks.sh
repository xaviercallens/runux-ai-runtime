#!/usr/bin/env bash
# Copyright (c) 2026 Xavier Callens / Socrate AI. All Rights Reserved.
# SPDX-License-Identifier: LicenseRef-RunuX-Commercial
# 
# RunuX Benchmark Execution Suite
# Automatically detects host architecture and executes the optimal validation pathway
# for TPU v5e (GCP) or SpacemiT K1/K3 (RISC-V).

set -e

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║                  RunuX AI Engine - Benchmark Suite                   ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"

# 1. Detect Host Architecture
ARCH=$(uname -m)
echo "[INFO] Detected Host Architecture: ${ARCH}"

# Prepare Rust environment if not present
if ! command -v cargo &> /dev/null; then
    echo "[WARN] cargo could not be found. Please install Rust: https://rustup.rs/"
    exit 1
fi

# 2. Run validations based on architecture
if [ "$ARCH" = "riscv64" ]; then
    echo "[INFO] RISC-V environment detected (Target: SpacemiT K1/K3)."
    echo "[INFO] Compiling with RVV (Vector) target features..."
    
    # We set specific RUSTFLAGS to ensure vector extensions are leveraged
    export RUSTFLAGS="-C target-feature=+v"
    
    # Force single-thread to get deterministic batch-1 metrics
    export RAYON_NUM_THREADS=1

    echo "[EXEC] Running benchmark report..."
    cargo run --bin runux-report --release --features "k3_a100"
    
elif [ "$ARCH" = "x86_64" ] || [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then
    echo "[INFO] x86_64 / ARM64 environment detected."
    
    # Check for TPU
    if lsmod | grep -q "libtpu" || [ -e "/dev/accel0" ]; then
        echo "[INFO] Google TPU detected (Target: TPU v5e)."
        echo "[INFO] Enabling FP8 native MXU paths..."
        
        # We simulate the env vars that the TPU runtime would expect
        export PJRT_DEVICE="TPU"
        export TPU_NUM_DEVICES=1
        
        echo "[EXEC] Running benchmark report with TPU simulations..."
        cargo run --bin runux-report --release
    else
        echo "[INFO] Standard CPU environment detected. Running baseline benchmark..."
        cargo run --bin runux-report --release
    fi
else
    echo "[ERROR] Unsupported architecture for optimized benchmark: $ARCH"
    exit 1
fi

echo "========================================================================"
echo "Benchmark Execution Complete."
echo "Please report the AUTORESEARCH_METRIC JSON to the dashboard."
