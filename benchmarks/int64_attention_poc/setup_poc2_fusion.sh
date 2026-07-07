#!/bin/bash
# Setup script for PoC 2 Triton/CUDA fusion phase
# Prepares environment and runs validation harness

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "PoC 2 Fusion Phase Setup"
echo "=========================================="

# Check GPU
echo ""
echo "[1/6] Checking GPU..."
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader
COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_capability --format=csv,noheader | head -1)
echo "Compute capability: $COMPUTE_CAP"

# Check PyTorch
echo ""
echo "[2/6] Checking PyTorch and CUDA..."
python3 << 'PYEOF'
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.get_device_name(0)}")
print(f"CUDA Available: {torch.cuda.is_available()}")
PYEOF

# Check Triton
echo ""
echo "[3/6] Checking Triton..."
python3 << 'PYEOF'
try:
    import triton
    print(f"Triton: {triton.__version__}")
except ImportError:
    print("Triton not installed - installing...")
    import subprocess
    subprocess.run(["pip", "install", "-U", "triton"], check=True)
    import triton
    print(f"Triton installed: {triton.__version__}")
PYEOF

# Set optimizations for faster Triton compilation on T4
echo ""
echo "[4/6] Configuring Triton for T4..."
export TRITON_CACHE_DIR="${SCRIPT_DIR}/.triton_cache"
mkdir -p "$TRITON_CACHE_DIR"
export TRITON_DISABLE_LINE_WRAPPING=1

# For T4, reduce optimization level to speed up compilation
export TRITON_CODEGEN_T4_OPT="-O1"  # Reduced from O2/O3
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

python3 << 'PYEOF'
import os
print(f"Triton cache: {os.environ.get('TRITON_CACHE_DIR')}")
print(f"CUDA visible: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
PYEOF

# Validate PoC 2 reference implementation
echo ""
echo "[5/6] Validating PoC 2 reference implementation..."
python3 << 'PYEOF'
import torch
from poc2_fused_kernel import build_int64_softmax_lut, lut_softmax_attention_reference

B, H, S, D = 1, 2, 64, 16
device = "cuda:0"
SCALE = 100
SCALE_SHIFT = 6
LUT_HALF = 128

q = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
k = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
v = torch.randn(B, H, S, D, device=device, dtype=torch.float16)

q_int = (q * SCALE).to(torch.int64)
k_int = (k * SCALE).to(torch.int64)
v_int = (v * SCALE).to(torch.int64)

lut = build_int64_softmax_lut(lut_half=LUT_HALF, exp_div=16, fixed_scale=1<<16, device=device)

# Run twice to check determinism
out1 = lut_softmax_attention_reference(q_int, k_int, v_int, lut, SCALE_SHIFT, LUT_HALF, chunk=32)
out2 = lut_softmax_attention_reference(q_int, k_int, v_int, lut, SCALE_SHIFT, LUT_HALF, chunk=32)

deterministic = torch.equal(out1, out2)
print(f"✓ PoC 2 reference validated (deterministic: {deterministic})")
PYEOF

# Create fusion test harness
echo ""
echo "[6/6] Creating fusion test harness..."
cat > fusion_harness.py << 'FUSIONEOF'
#!/usr/bin/env python3
"""Fast iteration harness for PoC 2 fusion kernel development.

Usage:
  python3 fusion_harness.py --kernel new_kernel.py --test
  python3 fusion_harness.py --benchmark
"""
import sys
import time
import torch
import argparse
import importlib.util

from poc2_fused_kernel import (
    build_int64_softmax_lut,
    lut_softmax_attention_reference,
)


def load_kernel_module(path):
    """Dynamically load a kernel module."""
    spec = importlib.util.spec_from_file_location("fusion_kernel", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_correctness(kernel_func, problem_size=(2, 4, 128, 32)):
    """Run quick correctness test against reference."""
    B, H, S, D = problem_size
    device = "cuda:0"
    SCALE = 100
    SCALE_SHIFT = 6
    LUT_HALF = 128

    q = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
    k = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
    v = torch.randn(B, H, S, D, device=device, dtype=torch.float16)

    q_int = (q * SCALE).to(torch.int64)
    k_int = (k * SCALE).to(torch.int64)
    v_int = (v * SCALE).to(torch.int64)

    lut = build_int64_softmax_lut(lut_half=LUT_HALF, exp_div=16, fixed_scale=1<<16, device=device)

    try:
        out_fused = kernel_func(q_int, k_int, v_int, lut, SCALE_SHIFT, LUT_HALF)
        out_ref = lut_softmax_attention_reference(q_int, k_int, v_int, lut, SCALE_SHIFT, LUT_HALF, chunk=32)

        # Check parity
        parity = torch.allclose(out_fused, out_ref, atol=1)

        # Check determinism
        out_fused2 = kernel_func(q_int, k_int, v_int, lut, SCALE_SHIFT, LUT_HALF)
        deterministic = torch.equal(out_fused, out_fused2)

        print(f"✓ Kernel test passed")
        print(f"  Parity (vs reference): {parity}")
        print(f"  Determinism: {deterministic}")
        print(f"  Output shape: {out_fused.shape}")
        return True

    except Exception as e:
        print(f"✗ Kernel test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def benchmark_kernel(kernel_func, problem_size=(4, 8, 256, 64), iters=3):
    """Quick benchmark of fusion kernel vs reference."""
    B, H, S, D = problem_size
    device = "cuda:0"
    SCALE = 100
    SCALE_SHIFT = 6
    LUT_HALF = 128

    q = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
    k = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
    v = torch.randn(B, H, S, D, device=device, dtype=torch.float16)

    q_int = (q * SCALE).to(torch.int64)
    k_int = (k * SCALE).to(torch.int64)
    v_int = (v * SCALE).to(torch.int64)

    lut = build_int64_softmax_lut(lut_half=LUT_HALF, exp_div=16, fixed_scale=1<<16, device=device)

    # Warmup
    for _ in range(2):
        kernel_func(q_int, k_int, v_int, lut, SCALE_SHIFT, LUT_HALF)
        lut_softmax_attention_reference(q_int, k_int, v_int, lut, SCALE_SHIFT, LUT_HALF, chunk=32)
    torch.cuda.synchronize()

    # Benchmark fusion
    start = time.time()
    for _ in range(iters):
        kernel_func(q_int, k_int, v_int, lut, SCALE_SHIFT, LUT_HALF)
    torch.cuda.synchronize()
    time_fused = (time.time() - start) / iters

    # Benchmark reference
    start = time.time()
    for _ in range(iters):
        lut_softmax_attention_reference(q_int, k_int, v_int, lut, SCALE_SHIFT, LUT_HALF, chunk=32)
    torch.cuda.synchronize()
    time_ref = (time.time() - start) / iters

    print(f"Benchmark (problem: {problem_size}):")
    print(f"  Reference:  {time_ref*1000:.2f} ms")
    print(f"  Fusion:     {time_fused*1000:.2f} ms")
    print(f"  Speedup:    {time_ref/time_fused:.2f}x")


def main():
    parser = argparse.ArgumentParser(description="PoC 2 fusion kernel harness")
    parser.add_argument("--kernel", type=str, default=None, help="Path to custom kernel module")
    parser.add_argument("--test", action="store_true", help="Run correctness test")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark")
    parser.add_argument("--problem-size", type=int, nargs=4, default=[2, 4, 128, 32],
                        help="Problem size (B H S D)")

    args = parser.parse_args()

    if args.kernel:
        print(f"Loading kernel from {args.kernel}...")
        module = load_kernel_module(args.kernel)

        # Find the kernel function
        kernel_funcs = [name for name in dir(module) if callable(getattr(module, name)) and 'attention' in name.lower()]
        if not kernel_funcs:
            print(f"✗ No attention kernel found in {args.kernel}")
            sys.exit(1)

        kernel_func = getattr(module, kernel_funcs[0])
        print(f"Found kernel: {kernel_funcs[0]}")

        if args.test:
            print("\nRunning correctness test...")
            test_correctness(kernel_func, tuple(args.problem_size))

        if args.benchmark:
            print("\nRunning benchmark...")
            benchmark_kernel(kernel_func, tuple(args.problem_size))
    else:
        print("PoC 2 Fusion Harness")
        print("  --kernel <path>      Load custom kernel module")
        print("  --test               Run correctness test")
        print("  --benchmark          Run benchmark")
        print("  --problem-size B H S D  Set problem size")
        print("\nExample:")
        print("  python3 fusion_harness.py --kernel my_fusion.py --test --benchmark")


if __name__ == "__main__":
    main()
FUSIONEOF

chmod +x fusion_harness.py
echo "✓ Fusion harness created"

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Next steps for Triton/CUDA fusion:"
echo "  1. Implement fusion kernel (e.g., fused_attention_kernel.py)"
echo "  2. Test with: python3 fusion_harness.py --kernel fused_attention_kernel.py --test"
echo "  3. Benchmark with: python3 fusion_harness.py --kernel fused_attention_kernel.py --benchmark"
echo ""
echo "Or run full PoC 2 benchmark:"
echo "  python3 benchmark_poc2_reference.py"
echo ""
