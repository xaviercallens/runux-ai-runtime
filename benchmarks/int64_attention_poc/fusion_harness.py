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
