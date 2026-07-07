#!/usr/bin/env python3
"""PoC 2: Fused Triton INT64 deterministic attention on NVIDIA T4.

Compares the fused Triton kernel (poc2_fused_kernel.py) against:
  1. PoC 1 reference (unfused, chunked INT64 with float softmax surrogate)
  2. FP16 baseline (standard attention via cuBLAS Tensor Cores)

Success metrics:
  - Bit-exact parity: Triton output == reference output (integer associativity)
  - Latency: closer to FP16 than PoC 1 (fusion keeps intermediates in SRAM)
  - Energy claim: measurable improvement (lower latency at similar power draw)
"""
import time
import threading

import numpy as np
import torch
import torch.nn.functional as F

try:
    import pynvml
except ImportError:
    print("Error: Please install pynvml via 'pip install pynvml'")
    raise SystemExit(1)

from poc2_fused_kernel import (
    build_int64_softmax_lut,
    lut_softmax_attention_reference,
    fused_int64_attention_triton,
)


# ==========================================
# 1. HARDWARE POWER MONITORING (NVML)
# ==========================================
class PowerMonitor:
    def __init__(self):
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        self.is_running = False
        self.readings = []

    def _monitor(self):
        while self.is_running:
            power_mw = pynvml.nvmlDeviceGetPowerUsage(self.handle)
            self.readings.append(power_mw / 1000.0)
            time.sleep(0.005)  # 5ms poll

    def start(self):
        self.is_running = True
        self.readings = []
        self.thread = threading.Thread(target=self._monitor)
        self.thread.start()

    def stop(self):
        self.is_running = False
        self.thread.join()
        return np.mean(self.readings) if self.readings else 0.0


# ==========================================
# 2. PoC 1 REFERENCE (UNFUSED, CHUNKED)
# ==========================================
def exact_int64_matmul_qkt(q_int, k_int, chunk=64):
    B, H, S, D = q_int.shape
    out = torch.empty(B, H, S, S, dtype=torch.int64, device=q_int.device)
    for i in range(0, S, chunk):
        qc = q_int[:, :, i:i + chunk, :]
        out[:, :, i:i + chunk, :] = (
            qc.unsqueeze(3) * k_int.unsqueeze(2)
        ).sum(dim=-1)
    return out


def exact_int64_matmul_av(attn_int, v_int, chunk=64):
    B, H, S, S2 = attn_int.shape
    D = v_int.shape[-1]
    out = torch.empty(B, H, S, D, dtype=torch.int64, device=attn_int.device)
    for i in range(0, S, chunk):
        ac = attn_int[:, :, i:i + chunk, :]
        out[:, :, i:i + chunk, :] = (
            ac.unsqueeze(-1) * v_int.unsqueeze(2)
        ).sum(dim=-2)
    return out


def poc1_reference_int64_kernel(q_int, k_int, v_int, scale_shift=10, chunk=64):
    """PoC 1 software reference: unfused, float softmax surrogate."""
    scores = exact_int64_matmul_qkt(q_int, k_int, chunk=chunk)
    scores = torch.bitwise_right_shift(scores, scale_shift)
    scores_float = scores.to(torch.float32) / (2 ** scale_shift)
    attn_weights = F.softmax(scores_float, dim=-1)
    attn_weights_int = (attn_weights * (1 << 16)).to(torch.int64)
    out_int = exact_int64_matmul_av(attn_weights_int, v_int, chunk=chunk)
    return torch.bitwise_right_shift(out_int, 16)


# ==========================================
# 3. FP16 BASELINE
# ==========================================
def fp16_baseline_attention(q, k, v):
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / (d_k ** 0.5)
    attn = F.softmax(scores, dim=-1)
    return torch.matmul(attn, v)


# ==========================================
# 4. BENCHMARK EXECUTION
# ==========================================
def run_benchmark():
    print("=== T4 GPU Benchmark: PoC 2 (Fused Triton) vs PoC 1 vs FP16 ===")

    # Smaller problem size to avoid OOM on T4 with fragmentation
    BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM = 4, 8, 512, 64
    device = torch.device("cuda:0")

    ITERATIONS_FP16 = 100
    ITERATIONS_TRITON = 30  # Reduced to fit in memory
    ITERATIONS_POC1 = 5     # Reduced - PoC 1 reference uses huge intermediates
    CHUNK = 32              # Smaller chunks to reduce peak memory

    print(f"Shapes: batch={BATCH_SIZE} heads={NUM_HEADS} seq={SEQ_LEN} head_dim={HEAD_DIM}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Create FP16 tensors and convert to INT64
    q = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM, device=device, dtype=torch.float16)
    k = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM, device=device, dtype=torch.float16)
    v = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM, device=device, dtype=torch.float16)

    SCALE = 1000
    q_int = (q * SCALE).to(torch.int64)
    k_int = (k * SCALE).to(torch.int64)
    v_int = (v * SCALE).to(torch.int64)

    # Build INT64 LUT softmax (one-time host precomputation)
    print("\nPrecomputing INT64 softmax LUT...")
    LUT_HALF = 128
    EXP_DIV = 16
    FIXED_SCALE = 1 << 16
    SCALE_SHIFT = 10
    lut = build_int64_softmax_lut(lut_half=LUT_HALF, exp_div=EXP_DIV, fixed_scale=FIXED_SCALE, device=device)
    print(f"  LUT shape: {lut.shape}, dtype: {lut.dtype}")

    monitor = PowerMonitor()

    # --- WARMUP ---
    print("\nWarmup...")
    for _ in range(5):
        fp16_baseline_attention(q, k, v)
    for _ in range(2):
        poc1_reference_int64_kernel(q_int, k_int, v_int, scale_shift=SCALE_SHIFT, chunk=CHUNK)
    try:
        for _ in range(2):
            fused_int64_attention_triton(q_int, k_int, v_int, lut, scale_shift=SCALE_SHIFT, lut_half=LUT_HALF)
    except Exception as e:
        print(f"  Triton warmup error (may need investigation): {e}")
    torch.cuda.synchronize()

    # --- A. FP16 Baseline ---
    print("\n[1/4] Benchmarking FP16 Baseline...")
    monitor.start()
    start_time = time.time()
    for _ in range(ITERATIONS_FP16):
        fp16_baseline_attention(q, k, v)
    torch.cuda.synchronize()
    time_fp16 = (time.time() - start_time) / ITERATIONS_FP16
    power_fp16 = monitor.stop()

    # --- B. PoC 1 Reference (unfused INT64) ---
    print("[2/4] Benchmarking PoC 1 Reference (unfused INT64)...")
    monitor.start()
    start_time = time.time()
    for _ in range(ITERATIONS_POC1):
        poc1_reference_int64_kernel(q_int, k_int, v_int, scale_shift=SCALE_SHIFT, chunk=CHUNK)
    torch.cuda.synchronize()
    time_poc1 = (time.time() - start_time) / ITERATIONS_POC1
    power_poc1 = monitor.stop()

    # --- C. PoC 2 Fused Triton Kernel ---
    print("[3/4] Benchmarking PoC 2 Fused Triton Kernel...")
    monitor.start()
    start_time = time.time()
    for _ in range(ITERATIONS_TRITON):
        fused_int64_attention_triton(q_int, k_int, v_int, lut, scale_shift=SCALE_SHIFT, lut_half=LUT_HALF)
    torch.cuda.synchronize()
    time_triton = (time.time() - start_time) / ITERATIONS_TRITON
    power_triton = monitor.stop()

    # --- D. Bit-Exact Parity Check ---
    print("[4/4] Verifying Bit-Exact Parity (Triton vs Reference)...")
    out_reference = lut_softmax_attention_reference(q_int, k_int, v_int, lut, SCALE_SHIFT, LUT_HALF, chunk=CHUNK)
    out_triton = fused_int64_attention_triton(q_int, k_int, v_int, lut, scale_shift=SCALE_SHIFT, lut_half=LUT_HALF)
    bit_exact_match = torch.equal(out_reference, out_triton)

    # Also verify Triton determinism (same input, two calls)
    out_triton_2 = fused_int64_attention_triton(q_int, k_int, v_int, lut, scale_shift=SCALE_SHIFT, lut_half=LUT_HALF)
    triton_deterministic = torch.equal(out_triton, out_triton_2)

    # --- E. RESULTS ---
    print("\n================ BENCHMARK RESULTS ================")
    print(f"FP16: {ITERATIONS_FP16} iters, PoC1: {ITERATIONS_POC1} iters, PoC2: {ITERATIONS_TRITON} iters")
    print("---------------------------------------------------")
    print("1. LATENCY (Execution Time per pass):")
    print(f"   - FP16 Baseline:    {time_fp16*1000:.3f} ms")
    print(f"   - PoC 1 (unfused):  {time_poc1*1000:.3f} ms ({time_poc1/time_fp16:.1f}× vs FP16)")
    print(f"   - PoC 2 (Triton):   {time_triton*1000:.3f} ms ({time_triton/time_fp16:.1f}× vs FP16)")
    if time_poc1 > 0:
        print(f"                        ({time_poc1/time_triton:.1f}× improvement over PoC 1)")
    print("---------------------------------------------------")
    print("2. AVERAGE POWER DRAW:")
    print(f"   - FP16 Baseline:    {power_fp16:.2f} W")
    print(f"   - PoC 1 (unfused):  {power_poc1:.2f} W")
    print(f"   - PoC 2 (Triton):   {power_triton:.2f} W")
    print("---------------------------------------------------")
    print("3. ENERGY PER ATTENTION CALL (latency × power):")
    energy_fp16 = time_fp16 * power_fp16
    energy_poc1 = time_poc1 * power_poc1
    energy_triton = time_triton * power_triton
    print(f"   - FP16 Baseline:    {energy_fp16:.2f} J (baseline)")
    print(f"   - PoC 1 (unfused):  {energy_poc1:.2f} J ({energy_poc1/energy_fp16:.1f}× vs FP16)")
    print(f"   - PoC 2 (Triton):   {energy_triton:.2f} J ({energy_triton/energy_fp16:.1f}× vs FP16)")
    print("---------------------------------------------------")
    print("4. CORRECTNESS & DETERMINISM:")
    print(f"   - Triton vs Reference (bit-exact): {'PASSED' if bit_exact_match else 'FAILED'}")
    print(f"   - Triton determinism (run-to-run):  {'PASSED' if triton_deterministic else 'FAILED'}")
    print("===================================================")

    # Summary
    if time_triton < time_poc1:
        improvement = (time_poc1 - time_triton) / time_poc1 * 100
        print(f"✓ PoC 2 achieves {improvement:.1f}% latency improvement over PoC 1")
    else:
        print("✗ PoC 2 did not improve latency (check Triton implementation)")

    if energy_triton < energy_fp16:
        energy_reduction = (energy_fp16 - energy_triton) / energy_fp16 * 100
        print(f"✓ Energy reduction vs FP16: {energy_reduction:.1f}%")
    else:
        print(f"✗ Energy still higher than FP16 by {(energy_triton/energy_fp16 - 1)*100:.1f}%")

    pynvml.nvmlShutdown()

    return {
        "time_fp16_ms": time_fp16 * 1000,
        "time_poc1_ms": time_poc1 * 1000,
        "time_triton_ms": time_triton * 1000,
        "power_fp16_w": power_fp16,
        "power_poc1_w": power_poc1,
        "power_triton_w": power_triton,
        "energy_fp16_j": energy_fp16,
        "energy_poc1_j": energy_poc1,
        "energy_triton_j": energy_triton,
        "bit_exact_match": bit_exact_match,
        "triton_deterministic": triton_deterministic,
    }


if __name__ == "__main__":
    run_benchmark()
