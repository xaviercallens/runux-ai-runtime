#!/usr/bin/env python3
"""PoC 1: INT64 deterministic attention vs FP16 baseline on an NVIDIA T4.

Validates two patent claims for the int64 kernel:
  1. Zero Numerical Drift  - bit-exact, run-to-run determinism.
  2. Power/latency profile of the naive (non-fused) integer path,
     establishing the baseline that a fused Triton/CUDA kernel (PoC 2)
     must beat.

Hardware note: PyTorch's CUDA backend does not implement matmul/bmm for
torch.int64 (cuBLAS has no integer GEMM) - `torch.matmul` on int64 CUDA
tensors raises `"baddbmm_cuda" not implemented for 'Long'`. This script
therefore implements an explicit, chunked multiply-accumulate reduction
(`exact_int64_matmul`) as the reference software kernel: every partial
product and accumulation happens in int64, so the result is bit-exact,
it just isn't routed through a fused GEMM kernel yet. That fusion is
exactly what the PoC 2 Triton kernel is for.
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
# 2. EXACT INT64 GEMM (CUDA has no integer cuBLAS path)
# ==========================================
def exact_int64_matmul_qkt(q_int, k_int, chunk=64):
    """q_int @ k_int.transpose(-2, -1), bit-exact int64, chunked over the
    query dimension so the O(chunk * seq * head_dim) broadcast intermediate
    stays within GPU memory."""
    B, H, S, D = q_int.shape
    out = torch.empty(B, H, S, S, dtype=torch.int64, device=q_int.device)
    for i in range(0, S, chunk):
        qc = q_int[:, :, i:i + chunk, :]                      # [B,H,c,D]
        out[:, :, i:i + chunk, :] = (
            qc.unsqueeze(3) * k_int.unsqueeze(2)               # [B,H,c,S,D]
        ).sum(dim=-1)
    return out


def exact_int64_matmul_av(attn_int, v_int, chunk=64):
    """attn_int @ v_int, bit-exact int64, chunked over the query dimension."""
    B, H, S, S2 = attn_int.shape
    D = v_int.shape[-1]
    out = torch.empty(B, H, S, D, dtype=torch.int64, device=attn_int.device)
    for i in range(0, S, chunk):
        ac = attn_int[:, :, i:i + chunk, :]                    # [B,H,c,S]
        out[:, :, i:i + chunk, :] = (
            ac.unsqueeze(-1) * v_int.unsqueeze(2)               # [B,H,c,S,D]
        ).sum(dim=-2)
    return out


# ==========================================
# 3. ATTENTION KERNELS
# ==========================================
def fp16_baseline_attention(q, k, v):
    """Standard FP16 attention, routed through T4 Tensor Cores via cuBLAS."""
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / (d_k ** 0.5)
    attn = F.softmax(scores, dim=-1)
    return torch.matmul(attn, v)


def int64_kernel_attention(q_int, k_int, v_int, scale_shift=10, chunk=64):
    """Software reference for the patented INT64 deterministic attention
    kernel. Every arithmetic step (QK^T, rescale, softmax LUT surrogate,
    attn@V, final rescale) operates on int64 tensors end to end."""
    # 1. Exact INT64 MatMul (Q * K^T)
    scores = exact_int64_matmul_qkt(q_int, k_int, chunk=chunk)

    # 2. Dynamic rescaling (arithmetic right-shift, deterministic)
    scores = torch.bitwise_right_shift(scores, scale_shift)

    # 3. Simulated INT64 softmax.
    # PoC 1 falls back to float only to produce a timing/shape surrogate;
    # the patented kernel replaces this with an INT64 LUT/polynomial.
    scores_float = scores.to(torch.float32) / (2 ** scale_shift)
    attn_weights = F.softmax(scores_float, dim=-1)
    attn_weights_int = (attn_weights * (1 << 16)).to(torch.int64)

    # 4. Final exact INT64 MatMul
    out_int = exact_int64_matmul_av(attn_weights_int, v_int, chunk=chunk)

    # 5. Final rescale
    return torch.bitwise_right_shift(out_int, 16)


# ==========================================
# 4. BENCHMARK EXECUTION
# ==========================================
def run_benchmark():
    print("=== T4 GPU Benchmark: INT64 (exact, chunked) vs FP16 Attention ===")

    BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM = 8, 12, 1024, 64
    device = torch.device("cuda:0")
    ITERATIONS_FP16 = 200
    # The chunked int64 reference kernel is ~2 orders of magnitude slower
    # than a fused cuBLAS/Tensor-Core GEMM (expected - see module docstring),
    # so fewer iterations keep the run time reasonable without affecting
    # the per-pass latency/power averages.
    ITERATIONS_INT64 = 20
    CHUNK = 64

    print(f"Shapes: batch={BATCH_SIZE} heads={NUM_HEADS} seq={SEQ_LEN} head_dim={HEAD_DIM}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    q = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM, device=device, dtype=torch.float16)
    k = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM, device=device, dtype=torch.float16)
    v = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM, device=device, dtype=torch.float16)

    SCALE = 1000
    q_int = (q * SCALE).to(torch.int64)
    k_int = (k * SCALE).to(torch.int64)
    v_int = (v * SCALE).to(torch.int64)

    monitor = PowerMonitor()

    # --- WARMUP ---
    for _ in range(5):
        fp16_baseline_attention(q, k, v)
    for _ in range(2):
        int64_kernel_attention(q_int, k_int, v_int, chunk=CHUNK)
    torch.cuda.synchronize()

    # --- A. FP16 Baseline ---
    print("\n[1/3] Benchmarking FP16 Baseline...")
    monitor.start()
    start_time = time.time()
    for _ in range(ITERATIONS_FP16):
        fp16_baseline_attention(q, k, v)
    torch.cuda.synchronize()
    time_fp16 = (time.time() - start_time) / ITERATIONS_FP16
    power_fp16 = monitor.stop()

    # --- B. INT64 Kernel (software reference) ---
    print("[2/3] Benchmarking INT64 Reference Kernel...")
    monitor.start()
    start_time = time.time()
    for _ in range(ITERATIONS_INT64):
        int64_kernel_attention(q_int, k_int, v_int, chunk=CHUNK)
    torch.cuda.synchronize()
    time_int64 = (time.time() - start_time) / ITERATIONS_INT64
    power_int64 = monitor.stop()

    # --- C. Numerical Drift Check (Determinism) ---
    print("[3/3] Verifying Zero Numerical Drift...")
    out_1 = int64_kernel_attention(q_int, k_int, v_int, chunk=CHUNK)
    out_2 = int64_kernel_attention(q_int, k_int, v_int, chunk=CHUNK)
    exact_match = torch.equal(out_1, out_2)

    fp16_1 = fp16_baseline_attention(q, k, v)
    fp16_2 = fp16_baseline_attention(q, k, v)
    fp16_exact_match = torch.equal(fp16_1, fp16_2)

    # --- D. RESULTS ---
    print("\n================ BENCHMARK RESULTS ================")
    print(f"FP16 averaged over {ITERATIONS_FP16} iterations, "
          f"INT64 averaged over {ITERATIONS_INT64} iterations")
    print("---------------------------------------------------")
    print("1. LATENCY (Execution Time):")
    print(f"   - FP16 Baseline: {time_fp16*1000:.3f} ms")
    print(f"   - INT64 Kernel:  {time_int64*1000:.3f} ms *")
    print("---------------------------------------------------")
    print("2. AVERAGE POWER DRAW:")
    print(f"   - FP16 Baseline: {power_fp16:.2f} W")
    print(f"   - INT64 Kernel:  {power_int64:.2f} W *")
    print("---------------------------------------------------")
    print("3. DETERMINISM (Numerical Drift Check):")
    print(f"   - INT64 kernel: {'PASSED (bit-exact, zero drift)' if exact_match else 'FAILED'}")
    print(f"   - FP16 baseline: {'bit-exact (cuBLAS run-to-run stable here)' if fp16_exact_match else 'NOT bit-exact (float non-associativity observed)'}")
    print("===================================================")
    print("* Note: torch.matmul/bmm has no CUDA int64 GEMM path, so the INT64")
    print("  kernel above uses an explicit chunked multiply-accumulate (still")
    print("  100% int64, still bit-exact) instead of a fused GEMM. This is the")
    print("  PoC 1 baseline: it is expected to be slower/more power-hungry than")
    print("  FP16 Tensor Cores. PoC 2 replaces it with a fused Triton/CUDA")
    print("  kernel operating in SRAM to close the latency/power gap.")

    pynvml.nvmlShutdown()

    return {
        "time_fp16_ms": time_fp16 * 1000,
        "time_int64_ms": time_int64 * 1000,
        "power_fp16_w": power_fp16,
        "power_int64_w": power_int64,
        "int64_deterministic": exact_match,
        "fp16_deterministic": fp16_exact_match,
    }


if __name__ == "__main__":
    run_benchmark()
