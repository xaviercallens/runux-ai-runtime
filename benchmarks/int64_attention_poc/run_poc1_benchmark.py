#!/usr/bin/env python3
"""Quick PoC 1 benchmark (reference INT64 kernel, no Triton compilation)."""
import time
import json
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np

try:
    import pynvml
    pynvml.nvmlInit()
    GPU_MONITOR = True
except:
    GPU_MONITOR = False

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def exact_int64_matmul_qkt(q_int, k_int, chunk=64):
    B, H, S, D = q_int.shape
    out = torch.empty(B, H, S, S, dtype=torch.int64, device=q_int.device)
    for i in range(0, S, chunk):
        qc = q_int[:, :, i:i + chunk, :]
        out[:, :, i:i + chunk, :] = (qc.unsqueeze(3) * k_int.unsqueeze(2)).sum(dim=-1)
    return out

def exact_int64_matmul_av(attn_int, v_int, chunk=64):
    B, H, S, S2 = attn_int.shape
    D = v_int.shape[-1]
    out = torch.empty(B, H, S, D, dtype=torch.int64, device=attn_int.device)
    for i in range(0, S, chunk):
        ac = attn_int[:, :, i:i + chunk, :]
        out[:, :, i:i + chunk, :] = (ac.unsqueeze(-1) * v_int.unsqueeze(2)).sum(dim=-2)
    return out

def poc1_reference_kernel(q_int, k_int, v_int, scale_shift=10, chunk=64):
    scores = exact_int64_matmul_qkt(q_int, k_int, chunk=chunk)
    scores = torch.bitwise_right_shift(scores, scale_shift)
    scores_float = scores.to(torch.float32) / (2 ** scale_shift)
    attn_weights = F.softmax(scores_float, dim=-1)
    attn_weights_int = (attn_weights * (1 << 16)).to(torch.int64)
    out_int = exact_int64_matmul_av(attn_weights_int, v_int, chunk=chunk)
    return torch.bitwise_right_shift(out_int, 16)

def fp16_baseline(q, k, v):
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / (d_k ** 0.5)
    attn = F.softmax(scores, dim=-1)
    return torch.matmul(attn, v)

log("=== PoC 1 Reference Kernel Benchmark ===")
log("(No Triton compilation — pure PyTorch)")

B, H, S, D = 4, 8, 512, 64
device = torch.device("cuda:0")
SCALE = 1000

log(f"Problem size: batch={B} heads={H} seq={S} head_dim={D}")
log("Creating tensors...")
q = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
k = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
v = torch.randn(B, H, S, D, device=device, dtype=torch.float16)

q_int = (q * SCALE).to(torch.int64)
k_int = (k * SCALE).to(torch.int64)
v_int = (v * SCALE).to(torch.int64)

log("Warmup...")
for _ in range(2):
    fp16_baseline(q, k, v)
for _ in range(1):
    poc1_reference_kernel(q_int, k_int, v_int)
torch.cuda.synchronize()

log("Benchmarking FP16 baseline (100 iterations)...")
start = time.time()
for _ in range(100):
    fp16_baseline(q, k, v)
torch.cuda.synchronize()
t_fp16 = (time.time() - start) / 100

log(f"Benchmarking PoC 1 (10 iterations, slower)...")
start = time.time()
for _ in range(10):
    poc1_reference_kernel(q_int, k_int, v_int)
torch.cuda.synchronize()
t_poc1 = (time.time() - start) / 10

log("Checking determinism...")
out1 = poc1_reference_kernel(q_int, k_int, v_int)
out2 = poc1_reference_kernel(q_int, k_int, v_int)
deterministic = torch.equal(out1, out2)

results = {
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "benchmark": "PoC 1 Reference (no Triton)",
    "problem_size": f"batch={B} heads={H} seq={S} head_dim={D}",
    "fp16_latency_ms": float(t_fp16 * 1000),
    "poc1_latency_ms": float(t_poc1 * 1000),
    "speedup_vs_fp16": float(t_poc1 / t_fp16),
    "deterministic": deterministic,
}

log("")
log("=== RESULTS ===")
log(f"FP16 baseline:      {t_fp16*1000:.2f} ms")
log(f"PoC 1 (unfused):    {t_poc1*1000:.2f} ms")
log(f"Slowdown vs FP16:   {t_poc1/t_fp16:.1f}×")
log(f"Determinism check:  {'✓ PASS' if deterministic else '✗ FAIL'}")

with open(Path(__file__).parent / "results_poc1_quick.json", "w") as f:
    json.dump(results, f, indent=2)

log(f"Results saved to results_poc1_quick.json")
