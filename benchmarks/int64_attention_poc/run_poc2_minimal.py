#!/usr/bin/env python3
"""PoC 2 Triton kernel with minimal problem size to avoid compilation bloat."""
import time
import json
from pathlib import Path
import torch
import torch.nn.functional as F

from poc2_fused_kernel import (
    build_int64_softmax_lut,
    lut_softmax_attention_reference,
    fused_int64_attention_triton,
)

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

log("=== PoC 2 Triton Kernel (Minimal Size) ===")

# Much smaller to reduce compilation load
B, H, S, D = 2, 4, 128, 64
device = torch.device("cuda:0")
SCALE = 1000

log(f"Problem size: batch={B} heads={H} seq={S} head_dim={D}")
log(f"(Reduced for faster Triton compilation)")

log("Creating tensors...")
q = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
k = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
v = torch.randn(B, H, S, D, device=device, dtype=torch.float16)

q_int = (q * SCALE).to(torch.int64)
k_int = (k * SCALE).to(torch.int64)
v_int = (v * SCALE).to(torch.int64)

log("Building INT64 softmax LUT...")
LUT_HALF = 128
EXP_DIV = 16
FIXED_SCALE = 1 << 16
SCALE_SHIFT = 10
lut = build_int64_softmax_lut(lut_half=LUT_HALF, exp_div=EXP_DIV, fixed_scale=FIXED_SCALE, device=device)

log("Compiling Triton kernel (this may take 5-10 min)...")
start_compile = time.time()
try:
    for _ in range(1):
        fused_int64_attention_triton(q_int, k_int, v_int, lut, scale_shift=SCALE_SHIFT, lut_half=LUT_HALF)
    compile_time = time.time() - start_compile
    log(f"✓ Compilation succeeded ({compile_time:.1f}s)")
except Exception as e:
    log(f"✗ Compilation failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

log("Getting reference output for parity check...")
out_ref = lut_softmax_attention_reference(q_int, k_int, v_int, lut, SCALE_SHIFT, LUT_HALF, chunk=32)

log("Running Triton kernel (5 iterations)...")
start = time.time()
for _ in range(5):
    out_triton = fused_int64_attention_triton(q_int, k_int, v_int, lut, scale_shift=SCALE_SHIFT, lut_half=LUT_HALF)
torch.cuda.synchronize()
t_triton = (time.time() - start) / 5

log("Checking parity...")
parity_pass = torch.equal(out_ref, out_triton)

log("Checking determinism...")
out_triton2 = fused_int64_attention_triton(q_int, k_int, v_int, lut, scale_shift=SCALE_SHIFT, lut_half=LUT_HALF)
deterministic = torch.equal(out_triton, out_triton2)

results = {
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "benchmark": "PoC 2 Triton (minimal)",
    "problem_size": f"batch={B} heads={H} seq={S} head_dim={D}",
    "compilation_time_s": float(compile_time),
    "triton_latency_ms": float(t_triton * 1000),
    "parity_check": parity_pass,
    "deterministic": deterministic,
    "note": "Reduced problem size to minimize compilation load",
}

log("")
log("=== RESULTS ===")
log(f"Compilation:     {compile_time:.1f}s")
log(f"Triton latency:  {t_triton*1000:.2f} ms")
log(f"Bit-exact match: {'✓ PASS' if parity_pass else '✗ FAIL'}")
log(f"Determinism:     {'✓ PASS' if deterministic else '✗ FAIL'}")

with open(Path(__file__).parent / "results_poc2_minimal.json", "w") as f:
    json.dump(results, f, indent=2)

log(f"Results saved to results_poc2_minimal.json")
