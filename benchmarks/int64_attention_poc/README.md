# PoC 1 — INT64 Deterministic Attention Kernel (NVIDIA T4)

Validates two patent claims ahead of the fused Triton/CUDA kernel (PoC 2):

1. **Zero Numerical Drift** — the INT64 attention path is bit-exact and
   reproducible run-to-run.
2. **Baseline latency/power profile** for the un-fused, pure-PyTorch INT64
   path, which the PoC 2 Triton kernel must beat to support the energy
   reduction claim.

## Environment

This repo's dev VM already carries the target hardware (`nvidia-smi` →
`Tesla T4`, 15 GB, GCE `n1-standard-8`), so PoC 1 was run in place —
no separate `gcloud compute instances create` was needed. If reproducing
elsewhere, provision a T4 Deep Learning VM:

```bash
gcloud compute instances create int64-attention-dev \
    --zone=us-central1-a \
    --machine-type=n1-standard-4 \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --image-family=pytorch-latest-cu121-debian-11 \
    --image-project=deeplearning-platform-release \
    --maintenance-policy=TERMINATE \
    --boot-disk-size=100GB

gcloud compute ssh int64-attention-dev --zone=us-central1-a
pip install pynvml numpy
```

## Run it

```bash
pip install pynvml numpy
python3 benchmarks/int64_attention_poc/benchmark_t4.py
```

## Critical finding vs. the original plan

The originally sketched script called `torch.matmul` directly on
`torch.int64` CUDA tensors. On this stack (torch 2.7.1+cu118) that raises:

```
RuntimeError: "baddbmm_cuda" not implemented for 'Long'
```

cuBLAS has no integer GEMM path, so PyTorch's CUDA backend doesn't
implement `matmul`/`bmm` for `int64` at all — this isn't a slow path, it's
a hard failure. `benchmark_t4.py` replaces it with `exact_int64_matmul_qkt`
/ `exact_int64_matmul_av`: a chunked, explicit multiply-then-`sum` reduction
that stays 100% in `int64` (elementwise multiply and `sum` **are**
implemented for `int64` on CUDA) and is verified bit-exact against a CPU
`torch.matmul` reference. Chunking over the query dimension (`chunk=64`
rows) keeps the broadcast intermediate (`batch × heads × chunk × seq ×
head_dim`) within the T4's 15 GB VRAM.

This is the real PoC 1 software kernel — not a Triton kernel yet, just the
reference math running natively on the GPU end to end in integer
arithmetic.

## Results (Tesla T4, 2026-07-07T13:37:02Z, driver 580.159.03, torch 2.7.1+cu118/cu11.8)

Shapes: `batch=8, heads=12, seq_len=1024, head_dim=64`. Full output in
[`results_t4_poc1.txt`](results_t4_poc1.txt).

| Metric | FP16 (Tensor Cores) | INT64 (chunked, unfused) |
|---|---:|---:|
| Latency / attention pass | 5.97 ms (n=200) | 911.06 ms (n=20) — **~153× slower** |
| Average power draw | 68.26 W | 67.10 W |
| Determinism (bit-exact repeat) | stable in this run | **PASSED — zero drift** |

### Interpretation

- **Determinism claim: confirmed.** Two back-to-back calls with identical
  inputs produce `torch.equal() == True` — the INT64 path has no
  floating-point non-associativity to introduce run-to-run drift.
- **Energy claim: not yet demonstrated — this is expected at PoC 1.**
  Instantaneous power draw is roughly the same as FP16 (~67-68 W, i.e. the
  chunked kernel isn't power-hungrier per se), but because it takes ~153×
  longer per pass, **total energy per attention call is ~153× higher**,
  not lower. This matches the plan's own caveat: unfused PyTorch INT64
  writes every intermediate tensor to VRAM (`scores`, the full
  `batch×heads×chunk×seq×head_dim` broadcast buffer, `attn_weights_int`,
  ...), so it's memory-bandwidth-bound, not compute-bound. The 80% energy
  reduction claim depends on PoC 2 (a fused Triton/CUDA kernel keeping
  the whole `MatMul → rescale → LUT softmax → MatMul` chain in SRAM),
  which has not been built or measured yet.

## Next steps (PoC 2)

- Implement the fused attention chain as a single `@triton.jit` kernel
  (or CUDA kernel) so QK^T, the bit-shift rescale, an INT64 LUT/polynomial
  softmax, and the final AV matmul never round-trip through HBM.
- Re-run this same harness against the Triton kernel; only then does the
  latency/power comparison against FP16 become a meaningful energy claim.
- Replace the float-based softmax surrogate (`scores_float = ... F.softmax
  ...`) with an actual INT64 LUT or polynomial approximation — the current
  PoC 1 kernel is bit-exact end-to-end in the matmuls but still bounces
  through float32 for softmax, which is a placeholder, not the patented
  mechanism.
