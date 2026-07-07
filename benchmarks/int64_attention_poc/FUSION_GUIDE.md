# PoC 2 Fusion Phase Guide

This guide prepares you for implementing the Triton/CUDA fusion kernel to complete PoC 2 and demonstrate the 80% energy reduction claim.

## Quick Start

All setup is done. To begin fusion kernel development:

```bash
# Test your kernel implementation
python3 fusion_harness.py --kernel my_fusion.py --test

# Benchmark against reference
python3 fusion_harness.py --kernel my_fusion.py --benchmark

# Full PoC 2 validation
python3 benchmark_poc2_reference.py
```

## What Needs to Be Done

The fusion kernel must implement a **single fused operation** that combines:

```
QK^T (int64) -> Rescale (bit shift) -> LUT Softmax -> AV (int64)
```

**All intermediate tensors stay in SRAM/registers** — no HBM round-trips for scores, weights, or attention matrices.

### Key Requirements

1. **Correctness**: Output must match reference kernel bit-exactly (determinism critical)
2. **Determinism**: Same input → identical output (int64 associativity)
3. **Latency**: Faster than reference (~2-5x improvement expected with fusion)
4. **Energy**: Lower per-call energy than FP16 baseline (requires real hardware measurement)

## Implementation Options

### Option A: Triton @triton.jit (Recommended starting point)

**Pros**: Fast to develop, works on T4, explicit control  
**Cons**: Compiler can be slow on T4

**Key tips for T4**:
- Use `tl.static_range()` for loops (avoid unrolling explosion)
- Reduce block sizes if compilation is slow (32x32 instead of 64x64)
- Set `TRITON_CODEGEN_T4_OPT=-O1` environment variable
- Cache compiled kernels in `.triton_cache/`

**Starting point**: `fusion_kernel_template.py` has a basic structure ready to refine.

```python
@triton.jit
def fused_int64_attention_kernel(...):
    # Load Q block (stays resident)
    # Loop over K/V blocks:
    #   - Load K, V
    #   - Compute QK^T (manual int64 reduce, no GEMM)
    #   - LUT softmax
    #   - Accumulate output
    # Store output and denominator
```

### Option B: Native CUDA C++ (`fused_attention_kernel.cu`)

**Pros**: Best performance, no compiler delays, explicit control  
**Cons**: More code, needs CUDA compilation

**Key implementation**:
- Use shared memory for Q block (stays resident across K/V loop)
- Use registers for accumulators
- Manual QK^T reduction (elementwise multiply + warp reduce)
- LUT gather (no expensive softmax)

**Compilation**:
```bash
nvcc -arch=sm_75 -O3 fused_attention_kernel.cu -o fused_attention.so
```

### Option C: PyTorch with Memory Optimization (Fallback)

Already working in `fusion_kernel_template.py::fused_int64_attention_pytorch`.  
This validates correctness but **won't achieve the energy improvement** (still bandwidth-bound).

## Development Workflow

### Phase 1: Get It Working (correctness)

```bash
# Implement your kernel (copy from template as starting point)
cp fusion_kernel_template.py my_fusion.py
# Edit my_fusion.py - implement the fusion logic

# Quick test
python3 fusion_harness.py --kernel my_fusion.py --test

# Expected output:
# ✓ Kernel test passed
# Parity (vs reference): True
# Determinism: True
```

### Phase 2: Optimize for Performance (latency)

```bash
# Benchmark against reference
python3 fusion_harness.py --kernel my_fusion.py --benchmark

# Expected improvement: 2-5x speedup (fusion keeps data in SRAM)
# If not seeing improvement, you're still bandwidth-bound - review kernel design
```

### Phase 3: Energy Validation (hardware measurement)

Once latency is improved, measure energy:

```bash
# Full benchmark with power monitoring
python3 benchmark_poc2_reference.py
# Or modify to test your fusion kernel instead
```

## Common Issues & Solutions

### Triton Compilation Too Slow

**Symptom**: Kernel compiles for >10 minutes on T4  
**Solution**:
```bash
export TRITON_CODEGEN_T4_OPT=-O1  # Reduce optimization level
export TRITON_CACHE_DIR=.triton_cache  # Cache compiled kernels
# Or reduce block sizes: 16x16 instead of 64x64
```

### Output Doesn't Match Reference (Parity Fails)

**Symptom**: `Parity (vs reference): False`  
**Debug**:
1. Check intermediate shapes (print in kernel)
2. Verify QK^T reduction is correct (sum over HEAD_DIM)
3. Confirm LUT indexing (must clamp correctly)
4. Check denominator accumulation (sum of weights)

### "baddbmm_cuda" not implemented for 'Long'

**Symptom**: `torch.matmul` fails on int64 tensors  
**Solution**: This is expected — you **cannot** use `torch.matmul` on int64 CUDA tensors.  
Use manual reduction instead:
```python
# Wrong:
scores = torch.matmul(q, k.transpose(-2, -1))  # FAILS

# Right:
scores = (q.unsqueeze(3) * k.unsqueeze(2)).sum(dim=-1)  # Manual reduce
```

## File Organization

```
benchmarks/int64_attention_poc/
├── poc2_fused_kernel.py              # Reference (working)
├── fusion_kernel_template.py          # Starting template
├── fusion_harness.py                  # Test & benchmark harness
├── benchmark_poc2_reference.py        # Full PoC 2 validation
├── my_fusion.py                       # YOUR kernel goes here
├── POC2_STATUS.md                     # Current status
└── FUSION_GUIDE.md                    # This file
```

## Success Criteria

Your fusion kernel is "done" when:

- ✓ Parity test passes (bit-exact vs reference)
- ✓ Determinism test passes (same input → same output)
- ✓ Latency is **2-5x faster** than reference (fusion working)
- ✓ Energy per call is **lower than FP16 baseline** (claim validated)

## References

- **PoC 1**: Determinism validated, results in `results_t4_poc1.txt`
- **PoC 2 Reference**: Correctness validated, LUT softmax proven in `results_poc2_reference.json`
- **Triton Docs**: https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html
- **Flash Attention**: https://arxiv.org/abs/2205.14135 (inspiration for tiling pattern)

## Next Steps After Fusion

Once fusion kernel is optimized:
1. Run full benchmark suite at multiple problem sizes
2. Submit results for patent validation
3. Consider further optimization (e.g., CUDA graphs, pre-compiled binaries)
4. Explore scaling to larger GPUs (A100, H100)

---

**Last Updated**: 2026-07-07  
**Status**: Setup complete, ready for fusion implementation
