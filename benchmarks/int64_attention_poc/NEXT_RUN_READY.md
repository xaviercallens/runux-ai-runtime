# PoC 2 Fusion Phase: Setup Complete ✓

## Setup Summary

**Completed on**: 2026-07-07 18:52 UTC  
**GPU**: Tesla T4 (compute capability 7.5), 15GB VRAM  
**Software**: PyTorch 2.7.1+cu118, Triton 3.3.1, CUDA 11.8

## What's Ready

### 1. **Validation Infrastructure** ✓
- ✓ Reference implementation (`poc2_fused_kernel.py`) — proven correct & deterministic
- ✓ Fusion test harness (`fusion_harness.py`) — quick test/benchmark
- ✓ Kernel template (`fusion_kernel_template.py`) — starting point for implementation

### 2. **Environment Setup** ✓
- ✓ GPU validated (Tesla T4 ready)
- ✓ Triton configured for T4 (cache dir: `.triton_cache`)
- ✓ Dependencies verified (PyTorch, Triton, CUDA)
- ✓ PoC 2 reference tested and determinism confirmed

### 3. **Documentation** ✓
- ✓ `POC2_STATUS.md` — current status and findings
- ✓ `FUSION_GUIDE.md` — complete implementation guide
- ✓ `fusion_kernel_template.py` — well-commented starting point

## Quick Start for Next Work

### To implement a fusion kernel:

```bash
cd benchmarks/int64_attention_poc

# 1. Copy template
cp fusion_kernel_template.py my_fusion.py

# 2. Edit my_fusion.py (implement your kernel)

# 3. Test correctness
python3 fusion_harness.py --kernel my_fusion.py --test

# 4. Benchmark performance
python3 fusion_harness.py --kernel my_fusion.py --benchmark
```

### Expected Results

When fusion is working correctly:
```
Benchmark (problem: (2, 4, 128, 32)):
  Reference:  1.08 ms
  Fusion:     0.3-0.5 ms          # Should be 2-5x faster
  Speedup:    2-3.6x              # Success if >2x
```

## Current Baseline

From PoC 2 reference validation:
```json
{
  "problem": "batch=2 heads=4 seq=256 head_dim=32",
  "poc1_float_ms": 3.695,    # Unfused (float softmax)
  "poc2_lut_ms": 3.735,      # Unfused (LUT softmax)
  "speedup": 0.99,           # Tied (unfused, bandwidth-bound)
  "poc2_deterministic": true # Determinism CONFIRMED
}
```

**Why tied?** Both unfused implementations materialize intermediates to HBM, making them bandwidth-bound. Fusion (keeping Q block in registers across K/V loop) will break the bandwidth bottleneck.

## Files Structure

```
benchmarks/int64_attention_poc/
├── setup_poc2_fusion.sh                # Setup script (already run)
├── fusion_harness.py                   # Test harness (ready to use)
├── fusion_kernel_template.py           # Triton template (copy this)
├── poc2_fused_kernel.py                # Reference (do not modify)
├── benchmark_poc2_reference.py         # Full PoC 2 validation
├── POC2_STATUS.md                      # Status doc
├── FUSION_GUIDE.md                     # Implementation guide
├── NEXT_RUN_READY.md                   # This file
├── .triton_cache/                      # Triton compiler cache
├── results_poc2_reference.json         # Baseline results
└── [other supporting files]
```

## Key Success Metrics

Your fusion kernel will demonstrate the patent claim when it achieves:

1. **Correctness**: `Parity (vs reference): True`
2. **Determinism**: `Determinism: True`
3. **Latency**: `Speedup: >2.0x` (fused faster than reference)
4. **Energy**: Total energy per call < FP16 baseline (TBD with power measurement)

## Next Phase Workflow

### Phase 1: Implement
1. Copy `fusion_kernel_template.py` → `my_fusion.py`
2. Implement your preferred approach (Triton, CUDA, or PyTorch)
3. Run `fusion_harness.py --kernel my_fusion.py --test` to validate

### Phase 2: Optimize
1. Run `fusion_harness.py --kernel my_fusion.py --benchmark` to measure
2. If speedup < 2x, review kernel design (might still be bandwidth-bound)
3. If speedup > 2x, fusion is working! Proceed to Phase 3

### Phase 3: Validate
1. Run `benchmark_poc2_reference.py` with your kernel to measure energy
2. Verify energy < FP16 baseline (claim validation)
3. Commit results and PR

## Troubleshooting

**Q: Triton compilation too slow?**  
A: Set `TRITON_CODEGEN_T4_OPT=-O1` and reduce block sizes (16x16 instead of 64x64)

**Q: Parity test fails?**  
A: Check QK^T reduction and LUT indexing in your kernel implementation

**Q: Still bandwidth-bound (speedup < 1.2x)?**  
A: Ensure Q block stays in registers across the K/V loop (not reloaded each iteration)

**Q: Ready to submit?**  
A: Make sure all three tests pass:
```bash
python3 fusion_harness.py --kernel my_fusion.py --test --benchmark
```

## Environment Variables

These were set up by `setup_poc2_fusion.sh`:

```bash
export TRITON_CACHE_DIR=".../.triton_cache"    # Compiler cache
export TRITON_CODEGEN_T4_OPT="-O1"             # Faster compilation
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
```

## References

- Patent claim: 80% energy reduction vs FP16
- PoC 1 results: `results_t4_poc1.txt` (determinism proven)
- PoC 2 reference: `results_poc2_reference.json` (LUT softmax proven)
- Implementation guide: `FUSION_GUIDE.md`

---

**Status**: ✓ Ready for fusion kernel implementation  
**Next Action**: Copy template, implement kernel, test, optimize

