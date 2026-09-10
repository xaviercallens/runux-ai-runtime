# RunuX AI Runtime — Deep Audit Report

**Date**: 2026-09-10  
**Scope**: Full codebase audit — 24 Rust crates, Python services, CI/CD, formal specs, deployment  
**Auditor**: Antigravity multi-agent deep audit (4 parallel auditors)  
**Codebase Version**: v0.3.0 (16,003 lines Rust, ~2,000 lines Python)

---

## Executive Summary

> [!CAUTION]
> **The RunuX AI Runtime is fundamentally a simulation/prototype framework, not a production runtime.** The vast majority of claimed capabilities — TPU PJRT bindings, RVV vector kernels, hardware drivers, MLGO ML models, benchmark numbers, and formal proofs — are either simulated, hardcoded, or stubbed. No real hardware interaction occurs anywhere in the codebase. Benchmark numbers printed by the CLI and published externally are computed from analytical formulas, not measured on actual hardware.

### Severity Summary

| Severity | Count | Description |
|:--------:|:-----:|-------------|
| 🔴 **CRITICAL** | 14 | Fake implementations presented as real; hardcoded benchmark fabrication |
| 🟠 **HIGH** | 12 | Major missing functionality; simulated-only with no real path |
| 🟡 **MEDIUM** | 8 | Incomplete but partially acknowledged; analytical-only models |
| 🟢 **LOW** | 2 | Minor gaps, acceptable for prototype stage |

### Build Status

| Check | Status |
|-------|--------|
| `cargo check --workspace` | ❌ Fails — Lockfile collision with sibling project |
| `cargo test --workspace` | ❌ Fails — Same lockfile collision |
| Real hardware tests | ❌ None exist |
| CI green status | ⚠️ Artificially forced green (failure suppression) |

---

## Part 1: Core Runtime Layer

### 1.1 `hal` — Hardware Abstraction Layer
**File**: [lib.rs](file:///home/xavkal/xdev/runux-ai-runtime/crates/hal/src/lib.rs) (1,092 lines, 83 functions, 13 tests)

| Finding | Severity |
|---------|:--------:|
| `TpuSimulatorBackend` delegates all operations to scalar `CpuBackend` — zero TPU interaction | 🔴 CRITICAL |
| `AppleSiliconBackend` claims Metal/MPS but uses nested scalar loops; `alloc_tensor` assigns a fake handle | 🔴 CRITICAL |
| Zero actual FFI bindings to any hardware API (no PJRT, no MPS, no OpenCL, no Vulkan) | 🔴 CRITICAL |
| All backends are type aliases for the same CPU scalar code | 🟠 HIGH |

**Reality**: The "Unified HAL" is a single CPU scalar backend with multiple type names. The `Accelerator` trait interface is well-designed but every implementation is identical scalar Rust.

---

### 1.2 `tpu_pjrt` — TPU PJRT Bindings
**File**: [lib.rs](file:///home/xavkal/xdev/runux-ai-runtime/crates/tpu_pjrt/src/lib.rs) (656 lines, 36 functions, 10 tests)

| Finding | Severity |
|---------|:--------:|
| Documentation states "Safe Rust bindings for Google TPU via PJRT C API" — zero `extern "C"` FFI exists | 🔴 CRITICAL |
| `alloc_buffer` creates a CPU `Vec<f32>`, not a PJRT device buffer | 🔴 CRITICAL |
| `compile` records a name + estimated FLOPs into a struct but compiles nothing | 🔴 CRITICAL |
| `transfer_to_device` copies into the CPU Vec | 🟠 HIGH |

**Reality**: Complete simulation. No `libtpu.so`, no `pjrt_c_api.h`, no device handles.

---

### 1.3 `stablehlo` — StableHLO Graph Builder
**File**: [lib.rs](file:///home/xavkal/xdev/runux-ai-runtime/crates/stablehlo/src/lib.rs) (639 lines, 34 functions, 7 tests)

| Finding | Severity |
|---------|:--------:|
| `serialize_text()` outputs fake human-readable strings, not MLIR bytecode | 🟠 HIGH |
| Source comment admits: *"In a real implementation, this would produce MLIR bytecode"* | 🟠 HIGH |
| Operations pushed to `Vec<HloOp>` — never compiled or executed | 🟠 HIGH |
| `estimate_flops` is hardcoded arithmetic | 🟡 MEDIUM |

---

### 1.4 `rvv_simd` — RISC-V Vector Kernels
**File**: [lib.rs](file:///home/xavkal/xdev/runux-ai-runtime/crates/rvv_simd/src/lib.rs) (851 lines, 44 functions, 13 tests)

| Finding | Severity |
|---------|:--------:|
| `matmul_rvv_f32` contains **real** RVV inline assembly (FP32 only) | ✅ REAL |
| INT4/INT8/FP8 kernels (`dequant_matmul_q4`, `fused_dot_q8`, `fused_dot_fp8`) are purely scalar loops | 🟠 HIGH |
| `VectorLength::detect()` has TODO to read `vlenb` CSR — uses compile-time feature flags instead | 🟡 MEDIUM |
| Documentation claims "32 FP32 or 128 INT8 elements per vector instruction" but only FP32 has real assembly | 🟠 HIGH |

**Reality**: 1 out of ~8 SIMD kernels has real RVV assembly. The rest are scalar fallbacks.

---

### 1.5 `arena_mem` — Memory Allocator
**File**: [lib.rs](file:///home/xavkal/xdev/runux-ai-runtime/crates/arena_mem/src/lib.rs) (565 lines, 35 functions, 8 tests)

| Finding | Severity |
|---------|:--------:|
| `BumpAllocator` is a real, functional bump allocator ✅ | ✅ REAL |
| Documentation claims "memory-mapped model weights (GGUF zero-copy)" — zero `mmap` calls exist | 🟠 HIGH |
| `PagedKvCache` is metadata/accounting only — no actual memory pages managed | 🟠 HIGH |

---

### 1.6 `ai_runtime` — Core Types
**File**: [lib.rs](file:///home/xavkal/xdev/runux-ai-runtime/crates/ai_runtime/src/lib.rs) (870 lines, 35 functions, 9 tests)

| Finding | Severity |
|---------|:--------:|
| `HardwareCaps` for K1/K3 are fully hardcoded — no OS/hardware probing | 🟡 MEDIUM |
| VRAM estimation formulas are analytical, not actual measurements | 🟡 MEDIUM |
| Type system and trait definitions are well-designed | ✅ REAL |

---

## Part 2: Optimization Layer

### 2.1 `flash_attention` — FlashAttention-2
**File**: [lib.rs](file:///home/xavkal/xdev/runux-ai-runtime/crates/flash_attention/src/lib.rs) (558 lines, 14 functions, 6 tests)

| Finding | Severity |
|---------|:--------:|
| Documentation claims "RISC-V vector loads/stores" — implementation is purely scalar `for dd in 0..d` loops | 🔴 CRITICAL |
| `estimate_memory` computes theoretical savings via formula, not actual measurements | 🟡 MEDIUM |
| Tiling logic (block decomposition) is real and correct ✅ | ✅ REAL |
| Online softmax implementation is real ✅ | ✅ REAL |

**Reality**: The algorithmic structure (tiled attention with online softmax) is correctly implemented in scalar Rust. The "7.36× speedup" claim is formula-derived.

---

### 2.2 `turbo_quant` — PolarQuant KV-Cache
**File**: [lib.rs](file:///home/xavkal/xdev/runux-ai-runtime/crates/turbo_quant/src/lib.rs) (649 lines, 24 functions, 7 tests)

| Finding | Severity |
|---------|:--------:|
| "Random orthogonal rotation" generates independent scalars via `xorshift64` — **not an orthogonal matrix** | 🔴 CRITICAL |
| Test `test_qjl_inner_product_preservation` explicitly skips correctness: *"Just ensure it produces a finite result"* | 🔴 CRITICAL |
| Compression ratio is a hardcoded formula | 🟡 MEDIUM |
| Scalar quantize/dequantize logic is functional | ✅ REAL |

**Reality**: The core mathematical claim (norm-preserving orthogonal rotation) is **not implemented correctly**. The random number generator produces arbitrary scalars, not an orthogonal transformation.

---

### 2.3 `speculative` — Speculative Decoding
**File**: [lib.rs](file:///home/xavkal/xdev/runux-ai-runtime/crates/speculative/src/lib.rs) (604 lines, 26 functions, 8 tests)

| Finding | Severity |
|---------|:--------:|
| Entire crate operates on pre-computed logits — no actual model execution | 🟡 MEDIUM |
| Rejection sampling algorithm is mathematically correct ✅ | ✅ REAL |
| Carbon-aware K scaling logic is implemented ✅ | ✅ REAL |
| Speedup metrics derived from acceptance rate formulas, not measured | 🟡 MEDIUM |

---

### 2.4 `mlgo_advisor` — MLGO Tiling Advisor
**File**: [lib.rs](file:///home/xavkal/xdev/runux-ai-runtime/crates/mlgo_advisor/src/lib.rs) (709 lines, 30 functions, 13 tests)

| Finding | Severity |
|---------|:--------:|
| `InliningCostModel` uses hardcoded float arrays for weights/biases — **not a trained ML model** | 🔴 CRITICAL |
| `MatmulPerformancePredictor` similarly uses hardcoded heuristics | 🔴 CRITICAL |
| `TilingAdvisor` is analytical formula-based, not hardware-profiled | 🟠 HIGH |
| Claims "ML-guided" and "learned" in documentation | 🔴 CRITICAL |

---

### 2.5 `gguf_loader` — GGUF File Parser
**File**: [lib.rs](file:///home/xavkal/xdev/runux-ai-runtime/crates/gguf_loader/src/lib.rs) (834 lines, 45 functions, 6 tests)

| Finding | Severity |
|---------|:--------:|
| Array parser for >1M elements reads 1,000 and returns — **file pointer misaligned for all subsequent reads** | 🔴 CRITICAL |
| GGUF magic/version parsing is real ✅ | ✅ REAL |
| Metadata key-value parsing is real ✅ | ✅ REAL |

---

### 2.6 `framework_bridge` — TF/PyTorch Bridge
**File**: [lib.rs](file:///home/xavkal/xdev/runux-ai-runtime/crates/framework_bridge/src/lib.rs) (354 lines, 16 functions, 8 tests)

| Finding | Severity |
|---------|:--------:|
| `runux_capabilities()` returns hardcoded bitmask claiming RISC-V + GPU support | 🔴 CRITICAL |
| Backends 1 (RISC-V) and 3 (GPU) in `runux_matmul`/`runux_flash_attention` fall through to error | 🔴 CRITICAL |
| CPU/TPU-sim backends (0, 2) delegate to scalar code | 🟡 MEDIUM |

---

### 2.7 `transformer` — Transformer Layers
**File**: [lib.rs](file:///home/xavkal/xdev/runux-ai-runtime/crates/transformer/src/lib.rs) (766 lines, 38 functions, 11 tests)

| Finding | Severity |
|---------|:--------:|
| Uses `rvv_simd::matmul_scalar_f32` (scalar fallback) not the RVV assembly version | 🟢 LOW |
| RoPE, GQA, SwiGLU implementations are real reference code ✅ | ✅ REAL |

---

### 2.8 `tokenizer` — BPE Tokenizer
**File**: [lib.rs](file:///home/xavkal/xdev/runux-ai-runtime/crates/tokenizer/src/lib.rs) (477 lines, 20 functions, 7 tests)

| Finding | Severity |
|---------|:--------:|
| Minimal but real byte-level BPE implementation | ✅ REAL |

---

## Part 3: Simulation & Infrastructure

### 3.1 `sim_bench` — Benchmark Runner
**Files**: [lib.rs](file:///home/xavkal/xdev/runux-ai-runtime/crates/sim_bench/src/lib.rs), [framework_comparison.rs](file:///home/xavkal/xdev/runux-ai-runtime/crates/sim_bench/src/framework_comparison.rs), [tpu_bench.rs](file:///home/xavkal/xdev/runux-ai-runtime/crates/sim_bench/src/tpu_bench.rs)

| Finding | Severity |
|---------|:--------:|
| `baseline_measurements` contains **massive hardcoded matrices** of fabricated TPS, TTFT, MXU for PyTorch/JAX/vLLM/RunuX | 🔴 CRITICAL |
| TPU benchmarks compute MXU utilization via formula: `0.38 + ((k % 128) / 10000.0)` | 🔴 CRITICAL |
| FlashAttention memory metrics derived from theoretical bandwidth | 🟠 HIGH |
| `tpu_bench.rs` has **0 tests** | 🟠 HIGH |
| Synthetic attention uses `approx_sin`/`approx_cos` instead of real embeddings | 🟢 LOW |

---

### 3.2 `sim_inference` — Inference Simulation
**File**: [lib.rs](file:///home/xavkal/xdev/runux-ai-runtime/crates/sim_inference/src/lib.rs) (566 lines)

| Finding | Severity |
|---------|:--------:|
| Weights generated from deterministic RNG (`WeightGen::new(42)`) — never loaded from files | 🟠 HIGH |
| Speculative decoding acceptance rates mocked with dummy draft logits | 🟠 HIGH |

---

### 3.3 `sim_train` — Training Simulation
**File**: [lib.rs](file:///home/xavkal/xdev/runux-ai-runtime/crates/sim_train/src/lib.rs) (1,049 lines)

| Finding | Severity |
|---------|:--------:|
| Loss decay hardcoded: `1.0 / (1.0 + step * 0.01)` to spoof convergence | 🟠 HIGH |
| LoRA training uses random inputs/targets, not actual model data | 🟠 HIGH |

---

### 3.4 `perf_model` — Roofline Model
**File**: [lib.rs](file:///home/xavkal/xdev/runux-ai-runtime/crates/perf_model/src/lib.rs) (652 lines)

| Finding | Severity |
|---------|:--------:|
| `HardwareSpec` values fully hardcoded (GFLOPS, bandwidth) | 🟡 MEDIUM |
| Token cost estimations are pure arithmetic (FLOPs / bytes) | 🟡 MEDIUM |

---

### 3.5 `power_monitor` — Energy & CO₂
**File**: [lib.rs](file:///home/xavkal/xdev/runux-ai-runtime/crates/power_monitor/src/lib.rs) (487 lines)

| Finding | Severity |
|---------|:--------:|
| No real energy sensors accessed — all computed from `watts / tokens_per_second` | 🟡 MEDIUM |
| Faked TPS inputs (`q4_tps = 8.0`, `spec_tps = 20.0`) drive the reports | 🟡 MEDIUM |

---

### 3.6 Hardware Drivers

#### `k3_a100` — SpacemiT K3 Driver
**File**: [lib.rs](file:///home/xavkal/xdev/runux-ai-runtime/crates/k3_a100/src/lib.rs) (85 lines, 3 functions, **0 tests**)

| Finding | Severity |
|---------|:--------:|
| `A100Context::new` hardcodes `vlen_1024_active: true` — no CPU feature detection | 🔴 CRITICAL |
| `matmul_fp8` falls back to `emulate_fp8_matmul` which **fills output with zeros** | 🔴 CRITICAL |

#### `gpu_compute` — PowerVR GPU
**File**: [lib.rs](file:///home/xavkal/xdev/runux-ai-runtime/crates/gpu_compute/src/lib.rs) (54 lines, 2 functions, **0 tests**)

| Finding | Severity |
|---------|:--------:|
| `GpuContext::new` mocks initialization (`ready: true`) | 🔴 CRITICAL |
| `embedding_lookup` returns `Ok(())` without any computation | 🔴 CRITICAL |

---

### 3.7 Other Infrastructure Crates

| Crate | Finding | Severity |
|-------|---------|:--------:|
| `federated` | FedAvg algorithm is structurally real; no real network I/O | 🟡 MEDIUM |
| `ai_bridge` | `runux_detect_hardware` returns mocked K1 profile (TODO to read CSRs) | 🟡 MEDIUM |
| `sched_fair` | `requires!`/`ensures!` macros are empty; `schedule()` is empty; **0 tests** | 🟡 MEDIUM |

---

## Part 4: Python, SymBrain & Infrastructure

### 4.1 SymBrain v4 Inference Server
**File**: `symbrain_v4/inference_server_v4.py`

| Finding | Severity |
|---------|:--------:|
| Default fallback `SimulationEngine` intercepts queries via regex/exact match and returns **hardcoded perfect mathematical answers** | 🔴 CRITICAL |
| Fabricated PFC routing metrics (σ_ded, complexity, MCTS) presented as neural router output | 🔴 CRITICAL |

---

### 4.2 Evaluation & Benchmarks
**Files**: `eval/benchmark_runner.py`, `eval/run_french_concours_benchmarks.py`

| Finding | Severity |
|---------|:--------:|
| Benchmark runner sends problems whose answers are hardcoded in the server → **rigged 100% accuracy** | 🔴 CRITICAL |
| Script admits: *"realized statistical performance is equivalent to our dry-run verification harness"* | 🔴 CRITICAL |
| Wilson-score confidence intervals computed on fabricated data | 🟠 HIGH |

---

### 4.3 TPU Benchmark Script
**File**: `scripts/tpu_llm_bench.py`

| Finding | Severity |
|---------|:--------:|
| Runs CPU `numpy.dot` but prints hardcoded values: `runux_tps = 56500.0`, `runux_tflops = 195.4` | 🔴 CRITICAL |
| Published as if measured on physical MXU hardware | 🔴 CRITICAL |

---

### 4.4 CI/CD Pipeline
**File**: `.github/workflows/qemu-inference.yml`

| Finding | Severity |
|---------|:--------:|
| Test failures suppressed: `\|\| echo "⚠️ Some QEMU tests may timeout"` followed by forced `✅ validated` | 🟠 HIGH |
| CI will always appear green regardless of test results | 🟠 HIGH |

---

### 4.5 Lean 4 Formal Proofs
**Files**: `spec/RunuxSpec/Basic.lean`, `spec/RunuxSpec/PFCRouter.lean`

| Finding | Severity |
|---------|:--------:|
| Proofs use `sorry` tactic — **Lean compiler skips actual proof obligation** | 🟠 HIGH |
| Theorem statements exist but are not actually proven | 🟠 HIGH |
| Cryptographic "certificate" hashes reference unproven theorems | 🟠 HIGH |

---

### 4.6 Deployment
**File**: `deploy/deploy.sh`

| Finding | Severity |
|---------|:--------:|
| Edge tier defaults to `SIMULATION_MODE=true` | 🟡 MEDIUM |
| 70B and 122B model deployments immediately exit | 🟠 HIGH |

---

### 4.7 CLI Binary
**File**: [main.rs](file:///home/xavkal/xdev/runux-ai-runtime/src/main.rs) (295 lines)

| Finding | Severity |
|---------|:--------:|
| Prints hardcoded performance comparison tables to stdout | 🔴 CRITICAL |
| Emits fake `AUTORESEARCH_METRIC` JSON logs for upstream consumption | 🔴 CRITICAL |

---

## Part 5: Feature Implementation Status Matrix

### Legend
- ✅ **Real** — Functional implementation with correct algorithms
- ⚠️ **Partial** — Structure exists but key parts are simulated/incomplete
- 🔶 **Simulated** — Computes expected results via formulas, no real execution
- ❌ **Stub/Fake** — Empty, hardcoded, or broken implementation
- 🚫 **Not Started** — Feature described in docs but no code exists

| # | Feature (as claimed) | Status | Evidence |
|---|---------------------|:------:|----------|
| 1 | `no_std` Rust workspace compiles | ⚠️ Partial | Lockfile collision prevents clean build |
| 2 | Accelerator trait + HAL dispatch | ✅ Real | Well-designed trait system |
| 3 | CpuBackend (scalar reference) | ✅ Real | Correct scalar implementations |
| 4 | TPU PJRT C API bindings | ❌ Fake | Zero FFI, CPU vectors only |
| 5 | StableHLO graph builder | 🔶 Simulated | String output, not MLIR bytecode |
| 6 | RVV FP32 matmul assembly | ✅ Real | Genuine `core::arch::asm!` |
| 7 | RVV INT4/INT8/FP8 kernels | ❌ Fake | Scalar loops, no assembly |
| 8 | RVV hardware detection (vlenb CSR) | ❌ Stub | Feature flags only |
| 9 | FlashAttention-2 tiled algorithm | ✅ Real | Correct tiling + online softmax (scalar) |
| 10 | FlashAttention vectorized kernels | ❌ Fake | Scalar `for` loops |
| 11 | PolarQuant orthogonal rotation | ❌ Fake | xorshift64 scalars ≠ orthogonal matrix |
| 12 | QJL error correction | ⚠️ Partial | Structure exists, correctness untested |
| 13 | Speculative rejection sampling | ✅ Real | Mathematically correct algorithm |
| 14 | Carbon-aware K scaling | ✅ Real | Formula-based scaling implemented |
| 15 | MLGO "ML-guided" cost models | ❌ Fake | Hardcoded float arrays, not learned |
| 16 | GGUF v3 parser | ⚠️ Partial | Broken for arrays >1M elements |
| 17 | BPE tokenizer | ✅ Real | Minimal but correct |
| 18 | Transformer (RoPE, GQA, SwiGLU) | ✅ Real | Reference scalar implementations |
| 19 | BumpAllocator | ✅ Real | Functional with alignment |
| 20 | PagedKvCache | 🔶 Simulated | Metadata only, no actual pages |
| 21 | GGUF mmap zero-copy | 🚫 Not Started | No mmap calls exist |
| 22 | SpacemiT K3 A100 driver | ❌ Fake | FP8 matmul fills zeros |
| 23 | PowerVR GPU compute | ❌ Fake | Empty stub (54 lines total) |
| 24 | Framework bridge FFI | ❌ Fake | Claims backends not implemented |
| 25 | Federated learning (FedAvg) | 🔶 Simulated | Algorithm correct, no network I/O |
| 26 | CO₂ per-token tracking | 🔶 Simulated | Formula-based, no real sensors |
| 27 | Roofline performance model | 🔶 Simulated | Hardcoded HW specs |
| 28 | LoRA training pipeline | 🔶 Simulated | Random data, hardcoded loss decay |
| 29 | SymBrain v4 inference engine | ❌ Fake | Regex-matched hardcoded answers |
| 30 | Benchmark suite (framework comparison) | ❌ Fake | Hardcoded result matrices |
| 31 | TPU real hardware benchmarks | ❌ Fake | numpy.dot + hardcoded metrics |
| 32 | Lean 4 formal proofs | ❌ Fake | `sorry` stubs, not proven |
| 33 | CI/CD pipeline | ❌ Fake | Failure suppression, forced green |
| 34 | Edge-cloud speculative protocol | 🚫 Not Started | No network transport |
| 35 | SETI-Fed P2P volunteer swarm | 🚫 Not Started | Only local simulation |
| 36 | MLGO RL compiler | 🚫 Not Started | No LLVM IR harvesting |
| 37 | Apple Silicon Metal/MPS | ❌ Fake | Scalar CPU loops |
| 38 | `sched_fair` scheduling | ❌ Stub | Empty `schedule()` function |

### Summary Counts

| Status | Count | Percentage |
|--------|:-----:|:----------:|
| ✅ Real | 9 | 23.7% |
| ⚠️ Partial | 3 | 7.9% |
| 🔶 Simulated | 6 | 15.8% |
| ❌ Fake/Stub | 15 | 39.5% |
| 🚫 Not Started | 5 | 13.2% |

---

## Part 6: Test Quality Assessment

| Metric | Value |
|--------|-------|
| Total `#[test]` functions across workspace | ~170 |
| Crates with **0 tests** | `k3_a100`, `gpu_compute`, `tpu_bench`, `sched_fair` |
| Tests that validate real correctness | ~40% |
| Tests that validate mock/simulated paths | ~55% |
| Tests explicitly skipping correctness | ~5% (turbo_quant QJL test) |
| Integration tests | None |
| Hardware-in-the-loop tests | None |
| Formal proofs completed | 0 (all use `sorry`) |

---

## Part 7: Documentation vs Reality Gap

| Documented Claim | Reality |
|-----------------|---------|
| "88% MXU Occupancy on TPU v5e" | Computed from formula `0.38 + k%128/10000` |
| "3.12× decode speedup validated" | Hardcoded in CLI output |
| "65× HBM traffic reduction" | Theoretical calculation |
| "Safe Rust bindings for PJRT C API" | Zero C FFI code |
| "1024-bit VLEN RVV operations" | Only FP32 has real assembly |
| "Orthogonal rotation preserves norm" | xorshift64 ≠ orthogonal matrix |
| "ML-guided systolic tiling" | Hardcoded float arrays |
| "99.92% GSM8K accuracy" | Regex-matched hardcoded answers |
| "Lean 4 cryptographic certificates" | `sorry`-stubbed proofs |
| "CI validated on QEMU RISC-V" | Failures suppressed |

---

*End of Audit Report*
