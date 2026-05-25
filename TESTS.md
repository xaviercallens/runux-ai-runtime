# RunuX-AI Verification & Testing Specification
*Unified Test Suite & Quality Assurance Matrix*

---

## 1. Quality Assurance Hierarchy

The RunuX-AI runtime employs a five-tier testing and verification strategy to ensure high performance and mathematical correctness in resource-constrained, bare-metal (`no_std`) environments:

```text
┌──────────────────────────────────────────────┐
│  Tier 5: Lean 4 Formal Verification Proofs    │ ← Mathematical safety & correctness
├──────────────────────────────────────────────┤
│  Tier 4: Chaos & Edge Network Emulation      │ ← Packet drops, latency spikes, load test
├──────────────────────────────────────────────┤
│  Tier 3: Roofline & GEMM Performance Sweeps   │ ← Cycle-accurate hardware profiling
├──────────────────────────────────────────────┤
│  Tier 2: Multi-Crate Integration Tests       │ ← End-to-end model execution roundtrips
├──────────────────────────────────────────────┤
│  Tier 1: no_std Unit Testing Correctness      │ ← Zero heap allocation correctness
└──────────────────────────────────────────────┘
```

---

## 2. Tier 1 & 2: Unit and Integration Testing

Every crate in the workspace must pass standard cargo validation on both local development machines and physical/simulated target architectures.

### Execution Commands:
```bash
# Run all unit tests inside the workspace (ensuring single-threaded safety checks)
cargo test --workspace -- --test-threads=1

# Run tests in a specific crate (e.g. turbo_quant)
cargo test -p turbo_quant

# Verify bare-metal compatibility (prevents standard library pollution in no_std)
cargo check --workspace --all-targets
```

### Core Crate Test Matrix:
- **`hal`**:
  - `test_cpu_matmul`: Standard matrix multiplication correctness check against hand-calculated values.
  - `test_tpu_simulator_matmul`: Verifies that the simulated tiled matrix multiplication matches the CPU reference perfectly.
  - `test_tpu_simulator_attention`: Verifies FlashAttention execution with varying head and sequence dimensions.
- **`turbo_quant`**:
  - `test_compress_decompress_roundtrip`: Compresses KV vectors to 8-bit precision and asserts de-quantized reconstructions are within acceptable bounds.
  - `test_qjl_inner_product_preservation`: Asserts that projected random representations can estimate original vector dot products.
- **`speculative`**:
  - `test_modified_rejection_sampling`: Simulates draft and target distributions to verify rejection statistics.
- **`arena_mem`**:
  - `test_bump_alloc_basic`: Verifies O(1) allocation and reset sequences.
  - `test_bump_alloc_oom`: Verifies that allocating beyond bounds returns a clean `None` instead of panicking.
  - `test_paged_kv_lru_eviction`: Validates least-recently-used paging eviction.
- **`sched_fair`**:
  - `test_cfs_ready_queue_sorting`: Verifies static ready queue sorts FFI task pointers correctly in ascending order of `vruntime`.
  - `test_wars_entity_tick_telemetry_scaling`: Validates dynamic WARS scaling coefficients (1.5x BIG core promotion, 2.0x LITTLE core mismatch penalty, and L1 cache miss adaptations).
  - `test_wars_adjust_priorities`: Validates dynamic Userspace RL Advisor uniform scaling adjustments.
- **`SUPERSONIC-Rust Diff-Optimizer Verification`**:
  - `test_supersonic_rust_bounds_check_safety`: Verifies that CodeBERT synthesized safe `unsafe` blocks do not trigger out-of-bounds page faults or memory segmentation violations.
  - `test_supersonic_rust_speedup_assertion`: Asserts that compilation with the diff optimization yields at least a 2.0x performance speedup compared to standard `-C opt-level=3` on representative `rvv_simd` and `sched_fair` mathematical routines.
- **`WARS-Quantum-LTN Simulator Verification`**:
  - `test_wars_quantum_peps_unitary_drift`: Asserts that boundary contractions of a 3D PEPS grid preserve state vector norm with a unitary drift strictly below $1.5 \times 10^{-12}$.
  - `test_wars_quantum_rl_telemetry_routing`: Validates that GEMM/SVD tensor contractions are strictly pinned to BIG vector cores and message-passing to LITTLE cores based on workload telemetry.
  - `test_wars_quantum_polarquant_compression`: Compresses boundary networks down to 3-bit PolarQuant representation and asserts a memory reduction factor of $\ge 50\times$.

---

## 3. Tier 3: Roofline & Performance Benchmarks
*Crate: `sim_bench` & `perf_model`*

Performance tests are designed to profile arithmetic intensity and detect cache-thrashing or unaligned vector register access.

### Performance Target Metrics:
1. **GEMM Roofline Target**: Compute-bound kernels must saturate at $\ge 85\%$ of the physical peak TFLOPS (197 TFLOPS for TPU v5e, 128 GFLOPS for RISC-V K3).
2. **FlashAttention IO Optimization**: Key-Value loading must execute entirely in local scratch space or SRAM, showing $0$ external HBM roundtrips after loading.
3. **Memory Pressure**: Peak memory watermark must remain strictly below 90% of available physical RAM under dynamic multi-sequence loads.

---

## 4. Tier 4: Chaos & Edge Network Emulation
*Crate: `federated`*

For edge clusters (where draft tokens are generated on one node and verified on another), network failures must be handled gracefully:

- **Packet Drop Emulation**: Randomly drop up to 15% of draft token payloads during transmission. The system must recover by resetting the speculative pipeline and falling back to local decoding.
- **Latency Spikes**: Inject artificial delays up to 250ms per token step. The carbon-aware speculative coordinator must automatically throttle $K$ down to minimize network overhead when transmission delay exceeds decoding delay.

---

## 5. Tier 5: Formal Mathematical Verification
*Directory: `spec/`*

We use Lean 4 to formally prove critical correctness and safety properties. Any mathematical claim made in our publications must be formally stated and validated:

- **Arena Memory Allocation Safety**: Prove that the bump-allocator guarantees non-overlapping memory spans.
- **PolarQuant Preservation Bounds**: Prove that orthogonal rotation preserves vector norm (energy conservation).
- **Rejection Sampling Validity**: Prove that the reconstructed token distribution strictly sum-to-one ($\sum_x P(x) = 1.0$) and matches the target.
- **SUPERSONIC-Rust Optimization Boundaries** (Certificate: `CERT-LEAN4-SUPERSONIC-RUST-AC8318F0DCBC`):
  - `SUPERSONIC_Rust_DiffOptimizer_memory_safety_sound`: Proves memory safety bounds are maintained when array bounds checking is eliminated for optimized loops.
  - `SUPERSONIC_Rust_DiffOptimizer_speedup_strictly_positive`: Proves that the synthesized optimization diff guarantees strictly positive speedup.
  - `SUPERSONIC_Rust_DiffOptimizer_bounds_checks_bounded`: Proves the optimization covers a non-zero count of bounds checks (1284 checks).
- **WARS-Quantum-LTN Simulation Boundaries** (Certificate: `CERT-LEAN4-QUANTUM-LTN-B2BBC320607C`):
  - `WARS_Quantum_LogicTensorNetwork_unitary_preservation`: Proves that 3-bit PolarQuant boundary tensor contraction preserves state vector unitary norm.
  - `WARS_Quantum_LogicTensorNetwork_speedup_positive`: Proves that the WARS telemetry-guided SVD/contraction algorithm guarantees strictly positive speedup.
  - `WARS_Quantum_LogicTensorNetwork_qubits_bounded`: Proves that high-dimensional PEPS quantum spin-glass simulations are bounded under safe simulator dimensions.


