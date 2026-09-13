# Engineering Brief: Navier-Stokes Numerical Proof via RunuX AI & LeanFlow
**Target System**: Claude Code / Autonomous Agent Process  
**Repository**: `runux-ai-runtime`  
**Active Branch**: `feature/leanflow-navier-stokes-proof` (or merged `main`)  
**Status**: Scale-Out HPC Kernel, Formal Lean 4 Verification, and Untrusted AI Guardrails Deployed  

---

## 1. Mission Overview & Mathematical Strategy

You are tasked with advancing the computer-assisted proof of **3D Incompressible Navier-Stokes Regularity** on the torus $\mathbb{T}^3 = [0, 2\pi]^3$:
$$\partial_t u + (u \cdot \nabla)u = -\nabla p + \nu \Delta u + f, \quad \nabla \cdot u = 0$$

### The Generate-and-Verify (Oracle) Paradigm
1. **The Untrusted Optimizer (Python / SymBrain)**: Recommends adaptive spectral mesh cutoffs $M$, preconditioners, and time steps $\Delta t$.
2. **The Untrusted Calculator (Rust HPC Kernel)**: Computes high-dimensional Fourier-Galerkin spectral truncations up to order $M$, propagating rigorous floating-point interval bounds across $\mathcal{O}(M^6)$ triadic convolution evaluations with zero-allocation hot loops.
3. **The Air-Gapped Bridge (`cert_forge`)**: Serializes the verified trapping region, enstrophy bounds, and contraction rate into `proof_certificate.json`.
4. **The Trusted Judge (Lean 4 / LeanFlow)**: Ingests rational interval bounds (`Rat`) and formally checks the algebraic boundary inequalities via proof by reflection under a strict **Zero Axiom Policy**.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       RunuX AI Advisor (Python)                         │
│   - Truncation Order M Heuristic    - CFL Adaptive Step Δt             │
│   - Preconditioner Guessing         - PhysicsGuard Conservation Checks  │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │ (Untrusted Suggestions)
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                   HPC Computational Kernel (Rust)                       │
│   - crates/interval_arith : Directed f64 ULP widening                  │
│   - crates/navier_stokes  : Strict SoA layout, O(M⁶) triadic conv      │
│   - crates/cert_forge     : Air-gapped JSON certificate export         │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │ proof_certificate.json
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    LeanFlow Formal Verifier (Lean 4)                    │
│   - spec/RunuxSpec/LeanFlow/ : RatInterval, RustBounds                 │
│   - lake build RunuxSpec     : Machine-checked regularity theorem       │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Repository Layout & Key File Map

```
runux-ai-runtime/
├── crates/
│   ├── interval_arith/              # Rigorous f64 interval arithmetic
│   │   ├── Cargo.toml               # no_std compatible
│   │   └── src/lib.rs               # Interval, IntervalFieldBounds, ULP widening, soft_sqrt
│   ├── navier_stokes/               # Spectral Galerkin solver
│   │   ├── Cargo.toml               # Rayon parallel feature
│   │   └── src/lib.rs               # SpectralField (SoA), WavevectorTable, workspaces, ODE state
│   └── cert_forge/                  # Air-gapped certificate serialization
│       ├── Cargo.toml               # serde, serde_json
│       └── src/lib.rs               # ProofCertificate, IntervalSerde, validate_consistency
├── runux/
│   ├── __init__.py                  # Public exports
│   └── navier_stokes_advisor.py     # NavierStokesAdvisor, PhysicsGuard, fallback logic
├── spec/
│   ├── RunuxSpec.lean               # Top-level import hub
│   └── RunuxSpec/
│       └── LeanFlow/                # Lean 4 formal certification subsystem
│           ├── Interval.lean        # RatInterval over ℚ (Rat)
│           ├── Certificate.lean     # RustBounds parser and verification predicates
│           ├── NavierStokes.lean    # FourierMode, wavenumber_sq, viscous decay
│           ├── InvariantRegion.lean # Trapping region checks & contraction lemmas
│           ├── SpectralDecay.lean   # Algebraic energy decay rate verifier
│           └── Regularity.lean      # Top-level regularity theorem (Zero Axiom Policy)
└── tests/
    └── test_navier_stokes_advisor.py# 21 unit tests for AI advisor and PhysicsGuard
```

---

## 3. Strict Architectural Rules & Invariants

You **MUST** adhere to the following 5 rules in all generated code:

### Rule 1: Structure-of-Arrays (Strict SoA) — No AoS
* **Forbidden**: `struct FourierMode { kx: i32, ky: i32, kz: i32, re: f64, im: f64 }` stored in a `Vec<FourierMode>`.
* **Required**: Separate contiguous flat vectors:
  ```rust
  pub struct SpectralField {
      pub re: Vec<f64>,
      pub im: Vec<f64>,
      pub n_modes: usize,
      pub truncation_m: usize,
  }
  ```
* **Rationale**: Guarantees L1/L2 cache alignment and enables LLVM SIMD auto-vectorization (AVX2 / AVX-512 `vfmadd231pd`).

### Rule 2: Zero-Allocation Hot Loops
* **Forbidden**: `Vec::new()`, `Box`, `.clone()`, or dynamic allocations inside convolution loops or ODE steppers.
* **Required**: Allocate all scratchpads once at startup (`ConvolutionWorkspace`, `OdeStepperState`). Compute kernels must accept only mutable slices:
  ```rust
  pub fn triadic_convolution_kernel(
      field: &SpectralField,
      waves: &WavevectorTable,
      out_re: &mut [f64],
      out_im: &mut [f64],
      workspace: &mut ConvolutionWorkspace,
  )
  ```

### Rule 3: Lock-Free Rayon Parallelization
* **Forbidden**: `std::thread`, `Mutex`, `RwLock` in compute paths (they destroy multi-core scaling).
* **Required**: Parallelize over disjoint slices using `rayon::par_chunks_mut()` and deterministic tree reductions (`rayon::iter::reduce`).

### Rule 4: Air-Gapped Certificate & Zero Axiom Policy
* **Forbidden**: Passing raw C-FFI pointers or shared memory between Rust and Lean 4.
* **Required**: Rust serializes interval bounds to `proof_certificate.json`. Lean 4 parses bounds as exact rationals (`Rat`) and validates all inequalities via reflection.
* **Invariant**: Theorem signatures in Lean 4 formulate Rust bounds as logical `Prop` hypotheses. `#print axioms` must stay completely clean.

### Rule 5: AI as an Untrusted Optimizer with Physical Fallback
* **Required**: All suggestions from `NavierStokesAdvisor` must pass through `PhysicsGuard`:
  - Energy conservation: $\Delta E / E \le 10^{-10}$
  - Incompressibility: $\|\nabla \cdot u\|_{L^2} \le 10^{-12}$
  - CFL condition: $u_{\max} \Delta t / \Delta x \le \text{CFL}_{\max}$
* **Fallback**: If an AI suggestion causes the residual norm ratio to exceed `max_residual_ratio` (default `1.1`), the Rust kernel immediately rejects the suggestion and falls back to the deterministic exact solver.

---

## 4. Operational Runbook for Claude Code

### Step 1: Environment Verification
Ensure Rust and Lean 4 toolchains are present and functional:
```bash
# Verify Rust workspace compilation
cargo check --workspace

# Run all HPC and certificate unit tests (31 tests)
cargo test -p interval_arith -p navier_stokes -p cert_forge

# Verify Lean 4 compilation (17 targets)
export PATH="$HOME/.elan/bin:$PATH"
(cd spec && lake build RunuxSpec)

# Verify Python AI module and fallback tests (21 tests)
PYTHONPATH=. pytest -p no:zarr tests/test_navier_stokes_advisor.py
```

### Step 2: Running the Spectral Galerkin Kernel (Rust)
Instantiate a `GalerkinSystem` and advance the spectral integration:
```rust
use navier_stokes::{GalerkinSystem, viscous_decay, compute_energy_spectrum};
use interval_arith::Interval;

// Initialize system with truncation order M = 10 (21³ = 9,261 modes)
let mut system = GalerkinSystem::new(10, 0.001, 32);

// Time integration step
let dt = 0.0005;
viscous_decay(&mut system.field, &system.waves, system.viscosity, dt);
let current_energy = compute_energy_spectrum(&system.field, &system.waves);
```

### Step 3: Exporting the Proof Certificate
When a trapping region $[L_k, U_k]$ is rigorously bounded, serialize the certificate:
```rust
use cert_forge::{ProofCertificate, IntervalSerde, CURRENT_SCHEMA_VERSION};

let cert = ProofCertificate {
    schema_version: CURRENT_SCHEMA_VERSION,
    truncation_m: 10,
    viscosity_rational: "1/1000".to_string(),
    time_start: IntervalSerde { lo: 0.0, hi: 0.0 },
    time_end: IntervalSerde { lo: 1.0, hi: 1.0 },
    h1_norm_bound: IntervalSerde { lo: 0.0, hi: 12.5 },
    enstrophy_bound: IntervalSerde { lo: 0.0, hi: 4.8 },
    mode_decay_alpha: IntervalSerde { lo: 2.1, hi: 2.3 },
    mode_decay_coefficients_lo: vec![0.1; 9261],
    mode_decay_coefficients_hi: vec![0.2; 9261],
    invariant_region_lo: vec![-1.0; 9261],
    invariant_region_hi: vec![1.0; 9261],
    residual_bound: IntervalSerde { lo: 0.0, hi: 1e-11 },
    contraction_rate: IntervalSerde { lo: 0.4, hi: 0.85 },
    n_time_steps: 2000,
    computed_at_utc: "2026-09-13T00:00:00Z".to_string(),
    runux_version: "0.4.0".to_string(),
    cpu_cores_used: 128,
    initial_data_hash: "3a7b9c...".to_string(),
};

// Validate consistency before writing
cert.validate_consistency().expect("Certificate validation failed");
cert.to_file(std::path::Path::new("proof_certificate.json")).unwrap();
```

### Step 4: Validating in Lean 4
Run the formal check:
```bash
export PATH="$HOME/.elan/bin:$PATH"
cd spec && lake build RunuxSpec
```
Inspect the Lean 4 environment:
* In `spec/RunuxSpec/LeanFlow/Regularity.lean`, the theorem `regularity` ensures that any valid `RustBounds` instance guarantees finite $H^1$ Sobolev norms for all $t \in [t_{\text{start}}, t_{\text{end}}]$.

### Step 5: Incorporating AI Optimization (Python)
Use `NavierStokesAdvisor` outside the physics loop:
```python
from runux import NavierStokesAdvisor, PhysicsGuard

advisor = NavierStokesAdvisor(cfl_limit=0.5, max_residual_ratio=1.1)

# 1. Propose adaptive time step
step_hint = advisor.suggest_time_step(
    current_dt=0.001, cfl_number=0.85, max_velocity=2.0, viscosity=0.001
)

# 2. PhysicsGuard sanity check
if PhysicsGuard.check_cfl_stability(step_hint.suggested_dt, dx=0.01, max_velocity=2.0, cfl_limit=0.5):
    # 3. Apply and validate residual
    val = advisor.validate_suggestion("time_step", step_hint, actual_residual=0.005, previous_residual=0.006)
    if not val.accepted:
        # Fall back to exact solver
        dt = 0.0005
```

---

## 5. Immediate Next Engineering Tasks

1. **Full $\mathcal{O}(M^6)$ Kernel Implementation**: Implement the inner loops in `crates/navier_stokes/src/lib.rs` for `triadic_convolution_kernel` using `rayon::par_chunks_mut()` across target wavevector chunks.
2. **De-aliased Pseudospectral Acceleration**: Evaluate 2/3 de-aliasing rule or 3/2 zero-padding using the in-place Cooley-Tukey FFT primitive from `examples/macos_m2_math_benchmark/` to accelerate the convolution from $\mathcal{O}(M^6)$ to $\mathcal{O}(M^3 \log M)$ while retaining rigorous interval error bounds on the aliasing tail.
3. **JSON Ingestion in Lean 4**: Add a JSON reflection parser in `spec/RunuxSpec/LeanFlow/Certificate.lean` to deserialize `proof_certificate.json` directly into `RustBounds` using Lean 4's `Lean.Json` parser.
