# Contributing to RunuX AI Runtime & LeanFlow

Thank you for your interest in contributing to the **RunuX AI Runtime** and the **LeanFlow Formal Navier-Stokes Project**!

This repository is actively developed and maintained by **Xavier Callens** and the **Socrate AI** team. To protect the integrity of the scientific simulations, formal mathematical proofs, and high-performance computing kernels, all contributions must strictly adhere to the guidelines outlined below.

---

## 1. Branch Protection & Contribution Model

To prevent accidental breakage or tampering with the codebase:
- **`main` is a Protected Branch:** Direct pushes, force-pushes (`git push --force`), and branch deletions on `main` are strictly blocked.
- **Pull Request Requirement:** All contributions—from internal collaborators and external community members alike—must be submitted via **Pull Requests (PRs)** against the `main` branch.
- **Mandatory Author Review:** Per [`.github/CODEOWNERS`](.github/CODEOWNERS), all PRs require explicit review and approval from repository authors (`@xaviercallens`) before merging.
- **Automated CI Gates:** Every PR triggers GitHub Actions CI checks that must pass completely before a PR is eligible for merge.

```
       ┌─────────────────────────────────────────────────────────┐
       │               RUNUX CONTRIBUTION WORKFLOW               │
       └─────────────────────────────────────────────────────────┘
                                    │
       1. Fork & Branch             ▼  (e.g., feature/my-enhancement)
       2. Local Development         ▼  (Rust SoA / Python / Lean 4)
       3. Run Quality Gates         ▼  (cargo test && lake build)
       4. Open Pull Request         ▼  (Fill .github/pull_request_template.md)
       5. Automated CI Checks       ▼  (GitHub Actions test suite)
       6. Author Code Review        ▼  (Mandatory review by @xaviercallens)
       7. Merge to Main             ▼  (Squash or linear merge)
```

---

## 2. Setting Up Your Development Environment

### Prerequisites
- **Rust Toolchain:** Nightly Rust (with RISC-V targets if testing cross-compilation).
- **Lean 4:** Version 4.30.0+ with `lake` build tool.
- **Python:** Python 3.10+ with `numpy`, `scipy`, `matplotlib`, `pytest`.
- **OpenFOAM (Optional for CFD benchmarks):** OpenFOAM v1912+ (`icoFoam`, `blockMesh`).

### Local Verification Commands
Before submitting a PR, verify all components pass locally:
```bash
# 1. Test all Rust crates (interval arithmetic, navier-stokes, cert-forge)
cargo test --workspace

# 2. Check Rust formatting and lints
cargo clippy --workspace -- -D warnings

# 3. Verify Lean 4 formal specifications (Zero Axiom Policy)
cd spec
lake build RunuxSpec
cd ..

# 4. Run Python unit tests
pytest tests/ -p no:zarr -v
```

---

## 3. Core Architectural & Scientific Invariants

When contributing to RunuX and LeanFlow, you must adhere to the following non-negotiable rules:

### A. Data-Oriented Design (Strict SoA)
- In performance-critical kernels (such as `crates/navier_stokes`), **never** use Arrays of Structures (AoS) for spectral fields.
- Use flat, contiguous Structure of Arrays (SoA):
  ```rust
  pub struct SpectralField {
      pub re: Vec<f64>,
      pub im: Vec<f64>,
      pub n_modes: usize,
      pub truncation_m: usize,
  }
  ```
- Prioritize L1/L2 cache locality and auto-vectorization.

### B. The Air-Gapped Formal Boundary
- **Never** pass raw C-FFI pointers or mutate state across the Rust $\leftrightarrow$ Lean 4 boundary.
- Communication between the Rust high-performance numerical kernel and Lean 4 formal certificates is mediated exclusively via cryptographically hashed JSON certificate files (`crates/cert_forge`).

### C. The Zero Axiom Policy in Lean 4
- Formal specifications in `spec/RunuxSpec/LeanFlow/` must compile with **zero axioms** and **no `sorry`**.
- Any theorem or lemma must be formally grounded in Mathlib or proven from first principles.

### D. Conservation Law Invariants (`PhysicsGuard`)
- Any algorithmic modification that updates fluid states must be validated by `runux.navier_stokes_advisor.PhysicsGuard`:
  - **Incompressibility:** $\|\nabla \cdot u\|_{L^\infty} \le 10^{-12}$ (or machine zero $\sim 10^{-32}$ for spectral solvers).
  - **Energy Dissipation:** $\frac{dE}{dt} \le 0$ unconditionally for unforced flow.
  - **CFL Stability:** Time steps must strictly satisfy the Courant-Friedrichs-Lewy criterion.

---

## 4. Submitting a Pull Request

1. Create a dedicated feature branch with a descriptive name:
   ```bash
   git checkout -b feature/my-feature-name
   ```
2. Commit your changes following [Conventional Commits](https://www.conventionalcommits.org/):
   - `feat(...)`: A new feature or capability.
   - `fix(...)`: A bug fix.
   - `perf(...)`: A performance improvement.
   - `docs(...)`: Documentation or paper updates.
   - `test(...)`: Adding or updating tests.
3. Push to your fork:
   ```bash
   git push origin feature/my-feature-name
   ```
4. Open a Pull Request on GitHub against `main`. Fill in all sections of the [Pull Request Template](.github/pull_request_template.md).
5. Ensure all CI checks pass and address any feedback from `@xaviercallens`.

---

## 5. Licensing & Intellectual Property

RunuX AI Runtime is proprietary scientific software:
- **Copyright (c) 2026 Xavier Callens / Socrate AI. All Rights Reserved.**
- **Patent Pending:** INPI Dossier Demande Provisoire de Brevet RunuX (2026).
- All contributed code becomes part of the RunuX codebase and is licensed under `LicenseRef-RunuX-Commercial` unless explicitly designated otherwise.
- Contributors must guarantee that contributions do not contain copyleft (GPL) code or infringe third-party patents.

---
*For questions, collaboration proposals, or academic inquiries, please open a GitHub Issue or reach out via Hugging Face discussions.*
