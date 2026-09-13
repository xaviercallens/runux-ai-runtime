<!-- ======================================================================= -->
<!-- RunuX AI Runtime — Pull Request Submission Template                       -->
<!-- Copyright (c) 2026 Xavier Callens / Socrate AI. All Rights Reserved.      -->
<!-- ======================================================================= -->

## Description of Changes

<!-- Provide a clear, detailed summary of what this PR introduces or fixes. -->

## Related Issues / RFCs

<!-- Link related issues, e.g. Closes #123 -->

## Mathematical & Physical Verification (If Applicable)

- [ ] **Conservation Laws**: Verified energy dissipation $\frac{dE}{dt} \le 0$ and incompressibility $\|\nabla \cdot u\|_\infty \le 10^{-12}$.
- [ ] **Lean 4 Formal Verification**: All formal specs in `spec/RunuxSpec/LeanFlow/` build with `lake build RunuxSpec` with **Zero Axioms** (no `sorry`).
- [ ] **Interval Arithmetic**: Certified numerical bounds computed via `crates/interval_arith` and signed by `crates/cert_forge`.
- [ ] **PhysicsGuard Compliance**: Tested against `runux.navier_stokes_advisor.PhysicsGuard`.

## Quality Gates & Verification Checklist

- [ ] **Rust Unit Tests**: `cargo test --workspace` passes with 0 failures.
- [ ] **Python Test Suite**: `pytest tests/ -p no:zarr` passes with 0 errors.
- [ ] **Clippy & Formatting**: `cargo clippy --workspace` passes without warnings.
- [ ] **Documentation**: Any newly introduced algorithms, types, or endpoints are documented.
- [ ] **Intellectual Property & Licensing**: All new source files contain the mandatory copyright notice:
  ```rust
  // Copyright (c) 2026 Xavier Callens / Socrate AI. All Rights Reserved.
  // Licensed under LicenseRef-RunuX-Commercial.
  ```

## Contributor Agreement

By submitting this pull request, I confirm that:
1. My contribution is original work created by me, or I have the necessary rights to submit it.
2. I grant the repository authors full rights to review, test, modify, and merge this contribution under the repository's licensing terms.
3. This PR does not introduce unverified axioms or violate the Zero Axiom Policy.
