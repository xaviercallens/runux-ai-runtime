-- Copyright (c) 2026 Xavier Callens / Socrate AI Lab. All Rights Reserved.
-- SPDX-License-Identifier: LicenseRef-RunuX-Commercial
--
-- spec/RunuxSpec/LeanFlow/Certificate.lean — Certificate parser structure

import RunuxSpec.LeanFlow.Interval

namespace RunuxSpec.LeanFlow

/-- The certificate parsed from the Rust bounds generator. -/
structure RustBounds where
  truncation_m : Nat
  h_m_pos : truncation_m > 0
  viscosity : Rat
  h_visc_pos : (0 : Rat) < viscosity
  h1_norm_bound : RatInterval
  enstrophy_bound : RatInterval
  mode_decay_alpha : RatInterval
  n_modes : Nat
  h_n_modes : n_modes = (2 * truncation_m + 1)^3
  invariant_lo : Fin n_modes → Rat
  invariant_hi : Fin n_modes → Rat
  residual_bound : RatInterval
  contraction_rate : RatInterval

def verify_invariant_bounds (cert : RustBounds) : Prop :=
  ∀ k, (0 : Rat) ≤ cert.invariant_lo k ∧ cert.invariant_lo k ≤ cert.invariant_hi k

def verify_contraction (cert : RustBounds) : Prop :=
  cert.contraction_rate.hi < 1

def verify_residual (cert : RustBounds) : Prop :=
  cert.residual_bound.is_nonneg

def verify_h1_bound (cert : RustBounds) : Prop :=
  cert.h1_norm_bound.is_positive

def verify_enstrophy (cert : RustBounds) : Prop :=
  cert.enstrophy_bound.is_nonneg

def verify_certificate (cert : RustBounds) : Prop :=
  verify_invariant_bounds cert ∧
  verify_contraction cert ∧
  verify_residual cert ∧
  verify_h1_bound cert ∧
  verify_enstrophy cert

end RunuxSpec.LeanFlow
