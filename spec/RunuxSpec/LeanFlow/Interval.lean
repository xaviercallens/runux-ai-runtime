-- Copyright (c) 2026 Xavier Callens / Socrate AI Lab. All Rights Reserved.
-- SPDX-License-Identifier: LicenseRef-RunuX-Commercial
--
-- spec/RunuxSpec/LeanFlow/Interval.lean — Rational interval arithmetic

namespace RunuxSpec.LeanFlow

/-- A closed interval of rational numbers [lo, hi]. -/
structure RatInterval where
  lo : Rat
  hi : Rat
  h_valid : lo ≤ hi

/-- Checks if a rational number x is within the interval I. -/
def RatInterval.contains (I : RatInterval) (x : Rat) : Prop := I.lo ≤ x ∧ x ≤ I.hi

/-- Width of the interval. -/
def RatInterval.width (I : RatInterval) : Rat := I.hi - I.lo

/-- Midpoint of the interval. -/
def RatInterval.midpoint (I : RatInterval) : Rat := (I.lo + I.hi) / 2

/-- Strictly positive interval. -/
def RatInterval.is_positive (I : RatInterval) : Prop := (0 : Rat) < I.lo

/-- Non-negative interval. -/
def RatInterval.is_nonneg (I : RatInterval) : Prop := (0 : Rat) ≤ I.lo

theorem RatInterval.lo_le_hi (I : RatInterval) : I.lo ≤ I.hi := I.h_valid

theorem RatInterval.contains_midpoint (I : RatInterval) : I.contains I.midpoint := by
  sorry -- PROOF OUTLINE: Requires proving I.lo <= (I.lo + I.hi) / 2 and (I.lo + I.hi) / 2 <= I.hi given I.lo <= I.hi.

theorem RatInterval.width_nonneg (I : RatInterval) : (0 : Rat) ≤ I.width := by
  sorry -- PROOF OUTLINE: Follows from I.h_valid : I.lo <= I.hi.

end RunuxSpec.LeanFlow
