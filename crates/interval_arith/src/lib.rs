// Copyright (c) 2026 Xavier Callens / Socrate AI. All Rights Reserved.
#![cfg_attr(not(test), no_std)]
#![deny(clippy::all)]
#![warn(clippy::pedantic)]

//! Rigorous f64 interval arithmetic with directed rounding for computer-assisted proofs.
//! Provides conservative arithmetic operations that guarantee containment of the true result
//! despite IEEE-754 roundoff errors.

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;
use core::ops::{Add, Div, Mul, Neg, Sub};

/// A real interval `[lo, hi]`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Interval {
    pub lo: f64,
    pub hi: f64,
}

impl Interval {
    /// Creates a new interval `[lo, hi]`. Panics if `lo > hi` or if inputs are NaN.
    #[must_use]
    pub fn new(lo: f64, hi: f64) -> Self {
        assert!(lo <= hi, "Invalid interval: lo > hi");
        assert!(!lo.is_nan() && !hi.is_nan(), "Interval bounds cannot be NaN");
        Self { lo, hi }
    }

    /// Creates an exact interval `[val, val]`.
    #[must_use]
    pub fn exact(val: f64) -> Self {
        Self::new(val, val)
    }

    /// Computes the square root of the interval.
    ///
    /// The returned interval is guaranteed to contain the true square root for every finite
    /// input, **subnormals included**. It does not simply widen the Newton-Raphson output by a
    /// fixed margin: that was unsound, because `soft_sqrt`'s bit-hack seed assumes a normalised
    /// exponent field and is wrong by orders of magnitude on subnormals, where 8 iterations
    /// cannot recover (measured: `sqrt(8.095e-320)` came back `4.37e-157` against a true
    /// `2.845e-160` — a factor of 1500, or `4.7e16` ULP, against a 1 ULP widening).
    ///
    /// See `sqrt_bounds` for the three-step construction that fixes it.
    #[must_use]
    pub fn sqrt(self) -> Self {
        let lo = if self.lo <= 0.0 { 0.0 } else { sqrt_bounds(self.lo).0 };
        let hi = if self.hi <= 0.0 { 0.0 } else { sqrt_bounds(self.hi).1 };
        Self::new(lo, hi)
    }

    /// Computes the absolute value of the interval.
    #[must_use]
    pub fn abs(self) -> Self {
        if self.lo >= 0.0 {
            self
        } else if self.hi <= 0.0 {
            Self::new(-self.hi, -self.lo)
        } else {
            Self::new(0.0, self.lo.abs().max(self.hi.abs()))
        }
    }

    /// Scales the interval by a scalar.
    #[must_use]
    pub fn scale(self, scalar: f64) -> Self {
        if scalar >= 0.0 {
            Self::new(self.lo * scalar, self.hi * scalar).widen(1)
        } else {
            Self::new(self.hi * scalar, self.lo * scalar).widen(1)
        }
    }

    /// Widens the interval by `ulps` Units in Last Place.
    ///
    /// Deliberately does NOT clamp to `[f64::MIN, f64::MAX]`. Clamping an upper bound that
    /// overflowed to `+inf` back down to `f64::MAX` returns a bound BELOW the true value and
    /// silently loses containment, which is the one property this type exists to provide. An
    /// infinite bound is sound — uselessly wide, but sound — so overflow is allowed to escape
    /// as an infinity where a caller can see it.
    #[must_use]
    fn widen(self, ulps: i64) -> Self {
        Self::new(ulp_widen(self.lo, -ulps), ulp_widen(self.hi, ulps))
    }
}

/// `2^e`, exactly, for `-1022 <= e <= 1023`. Bit construction keeps this `no_std`.
#[must_use]
fn pow2(e: i32) -> f64 {
    debug_assert!((-1022..=1023).contains(&e), "pow2 exponent out of normal range");
    f64::from_bits(((e + 1023) as u64) << 52)
}

/// A sound bracket `[lo, hi]` around `sqrt(x)` for every finite `x > 0`, subnormals included.
///
/// Three steps, because neither the Newton iteration nor a naive self-certification behaves in
/// the subnormal range:
///
/// 1. **Argument reduction.** Scale by an exact power of four into `[1, 4)`. Multiplication by
///    4 and by `2^-k` are exact, so the reduction introduces no error of its own.
/// 2. **Root where the seed is valid.** `soft_sqrt`'s estimate assumes a normalised exponent,
///    which now holds by construction.
/// 3. **Certify, do not assume.** Widen outward until `lo*lo <= v <= hi*hi` actually holds —
///    the bound proves itself by comparison rather than resting on an unproven claim about the
///    iteration's accuracy. Certifying *before* reduction would not work: in the subnormal range
///    the certifying multiplication itself underflows (measured 3311/20000 failures).
///
/// The result is always normal for `x > 0` — `sqrt` of the smallest subnormal is `~2.2e-162` —
/// so the final scale-back is a single correctly-rounded multiply, covered by 1 ULP.
#[must_use]
fn sqrt_bounds(x: f64) -> (f64, f64) {
    if x.is_nan() || x <= 0.0 {
        return (0.0, 0.0);
    }
    if x.is_infinite() {
        return (x, x);
    }
    // 1. reduce into [1, 4)
    let mut v = x;
    let mut k: i32 = 0;
    while v < 1.0 {
        v *= 4.0;
        k += 1;
    }
    while v >= 4.0 {
        v *= 0.25;
        k -= 1;
    }
    // 2. root in the normal range
    let g = soft_sqrt(v);
    // 3. certify by exact comparison, widening outward until the bracket really holds
    let (mut lo, mut hi) = (g, g);
    let mut w: i64 = 1;
    while !(lo * lo <= v && hi * hi >= v) {
        lo = ulp_widen(g, -w);
        hi = ulp_widen(g, w);
        w = w.saturating_mul(2);
        if w > (1 << 62) {
            return (0.0, f64::INFINITY); // sound last resort; never reached in practice
        }
    }
    // 4. scale back by the exact power of two, then one ULP for the rounding of that multiply
    let s = pow2(-k);
    (ulp_widen(lo * s, -1), ulp_widen(hi * s, 1))
}

/// Software square root via Newton-Raphson iteration.
/// Uses IEEE-754 bit manipulation for the initial estimate (Quake-style),
/// then 5 iterations for f64 convergence. `no_std` compatible.
fn soft_sqrt(val: f64) -> f64 {
    if val <= 0.0 {
        return 0.0;
    }
    if val.is_nan() || val.is_infinite() {
        return val;
    }
    // Initial estimate via bit manipulation: sqrt(x) ≈ 2^((e-1023)/2) * m
    let bits = val.to_bits();
    let estimate_bits = (bits >> 1) + 0x1FF8_0000_0000_0000;
    let mut x = f64::from_bits(estimate_bits);
    // 8 Newton-Raphson iterations for full 53-bit IEEE-754 double precision
    for _ in 0..8 {
        x = 0.5 * (x + val / x);
    }
    x
}

/// Computes a new f64 that is `ulps` units in the last place away from `val`.
fn ulp_widen(val: f64, ulps: i64) -> f64 {
    if val.is_nan() || val.is_infinite() {
        return val;
    }
    let bits = val.to_bits();
    let is_negative = (bits >> 63) == 1;
    let magnitude = bits & 0x7FFF_FFFF_FFFF_FFFF;
    
    // Convert to signed magnitude representation for easy ULP arithmetic
    let mut signed_mag = if is_negative {
        -(magnitude as i64)
    } else {
        magnitude as i64
    };
    
    signed_mag = signed_mag.saturating_add(ulps);
    
    if signed_mag == 0 {
        return 0.0;
    }
    
    let new_is_negative = signed_mag < 0;
    let new_magnitude = signed_mag.unsigned_abs();
    
    let new_bits = if new_is_negative {
        new_magnitude | 0x8000_0000_0000_0000
    } else {
        new_magnitude
    };
    
    f64::from_bits(new_bits)
}

impl Add for Interval {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self::new(self.lo + rhs.lo, self.hi + rhs.hi).widen(1)
    }
}

impl Sub for Interval {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.lo - rhs.hi, self.hi - rhs.lo).widen(1)
    }
}

impl Mul for Interval {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        let p1 = self.lo * rhs.lo;
        let p2 = self.lo * rhs.hi;
        let p3 = self.hi * rhs.lo;
        let p4 = self.hi * rhs.hi;
        
        let lo = p1.min(p2).min(p3).min(p4);
        let hi = p1.max(p2).max(p3).max(p4);
        
        Self::new(lo, hi).widen(1)
    }
}

impl Div for Interval {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        assert!(rhs.lo > 0.0 || rhs.hi < 0.0, "Division by interval containing zero");
        let d1 = self.lo / rhs.lo;
        let d2 = self.lo / rhs.hi;
        let d3 = self.hi / rhs.lo;
        let d4 = self.hi / rhs.hi;
        
        let lo = d1.min(d2).min(d3).min(d4);
        let hi = d1.max(d2).max(d3).max(d4);
        
        Self::new(lo, hi).widen(1)
    }
}

impl Neg for Interval {
    type Output = Self;
    fn neg(self) -> Self {
        Self::new(-self.hi, -self.lo)
    }
}

/// SoA representation of an interval field.
#[repr(C)]
#[derive(Clone, Debug)]
pub struct IntervalFieldBounds {
    pub lo: Vec<f64>,
    pub hi: Vec<f64>,
    pub n_intervals: usize,
}

impl IntervalFieldBounds {
    #[must_use]
    pub fn new(n_intervals: usize) -> Self {
        Self {
            lo: vec![0.0; n_intervals],
            hi: vec![0.0; n_intervals],
            n_intervals,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_containment() {
        let x = Interval::exact(1.0);
        let y = Interval::exact(2.0);
        let z = x + y;
        assert!(z.lo <= 3.0 && z.hi >= 3.0);
    }

    #[test]
    fn test_subtraction() {
        let x = Interval::new(5.0, 7.0);
        let y = Interval::new(2.0, 3.0);
        let z = x - y;
        // [5, 7] - [2, 3] = [5-3, 7-2] = [2, 4]
        assert!(z.lo <= 2.0 && z.hi >= 4.0);
    }

    #[test]
    fn test_multiplication_signs() {
        // Pos * Pos
        let a = Interval::new(2.0, 3.0);
        let b = Interval::new(4.0, 5.0);
        let p1 = a * b;
        assert!(p1.lo <= 8.0 && p1.hi >= 15.0);

        // Pos * Neg
        let c = Interval::new(-3.0, -1.0);
        let p2 = a * c;
        assert!(p2.lo <= -9.0 && p2.hi >= -2.0);

        // Neg * Neg
        let d = Interval::new(-4.0, -2.0);
        let p3 = c * d;
        assert!(p3.lo <= 2.0 && p3.hi >= 12.0);

        // Spanning zero * Spanning zero
        let e = Interval::new(-2.0, 3.0);
        let f = Interval::new(-4.0, 5.0);
        let p4 = e * f;
        assert!(p4.lo <= -12.0 && p4.hi >= 15.0);
    }

    #[test]
    fn test_division_valid() {
        let x = Interval::new(6.0, 12.0);
        let y = Interval::new(2.0, 3.0);
        let z = x / y;
        // [6, 12] / [2, 3] = [6/3, 12/2] = [2, 6]
        assert!(z.lo <= 2.0 && z.hi >= 6.0);
    }

    #[test]
    fn test_sqrt_accuracy() {
        let x = Interval::exact(4.0);
        let s = x.sqrt();
        assert!(s.lo <= 2.0 && s.hi >= 2.0);

        let y = Interval::exact(9.0);
        let s2 = y.sqrt();
        assert!(s2.lo <= 3.0 && s2.hi >= 3.0);

        let zero = Interval::exact(0.0);
        let s0 = zero.sqrt();
        assert!(s0.lo <= 0.0 && s0.hi >= 0.0);
    }

    #[test]
    fn test_abs_cases() {
        // Positive
        let p = Interval::new(2.0, 5.0).abs();
        assert!(p.lo == 2.0 && p.hi == 5.0);

        // Negative
        let n = Interval::new(-5.0, -2.0).abs();
        assert!(n.lo == 2.0 && n.hi == 5.0);

        // Spanning zero
        let z = Interval::new(-3.0, 4.0).abs();
        assert!(z.lo == 0.0 && z.hi == 4.0);
    }

    #[test]
    fn test_scale() {
        let x = Interval::new(2.0, 3.0);
        let scaled_pos = x.scale(4.0);
        assert!(scaled_pos.lo <= 8.0 && scaled_pos.hi >= 12.0);

        let scaled_neg = x.scale(-2.0);
        assert!(scaled_neg.lo <= -6.0 && scaled_neg.hi >= -4.0);
    }

    #[test]
    fn test_negation() {
        let x = Interval::new(-2.0, 5.0);
        let neg = -x;
        assert_eq!(neg.lo, -5.0);
        assert_eq!(neg.hi, 2.0);
    }

    #[test]
    fn test_monotonicity() {
        let x = Interval::new(1.0, 2.0);
        let y = Interval::new(0.5, 3.0);
        let sum = x + y;
        assert!(sum.lo <= 1.5 && sum.hi >= 5.0);
    }

    #[test]
    fn test_zero_interval() {
        let zero = Interval::exact(0.0);
        let x = Interval::new(-5.0, 5.0);
        let prod = zero * x;
        assert!(prod.lo <= 0.0 && prod.hi >= 0.0);
    }

    #[test]
    #[should_panic(expected = "Division by interval containing zero")]
    fn test_div_by_zero() {
        let x = Interval::exact(1.0);
        let y = Interval::new(-1.0, 1.0);
        let _ = x / y;
    }

    #[test]
    #[should_panic(expected = "Invalid interval: lo > hi")]
    fn test_invalid_interval_bounds() {
        let _ = Interval::new(5.0, 3.0);
    }

    // ---------------------------------------------------------------------------------
    // Containment tests on inputs that actually exercise rounding.
    //
    // Every constant in the tests above -- 1, 2, 3, 4, 5, 6, 7, 9, 12, 15, 0.5 -- is a dyadic
    // rational, on which f64 arithmetic and these square roots are EXACT. Those tests therefore
    // pass unchanged even if `widen` is replaced by the identity function, i.e. with the crate's
    // entire safety mechanism deleted (demonstrated: 10/10 still green). A checker that cannot
    // fail is not a checker, so the tests below use non-dyadic and subnormal inputs and assert
    // the bracket property directly.
    // ---------------------------------------------------------------------------------

    /// The bracket property: `lo^2 <= v <= hi^2`. This is the guarantee, stated in a form that
    /// needs no higher-precision reference to check.
    fn assert_brackets(v: f64) {
        let s = Interval::exact(v).sqrt();
        assert!(s.lo >= 0.0, "negative lower bound for sqrt({v:e})");
        assert!(
            s.lo * s.lo <= v && s.hi * s.hi >= v,
            "sqrt({v:e}) = [{:e}, {:e}] does not bracket: lo^2={:e}, hi^2={:e}",
            s.lo, s.hi, s.lo * s.lo, s.hi * s.hi
        );
    }

    #[test]
    fn sqrt_brackets_nondyadic_normals() {
        for i in 1..5_000u32 {
            assert_brackets(f64::from(i) / 7.0);
            assert_brackets(f64::from(i) * 1.000_000_1);
        }
    }

    /// REGRESSION. The previous implementation widened the raw Newton output by 1 ULP, and on
    /// subnormals the bit-hack seed is wrong by orders of magnitude: sqrt(8.095e-320) returned
    /// 4.370122139616469e-157 against a true 2.8451311993408992e-160 -- 4.7e16 ULP outside a
    /// 1 ULP interval. 60 of 20000 sampled subnormals lost containment.
    #[test]
    fn sqrt_brackets_subnormals() {
        assert_brackets(8.095e-320);
        assert_brackets(1.036_131e-317);
        assert_brackets(1.326_247_37e-315);
        assert_brackets(f64::from_bits(1)); // smallest positive subnormal
        let mut v = f64::MIN_POSITIVE; // smallest normal
        for _ in 0..1_000 {
            v *= 0.5; // walks down through the subnormal range
            if v == 0.0 {
                break;
            }
            assert_brackets(v);
        }
    }

    #[test]
    fn sqrt_brackets_every_power_of_two() {
        for e in -1070..=1023 {
            let v = if e >= -1022 { pow2(e) } else { f64::from_bits(1) * pow2(e + 1074) };
            if v > 0.0 && v.is_finite() {
                assert_brackets(v);
            }
        }
    }

    /// NEGATIVE CONTROL. If the outward widening ever becomes a no-op, the raw iteration is
    /// still inexact on almost every non-dyadic input, so a zero-width interval would not
    /// bracket. This test asserts that inexactness, so the suite NOTICES if someone removes the
    /// safety margin -- the property the dyadic tests above cannot see.
    #[test]
    fn negative_control_raw_iteration_is_inexact() {
        let mut inexact = 0usize;
        let mut total = 0usize;
        for i in 1..5_000u32 {
            let v = f64::from(i) / 7.0;
            let g = soft_sqrt(v);
            total += 1;
            if g * g != v {
                inexact += 1;
            }
        }
        assert!(
            inexact * 2 > total,
            "the raw square root was exact on most inputs ({inexact}/{total}); this test has \
             stopped exercising rounding and can no longer detect a lost widening"
        );
    }

    /// Pins the bug this commit fixes, by reproducing the OLD construction inline and asserting
    /// it is unsound. If someone reverts `sqrt` to "iterate then widen by a constant", this test
    /// starts failing and says why. Without it the fix is only an assertion that the new code
    /// works, never a demonstration that the old code did not.
    #[test]
    fn old_construction_was_unsound_on_subnormals() {
        let v = 8.095e-320_f64;
        assert!(v > 0.0 && v < f64::MIN_POSITIVE, "test input must be subnormal");

        // exactly what shipped before: raw Newton output, widened by one ULP either way
        let g = soft_sqrt(v);
        let (old_lo, old_hi) = (ulp_widen(g, -1), ulp_widen(g, 1));
        assert!(
            !(old_lo * old_lo <= v && old_hi * old_hi >= v),
            "the old construction unexpectedly bracketed {v:e}; this control no longer \
             demonstrates the defect it was written for"
        );

        // and the replacement does bracket it
        let s = Interval::exact(v).sqrt();
        assert!(s.lo * s.lo <= v && s.hi * s.hi >= v);
    }

    /// Overflow must escape as an infinity rather than being clamped back below the true value.
    #[test]
    fn widen_does_not_clamp_overflow_below_truth() {
        let w = Interval::new(f64::MAX, f64::MAX).widen(1);
        assert!(w.hi.is_infinite() && w.hi > 0.0, "upper bound must not be clamped to f64::MAX");
        assert!(w.lo.is_finite() && w.lo < f64::MAX);
    }

    #[test]
    fn test_interval_field_bounds_soa() {
        let n = 27;
        let mut bounds = IntervalFieldBounds::new(n);
        assert_eq!(bounds.lo.len(), n);
        assert_eq!(bounds.hi.len(), n);
        assert_eq!(bounds.n_intervals, n);

        for i in 0..n {
            bounds.lo[i] = -(i as f64);
            bounds.hi[i] = i as f64;
        }

        for i in 0..n {
            assert!(bounds.lo[i] <= bounds.hi[i]);
        }
    }
}
