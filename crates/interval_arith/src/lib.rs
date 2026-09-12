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
    /// Uses software sqrt (Newton-Raphson) to remain `no_std` compatible.
    #[must_use]
    pub fn sqrt(self) -> Self {
        let lo = if self.lo < 0.0 { 0.0 } else { soft_sqrt(self.lo) };
        let hi = soft_sqrt(self.hi);
        Self::new(lo, hi).widen(1)
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
    #[must_use]
    fn widen(self, ulps: i64) -> Self {
        let lo_w = ulp_widen(self.lo, -ulps);
        let hi_w = ulp_widen(self.hi, ulps);
        // Clamp to f64::MIN / f64::MAX to prevent blowing up to Infinity if we are close,
        // but typically ULP widening won't reach Inf unless already huge.
        Self::new(
            lo_w.clamp(f64::MIN, f64::MAX),
            hi_w.clamp(f64::MIN, f64::MAX),
        )
    }
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
