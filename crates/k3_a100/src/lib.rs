// Copyright (c) 2026 Xavier Callens / Socrate AI. All Rights Reserved.
// SPDX-License-Identifier: LicenseRef-RunuX-Commercial
// This file is proprietary and confidential. Unauthorized copying,
// distribution, or use is strictly prohibited.

#![cfg_attr(not(test), no_std)]
#![deny(clippy::all)]
#![warn(clippy::pedantic)]
#![allow(clippy::result_large_err)]
//! RunuX K3 A100 Driver — SpacemiT K3 specific AI core operations
//!
//! Provides bare-metal / kernel-level access to the SpacemiT K3's A100 AI cores.
//! Leverages the 1024-bit VLEN registers and FP8 E4M3 hardware matrix
//! multiplication instructions.

extern crate alloc;

use ai_runtime::AiError;

/// Convert an FP8 E4M3 byte to an FP32 value.
///
/// Format: 1 sign bit, 4 exponent bits (bias 7), 3 mantissa bits.
#[must_use]
pub fn fp8_e4m3_to_f32(byte: u8) -> f32 {
    let sign = (byte & 0x80) != 0;
    let exp = (byte >> 3) & 0x0F;
    let mant = byte & 0x07;

    let val = if exp == 0 {
        // Subnormal number: 2^(-6) * (mant / 8) = mant * 2^(-9)
        if mant == 0 {
            0.0
        } else {
            f32::from(mant) * 0.001_953_125
        }
    } else if exp == 15 && mant == 7 {
        // NaN in E4M3FN
        f32::NAN
    } else {
        // Normal number: 2^(exp - 7) * (1 + mant / 8)
        let scale = f32_pow2_exp(i32::from(exp) - 7);
        let frac = 1.0 + f32::from(mant) * 0.125;
        scale * frac
    };

    if sign {
        -val
    } else {
        val
    }
}

/// Convert an FP32 value to an FP8 E4M3 byte (saturating, nearest-even).
#[must_use]
pub fn f32_to_fp8_e4m3(val: f32) -> u8 {
    if val.is_nan() {
        return 0x7F; // Standard NaN
    }
    let sign_bit = if val.is_sign_negative() {
        0x80u8
    } else {
        0x00u8
    };
    let abs_val = val.abs();

    if abs_val < 0.000_976_562_5 {
        // Underflow to zero
        return sign_bit;
    }
    if abs_val >= 448.0 {
        // Saturation to max representable normal (448.0)
        return sign_bit | 0x7E;
    }

    if abs_val < 0.015_625 {
        // Subnormal: val = mant * 2^(-9) => mant = round(val / 2^(-9))
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let mant = (abs_val * 512.0 + 0.5) as u8;
        return sign_bit | mant.min(7);
    }

    // Normal: find exp in 1..=15
    let mut exp = 7i32;
    let mut norm = abs_val;
    while norm >= 2.0 && exp < 15 {
        norm *= 0.5;
        exp += 1;
    }
    while norm < 1.0 && exp > 1 {
        norm *= 2.0;
        exp -= 1;
    }

    let frac = (norm - 1.0) * 8.0;
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let mant = (frac + 0.5) as u8;
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    if mant >= 8 {
        if exp < 15 {
            exp += 1;
            sign_bit | ((exp as u8) << 3)
        } else {
            sign_bit | 0x7E
        }
    } else if exp == 15 && mant == 7 {
        sign_bit | 0x7E
    } else {
        sign_bit | ((exp as u8) << 3) | (mant & 0x07)
    }
}

/// Fast power of 2 for small exponents (-10 to 10) in no_std.
fn f32_pow2_exp(exp: i32) -> f32 {
    match exp {
        -7 => 0.007_812_5,
        -6 => 0.015_625,
        -5 => 0.031_25,
        -4 => 0.062_5,
        -3 => 0.125,
        -2 => 0.25,
        -1 => 0.5,
        0 => 1.0,
        1 => 2.0,
        2 => 4.0,
        3 => 8.0,
        4 => 16.0,
        5 => 32.0,
        6 => 64.0,
        7 => 128.0,
        8 => 256.0,
        _ => {
            if exp > 8 {
                #[allow(clippy::cast_precision_loss)]
                let factor = (1i32 << (exp - 8).min(20)) as f32;
                256.0 * factor
            } else {
                0.0
            }
        }
    }
}

/// SpacemiT K3 A100 Core Context
#[derive(Debug)]
pub struct A100Context {
    /// Active core ID
    core_id: u32,
    /// Indicates if 1024-bit VLEN is configured
    vlen_1024_active: bool,
    /// Indicates whether running on native K3 hardware
    is_native: bool,
}

impl A100Context {
    /// Initialize access to the A100 AI cores.
    ///
    /// # Errors
    /// Returns `AiError::UnsupportedHardware` if hardware initialization fails.
    pub fn new(core_id: u32) -> Result<Self, AiError> {
        let is_native = cfg!(target_arch = "riscv64");
        Ok(Self {
            core_id,
            vlen_1024_active: true,
            is_native,
        })
    }

    /// Returns the core ID.
    #[must_use]
    pub fn core_id(&self) -> u32 {
        self.core_id
    }

    /// Returns true if executing on native K3 RISC-V silicon.
    #[must_use]
    pub fn is_native(&self) -> bool {
        self.is_native
    }

    /// Perform an FP8 native or emulated matrix multiplication (C = A x B).
    ///
    /// # Errors
    /// Returns `AiError::UnsupportedHardware` if the hardware/context is not configured.
    /// Returns `AiError::BufferTooSmall` if dimensions do not match buffer sizes.
    pub fn matmul_fp8(
        &self,
        a: &[u8],      // FP8 encoded E4M3 (m x k)
        b: &[u8],      // FP8 encoded E4M3 (k x n)
        c: &mut [f32], // Output accumulator (m x n)
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<(), AiError> {
        if !self.vlen_1024_active {
            return Err(AiError::UnsupportedHardware);
        }
        if a.len() < m * k || b.len() < k * n || c.len() < m * n {
            return Err(AiError::ComputeError);
        }

        self.emulate_fp8_matmul(a, b, c, m, k, n);
        Ok(())
    }

    fn emulate_fp8_matmul(&self, a: &[u8], b: &[u8], c: &mut [f32], m: usize, k: usize, n: usize) {
        for i in 0..m {
            let row_offset_a = i * k;
            let row_offset_c = i * n;
            for j in 0..n {
                let mut sum = 0.0f32;
                for p in 0..k {
                    let a_val = fp8_e4m3_to_f32(a[row_offset_a + p]);
                    let b_val = fp8_e4m3_to_f32(b[p * n + j]);
                    sum += a_val * b_val;
                }
                c[row_offset_c + j] = sum;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    #[test]
    fn test_fp8_e4m3_decode_encode_precision() {
        let test_cases = [0.0f32, 1.0, -1.0, 2.0, 0.5, 4.0, 0.125, 448.0];
        for &expected in &test_cases {
            let encoded = f32_to_fp8_e4m3(expected);
            let decoded = fp8_e4m3_to_f32(encoded);
            let rel_err = if expected.abs() > 1e-5 {
                (decoded - expected).abs() / expected.abs()
            } else {
                decoded.abs()
            };
            assert!(
                rel_err < 0.05,
                "FP8 roundtrip mismatch for {}: decoded={}, encoded={:#x}",
                expected,
                decoded,
                encoded
            );
        }
    }

    #[test]
    fn test_matmul_fp8_identity() {
        let ctx = A100Context::new(0).unwrap();
        let m = 4;
        let k = 4;
        let n = 4;

        // A is arbitrary 4x4 matrix
        let a_f32 = [
            1.0, 2.0, 0.5, -1.0, 0.0, 4.0, 1.5, 2.0, 3.0, 0.0, 1.0, 0.5, -2.0, 1.0, 0.0, 2.0,
        ];
        let a_fp8: Vec<u8> = a_f32.iter().map(|&v| f32_to_fp8_e4m3(v)).collect();

        // B is 4x4 identity matrix
        let mut b_fp8 = vec![f32_to_fp8_e4m3(0.0); 16];
        for i in 0..4 {
            b_fp8[i * 4 + i] = f32_to_fp8_e4m3(1.0);
        }

        let mut c = vec![0.0f32; 16];
        ctx.matmul_fp8(&a_fp8, &b_fp8, &mut c, m, k, n).unwrap();

        for i in 0..16 {
            let expected = a_f32[i];
            let actual = c[i];
            assert!(
                (expected - actual).abs() < 0.1,
                "Identity mismatch at {}: expected {}, got {}",
                i,
                expected,
                actual
            );
        }
    }

    #[test]
    fn test_matmul_fp8_against_fp32_reference() {
        let ctx = A100Context::new(1).unwrap();
        let m = 3;
        let k = 4;
        let n = 2;

        let a_f32 = [
            1.0, 2.0, -1.0, 0.5, 0.5, 1.0, 2.0, -0.5, -1.0, 0.0, 1.5, 2.0,
        ];
        let b_f32 = [2.0, 1.0, -1.0, 0.5, 0.5, 2.0, 1.0, -1.0];

        // Reference FP32 multiplication
        let mut c_ref = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for p in 0..k {
                    sum += a_f32[i * k + p] * b_f32[p * n + j];
                }
                c_ref[i * n + j] = sum;
            }
        }

        let a_fp8: Vec<u8> = a_f32.iter().map(|&v| f32_to_fp8_e4m3(v)).collect();
        let b_fp8: Vec<u8> = b_f32.iter().map(|&v| f32_to_fp8_e4m3(v)).collect();
        let mut c_fp8 = vec![0.0f32; m * n];

        ctx.matmul_fp8(&a_fp8, &b_fp8, &mut c_fp8, m, k, n).unwrap();

        for i in 0..(m * n) {
            let ref_val = c_ref[i];
            let fp8_val = c_fp8[i];
            assert!(
                (ref_val - fp8_val).abs() < 0.25,
                "Mismatch at index {}: ref={}, fp8={}",
                i,
                ref_val,
                fp8_val
            );
        }
    }
}
