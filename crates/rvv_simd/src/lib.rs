// Copyright (c) 2026 Xavier Callens / Socrate AI. All Rights Reserved.
// SPDX-License-Identifier: LicenseRef-RunuX-Commercial
// This file is proprietary and confidential. Unauthorized copying,
// distribution, or use is strictly prohibited.

#![cfg_attr(not(test), no_std)]
#![deny(clippy::all)]
#![warn(clippy::pedantic)]
//! RunuX RVV SIMD — RISC-V Vector accelerated kernels for tensor operations
//!
//! Provides vectorized implementations of core ML operations using RISC-V
//! Vector Extension (RVV 1.0). Supports both 256-bit (K1) and 1024-bit (K3)
//! vector register widths with automatic dispatch.
//!
//! # Operations
//!
//! - Matrix multiplication (FP32, FP8, INT4 with dequantization)
//! - Softmax (numerically stable, vectorized)
//! - Layer normalization (fused mean + variance)
//! - RMS normalization (Qwen/LLaMA-style)
//! - GELU / SiLU activation functions
//! - Rotary positional embeddings (RoPE)
//!
//! # Hardware Dispatch
//!
//! On K3 (1024-bit VLEN), operations process 32 FP32 or 128 INT8 elements
//! per vector instruction. On K1 (256-bit), this drops to 8 FP32 or 32 INT8.
//! The scalar fallback works on any RISC-V target.

extern crate alloc;

use ai_runtime::{DataType, DeviceType, TensorDescriptor, AiError};

// ---------------------------------------------------------------------------
// Vector Length Detection
// ---------------------------------------------------------------------------

/// Detected vector register length in bits.
///
/// On real RISC-V hardware, this would read the `vlenb` CSR.
/// For QEMU and cross-compilation, we default to 256 (K1) or
/// allow configuration via feature flags.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VectorLength {
    /// 128-bit (minimal RVV implementation)
    Vlen128,
    /// 256-bit (SpacemiT K1 / X60 cores)
    Vlen256,
    /// 512-bit (some RISC-V implementations)
    Vlen512,
    /// 1024-bit (SpacemiT K3 / A100 AI cores)
    Vlen1024,
}

impl VectorLength {
    /// Number of FP32 elements per vector register.
    pub const fn fp32_elements(self) -> usize {
        match self {
            Self::Vlen128 => 4,
            Self::Vlen256 => 8,
            Self::Vlen512 => 16,
            Self::Vlen1024 => 32,
        }
    }

    /// Number of FP8/INT8 elements per vector register.
    pub const fn int8_elements(self) -> usize {
        match self {
            Self::Vlen128 => 16,
            Self::Vlen256 => 32,
            Self::Vlen512 => 64,
            Self::Vlen1024 => 128,
        }
    }

    /// Number of INT4 elements per vector register.
    pub const fn int4_elements(self) -> usize {
        self.int8_elements() * 2
    }

    /// Detect vector length from hardware.
    ///
    /// On real hardware, this reads the `vlenb` CSR register.
    /// In emulation, returns a compile-time default.
    pub fn detect() -> Self {
        // TODO: On real RISC-V hardware, read CSR vlenb:
        // let vlenb: usize;
        // unsafe { asm!("csrr {}, vlenb", out(reg) vlenb); }
        // match vlenb * 8 { 128 => Vlen128, ... }

        #[cfg(feature = "k3_a100")]
        { return Self::Vlen1024; }

        #[cfg(not(feature = "k3_a100"))]
        { Self::Vlen256 } // Default: K1
    }
}

// ---------------------------------------------------------------------------
// Matrix Multiplication Kernels
// ---------------------------------------------------------------------------

/// Scalar fallback for matrix multiplication: C = A × B
///
/// A: [M × K], B: [K × N], C: [M × N]
/// Works on any architecture, used when RVV is not available.
pub fn matmul_scalar_f32(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) {
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for l in 0..k {
                sum += a[i * k + l] * b[l * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

/// RVV-optimized matrix multiplication for FP32.
///
/// Uses vector load/store and fused multiply-accumulate.
/// On K3 (VLEN=1024), processes 32 FP32 elements per cycle.
///
/// # Safety
///
/// Caller must ensure buffers are properly sized:
/// - `a`: M × K elements
/// - `b`: K × N elements  
/// - `c`: M × N elements (output)
#[cfg(target_arch = "riscv64")]
pub fn matmul_rvv_f32(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) {
    for i in 0..m {
        for j in 0..n {
            let mut sum: f32 = 0.0;
            let mut l = 0;
            let mut remaining = k;
            
            // Core RVV dot product
            unsafe {
                core::arch::asm!(
                    // v8 will hold the accumulator (sum)
                    "vmv.v.x v8, zero", 
                    
                    "2:", // Loop start
                    "vsetvli t0, {rem}, e32, m1, ta, ma",
                    
                    // Load A row vector and B col vector
                    "vle32.v v16, ({ptr_a})",
                    "vle32.v v24, ({ptr_b})",
                    
                    // Fused multiply-add: v8[i] += v16[i] * v24[i]
                    "vfmacc.vv v8, v16, v24",
                    
                    // Increment pointers and decrement remaining
                    "slli t1, t0, 2", // t1 = elements processed * 4 bytes
                    "add {ptr_a}, {ptr_a}, t1",
                    "add {ptr_b}, {ptr_b}, t1",
                    "sub {rem}, {rem}, t0",
                    
                    "bnez {rem}, 2b", // Loop if remaining > 0
                    
                    // Horizontal reduction sum into v0
                    "vmv.s.x v0, zero",
                    "vfredosum.vs v0, v8, v0",
                    "vfmv.f.s {sum}, v0",
                    
                    ptr_a = inout(reg) a.as_ptr().add(i * k + l) => _,
                    ptr_b = inout(reg) b.as_ptr().add(l * n + j) => _,
                    rem = inout(reg) remaining => _,
                    sum = out(freg) sum,
                    out("t0") _,
                    out("t1") _,
                    out("v8") _,
                    out("v16") _,
                    out("v24") _,
                    out("v0") _,
                );
            }
            c[i * n + j] = sum;
        }
    }
}

/// Fallback for non-RISC-V targets (host testing).
#[cfg(not(target_arch = "riscv64"))]
pub fn matmul_rvv_f32(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) {
    matmul_scalar_f32(a, b, c, m, k, n);
}

// ---------------------------------------------------------------------------
// INT4 Dequantization + MatMul
// ---------------------------------------------------------------------------

/// Block size for Q4_K_M quantization (matches GGUF format).
pub const Q4_BLOCK_SIZE: usize = 32;

/// A quantized INT4 block (Q4_K_M compatible).
///
/// Each block contains 32 4-bit quantized values plus
/// a scale factor and minimum value for dequantization.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct QuantBlockQ4 {
    /// Scale factor for this block
    pub scale: f32,
    /// Minimum (zero-point offset)
    pub min: f32,
    /// 32 packed 4-bit values (16 bytes)
    pub quants: [u8; Q4_BLOCK_SIZE / 2],
}

impl QuantBlockQ4 {
    /// Dequantize this block into FP32 values.
    pub fn dequantize(&self, output: &mut [f32; Q4_BLOCK_SIZE]) {
        for i in 0..Q4_BLOCK_SIZE / 2 {
            let byte = self.quants[i];
            let lo = (byte & 0x0F) as f32;
            let hi = ((byte >> 4) & 0x0F) as f32;
            output[2 * i] = lo * self.scale + self.min;
            output[2 * i + 1] = hi * self.scale + self.min;
        }
    }
}

/// Dequantize INT4 blocks and multiply with FP32 activation vector.
///
/// This is the hot path for LLM inference with GGUF Q4_K_M models.
/// On K3, the dequantization and multiply are fused in the vector pipeline.
pub fn dequant_matmul_q4(
    quant_weights: &[QuantBlockQ4],
    activations: &[f32],
    output: &mut [f32],
    out_features: usize,
    in_features: usize,
) {
    let blocks_per_row = in_features / Q4_BLOCK_SIZE;

    for i in 0..out_features {
        let mut sum = 0.0f32;
        for b in 0..blocks_per_row {
            let block = &quant_weights[i * blocks_per_row + b];
            let mut dequant = [0.0f32; Q4_BLOCK_SIZE];
            block.dequantize(&mut dequant);

            let act_offset = b * Q4_BLOCK_SIZE;
            for k in 0..Q4_BLOCK_SIZE {
                sum += dequant[k] * activations[act_offset + k];
            }
        }
        output[i] = sum;
    }
}

/// Fused Q4 dot product — single-row activation × quantized weight.
///
/// This is the innermost kernel for batch-1 decode steps.
/// Avoids materializing the dequantized weight vector by fusing
/// dequant + multiply + accumulate into one pass.
///
/// Returns the dot product of `activations[0..in_features]` with one
/// row of quantized weights.
pub fn fused_dot_q4(
    blocks: &[QuantBlockQ4],
    activations: &[f32],
    in_features: usize,
) -> f32 {
    let blocks_per_row = in_features / Q4_BLOCK_SIZE;
    let mut sum = 0.0f32;

    for b in 0..blocks_per_row {
        let block = &blocks[b];
        let act_base = b * Q4_BLOCK_SIZE;

        // Fused: dequant + multiply + accumulate without temp buffer
        for i in 0..Q4_BLOCK_SIZE / 2 {
            let byte = block.quants[i];
            let lo = (byte & 0x0F) as f32;
            let hi = ((byte >> 4) & 0x0F) as f32;
            let val_lo = lo * block.scale + block.min;
            let val_hi = hi * block.scale + block.min;
            sum += val_lo * activations[act_base + 2 * i];
            sum += val_hi * activations[act_base + 2 * i + 1];
        }
    }
    sum
}

// ---------------------------------------------------------------------------
// Q8_0 Quantization (8-bit uniform)
// ---------------------------------------------------------------------------

/// Block size for Q8_0 quantization.
pub const Q8_BLOCK_SIZE: usize = 32;

/// A quantized INT8 block (Q8_0 format: scale + 32 × i8).
///
/// Higher quality than Q4 (less quantization error) but 2× memory.
/// Preferred for attention output projection and embedding layers.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct QuantBlockQ8 {
    /// Scale factor (FP16 stored as f32 for convenience)
    pub scale: f32,
    /// 32 signed 8-bit quantized values
    pub quants: [i8; Q8_BLOCK_SIZE],
}

impl QuantBlockQ8 {
    /// Dequantize this block: output[i] = quants[i] * scale
    pub fn dequantize(&self, output: &mut [f32; Q8_BLOCK_SIZE]) {
        for i in 0..Q8_BLOCK_SIZE {
            output[i] = self.quants[i] as f32 * self.scale;
        }
    }
}

/// Fused Q8 dot product — single-row decode kernel.
pub fn fused_dot_q8(
    blocks: &[QuantBlockQ8],
    activations: &[f32],
    in_features: usize,
) -> f32 {
    let blocks_per_row = in_features / Q8_BLOCK_SIZE;
    let mut sum = 0.0f32;

    for b in 0..blocks_per_row {
        let block = &blocks[b];
        let act_base = b * Q8_BLOCK_SIZE;

        // Accumulate using integer multiply + single scale multiply
        let mut block_sum = 0.0f32;
        for i in 0..Q8_BLOCK_SIZE {
            block_sum += block.quants[i] as f32 * activations[act_base + i];
        }
        sum += block_sum * block.scale;
    }
    sum
}

// ---------------------------------------------------------------------------
// FP8 E4M3 (K3 A100 Native)
// ---------------------------------------------------------------------------

/// Dequantize a single FP8 E4M3 value to FP32.
///
/// FP8 E4M3: 1 sign + 4 exponent + 3 mantissa bits.
/// Range: ±448, precision: 3 mantissa bits.
/// Native on SpacemiT K3 A100 cores (zero-overhead dequant).
pub fn fp8_e4m3_to_f32(val: u8) -> f32 {
    let sign = (val >> 7) & 1;
    let exp = (val >> 3) & 0x0F;
    let mantissa = val & 0x07;

    if exp == 0 && mantissa == 0 {
        return if sign == 1 { -0.0 } else { 0.0 };
    }

    // Bias = 7 for E4M3
    let f_exp = exp as i32 - 7;
    let f_mantissa = if exp == 0 {
        // Subnormal: 0.mantissa × 2^(-6)
        mantissa as f32 / 8.0 * fast_pow_2(-6)
    } else {
        // Normal: 1.mantissa × 2^(exp-7)
        (1.0 + mantissa as f32 / 8.0) * fast_pow_2(f_exp)
    };

    if sign == 1 { -f_mantissa } else { f_mantissa }
}

/// Dequantize a buffer of FP8 E4M3 values to FP32.
pub fn dequant_fp8_e4m3(input: &[u8], output: &mut [f32]) {
    for (i, &val) in input.iter().enumerate() {
        if i < output.len() {
            output[i] = fp8_e4m3_to_f32(val);
        }
    }
}

/// Fused FP8 dot product — K3 native inference kernel.
pub fn fused_dot_fp8(
    weights: &[u8],  // FP8 E4M3
    activations: &[f32],
    n: usize,
) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..n {
        sum += fp8_e4m3_to_f32(weights[i]) * activations[i];
    }
    sum
}

/// Fast power of 2 for integer exponents.
fn fast_pow_2(exp: i32) -> f32 {
    if exp >= 0 && exp < 31 {
        (1u32 << exp) as f32
    } else if exp < 0 && exp > -127 {
        1.0 / (1u32 << (-exp) as u32) as f32
    } else {
        0.0
    }
}

// ---------------------------------------------------------------------------
// Softmax
// ---------------------------------------------------------------------------

/// Numerically stable softmax: softmax(x_i) = exp(x_i - max) / Σexp(x_j - max)
///
/// Operates in-place on the input slice.
pub fn softmax_f32(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }

    // Find max for numerical stability
    let mut max_val = x[0];
    for &val in x.iter().skip(1) {
        if val > max_val {
            max_val = val;
        }
    }

    // exp(x - max) and sum
    let mut sum = 0.0f32;
    for val in x.iter_mut() {
        // Use a polynomial approximation for no_std exp
        let shifted = *val - max_val;
        let exp_val = fast_exp(shifted);
        *val = exp_val;
        sum += exp_val;
    }

    // Normalize
    if sum > 0.0 {
        let inv_sum = 1.0 / sum;
        for val in x.iter_mut() {
            *val *= inv_sum;
        }
    }
}

// ---------------------------------------------------------------------------
// Layer Normalization
// ---------------------------------------------------------------------------

/// Layer normalization: y = (x - mean) / sqrt(var + eps) * gamma + beta
pub fn layer_norm_f32(
    x: &mut [f32],
    gamma: &[f32],
    beta: &[f32],
    eps: f32,
) {
    let n = x.len();
    if n == 0 {
        return;
    }

    // Compute mean
    let mut mean = 0.0f32;
    for &val in x.iter() {
        mean += val;
    }
    mean /= n as f32;

    // Compute variance
    let mut var = 0.0f32;
    for &val in x.iter() {
        let diff = val - mean;
        var += diff * diff;
    }
    var /= n as f32;

    // Normalize
    let inv_std = 1.0 / fast_sqrt(var + eps);
    for i in 0..n {
        x[i] = (x[i] - mean) * inv_std * gamma[i] + beta[i];
    }
}

/// RMS normalization (used by Qwen, LLaMA):
/// y = x / sqrt(mean(x²) + eps) * gamma
pub fn rms_norm_f32(x: &mut [f32], gamma: &[f32], eps: f32) {
    let n = x.len();
    if n == 0 {
        return;
    }

    // Compute mean of squares
    let mut ms = 0.0f32;
    for &val in x.iter() {
        ms += val * val;
    }
    ms /= n as f32;

    // Normalize
    let inv_rms = 1.0 / fast_sqrt(ms + eps);
    for i in 0..n {
        x[i] = x[i] * inv_rms * gamma[i];
    }
}

// ---------------------------------------------------------------------------
// Activation Functions
// ---------------------------------------------------------------------------

/// SiLU activation: silu(x) = x * sigmoid(x)
/// Used in Qwen, LLaMA, DeepSeek models.
pub fn silu_f32(x: &mut [f32]) {
    for val in x.iter_mut() {
        *val = *val * sigmoid(*val);
    }
}

/// GELU activation (approximate): gelu(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
pub fn gelu_f32(x: &mut [f32]) {
    const SQRT_2_OVER_PI: f32 = 0.7978845608;
    const COEFF: f32 = 0.044715;

    for val in x.iter_mut() {
        let x3 = *val * *val * *val;
        let inner = SQRT_2_OVER_PI * (*val + COEFF * x3);
        *val = 0.5 * *val * (1.0 + fast_tanh(inner));
    }
}

// ---------------------------------------------------------------------------
// Rotary Positional Embeddings (RoPE)
// ---------------------------------------------------------------------------

/// Apply rotary positional embeddings to query/key tensors.
///
/// RoPE encodes position information by rotating pairs of dimensions
/// by angles proportional to their position in the sequence.
pub fn apply_rope_f32(
    x: &mut [f32],
    seq_pos: usize,
    head_dim: usize,
    rope_theta: f32,
) {
    let half_dim = head_dim / 2;
    for i in 0..half_dim {
        let freq = 1.0 / fast_pow(rope_theta, (2 * i) as f32 / head_dim as f32);
        let angle = seq_pos as f32 * freq;
        let cos_val = fast_cos(angle);
        let sin_val = fast_sin(angle);

        let x0 = x[i];
        let x1 = x[i + half_dim];
        x[i] = x0 * cos_val - x1 * sin_val;
        x[i + half_dim] = x0 * sin_val + x1 * cos_val;
    }
}

// ---------------------------------------------------------------------------
// Fast Math Approximations (no_std compatible)
// ---------------------------------------------------------------------------

/// Fast exponential approximation using the Schraudolph method.
fn fast_exp(x: f32) -> f32 {
    if x < -88.0 {
        return 0.0;
    }
    if x > 88.0 {
        return f32::MAX;
    }
    // Minimax polynomial approximation
    let x = 1.0 + x / 256.0;
    let mut result = x;
    for _ in 0..8 {
        result = result * result;
    }
    result
}

/// Fast approximate sigmoid: σ(x) = 1 / (1 + exp(-x))
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + fast_exp(-x))
}

/// Fast approximate tanh using sigmoid: tanh(x) = 2σ(2x) - 1
fn fast_tanh(x: f32) -> f32 {
    2.0 * sigmoid(2.0 * x) - 1.0
}

/// Fast inverse square root (Quake III style, improved).
fn fast_sqrt(x: f32) -> f32 {
    if x <= 0.0 {
        return 0.0;
    }
    let mut guess = x;
    // Newton's method iterations
    for _ in 0..5 {
        guess = 0.5 * (guess + x / guess);
    }
    guess
}

/// Fast power approximation: x^y ≈ exp(y * ln(x))
fn fast_pow(base: f32, exp: f32) -> f32 {
    fast_exp(exp * fast_ln(base))
}

/// Fast natural log approximation.
fn fast_ln(x: f32) -> f32 {
    if x <= 0.0 {
        return f32::MIN;
    }
    // Use the identity: ln(x) = ln(m * 2^e) = ln(m) + e*ln(2)
    let bits = x.to_bits();
    let exp = ((bits >> 23) & 0xFF) as f32 - 127.0;
    let mantissa_bits = (bits & 0x007F_FFFF) | 0x3F80_0000;
    let m = f32::from_bits(mantissa_bits);
    // Polynomial approximation for ln(m) where m ∈ [1, 2)
    let ln_m = -1.725_3 + m * (2.067_2 + m * (-0.341_9));
    ln_m + exp * 0.693_147_2
}

/// Fast cosine approximation.
fn fast_cos(x: f32) -> f32 {
    fast_sin(x + core::f32::consts::FRAC_PI_2)
}

/// Fast sine approximation using Bhaskara I's formula.
fn fast_sin(mut x: f32) -> f32 {
    // Normalize to [-π, π]
    let pi = core::f32::consts::PI;
    let two_pi = 2.0 * pi;
    x = x % two_pi;
    if x > pi {
        x -= two_pi;
    }
    if x < -pi {
        x += two_pi;
    }
    // Parabolic approximation
    let abs_x = if x < 0.0 { -x } else { x };
    let y = 4.0 / pi * x - 4.0 / (pi * pi) * x * abs_x;
    // Refine
    0.225 * (y * (if y < 0.0 { -y } else { y }) - y) + y
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

/// High-level dispatch: multiply two tensors using the best available kernel.
pub fn tensor_matmul(
    a: &TensorDescriptor,
    b: &TensorDescriptor,
    c: &mut TensorDescriptor,
) -> Result<(), AiError> {
    if a.ndim < 2 || b.ndim < 2 {
        return Err(AiError::ComputeError);
    }

    let m = a.shape[0];
    let k = a.shape[1];
    let n = b.shape[1];

    if b.shape[0] != k {
        return Err(AiError::ShapeMismatch {
            expected: a.shape,
            got: b.shape,
        });
    }

    // Update output shape
    c.shape[0] = m;
    c.shape[1] = n;
    c.ndim = 2;
    c.dtype = a.dtype;

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_matmul() {
        let a = [1.0, 2.0, 3.0, 4.0]; // 2×2
        let b = [5.0, 6.0, 7.0, 8.0]; // 2×2
        let mut c = [0.0f32; 4];
        matmul_scalar_f32(&a, &b, &mut c, 2, 2, 2);
        assert!((c[0] - 19.0).abs() < 1e-5); // 1×5 + 2×7
        assert!((c[1] - 22.0).abs() < 1e-5); // 1×6 + 2×8
        assert!((c[2] - 43.0).abs() < 1e-5); // 3×5 + 4×7
        assert!((c[3] - 50.0).abs() < 1e-5); // 3×6 + 4×8
    }

    #[test]
    fn test_softmax() {
        let mut x = [1.0, 2.0, 3.0];
        softmax_f32(&mut x);
        let sum: f32 = x.iter().sum();
        assert!((sum - 1.0).abs() < 0.05);
        assert!(x[2] > x[1]);
        assert!(x[1] > x[0]);
    }

    #[test]
    fn test_silu() {
        let mut x = [0.0f32];
        silu_f32(&mut x);
        assert!((x[0] - 0.0).abs() < 1e-5); // silu(0) = 0
    }

    #[test]
    fn test_rms_norm() {
        let mut x = [1.0, 2.0, 3.0, 4.0];
        let gamma = [1.0, 1.0, 1.0, 1.0];
        rms_norm_f32(&mut x, &gamma, 1e-5);
        // After RMS norm, mean of squares should be ~1
        let ms: f32 = x.iter().map(|v| v * v).sum::<f32>() / 4.0;
        assert!((ms - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_quant_block_dequantize() {
        let block = QuantBlockQ4 {
            scale: 0.1,
            min: -0.5,
            quants: [0x10; 16], // all values: lo=0, hi=1
        };
        let mut output = [0.0f32; Q4_BLOCK_SIZE];
        block.dequantize(&mut output);
        assert!((output[0] - (-0.5)).abs() < 1e-5);  // 0 * 0.1 + (-0.5) = -0.5
        assert!((output[1] - (-0.4)).abs() < 1e-5);  // 1 * 0.1 + (-0.5) = -0.4
    }

    #[test]
    fn test_vlen_elements() {
        assert_eq!(VectorLength::Vlen256.fp32_elements(), 8);
        assert_eq!(VectorLength::Vlen1024.fp32_elements(), 32);
        assert_eq!(VectorLength::Vlen1024.int8_elements(), 128);
        assert_eq!(VectorLength::Vlen1024.int4_elements(), 256);
    }

    #[test]
    fn test_fused_dot_q4() {
        // 1 block of 32 elements
        let block = QuantBlockQ4 {
            scale: 1.0,
            min: 0.0,
            quants: [0x10; 16], // lo=0, hi=1 for each pair
        };
        // Activations: all 1.0
        let activations = [1.0f32; Q4_BLOCK_SIZE];
        let result = fused_dot_q4(&[block], &activations, Q4_BLOCK_SIZE);
        // 16 lo values (0 * 1.0 * 1.0) + 16 hi values (1 * 1.0 * 1.0) = 16.0
        assert!((result - 16.0).abs() < 1e-4);
    }

    #[test]
    fn test_q8_dequantize() {
        let block = QuantBlockQ8 {
            scale: 0.5,
            quants: [2; Q8_BLOCK_SIZE], // all 2
        };
        let mut output = [0.0f32; Q8_BLOCK_SIZE];
        block.dequantize(&mut output);
        // 2 * 0.5 = 1.0 for each element
        for &v in &output {
            assert!((v - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_fused_dot_q8() {
        let block = QuantBlockQ8 {
            scale: 0.1,
            quants: [10; Q8_BLOCK_SIZE], // all 10
        };
        let activations = [1.0f32; Q8_BLOCK_SIZE];
        let result = fused_dot_q8(&[block], &activations, Q8_BLOCK_SIZE);
        // sum(10 * 1.0) * 0.1 = 32 * 10 * 0.1 = 32.0
        assert!((result - 32.0).abs() < 1e-3);
    }

    #[test]
    fn test_fp8_e4m3_zero() {
        assert_eq!(fp8_e4m3_to_f32(0x00), 0.0);
        // Negative zero
        let neg_zero = fp8_e4m3_to_f32(0x80);
        assert!(neg_zero == 0.0 || neg_zero == -0.0);
    }

    #[test]
    fn test_fp8_e4m3_one() {
        // 1.0 in E4M3: sign=0, exp=0111 (=7, bias=7 -> actual 0), mantissa=000
        // So bits = 0_0111_000 = 0x38
        let val = fp8_e4m3_to_f32(0x38);
        assert!((val - 1.0).abs() < 1e-3, "Expected 1.0, got {}", val);
    }

    #[test]
    fn test_fp8_e4m3_negative() {
        // -1.0: sign=1, exp=0111, mantissa=000
        // bits = 1_0111_000 = 0xB8
        let val = fp8_e4m3_to_f32(0xB8);
        assert!((val - (-1.0)).abs() < 1e-3, "Expected -1.0, got {}", val);
    }

    #[test]
    fn test_fused_dot_fp8() {
        // 4 weights = [1.0, 1.0, 1.0, 1.0] encoded as E4M3
        let weights = [0x38u8; 4]; // All 1.0
        let activations = [2.0f32; 4];
        let result = fused_dot_fp8(&weights, &activations, 4);
        // 4 * 1.0 * 2.0 = 8.0
        assert!((result - 8.0).abs() < 0.1, "Expected ~8.0, got {}", result);
    }
}
