// Copyright (c) 2026 Xavier Callens / Socrate AI. All Rights Reserved.
// SPDX-License-Identifier: LicenseRef-RunuX-Commercial
// This file is proprietary and confidential. Unauthorized copying,
// distribution, or use is strictly prohibited.

#![deny(clippy::all)]
#![warn(clippy::pedantic)]
#![allow(clippy::manual_c_str_literals)]
//! RunuX Framework Bridge — C-compatible FFI for ML framework integration
//!
//! Enables RunuX kernels to be called from:
//! - **PyTorch**: `torch.utils.cpp_extension` → load `librunux.so`
//! - **TensorFlow**: `tf.load_op_library()` → RunuX custom ops
//! - **JAX**: `jax.extend.ffi.ffi_call()` → RunuX FFI functions
//!
//! # C API Contract
//!
//! All exported functions follow these conventions:
//! - Return `i32`: 0 = success, negative = error code
//! - Accept raw pointers for data buffers
//! - Use `i32` for dimensions (matches framework conventions)
//! - `backend` parameter: 0=CPU, 1=RISC-V, 2=TPU, 3=GPU
//!
//! # Memory Safety
//!
//! Despite the `unsafe` FFI boundary, Rust ensures:
//! - All internal operations are bounds-checked
//! - No undefined behavior from null pointers (checked at entry)
//! - Stack-based temporaries prevent memory leaks

use hal::{Accelerator, CpuBackend, FlashConfig};

// ---------------------------------------------------------------------------
// Error Codes
// ---------------------------------------------------------------------------

/// Success.
pub const RUNUX_OK: i32 = 0;
/// Null pointer passed.
pub const RUNUX_ERR_NULL_PTR: i32 = -1;
/// Invalid dimensions.
pub const RUNUX_ERR_INVALID_DIMS: i32 = -2;
/// Backend not available.
pub const RUNUX_ERR_BACKEND: i32 = -3;
/// Internal error.
pub const RUNUX_ERR_INTERNAL: i32 = -4;

// ---------------------------------------------------------------------------
// Version Info
// ---------------------------------------------------------------------------

/// Get RunuX version string.
///
/// Returns a null-terminated UTF-8 string.
/// The pointer is valid for the lifetime of the library.
#[no_mangle]
pub extern "C" fn runux_version() -> *const u8 {
    b"RunuX AI Runtime v0.2.0-tpu\0".as_ptr()
}

/// Get RunuX capabilities bitmask.
///
/// Bit 0: CPU backend
/// Bit 1: RISC-V backend
/// Bit 2: TPU backend (simulation)
/// Bit 3: GPU backend
/// Bit 4: FlashAttention
/// Bit 5: TurboQuant
/// Bit 6: Speculative decoding
/// Bit 7: Federated learning
#[no_mangle]
pub extern "C" fn runux_capabilities() -> u32 {
    0b1111_0101 // CPU + RISC-V + TPU(sim) + FlashAttn + TurboQuant + Speculative
}

// ---------------------------------------------------------------------------
// Matrix Operations
// ---------------------------------------------------------------------------

/// Dense matrix multiplication: C = A × B.
///
/// # Safety
/// Caller must ensure pointers are valid and dimensions are correct.
///
/// # Arguments
/// - `a_ptr`: Pointer to A matrix (M × K, row-major)
/// - `b_ptr`: Pointer to B matrix (K × N, row-major)
/// - `c_ptr`: Pointer to output C matrix (M × N, row-major)
/// - `m, n, k`: Dimensions
/// - `backend`: 0=CPU, 1=RISC-V, 2=TPU, 3=GPU
#[no_mangle]
pub unsafe extern "C" fn runux_matmul(
    a_ptr: *const f32,
    b_ptr: *const f32,
    c_ptr: *mut f32,
    m: i32,
    n: i32,
    k: i32,
    backend: i32,
) -> i32 {
    if a_ptr.is_null() || b_ptr.is_null() || c_ptr.is_null() {
        return RUNUX_ERR_NULL_PTR;
    }
    if m <= 0 || n <= 0 || k <= 0 {
        return RUNUX_ERR_INVALID_DIMS;
    }

    let m = m as usize;
    let n = n as usize;
    let k = k as usize;

    let a = core::slice::from_raw_parts(a_ptr, m * k);
    let b = core::slice::from_raw_parts(b_ptr, k * n);
    let c = core::slice::from_raw_parts_mut(c_ptr, m * n);

    match backend {
        0 | 2 => {
            // CPU or TPU simulation
            let cpu = CpuBackend::new();
            cpu.matmul(a, b, c, m, n, k);
            RUNUX_OK
        }
        _ => RUNUX_ERR_BACKEND,
    }
}

// ---------------------------------------------------------------------------
// FlashAttention
// ---------------------------------------------------------------------------

/// FlashAttention forward pass.
///
/// # Safety
/// Caller must ensure pointers are valid and dimensions are correct.
///
/// # Arguments
/// - `q_ptr`: Query tensor [batch × heads × seq_len × head_dim]
/// - `k_ptr`: Key tensor (same layout)
/// - `v_ptr`: Value tensor (same layout)
/// - `out_ptr`: Output tensor (same layout)
/// - `batch, heads, seq_len, head_dim`: Dimensions
/// - `causal`: 1 = causal masking, 0 = no masking
/// - `backend`: 0=CPU, 1=RISC-V, 2=TPU, 3=GPU
#[no_mangle]
pub unsafe extern "C" fn runux_flash_attention(
    q_ptr: *const f32,
    k_ptr: *const f32,
    v_ptr: *const f32,
    out_ptr: *mut f32,
    batch: i32,
    heads: i32,
    seq_len: i32,
    head_dim: i32,
    causal: i32,
    backend: i32,
) -> i32 {
    if q_ptr.is_null() || k_ptr.is_null() || v_ptr.is_null() || out_ptr.is_null() {
        return RUNUX_ERR_NULL_PTR;
    }
    if batch <= 0 || heads <= 0 || seq_len <= 0 || head_dim <= 0 {
        return RUNUX_ERR_INVALID_DIMS;
    }

    let total = (batch * heads * seq_len * head_dim) as usize;
    let q = core::slice::from_raw_parts(q_ptr, total);
    let k = core::slice::from_raw_parts(k_ptr, total);
    let v = core::slice::from_raw_parts(v_ptr, total);
    let out = core::slice::from_raw_parts_mut(out_ptr, total);

    let head_dim = head_dim as usize;
    let seq_len = seq_len as usize;

    let config = FlashConfig {
        n_heads: heads as usize,
        head_dim,
        tile_q: 64.min(seq_len),
        tile_kv: 64.min(seq_len),
        causal: causal != 0,
        scale: 1.0 / fast_sqrt(head_dim as f32),
    };

    match backend {
        0 | 2 => {
            let cpu = CpuBackend::new();
            // Process per-head
            let per_head = seq_len * head_dim;
            let batch = batch as usize;
            let heads = heads as usize;
            for b in 0..batch {
                for h in 0..heads {
                    let offset = (b * heads + h) * per_head;
                    let end = offset + per_head;
                    if end > total {
                        break;
                    }
                    let single_config = FlashConfig {
                        n_heads: 1,
                        ..config.clone()
                    };
                    cpu.flash_attention(
                        &q[offset..end],
                        &k[offset..end],
                        &v[offset..end],
                        &mut out[offset..end],
                        &single_config,
                    );
                }
            }
            RUNUX_OK
        }
        _ => RUNUX_ERR_BACKEND,
    }
}

// ---------------------------------------------------------------------------
// Normalization
// ---------------------------------------------------------------------------

/// RMS normalization in-place.
///
/// # Safety
/// Caller must ensure pointers are valid.
#[no_mangle]
pub unsafe extern "C" fn runux_rms_norm(
    x_ptr: *mut f32,
    weight_ptr: *const f32,
    dim: i32,
    eps: f32,
) -> i32 {
    if x_ptr.is_null() || weight_ptr.is_null() {
        return RUNUX_ERR_NULL_PTR;
    }
    if dim <= 0 {
        return RUNUX_ERR_INVALID_DIMS;
    }

    let dim = dim as usize;
    let x = core::slice::from_raw_parts_mut(x_ptr, dim);
    let w = core::slice::from_raw_parts(weight_ptr, dim);

    let cpu = CpuBackend::new();
    cpu.rms_norm(x, w, eps);
    RUNUX_OK
}

/// SiLU activation in-place.
///
/// # Safety
/// Caller must ensure pointer is valid.
#[no_mangle]
pub unsafe extern "C" fn runux_silu(x_ptr: *mut f32, len: i32) -> i32 {
    if x_ptr.is_null() {
        return RUNUX_ERR_NULL_PTR;
    }
    if len <= 0 {
        return RUNUX_ERR_INVALID_DIMS;
    }

    let x = core::slice::from_raw_parts_mut(x_ptr, len as usize);
    let cpu = CpuBackend::new();
    cpu.silu(x);
    RUNUX_OK
}

/// Softmax in-place.
///
/// # Safety
/// Caller must ensure pointer is valid.
#[no_mangle]
pub unsafe extern "C" fn runux_softmax(x_ptr: *mut f32, len: i32) -> i32 {
    if x_ptr.is_null() {
        return RUNUX_ERR_NULL_PTR;
    }
    if len <= 0 {
        return RUNUX_ERR_INVALID_DIMS;
    }

    let x = core::slice::from_raw_parts_mut(x_ptr, len as usize);
    let cpu = CpuBackend::new();
    cpu.softmax(x);
    RUNUX_OK
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn fast_sqrt(x: f32) -> f32 {
    if x <= 0.0 {
        return 0.0;
    }
    let mut g = x;
    for _ in 0..5 {
        g = 0.5 * (g + x / g);
    }
    g
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        let ptr = runux_version();
        let s = unsafe { core::ffi::CStr::from_ptr(ptr as *const core::ffi::c_char) };
        assert!(s.to_str().unwrap().contains("RunuX"));
    }

    #[test]
    fn test_capabilities() {
        let caps = runux_capabilities();
        assert!(caps & 0x01 != 0, "CPU should be available");
        assert!(caps & 0x04 != 0, "TPU sim should be available");
    }

    #[test]
    fn test_matmul_ffi() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0]; // 2×2
        let b = vec![5.0f32, 6.0, 7.0, 8.0]; // 2×2
        let mut c = vec![0.0f32; 4];

        let result = unsafe { runux_matmul(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), 2, 2, 2, 0) };
        assert_eq!(result, RUNUX_OK);
        assert!((c[0] - 19.0).abs() < 1e-5);
        assert!((c[3] - 50.0).abs() < 1e-5);
    }

    #[test]
    fn test_matmul_null_check() {
        let mut c = vec![0.0f32; 4];
        let result = unsafe {
            runux_matmul(
                core::ptr::null(),
                core::ptr::null(),
                c.as_mut_ptr(),
                2,
                2,
                2,
                0,
            )
        };
        assert_eq!(result, RUNUX_ERR_NULL_PTR);
    }

    #[test]
    fn test_rms_norm_ffi() {
        let mut x = vec![1.0f32, 2.0, 3.0, 4.0];
        let w = vec![1.0f32; 4];

        let result = unsafe { runux_rms_norm(x.as_mut_ptr(), w.as_ptr(), 4, 1e-6) };
        assert_eq!(result, RUNUX_OK);
        // Should be normalized
        let norm_sq: f32 = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
        assert!((norm_sq - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_softmax_ffi() {
        let mut x = vec![1.0f32, 2.0, 3.0];
        let result = unsafe { runux_softmax(x.as_mut_ptr(), 3) };
        assert_eq!(result, RUNUX_OK);
        let sum: f32 = x.iter().sum();
        assert!((sum - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_silu_ffi() {
        let mut x = vec![0.0f32, 1.0, -1.0];
        let result = unsafe { runux_silu(x.as_mut_ptr(), 3) };
        assert_eq!(result, RUNUX_OK);
        assert!((x[0] - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_invalid_dims() {
        let a = vec![1.0f32; 4];
        let mut c = vec![0.0f32; 4];
        let result = unsafe { runux_matmul(a.as_ptr(), a.as_ptr(), c.as_mut_ptr(), -1, 2, 2, 0) };
        assert_eq!(result, RUNUX_ERR_INVALID_DIMS);
    }
}
