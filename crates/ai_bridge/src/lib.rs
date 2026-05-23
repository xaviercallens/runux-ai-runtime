// Copyright (c) 2026 Xavier Callens / Socrate AI. All Rights Reserved.
// SPDX-License-Identifier: LicenseRef-RunuX-Commercial
// This file is proprietary and confidential. Unauthorized copying,
// distribution, or use is strictly prohibited.

#![cfg_attr(not(test), no_std)]
#![deny(clippy::all)]
#![warn(clippy::pedantic)]
//! RunuX AI Bridge — C FFI bridge for Python/C++ ML framework interop
//!
//! Provides a C-compatible API for interacting with the RunuX AI runtime
//! from userspace Python (TensorFlow, PyTorch) or C++ (llama.cpp, ONNX Runtime)
//! applications. Enables zero-copy tensor sharing between the kernel AI
//! acceleration layer and userspace ML frameworks.
//!
//! # FFI Usage (from C/Python)
//!
//! ```c
//! #include "runux_ai.h"
//!
//! RunuxTensor* tensor = runux_tensor_create(shape, 3, RUNUX_DTYPE_FP32, RUNUX_DEV_CPU);
//! runux_tensor_fill(tensor, data, size);
//! RunuxTensor* result = runux_matmul(tensor_a, tensor_b);
//! runux_tensor_free(result);
//! ```
//!
//! # Safety
//!
//! All FFI functions are marked `unsafe` and use raw pointers for C interop.
//! The caller is responsible for proper memory management (create/free pairs).

extern crate alloc;
use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;

use ai_runtime::{
    DataType, DeviceType, HardwareCaps,
    TensorDescriptor, MAX_DIMS,
};
use rvv_simd::{matmul_rvv_f32, softmax_f32, silu_f32};

// ---------------------------------------------------------------------------
// C-Compatible Type Aliases
// ---------------------------------------------------------------------------

/// C-compatible data type enum (matches DataType ordinals).
pub type RunuxDtype = u8;
pub const RUNUX_DTYPE_FP32: RunuxDtype = 0;
pub const RUNUX_DTYPE_FP16: RunuxDtype = 1;
pub const RUNUX_DTYPE_BF16: RunuxDtype = 2;
pub const RUNUX_DTYPE_FP8: RunuxDtype = 3;
pub const RUNUX_DTYPE_INT8: RunuxDtype = 4;
pub const RUNUX_DTYPE_INT4: RunuxDtype = 5;
pub const RUNUX_DTYPE_UINT8: RunuxDtype = 6;

/// C-compatible device type enum.
pub type RunuxDevice = u8;
pub const RUNUX_DEV_CPU: RunuxDevice = 0;
pub const RUNUX_DEV_A100: RunuxDevice = 1;
pub const RUNUX_DEV_GPU: RunuxDevice = 2;

/// C-compatible error codes.
pub type RunuxError = i32;
pub const RUNUX_OK: RunuxError = 0;
pub const RUNUX_ERR_NULL_PTR: RunuxError = -1;
pub const RUNUX_ERR_OOM: RunuxError = -2;
pub const RUNUX_ERR_SHAPE: RunuxError = -3;
pub const RUNUX_ERR_UNSUPPORTED: RunuxError = -4;
pub const RUNUX_ERR_COMPUTE: RunuxError = -5;

// ---------------------------------------------------------------------------
// Opaque Handle Types
// ---------------------------------------------------------------------------

/// Opaque tensor handle for C FFI.
///
/// Contains a TensorDescriptor plus owned data storage.
#[repr(C)]
pub struct RunuxTensor {
    pub desc: TensorDescriptor,
    data: Vec<u8>,
}

impl RunuxTensor {
    /// Create a new tensor with allocated storage.
    pub fn new(shape: &[usize], dtype: DataType, device: DeviceType) -> Self {
        let desc = TensorDescriptor::new(shape, dtype, device);
        let size = desc.size_bytes();
        let data = vec![0u8; size];
        Self { desc, data }
    }

    /// Get a reference to the data as f32 slice (if dtype is FP32).
    pub fn as_f32_slice(&self) -> Option<&[f32]> {
        if self.desc.dtype != DataType::FP32 {
            return None;
        }
        let ptr = self.data.as_ptr() as *const f32;
        let len = self.data.len() / 4;
        Some(unsafe { core::slice::from_raw_parts(ptr, len) })
    }

    /// Get a mutable reference to the data as f32 slice.
    pub fn as_f32_slice_mut(&mut self) -> Option<&mut [f32]> {
        if self.desc.dtype != DataType::FP32 {
            return None;
        }
        let ptr = self.data.as_mut_ptr() as *mut f32;
        let len = self.data.len() / 4;
        Some(unsafe { core::slice::from_raw_parts_mut(ptr, len) })
    }
}

// ---------------------------------------------------------------------------
// FFI: Tensor Lifecycle
// ---------------------------------------------------------------------------

/// Create a new tensor. Returns null on failure.
///
/// # Safety
///
/// - `shape` must point to `ndim` valid usize values
/// - Caller must free the tensor with `runux_tensor_free`
#[no_mangle]
pub unsafe extern "C" fn runux_tensor_create(
    shape: *const usize,
    ndim: usize,
    dtype: RunuxDtype,
    device: RunuxDevice,
) -> *mut RunuxTensor {
    if shape.is_null() || ndim == 0 || ndim > MAX_DIMS {
        return core::ptr::null_mut();
    }

    let shape_slice = core::slice::from_raw_parts(shape, ndim);
    let dt = match dtype {
        RUNUX_DTYPE_FP32 => DataType::FP32,
        RUNUX_DTYPE_FP16 => DataType::FP16,
        RUNUX_DTYPE_BF16 => DataType::BF16,
        RUNUX_DTYPE_FP8 => DataType::FP8,
        RUNUX_DTYPE_INT8 => DataType::INT8,
        RUNUX_DTYPE_INT4 => DataType::INT4,
        RUNUX_DTYPE_UINT8 => DataType::UINT8,
        _ => return core::ptr::null_mut(),
    };
    let dev = match device {
        RUNUX_DEV_CPU => DeviceType::Cpu,
        RUNUX_DEV_A100 => DeviceType::A100AiCore,
        RUNUX_DEV_GPU => DeviceType::PowerVrGpu,
        _ => DeviceType::Cpu,
    };

    let tensor = RunuxTensor::new(shape_slice, dt, dev);
    Box::into_raw(Box::new(tensor))
}

/// Free a tensor created with `runux_tensor_create`.
///
/// # Safety
///
/// - `tensor` must be a valid pointer from `runux_tensor_create`
/// - Must not be called twice on the same tensor
#[no_mangle]
pub unsafe extern "C" fn runux_tensor_free(tensor: *mut RunuxTensor) {
    if !tensor.is_null() {
        drop(Box::from_raw(tensor));
    }
}

/// Fill a tensor with data from a buffer.
///
/// # Safety
///
/// - `tensor` must be valid
/// - `data` must point to at least `tensor.desc.size_bytes()` bytes
#[no_mangle]
pub unsafe extern "C" fn runux_tensor_fill(
    tensor: *mut RunuxTensor,
    data: *const u8,
    size: usize,
) -> RunuxError {
    if tensor.is_null() || data.is_null() {
        return RUNUX_ERR_NULL_PTR;
    }

    let t = &mut *tensor;
    let needed = t.desc.size_bytes();
    if size < needed {
        return RUNUX_ERR_SHAPE;
    }

    let src = core::slice::from_raw_parts(data, needed);
    t.data[..needed].copy_from_slice(src);
    RUNUX_OK
}

/// Get the total number of elements in a tensor.
///
/// # Safety
///
/// - `tensor` must be valid
#[no_mangle]
pub unsafe extern "C" fn runux_tensor_numel(tensor: *const RunuxTensor) -> usize {
    if tensor.is_null() {
        return 0;
    }
    (*tensor).desc.numel()
}

/// Get the size in bytes of a tensor's data.
///
/// # Safety
///
/// - `tensor` must be valid
#[no_mangle]
pub unsafe extern "C" fn runux_tensor_size_bytes(tensor: *const RunuxTensor) -> usize {
    if tensor.is_null() {
        return 0;
    }
    (*tensor).desc.size_bytes()
}

// ---------------------------------------------------------------------------
// FFI: Compute Operations
// ---------------------------------------------------------------------------

/// Matrix multiply two tensors: C = A × B
///
/// Both tensors must be FP32 and 2D. Returns a new tensor.
///
/// # Safety
///
/// - `a` and `b` must be valid tensors
/// - Caller must free the result with `runux_tensor_free`
#[no_mangle]
pub unsafe extern "C" fn runux_matmul(
    a: *const RunuxTensor,
    b: *const RunuxTensor,
) -> *mut RunuxTensor {
    if a.is_null() || b.is_null() {
        return core::ptr::null_mut();
    }

    let ta = &*a;
    let tb = &*b;

    if ta.desc.dtype != DataType::FP32 || tb.desc.dtype != DataType::FP32 {
        return core::ptr::null_mut();
    }

    let m = ta.desc.shape[0];
    let k = ta.desc.shape[1];
    let n = tb.desc.shape[1];

    if tb.desc.shape[0] != k || k == 0 {
        return core::ptr::null_mut();
    }

    let mut result = RunuxTensor::new(&[m, n], DataType::FP32, ta.desc.device);

    if let (Some(a_data), Some(b_data), Some(c_data)) = (
        ta.as_f32_slice(),
        tb.as_f32_slice(),
        result.as_f32_slice_mut(),
    ) {
        matmul_rvv_f32(a_data, b_data, c_data, m, k, n);
    } else {
        return core::ptr::null_mut();
    }

    Box::into_raw(Box::new(result))
}

/// Apply softmax in-place to a tensor.
///
/// # Safety
///
/// - `tensor` must be valid and FP32
#[no_mangle]
pub unsafe extern "C" fn runux_softmax(tensor: *mut RunuxTensor) -> RunuxError {
    if tensor.is_null() {
        return RUNUX_ERR_NULL_PTR;
    }

    let t = &mut *tensor;
    if let Some(data) = t.as_f32_slice_mut() {
        softmax_f32(data);
        RUNUX_OK
    } else {
        RUNUX_ERR_UNSUPPORTED
    }
}

/// Apply SiLU activation in-place.
///
/// # Safety
///
/// - `tensor` must be valid and FP32
#[no_mangle]
pub unsafe extern "C" fn runux_silu(tensor: *mut RunuxTensor) -> RunuxError {
    if tensor.is_null() {
        return RUNUX_ERR_NULL_PTR;
    }

    let t = &mut *tensor;
    if let Some(data) = t.as_f32_slice_mut() {
        silu_f32(data);
        RUNUX_OK
    } else {
        RUNUX_ERR_UNSUPPORTED
    }
}

// ---------------------------------------------------------------------------
// FFI: Hardware Detection
// ---------------------------------------------------------------------------

/// Detect hardware capabilities and return as a C struct.
///
/// # Safety
///
/// - `caps` must point to a valid HardwareCaps struct
#[no_mangle]
pub unsafe extern "C" fn runux_detect_hardware(caps: *mut HardwareCaps) -> RunuxError {
    if caps.is_null() {
        return RUNUX_ERR_NULL_PTR;
    }

    // TODO: Detect actual hardware via CSR reads and device tree parsing
    // For now, return K1 defaults
    *caps = HardwareCaps::spacemit_k1(4);
    RUNUX_OK
}

/// Query the optimal data type for a model of the given size.
///
/// # Safety
///
/// - `caps` must be valid
#[no_mangle]
pub unsafe extern "C" fn runux_optimal_dtype(
    caps: *const HardwareCaps,
    params_billions: u32,
) -> RunuxDtype {
    if caps.is_null() {
        return RUNUX_DTYPE_INT4; // Safe fallback
    }

    match (*caps).optimal_dtype(params_billions) {
        DataType::FP32 => RUNUX_DTYPE_FP32,
        DataType::FP16 => RUNUX_DTYPE_FP16,
        DataType::BF16 => RUNUX_DTYPE_BF16,
        DataType::FP8 => RUNUX_DTYPE_FP8,
        DataType::INT8 => RUNUX_DTYPE_INT8,
        DataType::INT4 => RUNUX_DTYPE_INT4,
        DataType::UINT8 => RUNUX_DTYPE_UINT8,
        _ => RUNUX_DTYPE_INT4,
    }
}

// ---------------------------------------------------------------------------
// FFI: Version Info
// ---------------------------------------------------------------------------

/// Returns the RunuX AI runtime version as a static string.
///
/// # Safety
///
/// The returned pointer is valid for the lifetime of the program.
#[no_mangle]
pub unsafe extern "C" fn runux_ai_version() -> *const u8 {
    b"RunuX AI Runtime v0.1.0 (RISC-V)\0".as_ptr()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_create_and_fill() {
        let shape = [2usize, 3usize];
        let tensor = unsafe {
            runux_tensor_create(shape.as_ptr(), 2, RUNUX_DTYPE_FP32, RUNUX_DEV_CPU)
        };
        assert!(!tensor.is_null());

        let data: [f32; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let err = unsafe {
            runux_tensor_fill(
                tensor,
                data.as_ptr() as *const u8,
                core::mem::size_of_val(&data),
            )
        };
        assert_eq!(err, RUNUX_OK);

        let numel = unsafe { runux_tensor_numel(tensor) };
        assert_eq!(numel, 6);

        unsafe { runux_tensor_free(tensor) };
    }

    #[test]
    fn test_matmul_ffi() {
        let shape_a = [2usize, 2usize];
        let shape_b = [2usize, 2usize];

        let ta = unsafe {
            runux_tensor_create(shape_a.as_ptr(), 2, RUNUX_DTYPE_FP32, RUNUX_DEV_CPU)
        };
        let tb = unsafe {
            runux_tensor_create(shape_b.as_ptr(), 2, RUNUX_DTYPE_FP32, RUNUX_DEV_CPU)
        };

        let a_data: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
        let b_data: [f32; 4] = [5.0, 6.0, 7.0, 8.0];

        unsafe {
            runux_tensor_fill(ta, a_data.as_ptr() as *const u8, 16);
            runux_tensor_fill(tb, b_data.as_ptr() as *const u8, 16);
        }

        let result = unsafe { runux_matmul(ta, tb) };
        assert!(!result.is_null());

        let result_ref = unsafe { &*result };
        let data = result_ref.as_f32_slice().unwrap();
        assert!((data[0] - 19.0).abs() < 1e-5);
        assert!((data[3] - 50.0).abs() < 1e-5);

        unsafe {
            runux_tensor_free(ta);
            runux_tensor_free(tb);
            runux_tensor_free(result);
        }
    }

    #[test]
    fn test_version_string() {
        let ptr = unsafe { runux_ai_version() };
        assert!(!ptr.is_null());
    }

    #[test]
    fn test_null_safety() {
        assert_eq!(unsafe { runux_tensor_numel(core::ptr::null()) }, 0);
        assert_eq!(unsafe { runux_tensor_size_bytes(core::ptr::null()) }, 0);
        assert!(unsafe { runux_matmul(core::ptr::null(), core::ptr::null()) }.is_null());
    }
}
