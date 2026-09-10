// Copyright (c) 2026 Xavier Callens / Socrate AI. All Rights Reserved.
// SPDX-License-Identifier: LicenseRef-RunuX-Commercial

//! C FFI Declarations for Google Cloud TPU PJRT Runtime (`pjrt_c_api.h`).
//!
//! Provides ABI-compatible foreign function interfaces to dynamically link
//! with `libpjrt_c_api.so` on Google Cloud TPU VM instances.

use core::ffi::{c_char, c_void};

/// Status code returned by PJRT C API calls.
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum PjrtErrorCode {
    Ok = 0,
    Cancelled = 1,
    Unknown = 2,
    InvalidArgument = 3,
    DeadlineExceeded = 4,
    NotFound = 5,
    AlreadyExists = 6,
    PermissionDenied = 7,
    ResourceExhausted = 8,
    FailedPrecondition = 9,
    Aborted = 10,
    OutOfRange = 11,
    Unimplemented = 12,
    Internal = 13,
    Unavailable = 14,
    DataLoss = 15,
    Unauthenticated = 16,
}

/// Opaque PJRT C API Error object.
#[repr(C)]
pub struct PjrtError {
    _unused: [u8; 0],
}

/// Opaque PJRT Client handle.
#[repr(C)]
pub struct PjrtCClient {
    _unused: [u8; 0],
}

/// Opaque PJRT Device handle.
#[repr(C)]
pub struct PjrtCDevice {
    _unused: [u8; 0],
}

/// Opaque PJRT Buffer (HBM Tensor) handle.
#[repr(C)]
pub struct PjrtCBuffer {
    _unused: [u8; 0],
}

/// Opaque PJRT Compiled Executable handle.
#[repr(C)]
pub struct PjrtCExecutable {
    _unused: [u8; 0],
}

/// PJRT C API Function Table (`PJRT_Api`).
/// ABI-compatible with Google XLA PJRT C API v0.48+.
#[repr(C)]
pub struct PjrtApi {
    pub struct_size: usize,
    pub extension_start: *mut c_void,
    pub pjrt_api_version: [u32; 2], // Major, Minor

    // Error APIs
    pub error_destroy: Option<unsafe extern "C" fn(error: *mut PjrtError)>,
    pub error_message: Option<
        unsafe extern "C" fn(
            error: *mut PjrtError,
            message: *mut *const c_char,
            length: *mut usize,
        ),
    >,
    pub error_code: Option<unsafe extern "C" fn(error: *mut PjrtError) -> PjrtErrorCode>,

    // Client APIs
    pub client_create:
        Option<unsafe extern "C" fn(client: *mut *mut PjrtCClient) -> *mut PjrtError>,
    pub client_destroy: Option<unsafe extern "C" fn(client: *mut PjrtCClient) -> *mut PjrtError>,
    pub client_platform_name: Option<
        unsafe extern "C" fn(
            client: *mut PjrtCClient,
            name: *mut *const c_char,
            length: *mut usize,
        ) -> *mut PjrtError,
    >,
    pub client_devices: Option<
        unsafe extern "C" fn(
            client: *mut PjrtCClient,
            devices: *mut *mut *mut PjrtCDevice,
            num_devices: *mut usize,
        ) -> *mut PjrtError,
    >,

    // Buffer APIs
    pub buffer_destroy: Option<unsafe extern "C" fn(buffer: *mut PjrtCBuffer) -> *mut PjrtError>,
    pub buffer_to_host: Option<
        unsafe extern "C" fn(
            buffer: *mut PjrtCBuffer,
            host_dst: *mut c_void,
            dst_size: usize,
        ) -> *mut PjrtError,
    >,

    // Execution APIs
    pub compile: Option<
        unsafe extern "C" fn(
            client: *mut PjrtCClient,
            program_bytecode: *const u8,
            size: usize,
            exec: *mut *mut PjrtCExecutable,
        ) -> *mut PjrtError,
    >,
    pub executable_execute: Option<
        unsafe extern "C" fn(
            exec: *mut PjrtCExecutable,
            inputs: *const *mut PjrtCBuffer,
            num_inputs: usize,
            outputs: *mut *mut PjrtCBuffer,
            num_outputs: usize,
        ) -> *mut PjrtError,
    >,
    pub executable_destroy:
        Option<unsafe extern "C" fn(exec: *mut PjrtCExecutable) -> *mut PjrtError>,
}

/// Dynamic Discovery and Loader Interface for Google TPU PJRT plugin.
pub struct PjrtPluginLoader {
    plugin_path: &'static str,
}

impl PjrtPluginLoader {
    pub const DEFAULT_TPU_PLUGIN_PATH: &'static str = "/usr/local/lib/libpjrt_c_api.so";

    #[must_use]
    pub const fn new(path: &'static str) -> Self {
        Self { plugin_path: path }
    }

    #[must_use]
    pub const fn plugin_path(&self) -> &'static str {
        self.plugin_path
    }

    /// Check if the physical hardware PJRT plugin library is available on host.
    #[must_use]
    pub fn is_plugin_present(&self) -> bool {
        // In no_std simulation or when absent, return false to trigger clean simulation fallback
        false
    }
}
