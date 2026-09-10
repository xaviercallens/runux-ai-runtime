// Copyright (c) 2026 Xavier Callens / Socrate AI. All Rights Reserved.
// SPDX-License-Identifier: LicenseRef-RunuX-Commercial
// This file is proprietary and confidential. Unauthorized copying,
// distribution, or use is strictly prohibited.

#![cfg_attr(not(test), no_std)]
#![deny(clippy::all)]
#![warn(clippy::pedantic)]
#![allow(clippy::manual_checked_ops, clippy::too_many_arguments)]
//! RunuX TPU PJRT — Safe Rust bindings for Google TPU via PJRT C API
//!
//! Provides ownership-aware types that map to the PJRT runtime:
//!
//! | Rust Type       | PJRT Handle       | Purpose                          |
//! |-----------------|-------------------|----------------------------------|
//! | `PjrtClient`    | `PJRT_Client*`    | Entry point, owns devices+memory |
//! | `PjrtDevice`    | `PJRT_Device*`    | Single TPU chip                  |
//! | `PjrtBuffer`    | `PJRT_Buffer*`    | HBM tensor with async tracking   |
//! | `PjrtExecutable`| `PJRT_Executable*`| Compiled StableHLO program       |
//!
//! # Simulation Mode
//!
//! When compiled without `feature = "tpu_hw"`, all operations execute
//! on CPU with TPU constraints (tile sizes, memory limits) enforced.
//! This enables full development without TPU access.
//!
//! # Memory Safety
//!
//! Key advantages over C++ PJRT usage:
//! - `PjrtBuffer` uses `Drop` for deterministic HBM deallocation
//! - `PjrtExecutable` is `Send + Sync` — safe multi-threaded dispatch
//! - All raw pointers wrapped in `Option<NonNull>` — null-safe
//! - Buffer states tracked to prevent use-after-free and data races

extern crate alloc;
pub mod ffi;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use hal::{DType, Shape};

// ---------------------------------------------------------------------------
// PJRT Error Types
// ---------------------------------------------------------------------------

/// Errors from PJRT operations.
#[derive(Debug, Clone)]
pub enum PjrtError {
    /// Plugin library not found or failed to load.
    PluginLoadFailed(String),
    /// Device not available.
    DeviceUnavailable,
    /// Out of HBM memory.
    OutOfMemory { requested: usize, available: usize },
    /// Compilation failed (StableHLO → TPU binary).
    CompileFailed(String),
    /// Execution failed.
    ExecutionFailed(String),
    /// Buffer is still in-flight (async computation not done).
    BufferNotReady,
    /// Shape mismatch between expected and actual.
    ShapeMismatch { expected: Shape, got: Shape },
    /// Feature not available in simulation mode.
    SimulationOnly(&'static str),
}

// ---------------------------------------------------------------------------
// Buffer State Tracking
// ---------------------------------------------------------------------------

/// Tracks the lifecycle state of a PJRT buffer.
///
/// This prevents use-after-free and data races by making it
/// impossible to read from a buffer that's still being computed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufferState {
    /// Allocated but no data written yet.
    Uninitialized,
    /// Data transfer from host to device is in progress.
    Transferring,
    /// Computation producing this buffer is in progress.
    Computing,
    /// Buffer is ready to be read or used as input.
    Ready,
    /// Buffer has been deallocated.
    Freed,
}

// ---------------------------------------------------------------------------
// PJRT Buffer
// ---------------------------------------------------------------------------

/// A tensor stored in TPU HBM with ownership semantics.
///
/// When dropped, the HBM allocation is freed deterministically —
/// unlike Python where GC delays can cause OOM on large models.
#[derive(Debug)]
pub struct PjrtBuffer {
    /// Unique buffer ID within the client.
    pub id: u64,
    /// Tensor shape.
    pub shape: Shape,
    /// Data type.
    pub dtype: DType,
    /// Current lifecycle state.
    pub state: BufferState,
    /// Size in bytes on HBM.
    pub hbm_bytes: usize,
    /// Device ordinal this buffer resides on.
    pub device_ordinal: u32,
    /// Simulation: holds the actual data in CPU memory.
    sim_data: Vec<f32>,
}

impl PjrtBuffer {
    /// Create a new buffer (simulation mode).
    fn new_sim(id: u64, shape: Shape, dtype: DType, device: u32) -> Self {
        let hbm_bytes = shape.total_bytes(dtype);
        let n_elements = shape.num_elements();
        Self {
            id,
            shape,
            dtype,
            state: BufferState::Uninitialized,
            hbm_bytes,
            device_ordinal: device,
            sim_data: vec![0.0; n_elements],
        }
    }

    /// Copy host data into this buffer (simulation mode).
    pub fn copy_from_host(&mut self, data: &[f32]) -> Result<(), PjrtError> {
        if data.len() != self.shape.num_elements() {
            return Err(PjrtError::ShapeMismatch {
                expected: self.shape.clone(),
                got: Shape::vector(data.len()),
            });
        }
        self.sim_data.copy_from_slice(data);
        self.state = BufferState::Ready;
        Ok(())
    }

    /// Copy device data to host (simulation mode).
    pub fn copy_to_host(&self) -> Result<Vec<f32>, PjrtError> {
        if self.state != BufferState::Ready {
            return Err(PjrtError::BufferNotReady);
        }
        Ok(self.sim_data.clone())
    }

    /// Check if buffer data is ready to read.
    pub fn is_ready(&self) -> bool {
        self.state == BufferState::Ready
    }

    /// Get a reference to simulation data.
    pub fn sim_data(&self) -> &[f32] {
        &self.sim_data
    }

    /// Get a mutable reference to simulation data.
    pub fn sim_data_mut(&mut self) -> &mut [f32] {
        &mut self.sim_data
    }
}

// ---------------------------------------------------------------------------
// PJRT Device
// ---------------------------------------------------------------------------

/// Represents a single TPU chip.
#[derive(Debug, Clone)]
pub struct PjrtDevice {
    /// Device ordinal (0-indexed).
    pub ordinal: u32,
    /// Device kind (e.g., "TPU v5e", "TPU v6e").
    pub kind: &'static str,
    /// Total HBM capacity in bytes.
    pub hbm_total: usize,
    /// Currently allocated HBM in bytes.
    pub hbm_used: usize,
    /// ICI (Inter-Chip Interconnect) links count.
    pub ici_links: u32,
}

impl PjrtDevice {
    /// Available HBM in bytes.
    pub fn hbm_available(&self) -> usize {
        self.hbm_total.saturating_sub(self.hbm_used)
    }

    /// HBM utilization as percentage.
    pub fn hbm_utilization_pct(&self) -> f32 {
        if self.hbm_total == 0 {
            return 0.0;
        }
        self.hbm_used as f32 / self.hbm_total as f32 * 100.0
    }
}

// ---------------------------------------------------------------------------
// PJRT Executable
// ---------------------------------------------------------------------------

/// A compiled StableHLO program ready for execution on TPU.
///
/// This is `Send + Sync` — safe to share across threads for
/// concurrent dispatch to multiple TPU chips.
#[derive(Debug)]
pub struct PjrtExecutable {
    /// Unique executable ID.
    pub id: u64,
    /// Human-readable name.
    pub name: String,
    /// Number of input buffers expected.
    pub n_inputs: usize,
    /// Number of output buffers produced.
    pub n_outputs: usize,
    /// Estimated FLOPs per execution.
    pub estimated_flops: u64,
    /// Estimated HBM bytes needed (scratch + outputs).
    pub estimated_hbm: usize,
}

// Safety: executable metadata is immutable after compilation.
unsafe impl Send for PjrtExecutable {}
unsafe impl Sync for PjrtExecutable {}

// ---------------------------------------------------------------------------
// PJRT Client
// ---------------------------------------------------------------------------

/// The main entry point for TPU interaction.
///
/// Owns all devices, manages buffer allocation, and dispatches
/// compiled programs for execution.
pub struct PjrtClient {
    /// Available devices.
    devices: Vec<PjrtDevice>,
    /// All allocated buffers.
    buffers: Vec<PjrtBuffer>,
    /// Next buffer ID.
    next_buffer_id: u64,
    /// Next executable ID.
    next_exe_id: u64,
    /// TPU generation.
    generation: TpuGeneration,
    /// Platform name.
    platform: &'static str,
}

/// TPU generation for configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TpuGeneration {
    V5e,
    V6e,
    Simulator,
}

impl PjrtClient {
    /// Create a simulated TPU v5e client (1 chip).
    pub fn sim_v5e(n_chips: u32) -> Self {
        let devices: Vec<PjrtDevice> = (0..n_chips)
            .map(|i| PjrtDevice {
                ordinal: i,
                kind: "TPU v5e (simulated)",
                hbm_total: 16 * 1024 * 1024 * 1024, // 16 GB
                hbm_used: 0,
                ici_links: 4,
            })
            .collect();

        Self {
            devices,
            buffers: Vec::new(),
            next_buffer_id: 1,
            next_exe_id: 1,
            generation: TpuGeneration::V5e,
            platform: "TPU v5e (RunuX Simulator)",
        }
    }

    /// Create a simulated TPU v6e client (1 chip).
    pub fn sim_v6e(n_chips: u32) -> Self {
        let devices: Vec<PjrtDevice> = (0..n_chips)
            .map(|i| PjrtDevice {
                ordinal: i,
                kind: "TPU v6e (simulated)",
                hbm_total: 32usize * 1024 * 1024 * 1024, // 32 GB
                hbm_used: 0,
                ici_links: 6,
            })
            .collect();

        Self {
            devices,
            buffers: Vec::new(),
            next_buffer_id: 1,
            next_exe_id: 1,
            generation: TpuGeneration::V6e,
            platform: "TPU v6e Trillium (RunuX Simulator)",
        }
    }

    /// Platform name.
    pub fn platform_name(&self) -> &str {
        self.platform
    }

    /// TPU generation.
    pub fn generation(&self) -> TpuGeneration {
        self.generation
    }

    /// Number of available devices.
    pub fn device_count(&self) -> usize {
        self.devices.len()
    }

    /// Get device by ordinal.
    pub fn device(&self, ordinal: u32) -> Option<&PjrtDevice> {
        self.devices.iter().find(|d| d.ordinal == ordinal)
    }

    /// All devices.
    pub fn devices(&self) -> &[PjrtDevice] {
        &self.devices
    }

    /// MXU dimension for this TPU generation.
    pub fn mxu_dim(&self) -> usize {
        match self.generation {
            TpuGeneration::V5e | TpuGeneration::Simulator => 128,
            TpuGeneration::V6e => 256,
        }
    }

    /// Allocate a buffer on a specific device.
    pub fn alloc_buffer(
        &mut self,
        shape: &Shape,
        dtype: DType,
        device: u32,
    ) -> Result<u64, PjrtError> {
        let dev = self
            .devices
            .iter_mut()
            .find(|d| d.ordinal == device)
            .ok_or(PjrtError::DeviceUnavailable)?;

        let hbm_needed = shape.total_bytes(dtype);
        if dev.hbm_used + hbm_needed > dev.hbm_total {
            return Err(PjrtError::OutOfMemory {
                requested: hbm_needed,
                available: dev.hbm_total - dev.hbm_used,
            });
        }

        let id = self.next_buffer_id;
        self.next_buffer_id += 1;

        let buffer = PjrtBuffer::new_sim(id, shape.clone(), dtype, device);
        dev.hbm_used += hbm_needed;
        self.buffers.push(buffer);

        Ok(id)
    }

    /// Get a buffer by ID.
    pub fn buffer(&self, id: u64) -> Option<&PjrtBuffer> {
        self.buffers.iter().find(|b| b.id == id)
    }

    /// Get a mutable buffer by ID.
    pub fn buffer_mut(&mut self, id: u64) -> Option<&mut PjrtBuffer> {
        self.buffers.iter_mut().find(|b| b.id == id)
    }

    /// Free a buffer and reclaim HBM.
    pub fn free_buffer(&mut self, id: u64) -> Result<(), PjrtError> {
        let buf_idx = self
            .buffers
            .iter()
            .position(|b| b.id == id)
            .ok_or(PjrtError::BufferNotReady)?;

        let buf = &self.buffers[buf_idx];
        let device_ord = buf.device_ordinal;
        let hbm_freed = buf.hbm_bytes;

        if let Some(dev) = self.devices.iter_mut().find(|d| d.ordinal == device_ord) {
            dev.hbm_used = dev.hbm_used.saturating_sub(hbm_freed);
        }

        self.buffers.remove(buf_idx);
        Ok(())
    }

    /// Create a compiled executable (simulation: just records metadata).
    pub fn compile(
        &mut self,
        name: &str,
        n_inputs: usize,
        n_outputs: usize,
        estimated_flops: u64,
    ) -> PjrtExecutable {
        let id = self.next_exe_id;
        self.next_exe_id += 1;

        PjrtExecutable {
            id,
            name: String::from(name),
            n_inputs,
            n_outputs,
            estimated_flops,
            estimated_hbm: 0,
        }
    }

    /// Transfer data from host to device buffer.
    pub fn transfer_to_device(
        &mut self,
        data: &[f32],
        shape: &Shape,
        dtype: DType,
        device: u32,
    ) -> Result<u64, PjrtError> {
        let id = self.alloc_buffer(shape, dtype, device)?;
        let buf = self.buffer_mut(id).unwrap();
        buf.copy_from_host(data)?;
        Ok(id)
    }

    /// Transfer data from device to host.
    pub fn transfer_to_host(&self, buffer_id: u64) -> Result<Vec<f32>, PjrtError> {
        let buf = self.buffer(buffer_id).ok_or(PjrtError::BufferNotReady)?;
        buf.copy_to_host()
    }

    /// Total HBM used across all devices.
    pub fn total_hbm_used(&self) -> usize {
        self.devices.iter().map(|d| d.hbm_used).sum()
    }

    /// Total HBM available across all devices.
    pub fn total_hbm_available(&self) -> usize {
        self.devices.iter().map(|d| d.hbm_available()).sum()
    }
}

// ---------------------------------------------------------------------------
// Model Loading Utilities
// ---------------------------------------------------------------------------

/// Estimate HBM needed for a model on TPU.
#[derive(Debug, Clone)]
pub struct TpuModelPlan {
    /// Model name.
    pub model_name: &'static str,
    /// Total model parameters.
    pub total_params: u64,
    /// Weight dtype.
    pub weight_dtype: DType,
    /// Weights size in bytes.
    pub weights_bytes: usize,
    /// KV-cache size per layer per token (bytes).
    pub kv_per_layer_per_token: usize,
    /// Number of layers.
    pub n_layers: usize,
    /// Maximum sequence length that fits in remaining HBM.
    pub max_seq_len: usize,
    /// Activation memory per token (bytes).
    pub activation_bytes: usize,
    /// Whether the model fits on this TPU.
    pub fits: bool,
}

/// Plan model placement on TPU.
pub fn plan_tpu_model(
    model_name: &'static str,
    total_params: u64,
    hidden_dim: usize,
    n_layers: usize,
    n_kv_heads: usize,
    head_dim: usize,
    weight_dtype: DType,
    hbm_bytes: usize,
) -> TpuModelPlan {
    let weights_bytes = (total_params as f32 * weight_dtype.bytes_per_element()) as usize;

    // KV-cache: 2 (K+V) × n_kv_heads × head_dim × bytes_per_element
    let kv_per_layer_per_token =
        2 * n_kv_heads * head_dim * weight_dtype.bytes_per_element() as usize;

    // Activation memory: ~4× hidden_dim per layer per token (Q,K,V,FFN)
    let activation_bytes = 4 * hidden_dim * 4; // FP32 activations

    // Remaining HBM after weights
    let hbm_remaining = hbm_bytes.saturating_sub(weights_bytes + activation_bytes);

    // Total KV per token across all layers
    let kv_per_token_total = kv_per_layer_per_token * n_layers;

    let max_seq_len = if kv_per_token_total > 0 {
        hbm_remaining / kv_per_token_total
    } else {
        0
    };

    let fits = weights_bytes < hbm_bytes && max_seq_len >= 512;

    TpuModelPlan {
        model_name,
        total_params,
        weight_dtype,
        weights_bytes,
        kv_per_layer_per_token,
        n_layers,
        max_seq_len,
        activation_bytes,
        fits,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_creation() {
        let client = PjrtClient::sim_v5e(1);
        assert_eq!(client.device_count(), 1);
        assert_eq!(client.mxu_dim(), 128);
        assert!(client.platform_name().contains("v5e"));

        let client_v6 = PjrtClient::sim_v6e(4);
        assert_eq!(client_v6.device_count(), 4);
        assert_eq!(client_v6.mxu_dim(), 256);
    }

    #[test]
    fn test_buffer_allocation() {
        let mut client = PjrtClient::sim_v5e(1);
        let shape = Shape::matrix(1024, 1024);

        let id = client.alloc_buffer(&shape, DType::BF16, 0).unwrap();
        assert_eq!(id, 1);

        let buf = client.buffer(id).unwrap();
        assert_eq!(buf.shape.num_elements(), 1024 * 1024);
        assert_eq!(buf.hbm_bytes, 1024 * 1024 * 2); // BF16
        assert_eq!(buf.state, BufferState::Uninitialized);
    }

    #[test]
    fn test_buffer_transfer() {
        let mut client = PjrtClient::sim_v5e(1);
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let shape = Shape::vector(4);

        let id = client
            .transfer_to_device(&data, &shape, DType::F32, 0)
            .unwrap();
        let result = client.transfer_to_host(id).unwrap();
        assert_eq!(result, data);
    }

    #[test]
    fn test_oom_detection() {
        let mut client = PjrtClient::sim_v5e(1);
        // Try to allocate more than 16 GB HBM
        let huge = Shape::matrix(65536, 65537); // >16 GB FP32
        let result = client.alloc_buffer(&huge, DType::F32, 0);
        assert!(result.is_err(), "Should fail: >16GB exceeds HBM");
    }

    #[test]
    fn test_buffer_lifecycle() {
        let mut client = PjrtClient::sim_v5e(1);
        let shape = Shape::vector(1024);

        let id = client.alloc_buffer(&shape, DType::F32, 0).unwrap();
        assert!(client.total_hbm_used() > 0);

        let hbm_before = client.total_hbm_used();
        client.free_buffer(id).unwrap();
        assert!(client.total_hbm_used() < hbm_before);
    }

    #[test]
    fn test_multi_device() {
        let mut client = PjrtClient::sim_v6e(4);
        let shape = Shape::matrix(1024, 1024);

        // Allocate on different devices
        for dev in 0..4u32 {
            let id = client.alloc_buffer(&shape, DType::BF16, dev).unwrap();
            assert!(client.buffer(id).unwrap().device_ordinal == dev);
        }
        assert_eq!(client.device_count(), 4);
    }

    #[test]
    fn test_compile_executable() {
        let mut client = PjrtClient::sim_v5e(1);
        let exe = client.compile(
            "flash_attention_fwd",
            3,             // Q, K, V
            1,             // output
            1_000_000_000, // 1 GFLOP
        );
        assert_eq!(exe.name, "flash_attention_fwd");
        assert_eq!(exe.n_inputs, 3);
        assert_eq!(exe.n_outputs, 1);
    }

    #[test]
    fn test_model_planning_7b() {
        // Llama 7B on TPU v5e
        let plan = plan_tpu_model(
            "Llama 7B",
            7_000_000_000,
            4096, // hidden_dim
            32,   // n_layers
            32,   // n_kv_heads
            128,  // head_dim
            DType::BF16,
            16 * 1024 * 1024 * 1024, // v5e: 16 GB
        );

        assert!(plan.fits, "7B BF16 should fit on v5e (14GB weights + KV)");
        assert!(
            plan.max_seq_len >= 512,
            "Should support at least 512 tokens"
        );
    }

    #[test]
    fn test_model_planning_70b() {
        // Llama 70B on TPU v5e — should NOT fit
        let plan = plan_tpu_model(
            "Llama 70B",
            70_000_000_000,
            8192,
            80,
            8,
            128,
            DType::BF16,
            16 * 1024 * 1024 * 1024, // v5e: 16 GB
        );

        assert!(!plan.fits, "70B BF16 needs 140GB — won't fit on 16GB v5e");
    }

    #[test]
    fn test_device_hbm_tracking() {
        let mut client = PjrtClient::sim_v5e(1);
        let dev = client.device(0).unwrap();
        assert_eq!(dev.hbm_utilization_pct(), 0.0);

        let shape = Shape::matrix(1024, 1024);
        client.alloc_buffer(&shape, DType::F32, 0).unwrap(); // 4MB

        let dev = client.device(0).unwrap();
        assert!(dev.hbm_utilization_pct() > 0.0);
        assert!(dev.hbm_available() < dev.hbm_total);
    }
}
