// Copyright (c) 2026 Xavier Callens / Socrate AI. All Rights Reserved.
// SPDX-License-Identifier: LicenseRef-RunuX-Commercial
// This file is proprietary and confidential. Unauthorized copying,
// distribution, or use is strictly prohibited.

#![cfg_attr(not(test), no_std)]
#![deny(clippy::all)]
#![warn(clippy::pedantic)]
//! RunuX HAL — Hardware Abstraction Layer
//!
//! Provides a unified interface across all accelerator backends:
//! - **RISC-V** (SpacemiT K1/K3): RVV 1.0 vector extensions
//! - **TPU** (Google v5e/v6e): PJRT C API + StableHLO programs
//! - **GPU** (PowerVR BXM-4-64): Compute shaders
//! - **CPU** (x86/ARM): Reference implementation
//!
//! # Design Principles
//!
//! 1. **Compile-time dispatch**: `impl<B: Accelerator>` — monomorphized,
//!    zero vtable overhead. Each backend is a concrete type.
//! 2. **no_std compatible**: Core abstractions work without allocator.
//! 3. **Backend-specific tiling**: Each accelerator provides optimal tile
//!    sizes for its hardware (MXU 128×128, RVV VLEN=256, etc.)
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────┐
//! │         Application Layer               │
//! │  (sim_inference, sim_train, sim_bench)   │
//! └───────────────┬─────────────────────────┘
//!                 │  impl<B: Accelerator>
//! ┌───────────────▼─────────────────────────┐
//! │          HAL Accelerator Trait           │
//! │  matmul · flash_attn · softmax · norm   │
//! └───┬───────┬───────┬───────┬─────────────┘
//!     │       │       │       │
//!  RiscV    Tpu     Gpu     Cpu
//!  (RVV)  (PJRT) (PowerVR) (scalar)
//! ```

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;
use alloc::string::String;

// ---------------------------------------------------------------------------
// Core Types
// ---------------------------------------------------------------------------

/// Data types supported across all backends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
    F16,
    BF16,
    FP8E4M3,
    FP8E5M2,
    INT8,
    INT4,
    Q4KM,
    Q8_0,
}

impl DType {
    /// Bytes per element (fractional for sub-byte types).
    pub fn bytes_per_element(&self) -> f32 {
        match self {
            DType::F32 => 4.0,
            DType::F16 | DType::BF16 => 2.0,
            DType::FP8E4M3 | DType::FP8E5M2 | DType::INT8 | DType::Q8_0 => 1.0,
            DType::INT4 | DType::Q4KM => 0.5,
        }
    }

    /// Whether this dtype is natively supported on TPU MXUs.
    pub fn is_tpu_native(&self) -> bool {
        matches!(self, DType::BF16 | DType::FP8E4M3 | DType::FP8E5M2 | DType::INT8)
    }

    /// Whether this dtype is natively supported on RISC-V RVV.
    pub fn is_riscv_native(&self) -> bool {
        matches!(self, DType::F32 | DType::F16 | DType::FP8E4M3)
    }
}

/// Tensor shape descriptor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shape {
    pub dims: Vec<usize>,
}

impl Shape {
    pub fn new(dims: &[usize]) -> Self {
        Self { dims: dims.to_vec() }
    }

    /// Scalar shape.
    pub fn scalar() -> Self {
        Self { dims: vec![] }
    }

    /// Vector shape.
    pub fn vector(n: usize) -> Self {
        Self { dims: vec![n] }
    }

    /// Matrix shape.
    pub fn matrix(rows: usize, cols: usize) -> Self {
        Self { dims: vec![rows, cols] }
    }

    /// Total number of elements.
    pub fn num_elements(&self) -> usize {
        if self.dims.is_empty() { 1 } else { self.dims.iter().product() }
    }

    /// Number of dimensions.
    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    /// Total bytes for a given dtype.
    pub fn total_bytes(&self, dtype: DType) -> usize {
        (self.num_elements() as f32 * dtype.bytes_per_element()) as usize
    }
}

/// Backend-agnostic tensor descriptor.
///
/// The actual data lives in backend-specific storage (host memory,
/// HBM, VMEM, etc.). This descriptor tracks shape, dtype, and
/// a backend-assigned handle.
#[derive(Debug, Clone)]
pub struct TensorDesc {
    pub shape: Shape,
    pub dtype: DType,
    /// Backend-specific handle ID (e.g., PJRT buffer ID, arena offset).
    pub handle: u64,
    /// Human-readable name for debugging.
    pub name: String,
}

impl TensorDesc {
    pub fn new(name: &str, shape: Shape, dtype: DType) -> Self {
        Self {
            shape,
            dtype,
            handle: 0,
            name: String::from(name),
        }
    }

    pub fn total_bytes(&self) -> usize {
        self.shape.total_bytes(self.dtype)
    }
}

// ---------------------------------------------------------------------------
// Hardware Capabilities
// ---------------------------------------------------------------------------

/// Describes the computational capabilities of a hardware backend.
#[derive(Debug, Clone)]
pub struct HardwareCaps {
    /// Backend name (e.g., "TPU v6e", "SpacemiT K1")
    pub name: &'static str,
    /// Backend type
    pub backend_type: BackendType,
    /// Peak compute in TFLOPS (at native dtype)
    pub peak_tflops: f32,
    /// Native compute dtype
    pub native_dtype: DType,
    /// Total accelerator memory in bytes
    pub memory_bytes: usize,
    /// Memory bandwidth in GB/s
    pub memory_bw_gbs: f32,
    /// Optimal matrix tile size (MXU dim, VLEN elements, etc.)
    pub optimal_tile_size: usize,
    /// Thermal Design Power in watts
    pub tdp_watts: f32,
    /// Roofline ridge point (FLOP/byte at native dtype)
    pub ridge_point: f32,
    /// Whether this backend supports distributed multi-device
    pub supports_distributed: bool,
    /// Inter-chip interconnect bandwidth (GB/s), 0 if N/A
    pub ici_bw_gbs: f32,
}

/// Backend type enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendType {
    /// RISC-V with Vector Extension (RVV 1.0)
    RiscV,
    /// Google TPU (v5e, v6e) via PJRT
    Tpu,
    /// GPU (PowerVR, NVIDIA, etc.)
    Gpu,
    /// CPU reference (x86, ARM)
    Cpu,
}

impl HardwareCaps {
    // --- RISC-V ---

    /// SpacemiT K1 (BPI-F3): 8× X60 cores, RVV 1.0, VLEN=256.
    pub fn spacemit_k1() -> Self {
        Self {
            name: "SpacemiT K1 (BPI-F3)",
            backend_type: BackendType::RiscV,
            peak_tflops: 0.016, // 16 GFLOPS
            native_dtype: DType::F32,
            memory_bytes: 4usize * 1024 * 1024 * 1024, // 4 GB
            memory_bw_gbs: 12.8,
            optimal_tile_size: 8, // VLEN=256 → 8 FP32
            tdp_watts: 8.0,
            ridge_point: 1.25, // 16 / 12.8
            supports_distributed: false,
            ici_bw_gbs: 0.0,
        }
    }

    /// SpacemiT K3 (AIBOX-K3): 8× X100 + 8× A100 AI, VLEN=1024.
    pub fn spacemit_k3() -> Self {
        Self {
            name: "SpacemiT K3 (AIBOX-K3)",
            backend_type: BackendType::RiscV,
            peak_tflops: 0.128, // 128 GFLOPS
            native_dtype: DType::F32,
            memory_bytes: 32usize * 1024 * 1024 * 1024, // 32 GB
            memory_bw_gbs: 51.2,
            optimal_tile_size: 32, // VLEN=1024 → 32 FP32
            tdp_watts: 15.0,
            ridge_point: 2.5,
            supports_distributed: false,
            ici_bw_gbs: 0.0,
        }
    }

    // --- Google TPU ---

    /// Google TPU v5e: 4× MXU 128×128, BF16, 16 GB HBM.
    pub fn tpu_v5e() -> Self {
        Self {
            name: "Google TPU v5e",
            backend_type: BackendType::Tpu,
            peak_tflops: 197.0,
            native_dtype: DType::BF16,
            memory_bytes: 16usize * 1024 * 1024 * 1024, // 16 GB HBM
            memory_bw_gbs: 800.0,
            optimal_tile_size: 128, // MXU 128×128
            tdp_watts: 200.0,
            ridge_point: 0.246, // 197 / 800
            supports_distributed: true,
            ici_bw_gbs: 400.0,
        }
    }

    /// Google TPU v6e (Trillium): 2× MXU 256×256, BF16/FP8, 32 GB HBM.
    pub fn tpu_v6e() -> Self {
        Self {
            name: "Google TPU v6e (Trillium)",
            backend_type: BackendType::Tpu,
            peak_tflops: 918.0,
            native_dtype: DType::BF16,
            memory_bytes: 32usize * 1024 * 1024 * 1024, // 32 GB HBM
            memory_bw_gbs: 1600.0,
            optimal_tile_size: 256, // MXU 256×256
            tdp_watts: 300.0,
            ridge_point: 0.574, // 918 / 1600
            supports_distributed: true,
            ici_bw_gbs: 800.0,
        }
    }

    // --- GPU ---

    /// PowerVR BXM-4-64 (AIBOX-K3 GPU).
    pub fn powervr_bxm4() -> Self {
        Self {
            name: "PowerVR BXM-4-64",
            backend_type: BackendType::Gpu,
            peak_tflops: 0.5,
            native_dtype: DType::F16,
            memory_bytes: 0, // Shared with CPU
            memory_bw_gbs: 51.2,
            optimal_tile_size: 64, // Workgroup size
            tdp_watts: 5.0,
            ridge_point: 0.01,
            supports_distributed: false,
            ici_bw_gbs: 0.0,
        }
    }

    // --- CPU Reference ---

    /// Generic CPU reference backend (for testing).
    pub fn cpu_reference() -> Self {
        Self {
            name: "CPU Reference",
            backend_type: BackendType::Cpu,
            peak_tflops: 0.1,
            native_dtype: DType::F32,
            memory_bytes: 16usize * 1024 * 1024 * 1024, // 16 GB
            memory_bw_gbs: 50.0,
            optimal_tile_size: 8, // AVX2 = 8 FP32
            tdp_watts: 65.0,
            ridge_point: 0.002,
            supports_distributed: false,
            ici_bw_gbs: 0.0,
        }
    }

    /// Energy efficiency: TFLOPS per watt.
    pub fn tflops_per_watt(&self) -> f32 {
        if self.tdp_watts > 0.0 {
            self.peak_tflops / self.tdp_watts
        } else {
            0.0
        }
    }

    /// Memory capacity in GB.
    pub fn memory_gb(&self) -> f32 {
        self.memory_bytes as f32 / (1024.0 * 1024.0 * 1024.0)
    }

    /// Whether an operation with given arithmetic intensity is memory-bound.
    pub fn is_memory_bound(&self, flops: u64, bytes: u64) -> bool {
        if bytes == 0 { return false; }
        let intensity = flops as f32 / bytes as f32;
        intensity < self.ridge_point
    }
}

// ---------------------------------------------------------------------------
// Accelerator Trait
// ---------------------------------------------------------------------------

/// Configuration for FlashAttention on any backend.
#[derive(Debug, Clone)]
pub struct FlashConfig {
    pub n_heads: usize,
    pub head_dim: usize,
    pub tile_q: usize,
    pub tile_kv: usize,
    pub causal: bool,
    pub scale: f32,
}

impl FlashConfig {
    /// Auto-configure for the given hardware.
    pub fn for_hardware(caps: &HardwareCaps, n_heads: usize, head_dim: usize) -> Self {
        let tile = caps.optimal_tile_size.max(16);
        Self {
            n_heads,
            head_dim,
            tile_q: tile,
            tile_kv: tile,
            causal: true,
            scale: 1.0 / fast_sqrt(head_dim as f32),
        }
    }
}

/// Tiling configuration for matrix operations.
#[derive(Debug, Clone, Copy)]
pub struct TileConfig {
    pub tile_m: usize,
    pub tile_n: usize,
    pub tile_k: usize,
}

impl TileConfig {
    /// Auto-configure for the given hardware.
    pub fn for_hardware(caps: &HardwareCaps, m: usize, n: usize, k: usize) -> Self {
        let tile = caps.optimal_tile_size.max(8);
        Self {
            tile_m: tile.min(m),
            tile_n: tile.min(n),
            tile_k: tile.min(k),
        }
    }
}

/// The core hardware abstraction trait.
///
/// All accelerator backends implement this trait, enabling
/// compile-time dispatch via monomorphization:
///
/// ```rust,ignore
/// fn inference<B: Accelerator>(backend: &B, ...) { ... }
/// ```
///
/// This generates specialized code for each backend with zero
/// vtable overhead — a key advantage of Rust's generics.
pub trait Accelerator {
    /// Hardware capabilities descriptor.
    fn caps(&self) -> &HardwareCaps;

    /// Name of this backend instance.
    fn name(&self) -> &str { self.caps().name }

    /// Allocate a tensor on this backend's memory.
    fn alloc_tensor(&self, shape: &Shape, dtype: DType) -> TensorDesc;

    /// Dense matrix multiplication: C = A × B.
    ///
    /// Shapes: A=[M×K], B=[K×N], C=[M×N].
    /// Backend chooses optimal tiling for its hardware.
    fn matmul(
        &self,
        a: &[f32], b: &[f32], c: &mut [f32],
        m: usize, n: usize, k: usize,
    );

    /// FlashAttention forward pass.
    ///
    /// Implements tiled, fused, IO-aware attention with online softmax.
    /// Tile sizes are backend-specific (MXU dim, VLEN, etc.)
    fn flash_attention(
        &self,
        q: &[f32], k: &[f32], v: &[f32],
        output: &mut [f32],
        config: &FlashConfig,
    );

    /// In-place softmax: x[i] = exp(x[i]) / sum(exp(x[j])).
    fn softmax(&self, x: &mut [f32]);

    /// In-place RMS normalization: x = x / RMS(x) * weight.
    fn rms_norm(&self, x: &mut [f32], weight: &[f32], eps: f32);

    /// SiLU activation: x = x * sigmoid(x).
    fn silu(&self, x: &mut [f32]);

    /// Rotary position embeddings.
    fn rope(&self, x: &mut [f32], position: usize, head_dim: usize, theta: f32);

    /// Optimal tile configuration for a matmul of given dimensions.
    fn optimal_tiles(&self, m: usize, n: usize, k: usize) -> TileConfig {
        TileConfig::for_hardware(self.caps(), m, n, k)
    }
}

// ---------------------------------------------------------------------------
// CPU Reference Backend
// ---------------------------------------------------------------------------

/// CPU reference backend — scalar implementation for testing.
///
/// This is the "gold standard" implementation. All other backends
/// must produce results within tolerance of CPU output.
pub struct CpuBackend {
    caps: HardwareCaps,
}

impl CpuBackend {
    pub fn new() -> Self {
        Self { caps: HardwareCaps::cpu_reference() }
    }
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl Accelerator for CpuBackend {
    fn caps(&self) -> &HardwareCaps { &self.caps }

    fn alloc_tensor(&self, shape: &Shape, dtype: DType) -> TensorDesc {
        TensorDesc::new("cpu_tensor", shape.clone(), dtype)
    }

    fn matmul(
        &self,
        a: &[f32], b: &[f32], c: &mut [f32],
        m: usize, n: usize, k: usize,
    ) {
        // Clear output
        for v in c.iter_mut() { *v = 0.0; }
        // Naive O(M×N×K) matmul
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for p in 0..k {
                    sum += a[i * k + p] * b[p * n + j];
                }
                c[i * n + j] = sum;
            }
        }
    }

    fn flash_attention(
        &self,
        q: &[f32], k: &[f32], v: &[f32],
        output: &mut [f32],
        config: &FlashConfig,
    ) {
        let d = config.head_dim;
        let n = q.len() / d;
        let m = k.len() / d;
        let scale = config.scale;

        // Standard O(N²) attention as reference
        for i in 0..n {
            // Compute scores
            let mut scores = vec![f32::NEG_INFINITY; m];
            let mut max_score = f32::NEG_INFINITY;

            for j in 0..m {
                if config.causal && j > i { continue; }
                let mut dot = 0.0f32;
                for dd in 0..d {
                    dot += q[i * d + dd] * k[j * d + dd];
                }
                scores[j] = dot * scale;
                if scores[j] > max_score { max_score = scores[j]; }
            }

            // Softmax
            let mut sum = 0.0f32;
            for j in 0..m {
                if scores[j] == f32::NEG_INFINITY { scores[j] = 0.0; continue; }
                scores[j] = fast_exp(scores[j] - max_score);
                sum += scores[j];
            }
            if sum > 0.0 {
                for j in 0..m { scores[j] /= sum; }
            }

            // Weighted sum
            for dd in 0..d {
                let mut val = 0.0f32;
                for j in 0..m {
                    val += scores[j] * v[j * d + dd];
                }
                output[i * d + dd] = val;
            }
        }
    }

    fn softmax(&self, x: &mut [f32]) {
        if x.is_empty() { return; }
        let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for v in x.iter_mut() {
            *v = fast_exp(*v - max);
            sum += *v;
        }
        if sum > 0.0 {
            for v in x.iter_mut() { *v /= sum; }
        }
    }

    fn rms_norm(&self, x: &mut [f32], weight: &[f32], eps: f32) {
        let n = x.len();
        let mut sum_sq = 0.0f32;
        for &v in x.iter() { sum_sq += v * v; }
        let rms = fast_sqrt(sum_sq / n as f32 + eps);
        let inv_rms = 1.0 / rms;
        for i in 0..n {
            x[i] = x[i] * inv_rms * weight[i.min(weight.len() - 1)];
        }
    }

    fn silu(&self, x: &mut [f32]) {
        for v in x.iter_mut() {
            let sigmoid = 1.0 / (1.0 + fast_exp(-*v));
            *v *= sigmoid;
        }
    }

    fn rope(&self, x: &mut [f32], position: usize, head_dim: usize, theta: f32) {
        let half = head_dim / 2;
        for i in 0..half {
            if 2 * i + 1 >= x.len() { break; }
            let freq = 1.0 / fast_pow(theta, (2 * i) as f32 / head_dim as f32);
            let angle = position as f32 * freq;
            let cos_a = fast_cos(angle);
            let sin_a = fast_sin(angle);
            let x0 = x[2 * i];
            let x1 = x[2 * i + 1];
            x[2 * i] = x0 * cos_a - x1 * sin_a;
            x[2 * i + 1] = x0 * sin_a + x1 * cos_a;
        }
    }
}

// ---------------------------------------------------------------------------
// TPU Simulator Backend
// ---------------------------------------------------------------------------

/// TPU simulator — executes ops on CPU but with TPU tiling and constraints.
///
/// This enables development and testing of the TPU code path without
/// actual TPU hardware. Operations use CPU math but tile sizes and
/// constraints match the target TPU generation.
pub struct TpuSimulatorBackend {
    caps: HardwareCaps,
    cpu: CpuBackend,
}

impl TpuSimulatorBackend {
    /// Create a v5e simulator.
    pub fn v5e() -> Self {
        Self {
            caps: HardwareCaps::tpu_v5e(),
            cpu: CpuBackend::new(),
        }
    }

    /// Create a v6e simulator.
    pub fn v6e() -> Self {
        Self {
            caps: HardwareCaps::tpu_v6e(),
            cpu: CpuBackend::new(),
        }
    }
}

impl Accelerator for TpuSimulatorBackend {
    fn caps(&self) -> &HardwareCaps { &self.caps }

    fn alloc_tensor(&self, shape: &Shape, dtype: DType) -> TensorDesc {
        // Simulate HBM allocation
        let mut desc = TensorDesc::new("tpu_hbm_tensor", shape.clone(), dtype);
        // In real TPU, this would call PJRT_Client_BufferFromHostBuffer
        desc.handle = shape.total_bytes(dtype) as u64;
        desc
    }

    fn matmul(
        &self,
        a: &[f32], b: &[f32], c: &mut [f32],
        m: usize, n: usize, k: usize,
    ) {
        // Simulate MXU-aligned tiled matmul
        // In real TPU: this would be a StableHLO dot_general op
        let tile = self.caps.optimal_tile_size; // 128 (v5e) or 256 (v6e)

        // Tiled matmul matching MXU dimensions
        for v in c.iter_mut() { *v = 0.0; }

        let tm = tile.min(m);
        let tn = tile.min(n);
        let tk = tile.min(k);

        for i0 in (0..m).step_by(tm) {
            for j0 in (0..n).step_by(tn) {
                for p0 in (0..k).step_by(tk) {
                    let i_end = (i0 + tm).min(m);
                    let j_end = (j0 + tn).min(n);
                    let p_end = (p0 + tk).min(k);

                    for i in i0..i_end {
                        for j in j0..j_end {
                            let mut sum = 0.0f32;
                            for p in p0..p_end {
                                sum += a[i * k + p] * b[p * n + j];
                            }
                            c[i * n + j] += sum;
                        }
                    }
                }
            }
        }
    }

    fn flash_attention(
        &self,
        q: &[f32], k: &[f32], v: &[f32],
        output: &mut [f32],
        config: &FlashConfig,
    ) {
        // Use CPU backend for correctness, but with TPU-aligned tile sizes
        let tpu_config = FlashConfig {
            tile_q: self.caps.optimal_tile_size.min(config.tile_q.max(16)),
            tile_kv: self.caps.optimal_tile_size.min(config.tile_kv.max(16)),
            ..config.clone()
        };
        self.cpu.flash_attention(q, k, v, output, &tpu_config);
    }

    fn softmax(&self, x: &mut [f32]) { self.cpu.softmax(x); }
    fn rms_norm(&self, x: &mut [f32], weight: &[f32], eps: f32) {
        self.cpu.rms_norm(x, weight, eps);
    }
    fn silu(&self, x: &mut [f32]) { self.cpu.silu(x); }
    fn rope(&self, x: &mut [f32], position: usize, head_dim: usize, theta: f32) {
        self.cpu.rope(x, position, head_dim, theta);
    }
}

// ---------------------------------------------------------------------------
// Cross-backend comparison utilities
// ---------------------------------------------------------------------------

/// Compare outputs from two backends for correctness.
pub fn compare_outputs(reference: &[f32], test: &[f32]) -> f32 {
    let mut max_err = 0.0f32;
    let len = reference.len().min(test.len());
    for i in 0..len {
        let err = (reference[i] - test[i]).abs();
        if err > max_err { max_err = err; }
    }
    max_err
}

/// Performance comparison result.
#[derive(Debug, Clone)]
pub struct BackendComparison {
    pub backend_a: &'static str,
    pub backend_b: &'static str,
    pub operation: &'static str,
    pub speedup: f32,
    pub max_error: f32,
    pub a_tflops_per_watt: f32,
    pub b_tflops_per_watt: f32,
}

// ---------------------------------------------------------------------------
// Fast math (no_std compatible)
// ---------------------------------------------------------------------------

fn fast_exp(x: f32) -> f32 {
    if x < -88.0 { return 0.0; }
    if x > 88.0 { return f32::MAX; }
    let a = 12_102_203.0f32;
    let b = 1_065_353_216.0f32;
    let bits = (a * x + b) as u32;
    f32::from_bits(bits.min(0x7F80_0000))
}

fn fast_sqrt(x: f32) -> f32 {
    if x <= 0.0 { return 0.0; }
    let mut g = x;
    for _ in 0..5 { g = 0.5 * (g + x / g); }
    g
}

fn fast_pow(base: f32, exp: f32) -> f32 {
    // For integer exponents: exact power-by-squaring
    let n = exp as u32;
    if n == 0 { return 1.0; }
    let mut result = 1.0f32;
    let mut b = base;
    let mut e = n;
    while e > 0 {
        if e & 1 == 1 { result *= b; }
        b *= b;
        e >>= 1;
    }
    result
}

fn fast_sin(x: f32) -> f32 {
    let pi = core::f32::consts::PI;
    let two_pi = 2.0 * pi;
    let mut a = x % two_pi;
    if a > pi { a -= two_pi; }
    if a < -pi { a += two_pi; }
    let abs_a = if a < 0.0 { -a } else { a };
    let y = 4.0 / pi * a - 4.0 / (pi * pi) * a * abs_a;
    0.225 * (y * (if y < 0.0 { -y } else { y }) - y) + y
}

fn fast_cos(x: f32) -> f32 {
    fast_sin(x + core::f32::consts::FRAC_PI_2)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_matmul() {
        let cpu = CpuBackend::new();
        let a = vec![1.0, 2.0, 3.0, 4.0]; // 2×2
        let b = vec![5.0, 6.0, 7.0, 8.0]; // 2×2
        let mut c = vec![0.0f32; 4];
        cpu.matmul(&a, &b, &mut c, 2, 2, 2);
        // C = [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]]
        //   = [[19, 22], [43, 50]]
        assert!((c[0] - 19.0).abs() < 1e-5);
        assert!((c[1] - 22.0).abs() < 1e-5);
        assert!((c[2] - 43.0).abs() < 1e-5);
        assert!((c[3] - 50.0).abs() < 1e-5);
    }

    #[test]
    fn test_cpu_softmax() {
        let cpu = CpuBackend::new();
        let mut x = vec![1.0, 2.0, 3.0];
        cpu.softmax(&mut x);
        let sum: f32 = x.iter().sum();
        assert!((sum - 1.0).abs() < 0.01, "Softmax should sum to 1.0, got {}", sum);
        assert!(x[2] > x[1] && x[1] > x[0], "Softmax should preserve ordering");
    }

    #[test]
    fn test_cpu_rms_norm() {
        let cpu = CpuBackend::new();
        let mut x = vec![1.0, 2.0, 3.0, 4.0];
        let w = vec![1.0; 4];
        cpu.rms_norm(&mut x, &w, 1e-6);
        // RMS of [1,2,3,4] = sqrt((1+4+9+16)/4) = sqrt(7.5) ≈ 2.738
        // Normalized: [0.365, 0.730, 1.095, 1.461]
        assert!(x[0] < x[1] && x[1] < x[2] && x[2] < x[3]);
        let norm_sq: f32 = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
        assert!((norm_sq - 1.0).abs() < 0.1, "RMS norm should normalize to ~1, got {}", norm_sq);
    }

    #[test]
    fn test_cpu_silu() {
        let cpu = CpuBackend::new();
        let mut x = vec![0.0, 1.0, -1.0];
        cpu.silu(&mut x);
        assert!((x[0] - 0.0).abs() < 0.01); // silu(0) = 0
        assert!(x[1] > 0.5); // silu(1) ≈ 0.731
        assert!(x[2] < 0.0 && x[2] > -0.5); // silu(-1) ≈ -0.269
    }

    #[test]
    fn test_cpu_flash_attention() {
        let cpu = CpuBackend::new();
        let d = 4;
        let n = 4;
        let q = vec![0.5f32; n * d];
        let k = vec![0.5f32; n * d];
        let v: Vec<f32> = (0..n * d).map(|i| i as f32 * 0.1).collect();
        let mut output = vec![0.0f32; n * d];

        let config = FlashConfig::for_hardware(&cpu.caps(), 1, d);
        cpu.flash_attention(&q, &k, &v, &mut output, &config);

        assert_eq!(output.len(), n * d);
        // First token attends only to itself (causal)
        assert!(!output[0].is_nan(), "Output should not be NaN");
    }

    #[test]
    fn test_tpu_simulator_matmul() {
        let tpu = TpuSimulatorBackend::v5e();
        let cpu = CpuBackend::new();

        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]; // 3×3
        let b = vec![9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]; // 3×3
        let mut c_tpu = vec![0.0f32; 9];
        let mut c_cpu = vec![0.0f32; 9];

        tpu.matmul(&a, &b, &mut c_tpu, 3, 3, 3);
        cpu.matmul(&a, &b, &mut c_cpu, 3, 3, 3);

        let err = compare_outputs(&c_cpu, &c_tpu);
        assert!(err < 1e-5, "TPU sim should match CPU, max error: {}", err);
    }

    #[test]
    fn test_tpu_simulator_attention() {
        let tpu = TpuSimulatorBackend::v6e();
        let cpu = CpuBackend::new();

        let d = 8;
        let n = 8;
        let q: Vec<f32> = (0..n * d).map(|i| fast_sin(i as f32 * 0.1)).collect();
        let k: Vec<f32> = (0..n * d).map(|i| fast_cos(i as f32 * 0.13)).collect();
        let v: Vec<f32> = (0..n * d).map(|i| fast_sin(i as f32 * 0.07)).collect();

        let mut out_tpu = vec![0.0f32; n * d];
        let mut out_cpu = vec![0.0f32; n * d];

        let config = FlashConfig::for_hardware(&tpu.caps(), 1, d);
        tpu.flash_attention(&q, &k, &v, &mut out_tpu, &config);
        cpu.flash_attention(&q, &k, &v, &mut out_cpu, &config);

        let err = compare_outputs(&out_cpu, &out_tpu);
        assert!(err < 0.2, "TPU sim should match CPU, max error: {}", err);
    }

    #[test]
    fn test_hardware_caps() {
        let k1 = HardwareCaps::spacemit_k1();
        let v5e = HardwareCaps::tpu_v5e();
        let v6e = HardwareCaps::tpu_v6e();

        // TPU should be much faster than RISC-V
        assert!(v5e.peak_tflops > k1.peak_tflops * 1000.0);
        assert!(v6e.peak_tflops > v5e.peak_tflops * 4.0);

        // TPU v6e is more energy efficient than v5e per TFLOP
        assert!(v6e.tflops_per_watt() > v5e.tflops_per_watt());

        // RISC-V K1 is much more power-efficient per-watt absolute
        assert!(k1.tdp_watts < v5e.tdp_watts);
    }

    #[test]
    fn test_dtype_properties() {
        assert!(DType::BF16.is_tpu_native());
        assert!(!DType::F32.is_tpu_native());
        assert!(DType::F32.is_riscv_native());
        assert!(!DType::BF16.is_riscv_native());
        assert_eq!(DType::Q4KM.bytes_per_element(), 0.5);
        assert_eq!(DType::BF16.bytes_per_element(), 2.0);
    }

    #[test]
    fn test_shape_operations() {
        let s = Shape::matrix(128, 256);
        assert_eq!(s.num_elements(), 128 * 256);
        assert_eq!(s.rank(), 2);
        assert_eq!(s.total_bytes(DType::F32), 128 * 256 * 4);
        assert_eq!(s.total_bytes(DType::BF16), 128 * 256 * 2);
    }

    #[test]
    fn test_tile_config() {
        let v5e = HardwareCaps::tpu_v5e();
        let tiles = TileConfig::for_hardware(&v5e, 512, 512, 512);
        assert_eq!(tiles.tile_m, 128); // MXU dim
        assert_eq!(tiles.tile_n, 128);

        let k1 = HardwareCaps::spacemit_k1();
        let tiles_rv = TileConfig::for_hardware(&k1, 512, 512, 512);
        assert_eq!(tiles_rv.tile_m, 8); // VLEN / 32-bit
    }

    #[test]
    fn test_roofline_classification() {
        let v5e = HardwareCaps::tpu_v5e();
        // Large matmul: compute-bound on TPU
        // 2*M*N*K FLOPs, M*K*2 + K*N*2 bytes (BF16)
        let m: u64 = 1024;
        let flops = 2 * m * m * m;
        let bytes = 3 * m * m * 2; // BF16
        assert!(!v5e.is_memory_bound(flops as u64, bytes as u64),
            "Large matmul should be compute-bound on TPU");

        // Small batch-1 decode: memory-bound
        let decode_flops: u64 = 2 * 1024; // 1 × 1024 × 1
        let decode_bytes: u64 = 1024 * 1024 * 2; // load full weight matrix
        assert!(v5e.is_memory_bound(decode_flops as u64, decode_bytes as u64),
            "Batch-1 decode should be memory-bound on TPU");
    }
}
