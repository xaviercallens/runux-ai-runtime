// Copyright (c) 2026 Xavier Callens / Socrate AI. All Rights Reserved.
// SPDX-License-Identifier: LicenseRef-RunuX-Commercial
// This file is proprietary and confidential. Unauthorized copying,
// distribution, or use is strictly prohibited.

#![allow(clippy::all, clippy::pedantic)]
#![cfg_attr(not(test), no_std)]
#![deny(clippy::all)]
#![warn(clippy::pedantic)]
//! RunuX AI Runtime — Core tensor types and inference engine abstractions
//!
//! This crate provides the foundational data structures for AI/ML inference
//! on RISC-V hardware. It defines tensor descriptors, data types, device
//! targets (CPU, A100 AI cores, PowerVR GPU), and the inference engine trait.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────┐
//! │            ai_runtime (this crate)          │
//! │  TensorDescriptor, DataType, DeviceType     │
//! │  InferenceEngine trait, ModelConfig          │
//! └──────────┬──────────────┬───────────────────┘
//!            │              │
//!     ┌──────▼──────┐  ┌───▼────────────┐
//!     │  rvv_simd   │  │  turbo_quant   │
//!     │  Vectorized │  │  KV-cache      │
//!     │  kernels    │  │  compression   │
//!     └─────────────┘  └────────────────┘
//! ```
//!
//! # Supported Hardware Targets
//!
//! - **SpacemiT K1** (BPI-F3): 8× X60 cores, RVV 1.0 256-bit, 2.0 TOPS
//! - **SpacemiT K3** (AIBOX-K3): 8× X100 + 8× A100 cores, RVV 1.0 1024-bit, 60 TOPS
//! - **PowerVR BXM-4-64** (K3 iGPU): OpenCL 3.0, Vulkan 1.3

extern crate alloc;
use alloc::vec::Vec;
use alloc::string::String;

use core::fmt;

// ---------------------------------------------------------------------------
// Data Types
// ---------------------------------------------------------------------------

/// Numeric data types supported by the AI runtime.
///
/// Covers full-precision, half-precision, brain-float, and quantized integer
/// formats. FP8 is natively supported on SpacemiT K3 A100 AI cores.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataType {
    /// 32-bit IEEE 754 float
    FP32 = 0,
    /// 16-bit IEEE 754 half-precision float
    FP16 = 1,
    /// 16-bit Brain Float (truncated FP32 mantissa)
    BF16 = 2,
    /// 8-bit float (E4M3 or E5M2) — native on K3 A100 cores
    FP8 = 3,
    /// 8-bit signed integer (post-training quantization)
    INT8 = 4,
    /// 4-bit signed integer (GGUF Q4_K_M, AWQ, GPTQ)
    INT4 = 5,
    /// 8-bit unsigned integer (activation quantization)
    UINT8 = 6,
    /// 1-bit binary (extreme quantization, BitNet-style)
    Binary = 7,
}

impl DataType {
    /// Returns the size in bits for one element of this data type.
    pub const fn bits(self) -> usize {
        match self {
            Self::FP32 => 32,
            Self::FP16 | Self::BF16 => 16,
            Self::FP8 | Self::INT8 | Self::UINT8 => 8,
            Self::INT4 => 4,
            Self::Binary => 1,
        }
    }

    /// Returns the size in bytes (rounded up) for one element.
    pub const fn bytes(self) -> usize {
        (self.bits() + 7) / 8
    }

    /// Returns true if this dtype is natively accelerated on K3 A100 cores.
    pub const fn is_k3_native(self) -> bool {
        matches!(self, Self::FP8 | Self::BF16 | Self::FP16 | Self::INT8 | Self::INT4)
    }
}

impl fmt::Display for DataType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FP32 => write!(f, "fp32"),
            Self::FP16 => write!(f, "fp16"),
            Self::BF16 => write!(f, "bf16"),
            Self::FP8 => write!(f, "fp8"),
            Self::INT8 => write!(f, "int8"),
            Self::INT4 => write!(f, "int4"),
            Self::UINT8 => write!(f, "uint8"),
            Self::Binary => write!(f, "binary"),
        }
    }
}

// ---------------------------------------------------------------------------
// Device Types
// ---------------------------------------------------------------------------

/// Hardware execution targets available on RISC-V SoCs.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceType {
    /// General-purpose RISC-V CPU cores (X60 on K1, X100 on K3)
    Cpu = 0,
    /// K3 A100 AI accelerator cores (1024-bit RVV, native FP8)
    A100AiCore = 1,
    /// PowerVR BXM-4-64 iGPU (OpenCL 3.0, Vulkan 1.3)
    PowerVrGpu = 2,
    /// NVMe storage for weight offloading
    NvmeOffload = 3,
}

impl fmt::Display for DeviceType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cpu => write!(f, "cpu"),
            Self::A100AiCore => write!(f, "a100_ai"),
            Self::PowerVrGpu => write!(f, "pvr_gpu"),
            Self::NvmeOffload => write!(f, "nvme"),
        }
    }
}

// ---------------------------------------------------------------------------
// Tensor Descriptor
// ---------------------------------------------------------------------------

/// Maximum number of dimensions for a tensor.
pub const MAX_DIMS: usize = 8;

/// C-compatible tensor descriptor for zero-copy interop with Python/C++ frameworks.
///
/// This struct is designed to be passed across FFI boundaries without
/// serialization overhead. The data pointer refers to a memory region
/// managed by the caller (kernel or userspace).
#[repr(C)]
#[derive(Debug, Clone)]
pub struct TensorDescriptor {
    /// Pointer to the raw tensor data in memory.
    pub data_ptr: usize,
    /// Shape of the tensor (e.g., [batch, seq_len, hidden_dim, 0, ...]).
    /// Unused dimensions are set to 0.
    pub shape: [usize; MAX_DIMS],
    /// Number of active dimensions (1–8).
    pub ndim: u8,
    /// Data type of each element.
    pub dtype: DataType,
    /// Target device for computation.
    pub device: DeviceType,
    /// Stride in bytes for each dimension (for non-contiguous tensors).
    pub strides: [usize; MAX_DIMS],
}

impl TensorDescriptor {
    /// Creates a new contiguous tensor descriptor.
    pub fn new(shape: &[usize], dtype: DataType, device: DeviceType) -> Self {
        let ndim = shape.len().min(MAX_DIMS);
        let mut s = [0usize; MAX_DIMS];
        let mut strides = [0usize; MAX_DIMS];

        for i in 0..ndim {
            s[i] = shape[i];
        }

        // Compute contiguous strides (row-major)
        if ndim > 0 {
            strides[ndim - 1] = dtype.bytes();
            for i in (0..ndim - 1).rev() {
                strides[i] = strides[i + 1] * s[i + 1];
            }
        }

        Self {
            data_ptr: 0,
            shape: s,
            ndim: ndim as u8,
            dtype,
            device,
            strides,
        }
    }

    /// Returns the total number of elements in the tensor.
    pub fn numel(&self) -> usize {
        let mut total = 1usize;
        for i in 0..self.ndim as usize {
            if self.shape[i] == 0 {
                break;
            }
            total = total.saturating_mul(self.shape[i]);
        }
        total
    }

    /// Returns the total size in bytes of the tensor data.
    pub fn size_bytes(&self) -> usize {
        let bits = self.numel() * self.dtype.bits();
        (bits + 7) / 8
    }

    /// Returns true if the tensor is stored contiguously in memory.
    pub fn is_contiguous(&self) -> bool {
        if self.ndim == 0 {
            return true;
        }
        let ndim = self.ndim as usize;
        let mut expected_stride = self.dtype.bytes();
        for i in (0..ndim).rev() {
            if self.strides[i] != expected_stride {
                return false;
            }
            expected_stride *= self.shape[i];
        }
        true
    }
}

impl fmt::Display for TensorDescriptor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tensor(")?;
        for i in 0..self.ndim as usize {
            if i > 0 {
                write!(f, "×")?;
            }
            write!(f, "{}", self.shape[i])?;
        }
        write!(f, ", {}, {})", self.dtype, self.device)
    }
}

// ---------------------------------------------------------------------------
// Model Configuration
// ---------------------------------------------------------------------------

/// Quantization format for model weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantFormat {
    /// No quantization (full precision)
    None,
    /// GGUF Q4_K_M — 4-bit with k-quant mixed precision
    GgufQ4KM,
    /// GGUF Q4_K_S — 4-bit with k-quant small
    GgufQ4KS,
    /// GGUF Q6_K — 6-bit with k-quant
    GgufQ6K,
    /// GGUF Q8_0 — 8-bit uniform
    GgufQ8_0,
    /// FP8 E4M3 — native on K3 A100 cores
    Fp8E4M3,
    /// AWQ — Activation-aware weight quantization
    Awq,
    /// GPTQ — Post-training quantization
    Gptq,
}

/// Model architecture family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelArch {
    /// Qwen transformer family (0.5B–72B)
    Qwen,
    /// LLaMA transformer family (7B–70B)
    Llama,
    /// DeepSeek R1 distilled (built on Qwen/Llama)
    DeepSeekR1,
    /// Mixtral MoE (8×7B, 8×22B)
    Mixtral,
    /// Generic GGUF-compatible transformer
    GenericTransformer,
}

/// Configuration for loading and running an ML model.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    /// Human-readable model name
    pub name: String,
    /// Model architecture family
    pub arch: ModelArch,
    /// Number of parameters (in billions, approximate)
    pub params_billions: u32,
    /// Weight quantization format
    pub quant_format: QuantFormat,
    /// Maximum context length (tokens)
    pub max_context_len: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Number of key-value heads (for GQA)
    pub num_kv_heads: usize,
    /// Hidden dimension size
    pub hidden_dim: usize,
    /// Number of transformer layers
    pub num_layers: usize,
    /// Vocabulary size
    pub vocab_size: usize,
    /// Target device for inference
    pub device: DeviceType,
    /// Path to model weights (GGUF file)
    pub weights_path: String,
}

impl ModelConfig {
    /// Estimates the RAM required (in bytes) for loading this model.
    ///
    /// Formula: params × bytes_per_param + KV_cache_estimate + overhead
    pub fn estimated_ram_bytes(&self) -> usize {
        let params = (self.params_billions as usize) * 1_000_000_000;
        let weight_bytes = match self.quant_format {
            QuantFormat::None => params * 4,             // FP32
            QuantFormat::Fp8E4M3 => params,              // 1 byte per param
            QuantFormat::GgufQ8_0 => params,             // ~1 byte per param
            QuantFormat::GgufQ6K => params * 6 / 8,      // 0.75 bytes per param
            QuantFormat::GgufQ4KM | QuantFormat::GgufQ4KS
            | QuantFormat::Awq | QuantFormat::Gptq => params / 2, // 0.5 bytes per param
        };

        // KV cache estimate: 2 * num_layers * num_kv_heads * hidden_dim/num_heads * max_ctx * 2bytes
        let head_dim = if self.num_heads > 0 {
            self.hidden_dim / self.num_heads
        } else {
            128
        };
        let kv_cache = 2 * self.num_layers * self.num_kv_heads * head_dim
            * self.max_context_len.min(4096) * 2;

        // ~500MB overhead for runtime, tokenizer, etc.
        let overhead = 500 * 1024 * 1024;

        weight_bytes + kv_cache + overhead
    }

    /// Returns true if this model can fit in the given RAM (in bytes).
    pub fn fits_in_ram(&self, available_ram: usize) -> bool {
        self.estimated_ram_bytes() <= available_ram
    }
}

// ---------------------------------------------------------------------------
// Inference Engine Trait
// ---------------------------------------------------------------------------

/// Errors that can occur during AI runtime operations.
#[derive(Debug, Clone)]
pub enum AiError {
    /// Model file not found or unreadable
    ModelNotFound,
    /// Insufficient memory to load model
    OutOfMemory { required: usize, available: usize },
    /// Unsupported quantization format on this hardware
    UnsupportedQuant(QuantFormat),
    /// Device not available (e.g., A100 cores on K1 hardware)
    DeviceNotAvailable(DeviceType),
    /// Tensor shape mismatch
    ShapeMismatch { expected: [usize; MAX_DIMS], got: [usize; MAX_DIMS] },
    /// Computation error
    ComputeError,
    /// Invalid configuration
    InvalidConfig(String),
    /// Unsupported hardware feature
    UnsupportedHardware,
}

impl fmt::Display for AiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ModelNotFound => write!(f, "Model file not found"),
            Self::OutOfMemory { required, available } => {
                write!(f, "OOM: need {}MB, have {}MB",
                    required / (1024 * 1024),
                    available / (1024 * 1024))
            }
            Self::UnsupportedQuant(q) => write!(f, "Unsupported quant: {:?}", q),
            Self::DeviceNotAvailable(d) => write!(f, "Device unavailable: {}", d),
            Self::ShapeMismatch { .. } => write!(f, "Tensor shape mismatch"),
            Self::ComputeError => write!(f, "Compute error"),
            Self::InvalidConfig(msg) => write!(f, "Invalid config: {}", msg),
            Self::UnsupportedHardware => write!(f, "Unsupported hardware feature"),
        }
    }
}

/// Hardware capability flags detected at runtime.
#[derive(Debug, Clone, Copy)]
pub struct HardwareCaps {
    /// RISC-V Vector extension version (0 = not available)
    pub rvv_version: u8,
    /// Vector register width in bits (128, 256, 512, 1024)
    pub vlen_bits: u16,
    /// Native FP8 support (K3 A100 cores)
    pub has_fp8: bool,
    /// Native BF16 support
    pub has_bf16: bool,
    /// Number of AI accelerator cores (0 on K1, 8 on K3)
    pub ai_core_count: u8,
    /// Total AI compute in TOPS (tera operations per second)
    pub ai_tops: u16,
    /// PowerVR GPU available
    pub has_gpu: bool,
    /// OpenCL version (0 = not available, 30 = 3.0)
    pub opencl_version: u8,
    /// Total system RAM in bytes
    pub total_ram: usize,
    /// Available (free) RAM in bytes
    pub available_ram: usize,
}

impl HardwareCaps {
    /// Returns capabilities for SpacemiT K1 (BPI-F3).
    pub fn spacemit_k1(ram_gb: usize) -> Self {
        Self {
            rvv_version: 1,
            vlen_bits: 256,
            has_fp8: false,
            has_bf16: false,
            ai_core_count: 0,
            ai_tops: 2,
            has_gpu: false,
            opencl_version: 0,
            total_ram: ram_gb * 1024 * 1024 * 1024,
            available_ram: (ram_gb * 1024 * 1024 * 1024) * 3 / 4, // ~75% available
        }
    }

    /// Returns capabilities for SpacemiT K3 (AIBOX-K3).
    pub fn spacemit_k3(ram_gb: usize) -> Self {
        Self {
            rvv_version: 1,
            vlen_bits: 1024,
            has_fp8: true,
            has_bf16: true,
            ai_core_count: 8,
            ai_tops: 60,
            has_gpu: true,
            opencl_version: 30,
            total_ram: ram_gb * 1024 * 1024 * 1024,
            available_ram: (ram_gb * 1024 * 1024 * 1024) * 3 / 4,
        }
    }

    /// Selects the optimal data type for inference given hardware capabilities.
    pub fn optimal_dtype(&self, model_params_b: u32) -> DataType {
        let model_fp8_bytes = (model_params_b as usize) * 1_000_000_000;
        let model_int4_bytes = model_fp8_bytes / 2;

        if self.has_fp8 && model_fp8_bytes < self.available_ram {
            // K3: prefer native FP8 (zero dequant overhead)
            DataType::FP8
        } else if model_int4_bytes < self.available_ram {
            // Falls back to INT4 quantization
            DataType::INT4
        } else {
            // Extreme case: even INT4 doesn't fit
            DataType::INT4
        }
    }
}

// ---------------------------------------------------------------------------
// Speculative Decoding
// ---------------------------------------------------------------------------

/// Configuration for speculative decoding (draft + verify architecture).
///
/// Uses a small "draft" model to generate candidate tokens quickly,
/// then verifies them in batch with the larger "target" model.
/// On K3, the draft runs on CPU X100 cores while verification
/// runs on A100 AI cores in parallel.
#[derive(Debug, Clone)]
pub struct SpeculativeConfig {
    /// Number of draft tokens to generate before verification
    pub draft_tokens: usize,
    /// Draft model configuration (e.g., Qwen 0.5B)
    pub draft_model: ModelConfig,
    /// Target model configuration (e.g., DeepSeek R1 14B)
    pub target_model: ModelConfig,
    /// Acceptance threshold (0.0–1.0)
    pub acceptance_threshold: f32,
}

// ---------------------------------------------------------------------------
// Model Registry — Pre-configured model profiles
// ---------------------------------------------------------------------------

/// Pre-configured model profiles for common open-weight LLMs.
pub struct ModelRegistry;

impl ModelRegistry {
    /// Qwen 2.5 0.5B — ultra-lightweight, ideal for draft model / classification
    pub fn qwen_0_5b(device: DeviceType) -> ModelConfig {
        ModelConfig {
            name: String::from("Qwen 2.5 0.5B"),
            arch: ModelArch::Qwen,
            params_billions: 1, // rounded up for estimation
            quant_format: QuantFormat::GgufQ4KM,
            max_context_len: 4096,
            num_heads: 14,
            num_kv_heads: 2,
            hidden_dim: 896,
            num_layers: 24,
            vocab_size: 151936,
            device,
            weights_path: String::new(),
        }
    }

    /// DeepSeek R1 1.5B — reasoning distill, good on BPI-F3 (8GB)
    pub fn deepseek_r1_1_5b(device: DeviceType) -> ModelConfig {
        ModelConfig {
            name: String::from("DeepSeek R1 1.5B"),
            arch: ModelArch::DeepSeekR1,
            params_billions: 2, // rounded up
            quant_format: QuantFormat::GgufQ4KM,
            max_context_len: 8192,
            num_heads: 12,
            num_kv_heads: 2,
            hidden_dim: 1536,
            num_layers: 28,
            vocab_size: 151936,
            device,
            weights_path: String::new(),
        }
    }

    /// DeepSeek R1 7B — advanced reasoning, needs AIBOX-K3
    pub fn deepseek_r1_7b(device: DeviceType) -> ModelConfig {
        ModelConfig {
            name: String::from("DeepSeek R1 7B"),
            arch: ModelArch::DeepSeekR1,
            params_billions: 7,
            quant_format: QuantFormat::Fp8E4M3,
            max_context_len: 8192,
            num_heads: 32,
            num_kv_heads: 8,
            hidden_dim: 4096,
            num_layers: 32,
            vocab_size: 151936,
            device,
            weights_path: String::new(),
        }
    }

    /// Qwen 2.5 14B — high-quality generation, AIBOX-K3 32GB
    pub fn qwen_14b(device: DeviceType) -> ModelConfig {
        ModelConfig {
            name: String::from("Qwen 2.5 14B"),
            arch: ModelArch::Qwen,
            params_billions: 14,
            quant_format: QuantFormat::GgufQ4KM,
            max_context_len: 8192,
            num_heads: 40,
            num_kv_heads: 8,
            hidden_dim: 5120,
            num_layers: 40,
            vocab_size: 151936,
            device,
            weights_path: String::new(),
        }
    }

    /// Returns all pre-configured model profiles.
    pub fn all_models() -> Vec<ModelConfig> {
        let dev = DeviceType::Cpu;
        alloc::vec![
            Self::qwen_0_5b(dev),
            Self::deepseek_r1_1_5b(dev),
            Self::deepseek_r1_7b(dev),
            Self::qwen_14b(dev),
        ]
    }
}

// ---------------------------------------------------------------------------
// SymBrain v3 Dual-Hemisphere Configuration & Quantization Mapping
// ---------------------------------------------------------------------------

/// Represents one of the dual hemispheres or components of SymBrain v3.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SymBrainHemisphere {
    /// Left Hemisphere (Qwen-7B-Reasoning): dense logical inference
    LeftHemisphere,
    /// Right Hemisphere (Ministral-8B-Creative): creative formulation & MCTS rollouts
    RightHemisphere,
    /// PFC Controller (WARS-CI-DFA Bridge): coordinating attention and routing
    PfcController,
}

/// Dynamic config describing the quantization mapping for one SymBrain hemisphere.
#[derive(Debug, Clone)]
pub struct SymBrainHemisphereConfig {
    /// Name of the hemisphere module
    pub name: String,
    /// Component type
    pub component: SymBrainHemisphere,
    /// Weight quantization format used
    pub weight_quant: QuantFormat,
    /// KV-cache compression or format used
    pub kv_cache_dtype: DataType,
    /// Size of the parameter set in billions
    pub params_billions: f32,
    /// Hidden dimension
    pub hidden_dim: usize,
    /// Number of attention layers
    pub num_layers: usize,
    /// Target execution device
    pub device: DeviceType,
}

impl SymBrainHemisphereConfig {
    /// Calculates the precise estimated memory footprint for this hemisphere.
    pub fn estimated_vram_bytes(&self, max_context_len: usize) -> usize {
        let params = (self.params_billions * 1_000_000_000.0) as usize;
        let weight_bytes = match self.weight_quant {
            QuantFormat::None => params * 4,
            QuantFormat::Fp8E4M3 | QuantFormat::GgufQ8_0 => params,
            QuantFormat::GgufQ6K => params * 6 / 8,
            QuantFormat::GgufQ4KM | QuantFormat::GgufQ4KS | QuantFormat::Awq | QuantFormat::Gptq => params * 9 / 16, // ~4.5 bits
        };

        // KV cache footprint
        let kv_bytes_per_element = match self.kv_cache_dtype {
            DataType::FP32 => 4,
            DataType::FP16 | DataType::BF16 => 2,
            DataType::FP8 | DataType::INT8 => 1,
            DataType::INT4 => 1, // PolarQuant 3-bit effective
            _ => 2,
        };

        // Standard sequence cache size: 2 * layers * kv_heads * head_dim * seq_len
        // Let's assume standard GQA with 8 KV heads and head_dim=128
        let head_dim = 128;
        let num_kv_heads = 8;
        let kv_cache_bytes = 2 * self.num_layers * num_kv_heads * head_dim * max_context_len * kv_bytes_per_element;

        weight_bytes + kv_cache_bytes
    }
}

/// Unified quantization mapping config for the complete SymBrain v3 system.
#[derive(Debug, Clone)]
pub struct SymBrainQuantConfig {
    /// Left hemisphere config
    pub left: SymBrainHemisphereConfig,
    /// Right hemisphere config
    pub right: SymBrainHemisphereConfig,
    /// PFC coordinator config
    pub pfc: SymBrainHemisphereConfig,
    /// Profile identifier
    pub profile_name: String,
}

impl SymBrainQuantConfig {
    /// Create the official SymBrain v3 Swarm Bourbaki preset.
    ///
    /// Configures precision targets adaptively based on hardware capabilities:
    /// - **Cloud (K3)**: Left (FP8), Right (Q8_0 + PolarQuant 3-bit), PFC (FP16)
    /// - **Edge (K1)**: Left (Q4_K_M), Right (Q8_0), PFC (FP16)
    pub fn v3_bourbaki(caps: &HardwareCaps) -> Self {
        let is_k3 = caps.has_fp8 && caps.vlen_bits >= 1024;
        let (left_quant, left_kv, left_device) = if is_k3 {
            (QuantFormat::Fp8E4M3, DataType::FP8, DeviceType::A100AiCore)
        } else {
            (QuantFormat::GgufQ4KM, DataType::FP16, DeviceType::Cpu)
        };

        let (right_quant, right_kv, right_device) = if is_k3 {
            (QuantFormat::GgufQ8_0, DataType::INT4, DeviceType::A100AiCore) // INT4 represent PolarQuant 3-bit
        } else {
            (QuantFormat::GgufQ8_0, DataType::FP16, DeviceType::Cpu)
        };

        let profile_name = if is_k3 {
            String::from("cloud_spacemit_k3")
        } else {
            String::from("edge_spacemit_k1")
        };

        Self {
            left: SymBrainHemisphereConfig {
                name: String::from("Qwen-7B-Reasoning"),
                component: SymBrainHemisphere::LeftHemisphere,
                weight_quant: left_quant,
                kv_cache_dtype: left_kv,
                params_billions: 7.0,
                hidden_dim: 4096,
                num_layers: 32,
                device: left_device,
            },
            right: SymBrainHemisphereConfig {
                name: String::from("Ministral-8B-Creative"),
                component: SymBrainHemisphere::RightHemisphere,
                weight_quant: right_quant,
                kv_cache_dtype: right_kv,
                params_billions: 8.0,
                hidden_dim: 4096,
                num_layers: 32,
                device: right_device,
            },
            pfc: SymBrainHemisphereConfig {
                name: String::from("WARS-CI-DFA-Bridge"),
                component: SymBrainHemisphere::PfcController,
                weight_quant: QuantFormat::None, // FP16 (represented as None in QuantFormat)
                kv_cache_dtype: DataType::FP16,
                params_billions: 0.5,
                hidden_dim: 1024,
                num_layers: 12,
                device: DeviceType::Cpu,
            },
            profile_name,
        }
    }

    /// Create the official SymBrain v3 macos_m2_unified preset for local edge inference.
    pub fn macos_m2_unified(_caps: &HardwareCaps) -> Self {
        Self {
            left: SymBrainHemisphereConfig {
                name: String::from("Qwen-7B-Reasoning"),
                component: SymBrainHemisphere::LeftHemisphere,
                weight_quant: QuantFormat::GgufQ4KM, // MPS-accelerated GGUF Q4
                kv_cache_dtype: DataType::FP16,     // standard F16 GPU cache
                params_billions: 7.0,
                hidden_dim: 4096,
                num_layers: 32,
                device: DeviceType::PowerVrGpu,      // represents Metal GPU core in HAL
            },
            right: SymBrainHemisphereConfig {
                name: String::from("Ministral-8B-Creative"),
                component: SymBrainHemisphere::RightHemisphere,
                weight_quant: QuantFormat::GgufQ8_0, // uniform Q8 weights
                kv_cache_dtype: DataType::INT4,      // PolarQuant 3-bit cache
                params_billions: 8.0,
                hidden_dim: 4096,
                num_layers: 32,
                device: DeviceType::PowerVrGpu,      // running on GPU
            },
            pfc: SymBrainHemisphereConfig {
                name: String::from("WARS-CI-DFA-Bridge"),
                component: SymBrainHemisphere::PfcController,
                weight_quant: QuantFormat::None,     // FP16 controller
                kv_cache_dtype: DataType::FP16,
                params_billions: 0.5,
                hidden_dim: 1024,
                num_layers: 12,
                device: DeviceType::Cpu,             // CPU AMX/NEON coordinator
            },
            profile_name: String::from("macos_m2_unified"),
        }
    }

    /// Computes the total VRAM/RAM required to execute this configuration.
    pub fn total_vram_bytes(&self, max_context_len: usize) -> usize {
        self.left.estimated_vram_bytes(max_context_len)
            + self.right.estimated_vram_bytes(max_context_len)
            + self.pfc.estimated_vram_bytes(max_context_len)
    }

    /// Checks if this configuration fits within the system's available RAM.
    pub fn fits_in_ram(&self, max_context_len: usize, available_ram: usize) -> bool {
        self.total_vram_bytes(max_context_len) <= available_ram
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtype_sizes() {
        assert_eq!(DataType::FP32.bits(), 32);
        assert_eq!(DataType::FP16.bits(), 16);
        assert_eq!(DataType::FP8.bits(), 8);
        assert_eq!(DataType::INT4.bits(), 4);
        assert_eq!(DataType::Binary.bits(), 1);
        assert_eq!(DataType::INT4.bytes(), 1); // rounded up
    }

    #[test]
    fn test_tensor_descriptor() {
        let t = TensorDescriptor::new(&[2, 3, 4], DataType::FP32, DeviceType::Cpu);
        assert_eq!(t.numel(), 24);
        assert_eq!(t.size_bytes(), 96); // 24 × 4 bytes
        assert!(t.is_contiguous());
        assert_eq!(t.ndim, 3);
    }

    #[test]
    fn test_model_ram_estimation() {
        let model = ModelRegistry::deepseek_r1_7b(DeviceType::A100AiCore);
        let ram = model.estimated_ram_bytes();
        // 7B FP8 ≈ 7GB + KV cache + overhead
        assert!(ram > 7 * 1024 * 1024 * 1024);
        assert!(ram < 12 * 1024 * 1024 * 1024);
    }

    #[test]
    fn test_hardware_caps_k1() {
        let k1 = HardwareCaps::spacemit_k1(8);
        assert_eq!(k1.vlen_bits, 256);
        assert!(!k1.has_fp8);
        assert_eq!(k1.ai_core_count, 0);
        assert_eq!(k1.optimal_dtype(1), DataType::INT4);
    }

    #[test]
    fn test_hardware_caps_k3() {
        let k3 = HardwareCaps::spacemit_k3(32);
        assert_eq!(k3.vlen_bits, 1024);
        assert!(k3.has_fp8);
        assert_eq!(k3.ai_core_count, 8);
        assert_eq!(k3.optimal_dtype(7), DataType::FP8);
    }

    #[test]
    fn test_model_fits_k1() {
        let k1 = HardwareCaps::spacemit_k1(8);
        let small = ModelRegistry::qwen_0_5b(DeviceType::Cpu);
        assert!(small.fits_in_ram(k1.available_ram));
    }

    #[test]
    fn test_symbrain_v3_config_edge() {
        let k1 = HardwareCaps::spacemit_k1(8);
        let config = SymBrainQuantConfig::v3_bourbaki(&k1);
        assert_eq!(config.profile_name, "edge_spacemit_k1");
        assert_eq!(config.left.weight_quant, QuantFormat::GgufQ4KM);
        assert_eq!(config.right.weight_quant, QuantFormat::GgufQ8_0);
        
        let total_ram_needed = config.total_vram_bytes(4096);
        // Total footprint should fit inside 8GB easily since 7B is Q4 and 8B is Q8
        assert!(total_ram_needed < 15 * 1024 * 1024 * 1024);
    }

    #[test]
    fn test_symbrain_v3_config_cloud() {
        let k3 = HardwareCaps::spacemit_k3(32);
        let config = SymBrainQuantConfig::v3_bourbaki(&k3);
        assert_eq!(config.profile_name, "cloud_spacemit_k3");
        assert_eq!(config.left.weight_quant, QuantFormat::Fp8E4M3);
        assert_eq!(config.right.weight_quant, QuantFormat::GgufQ8_0);
        assert_eq!(config.right.kv_cache_dtype, DataType::INT4); // PolarQuant 3-bit
    }

    #[test]
    fn test_symbrain_v3_config_m2() {
        let k1 = HardwareCaps::spacemit_k1(16); // mock
        let config = SymBrainQuantConfig::macos_m2_unified(&k1);
        assert_eq!(config.profile_name, "macos_m2_unified");
        assert_eq!(config.left.weight_quant, QuantFormat::GgufQ4KM);
        assert_eq!(config.right.weight_quant, QuantFormat::GgufQ8_0);
        assert_eq!(config.right.kv_cache_dtype, DataType::INT4); // PolarQuant 3-bit
        assert_eq!(config.left.device, DeviceType::PowerVrGpu); // represents Metal GPU
    }
}
