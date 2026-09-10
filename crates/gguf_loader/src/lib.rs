// Copyright (c) 2026 Xavier Callens / Socrate AI. All Rights Reserved.
// SPDX-License-Identifier: LicenseRef-RunuX-Commercial
// This file is proprietary and confidential. Unauthorized copying,
// distribution, or use is strictly prohibited.

#![cfg_attr(not(test), no_std)]
#![deny(clippy::all)]
#![warn(clippy::pedantic)]
#![allow(clippy::manual_div_ceil)]
//! RunuX GGUF Loader — Pure Rust parser for GGUF model files
//!
//! Parses GGUF v3 model files (llama.cpp format) to extract:
//! - Model metadata (architecture, tokenizer config, hyperparameters)
//! - Quantized tensor descriptors (Q4_K_M, Q6_K, Q8_0, FP8, FP16, FP32)
//! - Tensor data offsets for zero-copy memory-mapped loading
//!
//! # GGUF File Format (v3)
//!
//! ```text
//! ┌──────────────────────────────────────┐
//! │  Header (magic, version, counts)     │  24 bytes
//! ├──────────────────────────────────────┤
//! │  Metadata KV pairs                   │  variable
//! │  (architecture, context_length, etc) │
//! ├──────────────────────────────────────┤
//! │  Tensor Info array                   │  variable
//! │  (name, shape, dtype, offset)        │
//! ├──────────────────────────────────────┤
//! │  Alignment padding                   │  0-31 bytes
//! ├──────────────────────────────────────┤
//! │  Tensor Data (bulk weights)          │  majority of file
//! └──────────────────────────────────────┘
//! ```

extern crate alloc;
use alloc::string::String;
use alloc::vec::Vec;

use ai_runtime::DataType;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// GGUF magic number: "GGUF" in little-endian
pub const GGUF_MAGIC: u32 = 0x4655_4747; // "GGUF" LE

/// Supported GGUF version
pub const GGUF_VERSION_3: u32 = 3;

/// Default alignment for tensor data
pub const GGUF_DEFAULT_ALIGNMENT: usize = 32;

// ---------------------------------------------------------------------------
// GGUF Data Types (metadata value types)
// ---------------------------------------------------------------------------

/// GGUF metadata value types.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgufValueType {
    Uint8 = 0,
    Int8 = 1,
    Uint16 = 2,
    Int16 = 3,
    Uint32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    Uint64 = 10,
    Int64 = 11,
    Float64 = 12,
}

impl GgufValueType {
    /// Parse from raw u32 value.
    pub fn from_u32(v: u32) -> Option<Self> {
        match v {
            0 => Some(Self::Uint8),
            1 => Some(Self::Int8),
            2 => Some(Self::Uint16),
            3 => Some(Self::Int16),
            4 => Some(Self::Uint32),
            5 => Some(Self::Int32),
            6 => Some(Self::Float32),
            7 => Some(Self::Bool),
            8 => Some(Self::String),
            9 => Some(Self::Array),
            10 => Some(Self::Uint64),
            11 => Some(Self::Int64),
            12 => Some(Self::Float64),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// GGUF Tensor Types (quantization formats)
// ---------------------------------------------------------------------------

/// GGUF tensor quantization types.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgufTensorType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K_S = 11,
    Q3_K_M = 12,
    Q3_K_L = 13,
    Q4_K_S = 14,
    Q4_K_M = 15,
    Q5_K_S = 16,
    Q5_K_M = 17,
    Q6_K = 18,
    IQ2_XXS = 19,
    IQ2_XS = 20,
    IQ3_XXS = 21,
    IQ1_S = 22,
    IQ4_NL = 23,
    IQ3_S = 24,
    IQ2_S = 25,
    IQ4_XS = 26,
    I8 = 27,
    I16 = 28,
    I32 = 29,
    I64 = 30,
    F64 = 31,
    BF16 = 32,
}

impl GgufTensorType {
    /// Parse from raw u32 value.
    pub fn from_u32(v: u32) -> Option<Self> {
        match v {
            0 => Some(Self::F32),
            1 => Some(Self::F16),
            2 => Some(Self::Q4_0),
            3 => Some(Self::Q4_1),
            6 => Some(Self::Q5_0),
            7 => Some(Self::Q5_1),
            8 => Some(Self::Q8_0),
            9 => Some(Self::Q8_1),
            10 => Some(Self::Q2_K),
            11 => Some(Self::Q3_K_S),
            12 => Some(Self::Q3_K_M),
            13 => Some(Self::Q3_K_L),
            14 => Some(Self::Q4_K_S),
            15 => Some(Self::Q4_K_M),
            16 => Some(Self::Q5_K_S),
            17 => Some(Self::Q5_K_M),
            18 => Some(Self::Q6_K),
            19 => Some(Self::IQ2_XXS),
            20 => Some(Self::IQ2_XS),
            21 => Some(Self::IQ3_XXS),
            22 => Some(Self::IQ1_S),
            23 => Some(Self::IQ4_NL),
            24 => Some(Self::IQ3_S),
            25 => Some(Self::IQ2_S),
            26 => Some(Self::IQ4_XS),
            27 => Some(Self::I8),
            28 => Some(Self::I16),
            29 => Some(Self::I32),
            30 => Some(Self::I64),
            31 => Some(Self::F64),
            32 => Some(Self::BF16),
            _ => None,
        }
    }

    /// Returns the block size for this quantization type.
    /// Most k-quant types use blocks of 256 elements.
    pub fn block_size(self) -> usize {
        match self {
            Self::F32 | Self::F16 | Self::BF16 | Self::F64 => 1,
            Self::I8 | Self::I16 | Self::I32 | Self::I64 => 1,
            Self::Q4_0 | Self::Q4_1 => 32,
            Self::Q5_0 | Self::Q5_1 => 32,
            Self::Q8_0 | Self::Q8_1 => 32,
            _ => 256, // K-quant types
        }
    }

    /// Returns bytes per block for this quantization type.
    pub fn block_bytes(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::F64 => 8,
            Self::I8 => 1,
            Self::I16 => 2,
            Self::I32 => 4,
            Self::I64 => 8,
            Self::Q4_0 => 18, // 32 × 4-bit + 2-byte scale = 18
            Self::Q4_1 => 20, // 32 × 4-bit + 2-byte scale + 2-byte min = 20
            Self::Q5_0 => 22, // 32 × 5-bit + 2-byte scale = 22
            Self::Q5_1 => 24, // 32 × 5-bit + 2-byte scale + 2-byte min = 24
            Self::Q8_0 => 34, // 32 × 8-bit + 2-byte scale = 34
            Self::Q8_1 => 40, // 32 × 8-bit + 2-byte scale + 2-byte sum = 40
            Self::Q2_K => 84, // 256 elements
            Self::Q3_K_S => 110,
            Self::Q3_K_M => 110,
            Self::Q3_K_L => 110,
            Self::Q4_K_S => 144,
            Self::Q4_K_M => 144, // 256 elements per block
            Self::Q5_K_S => 176,
            Self::Q5_K_M => 176,
            Self::Q6_K => 210,
            _ => 1, // IQ types — simplified
        }
    }

    /// Convert to RunuX DataType (approximate mapping).
    pub fn to_runtime_dtype(self) -> DataType {
        match self {
            Self::F32 => DataType::FP32,
            Self::F16 => DataType::FP16,
            Self::BF16 => DataType::BF16,
            Self::Q8_0 | Self::Q8_1 | Self::I8 => DataType::INT8,
            Self::Q4_0 | Self::Q4_1 | Self::Q4_K_S | Self::Q4_K_M => DataType::INT4,
            _ => DataType::INT8, // Fallback
        }
    }
}

// ---------------------------------------------------------------------------
// GGUF Metadata Values
// ---------------------------------------------------------------------------

/// A parsed GGUF metadata value.
#[derive(Debug, Clone)]
pub enum GgufValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Array(Vec<GgufValue>),
    Uint64(u64),
    Int64(i64),
    Float64(f64),
}

impl GgufValue {
    /// Try to get as u32.
    pub fn as_u32(&self) -> Option<u32> {
        match self {
            Self::Uint8(v) => Some(*v as u32),
            Self::Uint16(v) => Some(*v as u32),
            Self::Uint32(v) => Some(*v),
            Self::Int32(v) => Some(*v as u32),
            _ => None,
        }
    }

    /// Try to get as u64.
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Self::Uint8(v) => Some(*v as u64),
            Self::Uint16(v) => Some(*v as u64),
            Self::Uint32(v) => Some(*v as u64),
            Self::Uint64(v) => Some(*v),
            _ => None,
        }
    }

    /// Try to get as f32.
    pub fn as_f32(&self) -> Option<f32> {
        match self {
            Self::Float32(v) => Some(*v),
            Self::Float64(v) => Some(*v as f32),
            _ => None,
        }
    }

    /// Try to get as string reference.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    /// Try to get as bool.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(v) => Some(*v),
            _ => None,
        }
    }
}

/// A metadata key-value pair.
#[derive(Debug, Clone)]
pub struct GgufMetadataKv {
    pub key: String,
    pub value: GgufValue,
}

// ---------------------------------------------------------------------------
// Tensor Info
// ---------------------------------------------------------------------------

/// Information about a single tensor in the GGUF file.
#[derive(Debug, Clone)]
pub struct GgufTensorInfo {
    /// Tensor name (e.g., "blk.0.attn_q.weight")
    pub name: String,
    /// Number of dimensions
    pub n_dims: u32,
    /// Shape (in GGUF order — may be reversed from PyTorch)
    pub shape: Vec<u64>,
    /// Quantization type
    pub dtype: GgufTensorType,
    /// Byte offset from start of tensor data section
    pub offset: u64,
}

impl GgufTensorInfo {
    /// Calculate total number of elements.
    pub fn n_elements(&self) -> u64 {
        self.shape.iter().product::<u64>().max(1)
    }

    /// Calculate the size in bytes for this tensor.
    pub fn size_bytes(&self) -> u64 {
        let n_elements = self.n_elements();
        let block_size = self.dtype.block_size() as u64;
        let block_bytes = self.dtype.block_bytes() as u64;
        let n_blocks = (n_elements + block_size - 1) / block_size;
        n_blocks * block_bytes
    }
}

// ---------------------------------------------------------------------------
// GGUF File Header
// ---------------------------------------------------------------------------

/// Parsed GGUF file header and contents.
#[derive(Debug, Clone)]
pub struct GgufFile {
    /// GGUF version (should be 3)
    pub version: u32,
    /// Number of tensors in the file
    pub n_tensors: u64,
    /// Number of metadata KV pairs
    pub n_metadata_kv: u64,
    /// Parsed metadata
    pub metadata: Vec<GgufMetadataKv>,
    /// Parsed tensor infos
    pub tensors: Vec<GgufTensorInfo>,
    /// Byte offset where tensor data begins
    pub data_offset: u64,
    /// Alignment (from metadata, default 32)
    pub alignment: usize,
}

impl GgufFile {
    /// Get a metadata value by key.
    pub fn get_metadata(&self, key: &str) -> Option<&GgufValue> {
        self.metadata
            .iter()
            .find(|kv| kv.key == key)
            .map(|kv| &kv.value)
    }

    /// Get architecture name (e.g., "llama", "qwen2")
    pub fn architecture(&self) -> Option<&str> {
        self.get_metadata("general.architecture")?.as_str()
    }

    /// Get model name
    pub fn model_name(&self) -> Option<&str> {
        self.get_metadata("general.name")?.as_str()
    }

    /// Get context length
    pub fn context_length(&self) -> Option<u32> {
        let arch = self.architecture()?;
        let key = alloc::format!("{}.context_length", arch);
        self.get_metadata(&key)?.as_u32()
    }

    /// Get embedding length (hidden dim)
    pub fn embedding_length(&self) -> Option<u32> {
        let arch = self.architecture()?;
        let key = alloc::format!("{}.embedding_length", arch);
        self.get_metadata(&key)?.as_u32()
    }

    /// Get number of attention heads
    pub fn head_count(&self) -> Option<u32> {
        let arch = self.architecture()?;
        let key = alloc::format!("{}.attention.head_count", arch);
        self.get_metadata(&key)?.as_u32()
    }

    /// Get number of KV heads (for GQA)
    pub fn head_count_kv(&self) -> Option<u32> {
        let arch = self.architecture()?;
        let key = alloc::format!("{}.attention.head_count_kv", arch);
        self.get_metadata(&key)?.as_u32()
    }

    /// Get number of layers
    pub fn block_count(&self) -> Option<u32> {
        let arch = self.architecture()?;
        let key = alloc::format!("{}.block_count", arch);
        self.get_metadata(&key)?.as_u32()
    }

    /// Get vocabulary size
    pub fn vocab_size(&self) -> Option<u32> {
        self.get_metadata("tokenizer.ggml.tokens")
            .and_then(|v| match v {
                GgufValue::Array(arr) => Some(arr.len() as u32),
                _ => None,
            })
    }

    /// Get a tensor by name.
    pub fn get_tensor(&self, name: &str) -> Option<&GgufTensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// Total size of all tensor data in bytes.
    pub fn total_tensor_bytes(&self) -> u64 {
        self.tensors.iter().map(|t| t.size_bytes()).sum()
    }
}

// ---------------------------------------------------------------------------
// Binary Reader
// ---------------------------------------------------------------------------

/// A simple cursor-based binary reader for `no_std`.
struct BinaryReader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> BinaryReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.pos)
    }

    fn read_u8(&mut self) -> Result<u8, GgufError> {
        if self.pos >= self.data.len() {
            return Err(GgufError::UnexpectedEof);
        }
        let v = self.data[self.pos];
        self.pos += 1;
        Ok(v)
    }

    fn read_u16_le(&mut self) -> Result<u16, GgufError> {
        if self.pos + 2 > self.data.len() {
            return Err(GgufError::UnexpectedEof);
        }
        let v = u16::from_le_bytes([self.data[self.pos], self.data[self.pos + 1]]);
        self.pos += 2;
        Ok(v)
    }

    fn read_u32_le(&mut self) -> Result<u32, GgufError> {
        if self.pos + 4 > self.data.len() {
            return Err(GgufError::UnexpectedEof);
        }
        let bytes = [
            self.data[self.pos],
            self.data[self.pos + 1],
            self.data[self.pos + 2],
            self.data[self.pos + 3],
        ];
        self.pos += 4;
        Ok(u32::from_le_bytes(bytes))
    }

    fn read_i32_le(&mut self) -> Result<i32, GgufError> {
        Ok(self.read_u32_le()? as i32)
    }

    fn read_u64_le(&mut self) -> Result<u64, GgufError> {
        if self.pos + 8 > self.data.len() {
            return Err(GgufError::UnexpectedEof);
        }
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(&self.data[self.pos..self.pos + 8]);
        self.pos += 8;
        Ok(u64::from_le_bytes(bytes))
    }

    fn read_i64_le(&mut self) -> Result<i64, GgufError> {
        Ok(self.read_u64_le()? as i64)
    }

    fn read_f32_le(&mut self) -> Result<f32, GgufError> {
        let bits = self.read_u32_le()?;
        Ok(f32::from_bits(bits))
    }

    fn read_f64_le(&mut self) -> Result<f64, GgufError> {
        let bits = self.read_u64_le()?;
        Ok(f64::from_bits(bits))
    }

    fn read_string(&mut self) -> Result<String, GgufError> {
        let len = self.read_u64_le()? as usize;
        if self.pos + len > self.data.len() {
            return Err(GgufError::UnexpectedEof);
        }
        let bytes = &self.data[self.pos..self.pos + len];
        self.pos += len;
        String::from_utf8(bytes.to_vec()).map_err(|_| GgufError::InvalidUtf8)
    }

    fn read_bool(&mut self) -> Result<bool, GgufError> {
        Ok(self.read_u8()? != 0)
    }

    fn skip_value(&mut self, vtype: GgufValueType) -> Result<(), GgufError> {
        match vtype {
            GgufValueType::Uint8 | GgufValueType::Int8 | GgufValueType::Bool => {
                if self.pos + 1 > self.data.len() {
                    return Err(GgufError::UnexpectedEof);
                }
                self.pos += 1;
                Ok(())
            }
            GgufValueType::Uint16 | GgufValueType::Int16 => {
                if self.pos + 2 > self.data.len() {
                    return Err(GgufError::UnexpectedEof);
                }
                self.pos += 2;
                Ok(())
            }
            GgufValueType::Uint32 | GgufValueType::Int32 | GgufValueType::Float32 => {
                if self.pos + 4 > self.data.len() {
                    return Err(GgufError::UnexpectedEof);
                }
                self.pos += 4;
                Ok(())
            }
            GgufValueType::Uint64 | GgufValueType::Int64 | GgufValueType::Float64 => {
                if self.pos + 8 > self.data.len() {
                    return Err(GgufError::UnexpectedEof);
                }
                self.pos += 8;
                Ok(())
            }
            GgufValueType::String => {
                let len = self.read_u64_le()? as usize;
                if self.pos + len > self.data.len() {
                    return Err(GgufError::UnexpectedEof);
                }
                self.pos += len;
                Ok(())
            }
            GgufValueType::Array => {
                let elem_type_raw = self.read_u32_le()?;
                let elem_type = GgufValueType::from_u32(elem_type_raw)
                    .ok_or(GgufError::InvalidValueType(elem_type_raw))?;
                let n = self.read_u64_le()? as usize;
                for _ in 0..n {
                    self.skip_value(elem_type)?;
                }
                Ok(())
            }
        }
    }

    fn read_value(&mut self, vtype: GgufValueType) -> Result<GgufValue, GgufError> {
        match vtype {
            GgufValueType::Uint8 => Ok(GgufValue::Uint8(self.read_u8()?)),
            GgufValueType::Int8 => Ok(GgufValue::Int8(self.read_u8()? as i8)),
            GgufValueType::Uint16 => Ok(GgufValue::Uint16(self.read_u16_le()?)),
            GgufValueType::Int16 => Ok(GgufValue::Int16(self.read_u16_le()? as i16)),
            GgufValueType::Uint32 => Ok(GgufValue::Uint32(self.read_u32_le()?)),
            GgufValueType::Int32 => Ok(GgufValue::Int32(self.read_i32_le()?)),
            GgufValueType::Float32 => Ok(GgufValue::Float32(self.read_f32_le()?)),
            GgufValueType::Bool => Ok(GgufValue::Bool(self.read_bool()?)),
            GgufValueType::String => Ok(GgufValue::String(self.read_string()?)),
            GgufValueType::Uint64 => Ok(GgufValue::Uint64(self.read_u64_le()?)),
            GgufValueType::Int64 => Ok(GgufValue::Int64(self.read_i64_le()?)),
            GgufValueType::Float64 => Ok(GgufValue::Float64(self.read_f64_le()?)),
            GgufValueType::Array => {
                let elem_type_raw = self.read_u32_le()?;
                let elem_type = GgufValueType::from_u32(elem_type_raw)
                    .ok_or(GgufError::InvalidValueType(elem_type_raw))?;
                let n = self.read_u64_le()? as usize;
                // If array exceeds 100,000 items, retain up to 10,000 and cleanly advance the file pointer past remaining elements
                if n > 100_000 {
                    let keep = 10_000;
                    let mut arr = Vec::with_capacity(keep);
                    for _ in 0..keep {
                        arr.push(self.read_value(elem_type)?);
                    }
                    for _ in keep..n {
                        self.skip_value(elem_type)?;
                    }
                    return Ok(GgufValue::Array(arr));
                }
                let mut arr = Vec::with_capacity(n);
                for _ in 0..n {
                    arr.push(self.read_value(elem_type)?);
                }
                Ok(GgufValue::Array(arr))
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors that can occur during GGUF parsing.
#[derive(Debug, Clone)]
pub enum GgufError {
    /// File is too short to contain a valid header
    FileTooShort,
    /// Invalid magic number (not "GGUF")
    InvalidMagic(u32),
    /// Unsupported GGUF version
    UnsupportedVersion(u32),
    /// Unexpected end of file during parsing
    UnexpectedEof,
    /// Invalid UTF-8 in string value
    InvalidUtf8,
    /// Unknown metadata value type
    InvalidValueType(u32),
    /// Unknown tensor quantization type
    InvalidTensorType(u32),
    /// Tensor count mismatch
    TensorCountMismatch { expected: u64, got: u64 },
}

impl core::fmt::Display for GgufError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::FileTooShort => write!(f, "GGUF file too short"),
            Self::InvalidMagic(m) => write!(f, "Invalid GGUF magic: 0x{:08x}", m),
            Self::UnsupportedVersion(v) => write!(f, "Unsupported GGUF version: {}", v),
            Self::UnexpectedEof => write!(f, "Unexpected end of file"),
            Self::InvalidUtf8 => write!(f, "Invalid UTF-8 string"),
            Self::InvalidValueType(t) => write!(f, "Unknown value type: {}", t),
            Self::InvalidTensorType(t) => write!(f, "Unknown tensor type: {}", t),
            Self::TensorCountMismatch { expected, got } => {
                write!(
                    f,
                    "Tensor count mismatch: expected {}, got {}",
                    expected, got
                )
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

/// Parse a GGUF file from a byte slice.
///
/// This is the main entry point. Pass in the full file contents
/// (or a memory-mapped region) and get back a parsed `GgufFile`.
///
/// # Example
///
/// ```ignore
/// let data = std::fs::read("model.gguf").unwrap();
/// let gguf = gguf_loader::parse(&data).unwrap();
/// println!("Model: {:?}", gguf.model_name());
/// println!("Layers: {:?}", gguf.block_count());
/// println!("Tensors: {}", gguf.tensors.len());
/// ```
pub fn parse(data: &[u8]) -> Result<GgufFile, GgufError> {
    if data.len() < 24 {
        return Err(GgufError::FileTooShort);
    }

    let mut reader = BinaryReader::new(data);

    // Header: magic (4) + version (4) + n_tensors (8) + n_metadata_kv (8) = 24 bytes
    let magic = reader.read_u32_le()?;
    if magic != GGUF_MAGIC {
        return Err(GgufError::InvalidMagic(magic));
    }

    let version = reader.read_u32_le()?;
    if version != GGUF_VERSION_3 {
        return Err(GgufError::UnsupportedVersion(version));
    }

    let n_tensors = reader.read_u64_le()?;
    let n_metadata_kv = reader.read_u64_le()?;

    // Parse metadata KV pairs
    let mut metadata = Vec::with_capacity(n_metadata_kv as usize);
    let mut alignment = GGUF_DEFAULT_ALIGNMENT;

    for _ in 0..n_metadata_kv {
        let key = reader.read_string()?;
        let vtype_raw = reader.read_u32_le()?;
        let vtype =
            GgufValueType::from_u32(vtype_raw).ok_or(GgufError::InvalidValueType(vtype_raw))?;
        let value = reader.read_value(vtype)?;

        // Check for alignment override
        if key == "general.alignment" {
            if let Some(a) = value.as_u32() {
                alignment = a as usize;
            }
        }

        metadata.push(GgufMetadataKv { key, value });
    }

    // Parse tensor info array
    let mut tensors = Vec::with_capacity(n_tensors as usize);
    for _ in 0..n_tensors {
        let name = reader.read_string()?;
        let n_dims = reader.read_u32_le()?;
        let mut shape = Vec::with_capacity(n_dims as usize);
        for _ in 0..n_dims {
            shape.push(reader.read_u64_le()?);
        }
        let dtype_raw = reader.read_u32_le()?;
        let dtype =
            GgufTensorType::from_u32(dtype_raw).ok_or(GgufError::InvalidTensorType(dtype_raw))?;
        let offset = reader.read_u64_le()?;

        tensors.push(GgufTensorInfo {
            name,
            n_dims,
            shape,
            dtype,
            offset,
        });
    }

    // Calculate data offset (aligned)
    let header_end = reader.pos as u64;
    let data_offset = (header_end + alignment as u64 - 1) & !(alignment as u64 - 1);

    Ok(GgufFile {
        version,
        n_tensors,
        n_metadata_kv,
        metadata,
        tensors,
        data_offset,
        alignment,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal valid GGUF v3 file in memory.
    fn build_minimal_gguf() -> Vec<u8> {
        let mut buf: Vec<u8> = Vec::new();

        // Magic
        buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        // Version 3
        buf.extend_from_slice(&3u32.to_le_bytes());
        // n_tensors = 1
        buf.extend_from_slice(&1u64.to_le_bytes());
        // n_metadata_kv = 1
        buf.extend_from_slice(&1u64.to_le_bytes());

        // Metadata: "general.architecture" = "llama"
        let key = b"general.architecture";
        buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
        buf.extend_from_slice(key);
        buf.extend_from_slice(&(GgufValueType::String as u32).to_le_bytes());
        let val = b"llama";
        buf.extend_from_slice(&(val.len() as u64).to_le_bytes());
        buf.extend_from_slice(val);

        // Tensor info: "output.weight", shape [4096, 32000], Q4_K_M
        let tname = b"output.weight";
        buf.extend_from_slice(&(tname.len() as u64).to_le_bytes());
        buf.extend_from_slice(tname);
        buf.extend_from_slice(&2u32.to_le_bytes()); // n_dims
        buf.extend_from_slice(&4096u64.to_le_bytes()); // dim 0
        buf.extend_from_slice(&32000u64.to_le_bytes()); // dim 1
        buf.extend_from_slice(&(GgufTensorType::Q4_K_M as u32).to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes()); // offset

        buf
    }

    #[test]
    fn test_parse_minimal_gguf() {
        let data = build_minimal_gguf();
        let gguf = parse(&data).unwrap();

        assert_eq!(gguf.version, 3);
        assert_eq!(gguf.n_tensors, 1);
        assert_eq!(gguf.n_metadata_kv, 1);
        assert_eq!(gguf.architecture(), Some("llama"));
        assert_eq!(gguf.tensors.len(), 1);
        assert_eq!(gguf.tensors[0].name, "output.weight");
        assert_eq!(gguf.tensors[0].dtype, GgufTensorType::Q4_K_M);
        assert_eq!(gguf.tensors[0].n_elements(), 4096 * 32000);
    }

    #[test]
    fn test_invalid_magic() {
        let data = [
            0x00, 0x00, 0x00, 0x00, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ];
        let result = parse(&data);
        assert!(matches!(result, Err(GgufError::InvalidMagic(_))));
    }

    #[test]
    fn test_file_too_short() {
        let data = [0x47, 0x47, 0x55, 0x46]; // "GGUF" but too short
        let result = parse(&data);
        assert!(matches!(result, Err(GgufError::FileTooShort)));
    }

    #[test]
    fn test_tensor_size_calculation() {
        let tensor = GgufTensorInfo {
            name: String::from("test"),
            n_dims: 2,
            shape: alloc::vec![4096, 4096],
            dtype: GgufTensorType::Q4_K_M,
            offset: 0,
        };
        // 4096 * 4096 = 16M elements / 256 block_size = 65536 blocks × 144 bytes
        assert_eq!(tensor.size_bytes(), 65536 * 144);
    }

    #[test]
    fn test_tensor_type_conversions() {
        assert_eq!(GgufTensorType::Q4_K_M.block_size(), 256);
        assert_eq!(GgufTensorType::Q4_K_M.block_bytes(), 144);
        assert_eq!(GgufTensorType::F32.block_size(), 1);
        assert_eq!(GgufTensorType::F32.block_bytes(), 4);
    }

    #[test]
    fn test_metadata_accessors() {
        let gguf = GgufFile {
            version: 3,
            n_tensors: 0,
            n_metadata_kv: 2,
            metadata: alloc::vec![
                GgufMetadataKv {
                    key: String::from("general.architecture"),
                    value: GgufValue::String(String::from("qwen2")),
                },
                GgufMetadataKv {
                    key: String::from("qwen2.context_length"),
                    value: GgufValue::Uint32(32768),
                },
            ],
            tensors: Vec::new(),
            data_offset: 0,
            alignment: 32,
        };

        assert_eq!(gguf.architecture(), Some("qwen2"));
        assert_eq!(gguf.context_length(), Some(32768));
    }

    #[test]
    fn test_large_array_skipping_preserves_offset() {
        let mut buffer = Vec::new();
        // Magic GGUF: 0x46554747
        buffer.extend_from_slice(&0x46554747u32.to_le_bytes());
        // Version 3
        buffer.extend_from_slice(&3u32.to_le_bytes());
        // n_tensors: 0
        buffer.extend_from_slice(&0u64.to_le_bytes());
        // n_metadata_kv: 2
        buffer.extend_from_slice(&2u64.to_le_bytes());

        // KV 1: key="large_array", type=Array (9), elem_type=Uint8 (0), count=100050
        let k1 = "large_array";
        buffer.extend_from_slice(&(k1.len() as u64).to_le_bytes());
        buffer.extend_from_slice(k1.as_bytes());
        buffer.extend_from_slice(&9u32.to_le_bytes()); // Array
        buffer.extend_from_slice(&0u32.to_le_bytes()); // Uint8
        let count: u64 = 100_050;
        buffer.extend_from_slice(&count.to_le_bytes());
        buffer.resize(buffer.len() + count as usize, 42u8);

        // KV 2: key="sentinel", type=String (8), value="intact"
        let k2 = "sentinel";
        buffer.extend_from_slice(&(k2.len() as u64).to_le_bytes());
        buffer.extend_from_slice(k2.as_bytes());
        buffer.extend_from_slice(&8u32.to_le_bytes()); // String
        let v2 = "intact";
        buffer.extend_from_slice(&(v2.len() as u64).to_le_bytes());
        buffer.extend_from_slice(v2.as_bytes());

        // Parse using parse()
        let parsed = parse(&buffer).expect("Must parse successfully");
        assert_eq!(parsed.metadata.len(), 2);
        assert_eq!(parsed.metadata[0].key, "large_array");
        assert_eq!(parsed.metadata[1].key, "sentinel");
        if let GgufValue::String(ref s) = parsed.metadata[1].value {
            assert_eq!(s, "intact");
        } else {
            panic!("Expected String value for sentinel metadata");
        }
    }
}
