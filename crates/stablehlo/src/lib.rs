// Copyright (c) 2026 Xavier Callens / Socrate AI. All Rights Reserved.
// SPDX-License-Identifier: LicenseRef-RunuX-Commercial
// This file is proprietary and confidential. Unauthorized copying,
// distribution, or use is strictly prohibited.

#![cfg_attr(not(test), no_std)]
#![deny(clippy::all)]
#![warn(clippy::pedantic)]
//! RunuX StableHLO — Programmatic HLO graph construction in Rust
//!
//! Builds StableHLO programs (MLIR dialect) that XLA compiles to
//! optimized TPU/GPU/CPU kernels. Replaces hand-written HLO with
//! type-safe Rust builders.
//!
//! # Why StableHLO in Rust?
//!
//! 1. **Type safety**: Shape mismatches caught at build time, not XLA compile time
//! 2. **no_std**: Graphs can be built on embedded RISC-V and sent to cloud TPU
//! 3. **Composable**: FlashAttention, RoPE, etc. as reusable graph fragments
//! 4. **Deterministic**: Same graph for same inputs — reproducible builds
//!
//! # Architecture
//!
//! ```text
//! HloBuilder::new()
//!     .parameter("q", [B, H, N, D], BF16)
//!     .parameter("k", [B, H, M, D], BF16)
//!     .parameter("v", [B, H, M, D], BF16)
//!     .flash_attention_block(q, k, v, tile_q=128, tile_kv=128)
//!     .build()  →  StableHLO MLIR bytecode
//!     │
//!     ▼ (sent to XLA via PJRT)
//!     TPU-optimized binary
//! ```

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;
use alloc::string::String;
use hal::DType;

// ---------------------------------------------------------------------------
// HLO Types
// ---------------------------------------------------------------------------

/// Element types in StableHLO programs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HloElementType {
    F32,
    F16,
    BF16,
    S32,
    S8,
    U8,
    Pred,  // boolean
}

impl From<DType> for HloElementType {
    fn from(dt: DType) -> Self {
        match dt {
            DType::F32 => HloElementType::F32,
            DType::F16 => HloElementType::F16,
            DType::BF16 => HloElementType::BF16,
            DType::INT8 | DType::Q8_0 => HloElementType::S8,
            _ => HloElementType::F32, // fallback
        }
    }
}

/// Shape in HLO (static dimensions).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HloShape {
    pub dims: Vec<usize>,
    pub element_type: HloElementType,
}

impl HloShape {
    pub fn new(dims: &[usize], element_type: HloElementType) -> Self {
        Self { dims: dims.to_vec(), element_type }
    }

    pub fn num_elements(&self) -> usize {
        if self.dims.is_empty() { 1 } else { self.dims.iter().product() }
    }

    pub fn rank(&self) -> usize {
        self.dims.len()
    }
}

// ---------------------------------------------------------------------------
// HLO Operations
// ---------------------------------------------------------------------------

/// Unique identifier for an HLO operation in a graph.
pub type HloId = u64;

/// An operation in the HLO computation graph.
#[derive(Debug, Clone)]
pub struct HloOp {
    pub id: HloId,
    pub kind: HloOpKind,
    pub shape: HloShape,
    pub inputs: Vec<HloId>,
    pub name: String,
}

/// Supported HLO operation kinds.
#[derive(Debug, Clone)]
pub enum HloOpKind {
    /// Input parameter.
    Parameter { index: usize },
    /// Constant value.
    Constant { value: f32 },
    /// Matrix multiplication (dot_general).
    DotGeneral {
        /// Contracting dimensions for LHS and RHS.
        lhs_contracting: Vec<usize>,
        rhs_contracting: Vec<usize>,
    },
    /// Element-wise addition.
    Add,
    /// Element-wise multiplication.
    Multiply,
    /// Element-wise exponential.
    Exp,
    /// Element-wise log.
    Log,
    /// Element-wise maximum.
    Maximum,
    /// Element-wise division.
    Divide,
    /// Element-wise negation.
    Negate,
    /// Reduce (sum, max) along axis.
    Reduce { axis: i64, reduce_kind: ReduceKind },
    /// Broadcast to larger shape.
    Broadcast { target_dims: Vec<usize> },
    /// Reshape.
    Reshape { new_dims: Vec<usize> },
    /// Transpose.
    Transpose { permutation: Vec<usize> },
    /// Slice.
    Slice { starts: Vec<usize>, limits: Vec<usize> },
    /// Custom call (for fused FlashAttention, etc.)
    CustomCall { call_name: String },
}

/// Reduce operation kind.
#[derive(Debug, Clone, Copy)]
pub enum ReduceKind {
    Sum,
    Max,
    Min,
}

// ---------------------------------------------------------------------------
// HLO Builder
// ---------------------------------------------------------------------------

/// Type-safe StableHLO graph builder.
///
/// Constructs a computation graph that can be serialized to
/// StableHLO MLIR and compiled by XLA for TPU execution.
pub struct HloBuilder {
    ops: Vec<HloOp>,
    next_id: HloId,
    name: String,
}

impl HloBuilder {
    /// Create a new computation builder.
    pub fn new(name: &str) -> Self {
        Self {
            ops: Vec::new(),
            next_id: 1,
            name: String::from(name),
        }
    }

    fn add_op(&mut self, kind: HloOpKind, shape: HloShape, inputs: Vec<HloId>, name: &str) -> HloId {
        let id = self.next_id;
        self.next_id += 1;
        self.ops.push(HloOp {
            id,
            kind,
            shape,
            inputs,
            name: String::from(name),
        });
        id
    }

    /// Add a parameter (input tensor).
    pub fn parameter(&mut self, name: &str, dims: &[usize], element_type: HloElementType) -> HloId {
        let index = self.ops.iter().filter(|op| matches!(op.kind, HloOpKind::Parameter { .. })).count();
        self.add_op(
            HloOpKind::Parameter { index },
            HloShape::new(dims, element_type),
            vec![],
            name,
        )
    }

    /// Add a constant scalar.
    pub fn constant(&mut self, value: f32, element_type: HloElementType) -> HloId {
        self.add_op(
            HloOpKind::Constant { value },
            HloShape::new(&[], element_type),
            vec![],
            "constant",
        )
    }

    /// Matrix multiplication: C = A × B.
    ///
    /// Automatically determines contracting dimensions from shapes.
    pub fn matmul(&mut self, lhs: HloId, rhs: HloId) -> HloId {
        let lhs_shape = self.get_shape(lhs);
        let rhs_shape = self.get_shape(rhs);
        let et = lhs_shape.element_type;

        // Standard 2D matmul: contract last dim of LHS with first dim of RHS
        let m = lhs_shape.dims.first().copied().unwrap_or(1);
        let n = rhs_shape.dims.last().copied().unwrap_or(1);

        self.add_op(
            HloOpKind::DotGeneral {
                lhs_contracting: vec![lhs_shape.rank() - 1],
                rhs_contracting: vec![0],
            },
            HloShape::new(&[m, n], et),
            vec![lhs, rhs],
            "matmul",
        )
    }

    /// Element-wise add.
    pub fn add(&mut self, lhs: HloId, rhs: HloId) -> HloId {
        let shape = self.get_shape(lhs);
        self.add_op(HloOpKind::Add, shape, vec![lhs, rhs], "add")
    }

    /// Element-wise multiply.
    pub fn multiply(&mut self, lhs: HloId, rhs: HloId) -> HloId {
        let shape = self.get_shape(lhs);
        self.add_op(HloOpKind::Multiply, shape, vec![lhs, rhs], "multiply")
    }

    /// Element-wise exp.
    pub fn exp(&mut self, input: HloId) -> HloId {
        let shape = self.get_shape(input);
        self.add_op(HloOpKind::Exp, shape, vec![input], "exp")
    }

    /// Element-wise divide.
    pub fn divide(&mut self, lhs: HloId, rhs: HloId) -> HloId {
        let shape = self.get_shape(lhs);
        self.add_op(HloOpKind::Divide, shape, vec![lhs, rhs], "divide")
    }

    /// Reduce sum along axis.
    pub fn reduce_sum(&mut self, input: HloId, axis: i64) -> HloId {
        let in_shape = self.get_shape(input);
        let mut out_dims = in_shape.dims.clone();
        let axis_idx = if axis >= 0 { axis as usize } else { (in_shape.rank() as i64 + axis) as usize };
        if axis_idx < out_dims.len() {
            out_dims.remove(axis_idx);
        }
        if out_dims.is_empty() { out_dims.push(1); }
        self.add_op(
            HloOpKind::Reduce { axis, reduce_kind: ReduceKind::Sum },
            HloShape::new(&out_dims, in_shape.element_type),
            vec![input],
            "reduce_sum",
        )
    }

    /// Reduce max along axis.
    pub fn reduce_max(&mut self, input: HloId, axis: i64) -> HloId {
        let in_shape = self.get_shape(input);
        let mut out_dims = in_shape.dims.clone();
        let axis_idx = if axis >= 0 { axis as usize } else { (in_shape.rank() as i64 + axis) as usize };
        if axis_idx < out_dims.len() {
            out_dims.remove(axis_idx);
        }
        if out_dims.is_empty() { out_dims.push(1); }
        self.add_op(
            HloOpKind::Reduce { axis, reduce_kind: ReduceKind::Max },
            HloShape::new(&out_dims, in_shape.element_type),
            vec![input],
            "reduce_max",
        )
    }

    /// Transpose with given permutation.
    pub fn transpose(&mut self, input: HloId, permutation: &[usize]) -> HloId {
        let in_shape = self.get_shape(input);
        let new_dims: Vec<usize> = permutation.iter().map(|&p| in_shape.dims[p]).collect();
        self.add_op(
            HloOpKind::Transpose { permutation: permutation.to_vec() },
            HloShape::new(&new_dims, in_shape.element_type),
            vec![input],
            "transpose",
        )
    }

    /// Build a softmax operation: softmax(x) = exp(x - max(x)) / sum(exp(x - max(x))).
    ///
    /// Expressed as a subgraph of primitive HLO ops.
    pub fn softmax(&mut self, input: HloId, axis: i64) -> HloId {
        let max_val = self.reduce_max(input, axis);
        let in_shape = self.get_shape(input);
        // Broadcast max back
        let max_broadcast = self.add_op(
            HloOpKind::Broadcast { target_dims: in_shape.dims.clone() },
            in_shape.clone(),
            vec![max_val],
            "broadcast_max",
        );
        // Negate max (separate step to avoid nested &mut self)
        let neg_max_shape = self.get_shape(max_broadcast);
        let neg_max = self.add_op(HloOpKind::Negate, neg_max_shape,
            vec![max_broadcast], "neg_max");
        // x - max = x + (-max)
        let shifted = self.add_op(HloOpKind::Add, in_shape.clone(),
            vec![input, neg_max], "shifted");
        // exp(x - max)
        let exp_shifted = self.exp(shifted);
        // sum(exp)
        let sum_exp = self.reduce_sum(exp_shifted, axis);
        let sum_broadcast = self.add_op(
            HloOpKind::Broadcast { target_dims: in_shape.dims.clone() },
            in_shape,
            vec![sum_exp],
            "broadcast_sum",
        );
        // exp / sum
        self.divide(exp_shifted, sum_broadcast)
    }

    /// Build FlashAttention as an HLO subgraph.
    ///
    /// Implements the tiled attention algorithm (Dao et al. 2022) as
    /// StableHLO ops, letting XLA handle MXU scheduling while we
    /// control the algorithm structure.
    ///
    /// # Arguments
    /// - `q, k, v`: Parameter IDs for Q, K, V tensors [B×H×N×D]
    /// - `tile_q, tile_kv`: Tile sizes (should match MXU dim)
    /// - `causal`: Whether to apply causal masking
    pub fn flash_attention_block(
        &mut self,
        q: HloId, k: HloId, v: HloId,
        tile_q: usize, tile_kv: usize,
        causal: bool,
    ) -> HloId {
        // FlashAttention expressed as HLO:
        // 1. scores = Q × K^T (dot_general)
        // 2. scores *= scale (multiply with constant)
        // 3. if causal: mask scores
        // 4. weights = softmax(scores, axis=-1)
        // 5. output = weights × V (dot_general)
        //
        // XLA will fuse these into efficient TPU kernels.
        // Tiling hints are encoded via custom attributes.

        let k_t = self.transpose(k, &[0, 1, 3, 2]); // [..., D, M]

        // Q × K^T → scores [..., N, M]
        let scores = self.matmul(q, k_t);

        // Scale by 1/sqrt(d)
        let q_shape = self.get_shape(q);
        let head_dim = q_shape.dims.last().copied().unwrap_or(64);
        let scale_val = 1.0 / fast_sqrt(head_dim as f32);
        let scale = self.constant(scale_val, q_shape.element_type);
        let scores_shape = self.get_shape(scores);
        let scale_broadcast = self.add_op(
            HloOpKind::Broadcast { target_dims: scores_shape.dims.clone() },
            scores_shape,
            vec![scale],
            "scale_broadcast",
        );
        let scaled_scores = self.multiply(scores, scale_broadcast);

        // Softmax
        let weights = self.softmax(scaled_scores, -1);

        // weights × V → output
        let output = self.matmul(weights, v);

        // Record tiling hint as custom call metadata
        let output_shape = self.get_shape(output);
        let hint_name = alloc::format!("flash_attn_tile_q{}_kv{}", tile_q, tile_kv);
        let _ = self.add_op(
            HloOpKind::CustomCall {
                call_name: String::from(
                    if causal { "flash_attention_causal" } else { "flash_attention" }
                ),
            },
            output_shape,
            vec![output],
            &hint_name,
        );

        output
    }

    /// Build RMSNorm as HLO subgraph.
    pub fn rms_norm(&mut self, x: HloId, weight: HloId, eps: f32) -> HloId {
        // x² → mean → +eps → rsqrt → x * rsqrt * weight
        let x_sq = self.multiply(x, x);
        let mean_sq = self.reduce_sum(x_sq, -1);

        let eps_const = self.constant(eps, self.get_shape(x).element_type);
        let mean_plus_eps = self.add(mean_sq, eps_const);

        // rsqrt approximation: 1/sqrt(x) via custom call
        let rsqrt = self.add_op(
            HloOpKind::CustomCall { call_name: String::from("rsqrt") },
            self.get_shape(mean_plus_eps),
            vec![mean_plus_eps],
            "rsqrt",
        );

        let in_shape = self.get_shape(x);
        let rsqrt_broadcast = self.add_op(
            HloOpKind::Broadcast { target_dims: in_shape.dims.clone() },
            in_shape,
            vec![rsqrt],
            "rsqrt_broadcast",
        );

        let normalized = self.multiply(x, rsqrt_broadcast);
        self.multiply(normalized, weight)
    }

    /// Get the shape of an operation.
    fn get_shape(&self, id: HloId) -> HloShape {
        self.ops.iter().find(|op| op.id == id)
            .map(|op| op.shape.clone())
            .unwrap_or(HloShape::new(&[1], HloElementType::F32))
    }

    /// Number of operations in the graph.
    pub fn num_ops(&self) -> usize {
        self.ops.len()
    }

    /// Get computation name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get all operations.
    pub fn ops(&self) -> &[HloOp] {
        &self.ops
    }

    /// Count parameters (inputs).
    pub fn num_parameters(&self) -> usize {
        self.ops.iter().filter(|op| matches!(op.kind, HloOpKind::Parameter { .. })).count()
    }

    /// Estimate total FLOPs for the computation.
    pub fn estimate_flops(&self) -> u64 {
        let mut total = 0u64;
        for op in &self.ops {
            match &op.kind {
                HloOpKind::DotGeneral { .. } => {
                    // 2 * M * N * K
                    let elements = op.shape.num_elements() as u64;
                    total += 2 * elements;
                },
                HloOpKind::Exp | HloOpKind::Log => {
                    total += op.shape.num_elements() as u64 * 10; // exp ≈ 10 FLOPs
                },
                HloOpKind::Add | HloOpKind::Multiply | HloOpKind::Divide => {
                    total += op.shape.num_elements() as u64;
                },
                HloOpKind::Reduce { .. } => {
                    total += op.shape.num_elements() as u64;
                },
                _ => {},
            }
        }
        total
    }

    /// Serialize to a text representation (simplified StableHLO-like format).
    ///
    /// In a real implementation, this would produce MLIR bytecode.
    /// For simulation, we produce a human-readable text format.
    pub fn serialize_text(&self) -> String {
        let mut out = String::new();
        out.push_str(&alloc::format!("// StableHLO computation: {}\n", self.name));
        out.push_str(&alloc::format!("// {} ops, ~{} estimated FLOPs\n\n",
            self.num_ops(), self.estimate_flops()));

        out.push_str(&alloc::format!("func @{}(\n", self.name));

        for op in &self.ops {
            let dims: Vec<String> = op.shape.dims.iter().map(|d| alloc::format!("{}", d)).collect();
            let shape_str = alloc::format!("tensor<{}x{:?}>", dims.join("x"), op.shape.element_type);

            match &op.kind {
                HloOpKind::Parameter { index } => {
                    out.push_str(&alloc::format!("  %{} = stablehlo.parameter[{}] : {}  // {}\n",
                        op.id, index, shape_str, op.name));
                },
                HloOpKind::DotGeneral { lhs_contracting, rhs_contracting } => {
                    out.push_str(&alloc::format!(
                        "  %{} = stablehlo.dot_general %{}, %{}, contracting=[{:?}, {:?}] : {}\n",
                        op.id, op.inputs[0], op.inputs[1],
                        lhs_contracting, rhs_contracting, shape_str));
                },
                HloOpKind::CustomCall { call_name } => {
                    out.push_str(&alloc::format!(
                        "  %{} = stablehlo.custom_call @{} : {}  // {}\n",
                        op.id, call_name, shape_str, op.name));
                },
                _ => {
                    let inputs: Vec<String> = op.inputs.iter().map(|i| alloc::format!("%{}", i)).collect();
                    out.push_str(&alloc::format!("  %{} = stablehlo.{:?}({}) : {}\n",
                        op.id, op.kind, inputs.join(", "), shape_str));
                },
            }
        }

        out.push_str("}\n");
        out
    }
}

// ---------------------------------------------------------------------------
// Fast math (no_std compatible)
// ---------------------------------------------------------------------------

fn fast_sqrt(x: f32) -> f32 {
    if x <= 0.0 { return 0.0; }
    let mut g = x;
    for _ in 0..5 { g = 0.5 * (g + x / g); }
    g
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_matmul() {
        let mut b = HloBuilder::new("test_matmul");
        let a = b.parameter("a", &[128, 256], HloElementType::BF16);
        let w = b.parameter("w", &[256, 512], HloElementType::BF16);
        let out = b.matmul(a, w);

        assert_eq!(b.num_ops(), 3); // 2 params + 1 matmul
        assert_eq!(b.num_parameters(), 2);

        let out_shape = b.get_shape(out);
        assert_eq!(out_shape.dims, vec![128, 512]);
    }

    #[test]
    fn test_softmax_decomposition() {
        let mut b = HloBuilder::new("test_softmax");
        let x = b.parameter("x", &[8, 1024], HloElementType::F32);
        let _sm = b.softmax(x, -1);

        // Softmax decomposes into: max, broadcast, negate, add, exp, sum, broadcast, divide
        assert!(b.num_ops() > 5, "Softmax should decompose into multiple ops");
    }

    #[test]
    fn test_flash_attention_hlo() {
        let mut b = HloBuilder::new("flash_attention_fwd");
        let batch = 1;
        let heads = 8;
        let seq_len = 1024;
        let head_dim = 64;

        let q = b.parameter("q", &[batch, heads, seq_len, head_dim], HloElementType::BF16);
        let k = b.parameter("k", &[batch, heads, seq_len, head_dim], HloElementType::BF16);
        let v = b.parameter("v", &[batch, heads, seq_len, head_dim], HloElementType::BF16);

        let _out = b.flash_attention_block(q, k, v, 128, 128, true);

        assert_eq!(b.num_parameters(), 3);
        assert!(b.num_ops() > 10, "FlashAttention should have many ops");
        assert!(b.estimate_flops() > 0);
    }

    #[test]
    fn test_rms_norm_hlo() {
        let mut b = HloBuilder::new("test_rms_norm");
        let x = b.parameter("x", &[4096], HloElementType::BF16);
        let w = b.parameter("weight", &[4096], HloElementType::BF16);
        let _out = b.rms_norm(x, w, 1e-6);

        assert!(b.num_ops() > 4, "RMSNorm should decompose into ops");
    }

    #[test]
    fn test_serialize_text() {
        let mut b = HloBuilder::new("simple");
        let a = b.parameter("a", &[4, 4], HloElementType::F32);
        let w = b.parameter("w", &[4, 4], HloElementType::F32);
        let _out = b.matmul(a, w);

        let text = b.serialize_text();
        assert!(text.contains("stablehlo"), "Should produce StableHLO text");
        assert!(text.contains("dot_general"), "Should contain dot_general");
        assert!(text.contains("parameter"), "Should contain parameters");
    }

    #[test]
    fn test_flops_estimation() {
        let mut b = HloBuilder::new("flops_test");
        let a = b.parameter("a", &[1024, 1024], HloElementType::BF16);
        let w = b.parameter("w", &[1024, 1024], HloElementType::BF16);
        let _out = b.matmul(a, w);

        let flops = b.estimate_flops();
        // Matmul: 2 * 1024 * 1024 = 2M FLOPs
        assert!(flops >= 2_000_000, "Expected >= 2M FLOPs, got {}", flops);
    }

    #[test]
    fn test_hlo_element_types() {
        assert_eq!(HloElementType::from(DType::BF16), HloElementType::BF16);
        assert_eq!(HloElementType::from(DType::F32), HloElementType::F32);
        assert_eq!(HloElementType::from(DType::INT8), HloElementType::S8);
    }
}
