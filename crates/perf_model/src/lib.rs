// Copyright (c) 2026 Xavier Callens / Socrate AI. All Rights Reserved.
// SPDX-License-Identifier: LicenseRef-RunuX-Commercial
// This file is proprietary and confidential. Unauthorized copying,
// distribution, or use is strictly prohibited.

#![cfg_attr(not(test), no_std)]
#![deny(clippy::all)]
#![warn(clippy::pedantic)]
//! RunuX Performance Model — Analytical roofline estimator for RISC-V LLM
//!
//! Models LLM inference performance using the **roofline model** to identify
//! whether each operation is compute-bound or memory-bound. This enables
//! accurate performance prediction without physical hardware.
//!
//! # Roofline Model
//!
//! ```text
//! Throughput (GFLOPS)
//! │         ┌────────── Compute Ceiling (K3: 128 GFLOPS)
//! │    ╱────┤
//! │   ╱     │
//! │  ╱      │← Compute-bound region
//! │ ╱       │
//! │╱        │
//! │← Mem-bound region
//! └──────────────────── Operational Intensity (FLOP/byte)
//! ```
//!
//! Autoregressive LLM decoding is **memory-bound** at batch_size=1:
//! - Each weight is loaded from DRAM but used for only 1 multiply
//! - Operational intensity ≈ 1 FLOP / 4 bytes = 0.25 (FP32)
//! - Performance limited by memory bandwidth, not compute peak
//!
//! Key insight: Quantization helps because it reduces bytes loaded per weight,
//! increasing operational intensity toward the compute ceiling.

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;

// ---------------------------------------------------------------------------
// Hardware Specifications
// ---------------------------------------------------------------------------

/// Complete hardware specification for performance modeling.
#[derive(Debug, Clone)]
pub struct HardwareSpec {
    /// Device name
    pub name: &'static str,
    /// Number of cores
    pub n_cores: usize,
    /// Clock frequency (GHz)
    pub clock_ghz: f32,
    /// VLEN (vector register width in bits)
    pub vlen_bits: usize,
    /// Peak FP32 GFLOPS per core
    pub fp32_gflops_per_core: f32,
    /// Peak INT8 TOPS per core
    pub int8_tops_per_core: f32,
    /// DRAM bandwidth (GB/s)
    pub dram_bw_gbs: f32,
    /// L1 cache size per core (bytes)
    pub l1_size: usize,
    /// L2 cache size (shared or per-core, bytes)
    pub l2_size: usize,
    /// L1 bandwidth (GB/s)
    pub l1_bw_gbs: f32,
    /// L2 bandwidth (GB/s)
    pub l2_bw_gbs: f32,
    /// Whether this device has dedicated AI accelerator cores
    pub has_ai_cores: bool,
    /// AI core peak TOPS (if has_ai_cores)
    pub ai_core_tops: f32,
}

impl HardwareSpec {
    /// SpacemiT K1 (BPI-F3): 8× X60 cores, RVV 1.0, LPDDR4.
    pub fn spacemit_k1() -> Self {
        Self {
            name: "SpacemiT K1 (BPI-F3)",
            n_cores: 8,
            clock_ghz: 1.6,
            vlen_bits: 256,
            fp32_gflops_per_core: 2.0,  // 256-bit × 1.6GHz / 32-bit × 2 (FMA)
            int8_tops_per_core: 0.25,
            dram_bw_gbs: 12.8,          // LPDDR4-3200 single-channel
            l1_size: 32 * 1024,         // 32KB per core
            l2_size: 1024 * 1024,       // 1MB shared
            l1_bw_gbs: 51.2,            // 2× clock × 256-bit / core
            l2_bw_gbs: 25.6,
            has_ai_cores: false,
            ai_core_tops: 0.0,
        }
    }

    /// SpacemiT K3 (AIBOX-K3): 8× X100 + 8× A100 AI cores, LPDDR5.
    pub fn spacemit_k3() -> Self {
        Self {
            name: "SpacemiT K3 (AIBOX-K3)",
            n_cores: 8,
            clock_ghz: 2.0,
            vlen_bits: 1024,
            fp32_gflops_per_core: 16.0,  // 1024-bit × 2GHz / 32-bit × 2 (FMA)
            int8_tops_per_core: 0.5,
            dram_bw_gbs: 51.2,           // LPDDR5-6400 dual-channel
            l1_size: 64 * 1024,          // 64KB per core
            l2_size: 4 * 1024 * 1024,    // 4MB shared
            l1_bw_gbs: 128.0,
            l2_bw_gbs: 64.0,
            has_ai_cores: true,
            ai_core_tops: 60.0,          // 8× A100 AI cores
        }
    }

    /// Total peak FP32 GFLOPS.
    pub fn total_fp32_gflops(&self) -> f32 {
        self.fp32_gflops_per_core * self.n_cores as f32
    }

    /// Roofline ridge point (FLOP/byte where compute = memory ceiling).
    pub fn ridge_point_fp32(&self) -> f32 {
        if self.dram_bw_gbs > 0.0 {
            self.total_fp32_gflops() / self.dram_bw_gbs
        } else {
            0.0
        }
    }
}

// ---------------------------------------------------------------------------
// Operation Cost Model
// ---------------------------------------------------------------------------

/// Type of operation for cost estimation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpType {
    /// Linear projection: y = Wx + b
    Linear,
    /// Attention score computation: S = Q·K^T
    AttentionScore,
    /// Attention output: O = softmax(S)·V
    AttentionOutput,
    /// Softmax (element-wise with reduction)
    Softmax,
    /// RMS normalization
    RmsNorm,
    /// SiLU / GELU activation
    Activation,
    /// RoPE positional encoding
    RoPE,
    /// Embedding lookup
    Embedding,
}

/// Cost estimate for a single operation.
#[derive(Debug, Clone)]
pub struct OpCost {
    /// Operation type
    pub op: OpType,
    /// FLOPs for this operation
    pub flops: u64,
    /// Bytes read from memory
    pub bytes_read: u64,
    /// Bytes written to memory
    pub bytes_written: u64,
    /// Operational intensity (FLOP / byte)
    pub op_intensity: f32,
    /// Whether this operation is memory-bound on the target hardware
    pub is_memory_bound: bool,
    /// Estimated time (microseconds)
    pub estimated_us: f32,
    /// Estimated throughput (GFLOPS achieved)
    pub achieved_gflops: f32,
}

impl OpCost {
    fn compute(
        op: OpType,
        flops: u64,
        bytes_read: u64,
        bytes_written: u64,
        hw: &HardwareSpec,
    ) -> Self {
        let total_bytes = bytes_read + bytes_written;
        let op_intensity = if total_bytes > 0 {
            flops as f32 / total_bytes as f32
        } else {
            f32::MAX
        };

        let is_memory_bound = op_intensity < hw.ridge_point_fp32();

        // Estimate time: min of compute-limited and memory-limited time
        let compute_time_us = if hw.total_fp32_gflops() > 0.0 {
            flops as f32 / (hw.total_fp32_gflops() * 1e3) // GFLOPS → µs
        } else {
            f32::MAX
        };

        let memory_time_us = if hw.dram_bw_gbs > 0.0 {
            total_bytes as f32 / (hw.dram_bw_gbs * 1e3) // GB/s → µs
        } else {
            f32::MAX
        };

        let estimated_us = compute_time_us.max(memory_time_us);

        let achieved_gflops = if estimated_us > 0.0 {
            flops as f32 / (estimated_us * 1e3)
        } else {
            0.0
        };

        Self {
            op,
            flops,
            bytes_read,
            bytes_written,
            op_intensity,
            is_memory_bound,
            estimated_us,
            achieved_gflops,
        }
    }
}

// ---------------------------------------------------------------------------
// Transformer Layer Cost Model
// ---------------------------------------------------------------------------

/// Model configuration for cost estimation.
#[derive(Debug, Clone)]
pub struct ModelParams {
    pub name: &'static str,
    pub hidden_dim: usize,
    pub intermediate_dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub vocab_size: usize,
    pub quant_bits: u32, // 4, 8, 16, or 32
}

impl ModelParams {
    pub fn head_dim(&self) -> usize {
        self.hidden_dim / self.n_heads
    }

    pub fn qwen2_0_5b_q4() -> Self {
        Self {
            name: "Qwen 2.5 0.5B Q4_K_M",
            hidden_dim: 896,
            intermediate_dim: 4864,
            n_layers: 24,
            n_heads: 14,
            n_kv_heads: 2,
            vocab_size: 151936,
            quant_bits: 4,
        }
    }

    pub fn deepseek_r1_1_5b_q4() -> Self {
        Self {
            name: "DeepSeek R1 1.5B Q4_K_M",
            hidden_dim: 1536,
            intermediate_dim: 8960,
            n_layers: 28,
            n_heads: 12,
            n_kv_heads: 2,
            vocab_size: 151936,
            quant_bits: 4,
        }
    }

    pub fn qwen2_7b_q4() -> Self {
        Self {
            name: "Qwen 2.5 7B Q4_K_M",
            hidden_dim: 3584,
            intermediate_dim: 18944,
            n_layers: 28,
            n_heads: 28,
            n_kv_heads: 4,
            vocab_size: 151936,
            quant_bits: 4,
        }
    }

    pub fn deepseek_r1_14b_q4() -> Self {
        Self {
            name: "DeepSeek R1 14B Q4_K_M",
            hidden_dim: 5120,
            intermediate_dim: 13824,
            n_layers: 40,
            n_heads: 40,
            n_kv_heads: 8,
            vocab_size: 151936,
            quant_bits: 4,
        }
    }

    /// Bytes per weight element for the given quantization.
    pub fn bytes_per_weight(&self) -> f32 {
        self.quant_bits as f32 / 8.0
    }
}

// ---------------------------------------------------------------------------
// Per-Token Cost Breakdown
// ---------------------------------------------------------------------------

/// Full cost breakdown for generating one token (decode step).
#[derive(Debug, Clone)]
pub struct TokenCostBreakdown {
    /// Model name
    pub model: &'static str,
    /// Hardware name
    pub hardware: &'static str,
    /// Sequence length at time of generation
    pub seq_len: usize,
    /// Per-operation costs
    pub ops: Vec<OpCost>,
    /// Total time per token (microseconds)
    pub total_us: f32,
    /// Estimated tokens per second
    pub tokens_per_second: f32,
    /// Time-to-first-token estimate (µs) — includes full prompt processing
    pub ttft_us: f32,
    /// Memory-bound fraction (0.0 – 1.0)
    pub memory_bound_fraction: f32,
}

/// Estimate per-token cost for autoregressive decoding at batch_size=1.
pub fn estimate_token_cost(
    model: &ModelParams,
    hw: &HardwareSpec,
    seq_len: usize,
) -> TokenCostBreakdown {
    let d = model.hidden_dim;
    let d_ff = model.intermediate_dim;
    let d_k = model.head_dim();
    let nh = model.n_heads;
    let nkv = model.n_kv_heads;
    let bpw = model.bytes_per_weight();

    let mut ops = Vec::new();

    // Per layer costs (decode step, batch_size=1):
    for _layer in 0..model.n_layers {
        // --- Attention block ---

        // 1. RMSNorm (pre-attention)
        ops.push(OpCost::compute(
            OpType::RmsNorm,
            (d * 3) as u64, // 3 ops per element: x², mean, normalize
            (d * 4 * 2) as u64, // read x + gamma
            (d * 4) as u64,
            hw,
        ));

        // 2. Q, K, V projections (linear layers)
        // Q: [1 × d] @ [d × d] → [1 × d]
        let q_flops = 2 * d * d; // 2 for FMA
        let q_bytes_read = (d as f32 * d as f32 * bpw) as u64 + (d * 4) as u64;
        ops.push(OpCost::compute(OpType::Linear, q_flops as u64, q_bytes_read, (d * 4) as u64, hw));

        // K: [1 × d] @ [d × (nkv×d_k)] → [1 × nkv×d_k]
        let kv_dim = nkv * d_k;
        let k_flops = 2 * d * kv_dim;
        let k_bytes_read = (d as f32 * kv_dim as f32 * bpw) as u64 + (d * 4) as u64;
        ops.push(OpCost::compute(OpType::Linear, k_flops as u64, k_bytes_read, (kv_dim * 4) as u64, hw));

        // V: same as K
        ops.push(OpCost::compute(OpType::Linear, k_flops as u64, k_bytes_read, (kv_dim * 4) as u64, hw));

        // 3. RoPE (on Q and K)
        ops.push(OpCost::compute(
            OpType::RoPE,
            (d_k * nh * 6) as u64, // sin, cos, 4 multiplies per pair
            (d_k * nh * 4 * 2) as u64, // read Q + freqs
            (d_k * nh * 4) as u64,
            hw,
        ));

        // 4. Attention scores: S = Q·K^T for all heads (seq_len positions)
        // Per head: [1 × d_k] · [d_k × seq_len] → [1 × seq_len]
        let attn_flops = 2 * nh * d_k * seq_len;
        let attn_bytes = (seq_len * nkv * d_k * 2) as u64 + (nh * d_k * 4) as u64; // KV-cache (FP16) + Q
        ops.push(OpCost::compute(OpType::AttentionScore, attn_flops as u64, attn_bytes, (nh * seq_len * 4) as u64, hw));

        // 5. Softmax over scores
        ops.push(OpCost::compute(
            OpType::Softmax,
            (nh * seq_len * 5) as u64, // exp, sum, normalize
            (nh * seq_len * 4) as u64,
            (nh * seq_len * 4) as u64,
            hw,
        ));

        // 6. Attention output: O = scores · V
        let out_flops = 2 * nh * seq_len * d_k;
        let out_bytes = (seq_len * nkv * d_k * 2) as u64 + (nh * seq_len * 4) as u64;
        ops.push(OpCost::compute(OpType::AttentionOutput, out_flops as u64, out_bytes, (d * 4) as u64, hw));

        // 7. Output projection: [1 × d] @ [d × d]
        ops.push(OpCost::compute(OpType::Linear, q_flops as u64, q_bytes_read, (d * 4) as u64, hw));

        // --- FFN block ---

        // 8. RMSNorm (pre-FFN)
        ops.push(OpCost::compute(
            OpType::RmsNorm,
            (d * 3) as u64,
            (d * 4 * 2) as u64,
            (d * 4) as u64,
            hw,
        ));

        // 9. Gate projection: [1 × d] @ [d × d_ff]
        let gate_flops = 2 * d * d_ff;
        let gate_bytes = (d as f32 * d_ff as f32 * bpw) as u64 + (d * 4) as u64;
        ops.push(OpCost::compute(OpType::Linear, gate_flops as u64, gate_bytes, (d_ff * 4) as u64, hw));

        // 10. Up projection: same as gate
        ops.push(OpCost::compute(OpType::Linear, gate_flops as u64, gate_bytes, (d_ff * 4) as u64, hw));

        // 11. SiLU activation
        ops.push(OpCost::compute(
            OpType::Activation,
            (d_ff * 5) as u64,
            (d_ff * 4) as u64,
            (d_ff * 4) as u64,
            hw,
        ));

        // 12. Down projection: [1 × d_ff] @ [d_ff × d]
        let down_flops = 2 * d_ff * d;
        let down_bytes = (d_ff as f32 * d as f32 * bpw) as u64 + (d_ff * 4) as u64;
        ops.push(OpCost::compute(OpType::Linear, down_flops as u64, down_bytes, (d * 4) as u64, hw));
    }

    // Final RMSNorm + LM Head
    ops.push(OpCost::compute(OpType::RmsNorm, (d * 3) as u64, (d * 4 * 2) as u64, (d * 4) as u64, hw));
    let lm_flops = 2 * d * model.vocab_size;
    let lm_bytes = (d as f32 * model.vocab_size as f32 * bpw) as u64 + (d * 4) as u64;
    ops.push(OpCost::compute(OpType::Linear, lm_flops as u64, lm_bytes, (model.vocab_size * 4) as u64, hw));

    // Aggregate
    let total_us: f32 = ops.iter().map(|o| o.estimated_us).sum();
    let mem_bound_time: f32 = ops.iter().filter(|o| o.is_memory_bound).map(|o| o.estimated_us).sum();
    let memory_bound_fraction = if total_us > 0.0 { mem_bound_time / total_us } else { 0.0 };
    let tokens_per_second = if total_us > 0.0 { 1_000_000.0 / total_us } else { 0.0 };

    // TTFT: roughly seq_len × per-token cost (prefill processes all tokens)
    let ttft_us = total_us * seq_len as f32;

    TokenCostBreakdown {
        model: model.name,
        hardware: hw.name,
        seq_len,
        ops,
        total_us,
        tokens_per_second,
        ttft_us,
        memory_bound_fraction,
    }
}

// ---------------------------------------------------------------------------
// Model Compatibility Matrix
// ---------------------------------------------------------------------------

/// Whether a model can run on given hardware.
#[derive(Debug, Clone)]
pub struct CompatibilityEntry {
    pub model: &'static str,
    pub hardware: &'static str,
    pub fits_in_ram: bool,
    pub tokens_per_second: f32,
    pub max_context: usize,
    pub memory_bound_pct: f32,
    pub recommendation: &'static str,
}

/// Generate the full compatibility matrix.
pub fn compatibility_matrix() -> Vec<CompatibilityEntry> {
    let models = [
        ModelParams::qwen2_0_5b_q4(),
        ModelParams::deepseek_r1_1_5b_q4(),
        ModelParams::qwen2_7b_q4(),
        ModelParams::deepseek_r1_14b_q4(),
    ];

    let hardware = [
        (HardwareSpec::spacemit_k1(), 4_294_967_296usize), // 4GB
        (HardwareSpec::spacemit_k3(), 8_589_934_592usize), // 8GB
        (HardwareSpec::spacemit_k3(), 34_359_738_368usize), // 32GB
    ];

    let hw_names = [
        "BPI-F3 (4GB)",
        "AIBOX-K3 (8GB)",
        "AIBOX-K3 (32GB)",
    ];

    let mut results = Vec::new();

    for model in &models {
        for (idx, (hw, ram)) in hardware.iter().enumerate() {
            let weight_bytes = (model.hidden_dim * model.hidden_dim * model.n_layers * 7) as f32
                * model.bytes_per_weight();
            let fits = (weight_bytes as usize) < *ram * 8 / 10;

            let cost = estimate_token_cost(model, hw, 512);

            let max_ctx = if fits {
                let kv_per_tok = 2 * model.n_layers * model.n_kv_heads * model.head_dim() * 2;
                let remaining = if *ram > weight_bytes as usize { *ram - weight_bytes as usize } else { 0 };
                if kv_per_tok > 0 { remaining / kv_per_tok } else { 0 }
            } else {
                0
            };

            let rec = match (fits, cost.tokens_per_second > 5.0) {
                (true, true) => "✅ Excellent — production-ready",
                (true, false) => "⚠️ Fits but slow — consider smaller model",
                (false, _) => "❌ Does not fit — need more RAM or lower quant",
            };

            results.push(CompatibilityEntry {
                model: model.name,
                hardware: hw_names[idx],
                fits_in_ram: fits,
                tokens_per_second: cost.tokens_per_second,
                max_context: max_ctx.min(131072),
                memory_bound_pct: cost.memory_bound_fraction * 100.0,
                recommendation: rec,
            });
        }
    }

    results
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_k1_specs() {
        let k1 = HardwareSpec::spacemit_k1();
        assert_eq!(k1.n_cores, 8);
        assert_eq!(k1.vlen_bits, 256);
        assert!(k1.total_fp32_gflops() > 0.0);
        assert!(k1.ridge_point_fp32() > 0.0);
    }

    #[test]
    fn test_k3_faster_than_k1() {
        let k1 = HardwareSpec::spacemit_k1();
        let k3 = HardwareSpec::spacemit_k3();
        assert!(k3.total_fp32_gflops() > k1.total_fp32_gflops());
        assert!(k3.dram_bw_gbs > k1.dram_bw_gbs);
    }

    #[test]
    fn test_token_cost_qwen_k1() {
        let model = ModelParams::qwen2_0_5b_q4();
        let hw = HardwareSpec::spacemit_k1();
        let cost = estimate_token_cost(&model, &hw, 256);

        assert!(cost.total_us > 0.0, "Should have non-zero cost");
        assert!(cost.tokens_per_second > 0.0, "Should estimate >0 tok/s");
        assert!(cost.memory_bound_fraction > 0.5,
            "Batch=1 decode should be mostly memory-bound, got {:.0}%",
            cost.memory_bound_fraction * 100.0);
    }

    #[test]
    fn test_token_cost_k3_faster() {
        let model = ModelParams::deepseek_r1_1_5b_q4();
        let k1 = HardwareSpec::spacemit_k1();
        let k3 = HardwareSpec::spacemit_k3();

        let cost_k1 = estimate_token_cost(&model, &k1, 256);
        let cost_k3 = estimate_token_cost(&model, &k3, 256);

        assert!(cost_k3.tokens_per_second > cost_k1.tokens_per_second,
            "K3 ({:.1} tok/s) should be faster than K1 ({:.1} tok/s)",
            cost_k3.tokens_per_second, cost_k1.tokens_per_second);
    }

    #[test]
    fn test_memory_bound_decode() {
        // At batch_size=1, autoregressive decode should be memory-bound
        let model = ModelParams::qwen2_0_5b_q4();
        let hw = HardwareSpec::spacemit_k1();
        let cost = estimate_token_cost(&model, &hw, 512);

        // The linear layers (weight loading) dominate and are memory-bound
        let linear_ops: Vec<_> = cost.ops.iter().filter(|o| o.op == OpType::Linear).collect();
        let mem_bound_linears = linear_ops.iter().filter(|o| o.is_memory_bound).count();
        assert!(mem_bound_linears > linear_ops.len() / 2,
            "Most linear ops should be memory-bound at batch=1");
    }

    #[test]
    fn test_compatibility_matrix() {
        let matrix = compatibility_matrix();
        assert!(!matrix.is_empty());

        // Qwen 0.5B Q4 on BPI-F3 should fit
        let qwen_bpi = matrix.iter()
            .find(|e| e.model.contains("0.5B") && e.hardware.contains("BPI-F3"))
            .expect("Should have Qwen 0.5B on BPI-F3");
        assert!(qwen_bpi.fits_in_ram, "Qwen 0.5B Q4 should fit on 4GB");

        // DeepSeek 14B on BPI-F3 should NOT fit
        let ds14_bpi = matrix.iter()
            .find(|e| e.model.contains("14B") && e.hardware.contains("BPI-F3"))
            .expect("Should have DS 14B on BPI-F3");
        assert!(!ds14_bpi.fits_in_ram, "14B model should not fit on 4GB BPI-F3");
    }

    #[test]
    fn test_longer_context_slower() {
        let model = ModelParams::qwen2_0_5b_q4();
        let hw = HardwareSpec::spacemit_k1();

        let cost_256 = estimate_token_cost(&model, &hw, 256);
        let cost_4096 = estimate_token_cost(&model, &hw, 4096);

        assert!(cost_4096.total_us > cost_256.total_us,
            "Longer context should be slower: 256={:.0}µs, 4096={:.0}µs",
            cost_256.total_us, cost_4096.total_us);
    }
}
