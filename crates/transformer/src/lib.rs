// Copyright (c) 2026 Xavier Callens / Socrate AI. All Rights Reserved.
// SPDX-License-Identifier: LicenseRef-RunuX-Commercial
// This file is proprietary and confidential. Unauthorized copying,
// distribution, or use is strictly prohibited.

#![cfg_attr(not(test), no_std)]
#![deny(clippy::all)]
#![warn(clippy::pedantic)]
//! RunuX Transformer — Complete transformer forward pass for LLM inference
//!
//! Implements the decoder-only transformer architecture used by:
//! - LLaMA / LLaMA 2 / LLaMA 3
//! - Qwen 2 / Qwen 2.5
//! - DeepSeek R1 (Qwen-based distillations)
//!
//! # Architecture
//!
//! ```text
//! Input Token IDs
//!       │
//!       ▼
//! ┌─────────────┐
//! │  Embedding   │  token_id → hidden_dim vector
//! └──────┬──────┘
//!        │
//! ┌──────▼──────┐ ×N layers
//! │  RMSNorm     │
//! │  Attention   │  Multi-head / GQA, RoPE, KV-cache
//! │  + Residual  │
//! │  RMSNorm     │
//! │  FFN (SiLU)  │  gate_proj, up_proj, down_proj
//! │  + Residual  │
//! └──────┬──────┘
//!        │
//! ┌──────▼──────┐
//! │  RMSNorm     │
//! │  LM Head     │  hidden_dim → vocab_size logits
//! └──────┬──────┘
//!        │
//!        ▼
//!    Logits → Sampling → Next Token
//! ```

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;

use rvv_simd::{matmul_scalar_f32, softmax_f32, rms_norm_f32, silu_f32};

// ---------------------------------------------------------------------------
// Model Configuration
// ---------------------------------------------------------------------------

/// Transformer model hyperparameters.
///
/// These are loaded from GGUF metadata at model load time.
#[derive(Debug, Clone)]
pub struct TransformerConfig {
    /// Hidden dimension (embedding size)
    pub hidden_dim: usize,
    /// Intermediate dimension (FFN size, typically 4× or 8/3× hidden_dim)
    pub intermediate_dim: usize,
    /// Number of transformer layers
    pub n_layers: usize,
    /// Number of attention heads
    pub n_heads: usize,
    /// Number of key-value heads (< n_heads for GQA)
    pub n_kv_heads: usize,
    /// Vocabulary size
    pub vocab_size: usize,
    /// Maximum sequence length
    pub max_seq_len: usize,
    /// RMSNorm epsilon
    pub norm_eps: f32,
    /// RoPE theta (base frequency)
    pub rope_theta: f32,
}

impl TransformerConfig {
    /// Head dimension (hidden_dim / n_heads).
    pub fn head_dim(&self) -> usize {
        self.hidden_dim / self.n_heads
    }

    /// Number of heads per KV group (for GQA).
    pub fn n_heads_per_kv(&self) -> usize {
        self.n_heads / self.n_kv_heads
    }

    /// Create a Qwen 2.5 0.5B configuration.
    pub fn qwen2_0_5b() -> Self {
        Self {
            hidden_dim: 896,
            intermediate_dim: 4864,
            n_layers: 24,
            n_heads: 14,
            n_kv_heads: 2,
            vocab_size: 151936,
            max_seq_len: 32768,
            norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
        }
    }

    /// Create a DeepSeek R1 1.5B (distill) configuration.
    pub fn deepseek_r1_1_5b() -> Self {
        Self {
            hidden_dim: 1536,
            intermediate_dim: 8960,
            n_layers: 28,
            n_heads: 12,
            n_kv_heads: 2,
            vocab_size: 151936,
            max_seq_len: 131072,
            norm_eps: 1e-6,
            rope_theta: 10_000.0,
        }
    }
}

// ---------------------------------------------------------------------------
// RoPE (Rotary Position Embeddings)
// ---------------------------------------------------------------------------

/// Precomputed RoPE frequency pairs for positional encoding.
pub struct RopeFreqs {
    /// Cosine values: [max_seq_len × head_dim/2]
    cos: Vec<f32>,
    /// Sine values: [max_seq_len × head_dim/2]
    sin: Vec<f32>,
    /// Half of head dimension
    half_dim: usize,
}

impl RopeFreqs {
    /// Precompute RoPE frequencies for the given config.
    pub fn new(head_dim: usize, max_seq_len: usize, theta: f32) -> Self {
        let half_dim = head_dim / 2;
        let mut cos = vec![0.0f32; max_seq_len * half_dim];
        let mut sin = vec![0.0f32; max_seq_len * half_dim];

        for pos in 0..max_seq_len {
            for i in 0..half_dim {
                let freq = 1.0 / fast_powf(theta, (2 * i) as f32 / head_dim as f32);
                let angle = pos as f32 * freq;
                cos[pos * half_dim + i] = fast_cos(angle);
                sin[pos * half_dim + i] = fast_sin(angle);
            }
        }

        Self { cos, sin, half_dim }
    }

    /// Apply RoPE to a query or key vector at a given position.
    ///
    /// Rotates pairs of elements: (x0, x1) → (x0·cos - x1·sin, x0·sin + x1·cos)
    pub fn apply(&self, vec: &mut [f32], position: usize) {
        let base = position * self.half_dim;
        for i in 0..self.half_dim {
            if base + i >= self.cos.len() {
                break;
            }
            let cos_val = self.cos[base + i];
            let sin_val = self.sin[base + i];
            let x0 = vec[i];
            let x1 = vec[i + self.half_dim];
            vec[i] = x0 * cos_val - x1 * sin_val;
            vec[i + self.half_dim] = x0 * sin_val + x1 * cos_val;
        }
    }
}

// ---------------------------------------------------------------------------
// KV Cache
// ---------------------------------------------------------------------------

/// Key-Value cache for autoregressive decoding.
///
/// Stores computed key and value projections for all past positions,
/// avoiding recomputation during generation.
#[derive(Debug, Clone)]
pub struct KvCache {
    /// Key cache: [n_layers × max_seq_len × n_kv_heads × head_dim] (INT8)
    pub keys: Vec<i8>,
    /// Value cache: [n_layers × max_seq_len × n_kv_heads × head_dim] (INT8)
    pub values: Vec<i8>,
    /// Key block scales: [n_layers × max_seq_len × n_kv_heads]
    pub key_scales: Vec<f32>,
    /// Value block scales: [n_layers × max_seq_len × n_kv_heads]
    pub value_scales: Vec<f32>,
    /// Current sequence length (number of cached positions)
    pub seq_len: usize,
    /// Config dimensions
    n_layers: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_seq_len: usize,
}

impl KvCache {
    /// Create a new empty KV cache.
    pub fn new(config: &TransformerConfig) -> Self {
        let head_dim = config.head_dim();
        let cache_size = config.n_layers * config.max_seq_len * config.n_kv_heads * head_dim;
        let scale_size = config.n_layers * config.max_seq_len * config.n_kv_heads;
        Self {
            keys: vec![0i8; cache_size],
            values: vec![0i8; cache_size],
            key_scales: vec![0.0f32; scale_size],
            value_scales: vec![0.0f32; scale_size],
            seq_len: 0,
            n_layers: config.n_layers,
            n_kv_heads: config.n_kv_heads,
            head_dim,
            max_seq_len: config.max_seq_len,
        }
    }

    /// Get the stride for accessing cache[layer][position][kv_head][dim].
    fn stride(&self) -> (usize, usize, usize) {
        let head_stride = self.head_dim;
        let pos_stride = self.n_kv_heads * head_stride;
        let layer_stride = self.max_seq_len * pos_stride;
        (layer_stride, pos_stride, head_stride)
    }

    /// Store a key vector for a given layer, position, and KV head.
    pub fn store_key(
        &mut self,
        layer: usize,
        position: usize,
        kv_head: usize,
        key: &[f32],
    ) {
        let (ls, ps, hs) = self.stride();
        let offset = layer * ls + position * ps + kv_head * hs;
        let scale_offset = layer * (self.max_seq_len * self.n_kv_heads) + position * self.n_kv_heads + kv_head;
        let len = key.len().min(self.head_dim);
        
        let mut max_abs = 0.0f32;
        for &v in &key[..len] {
            let abs_v = v.abs();
            if abs_v > max_abs { max_abs = abs_v; }
        }
        let scale = max_abs / 127.0;
        self.key_scales[scale_offset] = scale;
        
        for i in 0..len {
            let q = if scale == 0.0 { 0.0 } else { key[i] / scale };
            self.keys[offset + i] = q.clamp(-127.0, 127.0) as i8;
        }
    }

    /// Store a value vector for a given layer, position, and KV head.
    pub fn store_value(
        &mut self,
        layer: usize,
        position: usize,
        kv_head: usize,
        value: &[f32],
    ) {
        let (ls, ps, hs) = self.stride();
        let offset = layer * ls + position * ps + kv_head * hs;
        let scale_offset = layer * (self.max_seq_len * self.n_kv_heads) + position * self.n_kv_heads + kv_head;
        let len = value.len().min(self.head_dim);
        
        let mut max_abs = 0.0f32;
        for &v in &value[..len] {
            let abs_v = v.abs();
            if abs_v > max_abs { max_abs = abs_v; }
        }
        let scale = max_abs / 127.0;
        self.value_scales[scale_offset] = scale;
        
        for i in 0..len {
            let q = if scale == 0.0 { 0.0 } else { value[i] / scale };
            self.values[offset + i] = q.clamp(-127.0, 127.0) as i8;
        }
    }

    /// Get a key vector and its scale for a given layer, position, and KV head.
    pub fn get_key(&self, layer: usize, position: usize, kv_head: usize) -> (&[i8], f32) {
        let (ls, ps, hs) = self.stride();
        let offset = layer * ls + position * ps + kv_head * hs;
        let scale_offset = layer * (self.max_seq_len * self.n_kv_heads) + position * self.n_kv_heads + kv_head;
        (&self.keys[offset..offset + self.head_dim], self.key_scales[scale_offset])
    }

    /// Get a value vector and its scale for a given layer, position, and KV head.
    pub fn get_value(&self, layer: usize, position: usize, kv_head: usize) -> (&[i8], f32) {
        let (ls, ps, hs) = self.stride();
        let offset = layer * ls + position * ps + kv_head * hs;
        let scale_offset = layer * (self.max_seq_len * self.n_kv_heads) + position * self.n_kv_heads + kv_head;
        (&self.values[offset..offset + self.head_dim], self.value_scales[scale_offset])
    }

    /// Memory usage in bytes.
    pub fn memory_bytes(&self) -> usize {
        self.keys.len() + self.values.len() + (self.key_scales.len() + self.value_scales.len()) * 4
    }

    /// Reset the cache for a new sequence.
    pub fn reset(&mut self) {
        self.seq_len = 0;
        // Don't zero — old data is never read past seq_len
    }
}

// ---------------------------------------------------------------------------
// Sampling
// ---------------------------------------------------------------------------

/// Sampling configuration for token generation.
#[derive(Debug, Clone)]
pub struct SamplingConfig {
    /// Temperature (1.0 = no change, < 1.0 = more deterministic)
    pub temperature: f32,
    /// Top-p (nucleus) sampling threshold
    pub top_p: f32,
    /// Top-k sampling (0 = disabled)
    pub top_k: usize,
    /// Repetition penalty (1.0 = disabled)
    pub repetition_penalty: f32,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
            repetition_penalty: 1.1,
        }
    }
}

/// Sample the next token from logits.
///
/// Applies temperature scaling, top-k filtering, top-p (nucleus) filtering,
/// then samples from the resulting distribution.
pub fn sample_token(
    logits: &[f32],
    config: &SamplingConfig,
    rng_state: &mut u64,
) -> u32 {
    let n = logits.len();
    if n == 0 {
        return 0;
    }

    // Temperature scaling
    let mut scaled = vec![0.0f32; n];
    let temp = config.temperature.max(1e-6);
    for i in 0..n {
        scaled[i] = logits[i] / temp;
    }

    // Top-k filtering
    if config.top_k > 0 && config.top_k < n {
        // Find the top-k threshold
        let mut sorted_logits: Vec<f32> = scaled.clone();
        // Simple partial sort for top-k
        sorted_logits.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap_or(core::cmp::Ordering::Equal));
        let threshold = sorted_logits[config.top_k - 1];
        for v in &mut scaled {
            if *v < threshold {
                *v = f32::NEG_INFINITY;
            }
        }
    }

    // Softmax to get probabilities
    softmax_f32(&mut scaled);

    // Top-p (nucleus) filtering
    if config.top_p < 1.0 {
        // Sort by probability descending
        let mut indexed: Vec<(usize, f32)> = scaled.iter().copied().enumerate().collect();
        indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(core::cmp::Ordering::Equal));

        let mut cumsum = 0.0f32;
        let mut cutoff_idx = indexed.len();
        for (i, (_, prob)) in indexed.iter().enumerate() {
            cumsum += prob;
            if cumsum > config.top_p {
                cutoff_idx = i + 1;
                break;
            }
        }

        // Zero out tokens below the nucleus
        let mut keep_set = vec![false; n];
        for (idx, _) in indexed.iter().take(cutoff_idx) {
            keep_set[*idx] = true;
        }
        for i in 0..n {
            if !keep_set[i] {
                scaled[i] = 0.0;
            }
        }

        // Re-normalize
        let sum: f32 = scaled.iter().sum();
        if sum > 0.0 {
            for v in &mut scaled {
                *v /= sum;
            }
        }
    }

    // Sample from the distribution using xorshift64
    let r = xorshift64_f32(rng_state);
    let mut cumsum = 0.0f32;
    for (i, &prob) in scaled.iter().enumerate() {
        cumsum += prob;
        if r < cumsum {
            return i as u32;
        }
    }

    // Fallback: argmax
    scaled.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(core::cmp::Ordering::Equal))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

/// Greedy sampling (argmax).
pub fn sample_greedy(logits: &[f32]) -> u32 {
    logits.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(core::cmp::Ordering::Equal))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Transformer Forward Pass Components
// ---------------------------------------------------------------------------

/// Apply RMSNorm to a vector in-place.
pub fn apply_rms_norm(x: &mut [f32], weight: &[f32], eps: f32) {
    rms_norm_f32(x, weight, eps);
}

/// Compute attention scores for a single head.
///
/// Returns softmax(Q·K^T / sqrt(d_k)) · V
pub fn attention_head(
    query: &[f32],      // [head_dim]
    kv_cache: &KvCache,
    layer: usize,
    kv_head: usize,
    seq_len: usize,
    head_dim: usize,
) -> Vec<f32> {
    let scale = 1.0 / fast_sqrt(head_dim as f32);

    // Compute attention scores: Q · K^T for all cached positions
    let mut scores = vec![0.0f32; seq_len];
    for pos in 0..seq_len {
        let (key, key_scale) = kv_cache.get_key(layer, pos, kv_head);
        let mut dot = 0.0f32;
        for d in 0..head_dim {
            dot += query[d] * (key[d] as f32);
        }
        scores[pos] = dot * key_scale * scale;
    }

    // Causal masking is implicit — we only score up to seq_len

    // Softmax over scores
    softmax_f32(&mut scores);

    // Weighted sum of values: sum(score[pos] * V[pos])
    let mut output = vec![0.0f32; head_dim];
    for pos in 0..seq_len {
        let (value, val_scale) = kv_cache.get_value(layer, pos, kv_head);
        let s = scores[pos] * val_scale;
        for d in 0..head_dim {
            output[d] += s * (value[d] as f32);
        }
    }

    output
}

/// FFN (SwiGLU) forward pass.
///
/// out = down_proj(silu(gate_proj(x)) * up_proj(x))
pub fn ffn_forward(
    x: &[f32],
    gate_weight: &[f32],  // [intermediate_dim × hidden_dim]
    up_weight: &[f32],    // [intermediate_dim × hidden_dim]
    down_weight: &[f32],  // [hidden_dim × intermediate_dim]
    hidden_dim: usize,
    intermediate_dim: usize,
) -> Vec<f32> {
    // gate = gate_weight @ x
    let mut gate = vec![0.0f32; intermediate_dim];
    matmul_scalar_f32(gate_weight, x, &mut gate, intermediate_dim, hidden_dim, 1);

    // up = up_weight @ x
    let mut up = vec![0.0f32; intermediate_dim];
    matmul_scalar_f32(up_weight, x, &mut up, intermediate_dim, hidden_dim, 1);

    // gate = silu(gate) * up
    silu_f32(&mut gate);
    for i in 0..intermediate_dim {
        gate[i] *= up[i];
    }

    // output = down_weight @ gate
    let mut output = vec![0.0f32; hidden_dim];
    matmul_scalar_f32(down_weight, &gate, &mut output, hidden_dim, intermediate_dim, 1);

    output
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Xorshift64 PRNG — returns a float in [0, 1).
fn xorshift64_f32(state: &mut u64) -> f32 {
    let mut s = *state;
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    *state = s;
    (s as f32) / (u64::MAX as f32)
}

/// Fast approximate sqrt (Newton-Raphson).
fn fast_sqrt(x: f32) -> f32 {
    if x <= 0.0 { return 0.0; }
    let mut guess = x;
    for _ in 0..5 {
        guess = 0.5 * (guess + x / guess);
    }
    guess
}

/// Fast approximate cos using Taylor series.
fn fast_cos(x: f32) -> f32 {
    // Reduce to [0, 2π]
    let pi2 = 6.283_185_5;
    let mut a = x % pi2;
    if a < 0.0 { a += pi2; }
    // Taylor series: cos(x) ≈ 1 - x²/2 + x⁴/24 - x⁶/720
    // Center around 0 by shifting to [-π, π]
    let pi = 3.141_592_7;
    if a > pi { a -= pi2; }
    let x2 = a * a;
    let x4 = x2 * x2;
    let x6 = x4 * x2;
    1.0 - x2 * 0.5 + x4 * 0.041_666_668 - x6 * 0.001_388_889
}

/// Fast approximate sin using Taylor series.
fn fast_sin(x: f32) -> f32 {
    let pi2 = 6.283_185_5;
    let mut a = x % pi2;
    if a < 0.0 { a += pi2; }
    let pi = 3.141_592_7;
    if a > pi { a -= pi2; }
    let x2 = a * a;
    let x4 = x2 * x2;
    let x6 = x4 * x2;
    a - a * x2 * 0.166_666_67 + a * x4 * 0.008_333_334 - a * x6 * 0.000_198_413
}

/// Fast approximate power function.
fn fast_powf(base: f32, exp: f32) -> f32 {
    // Use exp(exp * ln(base))
    if base <= 0.0 { return 0.0; }
    fast_exp(exp * fast_ln(base))
}

/// Fast approximate natural logarithm.
fn fast_ln(x: f32) -> f32 {
    if x <= 0.0 { return f32::NEG_INFINITY; }
    let bits = x.to_bits();
    let exponent = ((bits >> 23) & 0xFF) as f32 - 127.0;
    let mantissa = f32::from_bits((bits & 0x007F_FFFF) | 0x3F80_0000);
    // ln(x) = exponent * ln(2) + ln(mantissa)
    // ln(mantissa) ≈ (mantissa - 1) - (mantissa - 1)²/2 for mantissa near 1
    let m = mantissa - 1.0;
    exponent * 0.693_147_2 + m - 0.5 * m * m + 0.333_333_3 * m * m * m
}

/// Fast approximate exponential.
fn fast_exp(x: f32) -> f32 {
    if x < -88.0 { return 0.0; }
    if x > 88.0 { return f32::MAX; }
    // Clamped range for bit manipulation trick
    let a = 12102203.0f32; // 2^23 / ln(2)
    let b = 1065353216.0f32; // 127 * 2^23 (IEEE 754 bias)
    let bits = ((a * x + b) as u32).max(0).min(0x7F80_0000);
    f32::from_bits(bits)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_head_dim() {
        let config = TransformerConfig::qwen2_0_5b();
        assert_eq!(config.head_dim(), 64); // 896 / 14
        assert_eq!(config.n_heads_per_kv(), 7); // 14 / 2
    }

    #[test]
    fn test_config_deepseek() {
        let config = TransformerConfig::deepseek_r1_1_5b();
        assert_eq!(config.head_dim(), 128); // 1536 / 12
        assert_eq!(config.n_heads_per_kv(), 6); // 12 / 2
    }

    #[test]
    fn test_rope_apply() {
        let rope = RopeFreqs::new(8, 16, 10_000.0);
        let mut vec = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let original = vec.clone();
        rope.apply(&mut vec, 0);
        // At position 0, cos=1 and sin=0, so vector should be unchanged
        for i in 0..8 {
            assert!((vec[i] - original[i]).abs() < 0.01,
                "RoPE at pos=0 should be near identity, idx={}: got={} expected={}",
                i, vec[i], original[i]);
        }
    }

    #[test]
    fn test_rope_different_positions() {
        let rope = RopeFreqs::new(8, 16, 10_000.0);
        let mut vec_pos0 = vec![1.0; 8];
        let mut vec_pos5 = vec![1.0; 8];
        rope.apply(&mut vec_pos0, 0);
        rope.apply(&mut vec_pos5, 5);
        // Different positions should give different results
        assert!(vec_pos0 != vec_pos5, "Different positions should produce different rotations");
    }

    #[test]
    fn test_kv_cache_store_retrieve() {
        let config = TransformerConfig {
            hidden_dim: 16,
            intermediate_dim: 32,
            n_layers: 2,
            n_heads: 4,
            n_kv_heads: 2,
            vocab_size: 100,
            max_seq_len: 8,
            norm_eps: 1e-6,
            rope_theta: 10_000.0,
        };
        let mut cache = KvCache::new(&config);
        let head_dim = config.head_dim(); // 4

        // Store a key at layer 0, position 0, kv_head 0
        let key = vec![1.0, 2.0, 3.0, 4.0];
        cache.store_key(0, 0, 0, &key);

        // Retrieve it
        let (retrieved, scale) = cache.get_key(0, 0, 0);
        // It's INT8, so let's dequantize to compare
        let mut deq = vec![0.0f32; 4];
        for i in 0..4 { deq[i] = retrieved[i] as f32 * scale; }
        // 4.0/127.0 * 127 = 4.0. We should expect near original values
        assert!((deq[3] - 4.0).abs() < 0.1);

        // Different position should be zeros
        let (other, _) = cache.get_key(0, 1, 0);
        assert_eq!(other, &[0, 0, 0, 0]);
    }

    #[test]
    fn test_kv_cache_memory() {
        let config = TransformerConfig::qwen2_0_5b();
        let cache = KvCache::new(&config);
        // 2 (K+V) × n_layers × max_seq_len × n_kv_heads × head_dim × 1 bytes + scales
        let items = 24 * 32768 * 2 * 64;
        let scales = 24 * 32768 * 2;
        let expected = 2 * items + 2 * scales * 4;
        assert_eq!(cache.memory_bytes(), expected);
    }

    #[test]
    fn test_sample_greedy() {
        let logits = vec![0.1, 0.3, 0.9, 0.2, 0.5];
        assert_eq!(sample_greedy(&logits), 2); // Index of max (0.9)
    }

    #[test]
    fn test_sample_deterministic_low_temp() {
        let logits = vec![0.1, 0.3, 0.9, 0.2, 0.5];
        let config = SamplingConfig {
            temperature: 0.01, // Near-greedy
            top_p: 1.0,
            top_k: 0,
            repetition_penalty: 1.0,
        };
        let mut rng = 42u64;
        let token = sample_token(&logits, &config, &mut rng);
        assert_eq!(token, 2); // Should pick argmax at very low temperature
    }

    #[test]
    fn test_fast_trig() {
        // Verify our fast trig is reasonably accurate
        let cos_0 = fast_cos(0.0);
        assert!((cos_0 - 1.0).abs() < 0.01, "cos(0) should be ~1.0, got {}", cos_0);

        let sin_0 = fast_sin(0.0);
        assert!(sin_0.abs() < 0.01, "sin(0) should be ~0.0, got {}", sin_0);
    }

    #[test]
    fn test_fast_exp_ln() {
        let x = 2.0f32;
        let ln_x = fast_ln(x);
        assert!((ln_x - 0.693).abs() < 0.05, "ln(2) should be ~0.693, got {}", ln_x);

        let exp_0 = fast_exp(0.0);
        assert!((exp_0 - 1.0).abs() < 0.01, "exp(0) should be ~1.0, got {}", exp_0);
    }

    #[test]
    fn test_attention_head_single_pos() {
        let config = TransformerConfig {
            hidden_dim: 8,
            intermediate_dim: 16,
            n_layers: 1,
            n_heads: 2,
            n_kv_heads: 1,
            vocab_size: 10,
            max_seq_len: 4,
            norm_eps: 1e-6,
            rope_theta: 10_000.0,
        };
        let mut cache = KvCache::new(&config);
        let head_dim = config.head_dim(); // 4

        // Store one KV pair
        let key = vec![1.0, 0.0, 0.0, 0.0];
        let value = vec![0.0, 1.0, 0.0, 0.0];
        cache.store_key(0, 0, 0, &key);
        cache.store_value(0, 0, 0, &value);

        // Query aligned with key
        let query = vec![1.0, 0.0, 0.0, 0.0];
        let out = attention_head(&query, &cache, 0, 0, 1, head_dim);

        // With single position, softmax([score]) = [1.0], so output = value
        assert_eq!(out.len(), head_dim);
        assert!((out[0] - 0.0).abs() < 0.01);
        assert!((out[1] - 1.0).abs() < 0.01);
    }
}
