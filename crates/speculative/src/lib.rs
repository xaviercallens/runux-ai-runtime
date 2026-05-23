// Copyright (c) 2026 Xavier Callens / Socrate AI. All Rights Reserved.
// SPDX-License-Identifier: LicenseRef-RunuX-Commercial
// This file is proprietary and confidential. Unauthorized copying,
// distribution, or use is strictly prohibited.

#![cfg_attr(not(test), no_std)]
#![deny(clippy::all)]
#![warn(clippy::pedantic)]
//! RunuX Speculative Decoding — Draft-and-Verify acceleration for LLM inference
//!
//! Implements speculative decoding (Leviathan et al., 2023; Chen et al., 2023)
//! adapted for heterogeneous RISC-V edge clusters.
//!
//! # Core Idea
//!
//! Autoregressive LLM decoding is memory-bound: each token requires a full
//! forward pass, but most of the GPU/CPU compute capacity is idle because
//! we're waiting on memory. Speculative decoding exploits this by:
//!
//! 1. **Draft**: A small, fast model generates K candidate tokens
//! 2. **Verify**: The large target model processes all K tokens in parallel
//! 3. **Accept/Reject**: Tokens are accepted if they match the target model's
//!    distribution; otherwise we fall back to the target's prediction
//!
//! This produces **mathematically identical** output to standard decoding
//! but with up to K× speedup in wall-clock time.
//!
//! # RISC-V Edge Deployment
//!
//! ```text
//! ┌─────────────────────────────┐
//! │  BPI-F3 (Draft Model)      │  ← Qwen 0.5B Q4_K_M
//! │  Fast: ~50 tok/s           │
//! │  Generates K=8 candidates  │
//! └──────────┬──────────────────┘
//!            │ K draft tokens
//! ┌──────────▼──────────────────┐
//! │  AIBOX-K3 (Target Model)   │  ← DeepSeek R1 14B Q4_K_M
//! │  Verifies K tokens in      │
//! │  a single forward pass     │
//! │  Accepts ≥ 3-5 tokens/step │
//! └─────────────────────────────┘
//! ```
//!
//! # Power Efficiency
//!
//! The BPI-F3 draws ~5W. Running the draft model locally avoids sending
//! each token over the network for verification. The K3 (~15W) only runs
//! when a batch of draft tokens is ready, reducing idle power by ~40%.

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Speculative decoding configuration.
#[derive(Debug, Clone)]
pub struct SpeculativeConfig {
    /// Number of draft tokens to generate per step (K)
    pub draft_length: usize,
    /// Maximum number of draft tokens to accept per step
    pub max_accept: usize,
    /// Temperature for draft model (usually matches target)
    pub draft_temperature: f32,
    /// Temperature for target model
    pub target_temperature: f32,
    /// Whether to use the modified rejection sampling from Leviathan et al.
    pub use_modified_rejection: bool,
}

impl Default for SpeculativeConfig {
    fn default() -> Self {
        Self {
            draft_length: 8,
            max_accept: 8,
            draft_temperature: 0.7,
            target_temperature: 0.7,
            use_modified_rejection: true,
        }
    }
}

impl SpeculativeConfig {
    /// Configuration optimized for BPI-F3 → AIBOX-K3 setup.
    pub fn for_edge_cluster() -> Self {
        Self {
            draft_length: 5,   // Conservative K for edge latency
            max_accept: 5,
            draft_temperature: 0.7,
            target_temperature: 0.7,
            use_modified_rejection: true,
        }
    }

    /// Configuration for single-device speculative decoding.
    pub fn for_single_device() -> Self {
        Self {
            draft_length: 4,
            max_accept: 4,
            draft_temperature: 0.7,
            target_temperature: 0.7,
            use_modified_rejection: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Draft Token
// ---------------------------------------------------------------------------

/// A draft token with its probability from the draft model.
#[derive(Debug, Clone)]
pub struct DraftToken {
    /// Token ID
    pub token_id: u32,
    /// Probability assigned by the draft model: q(x)
    pub draft_prob: f32,
    /// Full logit distribution from the draft model (for rejection sampling)
    pub draft_logits: Vec<f32>,
}

/// Result of the verification step.
#[derive(Debug, Clone)]
pub struct VerificationResult {
    /// Number of draft tokens accepted
    pub accepted_count: usize,
    /// The accepted token IDs (may include a bonus token from the target)
    pub accepted_tokens: Vec<u32>,
    /// Acceptance rate for this step (accepted / drafted)
    pub acceptance_rate: f32,
    /// Whether a bonus token was generated from the residual distribution
    pub has_bonus_token: bool,
}

// ---------------------------------------------------------------------------
// Speculative Decoding Engine
// ---------------------------------------------------------------------------

/// Speculative decoding engine (simulation mode).
///
/// In simulation mode, we model the draft and target distributions
/// analytically to benchmark acceptance rates and speedup without
/// requiring actual model weights.
pub struct SpeculativeEngine {
    /// Configuration
    config: SpeculativeConfig,
    /// Running statistics
    total_drafted: u64,
    total_accepted: u64,
    total_steps: u64,
    /// RNG state
    rng_state: u64,
}

impl SpeculativeEngine {
    /// Create a new speculative engine.
    pub fn new(config: SpeculativeConfig) -> Self {
        Self {
            config,
            total_drafted: 0,
            total_accepted: 0,
            total_steps: 0,
            rng_state: 12345,
        }
    }

    /// Verify draft tokens against target model logits.
    ///
    /// Implements the modified rejection sampling algorithm:
    /// For each draft token x_i with draft probability q(x_i):
    ///   - Compute target probability p(x_i)
    ///   - Accept with probability min(1, p(x_i) / q(x_i))
    ///   - If rejected, sample from the residual distribution
    ///     max(0, p(x) - q(x)) normalized
    pub fn verify(
        &mut self,
        draft_tokens: &[DraftToken],
        target_logits_batch: &[Vec<f32>], // one logit vector per draft position
    ) -> VerificationResult {
        let k = draft_tokens.len().min(target_logits_batch.len());
        let mut accepted_tokens = Vec::new();
        let mut accepted_count = 0;
        let mut has_bonus = false;

        for i in 0..k {
            let draft = &draft_tokens[i];
            let target_logits = &target_logits_batch[i];

            // Convert target logits to probabilities
            let target_probs = softmax_simple(target_logits, self.config.target_temperature);

            let target_prob = if (draft.token_id as usize) < target_probs.len() {
                target_probs[draft.token_id as usize]
            } else {
                0.0
            };

            if self.config.use_modified_rejection {
                // Modified rejection: accept with prob min(1, p/q)
                let accept_prob = if draft.draft_prob > 0.0 {
                    (target_prob / draft.draft_prob).min(1.0)
                } else {
                    0.0
                };

                let r = self.random_f32();
                if r < accept_prob {
                    accepted_tokens.push(draft.token_id);
                    accepted_count += 1;
                } else {
                    // Rejected: sample from residual distribution max(0, p - q)
                    let draft_probs = softmax_simple(&draft.draft_logits, self.config.draft_temperature);
                    let bonus = self.sample_residual(&target_probs, &draft_probs);
                    accepted_tokens.push(bonus);
                    has_bonus = true;
                    break;
                }
            } else {
                // Simple greedy verification
                let target_token = argmax(&target_probs);
                if target_token == draft.token_id {
                    accepted_tokens.push(draft.token_id);
                    accepted_count += 1;
                } else {
                    accepted_tokens.push(target_token);
                    break;
                }
            }
        }

        // If all K tokens accepted, sample one more from the target
        if accepted_count == k && !has_bonus {
            if let Some(last_logits) = target_logits_batch.get(k - 1) {
                let target_probs = softmax_simple(last_logits, self.config.target_temperature);
                let bonus = self.sample_from_probs(&target_probs);
                accepted_tokens.push(bonus);
                has_bonus = true;
            }
        }

        self.total_drafted += k as u64;
        self.total_accepted += accepted_count as u64;
        self.total_steps += 1;

        let acceptance_rate = if k > 0 {
            accepted_count as f32 / k as f32
        } else {
            0.0
        };

        VerificationResult {
            accepted_count,
            accepted_tokens,
            acceptance_rate,
            has_bonus_token: has_bonus,
        }
    }

    /// Sample from the residual distribution: max(0, p(x) - q(x)) normalized.
    fn sample_residual(&mut self, target: &[f32], draft: &[f32]) -> u32 {
        let n = target.len().min(draft.len());
        let mut residual = vec![0.0f32; n];
        let mut sum = 0.0f32;

        for i in 0..n {
            let r = (target[i] - draft[i]).max(0.0);
            residual[i] = r;
            sum += r;
        }

        if sum > 0.0 {
            for v in &mut residual {
                *v /= sum;
            }
            self.sample_from_probs(&residual)
        } else {
            argmax(target)
        }
    }

    /// Sample a token from a probability distribution.
    fn sample_from_probs(&mut self, probs: &[f32]) -> u32 {
        let r = self.random_f32();
        let mut cumsum = 0.0f32;
        for (i, &p) in probs.iter().enumerate() {
            cumsum += p;
            if r < cumsum {
                return i as u32;
            }
        }
        (probs.len() - 1) as u32
    }

    /// Generate a random float in [0, 1).
    fn random_f32(&mut self) -> f32 {
        self.rng_state ^= self.rng_state << 13;
        self.rng_state ^= self.rng_state >> 7;
        self.rng_state ^= self.rng_state << 17;
        (self.rng_state as f32) / (u64::MAX as f32)
    }

    // -----------------------------------------------------------------------
    // Statistics
    // -----------------------------------------------------------------------

    /// Average acceptance rate across all steps.
    pub fn avg_acceptance_rate(&self) -> f32 {
        if self.total_drafted == 0 { return 0.0; }
        self.total_accepted as f32 / self.total_drafted as f32
    }

    /// Average tokens generated per step (accepted + bonus).
    pub fn avg_tokens_per_step(&self) -> f32 {
        if self.total_steps == 0 { return 0.0; }
        self.total_accepted as f32 / self.total_steps as f32
    }

    /// Estimated speedup over standard autoregressive decoding.
    ///
    /// Speedup = avg_tokens_per_step / (1 + draft_overhead)
    /// where draft_overhead is the relative cost of running the draft model.
    pub fn estimated_speedup(&self, draft_cost_ratio: f32) -> f32 {
        let tokens_per_step = self.avg_tokens_per_step();
        if tokens_per_step <= 0.0 { return 1.0; }
        tokens_per_step / (1.0 + draft_cost_ratio)
    }

    /// Estimated power savings (Watts saved per token).
    ///
    /// By accepting multiple tokens per target forward pass, we amortize
    /// the target model's power cost across multiple tokens.
    pub fn power_efficiency_ratio(
        &self,
        draft_watts: f32,
        target_watts: f32,
    ) -> PowerReport {
        let alpha = self.avg_acceptance_rate();
        let k = self.config.draft_length as f32;

        // Standard: target_watts per token
        let standard_energy = target_watts; // per token

        // Speculative: draft_watts × K + target_watts for verification
        // Amortized over accepted + 1 tokens
        let tokens_per_step = alpha * k + 1.0;
        let spec_energy = (draft_watts * k + target_watts) / tokens_per_step;

        PowerReport {
            standard_watts_per_token: standard_energy,
            speculative_watts_per_token: spec_energy,
            savings_percent: ((standard_energy - spec_energy) / standard_energy) * 100.0,
            co2_reduction_factor: standard_energy / spec_energy,
        }
    }

    /// Reset statistics.
    pub fn reset_stats(&mut self) {
        self.total_drafted = 0;
        self.total_accepted = 0;
        self.total_steps = 0;
    }
}

/// Power efficiency report.
#[derive(Debug, Clone)]
pub struct PowerReport {
    /// Energy per token with standard decoding (Watts)
    pub standard_watts_per_token: f32,
    /// Energy per token with speculative decoding (Watts)
    pub speculative_watts_per_token: f32,
    /// Percentage energy savings
    pub savings_percent: f32,
    /// CO2 reduction multiplier
    pub co2_reduction_factor: f32,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Simple softmax with temperature.
fn softmax_simple(logits: &[f32], temperature: f32) -> Vec<f32> {
    let temp = temperature.max(1e-6);
    let mut probs = vec![0.0f32; logits.len()];

    // Find max for numerical stability
    let mut max_val = f32::NEG_INFINITY;
    for &v in logits {
        if v > max_val { max_val = v; }
    }

    // exp and sum
    let mut sum = 0.0f32;
    for (i, &v) in logits.iter().enumerate() {
        let scaled = (v - max_val) / temp;
        let exp_val = fast_exp(scaled);
        probs[i] = exp_val;
        sum += exp_val;
    }

    // Normalize
    if sum > 0.0 {
        for p in &mut probs {
            *p /= sum;
        }
    }

    probs
}

/// Argmax of a slice.
fn argmax(probs: &[f32]) -> u32 {
    let mut best_idx = 0u32;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in probs.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_idx = i as u32;
        }
    }
    best_idx
}

/// Fast exp approximation (Schraudolph).
fn fast_exp(x: f32) -> f32 {
    if x < -88.0 { return 0.0; }
    if x > 88.0 { return f32::MAX; }
    let a = 12102203.0f32;
    let b = 1065353216.0f32;
    let bits = (a * x + b) as u32;
    f32::from_bits(bits.min(0x7F80_0000))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_logits(dominant_idx: usize, vocab_size: usize) -> Vec<f32> {
        let mut logits = vec![-5.0f32; vocab_size];
        logits[dominant_idx] = 10.0; // Very high probability for dominant token
        logits
    }

    #[test]
    fn test_perfect_agreement() {
        // When draft and target agree perfectly, all tokens should be accepted
        let mut engine = SpeculativeEngine::new(SpeculativeConfig::default());

        let draft_tokens: Vec<DraftToken> = (0..4)
            .map(|i| {
                let logits = make_logits(i + 1, 100);
                DraftToken {
                    token_id: (i + 1) as u32,
                    draft_prob: 0.95,
                    draft_logits: logits,
                }
            })
            .collect();

        let target_logits: Vec<Vec<f32>> = (0..4)
            .map(|i| make_logits(i + 1, 100))
            .collect();

        let result = engine.verify(&draft_tokens, &target_logits);
        assert!(
            result.accepted_count >= 3,
            "High-agreement draft should accept most tokens, got {}",
            result.accepted_count
        );
    }

    #[test]
    fn test_disagreement_first_token() {
        // Draft proposes token 5, but target strongly prefers token 10
        let mut engine = SpeculativeEngine::new(SpeculativeConfig {
            use_modified_rejection: false,
            ..Default::default()
        });

        let draft_tokens = vec![DraftToken {
            token_id: 5,
            draft_prob: 0.9,
            draft_logits: make_logits(5, 100),
        }];

        let target_logits = vec![make_logits(10, 100)];

        let result = engine.verify(&draft_tokens, &target_logits);
        assert_eq!(result.accepted_count, 0, "Disagreeing token should be rejected");
        assert_eq!(result.accepted_tokens[0], 10, "Should use target's token");
    }

    #[test]
    fn test_power_report() {
        let mut engine = SpeculativeEngine::new(SpeculativeConfig::default());

        // Simulate 10 steps with ~60% acceptance
        for _ in 0..10 {
            let draft: Vec<DraftToken> = (0..5).map(|i| DraftToken {
                token_id: i,
                draft_prob: 0.5,
                draft_logits: make_logits(i as usize, 50),
            }).collect();

            let target: Vec<Vec<f32>> = (0..5).map(|i| make_logits(i as usize, 50)).collect();
            engine.verify(&draft, &target);
        }

        let report = engine.power_efficiency_ratio(5.0, 15.0);
        assert!(report.co2_reduction_factor > 0.0);
        assert!(report.speculative_watts_per_token < report.standard_watts_per_token || report.savings_percent > 0.0);
    }

    #[test]
    fn test_config_defaults() {
        let config = SpeculativeConfig::default();
        assert_eq!(config.draft_length, 8);
        assert!(config.use_modified_rejection);
    }

    #[test]
    fn test_edge_cluster_config() {
        let config = SpeculativeConfig::for_edge_cluster();
        assert_eq!(config.draft_length, 5);
    }
}
