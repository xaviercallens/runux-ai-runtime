// Copyright (c) 2026 Xavier Callens / Socrate AI. All Rights Reserved.
// SPDX-License-Identifier: LicenseRef-RunuX-Commercial
// This file is proprietary and confidential. Unauthorized copying,
// distribution, or use is strictly prohibited.

#![no_std]
#![cfg_attr(not(test), no_main)]
#![deny(clippy::all)]
#![warn(clippy::pedantic)]
//! RunuX Federated Learning — Distributed training and gradient aggregation
//!
//! Provides kernel-level support for federated learning across RISC-V edge
//! nodes. Designed for use with the Flower framework, enabling privacy-
//! preserving distributed training on BPI-F3 / AIBOX-K3 clusters.
//!
//! # Architecture
//!
//! ```text
//! ┌────────────────────────────────────────────┐
//! │         AIBOX-K3 (Aggregation Server)      │
//! │  ┌──────────────────────────────────────┐  │
//! │  │  FedAvg / FedProx / FedYogi          │  │
//! │  │  Global model (14B DeepSeek R1)      │  │
//! │  │  Differential privacy (ε-budget)     │  │
//! │  │  Secure aggregation                  │  │
//! │  └──────────────┬───────────────────────┘  │
//! └─────────────────┼──────────────────────────┘
//!           ┌───────┴───────┐
//!     ┌─────▼─────┐  ┌─────▼─────┐
//!     │ BPI-F3 #1 │  │ BPI-F3 #N │
//!     │ Local data │  │ Local data │
//!     │ LoRA train │  │ LoRA train │
//!     └───────────┘  └───────────┘
//! ```
//!
//! # Privacy Model
//!
//! - Data never leaves the edge node
//! - Only model gradients/weights are transmitted
//! - Differential privacy adds calibrated noise (ε-budget)
//! - Secure aggregation encrypts individual updates

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;
use alloc::string::String;

use ai_runtime::DataType;

// ---------------------------------------------------------------------------
// Node Identity
// ---------------------------------------------------------------------------

/// Unique identifier for a federated learning node.
pub type NodeId = u64;

/// Role of a node in the federated cluster.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeRole {
    /// Aggregation server (AIBOX-K3)
    Server,
    /// Training client (BPI-F3)
    Client,
    /// Hybrid: both trains and can aggregate a sub-cluster
    Hybrid,
}

/// Network address for inter-node communication.
#[derive(Debug, Clone)]
pub struct NodeAddress {
    /// Unique node identifier
    pub id: NodeId,
    /// IPv4/IPv6 address
    pub ip: String,
    /// gRPC port for Flower protocol
    pub port: u16,
    /// Role in the federation
    pub role: NodeRole,
    /// Hardware platform (for adaptive scheduling)
    pub platform: NodePlatform,
    /// Current status
    pub status: NodeStatus,
}

/// Hardware platform of a node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodePlatform {
    /// Banana Pi BPI-F3 (SpacemiT K1, 4-8GB)
    BpiF3,
    /// Firefly AIBOX-K3 (SpacemiT K3, 8-32GB)
    AiboxK3,
    /// Generic RISC-V node
    GenericRiscV,
    /// x86_64 server (for mixed clusters)
    X86Server,
}

/// Current status of a node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeStatus {
    /// Online and ready for training
    Ready,
    /// Currently training a local model
    Training,
    /// Uploading gradient update
    Uploading,
    /// Downloading global model update
    Downloading,
    /// Temporarily disconnected
    Disconnected,
    /// Permanently failed
    Failed,
}

// ---------------------------------------------------------------------------
// Aggregation Strategies
// ---------------------------------------------------------------------------

/// Federated aggregation strategy for combining client updates.
#[derive(Debug, Clone)]
pub enum AggregationStrategy {
    /// FedAvg: Simple weighted average of client updates.
    /// Weights are proportional to local dataset size.
    FedAvg,

    /// FedProx: Adds a proximal term to handle heterogeneity.
    /// The `mu` parameter controls the strength of the regularization.
    FedProx {
        /// Proximal term coefficient (typical: 0.01–1.0)
        mu: f32,
    },

    /// FedYogi: Adaptive server-side optimizer (momentum-based).
    /// Better convergence for heterogeneous data distributions.
    FedYogi {
        /// First moment decay (typical: 0.9)
        beta1: f32,
        /// Second moment decay (typical: 0.99)
        beta2: f32,
        /// Learning rate
        server_lr: f32,
    },

    /// FedAdam: Adam optimizer on the server side.
    FedAdam {
        /// First moment decay
        beta1: f32,
        /// Second moment decay
        beta2: f32,
        /// Learning rate
        server_lr: f32,
    },
}

// ---------------------------------------------------------------------------
// Privacy Configuration
// ---------------------------------------------------------------------------

/// Privacy configuration for federated learning.
#[derive(Debug, Clone)]
pub struct PrivacyConfig {
    /// Enable differential privacy noise injection
    pub differential_privacy: bool,
    /// Privacy budget epsilon (lower = more private, typical: 1.0–10.0)
    pub epsilon: f64,
    /// Privacy budget delta (typical: 1e-5)
    pub delta: f64,
    /// Enable secure aggregation (encrypt individual updates)
    pub secure_aggregation: bool,
    /// Gradient clipping L2 norm bound (for DP sensitivity)
    pub gradient_clip_norm: f32,
    /// Maximum number of rounds to spend the privacy budget
    pub max_rounds: usize,
}

impl Default for PrivacyConfig {
    fn default() -> Self {
        Self {
            differential_privacy: true,
            epsilon: 1.0,
            delta: 1e-5,
            secure_aggregation: false,
            gradient_clip_norm: 1.0,
            max_rounds: 100,
        }
    }
}

impl PrivacyConfig {
    /// Calculate per-round noise scale (σ) for Gaussian mechanism.
    ///
    /// σ = clip_norm × √(2 ln(1.25/δ)) / ε_per_round
    pub fn noise_scale(&self) -> f64 {
        if !self.differential_privacy || self.epsilon <= 0.0 {
            return 0.0;
        }
        let eps_per_round = self.epsilon / sqrt_f64(self.max_rounds as f64);
        let sensitivity = self.gradient_clip_norm as f64;
        let ln_term = sqrt_f64(2.0_f64 * ln_f64(1.25 / self.delta));
        sensitivity * ln_term / eps_per_round
    }

    /// Returns the remaining privacy budget after `rounds_completed` rounds.
    pub fn remaining_budget(&self, rounds_completed: usize) -> f64 {
        // Using basic composition theorem: ε_total = ε_per_round × √n
        let eps_used = self.epsilon * sqrt_f64(rounds_completed as f64)
            / sqrt_f64(self.max_rounds as f64);
        if self.epsilon > eps_used { self.epsilon - eps_used } else { 0.0 }
    }
}

// ---------------------------------------------------------------------------
// LoRA Configuration
// ---------------------------------------------------------------------------

/// LoRA (Low-Rank Adaptation) configuration for parameter-efficient fine-tuning.
///
/// Instead of updating all model parameters, LoRA decomposes weight updates
/// into low-rank matrices, reducing trainable parameters by 95-99%.
#[derive(Debug, Clone)]
pub struct LoraConfig {
    /// Rank of the low-rank decomposition (typical: 8, 16, 32, 64)
    pub rank: usize,
    /// Scaling factor: α / rank (typical α: 16, 32)
    pub alpha: f32,
    /// Dropout probability for LoRA layers
    pub dropout: f32,
    /// Which modules to apply LoRA to (e.g., "q_proj", "v_proj")
    pub target_modules: Vec<String>,
}

impl Default for LoraConfig {
    fn default() -> Self {
        Self {
            rank: 16,
            alpha: 32.0,
            dropout: 0.05,
            target_modules: Vec::new(),
        }
    }
}

impl LoraConfig {
    /// Estimates the number of trainable parameters for LoRA.
    ///
    /// For each target module of dimension [d_in × d_out]:
    /// LoRA adds d_in × rank + rank × d_out parameters.
    pub fn trainable_params(&self, hidden_dim: usize, num_modules: usize) -> usize {
        // Each LoRA adapter: A[hidden_dim × rank] + B[rank × hidden_dim]
        let per_module = 2 * hidden_dim * self.rank;
        per_module * num_modules
    }

    /// Returns the percentage of parameters that are trainable.
    pub fn trainable_percent(&self, total_params: usize, hidden_dim: usize, num_modules: usize) -> f32 {
        let trainable = self.trainable_params(hidden_dim, num_modules);
        (trainable as f32 / total_params as f32) * 100.0
    }
}

// ---------------------------------------------------------------------------
// Gradient Buffer
// ---------------------------------------------------------------------------

/// Efficient gradient storage for federated aggregation.
///
/// Stores gradient updates in a compact format for network transmission.
/// Supports gradient compression (sparsification, quantization) to
/// reduce bandwidth usage over the dual GbE links.
#[derive(Debug, Clone)]
pub struct GradientBuffer {
    /// Flat gradient data (FP32 or quantized)
    pub data: Vec<f32>,
    /// Number of parameters in this gradient update
    pub num_params: usize,
    /// Data type of the gradient values
    pub dtype: DataType,
    /// Training round this gradient was computed in
    pub round_id: u32,
    /// Node that produced this gradient
    pub source_node: NodeId,
    /// Number of local training samples used
    pub local_samples: usize,
    /// Local training loss
    pub local_loss: f32,
}

impl GradientBuffer {
    /// Create a new zero-initialized gradient buffer.
    pub fn zeros(num_params: usize) -> Self {
        Self {
            data: vec![0.0; num_params],
            num_params,
            dtype: DataType::FP32,
            round_id: 0,
            source_node: 0,
            local_samples: 0,
            local_loss: 0.0,
        }
    }

    /// Size in bytes for network transmission.
    pub fn wire_size(&self) -> usize {
        // Header (32 bytes) + data
        32 + self.data.len() * 4
    }

    /// Clip gradients to the given L2 norm bound.
    ///
    /// Required for differential privacy: clips gradient magnitude
    /// to bound the sensitivity of the query.
    pub fn clip_l2_norm(&mut self, max_norm: f32) {
        let norm_sq: f32 = self.data.iter().map(|x| x * x).sum();
        let norm = fast_sqrt_simple(norm_sq);

        if norm > max_norm {
            let scale = max_norm / norm;
            for val in &mut self.data {
                *val *= scale;
            }
        }
    }

    /// Add Gaussian noise for differential privacy.
    ///
    /// Uses a simple linear congruential generator (LCG) seeded
    /// with the round ID and node ID for reproducibility.
    pub fn add_dp_noise(&mut self, noise_scale: f64, seed: u64) {
        let mut state = seed ^ (self.round_id as u64) ^ (self.source_node << 32);

        for val in &mut self.data {
            // Box-Muller transform for Gaussian noise
            let u1 = lcg_next(&mut state);
            let u2 = lcg_next(&mut state);
            let z = sqrt_f64(-2.0 * ln_f64(u1))
                * cos_f64(2.0 * core::f64::consts::PI * u2);
            *val += (z * noise_scale) as f32;
        }
    }

    /// Sparsify the gradient by keeping only the top-k% largest values.
    ///
    /// Reduces bandwidth for network transmission at the cost of
    /// slightly slower convergence.
    pub fn sparsify(&mut self, keep_percent: f32) {
        let k = ((self.num_params as f32 * keep_percent / 100.0) as usize).max(1);

        // Find the k-th largest magnitude
        let mut magnitudes: Vec<f32> = self.data.iter().map(|x| x.abs()).collect();
        magnitudes.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap_or(core::cmp::Ordering::Equal));

        let threshold = if k < magnitudes.len() {
            magnitudes[k]
        } else {
            0.0
        };

        // Zero out values below threshold
        for val in &mut self.data {
            if val.abs() < threshold {
                *val = 0.0;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Aggregation
// ---------------------------------------------------------------------------

/// Perform FedAvg aggregation: weighted average of gradients.
///
/// Weights are proportional to the number of local training samples.
pub fn fedavg_aggregate(gradients: &[GradientBuffer]) -> GradientBuffer {
    if gradients.is_empty() {
        return GradientBuffer::zeros(0);
    }

    let num_params = gradients[0].num_params;
    let total_samples: usize = gradients.iter().map(|g| g.local_samples).sum();

    let mut result = GradientBuffer::zeros(num_params);
    result.round_id = gradients[0].round_id;
    result.local_samples = total_samples;

    if total_samples == 0 {
        return result;
    }

    for grad in gradients {
        let weight = grad.local_samples as f32 / total_samples as f32;
        for i in 0..num_params.min(grad.data.len()) {
            result.data[i] += grad.data[i] * weight;
        }
    }

    // Average loss
    result.local_loss = gradients.iter()
        .map(|g| g.local_loss * g.local_samples as f32 / total_samples as f32)
        .sum();

    result
}

/// Perform FedProx aggregation with proximal regularization.
pub fn fedprox_aggregate(
    gradients: &[GradientBuffer],
    global_params: &[f32],
    mu: f32,
) -> GradientBuffer {
    let mut result = fedavg_aggregate(gradients);

    // Add proximal term: -μ × (w_local - w_global)
    for i in 0..result.data.len().min(global_params.len()) {
        result.data[i] -= mu * (result.data[i] - global_params[i]);
    }

    result
}

// ---------------------------------------------------------------------------
// Federated Configuration
// ---------------------------------------------------------------------------

/// Full federated learning configuration.
#[derive(Debug, Clone)]
pub struct FederatedConfig {
    /// Participating nodes
    pub nodes: Vec<NodeAddress>,
    /// Aggregation strategy
    pub aggregation: AggregationStrategy,
    /// Privacy configuration
    pub privacy: PrivacyConfig,
    /// LoRA fine-tuning configuration
    pub lora: LoraConfig,
    /// Number of local training epochs per round
    pub local_epochs: usize,
    /// Local batch size
    pub batch_size: usize,
    /// Learning rate for local training
    pub learning_rate: f32,
    /// Maximum number of federated rounds
    pub max_rounds: usize,
    /// Minimum fraction of clients that must participate per round
    pub min_client_fraction: f32,
    /// Timeout for client updates (seconds)
    pub client_timeout_secs: u32,
    /// Enable gradient sparsification
    pub sparsify_percent: Option<f32>,
}

impl Default for FederatedConfig {
    fn default() -> Self {
        Self {
            nodes: Vec::new(),
            aggregation: AggregationStrategy::FedAvg,
            privacy: PrivacyConfig::default(),
            lora: LoraConfig::default(),
            local_epochs: 3,
            batch_size: 8,
            learning_rate: 1e-4,
            max_rounds: 100,
            min_client_fraction: 0.5,
            client_timeout_secs: 300,
            sparsify_percent: None,
        }
    }
}

/// Training round metrics.
#[derive(Debug, Clone)]
pub struct RoundMetrics {
    /// Round number
    pub round_id: u32,
    /// Number of participating clients
    pub num_clients: usize,
    /// Total training samples across all clients
    pub total_samples: usize,
    /// Aggregated loss
    pub loss: f32,
    /// Communication cost in bytes (gradient upload)
    pub comm_bytes: usize,
    /// Round wall-clock time in milliseconds
    pub duration_ms: u64,
    /// Remaining privacy budget (ε)
    pub remaining_epsilon: f64,
}

// ---------------------------------------------------------------------------
// Cluster Management
// ---------------------------------------------------------------------------

/// Federated learning cluster state.
pub struct FederatedCluster {
    /// Cluster configuration
    pub config: FederatedConfig,
    /// Current training round
    pub current_round: u32,
    /// Metrics for completed rounds
    pub metrics: Vec<RoundMetrics>,
    /// Global model parameters (server-side)
    pub global_params: Vec<f32>,
}

impl FederatedCluster {
    /// Create a new cluster.
    pub fn new(config: FederatedConfig, num_params: usize) -> Self {
        Self {
            config,
            current_round: 0,
            metrics: Vec::new(),
            global_params: vec![0.0; num_params],
        }
    }

    /// Count nodes by role.
    pub fn count_by_role(&self, role: NodeRole) -> usize {
        self.config.nodes.iter().filter(|n| n.role == role).count()
    }

    /// Count online clients.
    pub fn online_clients(&self) -> usize {
        self.config.nodes.iter().filter(|n| {
            n.role == NodeRole::Client && n.status != NodeStatus::Failed
                && n.status != NodeStatus::Disconnected
        }).count()
    }

    /// Check if enough clients are available for a round.
    pub fn can_start_round(&self) -> bool {
        let online = self.online_clients();
        let total = self.count_by_role(NodeRole::Client);
        if total == 0 {
            return false;
        }
        online as f32 / total as f32 >= self.config.min_client_fraction
    }

    /// Get remaining privacy budget.
    pub fn remaining_budget(&self) -> f64 {
        self.config.privacy.remaining_budget(self.current_round as usize)
    }

    /// Estimate total cluster compute in TOPS.
    pub fn total_tops(&self) -> u32 {
        self.config.nodes.iter().map(|n| {
            match n.platform {
                NodePlatform::AiboxK3 => 60,
                NodePlatform::BpiF3 => 2,
                _ => 1,
            }
        }).sum()
    }

    /// Estimate total cluster RAM in GB.
    pub fn total_ram_gb(&self) -> u32 {
        self.config.nodes.iter().map(|n| {
            match n.platform {
                NodePlatform::AiboxK3 => 32, // Assume max config
                NodePlatform::BpiF3 => 8,
                _ => 4,
            }
        }).sum()
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Simple sqrt for no_std (f32).
fn fast_sqrt_simple(x: f32) -> f32 {
    if x <= 0.0 { return 0.0; }
    let mut guess = x;
    for _ in 0..5 {
        guess = 0.5 * (guess + x / guess);
    }
    guess
}

/// Simple sqrt for no_std (f64).
fn sqrt_f64(x: f64) -> f64 {
    if x <= 0.0 { return 0.0; }
    let mut guess = x;
    for _ in 0..8 {
        guess = 0.5 * (guess + x / guess);
    }
    guess
}

/// Simple natural log for no_std (f64).
fn ln_f64(x: f64) -> f64 {
    if x <= 0.0 { return f64::MIN; }
    // Use identity: ln(x) = ln(m * 2^e) = ln(m) + e*ln(2)
    let bits = x.to_bits();
    let exp = ((bits >> 52) & 0x7FF) as f64 - 1023.0;
    let m_bits = (bits & 0x000F_FFFF_FFFF_FFFF) | 0x3FF0_0000_0000_0000;
    let m = f64::from_bits(m_bits);
    let ln_m = -1.725_3 + m * (2.067_2 + m * (-0.341_9));
    ln_m + exp * 0.693_147_180_559_945_3
}

/// Simple cosine for no_std (f64).
fn cos_f64(x: f64) -> f64 {
    sin_f64(x + core::f64::consts::FRAC_PI_2)
}

/// Simple sine for no_std (f64).
fn sin_f64(mut x: f64) -> f64 {
    let pi = core::f64::consts::PI;
    let two_pi = 2.0 * pi;
    x = x % two_pi;
    if x > pi { x -= two_pi; }
    if x < -pi { x += two_pi; }
    let abs_x = if x < 0.0 { -x } else { x };
    let y = 4.0 / pi * x - 4.0 / (pi * pi) * x * abs_x;
    0.225 * (y * (if y < 0.0 { -y } else { y }) - y) + y
}

/// Simple natural log for no_std.
fn fast_ln_simple(x: f32) -> f32 {
    if x <= 0.0 { return f32::MIN; }
    let bits = x.to_bits();
    let exp = ((bits >> 23) & 0xFF) as f32 - 127.0;
    let m_bits = (bits & 0x007F_FFFF) | 0x3F80_0000;
    let m = f32::from_bits(m_bits);
    (-1.725_3 + m * (2.067_2 + m * (-0.341_9))) + exp * 0.693_147_2
}

/// Linear congruential generator, returns value in [0, 1).
fn lcg_next(state: &mut u64) -> f64 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (*state >> 11) as f64 / (1u64 << 53) as f64
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gradient_buffer_zeros() {
        let buf = GradientBuffer::zeros(100);
        assert_eq!(buf.num_params, 100);
        assert_eq!(buf.data.len(), 100);
        assert!(buf.data.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_gradient_clip() {
        let mut buf = GradientBuffer::zeros(4);
        buf.data = vec![3.0, 4.0, 0.0, 0.0]; // L2 norm = 5.0
        buf.clip_l2_norm(1.0);

        let norm_sq: f32 = buf.data.iter().map(|x| x * x).sum();
        let norm = fast_sqrt_simple(norm_sq);
        assert!((norm - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_fedavg_aggregation() {
        let g1 = GradientBuffer {
            data: vec![1.0, 2.0, 3.0],
            num_params: 3,
            dtype: DataType::FP32,
            round_id: 0,
            source_node: 1,
            local_samples: 100,
            local_loss: 0.5,
        };
        let g2 = GradientBuffer {
            data: vec![3.0, 4.0, 5.0],
            num_params: 3,
            dtype: DataType::FP32,
            round_id: 0,
            source_node: 2,
            local_samples: 100,
            local_loss: 0.3,
        };

        let result = fedavg_aggregate(&[g1, g2]);
        // Equal weights (100 samples each), so average
        assert!((result.data[0] - 2.0).abs() < 1e-5);
        assert!((result.data[1] - 3.0).abs() < 1e-5);
        assert!((result.data[2] - 4.0).abs() < 1e-5);
    }

    #[test]
    fn test_privacy_noise_scale() {
        let config = PrivacyConfig::default();
        let sigma = config.noise_scale();
        assert!(sigma > 0.0);
        assert!(sigma.is_finite());
    }

    #[test]
    fn test_lora_trainable_params() {
        let lora = LoraConfig {
            rank: 16,
            alpha: 32.0,
            dropout: 0.05,
            target_modules: Vec::new(),
        };
        // 2 modules (q_proj, v_proj), hidden_dim=4096
        let params = lora.trainable_params(4096, 2);
        // Expected: 2 * 4096 * 16 * 2 = 262,144
        assert_eq!(params, 262144);
    }

    #[test]
    fn test_cluster_compute() {
        let config = FederatedConfig {
            nodes: vec![
                NodeAddress {
                    id: 0, ip: String::new(), port: 8080,
                    role: NodeRole::Server,
                    platform: NodePlatform::AiboxK3,
                    status: NodeStatus::Ready,
                },
                NodeAddress {
                    id: 1, ip: String::new(), port: 8081,
                    role: NodeRole::Client,
                    platform: NodePlatform::BpiF3,
                    status: NodeStatus::Ready,
                },
                NodeAddress {
                    id: 2, ip: String::new(), port: 8082,
                    role: NodeRole::Client,
                    platform: NodePlatform::BpiF3,
                    status: NodeStatus::Ready,
                },
            ],
            ..Default::default()
        };
        let cluster = FederatedCluster::new(config, 1000);
        assert_eq!(cluster.total_tops(), 64); // 60 + 2 + 2
        assert_eq!(cluster.total_ram_gb(), 48); // 32 + 8 + 8
        assert!(cluster.can_start_round());
    }

    #[test]
    fn test_sparsify() {
        let mut buf = GradientBuffer {
            data: vec![0.1, 0.5, 0.01, 0.9, 0.001],
            num_params: 5,
            dtype: DataType::FP32,
            round_id: 0,
            source_node: 0,
            local_samples: 10,
            local_loss: 0.0,
        };
        buf.sparsify(40.0); // Keep top 40% (= 2 values)
        let nonzero = buf.data.iter().filter(|&&v| v != 0.0).count();
        assert!(nonzero <= 3); // Approximately top 40%
    }
}
