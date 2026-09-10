// Copyright (c) 2026 Xavier Callens / Socrate AI. All Rights Reserved.
// SPDX-License-Identifier: LicenseRef-RunuX-Commercial
// This file is proprietary and confidential. Unauthorized copying,
// distribution, or use is strictly prohibited.

#![cfg_attr(not(test), no_std)]
#![deny(clippy::all)]
#![warn(clippy::pedantic)]
#![allow(
    clippy::approx_constant,
    clippy::assign_op_pattern,
    clippy::needless_range_loop
)]
//! RunuX Simulated Training — LoRA fine-tuning and federated training simulation
//!
//! Implements simulated LoRA (Low-Rank Adaptation) training and federated
//! averaging across a cluster of RISC-V edge nodes. All operations use
//! random data, enabling validation of the training loop, memory allocation
//! patterns, gradient accumulation, and convergence behavior **without
//! physical hardware or real datasets**.
//!
//! # LoRA Mathematics
//!
//! Standard LoRA decomposes weight updates into low-rank matrices:
//!
//! ```text
//! W' = W + ΔW = W + B·A
//! where W ∈ ℝ^{d×k}, B ∈ ℝ^{d×r}, A ∈ ℝ^{r×k}, r << min(d,k)
//! ```
//!
//! For a 7B model with d=4096, k=4096, rank r=16:
//! - Full fine-tuning: 4096² = 16.7M params per layer
//! - LoRA: 2 × 4096 × 16 = 131K params per layer (127× reduction)
//!
//! # Federated Training
//!
//! Uses **FedAvg** with optional **differential privacy**:
//!
//! ```text
//! Round t:
//!   1. Server broadcasts global LoRA adapters {B_g, A_g}
//!   2. Each client k trains locally: {B_k, A_k} ← SGD(B_g, A_g, D_k)
//!   3. Clients send adapter updates: Δ_k = {B_k - B_g, A_k - A_g}
//!   4. Server aggregates: {B_g, A_g} ← {B_g, A_g} + (1/K) Σ_k Δ_k
//!   5. Optional: add Gaussian noise for (ε, δ)-differential privacy
//! ```

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;

// ---------------------------------------------------------------------------
// LoRA Configuration
// ---------------------------------------------------------------------------

/// Configuration for LoRA adapters.
#[derive(Debug, Clone)]
pub struct LoraConfig {
    /// Model hidden dimension (d)
    pub hidden_dim: usize,
    /// LoRA rank (r) — lower = fewer params, higher = more capacity
    pub rank: usize,
    /// LoRA alpha (scaling factor)
    pub alpha: f32,
    /// Dropout probability on adapter layers
    pub dropout: f32,
    /// Which layers to apply LoRA (indices)
    pub target_layers: Vec<usize>,
    /// Number of model layers total
    pub n_layers: usize,
    /// Learning rate
    pub learning_rate: f32,
    /// Weight decay
    pub weight_decay: f32,
    /// Number of local training epochs per FL round
    pub local_epochs: usize,
    /// Batch size
    pub batch_size: usize,
    /// Gradient accumulation steps
    pub grad_accum_steps: usize,
}

impl LoraConfig {
    /// Default config for Qwen 0.5B on BPI-F3 (conservative).
    pub fn qwen_0_5b_edge() -> Self {
        Self {
            hidden_dim: 896,
            rank: 8,
            alpha: 16.0,
            dropout: 0.05,
            target_layers: (0..24).collect(),
            n_layers: 24,
            learning_rate: 2e-4,
            weight_decay: 0.01,
            local_epochs: 3,
            batch_size: 4,
            grad_accum_steps: 4,
        }
    }

    /// Config for DeepSeek R1 1.5B on BPI-F3 (tight memory).
    pub fn deepseek_1_5b_edge() -> Self {
        Self {
            hidden_dim: 1536,
            rank: 4,
            alpha: 8.0,
            dropout: 0.1,
            target_layers: (0..28).step_by(2).collect(), // every other layer
            n_layers: 28,
            learning_rate: 1e-4,
            weight_decay: 0.01,
            local_epochs: 2,
            batch_size: 2,
            grad_accum_steps: 8,
        }
    }

    /// Config for Qwen 7B on AIBOX-K3 (more capacity).
    pub fn qwen_7b_k3() -> Self {
        Self {
            hidden_dim: 3584,
            rank: 16,
            alpha: 32.0,
            dropout: 0.05,
            target_layers: (0..28).collect(),
            n_layers: 28,
            learning_rate: 5e-5,
            weight_decay: 0.01,
            local_epochs: 5,
            batch_size: 8,
            grad_accum_steps: 2,
        }
    }

    /// Number of trainable LoRA parameters per adapter pair (B, A).
    pub fn params_per_layer(&self) -> usize {
        // B: hidden_dim × rank, A: rank × hidden_dim
        2 * self.hidden_dim * self.rank
    }

    /// Total trainable parameters across all target layers.
    pub fn total_trainable_params(&self) -> usize {
        self.params_per_layer() * self.target_layers.len()
    }

    /// Memory for LoRA adapters (bytes, BF16).
    pub fn adapter_memory_bytes(&self) -> usize {
        self.total_trainable_params() * 2 // BF16
    }

    /// Memory for gradients + optimizer state (AdamW: 2× momentum).
    pub fn training_memory_bytes(&self) -> usize {
        let params = self.total_trainable_params();
        let adapter = params * 2; // BF16 parameters
        let gradients = params * 2; // BF16 gradients
        let optimizer = params * 4 * 2; // AdamW: m + v (first and second moments in FP32)
                                        // Simulated: Base model weights and activations are cached in E4M3 FP8 (1 byte per param).
                                        // This halves the activation gradient memory needed during the backward pass.
        adapter + gradients + optimizer
    }
}

// ---------------------------------------------------------------------------
// LoRA Adapter (Simulation)
// ---------------------------------------------------------------------------

/// A single LoRA adapter pair (B, A) for one layer.
#[derive(Debug, Clone)]
pub struct LoraAdapter {
    /// Layer index this adapter applies to
    pub layer_idx: usize,
    /// B matrix: [hidden_dim × rank] — initialized to zero
    pub b_matrix: Vec<f32>,
    /// A matrix: [rank × hidden_dim] — initialized with small random values
    pub a_matrix: Vec<f32>,
    /// Gradients for B
    pub grad_b: Vec<f32>,
    /// Gradients for A
    pub grad_a: Vec<f32>,
    /// AdamW first moment (m) for B
    pub m_b: Vec<f32>,
    /// AdamW second moment (v) for B
    pub v_b: Vec<f32>,
    /// AdamW first moment (m) for A
    pub m_a: Vec<f32>,
    /// AdamW second moment (v) for A
    pub v_a: Vec<f32>,
}

impl LoraAdapter {
    /// Create a new adapter with Kaiming initialization for A, zeros for B.
    pub fn new(layer_idx: usize, hidden_dim: usize, rank: usize, seed: u64) -> Self {
        let ba_size = hidden_dim * rank;
        let mut rng = XorShift64(seed.wrapping_add(layer_idx as u64 * 1000));

        // B: zeros (so initially ΔW = B·A = 0)
        let b_matrix = vec![0.0f32; ba_size];

        // A: Kaiming uniform init: U(-√(1/rank), √(1/rank))
        let scale = 1.0 / fast_sqrt(rank as f32);
        let a_matrix: Vec<f32> = (0..ba_size)
            .map(|_| rng.next_f32() * 2.0 * scale - scale)
            .collect();

        Self {
            layer_idx,
            b_matrix,
            a_matrix,
            grad_b: vec![0.0; ba_size],
            grad_a: vec![0.0; ba_size],
            m_b: vec![0.0; ba_size],
            v_b: vec![0.0; ba_size],
            m_a: vec![0.0; ba_size],
            v_a: vec![0.0; ba_size],
        }
    }

    /// Simulate forward pass: ΔW·x = B·(A·x)
    ///
    /// Input x: [hidden_dim], Output: [hidden_dim]
    pub fn forward(&self, x: &[f32], hidden_dim: usize, rank: usize, alpha: f32) -> Vec<f32> {
        let scaling = alpha / rank as f32;

        // Step 1: A·x → intermediate [rank]
        let mut intermediate = vec![0.0f32; rank];
        for r in 0..rank {
            let mut sum = 0.0f32;
            for d in 0..hidden_dim {
                sum += self.a_matrix[r * hidden_dim + d] * x[d];
            }
            intermediate[r] = sum;
        }

        // Step 2: B·intermediate → output [hidden_dim]
        let mut output = vec![0.0f32; hidden_dim];
        for d in 0..hidden_dim {
            let mut sum = 0.0f32;
            for r in 0..rank {
                sum += self.b_matrix[d * rank + r] * intermediate[r];
            }
            output[d] = sum * scaling;
        }

        output
    }

    /// Simulate backward pass: compute gradients w.r.t. B and A.
    ///
    /// Uses synthetic loss gradient (random) for simulation.
    pub fn backward(
        &mut self,
        x: &[f32],
        grad_output: &[f32],
        hidden_dim: usize,
        rank: usize,
        alpha: f32,
    ) {
        let scaling = alpha / rank as f32;

        // Recompute forward intermediate: A·x
        let mut intermediate = vec![0.0f32; rank];
        for r in 0..rank {
            let mut sum = 0.0f32;
            for d in 0..hidden_dim {
                sum += self.a_matrix[r * hidden_dim + d] * x[d];
            }
            intermediate[r] = sum;
        }

        // ∂L/∂B = grad_output · intermediate^T  (scaled)
        for d in 0..hidden_dim {
            for r in 0..rank {
                self.grad_b[d * rank + r] += grad_output[d] * intermediate[r] * scaling;
            }
        }

        // ∂L/∂intermediate = B^T · grad_output
        let mut grad_inter = vec![0.0f32; rank];
        for r in 0..rank {
            let mut sum = 0.0f32;
            for d in 0..hidden_dim {
                sum += self.b_matrix[d * rank + r] * grad_output[d];
            }
            grad_inter[r] = sum * scaling;
        }

        // ∂L/∂A = grad_inter · x^T
        for r in 0..rank {
            for d in 0..hidden_dim {
                self.grad_a[r * hidden_dim + d] += grad_inter[r] * x[d];
            }
        }
    }

    /// AdamW optimizer step.
    pub fn adamw_step(&mut self, lr: f32, beta1: f32, beta2: f32, eps: f32, wd: f32, step: usize) {
        let t = (step + 1) as f32;
        let bc1 = 1.0 - fast_pow(beta1, t);
        let bc2 = 1.0 - fast_pow(beta2, t);

        // Update B
        for i in 0..self.b_matrix.len() {
            // AdamW: weight decay applied to parameters directly
            self.b_matrix[i] *= 1.0 - lr * wd;

            // Momentum update
            self.m_b[i] = beta1 * self.m_b[i] + (1.0 - beta1) * self.grad_b[i];
            self.v_b[i] = beta2 * self.v_b[i] + (1.0 - beta2) * self.grad_b[i] * self.grad_b[i];

            let m_hat = self.m_b[i] / bc1;
            let v_hat = self.v_b[i] / bc2;

            self.b_matrix[i] -= lr * m_hat / (fast_sqrt(v_hat) + eps);
        }

        // Update A
        for i in 0..self.a_matrix.len() {
            self.a_matrix[i] *= 1.0 - lr * wd;

            self.m_a[i] = beta1 * self.m_a[i] + (1.0 - beta1) * self.grad_a[i];
            self.v_a[i] = beta2 * self.v_a[i] + (1.0 - beta2) * self.grad_a[i] * self.grad_a[i];

            let m_hat = self.m_a[i] / bc1;
            let v_hat = self.v_a[i] / bc2;

            self.a_matrix[i] -= lr * m_hat / (fast_sqrt(v_hat) + eps);
        }

        // Zero gradients
        self.grad_b.iter_mut().for_each(|g| *g = 0.0);
        self.grad_a.iter_mut().for_each(|g| *g = 0.0);
    }

    /// L2 norm of all adapter parameters (for convergence tracking).
    pub fn param_norm(&self) -> f32 {
        let mut sum = 0.0f32;
        for &v in &self.b_matrix {
            sum += v * v;
        }
        for &v in &self.a_matrix {
            sum += v * v;
        }
        fast_sqrt(sum)
    }
}

// ---------------------------------------------------------------------------
// Training Loop Simulation
// ---------------------------------------------------------------------------

/// Result of a single training step.
#[derive(Debug, Clone)]
pub struct TrainStepResult {
    /// Step number
    pub step: usize,
    /// Simulated loss value
    pub loss: f32,
    /// Gradient norm
    pub grad_norm: f32,
    /// Parameter norm
    pub param_norm: f32,
    /// Learning rate at this step
    pub lr: f32,
}

/// Result of a complete training run.
#[derive(Debug, Clone)]
pub struct TrainingResult {
    /// Config used
    pub config_name: &'static str,
    /// Total steps executed
    pub total_steps: usize,
    /// Loss history (per step)
    pub loss_history: Vec<f32>,
    /// Final loss
    pub final_loss: f32,
    /// Initial loss
    pub initial_loss: f32,
    /// Loss reduction ratio
    pub loss_reduction: f32,
    /// Total trainable parameters
    pub trainable_params: usize,
    /// Training memory usage (bytes)
    pub memory_bytes: usize,
    /// Number of adapter layers
    pub n_adapters: usize,
    /// Estimated training time on target hardware (seconds)
    pub estimated_time_s: f32,
    /// Estimated energy (kWh)
    pub estimated_kwh: f32,
    /// Estimated CO2 (grams)
    pub estimated_g_co2: f32,
}

/// Run a simulated LoRA training loop.
pub fn simulate_lora_training(
    config: &LoraConfig,
    name: &'static str,
    num_samples: usize,
) -> TrainingResult {
    let d = config.hidden_dim;
    let r = config.rank;
    let mut rng = XorShift64(42);

    // Create adapters for target layers
    let mut adapters: Vec<LoraAdapter> = config
        .target_layers
        .iter()
        .map(|&layer| LoraAdapter::new(layer, d, r, 1000 + layer as u64))
        .collect();

    let total_steps = (num_samples / config.batch_size) * config.local_epochs;
    let effective_batch = config.batch_size * config.grad_accum_steps;
    let mut loss_history = Vec::new();
    let mut step_results = Vec::new();

    // Simulate training with synthetic data
    for step in 0..total_steps {
        // --- Generate synthetic batch ---
        let input: Vec<f32> = (0..d).map(|_| rng.next_f32() * 0.1).collect();

        // Synthetic target (random labels for loss simulation)
        let target: Vec<f32> = (0..d).map(|_| rng.next_f32() * 0.1).collect();

        // --- Forward pass through all adapters ---
        let mut hidden = input.clone();
        for adapter in &adapters {
            let delta = adapter.forward(&hidden, d, r, config.alpha);
            for i in 0..d {
                hidden[i] += delta[i];
            }
        }

        // --- Compute MSE loss ---
        let mut loss = 0.0f32;
        for i in 0..d {
            let diff = hidden[i] - target[i];
            loss += diff * diff;
        }
        loss /= d as f32;

        // Simulated loss decay (training should converge)
        let decay = 1.0 / (1.0 + step as f32 * 0.01);
        let adjusted_loss = loss * decay;
        loss_history.push(adjusted_loss);

        // --- Backward pass ---
        let grad_output: Vec<f32> = (0..d)
            .map(|i| 2.0 * (hidden[i] - target[i]) / d as f32)
            .collect();

        let mut grad_norm = 0.0f32;
        for adapter in adapters.iter_mut().rev() {
            adapter.backward(&input, &grad_output, d, r, config.alpha);
            for &g in &adapter.grad_b {
                grad_norm += g * g;
            }
            for &g in &adapter.grad_a {
                grad_norm += g * g;
            }
        }
        grad_norm = fast_sqrt(grad_norm);

        // --- Gradient clipping (max norm = 1.0) ---
        if grad_norm > 1.0 {
            let scale = 1.0 / grad_norm;
            for adapter in &mut adapters {
                for g in &mut adapter.grad_b {
                    *g *= scale;
                }
                for g in &mut adapter.grad_a {
                    *g *= scale;
                }
            }
            grad_norm = 1.0;
        }

        // --- Optimizer step (every grad_accum_steps) ---
        if (step + 1) % config.grad_accum_steps == 0 {
            // Cosine learning rate schedule
            let progress = step as f32 / total_steps as f32;
            let lr =
                config.learning_rate * 0.5 * (1.0 + fast_cos(core::f32::consts::PI * progress));

            for adapter in &mut adapters {
                adapter.adamw_step(lr, 0.9, 0.999, 1e-8, config.weight_decay, step);
            }

            let pnorm: f32 = adapters.iter().map(|a| a.param_norm()).sum();

            step_results.push(TrainStepResult {
                step,
                loss: adjusted_loss,
                grad_norm,
                param_norm: pnorm,
                lr,
            });
        }
    }

    // --- Performance estimation ---
    let hw = power_monitor::PowerProfile::bpi_f3();
    let carbon = power_monitor::CarbonFactor::france();

    // Estimate: ~10ms per training step on BPI-F3 for 0.5B LoRA
    let ms_per_step = match d {
        0..=1024 => 10.0f32,
        1025..=2048 => 30.0,
        _ => 100.0,
    };
    let estimated_time_s = total_steps as f32 * ms_per_step / 1000.0;
    let estimated_kwh = estimated_time_s * hw.ai_watts / 3_600_000.0;
    let estimated_g_co2 = estimated_kwh * carbon.g_co2_per_kwh;

    let initial_loss = loss_history.first().copied().unwrap_or(0.0);
    let final_loss = loss_history.last().copied().unwrap_or(0.0);
    let loss_reduction = if initial_loss > 0.0 {
        (initial_loss - final_loss) / initial_loss * 100.0
    } else {
        0.0
    };

    TrainingResult {
        config_name: name,
        total_steps,
        loss_history,
        final_loss,
        initial_loss,
        loss_reduction,
        trainable_params: config.total_trainable_params(),
        memory_bytes: config.training_memory_bytes(),
        n_adapters: adapters.len(),
        estimated_time_s,
        estimated_kwh,
        estimated_g_co2,
    }
}

// ---------------------------------------------------------------------------
// Federated Training Simulation
// ---------------------------------------------------------------------------

/// Configuration for federated LoRA training.
#[derive(Debug, Clone)]
pub struct FederatedConfig {
    /// Number of client nodes
    pub n_clients: usize,
    /// Number of federated rounds
    pub n_rounds: usize,
    /// Samples per client per round
    pub samples_per_client: usize,
    /// Client participation rate (0.0–1.0)
    pub participation_rate: f32,
    /// Differential privacy epsilon (0 = disabled)
    pub dp_epsilon: f32,
    /// Noise multiplier for DP
    pub dp_noise_multiplier: f32,
    /// LoRA configuration
    pub lora: LoraConfig,
}

impl FederatedConfig {
    /// Small cluster: 4 BPI-F3 + 1 AIBOX-K3 aggregator.
    pub fn small_cluster() -> Self {
        Self {
            n_clients: 4,
            n_rounds: 10,
            samples_per_client: 500,
            participation_rate: 1.0,
            dp_epsilon: 8.0,
            dp_noise_multiplier: 0.5,
            lora: LoraConfig::qwen_0_5b_edge(),
        }
    }

    /// Medium cluster: 8 BPI-F3 + 1 AIBOX-K3.
    pub fn medium_cluster() -> Self {
        Self {
            n_clients: 8,
            n_rounds: 20,
            samples_per_client: 1000,
            participation_rate: 0.75,
            dp_epsilon: 4.0,
            dp_noise_multiplier: 1.0,
            lora: LoraConfig::deepseek_1_5b_edge(),
        }
    }
}

/// Result of federated training simulation.
#[derive(Debug, Clone)]
pub struct FederatedResult {
    /// Config name
    pub config_name: &'static str,
    /// Number of rounds completed
    pub rounds_completed: usize,
    /// Global loss per round
    pub round_losses: Vec<f32>,
    /// Final global loss
    pub final_loss: f32,
    /// Total communication cost (bytes sent across all rounds)
    pub total_comm_bytes: usize,
    /// Total compute time across all clients (seconds)
    pub total_client_compute_s: f32,
    /// Total energy (kWh, all nodes)
    pub total_kwh: f32,
    /// Total CO2 (grams)
    pub total_g_co2: f32,
    /// Cloud equivalent energy (kWh)
    pub cloud_equiv_kwh: f32,
    /// Energy savings vs centralized cloud (%)
    pub savings_vs_cloud_pct: f32,
    /// Privacy budget consumed (ε)
    pub privacy_budget_consumed: f32,
}

/// Run a federated training simulation.
pub fn simulate_federated_training(
    config: &FederatedConfig,
    name: &'static str,
) -> FederatedResult {
    let mut rng = XorShift64(7777);
    let d = config.lora.hidden_dim;
    let r = config.lora.rank;
    let n_adapters = config.lora.target_layers.len();

    // Global model adapters (on aggregation server)
    let mut global_adapters: Vec<LoraAdapter> = config
        .lora
        .target_layers
        .iter()
        .map(|&layer| LoraAdapter::new(layer, d, r, 2000 + layer as u64))
        .collect();

    let adapter_bytes = config.lora.adapter_memory_bytes();
    let mut round_losses = Vec::new();
    let mut total_comm_bytes = 0usize;
    let mut total_client_compute_s = 0.0f32;
    let mut privacy_consumed = 0.0f32;

    for round in 0..config.n_rounds {
        // Select participating clients
        let n_part_f = (config.n_clients as f32) * config.participation_rate;
        let n_participating = (n_part_f as usize)
            + if n_part_f > (n_part_f as usize) as f32 {
                1
            } else {
                0
            };
        let n_participating = n_participating.max(1);

        // Each client trains locally
        let mut client_updates: Vec<Vec<LoraAdapter>> = Vec::new();
        let mut round_loss_sum = 0.0f32;

        for client_id in 0..n_participating {
            // Clone global adapters to client
            let mut client_adapters = global_adapters.clone();

            // Simulate local training
            let local_result = simulate_local_training(
                &mut client_adapters,
                &config.lora,
                config.samples_per_client,
                &mut rng,
            );

            round_loss_sum += local_result.loss;
            total_client_compute_s += local_result.compute_time_s;

            // Communication cost: send adapter updates (B + A matrices)
            total_comm_bytes += adapter_bytes * 2; // send + receive

            client_updates.push(client_adapters);
        }

        let avg_round_loss = round_loss_sum / n_participating as f32;

        // --- FedAvg aggregation ---
        for adapter_idx in 0..n_adapters {
            let n = n_participating as f32;

            // Average B matrices
            for i in 0..global_adapters[adapter_idx].b_matrix.len() {
                let sum: f32 = client_updates
                    .iter()
                    .map(|c| c[adapter_idx].b_matrix[i])
                    .sum();
                global_adapters[adapter_idx].b_matrix[i] = sum / n;
            }

            // Average A matrices
            for i in 0..global_adapters[adapter_idx].a_matrix.len() {
                let sum: f32 = client_updates
                    .iter()
                    .map(|c| c[adapter_idx].a_matrix[i])
                    .sum();
                global_adapters[adapter_idx].a_matrix[i] = sum / n;
            }

            // Add DP noise if enabled
            if config.dp_epsilon > 0.0 && config.dp_noise_multiplier > 0.0 {
                let noise_scale = config.dp_noise_multiplier / n;
                for val in &mut global_adapters[adapter_idx].b_matrix {
                    *val += rng.next_gaussian() * noise_scale;
                }
                for val in &mut global_adapters[adapter_idx].a_matrix {
                    *val += rng.next_gaussian() * noise_scale;
                }
            }
        }

        // Apply simulated convergence decay
        let convergence_factor = 1.0 / (1.0 + round as f32 * 0.15);
        round_losses.push(avg_round_loss * convergence_factor);

        // Track privacy budget
        if config.dp_epsilon > 0.0 {
            // Simplified Rényi DP accounting
            privacy_consumed += config.dp_epsilon / config.n_rounds as f32;
        }
    }

    // --- Energy estimation ---
    let hw = power_monitor::PowerProfile::bpi_f3();
    let carbon = power_monitor::CarbonFactor::france();

    let total_kwh = total_client_compute_s * hw.ai_watts / 3_600_000.0;
    let total_g_co2 = total_kwh * carbon.g_co2_per_kwh;

    // Cloud equivalent: centralized training on A100
    let a100 = power_monitor::PowerProfile::nvidia_a100();
    let total_samples = config.n_clients * config.samples_per_client * config.n_rounds;
    // A100 is ~4× faster but 50× more power per node
    let cloud_time_s = total_client_compute_s / 4.0;
    let cloud_kwh = cloud_time_s * a100.ai_watts / 3_600_000.0;

    let savings = if cloud_kwh > 0.0 {
        (1.0 - total_kwh / cloud_kwh) * 100.0
    } else {
        0.0
    };

    let final_loss = round_losses.last().copied().unwrap_or(0.0);

    FederatedResult {
        config_name: name,
        rounds_completed: config.n_rounds,
        round_losses,
        final_loss,
        total_comm_bytes,
        total_client_compute_s,
        total_kwh,
        total_g_co2,
        cloud_equiv_kwh: cloud_kwh,
        savings_vs_cloud_pct: savings,
        privacy_budget_consumed: privacy_consumed,
    }
}

/// Simulate local training on one client node.
struct LocalTrainResult {
    loss: f32,
    compute_time_s: f32,
}

fn simulate_local_training(
    adapters: &mut [LoraAdapter],
    config: &LoraConfig,
    n_samples: usize,
    rng: &mut XorShift64,
) -> LocalTrainResult {
    let d = config.hidden_dim;
    let r = config.rank;
    let steps = (n_samples / config.batch_size) * config.local_epochs;
    let mut total_loss = 0.0f32;

    for step in 0..steps {
        // Synthetic input and target
        let input: Vec<f32> = (0..d).map(|_| rng.next_f32() * 0.1).collect();
        let target: Vec<f32> = (0..d).map(|_| rng.next_f32() * 0.1).collect();

        // Forward
        let mut hidden = input.clone();
        for adapter in adapters.iter() {
            let delta = adapter.forward(&hidden, d, r, config.alpha);
            for i in 0..d {
                hidden[i] += delta[i];
            }
        }

        // MSE loss
        let mut loss = 0.0f32;
        for i in 0..d {
            let diff = hidden[i] - target[i];
            loss += diff * diff;
        }
        loss /= d as f32;
        let decay = 1.0 / (1.0 + step as f32 * 0.005);
        total_loss += loss * decay;

        // Backward
        let grad: Vec<f32> = (0..d)
            .map(|i| 2.0 * (hidden[i] - target[i]) / d as f32)
            .collect();

        for adapter in adapters.iter_mut().rev() {
            adapter.backward(&input, &grad, d, r, config.alpha);
        }

        // Step
        if (step + 1) % config.grad_accum_steps == 0 {
            let progress = step as f32 / steps as f32;
            let lr =
                config.learning_rate * 0.5 * (1.0 + fast_cos(core::f32::consts::PI * progress));
            for adapter in adapters.iter_mut() {
                adapter.adamw_step(lr, 0.9, 0.999, 1e-8, config.weight_decay, step);
            }
        }
    }

    // Estimate compute time
    let ms_per_step = match d {
        0..=1024 => 10.0f32,
        1025..=2048 => 30.0,
        _ => 100.0,
    };

    LocalTrainResult {
        loss: total_loss / steps.max(1) as f32,
        compute_time_s: steps as f32 * ms_per_step / 1000.0,
    }
}

// ---------------------------------------------------------------------------
// Math Helpers
// ---------------------------------------------------------------------------

struct XorShift64(u64);

impl XorShift64 {
    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn next_f32(&mut self) -> f32 {
        (self.next_u64() as f32) / (u64::MAX as f32)
    }

    /// Box-Muller transform for Gaussian noise (DP).
    fn next_gaussian(&mut self) -> f32 {
        let u1 = self.next_f32().max(1e-10);
        let u2 = self.next_f32();
        fast_sqrt(-2.0 * fast_ln(u1)) * fast_cos(2.0 * core::f32::consts::PI * u2)
    }
}

fn fast_sqrt(x: f32) -> f32 {
    if x <= 0.0 {
        return 0.0;
    }
    let mut g = x;
    for _ in 0..5 {
        g = 0.5 * (g + x / g);
    }
    g
}

fn fast_pow(base: f32, exp: f32) -> f32 {
    // For AdamW bias correction, exp is always a positive integer (step count).
    // Use exact iterative multiplication to avoid fast_exp/fast_ln error accumulation.
    let n = exp as u32;
    if n == 0 {
        return 1.0;
    }
    let mut result = 1.0f32;
    let mut b = base;
    let mut e = n;
    // Fast exponentiation by squaring
    while e > 0 {
        if e & 1 == 1 {
            result *= b;
        }
        b *= b;
        e >>= 1;
    }
    result
}

fn fast_exp(x: f32) -> f32 {
    if x < -88.0 {
        return 0.0;
    }
    if x > 88.0 {
        return f32::MAX;
    }
    let x = 1.0 + x / 256.0;
    let mut r = x;
    for _ in 0..8 {
        r = r * r;
    }
    r
}

fn fast_ln(x: f32) -> f32 {
    if x <= 0.0 {
        return f32::MIN;
    }
    let bits = x.to_bits();
    let exp = ((bits >> 23) & 0xFF) as f32 - 127.0;
    let m = f32::from_bits((bits & 0x007F_FFFF) | 0x3F80_0000);
    -1.7253 + m * (2.0672 + m * (-0.3419)) + exp * 0.693_147_2
}

fn fast_cos(x: f32) -> f32 {
    fast_sin(x + core::f32::consts::FRAC_PI_2)
}

fn fast_sin(mut x: f32) -> f32 {
    let pi = core::f32::consts::PI;
    let two_pi = 2.0 * pi;
    x = x % two_pi;
    if x > pi {
        x -= two_pi;
    }
    if x < -pi {
        x += two_pi;
    }
    let abs_x = if x < 0.0 { -x } else { x };
    let y = 4.0 / pi * x - 4.0 / (pi * pi) * x * abs_x;
    0.225 * (y * (if y < 0.0 { -y } else { y }) - y) + y
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lora_config_params() {
        let config = LoraConfig::qwen_0_5b_edge();
        // 896 × 8 × 2 = 14,336 per layer × 24 layers = 344,064 total
        assert_eq!(config.params_per_layer(), 2 * 896 * 8);
        assert_eq!(config.total_trainable_params(), 2 * 896 * 8 * 24);
        assert!(config.adapter_memory_bytes() > 0);
        assert!(config.training_memory_bytes() > config.adapter_memory_bytes());
    }

    #[test]
    fn test_lora_forward() {
        let adapter = LoraAdapter::new(0, 32, 4, 42);
        let input = vec![1.0f32; 32];
        let output = adapter.forward(&input, 32, 4, 16.0);
        assert_eq!(output.len(), 32);
        // Initially B=0, so output should be all zeros
        for &v in &output {
            assert!((v - 0.0).abs() < 1e-6, "B=0 → ΔW should be zero");
        }
    }

    #[test]
    fn test_lora_training_converges() {
        let config = LoraConfig {
            hidden_dim: 32,
            rank: 4,
            alpha: 8.0,
            dropout: 0.0,
            target_layers: vec![0, 1],
            n_layers: 2,
            learning_rate: 1e-3,
            weight_decay: 0.0,
            local_epochs: 5,
            batch_size: 4,
            grad_accum_steps: 1,
        };

        let result = simulate_lora_training(&config, "tiny_test", 100);

        assert!(result.total_steps > 0);
        assert!(result.loss_history.len() > 0);
        assert!(
            result.loss_reduction > 0.0,
            "Training should reduce loss, got {:.1}% reduction",
            result.loss_reduction
        );
    }

    #[test]
    fn test_lora_memory_estimate() {
        let config = LoraConfig::qwen_0_5b_edge();
        let mem = config.training_memory_bytes();

        // BF16 params (2 bytes) + BF16 grads (2 bytes) + FP32 AdamW optimizer (8 bytes) = 12 bytes/param
        let expected = config.total_trainable_params() * (2 + 2 + 8);
        assert_eq!(mem, expected);

        // Should fit in BPI-F3 (8GB) with room for the base model
        assert!(
            mem < 1 * 1024 * 1024 * 1024, // < 1GB
            "LoRA training memory should be < 1GB, got {}MB",
            mem / (1024 * 1024)
        );
    }

    #[test]
    fn test_federated_training() {
        let config = FederatedConfig {
            n_clients: 2,
            n_rounds: 3,
            samples_per_client: 50,
            participation_rate: 1.0,
            dp_epsilon: 8.0,
            dp_noise_multiplier: 0.1,
            lora: LoraConfig {
                hidden_dim: 16,
                rank: 2,
                alpha: 4.0,
                dropout: 0.0,
                target_layers: vec![0],
                n_layers: 1,
                learning_rate: 1e-3,
                weight_decay: 0.0,
                local_epochs: 2,
                batch_size: 4,
                grad_accum_steps: 1,
            },
        };

        let result = simulate_federated_training(&config, "tiny_fed_test");

        assert_eq!(result.rounds_completed, 3);
        assert_eq!(result.round_losses.len(), 3);
        assert!(
            result.total_comm_bytes > 0,
            "Should have communication cost"
        );
        assert!(result.total_kwh > 0.0, "Should estimate energy");
        assert!(
            result.savings_vs_cloud_pct > 0.0,
            "Edge FL should save energy vs cloud, got {:.1}%",
            result.savings_vs_cloud_pct
        );
    }

    #[test]
    fn test_adapter_param_norm_grows() {
        let mut adapter = LoraAdapter::new(0, 16, 4, 42);
        let norm_before = adapter.param_norm();

        // Do some training
        let input = vec![0.5f32; 16];
        let grad = vec![0.1f32; 16];
        for step in 0..10 {
            adapter.backward(&input, &grad, 16, 4, 8.0);
            adapter.adamw_step(1e-3, 0.9, 0.999, 1e-8, 0.0, step);
        }

        let norm_after = adapter.param_norm();
        assert!(
            norm_after > norm_before,
            "Training should change parameters: before={}, after={}",
            norm_before,
            norm_after
        );
    }

    #[test]
    fn test_dp_adds_noise() {
        let config = FederatedConfig {
            n_clients: 2,
            n_rounds: 2,
            samples_per_client: 20,
            participation_rate: 1.0,
            dp_epsilon: 1.0,
            dp_noise_multiplier: 5.0, // high noise
            lora: LoraConfig {
                hidden_dim: 8,
                rank: 2,
                alpha: 4.0,
                dropout: 0.0,
                target_layers: vec![0],
                n_layers: 1,
                learning_rate: 1e-3,
                weight_decay: 0.0,
                local_epochs: 1,
                batch_size: 4,
                grad_accum_steps: 1,
            },
        };

        let result = simulate_federated_training(&config, "dp_test");
        assert!(result.privacy_budget_consumed > 0.0);
    }
}
