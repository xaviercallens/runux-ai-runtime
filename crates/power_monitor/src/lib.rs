// Copyright (c) 2026 Xavier Callens / Socrate AI. All Rights Reserved.
// SPDX-License-Identifier: LicenseRef-RunuX-Commercial
// This file is proprietary and confidential. Unauthorized copying,
// distribution, or use is strictly prohibited.

#![cfg_attr(not(test), no_std)]
#![deny(clippy::all)]
#![warn(clippy::pedantic)]
//! RunuX Power Monitor — Energy profiling and CO2 estimation for sustainable AI
//!
//! Tracks power consumption, computes per-token energy costs, and estimates
//! CO2 emissions for LLM inference and training on RISC-V edge hardware.
//!
//! # Why this matters
//!
//! A single GPT-4 query on cloud infrastructure consumes ~0.0017 kWh and
//! emits ~0.7g CO2 (Luccioni et al., 2024). RunuX on RISC-V edge hardware
//! targets 10-50× lower energy per token through:
//!
//! - **Efficient hardware**: SpacemiT K1 (5W TDP) vs. A100 (400W TDP)
//! - **Quantization**: Q4_K_M reduces compute 4× vs. FP16
//! - **FlashAttention**: Eliminates redundant memory traffic
//! - **Speculative decoding**: Amortizes target model cost
//! - **Federated learning**: No data center round-trip
//!
//! # Hardware power profiles
//!
//! | Device | TDP | AI Workload | Idle |
//! |--------|-----|-------------|------|
//! | BPI-F3 (K1) | 10W | 5-7W | 2W |
//! | AIBOX-K3 (K3) | 25W | 12-18W | 5W |
//! | A100 (reference) | 400W | 250-350W | 50W |

extern crate alloc;
use alloc::vec::Vec;

// ---------------------------------------------------------------------------
// Hardware Power Profiles
// ---------------------------------------------------------------------------

/// Power profile for a specific hardware platform.
#[derive(Debug, Clone)]
pub struct PowerProfile {
    /// Platform name
    pub name: &'static str,
    /// Thermal Design Power (Watts)
    pub tdp_watts: f32,
    /// Typical AI workload power (Watts)
    pub ai_watts: f32,
    /// Idle power (Watts)
    pub idle_watts: f32,
    /// Peak INT8 TOPS
    pub peak_tops: f32,
    /// Peak FP32 GFLOPS
    pub peak_gflops: f32,
    /// RAM bandwidth (GB/s)
    pub mem_bandwidth_gbs: f32,
}

impl PowerProfile {
    /// BPI-F3: SpacemiT K1 8-core, 2.0 TOPS, LPDDR4
    pub fn bpi_f3() -> Self {
        Self {
            name: "BPI-F3 (SpacemiT K1)",
            tdp_watts: 10.0,
            ai_watts: 6.0,
            idle_watts: 2.0,
            peak_tops: 2.0,
            peak_gflops: 16.0, // 8 cores × 1.6GHz × 2 FMA
            mem_bandwidth_gbs: 12.8, // LPDDR4-3200 single-channel
        }
    }

    /// AIBOX-K3: SpacemiT K3, 60 TOPS (A100 cores), PowerVR BXM
    pub fn aibox_k3() -> Self {
        Self {
            name: "AIBOX-K3 (SpacemiT K3)",
            tdp_watts: 25.0,
            ai_watts: 15.0,
            idle_watts: 5.0,
            peak_tops: 60.0,
            peak_gflops: 128.0,
            mem_bandwidth_gbs: 51.2, // LPDDR5-6400 dual-channel
        }
    }

    /// NVIDIA A100 (for comparison only).
    pub fn nvidia_a100() -> Self {
        Self {
            name: "NVIDIA A100 (reference)",
            tdp_watts: 400.0,
            ai_watts: 300.0,
            idle_watts: 50.0,
            peak_tops: 624.0, // INT8
            peak_gflops: 19500.0, // FP32
            mem_bandwidth_gbs: 2039.0, // HBM2e
        }
    }

    /// Energy efficiency: TOPS per Watt (higher = better).
    pub fn tops_per_watt(&self) -> f32 {
        if self.ai_watts > 0.0 {
            self.peak_tops / self.ai_watts
        } else {
            0.0
        }
    }

    /// GFLOPS per Watt.
    pub fn gflops_per_watt(&self) -> f32 {
        if self.ai_watts > 0.0 {
            self.peak_gflops / self.ai_watts
        } else {
            0.0
        }
    }
}

// ---------------------------------------------------------------------------
// CO2 Emission Model
// ---------------------------------------------------------------------------

/// Regional CO2 emission factors (gCO2/kWh).
#[derive(Debug, Clone, Copy)]
pub struct CarbonFactor {
    /// Region name
    pub region: &'static str,
    /// Grams of CO2 per kWh
    pub g_co2_per_kwh: f32,
}

impl CarbonFactor {
    pub fn france() -> Self {
        Self { region: "France", g_co2_per_kwh: 56.0 } // Nuclear-dominated
    }
    pub fn china_avg() -> Self {
        Self { region: "China (avg)", g_co2_per_kwh: 555.0 } // Coal-heavy
    }
    pub fn china_yunnan() -> Self {
        Self { region: "China (Yunnan)", g_co2_per_kwh: 120.0 } // Hydro
    }
    pub fn germany() -> Self {
        Self { region: "Germany", g_co2_per_kwh: 350.0 }
    }
    pub fn us_avg() -> Self {
        Self { region: "USA (avg)", g_co2_per_kwh: 386.0 }
    }
    pub fn us_california() -> Self {
        Self { region: "USA (CA)", g_co2_per_kwh: 210.0 }
    }
    pub fn iceland() -> Self {
        Self { region: "Iceland", g_co2_per_kwh: 28.0 } // Geothermal
    }
}

// ---------------------------------------------------------------------------
// Inference Energy Estimator
// ---------------------------------------------------------------------------

/// Per-token energy estimate.
#[derive(Debug, Clone)]
pub struct TokenEnergyEstimate {
    /// Energy per token (Joules)
    pub joules_per_token: f32,
    /// Energy per token (kWh)
    pub kwh_per_token: f32,
    /// CO2 per token (grams)
    pub g_co2_per_token: f32,
    /// CO2 per 1000 tokens (grams) — easier to reason about
    pub g_co2_per_1k_tokens: f32,
    /// Tokens per kWh (efficiency metric)
    pub tokens_per_kwh: f32,
}

/// Estimate per-token energy and CO2 for a given hardware + model config.
pub fn estimate_token_energy(
    profile: &PowerProfile,
    tokens_per_second: f32,
    carbon: &CarbonFactor,
) -> TokenEnergyEstimate {
    // Energy per token = power / throughput
    let joules = if tokens_per_second > 0.0 {
        profile.ai_watts / tokens_per_second
    } else {
        0.0
    };

    let kwh = joules / 3_600_000.0;
    let g_co2 = kwh * carbon.g_co2_per_kwh;

    TokenEnergyEstimate {
        joules_per_token: joules,
        kwh_per_token: kwh,
        g_co2_per_token: g_co2,
        g_co2_per_1k_tokens: g_co2 * 1000.0,
        tokens_per_kwh: if kwh > 0.0 { 1.0 / kwh } else { 0.0 },
    }
}

// ---------------------------------------------------------------------------
// Training Energy Estimator
// ---------------------------------------------------------------------------

/// Training energy estimate for federated learning.
#[derive(Debug, Clone)]
pub struct TrainingEnergyEstimate {
    /// Total energy for training (kWh)
    pub total_kwh: f32,
    /// Total CO2 emissions (kg)
    pub total_kg_co2: f32,
    /// Energy per training sample (Joules)
    pub joules_per_sample: f32,
    /// Comparison: A100 cloud equivalent (kWh)
    pub cloud_equivalent_kwh: f32,
    /// Energy savings vs cloud (%)
    pub savings_vs_cloud_percent: f32,
}

/// Estimate training energy for federated LoRA fine-tuning.
pub fn estimate_training_energy(
    num_nodes: usize,
    profile: &PowerProfile,
    training_hours: f32,
    samples_per_node: usize,
    num_rounds: usize,
    carbon: &CarbonFactor,
) -> TrainingEnergyEstimate {
    // Edge training energy
    let total_node_hours = num_nodes as f32 * training_hours;
    let total_kwh = total_node_hours * profile.ai_watts / 1000.0;
    let total_kg_co2 = total_kwh * carbon.g_co2_per_kwh / 1000.0;

    let total_samples = (samples_per_node * num_rounds * num_nodes) as f32;
    let joules_per_sample = if total_samples > 0.0 {
        (total_kwh * 3_600_000.0) / total_samples
    } else {
        0.0
    };

    // Cloud equivalent (A100): assume 4× faster but 30× more power
    let a100 = PowerProfile::nvidia_a100();
    let cloud_hours = training_hours / 4.0; // A100 is ~4× faster
    let cloud_kwh = cloud_hours * a100.ai_watts / 1000.0;

    let savings = if cloud_kwh > 0.0 {
        ((cloud_kwh - total_kwh) / cloud_kwh) * 100.0
    } else {
        0.0
    };

    TrainingEnergyEstimate {
        total_kwh,
        total_kg_co2,
        joules_per_sample,
        cloud_equivalent_kwh: cloud_kwh,
        savings_vs_cloud_percent: savings,
    }
}

// ---------------------------------------------------------------------------
// Simulation Benchmark Runner
// ---------------------------------------------------------------------------

/// Run a full power simulation comparing different hardware + optimization
/// configurations for a target model.
#[derive(Debug, Clone)]
pub struct SimulationResult {
    /// Configuration name
    pub config_name: &'static str,
    /// Hardware used
    pub hardware: &'static str,
    /// Estimated tokens/second
    pub tokens_per_second: f32,
    /// Energy per token (Joules)
    pub joules_per_token: f32,
    /// CO2 per 1K tokens (grams)
    pub g_co2_per_1k: f32,
    /// Speedup vs baseline
    pub speedup_vs_baseline: f32,
    /// Energy savings vs baseline (%)
    pub energy_savings_percent: f32,
}

/// Run power simulation across hardware configurations.
pub fn run_power_simulation() -> Vec<SimulationResult> {
    let france = CarbonFactor::france();
    let mut results = Vec::new();

    // --- Baseline: Standard attention, FP32, BPI-F3 ---
    let bpi = PowerProfile::bpi_f3();
    let baseline_tps = 2.0; // ~2 tok/s for 1.5B FP32 on K1
    let baseline = estimate_token_energy(&bpi, baseline_tps, &france);

    results.push(SimulationResult {
        config_name: "Baseline (FP32, standard attn)",
        hardware: bpi.name,
        tokens_per_second: baseline_tps,
        joules_per_token: baseline.joules_per_token,
        g_co2_per_1k: baseline.g_co2_per_1k_tokens,
        speedup_vs_baseline: 1.0,
        energy_savings_percent: 0.0,
    });

    // --- Q4_K_M quantization ---
    let q4_tps = 8.0; // ~4× speedup from quantization
    let q4 = estimate_token_energy(&bpi, q4_tps, &france);
    results.push(SimulationResult {
        config_name: "Q4_K_M quantization",
        hardware: bpi.name,
        tokens_per_second: q4_tps,
        joules_per_token: q4.joules_per_token,
        g_co2_per_1k: q4.g_co2_per_1k_tokens,
        speedup_vs_baseline: q4_tps / baseline_tps,
        energy_savings_percent: (1.0 - q4.joules_per_token / baseline.joules_per_token) * 100.0,
    });

    // --- Q4_K_M + FlashAttention ---
    let flash_tps = 12.0; // +50% from FlashAttention memory savings
    let flash = estimate_token_energy(&bpi, flash_tps, &france);
    results.push(SimulationResult {
        config_name: "Q4_K_M + FlashAttention",
        hardware: bpi.name,
        tokens_per_second: flash_tps,
        joules_per_token: flash.joules_per_token,
        g_co2_per_1k: flash.g_co2_per_1k_tokens,
        speedup_vs_baseline: flash_tps / baseline_tps,
        energy_savings_percent: (1.0 - flash.joules_per_token / baseline.joules_per_token) * 100.0,
    });

    // --- Q4_K_M + FlashAttn + Speculative decoding (edge cluster) ---
    let spec_tps = 20.0; // ~2.5× from speculative (K=5, α≈0.6)
    let spec = estimate_token_energy(&bpi, spec_tps, &france);
    results.push(SimulationResult {
        config_name: "Q4 + FlashAttn + Speculative (K=5)",
        hardware: "BPI-F3 → AIBOX-K3 cluster",
        tokens_per_second: spec_tps,
        joules_per_token: spec.joules_per_token,
        g_co2_per_1k: spec.g_co2_per_1k_tokens,
        speedup_vs_baseline: spec_tps / baseline_tps,
        energy_savings_percent: (1.0 - spec.joules_per_token / baseline.joules_per_token) * 100.0,
    });

    // --- AIBOX-K3 native FP8 ---
    let k3 = PowerProfile::aibox_k3();
    let k3_tps = 45.0; // K3 with FP8 + A100 cores
    let k3_est = estimate_token_energy(&k3, k3_tps, &france);
    results.push(SimulationResult {
        config_name: "K3 FP8 native + FlashAttn",
        hardware: k3.name,
        tokens_per_second: k3_tps,
        joules_per_token: k3_est.joules_per_token,
        g_co2_per_1k: k3_est.g_co2_per_1k_tokens,
        speedup_vs_baseline: k3_tps / baseline_tps,
        energy_savings_percent: (1.0 - k3_est.joules_per_token / baseline.joules_per_token) * 100.0,
    });

    // --- Reference: A100 cloud ---
    let a100 = PowerProfile::nvidia_a100();
    let a100_tps = 150.0;
    let a100_est = estimate_token_energy(&a100, a100_tps, &france);
    results.push(SimulationResult {
        config_name: "NVIDIA A100 (cloud reference)",
        hardware: a100.name,
        tokens_per_second: a100_tps,
        joules_per_token: a100_est.joules_per_token,
        g_co2_per_1k: a100_est.g_co2_per_1k_tokens,
        speedup_vs_baseline: a100_tps / baseline_tps,
        energy_savings_percent: (1.0 - a100_est.joules_per_token / baseline.joules_per_token) * 100.0,
    });

    results
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bpi_f3_profile() {
        let p = PowerProfile::bpi_f3();
        assert!(p.tops_per_watt() > 0.0);
        assert!(p.gflops_per_watt() > 0.0);
        assert!(p.ai_watts < p.tdp_watts);
    }

    #[test]
    fn test_k3_more_efficient() {
        let k1 = PowerProfile::bpi_f3();
        let k3 = PowerProfile::aibox_k3();
        // K3 should have better TOPS/W due to dedicated AI cores
        assert!(
            k3.tops_per_watt() > k1.tops_per_watt(),
            "K3 ({:.2} TOPS/W) should beat K1 ({:.2} TOPS/W)",
            k3.tops_per_watt(), k1.tops_per_watt()
        );
    }

    #[test]
    fn test_token_energy_estimate() {
        let profile = PowerProfile::bpi_f3();
        let carbon = CarbonFactor::france();
        let est = estimate_token_energy(&profile, 10.0, &carbon);

        assert!(est.joules_per_token > 0.0);
        assert!(est.g_co2_per_token > 0.0);
        assert!(est.tokens_per_kwh > 0.0);
    }

    #[test]
    fn test_edge_vs_cloud_co2() {
        let carbon = CarbonFactor::france();

        let bpi = PowerProfile::bpi_f3();
        let edge = estimate_token_energy(&bpi, 10.0, &carbon);

        let a100 = PowerProfile::nvidia_a100();
        let cloud = estimate_token_energy(&a100, 150.0, &carbon);

        // Edge should have lower CO2 per token despite lower throughput
        // because power draw is dramatically lower
        assert!(
            edge.g_co2_per_token < cloud.g_co2_per_token,
            "Edge ({:.6} gCO2/tok) should be greener than cloud ({:.6} gCO2/tok)",
            edge.g_co2_per_token, cloud.g_co2_per_token
        );
    }

    #[test]
    fn test_training_energy() {
        let est = estimate_training_energy(
            4,                            // 4 BPI-F3 nodes
            &PowerProfile::bpi_f3(),
            2.0,                          // 2 hours training
            1000,                         // 1000 samples/node
            10,                           // 10 rounds
            &CarbonFactor::france(),
        );

        assert!(est.total_kwh > 0.0);
        assert!(est.savings_vs_cloud_percent > 0.0,
            "Edge training should be more energy-efficient than cloud");
    }

    #[test]
    fn test_power_simulation() {
        let results = run_power_simulation();
        assert!(results.len() >= 5);

        // Verify that BPI-F3 optimizations (indices 0-3) progressively improve
        // energy efficiency (lower joules_per_token)
        for i in 1..4 {
            assert!(
                results[i].joules_per_token <= results[i - 1].joules_per_token * 1.01,
                "BPI-F3 config '{}' ({:.3} J/tok) should improve on '{}' ({:.3} J/tok)",
                results[i].config_name, results[i].joules_per_token,
                results[i - 1].config_name, results[i - 1].joules_per_token
            );
        }

        // K3 should be faster than baseline BPI-F3 (higher tok/s)
        assert!(results[4].tokens_per_second > results[0].tokens_per_second,
            "K3 should have higher throughput than baseline BPI-F3");

        // A100 should be fastest overall
        let last = results.len() - 1;
        assert!(results[last].tokens_per_second > results[0].tokens_per_second * 10.0,
            "A100 should be >10× faster than baseline");
    }

    #[test]
    fn test_carbon_factors() {
        let fr = CarbonFactor::france();
        let cn = CarbonFactor::china_avg();
        assert!(fr.g_co2_per_kwh < cn.g_co2_per_kwh,
            "France (nuclear) should have lower carbon intensity than China avg");
    }
}
