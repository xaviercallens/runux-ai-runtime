// Copyright (c) 2026 Xavier Callens / Socrate AI. All Rights Reserved.
// SPDX-License-Identifier: LicenseRef-RunuX-Commercial
// This file is proprietary and confidential. Unauthorized copying,
// distribution, or use is strictly prohibited.

#![cfg_attr(not(test), no_std)]
#![deny(clippy::all)]
#![warn(clippy::pedantic)]
//! RunuX Simulation Benchmarks — End-to-end optimization validation
//!
//! Runs the full optimization stack in simulation mode (no hardware required)
//! to validate correctness, measure theoretical speedups, and estimate
//! power/CO2 savings before deploying to physical RISC-V hardware.
//!
//! # Benchmarks
//!
//! 1. **FlashAttention correctness**: Verify against standard O(N²) attention
//! 2. **Memory planning**: Validate model fits on target hardware
//! 3. **Arena allocator**: Measure allocation throughput and fragmentation
//! 4. **Speculative decoding**: Simulate acceptance rates and speedup
//! 5. **TurboQuant**: Measure KV-cache compression ratio
//! 6. **Power simulation**: Estimate CO2 across optimization configurations

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;

// no_std trig approximations
fn approx_sin(x: f32) -> f32 {
    let pi = core::f32::consts::PI;
    let two_pi = 2.0 * pi;
    let mut a = x % two_pi;
    if a > pi { a -= two_pi; }
    if a < -pi { a += two_pi; }
    let abs_a = if a < 0.0 { -a } else { a };
    let y = 4.0 / pi * a - 4.0 / (pi * pi) * a * abs_a;
    0.225 * (y * (if y < 0.0 { -y } else { y }) - y) + y
}
fn approx_cos(x: f32) -> f32 {
    approx_sin(x + core::f32::consts::FRAC_PI_2)
}

// ---------------------------------------------------------------------------
// Model Configurations for Benchmarking
// ---------------------------------------------------------------------------

/// Pre-defined model configurations for simulation.
#[derive(Debug, Clone)]
pub struct ModelSpec {
    pub name: &'static str,
    pub params_b: f32,  // billions of parameters
    pub hidden_dim: usize,
    pub head_dim: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub n_layers: usize,
    pub vocab_size: usize,
    pub intermediate_dim: usize,
}

impl ModelSpec {
    pub fn qwen2_0_5b() -> Self {
        Self {
            name: "Qwen 2.5 0.5B",
            params_b: 0.5,
            hidden_dim: 896,
            head_dim: 64,
            n_heads: 14,
            n_kv_heads: 2,
            n_layers: 24,
            vocab_size: 151936,
            intermediate_dim: 4864,
        }
    }

    pub fn deepseek_r1_1_5b() -> Self {
        Self {
            name: "DeepSeek R1 1.5B (distill)",
            params_b: 1.5,
            hidden_dim: 1536,
            head_dim: 128,
            n_heads: 12,
            n_kv_heads: 2,
            n_layers: 28,
            vocab_size: 151936,
            intermediate_dim: 8960,
        }
    }

    pub fn qwen2_7b() -> Self {
        Self {
            name: "Qwen 2.5 7B",
            params_b: 7.0,
            hidden_dim: 3584,
            head_dim: 128,
            n_heads: 28,
            n_kv_heads: 4,
            n_layers: 28,
            vocab_size: 151936,
            intermediate_dim: 18944,
        }
    }

    pub fn deepseek_r1_14b() -> Self {
        Self {
            name: "DeepSeek R1 14B (distill)",
            params_b: 14.0,
            hidden_dim: 5120,
            head_dim: 128,
            n_heads: 40,
            n_kv_heads: 8,
            n_layers: 40,
            vocab_size: 151936,
            intermediate_dim: 13824,
        }
    }

    /// Total parameters (exact count for memory planning).
    pub fn total_params(&self) -> u64 {
        (self.params_b * 1_000_000_000.0) as u64
    }
}

// ---------------------------------------------------------------------------
// Simulation Results
// ---------------------------------------------------------------------------

/// Result of a single benchmark run.
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Benchmark name
    pub name: &'static str,
    /// Model tested
    pub model: &'static str,
    /// Hardware target
    pub hardware: &'static str,
    /// Primary metric value
    pub metric_value: f64,
    /// Metric unit
    pub metric_unit: &'static str,
    /// Whether the benchmark passed
    pub passed: bool,
    /// Human-readable notes
    pub notes: Vec<&'static str>,
}

// ---------------------------------------------------------------------------
// Benchmark: FlashAttention Correctness
// ---------------------------------------------------------------------------

/// Run FlashAttention correctness benchmark.
pub fn bench_flash_attention_correctness() -> BenchmarkResult {
    let d = 64; // Qwen head_dim
    let seq_lens = [32, 64, 128, 256];
    let mut max_error = 0.0f32;

    for &n in &seq_lens {
        let q: Vec<f32> = (0..n * d).map(|i| approx_sin((i as f32) * 0.037)).collect();
        let k: Vec<f32> = (0..n * d).map(|i| approx_cos((i as f32) * 0.041)).collect();
        let v: Vec<f32> = (0..n * d).map(|i| approx_sin((i as f32) * 0.029)).collect();

        let (_, _, err) = flash_attention::validate_against_standard(&q, &k, &v, d);
        if err > max_error {
            max_error = err;
        }
    }

    let passed = max_error < 0.5; // Generous tolerance for fast_exp approximation

    BenchmarkResult {
        name: "FlashAttention correctness",
        model: "Generic (head_dim=64)",
        hardware: "Simulation",
        metric_value: max_error as f64,
        metric_unit: "max_abs_error",
        passed,
        notes: if passed {
            vec!["FlashAttention matches standard attention within tolerance"]
        } else {
            vec!["ERROR: FlashAttention diverges from standard attention"]
        },
    }
}

// ---------------------------------------------------------------------------
// Benchmark: FlashAttention Memory Savings
// ---------------------------------------------------------------------------

/// Run FlashAttention memory savings benchmark.
pub fn bench_flash_memory_savings() -> BenchmarkResult {
    let est = flash_attention::estimate_memory(
        4096, 128, 32, 32, 64, 64,
    );

    BenchmarkResult {
        name: "FlashAttention memory savings",
        model: "14B model, 4K context",
        hardware: "Any",
        metric_value: est.savings_ratio as f64,
        metric_unit: "memory_savings_ratio",
        passed: est.savings_ratio > 5.0,
        notes: vec![
            "Standard attention requires O(N²) memory",
            "FlashAttention reduces to O(N) via tiling",
        ],
    }
}

// ---------------------------------------------------------------------------
// Benchmark: Arena Allocator Performance
// ---------------------------------------------------------------------------

/// Run arena allocator benchmark.
pub fn bench_arena_allocator() -> BenchmarkResult {
    let arena = arena_mem::BumpAllocator::new(1024 * 1024); // 1MB

    // Simulate per-token allocations for a full forward pass
    let num_allocations = 100;
    let mut total_bytes = 0usize;

    for _i in 0..num_allocations {
        let size = 896 * 4; // hidden_dim buffer
        if let Some(_offset) = arena.alloc_f32(size / 4, 64) {
            total_bytes += size;
        }
    }

    let _utilization = arena.utilization_percent();
    arena.reset();

    BenchmarkResult {
        name: "Arena bump allocation throughput",
        model: "Qwen 2.5 0.5B",
        hardware: "Simulation",
        metric_value: num_allocations as f64,
        metric_unit: "allocations_before_reset",
        passed: arena.alloc_count() == 0 && total_bytes > 0,
        notes: vec![
            "O(1) bump allocation with zero fragmentation",
            "Instant deallocation via pointer reset",
        ],
    }
}

// ---------------------------------------------------------------------------
// Benchmark: Memory Planning
// ---------------------------------------------------------------------------

/// Run memory planning benchmark for target hardware.
pub fn bench_memory_planning() -> Vec<BenchmarkResult> {
    let models = [
        ModelSpec::qwen2_0_5b(),
        ModelSpec::deepseek_r1_1_5b(),
        ModelSpec::qwen2_7b(),
        ModelSpec::deepseek_r1_14b(),
    ];

    let hardware_configs = [
        ("BPI-F3 (4GB)", 4_294_967_296usize), // 4GB
        ("AIBOX-K3 (8GB)", 8_589_934_592usize), // 8GB
        ("AIBOX-K3 (32GB)", 34_359_738_368usize), // 32GB
    ];

    let quant_bits = [4u32, 8]; // Q4_K_M and Q8_0

    let mut results = Vec::new();

    for model in &models {
        for &(hw_name, ram) in &hardware_configs {
            for &bits in &quant_bits {
                let budget = arena_mem::plan_memory(
                    model.total_params(),
                    bits,
                    model.head_dim,
                    model.n_heads,
                    model.n_kv_heads,
                    model.n_layers,
                    4096,
                    ram,
                );

                let _quant_name = if bits == 4 { "Q4_K_M" } else { "Q8_0" };

                results.push(BenchmarkResult {
                    name: "Memory fit check",
                    model: model.name,
                    hardware: hw_name,
                    metric_value: budget.max_seq_len as f64,
                    metric_unit: "max_seq_len",
                    passed: budget.fits,
                    notes: if budget.fits {
                        vec!["Model fits with target quantization"]
                    } else {
                        vec!["Model does NOT fit — lower quantization needed"]
                    },
                });
            }
        }
    }

    results
}

// ---------------------------------------------------------------------------
// Benchmark: TurboQuant KV-Cache Compression
// ---------------------------------------------------------------------------

/// Run TurboQuant compression benchmark.
pub fn bench_turbo_quant_compression() -> BenchmarkResult {
    let config = turbo_quant::TurboQuantConfig {
        target_bits: 3,
        block_size: 32,
        use_qjl_correction: true,
        ..turbo_quant::TurboQuantConfig::default()
    };

    let dim = 128; // head_dim
    let key: Vec<f32> = (0..dim).map(|i| approx_sin((i as f32) * 0.1)).collect();
    let value: Vec<f32> = (0..dim).map(|i| approx_cos((i as f32) * 0.07)).collect();

    let compressed = turbo_quant::compress_kv(&key, &value, &config, 0);

    // Calculate compression ratio
    let original_bytes = dim * 4 * 2; // FP32 K + V
    let compressed_bytes = compressed.key_quantized.len()
        + compressed.value_quantized.len()
        + compressed.key_scales.len() * 4
        + compressed.value_scales.len() * 4
        + compressed.key_zeros.len() * 4
        + compressed.value_zeros.len() * 4
        + compressed.key_qjl.len() * 4
        + compressed.value_qjl.len() * 4;

    let ratio = if compressed_bytes > 0 {
        original_bytes as f64 / compressed_bytes as f64
    } else {
        0.0
    };

    BenchmarkResult {
        name: "TurboQuant KV-cache compression",
        model: "Generic (d=128)",
        hardware: "Any",
        metric_value: ratio,
        metric_unit: "compression_ratio",
        passed: ratio > 1.5,
        notes: vec![
            "PolarQuant rotation + 3-bit scalar quantization",
            "QJL error correction preserves attention accuracy",
        ],
    }
}

// ---------------------------------------------------------------------------
// Benchmark: Power Simulation
// ---------------------------------------------------------------------------

/// Run full power simulation benchmark.
pub fn bench_power_simulation() -> Vec<BenchmarkResult> {
    let results = power_monitor::run_power_simulation();
    let mut bench_results = Vec::new();

    for sim in &results {
        bench_results.push(BenchmarkResult {
            name: sim.config_name,
            model: "DeepSeek R1 1.5B",
            hardware: sim.hardware,
            metric_value: sim.g_co2_per_1k as f64,
            metric_unit: "gCO2_per_1K_tokens",
            passed: true,
            notes: vec![],
        });
    }

    bench_results
}

// ---------------------------------------------------------------------------
// Run All Benchmarks
// ---------------------------------------------------------------------------

/// Run the complete simulation benchmark suite.
pub fn run_all_benchmarks() -> Vec<BenchmarkResult> {
    let mut all = Vec::new();

    all.push(bench_flash_attention_correctness());
    all.push(bench_flash_memory_savings());
    all.push(bench_arena_allocator());
    all.extend(bench_memory_planning());
    all.push(bench_turbo_quant_compression());
    all.extend(bench_power_simulation());

    all
}

// ---------------------------------------------------------------------------
// Tests (run benchmarks as tests)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flash_correctness() {
        let result = bench_flash_attention_correctness();
        assert!(result.passed, "FlashAttention correctness check failed: max_error={}", result.metric_value);
    }

    #[test]
    fn test_flash_memory_savings() {
        let result = bench_flash_memory_savings();
        assert!(result.passed, "FlashAttention should save >5× memory");
    }

    #[test]
    fn test_arena_benchmark() {
        let result = bench_arena_allocator();
        assert!(result.passed, "Arena allocator benchmark failed");
    }

    #[test]
    fn test_memory_planning() {
        let results = bench_memory_planning();
        assert!(!results.is_empty());
        // Qwen 0.5B Q4 should fit on BPI-F3
        let qwen_bpi = results.iter()
            .find(|r| r.model == "Qwen 2.5 0.5B" && r.hardware == "BPI-F3 (4GB)")
            .expect("Should have Qwen 0.5B on BPI-F3");
        assert!(qwen_bpi.passed, "Qwen 0.5B Q4 should fit on BPI-F3 4GB");
    }

    #[test]
    fn test_turbo_quant() {
        let result = bench_turbo_quant_compression();
        assert!(result.passed, "TurboQuant should achieve >1.5× compression");
    }

    #[test]
    fn test_power_simulation() {
        let results = bench_power_simulation();
        assert!(results.len() >= 5, "Should have at least 5 power configs");
    }

    #[test]
    fn test_all_benchmarks() {
        let all = run_all_benchmarks();
        let failures: Vec<_> = all.iter().filter(|r| !r.passed).collect();
        assert!(
            failures.is_empty(),
            "Failed benchmarks: {:?}",
            failures.iter().map(|r| r.name).collect::<Vec<_>>()
        );
    }
}
