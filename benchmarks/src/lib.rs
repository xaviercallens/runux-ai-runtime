#![no_std]
#![cfg_attr(not(test), no_main)]
// Copyright (c) 2026 Xavier Callens / Socrate AI. All Rights Reserved.
// SPDX-License-Identifier: LicenseRef-RunuX-Commercial

//! RunuX AI Runtime — Benchmark Suite
//!
//! Standardized benchmarks for measuring RunuX AI Runtime performance.
//! Results are publishable; implementation details are proprietary.

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;
use alloc::string::String;

use ai_runtime::{DataType, DeviceType, HardwareCaps};
use rvv_simd::{matmul_scalar_f32, matmul_rvv_f32, softmax_f32, rms_norm_f32, silu_f32};
use turbo_quant::{TurboQuantConfig, compress_kv, decompress_kv, CompressedKvCache};

/// Benchmark result for a single operation.
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Operation name
    pub name: String,
    /// Input dimensions
    pub dimensions: String,
    /// Number of iterations
    pub iterations: usize,
    /// Total time in microseconds (to be measured by caller)
    pub total_us: u64,
    /// Throughput in operations per second
    pub ops_per_sec: f64,
    /// Memory usage in bytes
    pub memory_bytes: usize,
    /// Hardware platform
    pub platform: String,
}

/// Run matmul benchmark for various sizes.
pub fn bench_matmul_sizes() -> Vec<(usize, usize, usize)> {
    vec![
        // (M, K, N) — typical transformer dimensions
        (1, 4096, 4096),      // Single token, hidden_dim=4096
        (1, 4096, 11008),     // Single token, FFN up-projection
        (32, 4096, 4096),     // Batch=32 prefill
        (128, 4096, 4096),    // Batch=128 prefill
        (1, 2048, 2048),      // Smaller model (Qwen 0.5B)
        (1, 5120, 5120),      // Larger model (Qwen 14B)
    ]
}

/// Prepare matmul input data for benchmarking.
pub fn prepare_matmul_data(m: usize, k: usize, n: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let a: Vec<f32> = (0..m * k).map(|i| ((i % 17) as f32 - 8.0) * 0.01).collect();
    let b: Vec<f32> = (0..k * n).map(|i| ((i % 13) as f32 - 6.0) * 0.01).collect();
    let c = vec![0.0f32; m * n];
    (a, b, c)
}

/// Run matmul and return the output (for correctness verification).
pub fn run_matmul_benchmark(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    use_rvv: bool,
) {
    if use_rvv {
        matmul_rvv_f32(a, b, c, m, k, n);
    } else {
        matmul_scalar_f32(a, b, c, m, k, n);
    }
}

/// TurboQuant compression ratio benchmark.
pub fn bench_turbo_quant_compression(dim: usize, target_bits: u8) -> (usize, usize, f32) {
    let config = TurboQuantConfig {
        target_bits,
        block_size: 128,
        use_qjl_correction: true,
        ..Default::default()
    };

    let key: Vec<f32> = (0..dim).map(|i| ((i % 31) as f32 - 15.0) * 0.1).collect();
    let value: Vec<f32> = (0..dim).map(|i| ((i % 23) as f32 - 11.0) * 0.1).collect();

    let original_size = dim * 2 * 4; // K + V, FP32
    let entry = compress_kv(&key, &value, &config, 0);
    let compressed_size = entry.key_quantized.len()
        + entry.value_quantized.len()
        + entry.key_scales.len() * 4
        + entry.value_scales.len() * 4
        + entry.key_zeros.len() * 4
        + entry.value_zeros.len() * 4
        + entry.key_qjl.len() * 4
        + entry.value_qjl.len() * 4;

    let ratio = original_size as f32 / compressed_size as f32;
    (original_size, compressed_size, ratio)
}

/// TurboQuant accuracy benchmark (reconstruction error).
pub fn bench_turbo_quant_accuracy(dim: usize, target_bits: u8) -> (f32, f32) {
    let config = TurboQuantConfig {
        target_bits,
        block_size: 128,
        use_qjl_correction: true,
        ..Default::default()
    };

    let key: Vec<f32> = (0..dim).map(|i| ((i % 31) as f32 - 15.0) * 0.1).collect();
    let value: Vec<f32> = (0..dim).map(|i| ((i % 23) as f32 - 11.0) * 0.1).collect();

    let entry = compress_kv(&key, &value, &config, 0);
    let (decompressed_key, decompressed_value) = decompress_kv(&entry, &config);

    // Mean Squared Error
    let key_mse: f32 = key.iter().zip(decompressed_key.iter())
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f32>() / dim as f32;

    let value_mse: f32 = value.iter().zip(decompressed_value.iter())
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f32>() / dim as f32;

    (key_mse, value_mse)
}

/// Memory estimation benchmark for different model configurations.
pub fn bench_model_memory_estimates() -> Vec<(String, u64, u64, bool)> {
    let caps_k1 = HardwareCaps::spacemit_k1(8);
    let caps_k3 = HardwareCaps::spacemit_k3(32);

    let models = ai_runtime::ModelRegistry::all_models();
    let mut results = Vec::new();

    for model in &models {
        let ram = model.estimated_ram_bytes();
        let fits_k1 = model.fits_in_ram(caps_k1.available_ram);
        let fits_k3 = model.fits_in_ram(caps_k3.available_ram);
        results.push((
            model.name.clone(),
            ram as u64,
            model.params_billions as u64,
            fits_k1,
        ));
        results.push((
            model.name.clone(),
            ram as u64,
            model.params_billions as u64,
            fits_k3,
        ));
    }
    results
}

// ---------------------------------------------------------------------------
// Tests (run as validation)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matmul_correctness() {
        let (a, b, mut c_scalar) = prepare_matmul_data(4, 4, 4);
        let mut c_rvv = vec![0.0f32; 16];

        run_matmul_benchmark(&a, &b, &mut c_scalar, 4, 4, 4, false);
        run_matmul_benchmark(&a, &b, &mut c_rvv, 4, 4, 4, true);

        for i in 0..16 {
            assert!((c_scalar[i] - c_rvv[i]).abs() < 1e-3,
                "Mismatch at index {}: scalar={}, rvv={}",
                i, c_scalar[i], c_rvv[i]);
        }
    }

    #[test]
    fn test_turbo_quant_compression_ratio() {
        let (orig, compressed, ratio) = bench_turbo_quant_compression(4096, 3);
        assert!(ratio > 1.0, "Compression ratio should be > 1.0, got {}", ratio);
        assert!(compressed < orig, "Compressed should be smaller");
    }

    #[test]
    fn test_turbo_quant_8bit_accuracy() {
        let (key_mse, value_mse) = bench_turbo_quant_accuracy(256, 8);
        // 8-bit should have very low reconstruction error
        assert!(key_mse < 1.0, "Key MSE too high: {}", key_mse);
        assert!(value_mse < 1.0, "Value MSE too high: {}", value_mse);
    }
}
