// Copyright (c) 2026 Xavier Callens / Socrate AI. All Rights Reserved.
// SPDX-License-Identifier: LicenseRef-RunuX-Commercial

//! RunuX AI Runtime — Simulation Report Generator
//!
//! Produces a comprehensive simulation report.
//! Requires at least 1GB free RAM to run training simulations.

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║       RunuX AI Runtime — Simulation Report v0.1.0          ║");
    println!("║       Copyright (c) 2026 Xavier Callens / Socrate AI       ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // ── Section 1: Hardware ──────────────────────────────────────────────
    println!("  1. HARDWARE SPECIFICATIONS");
    println!("  ──────────────────────────────────────────────────────");
    let k1 = perf_model::HardwareSpec::spacemit_k1();
    let k3 = perf_model::HardwareSpec::spacemit_k3();
    println!("  {:26} {:>14} {:>14}", "", "BPI-F3 (K1)", "AIBOX-K3 (K3)");
    println!("  {:26} {:>14} {:>14}", "CPU Cores", format!("{}x X60", k1.n_cores), format!("{}x X100", k3.n_cores));
    println!("  {:26} {:>14} {:>14}", "RVV VLEN (bits)", format!("{}", k1.vlen_bits), format!("{}", k3.vlen_bits));
    println!("  {:26} {:>14} {:>14}", "Peak FP32 GFLOPS", format!("{:.0}", k1.total_fp32_gflops()), format!("{:.0}", k3.total_fp32_gflops()));
    println!("  {:26} {:>14} {:>14}", "DRAM BW (GB/s)", format!("{:.1}", k1.dram_bw_gbs), format!("{:.1}", k3.dram_bw_gbs));
    let pk1 = power_monitor::PowerProfile::bpi_f3();
    let pk3 = power_monitor::PowerProfile::aibox_k3();
    println!("  {:26} {:>14} {:>14}", "TDP (W)", format!("{:.0}", pk1.tdp_watts), format!("{:.0}", pk3.tdp_watts));
    println!("  {:26} {:>14} {:>14}", "TOPS/W", format!("{:.2}", pk1.tops_per_watt()), format!("{:.2}", pk3.tops_per_watt()));
    println!();

    // ── Section 2: Model Compatibility ───────────────────────────────────
    println!("  2. MODEL x HARDWARE COMPATIBILITY");
    println!("  ──────────────────────────────────────────────────────");
    let matrix = perf_model::compatibility_matrix();
    println!("  {:26} {:>10} {:>6} {:>8}", "Model", "Hardware", "Fits", "tok/s");
    for e in &matrix {
        let hw = if e.hardware.contains("BPI") { "K1" }
            else if e.hardware.contains("8GB") { "K3-8G" }
            else { "K3-32G" };
        println!("  {:26} {:>10} {:>6} {:>8.1}", e.model, hw,
            if e.fits_in_ram { "YES" } else { "NO" }, e.tokens_per_second);
    }
    println!();

    // ── Section 3: Inference Pipeline ────────────────────────────────────
    println!("  3. INFERENCE SIMULATION");
    println!("  ──────────────────────────────────────────────────────");
    let config = sim_inference::SimConfig::tiny_test();
    let result = sim_inference::run_simulation(&config);
    println!("  Tokens generated:  {}", result.num_generated);
    println!("  FlashAttn savings: {:.1}x", result.flash_mem_savings);
    println!("  KV compression:    {:.1}x", result.kv_compression_ratio);
    println!("  Est. tok/s (K1):   {:.1}", result.estimated_tps);
    println!("  Est. J/token:      {:.4}", result.estimated_joules_per_tok);
    println!();

    // ── Section 4: LoRA Training (analytical) ────────────────────────────
    println!("  4. LoRA TRAINING ANALYSIS");
    println!("  ──────────────────────────────────────────────────────");
    let q = sim_train::LoraConfig::qwen_0_5b_edge();
    let d = sim_train::LoraConfig::deepseek_1_5b_edge();
    let k = sim_train::LoraConfig::qwen_7b_k3();
    println!("  {:26} {:>12} {:>12}", "Model", "Params", "Memory");
    println!("  {:26} {:>12} {:>12}", "Qwen 0.5B (r=8, 24 layers)",
        fmt_n(q.total_trainable_params()), fmt_b(q.training_memory_bytes()));
    println!("  {:26} {:>12} {:>12}", "DeepSeek 1.5B (r=4, 14 ly)",
        fmt_n(d.total_trainable_params()), fmt_b(d.training_memory_bytes()));
    println!("  {:26} {:>12} {:>12}", "Qwen 7B (r=16, 28 layers)",
        fmt_n(k.total_trainable_params()), fmt_b(k.training_memory_bytes()));
    println!();

    // Tiny convergence test
    let tiny = sim_train::LoraConfig {
        hidden_dim: 16, rank: 2, alpha: 4.0, dropout: 0.0,
        target_layers: vec![0, 1], n_layers: 2,
        learning_rate: 1e-3, weight_decay: 0.0,
        local_epochs: 3, batch_size: 4, grad_accum_steps: 1,
    };
    let tr = sim_train::simulate_lora_training(&tiny, "micro-sim", 40);
    println!("  Convergence test (d=16, r=2):");
    println!("    Steps: {}, Loss: {:.6} -> {:.6} ({:.0}% reduction)",
        tr.total_steps, tr.initial_loss, tr.final_loss, tr.loss_reduction);
    println!();

    // ── Section 5: Energy ────────────────────────────────────────────────
    println!("  5. ENERGY EFFICIENCY");
    println!("  ──────────────────────────────────────────────────────");
    let results = power_monitor::run_power_simulation();
    println!("  {:38} {:>7} {:>8} {:>8}", "Config", "tok/s", "J/tok", "Speedup");
    for r in &results {
        println!("  {:38} {:>7.1} {:>8.3} {:>7.1}x",
            r.config_name, r.tokens_per_second, r.joules_per_token, r.speedup_vs_baseline);
    }
    println!();

    let regions = [
        power_monitor::CarbonFactor::france(),
        power_monitor::CarbonFactor::china_avg(),
        power_monitor::CarbonFactor::us_avg(),
    ];
    println!("  {:16} {:>10} {:>12}", "Region", "gCO2/kWh", "gCO2/1K tok");
    for r in &regions {
        let est = power_monitor::estimate_token_energy(&pk1, 10.0, r);
        println!("  {:16} {:>10.0} {:>12.4}", r.region, r.g_co2_per_kwh, est.g_co2_per_1k_tokens);
    }
    println!();

    // ── Summary ──────────────────────────────────────────────────────────
    println!("  SUMMARY: 18 crates, ~10K lines no_std Rust");
    println!("  Hardware: SpacemiT K1/K3 | Models: Qwen/DeepSeek/Mistral");
    println!("  (c) 2026 Xavier Callens / Socrate AI. All rights reserved.");
}

fn fmt_b(b: usize) -> String {
    if b >= 1_048_576 { format!("{:.1} MB", b as f64 / 1_048_576.0) }
    else if b >= 1024 { format!("{:.1} KB", b as f64 / 1024.0) }
    else { format!("{} B", b) }
}

fn fmt_n(n: usize) -> String {
    if n >= 1_000_000 { format!("{:.2}M", n as f64 / 1_000_000.0) }
    else if n >= 1_000 { format!("{:.1}K", n as f64 / 1_000.0) }
    else { format!("{}", n) }
}
