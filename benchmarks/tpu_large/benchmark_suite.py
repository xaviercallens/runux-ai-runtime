#!/usr/bin/env python3
"""
RunuX-AI — TPU v5e Real-World Benchmark Suite
Copyright (c) 2026 Xavier Callens / Socrate AI Lab
SPDX-License-Identifier: LicenseRef-RunuX-Commercial

Benchmark Suite for measuring real framework performance on Google TPU v5e.
Tests: PyTorch (torch_xla), TensorFlow/JAX, and collects JSON results.
Budget target: < $80 total. Estimated time: ~1-2 hours on preemptible TPU.
"""

import json
import time
import os
import subprocess
import sys
from datetime import datetime

# ============================================================================
# Configuration
# ============================================================================
MODELS = {
    "qwen2.5-0.5b": {
        "hf_id": "Qwen/Qwen2.5-0.5B",
        "params_b": 0.5,
        "hidden_dim": 896,
        "n_layers": 24,
    },
    "deepseek-r1-1.5b": {
        "hf_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "params_b": 1.5,
        "hidden_dim": 1536,
        "n_layers": 28,
    },
    "mistral-7b-v0.3": {
        "hf_id": "mistralai/Mistral-7B-v0.3",
        "params_b": 7.0,
        "hidden_dim": 4096,
        "n_layers": 32,
    },
    "gemma-2-9b": {
        "hf_id": "google/gemma-2-9b",
        "params_b": 9.0,
        "hidden_dim": 3584,
        "n_layers": 42,
    },
}

BATCH_SIZES = [1, 8, 32]
SEQ_LEN = 512
DECODE_TOKENS = 128
WARMUP_STEPS = 3
MEASURE_STEPS = 10
TPU_TDP_WATTS = 200.0  # TPU v5e TDP

# ============================================================================
# Utility
# ============================================================================
def get_timestamp():
    return datetime.utcnow().isoformat() + "Z"

def save_results(results, filename):
    """Save benchmark results to JSON."""
    output = {
        "benchmark": "RunuX-AI TPU v5e Multi-Framework Comparison",
        "version": "1.0.0",
        "author": "Xavier Callens / Socrate AI Lab",
        "timestamp": get_timestamp(),
        "hardware": {
            "accelerator": "TPU v5e-1",
            "tflops_bf16": 197.0,
            "hbm_gb": 16,
            "hbm_bw_gbps": 819.2,
            "tdp_watts": TPU_TDP_WATTS,
        },
        "config": {
            "seq_len": SEQ_LEN,
            "decode_tokens": DECODE_TOKENS,
            "warmup_steps": WARMUP_STEPS,
            "measure_steps": MEASURE_STEPS,
            "dtype": "bfloat16",
        },
        "results": results,
    }
    with open(filename, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {filename}")


# ============================================================================
# PyTorch / torch_xla benchmark
# ============================================================================
def benchmark_pytorch_xla():
    """Benchmark models using PyTorch with torch_xla on TPU."""
    print("\n" + "="*70)
    print("  PyTorch (torch_xla) — TPU v5e Benchmark")
    print("="*70)
    
    try:
        import torch
        import torch_xla
        import torch_xla.core.xla_model as xm
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        print(f"  [SKIP] PyTorch/torch_xla not available: {e}")
        return []
    
    device = xm.xla_device()
    results = []
    
    for model_name, model_config in MODELS.items():
        for batch_size in BATCH_SIZES:
            print(f"\n  [{model_name}] BS={batch_size} — Loading...", flush=True)
            
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_config["hf_id"], trust_remote_code=True
                )
                model = AutoModelForCausalLM.from_pretrained(
                    model_config["hf_id"],
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                ).to(device)
                model.eval()
                
                # Prepare input
                input_text = "The future of sustainable AI computing " * (SEQ_LEN // 8)
                inputs = tokenizer(
                    [input_text] * batch_size,
                    return_tensors="pt",
                    max_length=SEQ_LEN,
                    truncation=True,
                    padding="max_length",
                ).to(device)
                
                # Warmup
                print(f"  [{model_name}] BS={batch_size} — Warmup ({WARMUP_STEPS} steps)...", flush=True)
                for _ in range(WARMUP_STEPS):
                    with torch.no_grad():
                        _ = model.generate(
                            **inputs,
                            max_new_tokens=DECODE_TOKENS,
                            do_sample=False,
                        )
                    xm.mark_step()
                
                # Measure
                print(f"  [{model_name}] BS={batch_size} — Measuring ({MEASURE_STEPS} steps)...", flush=True)
                latencies = []
                for step in range(MEASURE_STEPS):
                    xm.mark_step()
                    t0 = time.perf_counter()
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=DECODE_TOKENS,
                            do_sample=False,
                        )
                    xm.mark_step()
                    t1 = time.perf_counter()
                    latencies.append(t1 - t0)
                    total_tokens = batch_size * DECODE_TOKENS
                    tps = total_tokens / (t1 - t0)
                    print(f"    Step {step+1}: {tps:.1f} tok/s ({(t1-t0)*1000:.1f} ms)", flush=True)
                
                # Compute metrics
                avg_latency = sum(latencies) / len(latencies)
                total_tokens = batch_size * DECODE_TOKENS
                avg_tps = total_tokens / avg_latency
                joules_per_tok = TPU_TDP_WATTS / avg_tps
                p50_latency = sorted(latencies)[len(latencies)//2]
                p99_latency = sorted(latencies)[int(len(latencies)*0.99)]
                
                result = {
                    "framework": "pytorch_xla",
                    "model": model_name,
                    "model_hf": model_config["hf_id"],
                    "params_b": model_config["params_b"],
                    "batch_size": batch_size,
                    "seq_len": SEQ_LEN,
                    "decode_tokens": DECODE_TOKENS,
                    "avg_throughput_tps": round(avg_tps, 2),
                    "avg_latency_ms": round(avg_latency * 1000, 2),
                    "p50_latency_ms": round(p50_latency * 1000, 2),
                    "p99_latency_ms": round(p99_latency * 1000, 2),
                    "joules_per_token": round(joules_per_tok, 4),
                    "dtype": "bfloat16",
                }
                results.append(result)
                print(f"  ✓ [{model_name}] BS={batch_size}: {avg_tps:.1f} tok/s, {joules_per_tok:.2f} J/tok")
                
                # Free memory
                del model, tokenizer, inputs, outputs
                import gc; gc.collect()
                
            except Exception as e:
                print(f"  ✗ [{model_name}] BS={batch_size}: ERROR — {e}")
                results.append({
                    "framework": "pytorch_xla",
                    "model": model_name,
                    "batch_size": batch_size,
                    "error": str(e),
                })
    
    return results


# ============================================================================
# JAX / Flax benchmark
# ============================================================================
def benchmark_jax():
    """Benchmark models using JAX/Flax on TPU."""
    print("\n" + "="*70)
    print("  JAX/Flax — TPU v5e Benchmark")
    print("="*70)
    
    try:
        import jax
        import jax.numpy as jnp
        from transformers import FlaxAutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        print(f"  [SKIP] JAX/Flax not available: {e}")
        return []
    
    print(f"  JAX devices: {jax.devices()}")
    results = []
    
    for model_name, model_config in MODELS.items():
        for batch_size in BATCH_SIZES:
            print(f"\n  [{model_name}] BS={batch_size} — Loading...", flush=True)
            
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_config["hf_id"], trust_remote_code=True
                )
                model = FlaxAutoModelForCausalLM.from_pretrained(
                    model_config["hf_id"],
                    dtype=jnp.bfloat16,
                    trust_remote_code=True,
                )
                
                # Prepare input
                input_text = "The future of sustainable AI computing " * (SEQ_LEN // 8)
                inputs = tokenizer(
                    [input_text] * batch_size,
                    return_tensors="jax",
                    max_length=SEQ_LEN,
                    truncation=True,
                    padding="max_length",
                )
                
                # Warmup
                print(f"  [{model_name}] BS={batch_size} — Warmup...", flush=True)
                for _ in range(WARMUP_STEPS):
                    _ = model.generate(
                        inputs["input_ids"],
                        max_new_tokens=DECODE_TOKENS,
                        do_sample=False,
                    )
                    jax.effects_barrier()
                
                # Measure
                print(f"  [{model_name}] BS={batch_size} — Measuring...", flush=True)
                latencies = []
                for step in range(MEASURE_STEPS):
                    jax.effects_barrier()
                    t0 = time.perf_counter()
                    outputs = model.generate(
                        inputs["input_ids"],
                        max_new_tokens=DECODE_TOKENS,
                        do_sample=False,
                    )
                    jax.effects_barrier()
                    t1 = time.perf_counter()
                    latencies.append(t1 - t0)
                    tps = batch_size * DECODE_TOKENS / (t1 - t0)
                    print(f"    Step {step+1}: {tps:.1f} tok/s", flush=True)
                
                avg_latency = sum(latencies) / len(latencies)
                avg_tps = batch_size * DECODE_TOKENS / avg_latency
                joules_per_tok = TPU_TDP_WATTS / avg_tps
                
                result = {
                    "framework": "jax_flax",
                    "model": model_name,
                    "model_hf": model_config["hf_id"],
                    "params_b": model_config["params_b"],
                    "batch_size": batch_size,
                    "seq_len": SEQ_LEN,
                    "decode_tokens": DECODE_TOKENS,
                    "avg_throughput_tps": round(avg_tps, 2),
                    "avg_latency_ms": round(avg_latency * 1000, 2),
                    "joules_per_token": round(joules_per_tok, 4),
                    "dtype": "bfloat16",
                }
                results.append(result)
                print(f"  ✓ [{model_name}] BS={batch_size}: {avg_tps:.1f} tok/s, {joules_per_tok:.2f} J/tok")
                
                del model, tokenizer, inputs
                import gc; gc.collect()
                
            except Exception as e:
                print(f"  ✗ [{model_name}] BS={batch_size}: ERROR — {e}")
                results.append({
                    "framework": "jax_flax",
                    "model": model_name,
                    "batch_size": batch_size,
                    "error": str(e),
                })
    
    return results


# ============================================================================
# Main
# ============================================================================
def main():
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  RunuX-AI — TPU v5e Real-World Multi-Framework Benchmark       ║")
    print("║  Copyright (c) 2026 Xavier Callens / Socrate AI Lab            ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print()
    print(f"  Timestamp: {get_timestamp()}")
    print(f"  Models: {', '.join(MODELS.keys())}")
    print(f"  Batch sizes: {BATCH_SIZES}")
    print(f"  Sequence length: {SEQ_LEN}, Decode tokens: {DECODE_TOKENS}")
    print()
    
    all_results = []
    
    # Phase 1: PyTorch / torch_xla
    pytorch_results = benchmark_pytorch_xla()
    all_results.extend(pytorch_results)
    save_results(all_results, "/tmp/runux_benchmark_pytorch.json")
    
    # Phase 2: JAX / Flax
    jax_results = benchmark_jax()
    all_results.extend(jax_results)
    save_results(all_results, "/tmp/runux_benchmark_all.json")
    
    # Summary
    print("\n" + "="*70)
    print("  BENCHMARK SUMMARY")
    print("="*70)
    successful = [r for r in all_results if "error" not in r]
    errors = [r for r in all_results if "error" in r]
    print(f"  Total runs: {len(all_results)}")
    print(f"  Successful: {len(successful)}")
    print(f"  Errors: {len(errors)}")
    
    if successful:
        print(f"\n  {'Framework':<16} {'Model':<20} {'BS':>4} {'tok/s':>10} {'J/tok':>8}")
        print("  " + "-"*62)
        for r in sorted(successful, key=lambda x: (x["model"], x["batch_size"])):
            print(f"  {r['framework']:<16} {r['model']:<20} {r['batch_size']:>4} "
                  f"{r['avg_throughput_tps']:>10.1f} {r['joules_per_token']:>8.4f}")
    
    print(f"\n  Results saved to /tmp/runux_benchmark_all.json")
    print(f"  Total cost estimate: <$5 (preemptible TPU v5e-1)")
    return all_results


if __name__ == "__main__":
    main()
