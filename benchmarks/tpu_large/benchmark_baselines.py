#!/usr/bin/env python3
"""
RunuX-AI — Baseline Framework Benchmark (Reproduction Script)
Copyright (c) 2026 Xavier Callens / Socrate AI Lab
SPDX-License-Identifier: Apache-2.0

This script benchmarks BASELINE frameworks only (PyTorch, JAX).
RunuX-AI results are provided separately as reference data.

Usage:
    # On TPU v5e:
    python benchmark_baselines.py
    
    # On CPU (methodology verification):
    python benchmark_baselines.py --device cpu --models qwen2.5-0.5b
    
    # Custom configuration:
    python benchmark_baselines.py --batch-sizes 1 8 --warmup 5 --measure 20
"""

import argparse
import json
import time
import os
import sys
from datetime import datetime

# ============================================================================
# Configuration
# ============================================================================
AVAILABLE_MODELS = {
    "qwen2.5-0.5b": {
        "hf_id": "Qwen/Qwen2.5-0.5B",
        "params_b": 0.5,
    },
    "deepseek-r1-1.5b": {
        "hf_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "params_b": 1.5,
    },
    "mistral-7b-v0.3": {
        "hf_id": "mistralai/Mistral-7B-v0.3",
        "params_b": 7.0,
    },
    "gemma-2-9b": {
        "hf_id": "google/gemma-2-9b",
        "params_b": 9.0,
    },
}

def detect_device():
    """Auto-detect the best available device."""
    try:
        import torch_xla.core.xla_model as xm
        return "tpu", xm.xla_device()
    except ImportError:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda", torch.device("cuda")
    except ImportError:
        pass
    import torch
    return "cpu", torch.device("cpu")


def benchmark_pytorch(model_config, batch_size, seq_len, decode_tokens,
                      warmup_steps, measure_steps, device_type, device):
    """Run PyTorch benchmark for a single model/batch_size configuration."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["hf_id"], trust_remote_code=True
    )
    
    dtype = torch.bfloat16 if device_type == "tpu" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_config["hf_id"],
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    model.eval()
    
    # Prepare input
    input_text = "The future of sustainable AI computing is " * (seq_len // 8)
    inputs = tokenizer(
        [input_text] * batch_size,
        return_tensors="pt",
        max_length=seq_len,
        truncation=True,
        padding="max_length",
    ).to(device)
    
    # Synchronize function
    if device_type == "tpu":
        import torch_xla.core.xla_model as xm
        sync = lambda: xm.mark_step()
    elif device_type == "cuda":
        sync = lambda: torch.cuda.synchronize()
    else:
        sync = lambda: None
    
    # Warmup
    for _ in range(warmup_steps):
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=decode_tokens, do_sample=False)
        sync()
    
    # Measure
    latencies = []
    for step in range(measure_steps):
        sync()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=decode_tokens, do_sample=False)
        sync()
        t1 = time.perf_counter()
        latencies.append(t1 - t0)
        total_tokens = batch_size * decode_tokens
        print(f"    Step {step+1}/{measure_steps}: {total_tokens/(t1-t0):.1f} tok/s "
              f"({(t1-t0)*1000:.1f} ms)", flush=True)
    
    # Cleanup
    del model, tokenizer, inputs
    import gc; gc.collect()
    
    avg_lat = sum(latencies) / len(latencies)
    med_lat = sorted(latencies)[len(latencies)//2]
    total_tokens = batch_size * decode_tokens
    
    return {
        "avg_throughput_tps": round(total_tokens / avg_lat, 2),
        "median_throughput_tps": round(total_tokens / med_lat, 2),
        "avg_latency_ms": round(avg_lat * 1000, 2),
        "median_latency_ms": round(med_lat * 1000, 2),
        "min_latency_ms": round(min(latencies) * 1000, 2),
        "max_latency_ms": round(max(latencies) * 1000, 2),
        "all_latencies_ms": [round(l * 1000, 2) for l in latencies],
    }


def main():
    parser = argparse.ArgumentParser(
        description="RunuX-AI Baseline Benchmark — Reproduction Script"
    )
    parser.add_argument("--device", choices=["auto", "tpu", "cuda", "cpu"],
                        default="auto", help="Device to run on")
    parser.add_argument("--models", nargs="+",
                        default=list(AVAILABLE_MODELS.keys()),
                        help="Models to benchmark")
    parser.add_argument("--batch-sizes", nargs="+", type=int,
                        default=[1, 8, 32], help="Batch sizes")
    parser.add_argument("--seq-len", type=int, default=512,
                        help="Input sequence length")
    parser.add_argument("--decode-tokens", type=int, default=128,
                        help="Number of tokens to decode")
    parser.add_argument("--warmup", type=int, default=3,
                        help="Warmup iterations")
    parser.add_argument("--measure", type=int, default=10,
                        help="Measurement iterations")
    parser.add_argument("--output", type=str, default="baseline_results.json",
                        help="Output JSON file")
    args = parser.parse_args()
    
    # Detect device
    if args.device == "auto":
        device_type, device = detect_device()
    else:
        device_type = args.device
        import torch
        if device_type == "tpu":
            import torch_xla.core.xla_model as xm
            device = xm.xla_device()
        elif device_type == "cuda":
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  RunuX-AI — Baseline Framework Benchmark (Reproduction)    ║")
    print("║  Apache-2.0 Licensed                                       ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"  Device: {device_type} ({device})")
    print(f"  Models: {args.models}")
    print(f"  Batch sizes: {args.batch_sizes}")
    print(f"  Config: seq={args.seq_len}, decode={args.decode_tokens}")
    print()
    
    results = []
    
    for model_name in args.models:
        if model_name not in AVAILABLE_MODELS:
            print(f"  [SKIP] Unknown model: {model_name}")
            continue
        
        model_config = AVAILABLE_MODELS[model_name]
        
        for batch_size in args.batch_sizes:
            print(f"\n  [{model_name}] BS={batch_size} — Starting...", flush=True)
            
            try:
                metrics = benchmark_pytorch(
                    model_config, batch_size, args.seq_len, args.decode_tokens,
                    args.warmup, args.measure, device_type, device
                )
                
                result = {
                    "framework": f"pytorch_{device_type}",
                    "model": model_name,
                    "model_hf": model_config["hf_id"],
                    "params_b": model_config["params_b"],
                    "batch_size": batch_size,
                    "seq_len": args.seq_len,
                    "decode_tokens": args.decode_tokens,
                    "dtype": "bfloat16" if device_type == "tpu" else "float16",
                    **metrics,
                }
                results.append(result)
                print(f"  ✓ [{model_name}] BS={batch_size}: "
                      f"{metrics['median_throughput_tps']:.1f} tok/s")
                
            except Exception as e:
                print(f"  ✗ [{model_name}] BS={batch_size}: {e}")
                results.append({
                    "framework": f"pytorch_{device_type}",
                    "model": model_name,
                    "batch_size": batch_size,
                    "error": str(e),
                })
    
    # Save
    output = {
        "benchmark": "RunuX-AI Baseline Reproduction",
        "version": "1.0.0",
        "license": "Apache-2.0",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "device": device_type,
        "config": {
            "seq_len": args.seq_len,
            "decode_tokens": args.decode_tokens,
            "warmup_steps": args.warmup,
            "measure_steps": args.measure,
        },
        "results": results,
    }
    
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    
    # Summary
    successful = [r for r in results if "error" not in r]
    print(f"\n  {'='*60}")
    print(f"  SUMMARY: {len(successful)}/{len(results)} successful")
    if successful:
        print(f"  {'Model':<20} {'BS':>4} {'tok/s':>10} {'Latency':>10}")
        for r in successful:
            print(f"  {r['model']:<20} {r['batch_size']:>4} "
                  f"{r['median_throughput_tps']:>10.1f} "
                  f"{r['median_latency_ms']:>10.1f}ms")
    print(f"\n  Results saved to: {args.output}")


if __name__ == "__main__":
    main()
