#!/usr/bin/env python3
"""
RunuX-AI TPU v5e Benchmark — Real Hardware Measurement
Copyright (c) 2026 Xavier Callens / Socrate AI Lab
SPDX-License-Identifier: Apache-2.0

Runs actual inference benchmarks on Google TPU v5e.
Measures: latency, throughput, energy efficiency.
"""
import json
import time
import sys
import os
import gc
from datetime import datetime

# Results collector
RESULTS = {
    "benchmark_id": "runux-tpu-v5e-real-hw",
    "timestamp": datetime.utcnow().isoformat() + "Z",
    "hardware": {
        "accelerator": "Google TPU v5e",
        "type": "v5litepod-1",
        "bf16_tflops": 197.0,
        "hbm_gb": 16,
        "hbm_bw_gbps": 819.2,
        "tdp_watts": 200,
    },
    "software": {},
    "models": [],
}

def get_system_info():
    """Collect system information."""
    import torch
    import torch_xla
    import transformers
    RESULTS["software"] = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_xla": torch_xla.__version__,
        "transformers": transformers.__version__,
    }
    print(f"Python {RESULTS['software']['python']}")
    print(f"PyTorch {RESULTS['software']['torch']}")
    print(f"torch_xla {RESULTS['software']['torch_xla']}")
    print(f"Transformers {RESULTS['software']['transformers']}")

def benchmark_model(model_name, model_id, num_warmup=3, num_runs=10):
    """Benchmark a single model on TPU."""
    import torch
    import torch_xla.core.xla_model as xm
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    print(f"\n{'='*60}")
    print(f"  Benchmarking: {model_name} ({model_id})")
    print(f"{'='*60}")
    
    device = xm.xla_device()
    result = {
        "name": model_name,
        "model_id": model_id,
        "status": "error",
    }
    
    try:
        # Load tokenizer
        print(f"  Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model in BF16
        print(f"  Loading model (BF16)...")
        t0 = time.time()
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        model = model.to(device)
        model.eval()
        xm.mark_step()
        load_time = time.time() - t0
        
        # Count parameters
        num_params = sum(p.numel() for p in model.parameters()) / 1e9
        print(f"  Parameters: {num_params:.2f}B")
        print(f"  Load time: {load_time:.1f}s")
        
        # Prepare input
        prompt = "The future of artificial intelligence in sustainable computing is"
        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        input_len = input_ids.shape[1]
        
        gen_kwargs = {
            "max_new_tokens": 128,
            "do_sample": False,
            "use_cache": True,
        }
        
        # ---- Prefill benchmark (BS=1) ----
        print(f"  Running prefill benchmark...")
        # Warmup
        for _ in range(num_warmup):
            with torch.no_grad():
                out = model(input_ids, attention_mask=attention_mask)
                xm.mark_step()
            del out
        
        prefill_times = []
        for _ in range(num_runs):
            xm.mark_step()
            t0 = time.perf_counter()
            with torch.no_grad():
                out = model(input_ids, attention_mask=attention_mask)
                xm.mark_step()
            t1 = time.perf_counter()
            prefill_times.append(t1 - t0)
            del out
        
        prefill_median = sorted(prefill_times)[len(prefill_times) // 2]
        print(f"  Prefill latency: {prefill_median*1000:.2f} ms (median)")
        
        # ---- Decode benchmark (BS=1, 128 tokens) ----
        print(f"  Running decode benchmark (128 tokens)...")
        for _ in range(num_warmup):
            with torch.no_grad():
                gen = model.generate(input_ids, attention_mask=attention_mask, **gen_kwargs)
                xm.mark_step()
            del gen
        
        decode_times = []
        decode_tokens_list = []
        for _ in range(num_runs):
            xm.mark_step()
            t0 = time.perf_counter()
            with torch.no_grad():
                gen = model.generate(input_ids, attention_mask=attention_mask, **gen_kwargs)
                xm.mark_step()
            t1 = time.perf_counter()
            new_tokens = gen.shape[1] - input_len
            decode_times.append(t1 - t0)
            decode_tokens_list.append(new_tokens)
            del gen
        
        decode_median = sorted(decode_times)[len(decode_times) // 2]
        avg_new_tokens = sum(decode_tokens_list) / len(decode_tokens_list)
        throughput = avg_new_tokens / decode_median
        energy_per_token = 200.0 / throughput  # Watts / (tokens/s) = J/token
        
        print(f"  Decode latency: {decode_median*1000:.1f} ms")
        print(f"  New tokens: {avg_new_tokens:.0f}")
        print(f"  Throughput: {throughput:.1f} tok/s")
        print(f"  Energy: {energy_per_token:.2f} J/tok")
        
        # ---- Batch=8 decode ----
        print(f"  Running BS=8 decode benchmark...")
        try:
            batch_input = input_ids.repeat(8, 1)
            batch_mask = attention_mask.repeat(8, 1)
            
            for _ in range(num_warmup):
                with torch.no_grad():
                    gen8 = model.generate(batch_input, attention_mask=batch_mask, **gen_kwargs)
                    xm.mark_step()
                del gen8
            
            bs8_times = []
            bs8_tokens = []
            for _ in range(num_runs):
                xm.mark_step()
                t0 = time.perf_counter()
                with torch.no_grad():
                    gen8 = model.generate(batch_input, attention_mask=batch_mask, **gen_kwargs)
                    xm.mark_step()
                t1 = time.perf_counter()
                bs8_new = gen8.shape[1] - input_len
                bs8_times.append(t1 - t0)
                bs8_tokens.append(bs8_new * 8)  # total tokens across batch
                del gen8
            
            bs8_median = sorted(bs8_times)[len(bs8_times) // 2]
            bs8_avg_tokens = sum(bs8_tokens) / len(bs8_tokens)
            bs8_throughput = bs8_avg_tokens / bs8_median
            bs8_energy = 200.0 / bs8_throughput
            
            print(f"  BS=8 throughput: {bs8_throughput:.1f} tok/s")
            print(f"  BS=8 energy: {bs8_energy:.3f} J/tok")
            del batch_input, batch_mask
        except Exception as e:
            print(f"  BS=8 failed (OOM?): {e}")
            bs8_throughput = None
            bs8_energy = None
            bs8_median = None
        
        result = {
            "name": model_name,
            "model_id": model_id,
            "params_b": round(num_params, 2),
            "load_time_s": round(load_time, 1),
            "status": "success",
            "prefill_bs1": {
                "latency_ms": round(prefill_median * 1000, 2),
                "all_latencies_ms": [round(t*1000, 2) for t in prefill_times],
            },
            "decode_bs1": {
                "latency_ms": round(decode_median * 1000, 1),
                "tokens_generated": round(avg_new_tokens),
                "throughput_tps": round(throughput, 1),
                "energy_j_per_tok": round(energy_per_token, 3),
                "all_latencies_ms": [round(t*1000, 1) for t in decode_times],
            },
            "decode_bs8": {
                "throughput_tps": round(bs8_throughput, 1) if bs8_throughput else None,
                "energy_j_per_tok": round(bs8_energy, 3) if bs8_energy else None,
                "latency_ms": round(bs8_median * 1000, 1) if bs8_median else None,
            },
        }
        
        # Clean up
        del model, tokenizer, input_ids, attention_mask
        gc.collect()
        
    except Exception as e:
        print(f"  ERROR: {e}")
        result["error"] = str(e)
    
    RESULTS["models"].append(result)
    return result

def main():
    print("="*60)
    print("  RunuX-AI TPU v5e Benchmark Suite")
    print("  Xavier Callens / Socrate AI Lab")
    print("="*60)
    
    get_system_info()
    
    # Models that fit on 16GB HBM (BF16)
    models = [
        ("Qwen 2.5 0.5B", "Qwen/Qwen2.5-0.5B"),
        ("DeepSeek R1 Distill 1.5B", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"),
        ("Gemma 2 2B", "google/gemma-2-2b"),
    ]
    
    total_start = time.time()
    
    for name, model_id in models:
        try:
            benchmark_model(name, model_id)
        except Exception as e:
            print(f"FATAL ERROR with {name}: {e}")
            RESULTS["models"].append({
                "name": name,
                "model_id": model_id,
                "status": "fatal_error",
                "error": str(e),
            })
    
    total_time = time.time() - total_start
    RESULTS["total_runtime_s"] = round(total_time, 1)
    RESULTS["estimated_cost_usd"] = round(total_time / 3600 * 1.20, 2)
    
    # Save results
    output_path = os.path.expanduser("~/tpu_benchmark_results.json")
    with open(output_path, 'w') as f:
        json.dump(RESULTS, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"  BENCHMARK COMPLETE")
    print(f"  Total time: {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"  Est. cost: ${RESULTS['estimated_cost_usd']:.2f}")
    print(f"  Results: {output_path}")
    print(f"{'='*60}")
    
    # Print summary table
    print(f"\n{'Model':<30} {'TPS(BS1)':>10} {'J/tok':>8} {'TPS(BS8)':>10}")
    print("-" * 62)
    for m in RESULTS["models"]:
        if m["status"] == "success":
            bs8 = m["decode_bs8"]["throughput_tps"] or "N/A"
            print(f"{m['name']:<30} {m['decode_bs1']['throughput_tps']:>10.1f} {m['decode_bs1']['energy_j_per_tok']:>8.3f} {str(bs8):>10}")
        else:
            print(f"{m['name']:<30} {'ERROR':>10}")
    
    # Print JSON to stdout for capture
    print("\n=== RESULTS_JSON_START ===")
    print(json.dumps(RESULTS, indent=2))
    print("=== RESULTS_JSON_END ===")

if __name__ == "__main__":
    main()
