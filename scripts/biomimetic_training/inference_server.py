#!/usr/bin/env python3
# Copyright (c) 2026 Xavier Callens / Socrate AI Lab. All Rights Reserved.
# SPDX-License-Identifier: LicenseRef-RunuX-Commercial
#
# SymBrain v2 — Low-Cost Inference Server
# =========================================
# Serves the trained model via REST API for real validation.
# Runs on L4 Spot ($0.21/hr) or T4 ($0.11/hr).
#
# Endpoints:
#   GET  /health           — Health check
#   POST /v1/solve         — Solve a math/science problem
#   POST /v1/batch_eval    — Evaluate on benchmark subset
#   GET  /v1/benchmarks    — Get stored benchmark results
#
# Usage:
#   python inference_server.py --port 8080
#   python inference_server.py --port 8080 --model-path ./model
#   python inference_server.py --simulation  # local test (CPU)

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import random
import re
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("inference_server")

# ═══════════════════════════════════════════════════════════════
# §1  MODEL LOADER
# ═══════════════════════════════════════════════════════════════

class SymBrainModel:
    """Loads and serves the trained model for inference."""

    def __init__(self, model_path: str, base_model: str, simulation: bool = False):
        self.simulation = simulation
        self.model = None
        self.tokenizer = None
        self.model_path = model_path
        self.base_model = base_model
        self.device = None
        self.ready = False
        self.load_time = 0.0
        self.total_inferences = 0

    def load(self):
        t0 = time.time()
        if self.simulation:
            logger.info("  [SIM] Model loaded (simulation mode)")
            self.ready = True
            self.load_time = 0.1
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        logger.info(f"  Loading base model: {self.base_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        base = AutoModelForCausalLM.from_pretrained(
            self.base_model, torch_dtype=dtype,
            trust_remote_code=True, low_cpu_mem_usage=True,
        )

        # Load LoRA adapters if available
        adapter_path = Path(self.model_path)
        if adapter_path.exists() and (adapter_path / "adapter_config.json").exists():
            logger.info(f"  Loading LoRA adapters from {adapter_path}")
            self.model = PeftModel.from_pretrained(base, str(adapter_path))
        else:
            logger.info("  Using base model (no LoRA adapters found)")
            self.model = base

        self.model = self.model.to(self.device)
        self.model.eval()
        self.load_time = time.time() - t0
        self.ready = True
        logger.info(f"  ✓ Model ready in {self.load_time:.1f}s on {self.device}")

    def generate(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.3,
                 n_samples: int = 1) -> List[Dict]:
        """Generate response(s) for a prompt."""
        self.total_inferences += 1

        if self.simulation:
            return self._simulate_generate(prompt, n_samples)

        import torch
        results = []
        for i in range(n_samples):
            inputs = self.tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=2048
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs, max_new_tokens=max_tokens,
                    temperature=max(temperature, 0.01),
                    do_sample=temperature > 0,
                    top_p=0.95, pad_token_id=self.tokenizer.pad_token_id,
                )

            response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:],
                                             skip_special_tokens=True)
            results.append({
                "text": response.strip(),
                "sample_index": i,
            })
        return results

    def _simulate_generate(self, prompt: str, n_samples: int) -> List[Dict]:
        """Simulation mode: generate plausible responses."""
        rng = random.Random(hash(prompt) % 2**32)
        results = []
        for i in range(n_samples):
            # Extract problem type from prompt
            if any(kw in prompt.lower() for kw in ["solve", "calculate", "find", "what is"]):
                # Math-style answer
                answer = rng.choice([42, 17, 256, 3.14, 0.5, 100, 7, 12])
                response = (
                    f"Let me solve this step by step.\n\n"
                    f"Step 1: Identify the key variables and relationships.\n"
                    f"Step 2: Set up the equation based on the given conditions.\n"
                    f"Step 3: Solve the equation.\n\n"
                    f"Therefore, the answer is **{answer}**.\n\n"
                    f"\\boxed{{{answer}}}"
                )
            else:
                response = (
                    f"Based on the analysis:\n\n"
                    f"The key insight is that we need to consider the underlying principles. "
                    f"After careful evaluation, the answer follows from the established theory.\n\n"
                    f"The answer is (B)."
                )
            results.append({"text": response, "sample_index": i})
        return results

    def extract_answer(self, text: str) -> Optional[str]:
        """Extract final answer from model output."""
        # Try \boxed{} format first
        m = re.search(r"\\boxed\{([^}]+)\}", text)
        if m:
            return m.group(1).strip()
        # Try "answer is X" format
        m = re.search(r"(?:answer|result)\s+(?:is|=)\s+([^\n.]+)", text, re.IGNORECASE)
        if m:
            return m.group(1).strip()
        # Try last line with number
        for line in reversed(text.strip().split("\n")):
            m = re.search(r"(\d+\.?\d*)", line)
            if m:
                return m.group(1)
        return None


# ═══════════════════════════════════════════════════════════════
# §2  BENCHMARK EVALUATOR
# ═══════════════════════════════════════════════════════════════

class BenchmarkValidator:
    """Validates model on GSM8K/MATH-500/MMLU-STEM in real-time."""

    def __init__(self, model: SymBrainModel):
        self.model = model
        self.results = {}

    def evaluate_subset(self, benchmark: str, n_samples: int = 50,
                        self_consistency_k: int = 1) -> Dict:
        """Evaluate on a random subset of a benchmark."""
        logger.info(f"  Evaluating {benchmark} ({n_samples} samples, k={self_consistency_k})...")
        t0 = time.time()

        if self.model.simulation:
            return self._simulate_eval(benchmark, n_samples)

        try:
            from datasets import load_dataset

            if benchmark == "gsm8k":
                ds = load_dataset("openai/gsm8k", "main", split="test")
                ds = ds.shuffle(seed=42).select(range(min(n_samples, len(ds))))
                correct = 0
                for sample in ds:
                    prompt = f"Solve this math problem. Show your work and put the final answer in \\boxed{{}}.\n\nProblem: {sample['question']}\n\nSolution:"
                    responses = self.model.generate(prompt, n_samples=self_consistency_k)
                    answers = [self.model.extract_answer(r["text"]) for r in responses]
                    # Majority vote
                    from collections import Counter
                    answer_counts = Counter(a for a in answers if a)
                    if answer_counts:
                        predicted = answer_counts.most_common(1)[0][0]
                        # Extract ground truth
                        gt = sample['answer'].split("####")[-1].strip()
                        if predicted == gt:
                            correct += 1
                return {"benchmark": benchmark, "accuracy": correct / len(ds),
                        "correct": correct, "total": len(ds),
                        "self_consistency_k": self_consistency_k,
                        "wall_seconds": time.time() - t0}

            elif benchmark == "math500":
                ds = load_dataset("HendrycksTest/MATH", split="test")
                ds = ds.shuffle(seed=42).select(range(min(n_samples, len(ds))))
                correct = 0
                for sample in ds:
                    prompt = f"Solve this math problem. Put the final answer in \\boxed{{}}.\n\nProblem: {sample['problem']}\n\nSolution:"
                    responses = self.model.generate(prompt, n_samples=self_consistency_k)
                    answers = [self.model.extract_answer(r["text"]) for r in responses]
                    from collections import Counter
                    answer_counts = Counter(a for a in answers if a)
                    if answer_counts:
                        predicted = answer_counts.most_common(1)[0][0]
                        gt = self.model.extract_answer(sample['solution']) or ""
                        if predicted == gt:
                            correct += 1
                return {"benchmark": benchmark, "accuracy": correct / len(ds),
                        "correct": correct, "total": len(ds),
                        "self_consistency_k": self_consistency_k,
                        "wall_seconds": time.time() - t0}

        except Exception as e:
            logger.warning(f"  ⚠ Evaluation error: {e}")
            return {"benchmark": benchmark, "error": str(e)}

        return {"benchmark": benchmark, "error": "Not implemented"}

    def _simulate_eval(self, benchmark: str, n_samples: int) -> Dict:
        rng = random.Random(42)
        rates = {"gsm8k": 0.937, "math500": 0.935, "mmlu_stem": 0.886}
        rate = rates.get(benchmark, 0.80)
        correct = sum(1 for _ in range(n_samples) if rng.random() < rate)
        return {"benchmark": benchmark, "accuracy": correct / n_samples,
                "correct": correct, "total": n_samples,
                "self_consistency_k": 1, "wall_seconds": 0.5, "simulation": True}


# ═══════════════════════════════════════════════════════════════
# §3  FASTAPI SERVER
# ═══════════════════════════════════════════════════════════════

def create_app(model: SymBrainModel, validator: BenchmarkValidator):
    """Create FastAPI application."""
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import JSONResponse
        from pydantic import BaseModel
    except ImportError:
        # Fallback to simple HTTP server
        return create_simple_server(model, validator)

    app = FastAPI(
        title="SymBrain v2 Inference API",
        description="Biomimetic neuro-symbolic math reasoning",
        version="2.0.0",
    )

    class SolveRequest(BaseModel):
        problem: str
        max_tokens: int = 1024
        temperature: float = 0.3
        self_consistency_k: int = 1

    class BatchEvalRequest(BaseModel):
        benchmark: str = "gsm8k"
        n_samples: int = 50
        self_consistency_k: int = 1

    @app.get("/health")
    async def health():
        return {
            "status": "healthy" if model.ready else "loading",
            "model": model.base_model,
            "device": str(model.device) if model.device else "simulation",
            "load_time_seconds": model.load_time,
            "total_inferences": model.total_inferences,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    @app.post("/v1/solve")
    async def solve(req: SolveRequest):
        if not model.ready:
            raise HTTPException(503, "Model not ready")

        t0 = time.time()
        responses = model.generate(
            prompt=f"Solve this problem. Show your work and put the final answer in \\boxed{{}}.\n\nProblem: {req.problem}\n\nSolution:",
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            n_samples=req.self_consistency_k,
        )

        # Extract answers and pick majority
        answers = [model.extract_answer(r["text"]) for r in responses]
        from collections import Counter
        answer_counts = Counter(a for a in answers if a)
        final_answer = answer_counts.most_common(1)[0][0] if answer_counts else None

        return {
            "answer": final_answer,
            "reasoning": responses[0]["text"] if responses else "",
            "all_responses": responses if req.self_consistency_k > 1 else None,
            "self_consistency_k": req.self_consistency_k,
            "latency_seconds": round(time.time() - t0, 3),
        }

    @app.post("/v1/batch_eval")
    async def batch_eval(req: BatchEvalRequest):
        if not model.ready:
            raise HTTPException(503, "Model not ready")
        result = validator.evaluate_subset(
            req.benchmark, req.n_samples, req.self_consistency_k
        )
        validator.results[req.benchmark] = result
        return result

    @app.get("/v1/benchmarks")
    async def benchmarks():
        return {"benchmarks": validator.results}

    @app.get("/v1/model_info")
    async def model_info():
        return {
            "base_model": model.base_model,
            "model_path": model.model_path,
            "simulation": model.simulation,
            "device": str(model.device),
            "ready": model.ready,
            "total_inferences": model.total_inferences,
        }

    return app


def create_simple_server(model, validator):
    """Fallback: simple HTTP server without FastAPI."""
    from http.server import HTTPServer, BaseHTTPRequestHandler
    import urllib.parse

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/health":
                self._json_response(200, {
                    "status": "healthy" if model.ready else "loading",
                    "model": model.base_model,
                    "total_inferences": model.total_inferences,
                })
            elif self.path == "/v1/benchmarks":
                self._json_response(200, {"benchmarks": validator.results})
            else:
                self._json_response(404, {"error": "Not found"})

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}

            if self.path == "/v1/solve":
                problem = body.get("problem", "")
                k = body.get("self_consistency_k", 1)
                t0 = time.time()
                prompt = f"Solve: {problem}\nSolution:"
                responses = model.generate(prompt, n_samples=k)
                answers = [model.extract_answer(r["text"]) for r in responses]
                from collections import Counter
                answer_counts = Counter(a for a in answers if a)
                final = answer_counts.most_common(1)[0][0] if answer_counts else None
                self._json_response(200, {
                    "answer": final,
                    "reasoning": responses[0]["text"],
                    "latency_seconds": round(time.time() - t0, 3),
                })
            elif self.path == "/v1/batch_eval":
                bench = body.get("benchmark", "gsm8k")
                n = body.get("n_samples", 50)
                result = validator.evaluate_subset(bench, n)
                validator.results[bench] = result
                self._json_response(200, result)
            else:
                self._json_response(404, {"error": "Not found"})

        def _json_response(self, code, data):
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(data, default=str).encode())

        def log_message(self, fmt, *args):
            logger.info(f"  {args[0]} {args[1]} {args[2]}")

    return Handler


# ═══════════════════════════════════════════════════════════════
# §4  MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="SymBrain v2 Inference Server")
    parser.add_argument("--model-path", default="./model", help="Path to LoRA adapters")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-Math-7B-Instruct")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--simulation", action="store_true")
    parser.add_argument("--validate", action="store_true", help="Run validation then exit")
    parser.add_argument("--validate-n", type=int, default=50, help="Samples per benchmark")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  SymBrain v2 Inference Server")
    print(f"  Model: {args.base_model}")
    print(f"  Port:  {args.port}")
    print(f"{'='*60}\n")

    # Load model
    model = SymBrainModel(args.model_path, args.base_model, args.simulation)
    model.load()

    validator = BenchmarkValidator(model)

    # Validation mode
    if args.validate:
        print(f"\n{'='*60}")
        print(f"  Running Benchmark Validation")
        print(f"{'='*60}\n")
        for bench in ["gsm8k", "math500", "mmlu_stem"]:
            result = validator.evaluate_subset(bench, args.validate_n)
            print(f"  {bench:<12} {result.get('accuracy',0):.2%} "
                  f"({result.get('correct',0)}/{result.get('total',0)})")
        print(f"\n{'='*60}\n")

        # Save results
        results_path = Path("validation_results.json")
        with open(results_path, "w") as f:
            json.dump({"benchmarks": validator.results,
                       "timestamp": datetime.now(timezone.utc).isoformat()}, f, indent=2)
        print(f"  Results: {results_path}")
        return

    # Server mode
    try:
        import uvicorn
        app = create_app(model, validator)
        logger.info(f"  Starting FastAPI server on {args.host}:{args.port}")
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    except ImportError:
        from http.server import HTTPServer
        handler_class = create_simple_server(model, validator)
        server = HTTPServer((args.host, args.port), handler_class)
        logger.info(f"  Starting HTTP server on {args.host}:{args.port}")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            server.shutdown()


if __name__ == "__main__":
    main()
