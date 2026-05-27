#!/usr/bin/env python3
# Copyright (c) 2026 Xavier Callens / Socrate AI Lab. All Rights Reserved.
# SPDX-License-Identifier: LicenseRef-RunuX-Commercial
#
# Gemini 2.5 Pro Deep Think — Peer Review & Training Monitor
# =============================================================
# Uses Gemini 2.5 Pro with extended thinking for:
#   1. Peer review of training results & implementation
#   2. Real-time interpretation of training metrics
#   3. Checkpoint validation & anomaly detection
#   4. Optimization recommendations
#
# Usage:
#   python gemini_peer_review.py --mode review    # Full implementation review
#   python gemini_peer_review.py --mode monitor   # Continuous 10-min monitoring
#   python gemini_peer_review.py --mode interpret # Interpret latest results

from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("gemini_peer_review")

BOLD, GREEN, CYAN, YELLOW, RED, NC = "\033[1m", "\033[0;32m", "\033[0;36m", "\033[0;33m", "\033[0;31m", "\033[0m"

# ═══════════════════════════════════════════════════════════════
# §1  GEMINI CLIENT — Deep Think via 2.5 Pro
# ═══════════════════════════════════════════════════════════════

class GeminiDeepThink:
    """Gemini 2.5 Pro with extended thinking for deep analysis."""

    def __init__(self):
        self.api_key = os.environ.get("GEMINI_API_KEY", "")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not set. Export it first.")

        # Try google-genai SDK first, fallback to REST
        self.client = None
        self.use_sdk = False
        try:
            from google import genai
            self.client = genai.Client(api_key=self.api_key)
            self.use_sdk = True
            logger.info(f"  {GREEN}✓ Gemini SDK initialized{NC}")
        except ImportError:
            logger.info(f"  {YELLOW}Using REST API (install google-genai for SDK){NC}")

        # Model selection: best available
        self.model = "gemini-2.5-pro"  # Deep thinking model
        self.fallback_model = "gemini-2.5-flash"
        logger.info(f"  Model: {self.model} (deep think)")

    def think(self, prompt: str, system_instruction: str = "",
              thinking_budget: int = 16384, max_output: int = 8192) -> Dict:
        """Send prompt with extended thinking enabled."""
        t0 = time.time()

        if self.use_sdk:
            return self._think_sdk(prompt, system_instruction, thinking_budget, max_output, t0)
        else:
            return self._think_rest(prompt, system_instruction, thinking_budget, max_output, t0)

    def _think_sdk(self, prompt, system_instruction, thinking_budget, max_output, t0):
        from google.genai import types

        config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(
                thinking_budget=thinking_budget,
            ),
            max_output_tokens=max_output,
            temperature=1.0,  # Required for thinking
        )
        if system_instruction:
            config.system_instruction = system_instruction

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=config,
            )

            thinking_text = ""
            response_text = ""
            for part in response.candidates[0].content.parts:
                if part.thought:
                    thinking_text += part.text
                else:
                    response_text += part.text

            return {
                "response": response_text,
                "thinking": thinking_text,
                "model": self.model,
                "latency_seconds": round(time.time() - t0, 2),
                "thinking_tokens": len(thinking_text.split()),
            }
        except Exception as e:
            logger.warning(f"  {YELLOW}2.5 Pro failed: {e}, trying Flash{NC}")
            try:
                response = self.client.models.generate_content(
                    model=self.fallback_model,
                    contents=prompt,
                    config=config,
                )
                response_text = ""
                thinking_text = ""
                for part in response.candidates[0].content.parts:
                    if part.thought:
                        thinking_text += part.text
                    else:
                        response_text += part.text
                return {
                    "response": response_text, "thinking": thinking_text,
                    "model": self.fallback_model,
                    "latency_seconds": round(time.time() - t0, 2),
                    "thinking_tokens": len(thinking_text.split()),
                }
            except Exception as e2:
                return {"response": f"Error: {e2}", "thinking": "", "model": "error",
                        "latency_seconds": round(time.time() - t0, 2), "thinking_tokens": 0}

    def _think_rest(self, prompt, system_instruction, thinking_budget, max_output, t0):
        import requests
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"

        body = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "thinkingConfig": {"thinkingBudget": thinking_budget},
                "maxOutputTokens": max_output,
                "temperature": 1.0,
            },
        }
        if system_instruction:
            body["systemInstruction"] = {"parts": [{"text": system_instruction}]}

        try:
            r = requests.post(url, json=body, timeout=120)
            if not r.ok:
                # Fallback to Flash
                url = url.replace(self.model, self.fallback_model)
                r = requests.post(url, json=body, timeout=120)

            data = r.json()
            parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
            thinking = ""
            response = ""
            for p in parts:
                if p.get("thought"):
                    thinking += p.get("text", "")
                else:
                    response += p.get("text", "")

            return {
                "response": response, "thinking": thinking,
                "model": self.model, "latency_seconds": round(time.time() - t0, 2),
                "thinking_tokens": len(thinking.split()),
            }
        except Exception as e:
            return {"response": f"Error: {e}", "thinking": "", "model": "error",
                    "latency_seconds": round(time.time() - t0, 2), "thinking_tokens": 0}


# ═══════════════════════════════════════════════════════════════
# §2  PEER REVIEWER
# ═══════════════════════════════════════════════════════════════

SYSTEM_INSTRUCTION = """You are a senior ML researcher conducting peer review.
You specialize in mathematical reasoning, neuro-symbolic AI, and efficient fine-tuning.
Be rigorous, constructive, and specific. Flag any concerns about:
- Statistical validity (confidence intervals, seed variance)
- Methodology (data contamination, leakage, benchmark integrity)
- Claims vs evidence (are results reproducible and fairly presented?)
- Efficiency claims (cost, compute, scalability)
Provide a structured review with: Summary, Strengths, Weaknesses, Questions, Recommendations."""


class PeerReviewer:
    """Gemini-powered peer review of SymBrain implementation."""

    def __init__(self, gemini: GeminiDeepThink, output_dir: Path):
        self.gemini = gemini
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.reviews: List[Dict] = []

    def review_implementation(self, code_files: List[Path]) -> Dict:
        """Full implementation peer review."""
        logger.info(f"\n{BOLD}{'='*72}{NC}")
        logger.info(f"{BOLD}  Gemini 2.5 Pro — Implementation Peer Review{NC}")
        logger.info(f"{'='*72}\n")

        # Gather code context
        code_context = ""
        for f in code_files:
            if f.exists():
                content = f.read_text()[:4000]  # Truncate for token limits
                code_context += f"\n### {f.name}\n```python\n{content}\n```\n"

        prompt = f"""Review this ML training implementation for a mathematical reasoning system.

The system (SymBrain v2) fine-tunes Qwen2.5-Math-7B with LoRA for GSM8K/MATH/Physics benchmarks.

Key claims:
- GSM8K: 88.50% (baseline 83.00%)
- MATH-500: 58.41% (baseline 52.00%)
- Physics: 56.09% (baseline 45.00%)
- Training cost: <$150 on TPU/GPU spot
- LoRA r=128 with RSLoRA across all linear layers

Implementation code:
{code_context}

Please provide a thorough peer review addressing:
1. Are the claimed results plausible for a 7B model with LoRA?
2. Is the training pipeline sound (data, hyperparameters, evaluation)?
3. Are there risks of data contamination or benchmark gaming?
4. Is the cost claim realistic?
5. What improvements would you recommend?"""

        result = self.gemini.think(prompt, SYSTEM_INSTRUCTION, thinking_budget=24576)

        review = {
            "type": "implementation_review",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": result["model"],
            "thinking_tokens": result["thinking_tokens"],
            "latency_seconds": result["latency_seconds"],
            "review": result["response"],
            "thinking_trace": result["thinking"][:2000],
        }
        self.reviews.append(review)
        self._save_review(review, "implementation_review")
        return review

    def review_results(self, results: Dict) -> Dict:
        """Review training results and metrics."""
        logger.info(f"\n{BOLD}  Reviewing Training Results...{NC}")

        prompt = f"""Analyze these training results from a mathematical reasoning model fine-tuning:

```json
{json.dumps(results, indent=2, default=str)}
```

Evaluate:
1. Loss curve: Is the training converging properly? Any signs of overfitting?
2. Benchmark scores: Are they consistent across seeds? Statistical significance?
3. Ablation contributions: Do they make scientific sense?
4. Cost efficiency: How does this compare to similar published work?
5. Red flags: Anything that looks too good to be true?

Provide specific, actionable recommendations for the next training iteration."""

        result = self.gemini.think(prompt, SYSTEM_INSTRUCTION, thinking_budget=16384)

        review = {
            "type": "results_review",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": result["model"],
            "review": result["response"],
            "thinking_trace": result["thinking"][:2000],
        }
        self.reviews.append(review)
        self._save_review(review, "results_review")
        return review

    def review_paper(self, paper_path: Path) -> Dict:
        """Peer review the paper draft."""
        logger.info(f"\n{BOLD}  Reviewing Paper Draft...{NC}")

        paper_text = paper_path.read_text()[:8000]

        prompt = f"""Conduct a rigorous peer review of this ML research paper:

{paper_text}

Review criteria (NeurIPS/ICML standard):
1. Novelty: Is the contribution significant and original?
2. Soundness: Are methods and experiments rigorous?
3. Clarity: Is the paper well-written and easy to follow?
4. Significance: Will this impact the community?
5. Reproducibility: Can others replicate the results?
6. Ethics: Any IP or ethical concerns?

Score each criterion 1-10 and provide an overall recommendation:
Accept / Weak Accept / Borderline / Weak Reject / Reject"""

        result = self.gemini.think(prompt, SYSTEM_INSTRUCTION, thinking_budget=24576)

        review = {
            "type": "paper_review",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": result["model"],
            "review": result["response"],
            "thinking_trace": result["thinking"][:2000],
        }
        self.reviews.append(review)
        self._save_review(review, "paper_review")
        return review

    def _save_review(self, review: Dict, name: str):
        path = self.output_dir / f"{name}_{datetime.now().strftime('%Y%m%dT%H%M%S')}.json"
        with open(path, "w") as f:
            json.dump(review, f, indent=2, default=str)

        # Also save readable markdown
        md_path = self.output_dir / f"{name}_{datetime.now().strftime('%Y%m%dT%H%M%S')}.md"
        with open(md_path, "w") as f:
            f.write(f"# {review['type'].replace('_', ' ').title()}\n\n")
            f.write(f"**Model**: {review['model']}  \n")
            f.write(f"**Date**: {review['timestamp']}  \n")
            f.write(f"**Thinking tokens**: {review.get('thinking_tokens', 'N/A')}  \n\n")
            f.write("---\n\n")
            f.write(review["review"])
        logger.info(f"  {GREEN}✓ Review saved: {md_path}{NC}")


# ═══════════════════════════════════════════════════════════════
# §3  TRAINING MONITOR (10-min intervals)
# ═══════════════════════════════════════════════════════════════

class TrainingMonitor:
    """Monitors training progress every 10 minutes."""

    def __init__(self, gemini: GeminiDeepThink, output_dir: Path,
                 gcs_bucket: str = "gs://symbrain-v2-models"):
        self.gemini = gemini
        self.output_dir = output_dir
        self.gcs_bucket = gcs_bucket
        self.history: List[Dict] = []
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def check_training_status(self) -> Dict:
        """Check current training status from GCS or local files."""
        status = {"timestamp": datetime.now(timezone.utc).isoformat()}

        # Check GCS for training outputs
        try:
            import subprocess
            result = subprocess.run(
                ["gsutil", "ls", "-l", f"{self.gcs_bucket}/training_output/"],
                capture_output=True, text=True, timeout=15
            )
            status["gcs_files"] = result.stdout.strip()
        except Exception:
            status["gcs_files"] = "N/A"

        # Check Cloud Build status
        try:
            result = subprocess.run(
                ["gcloud", "builds", "list", "--project=gen-lang-client-0625573011",
                 "--region=us-central1", "--limit=1", "--format=json"],
                capture_output=True, text=True, timeout=15
            )
            builds = json.loads(result.stdout) if result.stdout.strip() else []
            if builds:
                b = builds[0]
                status["build_status"] = b.get("status", "UNKNOWN")
                status["build_id"] = b.get("id", "N/A")[:12]
                status["build_duration"] = b.get("timing", {}).get("BUILD", {}).get("endTime", "running")
        except Exception:
            status["build_status"] = "UNKNOWN"

        # Check Vertex AI jobs
        try:
            result = subprocess.run(
                ["gcloud", "ai", "custom-jobs", "list",
                 "--project=gen-lang-client-0625573011", "--region=us-central1",
                 "--limit=1", "--format=json"],
                capture_output=True, text=True, timeout=15
            )
            jobs = json.loads(result.stdout) if result.stdout.strip() else []
            if jobs:
                j = jobs[0]
                status["job_state"] = j.get("state", "UNKNOWN")
                status["job_name"] = j.get("displayName", "N/A")
        except Exception:
            status["job_state"] = "UNKNOWN"

        # Check Cloud Run jobs
        try:
            result = subprocess.run(
                ["gcloud", "run", "jobs", "list",
                 "--project=gen-lang-client-0625573011", "--region=us-central1",
                 "--format=json"],
                capture_output=True, text=True, timeout=15
            )
            jobs = json.loads(result.stdout) if result.stdout.strip() else []
            for j in jobs:
                if "symbrain" in j.get("metadata", {}).get("name", "").lower():
                    status["cloudrun_job"] = j.get("metadata", {}).get("name", "N/A")
                    status["cloudrun_status"] = j.get("status", {}).get("conditions", [{}])[-1].get("type", "UNKNOWN")
        except Exception:
            pass

        # Check local checkpoints
        local_ckpt_dir = Path("real_training_output/checkpoints")
        if local_ckpt_dir.exists():
            ckpts = sorted(local_ckpt_dir.iterdir())
            if ckpts:
                latest = ckpts[-1]
                meta_file = latest / "metadata.json"
                if meta_file.exists():
                    meta = json.loads(meta_file.read_text())
                    status["latest_checkpoint"] = meta

        # Check SSD backup
        ssd_dir = Path("/Volumes/MacCleanerStorage/symbrain_v2_models")
        if ssd_dir.exists():
            runs = sorted(ssd_dir.iterdir())
            status["ssd_backups"] = len(runs)
            if runs:
                size_mb = sum(f.stat().st_size for f in runs[-1].rglob("*") if f.is_file()) / 1e6
                status["latest_ssd_backup_mb"] = round(size_mb, 1)

        return status

    def interpret_results(self, status: Dict) -> Dict:
        """Use Gemini to interpret training progress."""
        prompt = f"""Analyze this training status report and provide interpretation:

```json
{json.dumps(status, indent=2, default=str)}
```

Previous monitoring history ({len(self.history)} checks):
{json.dumps(self.history[-3:], indent=2, default=str) if self.history else 'First check'}

Please provide:
1. **Status Summary**: Is training running/complete/failed?
2. **Progress Assessment**: How far along is the pipeline?
3. **Health Indicators**: Any warning signs (loss spikes, OOM, stalled)?
4. **Checkpoint Status**: Are checkpoints being saved properly?
5. **Cost Estimate**: Estimated cost so far based on elapsed time
6. **Recommendations**: Next actions needed

Be concise but precise. Use ✅ ⚠️ ❌ indicators."""

        result = self.gemini.think(prompt,
            system_instruction="You are a ML training ops expert monitoring a fine-tuning job.",
            thinking_budget=8192)

        interpretation = {
            "timestamp": status["timestamp"],
            "status": status,
            "interpretation": result["response"],
            "model": result["model"],
        }
        self.history.append(interpretation)
        return interpretation

    def run_monitoring_loop(self, interval_seconds: int = 600, max_checks: int = 144):
        """Run continuous monitoring every 10 minutes."""
        logger.info(f"\n{CYAN}{BOLD}{'='*72}{NC}")
        logger.info(f"{CYAN}{BOLD}  Training Monitor — Every {interval_seconds//60} minutes{NC}")
        logger.info(f"{CYAN}{BOLD}{'='*72}{NC}\n")

        for i in range(max_checks):
            try:
                logger.info(f"\n{BOLD}📊 Check #{i+1} at {datetime.now().strftime('%H:%M:%S')}{NC}")

                status = self.check_training_status()
                interpretation = self.interpret_results(status)

                print(f"\n{interpretation['interpretation']}")

                # Save to log
                log_path = self.output_dir / f"monitor_{datetime.now().strftime('%Y%m%d')}.jsonl"
                with open(log_path, "a") as f:
                    f.write(json.dumps(interpretation, default=str) + "\n")

                # Check for completion
                if status.get("build_status") == "SUCCESS":
                    logger.info(f"\n  {GREEN}✅ Build completed! Proceeding to deploy...{NC}")
                elif status.get("build_status") == "FAILURE":
                    logger.info(f"\n  {RED}❌ Build failed! Check logs.{NC}")
                elif status.get("job_state") == "JOB_STATE_SUCCEEDED":
                    logger.info(f"\n  {GREEN}✅ Training job completed!{NC}")
                    break

                if i < max_checks - 1:
                    logger.info(f"\n  💤 Next check in {interval_seconds//60} minutes...")
                    time.sleep(interval_seconds)

            except KeyboardInterrupt:
                logger.info(f"\n  {YELLOW}Monitoring stopped by user{NC}")
                break
            except Exception as e:
                logger.warning(f"  {YELLOW}Monitor error: {e}{NC}")
                time.sleep(60)

        # Final summary
        self._save_summary()

    def _save_summary(self):
        summary_path = self.output_dir / "monitoring_summary.json"
        with open(summary_path, "w") as f:
            json.dump({
                "total_checks": len(self.history),
                "start": self.history[0]["timestamp"] if self.history else None,
                "end": self.history[-1]["timestamp"] if self.history else None,
                "history": self.history,
            }, f, indent=2, default=str)
        logger.info(f"  📄 Summary: {summary_path}")


# ═══════════════════════════════════════════════════════════════
# §4  MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Gemini 2.5 Pro Peer Review & Monitor")
    parser.add_argument("--mode", choices=["review", "monitor", "interpret", "paper", "all"],
                        default="all")
    parser.add_argument("--interval", type=int, default=600, help="Monitor interval (seconds)")
    parser.add_argument("--output-dir", default="./peer_review_output")
    parser.add_argument("--gcs-bucket", default="gs://symbrain-v2-models")
    args = parser.parse_args()

    print(f"\n{CYAN}{BOLD}{'='*72}{NC}")
    print(f"{CYAN}{BOLD}  Gemini 2.5 Pro — Deep Think Peer Review{NC}")
    print(f"{CYAN}{BOLD}{'='*72}{NC}\n")

    gemini = GeminiDeepThink()
    output_dir = Path(args.output_dir)
    reviewer = PeerReviewer(gemini, output_dir)
    monitor = TrainingMonitor(gemini, output_dir, args.gcs_bucket)

    script_dir = Path(__file__).parent

    if args.mode in ("review", "all"):
        # Review implementation
        code_files = [
            script_dir / "real_training_pipeline.py",
            script_dir / "neuro_symbolic_brain.py",
            script_dir / "wars_ci_dfa_bridge.py",
            script_dir / "inference_server.py",
        ]
        review = reviewer.review_implementation(code_files)
        print(f"\n{BOLD}{'='*72}{NC}")
        print(f"{BOLD}Implementation Review ({review['model']}){NC}")
        print(f"{'='*72}")
        print(review["review"])

    if args.mode in ("paper", "all"):
        paper_path = script_dir / "SYMBRAIN_V2_PAPER.md"
        if paper_path.exists():
            review = reviewer.review_paper(paper_path)
            print(f"\n{BOLD}{'='*72}{NC}")
            print(f"{BOLD}Paper Review ({review['model']}){NC}")
            print(f"{'='*72}")
            print(review["review"])

    if args.mode in ("interpret", "all"):
        status = monitor.check_training_status()
        interpretation = monitor.interpret_results(status)
        print(f"\n{BOLD}{'='*72}{NC}")
        print(f"{BOLD}Training Status Interpretation{NC}")
        print(f"{'='*72}")
        print(interpretation["interpretation"])

    if args.mode == "monitor":
        monitor.run_monitoring_loop(interval_seconds=args.interval)

    print(f"\n{GREEN}✓ All reviews saved to: {output_dir}{NC}\n")


if __name__ == "__main__":
    main()
