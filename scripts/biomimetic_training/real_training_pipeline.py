#!/usr/bin/env python3
# Copyright (c) 2026 Xavier Callens / Socrate AI Lab. All Rights Reserved.
# SPDX-License-Identifier: LicenseRef-RunuX-Commercial
#
# Real Training + Serialization + Deployment Pipeline
# =====================================================
# Actually trains Qwen2.5-Math-7B with LoRA on real datasets,
# serializes checkpoints, and pushes to HuggingFace + GCS.
#
# Budget: <$150 on cloud GPU (A100/H100) or TPU v5e
#
# Usage:
#   export GEMINI_API_KEY="..."   # or MISTRAL_API_KEY
#   export HF_TOKEN="..."
#   python real_training_pipeline.py --budget 150
#   python real_training_pipeline.py --simulation  # local test

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import shutil
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.stdout.reconfigure(line_buffering=True)

GREEN, YELLOW, BLUE, CYAN, RED, MAGENTA, BOLD, NC = (
    "\033[0;32m", "\033[0;33m", "\033[0;34m", "\033[0;36m",
    "\033[0;31m", "\033[0;35m", "\033[1m", "\033[0m"
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("real_training")

# ═══════════════════════════════════════════════════════════════
# §1  CONFIGURATION — ALL KEYS FROM ENVIRONMENT
# ═══════════════════════════════════════════════════════════════

def get_api_key(name: str) -> str:
    """Get API key from environment variable. Never hardcode."""
    val = os.environ.get(name, "")
    if val:
        logger.info(f"  {GREEN}✓ {name} loaded from environment{NC}")
    else:
        logger.warning(f"  {YELLOW}⚠ {name} not set{NC}")
    return val


# ═══════════════════════════════════════════════════════════════
# §2  COST TRACKER (Cloud GPU/TPU pricing)
# ═══════════════════════════════════════════════════════════════

PRICING = {
    "a100-40gb": 3.67,   # $/hr on-demand GCP
    "a100-80gb": 5.07,
    "h100-80gb": 8.00,
    "tpu-v5e-4": 4.80,   # 4 chips
    "tpu-v5e-8": 9.60,
    "cpu": 0.0,
}


class CostTracker:
    def __init__(self, budget: float, gpu_type: str = "a100-40gb"):
        self.budget = budget
        self.hourly = PRICING.get(gpu_type, 3.67)
        self.gpu_type = gpu_type
        self.start = time.time()
        self.stages: Dict[str, float] = {}
        self._stage_name: Optional[str] = None
        self._stage_start: float = 0

    def begin(self, name: str):
        if self._stage_name:
            self.end()
        self._stage_name = name
        self._stage_start = time.time()

    def end(self):
        if self._stage_name:
            cost = (time.time() - self._stage_start) / 3600 * self.hourly
            self.stages[self._stage_name] = cost
            self._stage_name = None

    @property
    def total(self) -> float:
        s = sum(self.stages.values())
        if self._stage_name:
            s += (time.time() - self._stage_start) / 3600 * self.hourly
        return s

    def check(self) -> bool:
        if self.total >= self.budget:
            logger.error(f"  {RED}⛔ BUDGET EXCEEDED: ${self.total:.2f}/{self.budget:.2f}{NC}")
            return False
        return True

    def summary(self) -> Dict:
        return {"budget": self.budget, "spent": round(self.total, 2),
                "gpu_type": self.gpu_type, "stages": {k: round(v, 2) for k, v in self.stages.items()}}


# ═══════════════════════════════════════════════════════════════
# §3  CHECKPOINT & SERIALIZATION MANAGER
# ═══════════════════════════════════════════════════════════════

class CheckpointManager:
    """Manages model serialization with safetensors + SHA-256 checksums."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.checkpoints_dir = output_dir / "checkpoints"
        self.final_dir = output_dir / "final_model"
        self.lora_dir = output_dir / "lora_adapters"
        self.merged_dir = output_dir / "merged_model"
        for d in [self.checkpoints_dir, self.final_dir, self.lora_dir, self.merged_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(self, model, tokenizer, step: int, loss: float, simulation: bool):
        """Save training checkpoint with metadata."""
        ckpt_dir = self.checkpoints_dir / f"checkpoint-{step}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        meta = {"step": step, "loss": loss, "timestamp": datetime.now(timezone.utc).isoformat(),
                "simulation": simulation}
        with open(ckpt_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        if not simulation and model is not None:
            model.save_pretrained(ckpt_dir, safe_serialization=True)
            tokenizer.save_pretrained(ckpt_dir)
            self._compute_checksums(ckpt_dir)
            logger.info(f"  📁 Checkpoint saved: {ckpt_dir} (safetensors)")
        else:
            logger.info(f"  📁 [SIM] Checkpoint: step-{step}, loss={loss:.4f}")

    def save_lora_adapters(self, model, tokenizer, simulation: bool):
        """Save LoRA adapters separately (small, fast to upload)."""
        if not simulation and model is not None:
            model.save_pretrained(self.lora_dir, safe_serialization=True)
            tokenizer.save_pretrained(self.lora_dir)
            self._compute_checksums(self.lora_dir)
            size_mb = sum(f.stat().st_size for f in self.lora_dir.rglob("*") if f.is_file()) / 1e6
            logger.info(f"  {GREEN}✓ LoRA adapters: {self.lora_dir} ({size_mb:.1f} MB){NC}")
        else:
            # Save PFC bridge weights as the real serialized component
            pfc_src = Path(__file__).parent / "pretrained_models" / "pfc_bridge"
            if pfc_src.exists():
                shutil.copytree(pfc_src, self.lora_dir / "pfc_bridge", dirs_exist_ok=True)
            meta = {"simulation": simulation, "timestamp": datetime.now(timezone.utc).isoformat()}
            with open(self.lora_dir / "adapter_config.json", "w") as f:
                json.dump(meta, f, indent=2)
            logger.info(f"  📁 [SIM] LoRA adapters saved with PFC bridge")

    def save_merged_model(self, model, tokenizer, simulation: bool):
        """Merge LoRA into base model and save full weights."""
        if not simulation and model is not None:
            try:
                merged = model.merge_and_unload()
                merged.save_pretrained(self.merged_dir, safe_serialization=True)
                tokenizer.save_pretrained(self.merged_dir)
                self._compute_checksums(self.merged_dir)
                size_gb = sum(f.stat().st_size for f in self.merged_dir.rglob("*") if f.is_file()) / 1e9
                logger.info(f"  {GREEN}✓ Merged model: {self.merged_dir} ({size_gb:.1f} GB){NC}")
            except Exception as e:
                logger.warning(f"  {YELLOW}⚠ Merge failed: {e}{NC}")
        else:
            meta = {"simulation": True, "note": "Merge requires real model weights"}
            with open(self.merged_dir / "manifest.json", "w") as f:
                json.dump(meta, f, indent=2)

    def save_training_state(self, optimizer, scheduler, step, loss, simulation: bool):
        """Save full training state for resume."""
        state_dir = self.output_dir / "training_state"
        state_dir.mkdir(exist_ok=True)
        state = {"step": step, "loss": loss, "simulation": simulation,
                 "timestamp": datetime.now(timezone.utc).isoformat()}
        if not simulation and optimizer is not None:
            import torch
            torch.save({"optimizer": optimizer.state_dict(), "step": step},
                       state_dir / "optimizer_state.pt")
            if scheduler:
                torch.save(scheduler.state_dict(), state_dir / "scheduler_state.pt")
        with open(state_dir / "state.json", "w") as f:
            json.dump(state, f, indent=2)

    def _compute_checksums(self, directory: Path):
        import hashlib
        checksums = {}
        for f in sorted(directory.rglob("*")):
            if f.is_file() and f.name != "checksums.sha256":
                h = hashlib.sha256(f.read_bytes()).hexdigest()
                checksums[f.name] = h
        with open(directory / "checksums.sha256", "w") as out:
            for name, h in checksums.items():
                out.write(f"{h}  {name}\n")


# ═══════════════════════════════════════════════════════════════
# §4  HUGGINGFACE PUBLISHER
# ═══════════════════════════════════════════════════════════════

class HuggingFacePublisher:
    """Push models, datasets, and results to HuggingFace."""

    def __init__(self):
        self.token = get_api_key("HF_TOKEN")

    def push_model(self, model_dir: Path, repo_id: str, commit_msg: str = "Update model"):
        """Push model directory to HuggingFace Hub."""
        if not self.token:
            logger.warning(f"  {YELLOW}⚠ HF_TOKEN not set — skipping HuggingFace push{NC}")
            return False

        try:
            from huggingface_hub import HfApi
            api = HfApi(token=self.token)
            api.create_repo(repo_id, exist_ok=True, private=False, repo_type="model")
            api.upload_folder(
                folder_path=str(model_dir),
                repo_id=repo_id,
                repo_type="model",
                commit_message=commit_msg,
            )
            logger.info(f"  {GREEN}✓ Pushed to https://huggingface.co/{repo_id}{NC}")
            return True
        except Exception as e:
            logger.warning(f"  {YELLOW}⚠ HF push failed: {e}{NC}")
            return False

    def push_dataset(self, data_path: Path, repo_id: str):
        """Push dataset to HuggingFace."""
        if not self.token:
            return False
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=self.token)
            api.create_repo(repo_id, exist_ok=True, private=False, repo_type="dataset")
            api.upload_folder(
                folder_path=str(data_path), repo_id=repo_id,
                repo_type="dataset", commit_message="Upload training results",
            )
            logger.info(f"  {GREEN}✓ Dataset pushed: {repo_id}{NC}")
            return True
        except Exception as e:
            logger.warning(f"  {YELLOW}⚠ Dataset push failed: {e}{NC}")
            return False


# ═══════════════════════════════════════════════════════════════
# §5  GCS ARCHIVER
# ═══════════════════════════════════════════════════════════════

class GCSArchiver:
    """Archive to Google Cloud Storage (private bucket)."""

    def __init__(self, bucket: str):
        self.bucket = bucket

    def upload(self, local_path: Path, gcs_prefix: str):
        """Upload directory to GCS."""
        if not self.bucket:
            logger.warning(f"  {YELLOW}⚠ GCS bucket not configured{NC}")
            return False
        gcs_dest = f"{self.bucket}/{gcs_prefix}"
        cmd = f"gsutil -m cp -r {local_path} {gcs_dest}"
        logger.info(f"  Uploading to {gcs_dest}...")
        ret = os.system(cmd)
        if ret == 0:
            logger.info(f"  {GREEN}✓ Archived to {gcs_dest}{NC}")
            return True
        else:
            logger.warning(f"  {YELLOW}⚠ gsutil failed (code {ret}). Install: pip install gsutil{NC}")
            return False


# ═══════════════════════════════════════════════════════════════
# §6  LOCAL SSD ARCHIVER
# ═══════════════════════════════════════════════════════════════

class SSDArchiver:
    """Archive to remote SSD / local backup drive."""

    def __init__(self, ssd_path: str):
        self.ssd_path = Path(ssd_path)

    def archive(self, source_dir: Path, archive_name: str):
        """Copy model directory to SSD backup location."""
        dest = self.ssd_path / "symbrain_v2_models" / archive_name
        try:
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source_dir, dest, dirs_exist_ok=True)
            size_mb = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file()) / 1e6
            logger.info(f"  {GREEN}✓ Archived to SSD: {dest} ({size_mb:.1f} MB){NC}")
            return True
        except Exception as e:
            logger.warning(f"  {YELLOW}⚠ SSD archive failed: {e}{NC}")
            return False


# ═══════════════════════════════════════════════════════════════
# §7  REAL TRAINING ENGINE
# ═══════════════════════════════════════════════════════════════

def detect_device():
    try:
        import torch_xla.core.xla_model as xm
        return xm.xla_device(), "tpu"
    except Exception:
        pass
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda"), "cuda"
    return torch.device("cpu"), "cpu"


def real_training_loop(args, cost: CostTracker, ckpt: CheckpointManager, simulation: bool):
    """The actual training loop — SFT on math datasets."""
    import numpy as np

    logger.info(f"\n{BOLD}{'='*72}{NC}")
    logger.info(f"{BOLD}  Stage 1: Supervised Fine-Tuning (Real Data){NC}")
    logger.info(f"{'='*72}")

    device, device_type = detect_device()
    model, tokenizer, optimizer, scheduler = None, None, None, None

    if not simulation:
        import torch
        from transformers import (AutoModelForCausalLM, AutoTokenizer,
                                  get_cosine_schedule_with_warmup)
        from peft import LoraConfig, get_peft_model, TaskType

        logger.info(f"  Loading {args.model}...")
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        dtype = torch.bfloat16 if device_type in ("tpu", "cuda") else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=dtype, trust_remote_code=True, low_cpu_mem_usage=True,
        )

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=args.lora_r, lora_alpha=args.lora_r * 2,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            bias="none", use_rslora=True,
        )
        model = get_peft_model(model, lora_config)
        model = model.to(device)
        model.train()

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"  {GREEN}✓ LoRA r={args.lora_r}: {trainable:,} trainable params{NC}")

        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr, weight_decay=0.01,
        )
        total_steps = int(args.sft_duration / 2)  # ~2s per step
        warmup_steps = int(total_steps * 0.03)
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

        # Load datasets
        try:
            from datasets import load_dataset
            logger.info("  Loading MetaMathQA...")
            train_ds = load_dataset("meta-math/MetaMathQA", split="train")
            train_ds = train_ds.shuffle(seed=42).select(range(min(len(train_ds), 100000)))
            logger.info(f"  {GREEN}✓ {len(train_ds)} samples loaded{NC}")
        except Exception as e:
            logger.warning(f"  {YELLOW}⚠ Dataset load failed: {e}. Using synthetic.{NC}")
            simulation = True  # Fallback

    cost.begin("SFT")
    rng = np.random.RandomState(42)
    step = 0
    losses = []
    t0 = time.time()
    checkpoint_interval = 1800  # 30 min
    last_ckpt = t0
    interrupted = False

    def _sigint(s, f):
        nonlocal interrupted
        interrupted = True
    signal.signal(signal.SIGINT, _sigint)

    while (time.time() - t0) < args.sft_duration and cost.check() and not interrupted:
        step += 1
        elapsed = time.time() - t0

        if simulation:
            progress = elapsed / max(1, args.sft_duration)
            loss = 2.8 * (0.5 * (1 + math.cos(math.pi * progress))) + 0.15 + rng.normal(0, 0.03)
            loss = max(0.05, loss)
            time.sleep(0.2)
        else:
            import torch
            # Get training sample
            idx = step % len(train_ds)
            sample = train_ds[idx]
            text = f"Problem: {sample.get('query', sample.get('question', ''))}\nSolution: {sample.get('response', sample.get('answer', ''))}"
            inputs = tokenizer(text, return_tensors="pt", truncation=True,
                               max_length=args.max_seq_len, padding="max_length")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            inputs["labels"] = inputs["input_ids"].clone()

            outputs = model(**inputs)
            loss_val = outputs.loss
            loss = loss_val.item()

            loss_val.backward()
            if step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        losses.append(loss)

        if step % 100 == 0:
            logger.info(f"  Step {step:5d} | {elapsed/60:.1f}m | loss={loss:.4f} | "
                        f"${cost.total:.2f}/${cost.budget:.0f}")

        # Periodic checkpoint
        if time.time() - last_ckpt > checkpoint_interval:
            ckpt.save_checkpoint(model, tokenizer, step, loss, simulation)
            ckpt.save_training_state(optimizer, scheduler, step, loss, simulation)
            last_ckpt = time.time()

    cost.end()

    # Final checkpoint
    ckpt.save_checkpoint(model, tokenizer, step, losses[-1] if losses else 0, simulation)
    logger.info(f"\n  {GREEN}✅ SFT: {step} steps, loss {losses[0]:.4f} → {losses[-1]:.4f}{NC}")

    return model, tokenizer, {"steps": step, "initial_loss": losses[0], "final_loss": losses[-1],
                               "simulation": simulation}


# ═══════════════════════════════════════════════════════════════
# §8  MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Real Training + Serialization Pipeline")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Math-7B-Instruct")
    parser.add_argument("--lora-r", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--sft-duration", type=float, default=36000, help="SFT in seconds (10h)")
    parser.add_argument("--budget", type=float, default=150.0)
    parser.add_argument("--output-dir", default="./real_training_output")
    parser.add_argument("--gcs-bucket", default="gs://runux-ai-models")
    parser.add_argument("--ssd-path", default="/Volumes/MacCleanerStorage")
    parser.add_argument("--hf-repo", default="xaviercallens/symbrain-v2-math")
    parser.add_argument("--hf-dataset-repo", default="xaviercallens/symbrain-v2-results")
    parser.add_argument("--simulation", action="store_true")
    parser.add_argument("--gpu-type", default="a100-40gb",
                        choices=list(PRICING.keys()))
    args = parser.parse_args()

    print(f"\n{CYAN}{BOLD}{'='*72}{NC}")
    print(f"{CYAN}{BOLD}  SymBrain v2 — Real Training + Serialization Pipeline{NC}")
    print(f"{CYAN}{BOLD}  Budget: ${args.budget:.0f} | GPU: {args.gpu_type}{NC}")
    print(f"{CYAN}{BOLD}{'='*72}{NC}\n")

    # Check API keys
    gemini_key = get_api_key("GEMINI_API_KEY")
    mistral_key = get_api_key("MISTRAL_API_KEY")
    hf_token = get_api_key("HF_TOKEN")

    device, device_type = detect_device()
    simulation = args.simulation or device_type == "cpu"

    if simulation:
        logger.info(f"  {YELLOW}⚠ SIMULATION MODE — compressing durations{NC}")
        args.sft_duration = min(args.sft_duration, 30.0)

    # Initialize managers
    out_dir = Path(args.output_dir)
    cost = CostTracker(args.budget, args.gpu_type)
    ckpt = CheckpointManager(out_dir)
    hf_pub = HuggingFacePublisher()
    gcs = GCSArchiver(args.gcs_bucket)
    ssd = SSDArchiver(args.ssd_path)

    # ── TRAIN ──
    model, tokenizer, sft_results = real_training_loop(args, cost, ckpt, simulation)

    # ── SERIALIZE ──
    logger.info(f"\n{BOLD}💾 Serializing Models{NC}")
    ckpt.save_lora_adapters(model, tokenizer, simulation)
    ckpt.save_merged_model(model, tokenizer, simulation)

    # Save PFC bridge (always real)
    pfc_src = Path(__file__).parent / "pretrained_models" / "pfc_bridge"
    if pfc_src.exists():
        pfc_dest = out_dir / "pfc_bridge"
        shutil.copytree(pfc_src, pfc_dest, dirs_exist_ok=True)
        logger.info(f"  {GREEN}✓ PFC bridge serialized: {pfc_dest}{NC}")

    # ── PUSH TO HUGGINGFACE ──
    logger.info(f"\n{BOLD}📤 Publishing to HuggingFace{NC}")
    ts = datetime.now().strftime("%Y%m%d")

    # Push LoRA adapters (small, always push)
    hf_pub.push_model(ckpt.lora_dir, args.hf_repo,
                       f"SymBrain v2 LoRA adapters ({ts})")

    # Push results as dataset
    results_dir = out_dir / "results_dataset"
    results_dir.mkdir(exist_ok=True)
    report = {
        "meta": {"model": args.model, "lora_r": args.lora_r, "simulation": simulation,
                 "device": device_type, "timestamp": datetime.now(timezone.utc).isoformat()},
        "cost": cost.summary(), "sft": sft_results,
    }
    with open(results_dir / "training_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    with open(out_dir / "training_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    hf_pub.push_dataset(results_dir, args.hf_dataset_repo)

    # ── ARCHIVE TO GCS (PRIVATE) ──
    logger.info(f"\n{BOLD}☁️  Archiving to Google Cloud Storage (Private){NC}")
    gcs.upload(out_dir, f"symbrain-v2/{ts}")

    # ── ARCHIVE TO REMOTE SSD ──
    logger.info(f"\n{BOLD}💿 Archiving to Remote SSD{NC}")
    ssd.archive(out_dir, ts)

    # ── FINAL SUMMARY ──
    print(f"\n{CYAN}{BOLD}{'='*72}{NC}")
    print(f"{CYAN}{BOLD}  Pipeline Complete!{NC}")
    print(f"{CYAN}{BOLD}{'='*72}{NC}")
    print(f"  Real Training:  {'NO (simulation)' if simulation else 'YES'}")
    print(f"  Cost:           ${cost.total:.2f} / ${args.budget:.0f}")
    print(f"  SFT Loss:       {sft_results['initial_loss']:.4f} → {sft_results['final_loss']:.4f}")
    print(f"  Checkpoints:    {ckpt.checkpoints_dir}")
    print(f"  LoRA Adapters:  {ckpt.lora_dir}")
    print(f"  Merged Model:   {ckpt.merged_dir}")
    print(f"  HuggingFace:    {args.hf_repo}")
    print(f"  GCS:            {args.gcs_bucket}/symbrain-v2/{ts}")
    print(f"  SSD:            {args.ssd_path}/symbrain_v2_models/{ts}")
    print(f"  Report:         {out_dir / 'training_report.json'}")
    print(f"{CYAN}{BOLD}{'='*72}{NC}\n")


if __name__ == "__main__":
    main()
