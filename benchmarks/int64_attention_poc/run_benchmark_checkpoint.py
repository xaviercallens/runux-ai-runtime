#!/usr/bin/env python3
"""Checkpoint-aware runner for PoC 2 benchmark.

Saves progress incrementally to a checkpoint file, allowing recovery if
interrupted. Runs independently and can be monitored via tail/grep of
the log file.
"""
import os
import sys
import json
import time
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from benchmark_t4_poc2 import run_benchmark

CHECKPOINT_FILE = Path(__file__).parent / "checkpoint_poc2.json"
LOG_FILE = Path(__file__).parent / "benchmark_poc2_run.log"


def log(msg):
    """Write message to log with timestamp."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line, file=sys.stderr)
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")
        f.flush()
    sys.stdout.flush()
    sys.stderr.flush()


def load_checkpoint():
    """Load checkpoint if it exists."""
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return None


def save_checkpoint(data):
    """Save checkpoint atomically."""
    tmp = CHECKPOINT_FILE.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    tmp.replace(CHECKPOINT_FILE)
    log(f"Checkpoint saved: {CHECKPOINT_FILE}")


def main():
    log("="*70)
    log("PoC 2 INT64 Attention Benchmark (Checkpoint-aware)")
    log("="*70)
    log(f"Log: {LOG_FILE}")
    log(f"Checkpoint: {CHECKPOINT_FILE}")

    checkpoint = load_checkpoint()
    if checkpoint:
        log(f"Resuming from checkpoint (completed at {checkpoint.get('timestamp', 'unknown')})")
        log("Results already saved:")
        for key, val in checkpoint.items():
            if key != "timestamp":
                if isinstance(val, float):
                    log(f"  {key}: {val:.3f}")
                else:
                    log(f"  {key}: {val}")
        return

    log("Starting benchmark (this may take 5-10 minutes on T4)...")
    log("")

    try:
        start_time = time.time()
        log("Running benchmark...")
        results = run_benchmark()
        elapsed = time.time() - start_time

        log("")
        log("="*70)
        log("BENCHMARK COMPLETED SUCCESSFULLY")
        log("="*70)
        log(f"Total time: {elapsed:.1f}s")
        log("")

        # Save checkpoint with results
        checkpoint = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_seconds": elapsed,
        }
        checkpoint.update(results)
        save_checkpoint(checkpoint)

        log("")
        log("Results saved to checkpoint. Run again to display results.")
        log("")

    except Exception as e:
        log(f"ERROR: {type(e).__name__}: {e}")
        import traceback
        log(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
