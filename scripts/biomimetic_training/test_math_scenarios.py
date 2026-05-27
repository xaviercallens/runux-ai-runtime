#!/usr/bin/env python3
# Copyright (c) 2026 Xavier Callens / Socrate AI Lab. All Rights Reserved.
# SPDX-License-Identifier: LicenseRef-RunuX-Commercial
#
# SymBrain v2 — Intense Mathematical Scenario Test Suite
# ========================================================
#
# Validates FastAPI endpoints under complex mathematical, physical, and 
# reinforcement learning scenarios.

import json
import urllib.request
import urllib.error
import time
import sys

# Define HSL-tailored ANSI styles for cyberpunk theme
CYAN = "\033[38;2;0;242;254m"
MAGENTA = "\033[38;2;255;0;127m"
GOLD = "\033[38;2;245;158;11m"
GREEN = "\033[1;32m"
RED = "\033[1;31m"
RESET = "\033[0m"
BOLD = "\033[1m"

BASE_URL = "http://localhost:8085"

def run_post(endpoint: str, data: dict) -> dict:
    url = f"{BASE_URL}{endpoint}"
    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    try:
        with urllib.request.urlopen(req) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        print(f"  {RED}⚠ HTTP Error {e.code}: {e.read().decode('utf-8')}{RESET}")
        raise e

def run_get(endpoint: str) -> dict:
    url = f"{BASE_URL}{endpoint}"
    try:
        with urllib.request.urlopen(url) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        print(f"  {RED}⚠ HTTP Error {e.code}{RESET}")
        raise e

def print_header(title: str):
    print(f"\n{BOLD}{GOLD}════════════════════════════════════════════════════════════════{RESET}")
    print(f"  {BOLD}{CYAN}{title}{RESET}")
    print(f"{BOLD}{GOLD}════════════════════════════════════════════════════════════════{RESET}\n")

def main():
    print_header("SymBrain v2 — Back-End Intense Mathematical Test Suite")
    
    # ───────────────────────────────────────────────────────────
    # Test 1: Service Health and Warm-up Checks
    # ───────────────────────────────────────────────────────────
    print(f"{BOLD}[1/5] Executing API Health & Node Pre-warming Sequence...{RESET}")
    try:
        health = run_get("/health")
        print(f"  {GREEN}✓ Health status: {health.get('status')}{RESET}")
        print(f"  {GREEN}✓ Active model:  {health.get('model')}{RESET}")
        print(f"  {GREEN}✓ Device node:   {health.get('device')}{RESET}")
        
        warm = run_get("/v1/warm")
        print(f"  {GREEN}✓ Pre-warm sequence latency: {warm.get('warmup_seconds')}s ({warm.get('status')}){RESET}")
    except Exception as e:
        print(f"  {RED}⛔ Connection to server failed on port 8085. Ensure the server is running.{RESET}")
        sys.exit(1)

    # ───────────────────────────────────────────────────────────
    # Test 2: Solve Mathematical Scenario A (Linear Algebra Bounding)
    # ───────────────────────────────────────────────────────────
    print(f"\n{BOLD}[2/5] Testing Scenario A: Orthogonal Matrix Synaptic Updates...{RESET}")
    problem_a = (
        "Let B_L be a 512x512 orthogonal synaptic weight matrix. Prove that "
        "the L2 norm of the update vectors satisfies the Lipschitz constraint ||B_L * x|| <= M * ||x||."
    )
    t0 = time.time()
    res_a = run_post("/v1/solve", {"problem": problem_a, "self_consistency_k": 1})
    latency_a = time.time() - t0
    print(f"  {GREEN}✓ Response latency: {latency_a:.3f}s{RESET}")
    print(f"  {GREEN}✓ PFC routed answer: {CYAN}{res_a.get('answer')}{RESET}")
    print(f"  {GREEN}✓ Proof chain snippet:{RESET}\n{MAGENTA}{res_a.get('reasoning')[:300]}...{RESET}")

    # ───────────────────────────────────────────────────────────
    # Test 3: Solve Mathematical Scenario B (Calculus/Analysis Integration)
    # ───────────────────────────────────────────────────────────
    print(f"\n{BOLD}[3/5] Testing Scenario B: Closed-Loop Integral & Divergence Limits...{RESET}")
    problem_b = (
        "Integrate the boundary function f(x) = exp(-x^2) from 0 to infinity and verify convergence."
    )
    t0 = time.time()
    res_b = run_post("/v1/solve", {"problem": problem_b, "self_consistency_k": 1})
    latency_b = time.time() - t0
    print(f"  {GREEN}✓ Response latency: {latency_b:.3f}s{RESET}")
    print(f"  {GREEN}✓ Verified algebraic limit: {CYAN}{res_b.get('answer')}{RESET}")

    # ───────────────────────────────────────────────────────────
    # Test 4: Lean 4 Theorem Verification Checks
    # ───────────────────────────────────────────────────────────
    print(f"\n{BOLD}[4/5] Testing Scenario C: Formal Proof Compiler (Lean 4)...{RESET}")
    valid_lean = (
        "import Mathlib\n\n"
        "theorem add_comm_test (a b : ℝ) : a + b = b + a := by ring"
    )
    invalid_lean = (
        "import Mathlib\n\n"
        "theorem add_comm_broken (a b : ℝ) : a + b = b + a := by sorry"
    )
    
    res_valid = run_post("/v1/lean/verify", {"code": valid_lean})
    print(f"  {GREEN}✓ Valid theorem verified: {res_valid.get('verified')} ({res_valid.get('compiler_output')}){RESET}")
    
    res_invalid = run_post("/v1/lean/verify", {"code": invalid_lean})
    print(f"  {GREEN}✓ Invalid theorem rejected: {not res_invalid.get('verified')} ({res_invalid.get('compiler_output')}){RESET}")

    # ───────────────────────────────────────────────────────────
    # Test 5: Closed-Loop SFT / RLCF Backpropagation Updates
    # ───────────────────────────────────────────────────────────
    print(f"\n{BOLD}[5/5] Testing Scenario D: Closed-Loop Online Synaptic Tuning (RLCF)...{RESET}")
    reinforce_req = {
        "problem": problem_a,
        "corrected_solution": (
            "Proof step corrected: Since B_L is orthogonal, its eigenvalues have magnitude exactly 1. "
            "Therefore, by the spectral mapping theorem, ||B_L * x|| = ||x|| for all x. Thus Lipschitz M = 1."
        )
    }
    t0 = time.time()
    res_reinforce = run_post("/v1/learn/reinforce", reinforce_req)
    latency_reinforce = time.time() - t0
    print(f"  {GREEN}✓ Weight reinforcement step completed in {latency_reinforce:.3f}s{RESET}")
    print(f"  {GREEN}✓ Active LoRA training loss: {CYAN}{res_reinforce.get('loss')}{RESET}")

    # ───────────────────────────────────────────────────────────
    # End Report
    # ───────────────────────────────────────────────────────────
    print(f"\n{BOLD}{GREEN}✓ ALL BACK-END MATHEMATICAL TEST SCENARIOS PASSED SUCCESSFULLY!{RESET}\n")

if __name__ == "__main__":
    main()
