#!/usr/bin/env python3
# Copyright (c) 2026 Xavier Callens / Socrate AI Lab. All Rights Reserved.
# SPDX-License-Identifier: LicenseRef-RunuX-Commercial

import os
import sys
import ssl
import json
import urllib.request
from pathlib import Path

# Disable SSL verification globally
ssl._create_default_https_context = ssl._create_unverified_context

ENDPOINT = "https://symbrain-v3-1003063861791.us-central1.run.app"
OUTPUT_FILE = Path("/Volumes/MacCleanerStorage/xdev/xavux/runux-ai-runtime/scripts/biomimetic_training/symbrain_math_solution.json")

def main():
    print("=" * 80)
    print("  Socrate AI Lab — Live SymBrain v3 Mathematical Physics Validation Client")
    print(f"  Target Endpoint: {ENDPOINT}")
    print("=" * 80)

    # 1. Formulate the complex Weyl Spinor and RLCF stability problem
    problem_desc = (
        "Compute the eigenvalue spectrum \\lambda_n of the localized Weyl spinor boundary perturbation operator D_\\partial "
        "on a toroidal 3-manifold \\mathbb{T}^3 under a dynamic Ricci-Le'vy Curvature Flow (RLCF):\n"
        "  \\frac{\\partial g_{\\mu\\nu}}{\\partial t} = -2 R_{\\mu\\nu} + \\alpha |R|^{\\beta - 1} \\Delta^{\\gamma} R_{\\mu\\nu} + \\eta_t\n"
        "where \\eta_t represents an infinite-variance \\alpha-stable Le'vy noise vector field.\n"
        "Prove that the Socratic WARS-CI-DFA feedback control mechanism guarantees an upper bound on energy drift "
        "E(t) = \\int_{\\mathbb{T}^3} |\\psi|^2 dV_g \\le E_0 e^{-\\kappa t} in the presence of heavy-tailed noise, "
        "and analyze the mathematical information bottleneck caused by sequence mean-pooling in the PFC state embedding "
        "s_{pfc} = \\frac{1}{T} \\sum_{t=1}^T h_t."
    )

    payload = {
        "problem": problem_desc,
        "max_steps": 5,
        "use_mcts": False
    }

    # 2. Query endpoint
    url = f"{ENDPOINT}/v1/solve"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    print("\n[+] Transmitting mathematical formulation to SymBrain Swarm Bourbaki (32B)...")
    print("    This may take several seconds due to serverless cold-start or heavy cognitive traces...")
    
    try:
        with urllib.request.urlopen(req, timeout=90) as response:
            res = json.loads(response.read().decode("utf-8"))
            print("\n[✓] Received mathematical reasoning trace successfully!")
            
            # Print the reasoning and solution
            reasoning = res.get("reasoning", "No reasoning returned.")
            solution = res.get("solution", "No solution returned.")
            
            print("\n" + "-" * 80)
            print("REASONING TRACE:")
            print("-" * 80)
            print(reasoning)
            print("-" * 80)
            print("FINAL RESOLUTION:")
            print("-" * 80)
            print(solution)
            print("-" * 80)

            # Save response to file
            OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(res, f, indent=2)
            print(f"\n[x] Logged full response payload to: {OUTPUT_FILE}")

    except Exception as e:
        print(f"\n❌ Error querying SymBrain endpoint: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
