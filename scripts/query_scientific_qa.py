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
OUTPUT_FILE = Path("/Volumes/MacCleanerStorage/xdev/xavux/runux-ai-runtime/scripts/biomimetic_training/scientific_qa_solutions.json")

def query_solve(problem_desc: str) -> dict:
    url = f"{ENDPOINT}/v1/solve"
    payload = {
        "problem": problem_desc,
        "max_steps": 5,
        "use_mcts": False
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception as e:
        print(f"Error querying {url}: {e}")
        return {"reasoning": f"Failed to connect: {e}", "answer": "Error"}

def main():
    print("=" * 80)
    print("  Socrate AI Lab — Live SymBrain v3 Scientific Q&A and Evaluation Suite")
    print("=" * 80)

    # Problem 1: Algebraic Topology & Sheaf Theory
    p1 = (
        "Consider a presheaf of dialectical logical propositions \\mathcal{P} on a sequence manifold \\mathcal{M} "
        "representing transformer token steps. Prove that the sequence mean-pooling state embedding "
        "s = \\frac{1}{T}\\sum h_t acts as a singular topological defect or sheaf-theoretic obstruction "
        "\\check{H}^1(\\mathcal{M}, \\mathcal{P}) \\neq 0$ that discards syntactic proof-path connections. "
        "Formulate a sheaf-reconstruction method to recover global proof-theoretic consistency."
    )

    # Problem 2: Astrophysics & General Relativity
    p2 = (
        "Formulate the propagation equations of high-frequency gravitational waves h_{\\mu\\nu} passing through a "
        "toroidal magnetohydrodynamic (MHD) plasma torus with background magnetic field B_0. Assume the metric g_{\\mu\\nu} "
        "is subject to a Ricci-Le'vy Curvature Flow (RLCF) with heavy-tailed stochastic perturbations \\eta_t. "
        "Derive the dispersion relation and examine if energy dissipation bounds are preserved under the fluctuation-dissipation theorem."
    )

    # Problem 3: Control Theory & Lyapunov Stability
    p3 = (
        "Let G(d, g) be the prefrontal cortex gating function regulating dual-hemisphere updates. Under Axioms 1, 2, and 3: "
        "(1) prove the existence of a stochastic Lyapunov candidate function V(W) for the weights W_t satisfying "
        "\\mathbb{E}[dV/dt] \\le 0$ under the Ricci-Le'vy Curvature Flow (RLCF) update, and (2) verify how the "
        "homeostatic stability bounds in Axiom 2 quench infinite-variance Le'vy flight noise dZ_t near a local minimum."
    )

    scenarios = [
        ("Algebraic Topology & Sheaf Theory", p1),
        ("Astrophysics & General Relativity", p2),
        ("Control Theory & Lyapunov Stability", p3)
    ]

    results = {}

    for name, query in scenarios:
        print(f"\n[+] Querying SymBrain Swarm (32B) for: {name}...")
        print("    Processing cognitive reasoning trace...")
        res = query_solve(query)
        results[name] = res
        print(f"    [✓] Completed: {name}")
        print("-" * 80)
        print(f"REASONING:\n{res.get('reasoning')[:500]}...")
        print("-" * 80)

    # Save to file
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[x] Logged all scientific Q&A solutions to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
