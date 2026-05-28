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
OUTPUT_FILE = Path("/Volumes/MacCleanerStorage/xdev/xavux/runux-ai-runtime/scripts/biomimetic_training/high_school_exam_solutions.json")

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
    print("  Socrate AI Lab — Live SymBrain v3 High School Scientific Exam")
    print("=" * 80)

    # Question 1: Mathematics (French Bac S)
    q1 = (
        "Let f(x) = e^x / (e^x + 1) be a function defined on \\mathbb{R}.\n"
        "1. Determine the limits of f(x) as x approaches -\\infty and +\\infty.\n"
        "2. Prove that the derivative is f'(x) = e^x / (e^x + 1)^2.\n"
        "3. Determine the variation table of f(x) on \\mathbb{R} (stating if it is increasing, decreasing, or constant)."
    )

    # Question 2: Physics (AP Physics C Mechanics)
    q2 = (
        "A block of mass m = 2.0 kg is placed on a rough inclined plane at an angle \\theta = 30^\\circ to the horizontal. "
        "The coefficient of static friction is \\mu_s = 0.40 and the coefficient of kinetic friction is \\mu_k = 0.30.\n"
        "1. Draw/describe the free-body diagram of the block.\n"
        "2. Determine if the block remains in static equilibrium or slides down the incline.\n"
        "3. If the block slides, calculate the magnitude of its acceleration a down the incline (use g = 9.8 m/s^2)."
    )

    # Question 3: Chemistry (AP Chemistry)
    q3 = (
        "Calculate the pH of a 0.10 M solution of acetic acid (CH_3COOH) at 25^\\circ C. "
        "The acid dissociation constant of acetic acid is K_a = 1.8 \\times 10^{-5}. Show your assumptions and steps."
    )

    scenarios = [
        ("Mathematics (French Bac S)", q1),
        ("Physics (AP Physics C)", q2),
        ("Chemistry (AP Chemistry)", q3)
    ]

    results = {}

    for name, query in scenarios:
        print(f"\n[+] Querying SymBrain Swarm (32B) for: {name}...")
        res = query_solve(query)
        results[name] = res
        print(f"    [✓] Completed: {name}")
        print("-" * 80)
        print(f"REASONING:\n{res.get('reasoning')[:400]}...")
        print("-" * 80)

    # Save to file
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[x] Logged all exam solutions to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
