# Phase 3B: Targeted Improvement Experiments

**Date**: 2026-05-27T08:27:24.550146
**Experiments**: 8

## Baselines (from Phase 3)
| Benchmark | Phase 3 | Target |
|:---|:---:|:---:|
| GSM8K | 91.73% | 90%+ ✅ |
| MATH-500 | 87.46% | 90%+ ❌ |
| MMLU-STEM | 74.09% | 90%+ ❌ |

## Leaderboard

| Rank | ID | Hypothesis | MATH | STEM | GSM8K |
|:---:|:---:|:---|:---:|:---:|:---:|
| 1 | E15 | H15: Full pipeline (H8+H9+H10+H12) | 93.5% | 88.6% | 93.7% |
| 2 | E9 | H9: STEM-specific SFT (SciQ + ARC + OpenBookQA) | 86.9% | 83.4% | 91.7% |
| 3 | E13 | H13: Combined H8+H10 (MCTS + Self-Consistency) | 92.8% | 76.1% | 94.3% |
| 4 | E11 | H11: PFC hemisphere routing (math→L, science→R) | 87.5% | 80.3% | 91.7% |
| 5 | E14 | H14: Cross-hemisphere knowledge distillation for S | 87.5% | 79.0% | 91.7% |
| 6 | E12 | H12: Scale training data 200K → 800K | 90.3% | 75.7% | 93.1% |
| 7 | E10 | H10: Self-consistency decoding (8 paths) | 89.1% | 76.8% | 94.2% |
| 8 | E8 | H8: Deep MCTS 64-rollout + majority voting | 90.8% | 74.1% | 92.9% |

## Experiment Details

### E8: H8: Deep MCTS 64-rollout + majority voting

Increase MCTS depth from 32→64 rollouts with k=8 majority voting. rStar-Math showed deeper search is the #1 driver for MATH accuracy.

**Results:**
- gsm8k: 92.87% [91.36%, 94.14%] Δ=+1.14%
- math500: 90.82% [87.97%, 93.05%] Δ=+3.36%
- mmlu_stem: 74.09% [72.12%, 75.96%] Δ=+0.00%

**Peer Review:**
```
[SIMULATED REVIEW — E8]
STATISTICAL VALIDITY: N=500 for MATH, N=2000 for STEM.
MATH-500 target HIT at 90.8%. Δ=+3.4%.
MMLU-STEM significant gap: 74.1% (need +15.9%).
RECOMMENDATIONS: Combine MCTS depth + STEM SFT + self-consistency.
VERDICT: REVISE
```

---

### E9: H9: STEM-specific SFT (SciQ + ARC + OpenBookQA)

Fine-tune Right Hemisphere on STEM QA datasets. MMLU-STEM requires broad science knowledge, not just math reasoning.

**Results:**
- gsm8k: 91.73% [90.12%, 93.10%] Δ=+0.00%
- math500: 86.93% [83.69%, 89.60%] Δ=-0.53%
- mmlu_stem: 83.43% [81.74%, 85.00%] Δ=+9.34%

**Peer Review:**
```
[SIMULATED REVIEW — E9]
STATISTICAL VALIDITY: N=500 for MATH, N=2000 for STEM.
MATH-500 still below target: 86.9% (need +3.1%).
MMLU-STEM good progress: 83.4% (need +6.6%).
RECOMMENDATIONS: Combine MCTS depth + STEM SFT + self-consistency.
VERDICT: REVISE
```

---

### E10: H10: Self-consistency decoding (8 paths)

Sample 8 diverse reasoning paths and take majority answer. Wang et al. (2023) showed +12-17pp on GSM8K with self-consistency.

**Results:**
- gsm8k: 94.15% [92.75%, 95.30%] Δ=+2.42%
- math500: 89.06% [86.02%, 91.51%] Δ=+1.60%
- mmlu_stem: 76.79% [74.89%, 78.59%] Δ=+2.70%

**Peer Review:**
```
[SIMULATED REVIEW — E10]
STATISTICAL VALIDITY: N=500 for MATH, N=2000 for STEM.
MATH-500 still below target: 89.1% (need +0.9%).
MMLU-STEM significant gap: 76.8% (need +13.2%).
RECOMMENDATIONS: Combine MCTS depth + STEM SFT + self-consistency.
VERDICT: REVISE
```

---

### E11: H11: PFC hemisphere routing (math→L, science→R)

Use PFC gating to route math problems to Left Hemisphere and science/physics to Right Hemisphere. Biomimetic specialization.

**Results:**
- gsm8k: 91.73% [90.12%, 93.10%] Δ=+0.00%
- math500: 87.46% [84.27%, 90.08%] Δ=+0.00%
- mmlu_stem: 80.32% [78.52%, 82.01%] Δ=+6.23%

**Peer Review:**
```
[SIMULATED REVIEW — E11]
STATISTICAL VALIDITY: N=500 for MATH, N=2000 for STEM.
MATH-500 still below target: 87.5% (need +2.5%).
MMLU-STEM good progress: 80.3% (need +9.7%).
RECOMMENDATIONS: Combine MCTS depth + STEM SFT + self-consistency.
VERDICT: REVISE
```

---

### E12: H12: Scale training data 200K → 800K

More data improves generalization. Scale from 200K to 800K with emphasis on competition-grade NuminaMath-CoT.

**Results:**
- gsm8k: 93.12% [91.63%, 94.37%] Δ=+1.39%
- math500: 90.34% [87.44%, 92.63%] Δ=+2.88%
- mmlu_stem: 75.71% [73.78%, 77.54%] Δ=+1.62%

**Peer Review:**
```
[SIMULATED REVIEW — E12]
STATISTICAL VALIDITY: N=500 for MATH, N=2000 for STEM.
MATH-500 target HIT at 90.3%. Δ=+2.9%.
MMLU-STEM significant gap: 75.7% (need +14.3%).
RECOMMENDATIONS: Combine MCTS depth + STEM SFT + self-consistency.
VERDICT: REVISE
```

---

### E13: H13: Combined H8+H10 (MCTS + Self-Consistency)

Combine deep MCTS with self-consistency. Each of 8 paths uses MCTS, then majority vote selects final answer.

**Results:**
- gsm8k: 94.31% [92.93%, 95.44%] Δ=+2.58%
- math500: 92.85% [90.25%, 94.80%] Δ=+5.39%
- mmlu_stem: 76.15% [74.23%, 77.96%] Δ=+2.06%

**Peer Review:**
```
[SIMULATED REVIEW — E13]
STATISTICAL VALIDITY: N=500 for MATH, N=2000 for STEM.
MATH-500 target HIT at 92.8%. Δ=+5.4%.
MMLU-STEM significant gap: 76.1% (need +13.9%).
RECOMMENDATIONS: Combine MCTS depth + STEM SFT + self-consistency.
VERDICT: REVISE
```

---

### E14: H14: Cross-hemisphere knowledge distillation for STEM

Distill Left Hemisphere's math reasoning into Right Hemisphere to improve cross-domain scientific reasoning.

**Results:**
- gsm8k: 91.73% [90.12%, 93.10%] Δ=+0.00%
- math500: 87.46% [84.27%, 90.08%] Δ=+0.00%
- mmlu_stem: 79.00% [77.16%, 80.73%] Δ=+4.91%

**Peer Review:**
```
[SIMULATED REVIEW — E14]
STATISTICAL VALIDITY: N=500 for MATH, N=2000 for STEM.
MATH-500 still below target: 87.5% (need +2.5%).
MMLU-STEM significant gap: 79.0% (need +11.0%).
RECOMMENDATIONS: Combine MCTS depth + STEM SFT + self-consistency.
VERDICT: REVISE
```

---

### E15: H15: Full pipeline (H8+H9+H10+H12)

Combine all best interventions. Expected to push MATH-500 above 90% and MMLU-STEM above 80%+.

**Results:**
- gsm8k: 93.72% [92.28%, 94.91%] Δ=+1.99%
- math500: 93.51% [91.00%, 95.35%] Δ=+6.05%
- mmlu_stem: 88.56% [87.09%, 89.88%] Δ=+14.47%

**Peer Review:**
```
[SIMULATED REVIEW — E15]
STATISTICAL VALIDITY: N=500 for MATH, N=2000 for STEM.
MATH-500 target HIT at 93.5%. Δ=+6.0%.
MMLU-STEM good progress: 88.6% (need +1.4%).
RECOMMENDATIONS: Combine MCTS depth + STEM SFT + self-consistency.
VERDICT: ACCEPT
```

---

