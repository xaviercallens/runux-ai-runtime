# RunuX AI Runtime — Enterprise Partnership Evaluation Package

**Document Version**: 1.0.0-EVAL  
**Confidentiality**: STRICTLY CONFIDENTIAL — UNDER MUTUAL NDA  
**Governing License**: [`legal/EVALUATION_LICENSE.md`](../legal/EVALUATION_LICENSE.md)  
**Jurisdiction**: Commercial Court of Paris, France  

---

## 1. Executive Purpose

This evaluation package provides technical leads, research engineers, and compiler teams at **Mistral AI**, **NVIDIA Corporation**, and **Google Cloud** with reproducible validation harnesses to assess the performance, energy efficiency, and mathematical determinism of the **RunuX AI Runtime** on their targeted workloads.

### Target Alignments

| Partner | Specialized Evaluation Pack | Focus Area & Metric |
|:---|:---|:---|
| **Mistral AI** | `harness_mistral.py` | **PolarQuant 3-bit KV Cache** (4.92× memory reduction on Mistral Large 2) & **RTE Carbon-Aware Speculative Scheduling** (64.9% carbon reduction). |
| **NVIDIA** | `harness_nvidia.py` | **INT64 Deterministic Attention** (Zero numerical drift, bit-exact reproducibility) & **1-Bit SignSGD** (32× communication compression for Megatron-LM). |
| **Google Cloud** | `harness_google.py` | **Cloud TPU v5e/v6e Systolic Tiling** (88.0% MXU occupancy vs 38% baseline, 2.32× speedup on Gemma 2) & **Safe Rust PJRT**. |

---

## 2. Quickstart Execution

To run all evaluation harnesses with a single command:

```bash
./evaluation_package/verify_evaluation.sh
```

To run individual partner evaluation suites:

```bash
# 1. Mistral AI Evaluation:
python3 evaluation_package/harness_mistral.py

# 2. NVIDIA Corporation Evaluation:
python3 evaluation_package/harness_nvidia.py

# 3. Google Cloud Evaluation:
python3 evaluation_package/harness_google.py
```

---

## 3. Evaluation Package Structure

```text
evaluation_package/
├── README.md                 # This evaluation guide
├── verify_evaluation.sh      # Unified master execution script
├── harness_mistral.py        # PolarQuant KV-cache & RTE Green AI harness
├── harness_nvidia.py         # INT64 Deterministic Attention & SignSGD harness
└── harness_google.py         # TPU v5e/v6e MXU Systolic Tiling & StableHLO harness
```

---

## 4. Evaluation License Terms & Contact

- **Evaluation Window**: 90 Calendar Days from delivery.
- **Restrictions**: Internal benchmarking only; unauthorized benchmark publishing, reverse-engineering, or decompilation is strictly prohibited under the French Intellectual Property Code (*Code de la propriété intellectuelle*).
- **Commercial Inquiries & Conversion**:
  - Xavier Callens (`xavier.callens@socrate.ai`)
  - Web: [https://runux.ai](https://runux.ai)

---
*(c) 2026 Xavier Callens / Socrate AI Lab. All Rights Reserved.*
