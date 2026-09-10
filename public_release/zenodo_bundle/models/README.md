---
license: apache-2.0
tags:
- runux
- scientific-proof
- benchmark
- polarquant
- signsgd
- tpu-trillium
- deterministic-attention
---

# RunuX AI Runtime — Scientific Benchmarking & Evaluation Artifacts

This repository provides open, reproducible evaluation datasets, hardware benchmarks, and model configurations accompanying the paper:
**"RunuX: Deterministic Fixed-Point Attention, Isometric PolarQuant KV-Compression, and Grid-Carbon Adaptive Scheduling for High-Efficiency Foundation Model Inference"** (Xavier Callens, Socrate AI Lab, 2026).

## Key Benchmarking Highlights
- **Mistral Large 2 (128K context)**: 4.92× KV-Cache memory reduction via PolarQuant 3-bit.
- **NVIDIA Distributed Megatron-LM 70B**: 32.0× gradient sync bandwidth reduction via 1-bit SignSGD.
- **Google Cloud TPU v5e / v6e Trillium**: 88.0% MXU systolic occupancy on Gemma 2 projections.
- **Reproducibility**: Evaluated on NVIDIA Tesla T4 and modeled across GCP Spot compute within a $50 budget.

*Note: Proprietary RunuX runtime engine source code and binaries are excluded.*
