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
- 1hour-soak-test
- zenodo-archived
---

# RunuX AI Runtime — Scientific Benchmarking & Open Evaluation Artifacts

[![Zenodo DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.14992026-blue.svg)](https://doi.org/10.5281/zenodo.14992026)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Paper PDF](https://img.shields.io/badge/Paper-PDF_Available-red.svg)](paper/runux_scientific_proof_paper.pdf)

This repository provides open, reproducible evaluation datasets, empirical hardware telemetry, and model configuration cards accompanying the scientific research paper:
> **"RunuX: Deterministic Fixed-Point Attention, Isometric PolarQuant KV-Compression, and Grid-Carbon Adaptive Scheduling for High-Efficiency Foundation Model Inference"**  
> *Author: Xavier Callens (Socrate AI Lab / RunuX Research, 2026)*  
> *Permanent Zenodo Archive: [10.5281/zenodo.14992026](https://doi.org/10.5281/zenodo.14992026)*

---

## 1. Key Scientific Benchmarking Highlights

### A. Long-Duration 1-Hour Sustained Soak Test (NVIDIA Tesla T4 GPU)
* **Duration**: 60 consecutive observation windows (3,600s cumulative execution, 15,000 attention passes, 15.36M tokens).
* **Throughput & Jitter**: Sustained **2.76 TFLOPS** with steady-state jitter under $\pm 1.2\%$.
* **Latency Profile**: Steady-state $p_{50} = \mathbf{0.368\text{ ms}}$, tail $p_{99} = \mathbf{0.450\text{ ms}}$.
* **Thermal Equilibrium**: Clean convergence from $68^\circ\text{C}$ to $70^\circ\text{C}$ with peak power draw of 58.17 W (comfortably below the 70 W TDP cap, zero thermal throttling).
* **Zero Memory Leakage**: Flat VRAM footprint at 28.12 MB throughout ($\Delta_{\text{leak}} = \mathbf{0.000\text{ MB}}$), empirically validating the Lean 4 formal bump allocator safety proof.
* **Deterministic Stability**: Max fixed-point INT64 divergence $\Delta_{\text{num}} = \mathbf{0.000}$ (bit-exact zero numerical drift across all 15,000 passes).

### B. Cloud Spot Serverless TPU Protocol (Google Cloud TPU v5e/v6e Trillium)
* **Systolic MXU Occupancy**: Elevated from 38.0% to **88.0%** on Google Gemma 2 9B/27B projection layers (**2.32× throughput speedup**).
* **Spot Preemption Resilience**: Handled unannounced Spot preemption notices via asynchronous page-boundary DMA flushing in **9.50 ms** ($< 12\text{ ms}$ budget).
* **Zero Token Loss**: **0 tokens lost** during migration; achieved **65.2% cost reduction** (\$0.40/h Spot vs \$1.15/h On-Demand).

### C. Foundation Model Evaluation Configurations
* **Mistral Large 2 (128K context)**: PolarQuant 3-bit KV-cache reduction from 44.0 GB to **8.94 GB** (**4.92× compression**, 35.06 GB VRAM saved) with bounded norm distortion ($\Delta < 0.35$).
* **NVIDIA Megatron-LM 70B**: 1-bit SignSGD with error feedback reduces 400 Gbps InfiniBand gradient synchronization from 521.5 GB to **16.3 GB** (**32.0× bandwidth reduction**, latency reduced from 10.43 ms to 0.33 ms).
* **Carbon-Adaptive Speculative Decoding**: Dynamic asservissement to RTE Eco2Mix real-time grid telemetry yielding a **26.9× carbon emissions reduction** on low-carbon baseload electricity.
* **GCP Spot & Serverless Economic Envelope**: Fully reproducible experimental schedule strictly capped at **\$50.00 USD**.

---

## 2. Intellectual Property & Trade Secret Notice

> **IMPORTANT**: In accordance with commercial IP protection and pending patent applications before the Institut National de la Propriété Industrielle (INPI), the proprietary RunuX runtime engine source code, compiled Rust kernel binaries, internal memory layout structures, and private cryptographic salts are strictly proprietary and excluded from this public distribution. Only open evaluation configurations, empirical telemetry, benchmark datasets, and academic documentation are made publicly available under CC-BY-4.0.
