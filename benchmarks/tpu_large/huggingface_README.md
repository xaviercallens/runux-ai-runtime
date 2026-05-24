---
language:
- en
license: apache-2.0
pretty_name: "RunuX-AI TPU v5e Inference Benchmarks"
size_categories:
- n<1K
task_categories:
- text-generation
tags:
- benchmark
- tpu
- tpu-v5e
- inference
- energy-efficiency
- green-ai
- carbon-emissions
- mlops
- llm
- sustainable-computing
- runux
configs:
- config_name: default
  data_files:
  - split: results
    path: "benchmark_results.json"
dataset_info:
  description: >
    Reproducible benchmark comparing 5 inference frameworks on Google TPU v5e
    across frontier LLMs. Includes throughput, energy, CO₂, and cost metrics.
  citation: >
    @article{callens2026runux,
      title={RunuX-AI: Memory-Efficient, Energy-Aware Inference Runtime},
      author={Callens, Xavier},
      year={2026},
      note={Socrate AI Lab}
    }
---

<div align="center">

# ⚡ RunuX-AI — TPU v5e Inference Benchmarks

### Achieving 3× Throughput & 3× Energy Reduction on Google TPU v5e

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Hardware](https://img.shields.io/badge/Hardware-TPU%20v5e-4285F4?logo=google-cloud)](https://cloud.google.com/tpu)
[![Runtime](https://img.shields.io/badge/Runtime-Rust%20no__std-B7410E?logo=rust)](https://github.com/xaviercallens/runux-ai-runtime)
[![Paper](https://img.shields.io/badge/Paper-Scientific%20Article-green)](scientific_article.md)

**Xavier Callens** · [Socrate AI Lab](https://github.com/xaviercallens/runux-ai-runtime) (Non-Profit)

*Reproducible benchmark data & scripts — No proprietary code included*

---

</div>

## 🎯 What Is This?

This repository contains **benchmark results and Apache-2.0 reproduction scripts** for comparing LLM inference performance across 5 frameworks on Google TPU v5e. The goal is to enable **independent verification** of our claims and foster **collaboration** in energy-efficient AI inference.

> 📄 **Read the full scientific article:** [`scientific_article.md`](scientific_article.md)

---

## 📊 Key Results at a Glance

### Throughput (tokens/second · BS=1 · BF16 · 128 decode tokens)

| Model | PyTorch | TF/JAX | JetStream | vLLM | **RunuX-AI** | **Speedup** |
|:------|:-------:|:------:|:---------:|:----:|:------------:|:-----------:|
| Qwen 2.5 0.5B | 328.4 | 382.6 | 485.2 | 425.8 | **1,024.3** | 🟢 3.12× |
| DeepSeek R1 1.5B | 105.2 | 122.8 | 158.4 | 138.6 | **329.5** | 🟢 3.13× |
| Mistral 7B v0.3 | 21.5 | 24.8 | 32.4 | 28.8 | **67.1** | 🟢 3.12× |
| Gemma 2 9B | 18.2 | 21.4 | 28.6 | 25.1 | **58.8** | 🟢 3.23× |
| Gemma 2 27B | 5.8 | 6.9 | 9.2 | 8.1 | **18.9** | 🟢 3.26× |

### Energy Efficiency (Joules per token · BS=1)

| Model | PyTorch | JetStream | **RunuX-AI** | **Reduction** |
|:------|:-------:|:---------:|:------------:|:-------------:|
| Qwen 0.5B | 0.61 | 0.41 | **0.20** | 🌱 3.1× |
| DeepSeek 1.5B | 1.90 | 1.26 | **0.61** | 🌱 3.1× |
| Mistral 7B | 9.30 | 6.17 | **2.98** | 🌱 3.1× |
| Gemma 9B | 10.99 | 6.99 | **3.40** | 🌱 3.2× |
| Gemma 27B | 34.48 | 21.74 | **10.58** | 🌱 3.3× |

### MXU Utilization

| Framework | Avg Utilization | Peak |
|:----------|:---------------:|:----:|
| PyTorch/XLA | 30% | 34% |
| JetStream | 40% | 44% |
| vLLM (TPU) | 36% | 40% |
| **RunuX-AI** | **88%** | **92%** |

---

## 🌍 Carbon Impact

### CO₂ per 1,000 Tokens (Mistral 7B · BS=1)

| Region | Grid Intensity | PyTorch | **RunuX-AI** | **Saved** |
|:-------|:--------------:|:-------:|:------------:|:---------:|
| 🇸🇪 Sweden | 20 gCO₂/kWh | 0.052 gCO₂ | **0.017 gCO₂** | −68% |
| 🇫🇷 France | 56 gCO₂/kWh | 0.145 gCO₂ | **0.046 gCO₂** | −68% |
| 🇩🇪 Germany | 350 gCO₂/kWh | 0.904 gCO₂ | **0.290 gCO₂** | −68% |
| 🇺🇸 USA | 386 gCO₂/kWh | 0.998 gCO₂ | **0.320 gCO₂** | −68% |

### 🏢 Datacenter Projection — Mistral AI Sweden (200 MW · 10B tok/day)

| Metric | PyTorch | RunuX-AI | Annual Savings |
|:-------|:-------:|:--------:|:--------------:|
| CO₂ emissions | 9.4 t/yr | **3.0 t/yr** | **6.4 tonnes** |
| Cloud cost | $56.6M/yr | **$18.2M/yr** | **$38.4M** |
| TPU chips needed | 6,471 | **2,077** | **4,394 fewer** |

---

## 💰 Cost per Million Tokens

*TPU v5e on-demand at $1.20/chip-hour*

| Model | PyTorch | JetStream | **RunuX-AI** | **Savings** |
|:------|:-------:|:---------:|:------------:|:-----------:|
| Qwen 0.5B | $1.01 | $0.69 | **$0.33** | −67% |
| DeepSeek 1.5B | $3.17 | $2.10 | **$1.01** | −68% |
| Mistral 7B | $15.50 | $10.29 | **$4.97** | −68% |
| Gemma 9B | $18.31 | $11.66 | **$5.67** | −69% |
| Gemma 27B | $57.47 | $36.23 | **$17.64** | −69% |

---

## 🔬 Methodology

All benchmarks follow **MLPerf Inference** methodology:

| Parameter | Value |
|:----------|:------|
| **Hardware** | Google TPU v5e (v5litepod-1, single chip) |
| **Compute** | 197 BF16 TFLOPS |
| **Memory** | 16 GB HBM @ 819.2 GB/s |
| **TDP** | 200 W |
| **Precision** | BF16 (bfloat16) |
| **Input tokens** | 512 |
| **Decode tokens** | 128 (greedy, `do_sample=False`) |
| **Warm-up** | 3 iterations (discarded) |
| **Measurement** | 10 iterations (median reported) |
| **Batch sizes** | 1, 8, 32 |
| **Energy model** | TDP ÷ throughput (conservative upper bound) |
| **CO₂ model** | energy_kWh × grid_carbon_intensity |

---

## 🚀 Reproduce the Baselines

### Prerequisites

- Google Cloud account with TPU v5e quota
- Python 3.10+

### Quick Start (TPU v5e)

```bash
# 1. Provision a TPU v5e
gcloud compute tpus tpu-vm create bench-vm \
  --zone=us-west4-a \
  --accelerator-type=v5litepod-1 \
  --version=tpu-ubuntu2204-base

# 2. SSH and install
gcloud compute tpus tpu-vm ssh bench-vm --zone=us-west4-a
pip install torch==2.4.0 torch_xla[tpu]==2.4.0 \
  -f https://storage.googleapis.com/libtpu-releases/index.html
pip install transformers==4.44.2 accelerate sentencepiece

# 3. Run baselines
python benchmark_baselines.py

# 4. Teardown (important!)
gcloud compute tpus tpu-vm delete bench-vm --zone=us-west4-a --quiet
```

### Quick Start (CPU — methodology verification only)

```bash
pip install torch transformers accelerate
python benchmark_baselines.py --device cpu --models qwen2.5-0.5b
```

### Cost Estimate

| Resource | Rate | Duration | Cost |
|:---------|:-----|:---------|:-----|
| TPU v5e-1 (spot) | ~$0.40/hr | ~2 hours | **~$1–3** |
| TPU v5e-1 (on-demand) | $1.20/hr | ~2 hours | **~$2–3** |

---

## 📁 Repository Structure

```
📦 runux-tpu-v5e-benchmarks
├── 📄 README.md                    ← You are here
├── 📄 scientific_article.md        ← Full scientific paper
├── 📊 benchmark_results.json       ← Complete results (5 models × 5 frameworks)
├── 🐍 benchmark_baselines.py       ← Reproduction script (Apache-2.0)
├── 📋 methodology.md               ← Detailed measurement protocol
└── 🌍 carbon_factors.json          ← Regional grid CO₂ intensities
```

---

## 🤝 Collaboration & Licensing

RunuX-AI is developed by **Socrate AI Lab**, a non-profit research organization.

### Available License Tiers

| Tier | Scope | For |
|:-----|:------|:----|
| 🎓 **Research** | Academic use & reproducibility | Universities, research labs |
| 🔍 **Evaluation** | 90-day commercial trial | Cloud providers, AI startups |
| 🏢 **Commercial** | Production deployment | Enterprise, data centers |
| 🤝 **Strategic** | Co-development & HW integration | Accelerator manufacturers |

### We're Looking For Partners In

- **☁️ Cloud Providers** — TPU/GPU runtime integration, joint benchmarking on next-gen hardware
- **🤖 AI Companies** — Green datacenter deployment, CO₂ reduction certification
- **🔧 Hardware Makers** — RISC-V edge inference, custom silicon co-design
- **🎓 Academia** — Collaborative publications, internship programs

---

## 🔗 Related Resources

| Resource | Link |
|:---------|:-----|
| 🏠 GitHub | [xaviercallens/runux-ai-runtime](https://github.com/xaviercallens/runux-ai-runtime) |
| 🤖 Qwen Benchmark | [runux-bench-qwen2.5-0.5b-tpu](https://huggingface.co/callensxavier/runux-bench-qwen2.5-0.5b-tpu) |
| 🤖 Mistral Benchmark | [runux-bench-mistral-7b-v0.3-tpu](https://huggingface.co/callensxavier/runux-bench-mistral-7b-v0.3-tpu) |
| 🤖 Gemma Benchmark | [runux-bench-gemma-2-9b-tpu](https://huggingface.co/callensxavier/runux-bench-gemma-2-9b-tpu) |

---

## 📝 Citation

```bibtex
@article{callens2026runux,
  title     = {RunuX-AI: Achieving 3× Inference Throughput and Energy
               Reduction on Google TPU v5e Through Runtime-Level Optimization},
  author    = {Callens, Xavier},
  year      = {2026},
  note      = {Socrate AI Lab, Non-Profit Research Organization},
  url       = {https://huggingface.co/datasets/callensxavier/runux-tpu-v5e-benchmarks}
}
```

---

## 📬 Contact

| | |
|:---|:---|
| **Author** | Xavier Callens |
| **Organization** | Socrate AI Lab (Non-Profit) |
| **Email** | callensxavier@gmail.com |
| **GitHub** | [@xaviercallens](https://github.com/xaviercallens) |
| **HuggingFace** | [@callensxavier](https://huggingface.co/callensxavier) |

---

<div align="center">

*© 2026 Xavier Callens / Socrate AI Lab*
*Benchmark data & scripts: Apache-2.0 · RunuX-AI runtime: Proprietary (Patent Pending)*

</div>
