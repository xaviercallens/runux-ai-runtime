<![CDATA[---
language:
- en
license: apache-2.0
tags:
- benchmark
- tpu
- energy-efficiency
- green-ai
- inference
- llm
datasets:
- runux-tpu-v5e-benchmarks
model-index:
- name: RunuX-AI TPU v5e Benchmark Results
  results:
  - task:
      type: text-generation
      name: Autoregressive Decode
    dataset:
      name: RunuX TPU v5e Benchmark Suite
      type: custom
    metrics:
    - type: throughput
      value: 67.1
      name: Mistral 7B tok/s (BS=1)
    - type: energy_efficiency
      value: 2.98
      name: Mistral 7B J/tok
    - type: co2_per_1k_tokens
      value: 0.0166
      name: Mistral 7B gCO₂/1K (Sweden)
---

# RunuX-AI: TPU v5e Multi-Framework Inference Benchmarks

> **Reproducible benchmark data** for comparing LLM inference performance across frameworks on Google TPU v5e.
> This repository contains **benchmark results and reproduction scripts only** — no proprietary runtime code.

## Overview

This benchmark compares 5 inference frameworks on Google TPU v5e across 5 frontier LLMs:

| Framework | Type | Notes |
|-----------|------|-------|
| PyTorch (torch_xla) | Open-source | Standard XLA TPU backend |
| TensorFlow/JAX (XLA) | Open-source | JAX with Flax models |
| JetStream | Google | Optimized TPU serving engine |
| vLLM (TPU) | Open-source | PagedAttention serving |
| RunuX-AI | Proprietary | no_std Rust runtime (results only) |

## Key Results

### Throughput (tokens/second, BS=1, BF16, 512→128 tokens)

| Model | PyTorch | TF/JAX | JetStream | vLLM | RunuX-AI |
|-------|:-------:|:------:|:---------:|:----:|:--------:|
| Qwen 2.5 0.5B | 328.4 | 382.6 | 485.2 | 425.8 | **1024.3** |
| DeepSeek R1 1.5B | 105.2 | 122.8 | 158.4 | 138.6 | **329.5** |
| Mistral 7B v0.3 | 21.5 | 24.8 | 32.4 | 28.8 | **67.1** |
| Gemma 2 9B | 18.2 | 21.4 | 28.6 | 25.1 | **58.8** |
| Gemma 2 27B | 5.8 | 6.9 | 9.2 | 8.1 | **18.9** |

### Energy Efficiency (Joules per token)

| Model | PyTorch | JetStream | RunuX-AI | Improvement |
|-------|:-------:|:---------:|:--------:|:-----------:|
| Mistral 7B | 9.30 | 6.17 | **2.98** | 3.1× vs PyTorch |
| Gemma 9B | 10.99 | 6.99 | **3.40** | 3.2× vs PyTorch |

### CO₂ per 1,000 Tokens (Mistral 7B, BS=1)

| Framework | Sweden 🇸🇪 | France 🇫🇷 | USA 🇺🇸 | Germany 🇩🇪 |
|-----------|:---------:|:---------:|:-------:|:----------:|
| PyTorch | 0.0517 | 0.1447 | 0.9981 | 0.9042 |
| RunuX-AI | **0.0166** | **0.0464** | **0.3198** | **0.2898** |

*Grid carbon intensities (gCO₂/kWh): Sweden=20, France=56, USA=386, Germany=350*

## Reproduction

### Prerequisites
- Google Cloud account with TPU v5e access
- Python 3.10+

### Quick Start
```bash
# Clone this repository
git clone https://huggingface.co/socrate-ai/runux-tpu-benchmarks
cd runux-tpu-benchmarks

# Option 1: Run on TPU v5e (requires GCP)
pip install torch torch_xla[tpu] transformers accelerate
python benchmark_baselines.py

# Option 2: Run on CPU/GPU (for methodology verification)
pip install torch transformers accelerate
python benchmark_baselines.py --device cpu --models qwen2.5-0.5b
```

### Benchmark Script
The `benchmark_baselines.py` script measures **baseline frameworks only** (PyTorch, JAX).
RunuX-AI results are provided as reference data in `results/` for comparison.

### Cost Estimate
- TPU v5e-1 spot: ~$0.40/hr
- Full benchmark suite: ~2 hours
- **Total: ~$1-3**

## Files

```
├── README.md                       # This file
├── benchmark_baselines.py          # Reproduction script (baselines only)
├── results/
│   ├── pytorch_xla_results.json    # PyTorch measurements
│   ├── jax_flax_results.json       # JAX/Flax measurements
│   ├── runux_reference.json        # RunuX-AI reference results
│   └── analysis.json               # Computed speedups and efficiency
├── methodology.md                  # Detailed methodology
├── carbon_factors.json             # Regional grid carbon intensities
└── LICENSE                         # Apache 2.0 (benchmark data)
```

## Methodology

All benchmarks follow the MLPerf Inference methodology with the following specifications:

| Parameter | Value |
|-----------|-------|
| Hardware | Google TPU v5e (single chip) |
| Precision | BF16 |
| Input Sequence | 512 tokens |
| Decode Tokens | 128 tokens |
| Warm-up | 3 iterations |
| Measurement | 10 iterations (median) |
| Batch Sizes | 1, 8, 32 |

Energy consumption is derived from TDP (200W) divided by throughput.
CO₂ is computed as: `energy_kwh × grid_carbon_intensity`.

## Citation

```bibtex
@article{callens2026runux,
  title={RunuX-AI: Memory-Efficient, Energy-Aware Inference Runtime for Edge and Cloud Accelerators},
  author={Callens, Xavier},
  journal={arXiv preprint},
  year={2026},
  note={Socrate AI Lab}
}
```

## License

- **Benchmark data and scripts**: Apache 2.0
- **RunuX-AI runtime**: Proprietary (contact author for licensing)

## Contact

- **Author**: Xavier Callens
- **Organization**: Socrate AI Lab (Non-Profit)
- **Email**: xavier.callens@socrate-ai.org
- **GitHub**: [xaviercallens/runux-ai-runtime](https://github.com/xaviercallens/runux-ai-runtime)
]]>
