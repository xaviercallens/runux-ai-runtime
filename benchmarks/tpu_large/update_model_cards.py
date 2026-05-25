#!/usr/bin/env python3
"""Update HuggingFace model cards with polished formatting."""
import tempfile, os
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
try:
    import httpx
    original_httpx_init = httpx.Client.__init__
    def patched_httpx_init(self, *args, **kwargs):
        kwargs['verify'] = False
        original_httpx_init(self, *args, **kwargs)
    httpx.Client.__init__ = patched_httpx_init
except ImportError:
    pass
try:
    import requests
    original_requests_init = requests.Session.__init__
    def patched_requests_init(self, *args, **kwargs):
        original_requests_init(self, *args, **kwargs)
        self.verify = False
    requests.Session.__init__ = patched_requests_init
except ImportError:
    pass

from huggingface_hub import HfApi

TOKEN = os.environ.get("HF_TOKEN", "")
api = HfApi(token=TOKEN)
username = "callensxavier"

models = [
    {
        "repo": f"{username}/runux-bench-qwen2.5-0.5b-tpu",
        "name": "Qwen 2.5 0.5B",
        "base": "Qwen/Qwen2.5-0.5B",
        "params": "0.5B",
        "tps_pt": 328.4, "tps_js": 485.2, "tps_rx": 1024.3,
        "energy_pt": 0.61, "energy_rx": 0.20,
        "cost_pt": 1.01, "cost_rx": 0.33,
        "speedup": "3.12x",
        "tags": "qwen2",
    },
    {
        "repo": f"{username}/runux-bench-mistral-7b-v0.3-tpu",
        "name": "Mistral 7B v0.3",
        "base": "mistralai/Mistral-7B-v0.3",
        "params": "7.0B",
        "tps_pt": 21.5, "tps_js": 32.4, "tps_rx": 67.1,
        "energy_pt": 9.30, "energy_rx": 2.98,
        "cost_pt": 15.50, "cost_rx": 4.97,
        "speedup": "3.12x",
        "tags": "mistral",
    },
    {
        "repo": f"{username}/runux-bench-gemma-2-9b-tpu",
        "name": "Gemma 2 9B",
        "base": "google/gemma-2-9b",
        "params": "9.0B",
        "tps_pt": 18.2, "tps_js": 28.6, "tps_rx": 58.8,
        "energy_pt": 10.99, "energy_rx": 3.40,
        "cost_pt": 18.31, "cost_rx": 5.67,
        "speedup": "3.23x",
        "tags": "gemma2",
    },
]

TEMPLATE = """---
language:
- en
license: apache-2.0
tags:
- benchmark
- tpu-v5e
- energy-efficient
- green-ai
- runux
- {tags}
- inference
base_model: {base}
pipeline_tag: text-generation
---

<div align="center">

# RunuX-AI Benchmark: {name}

### Google TPU v5e | {speedup} Faster | {speedup} Less Energy

**Xavier Callens** | [Socrate AI Lab](https://github.com/xaviercallens/runux-ai-runtime)

[Full Dataset](https://huggingface.co/datasets/callensxavier/runux-tpu-v5e-benchmarks) | [Scientific Article](https://huggingface.co/datasets/callensxavier/runux-tpu-v5e-benchmarks/blob/main/scientific_article.md)

</div>

---

## Results (BS=1, BF16, TPU v5e)

| Metric | PyTorch/XLA | JetStream | **RunuX-AI** | **vs PyTorch** |
|:-------|:-----------:|:---------:|:------------:|:--------------:|
| Throughput | {tps_pt} tok/s | {tps_js} tok/s | **{tps_rx} tok/s** | **{speedup}** |
| Energy | {energy_pt} J/tok | - | **{energy_rx} J/tok** | **{speedup}** |
| MXU Util | ~32% | ~40% | **88%** | **2.75x** |
| Cost/M tok | ${cost_pt} | - | **${cost_rx}** | **-68%** |

## Model Details

| | |
|:---|:---|
| **Base Model** | [{base}](https://huggingface.co/{base}) |
| **Parameters** | {params} |
| **Precision** | BF16 (bfloat16) |
| **Hardware** | Google TPU v5e (v5litepod-1, 197 TFLOPS) |
| **Runtime** | RunuX-AI v0.2.0 (no_std Rust, 23 crates) |

## Methodology

| Parameter | Value |
|:----------|:------|
| Input tokens | 512 |
| Decode tokens | 128 (greedy) |
| Warmup | 3 iterations |
| Measurement | 10 iterations (median) |
| Framework | PyTorch 2.4.0 + torch_xla 2.4.0 |

See [full methodology](https://huggingface.co/datasets/callensxavier/runux-tpu-v5e-benchmarks/blob/main/methodology.md).

## Reproduce Baselines

```bash
pip install torch torch_xla[tpu] transformers accelerate
python benchmark_baselines.py
```

## Collaboration

RunuX-AI is available for licensing through Socrate AI Lab. See our [collaboration page](https://huggingface.co/datasets/callensxavier/runux-tpu-v5e-benchmarks) for details.

## Citation

```bibtex
@article{{callens2026runux,
  title={{RunuX-AI: 3x Inference Throughput on TPU v5e}},
  author={{Callens, Xavier}},
  year={{2026}},
  note={{Socrate AI Lab}}
}}
```

---

*2026 Xavier Callens / Socrate AI Lab | Data: Apache-2.0 | Runtime: Patent Pending*
"""

for m in models:
    card = TEMPLATE.format(**m)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
        f.write(card)
        path = f.name
    
    api.upload_file(
        path_or_fileobj=path,
        path_in_repo='README.md',
        repo_id=m['repo'],
        repo_type='model',
        token=TOKEN,
    )
    os.unlink(path)
    print(f"Updated: https://huggingface.co/{m['repo']}")

print("All model cards updated!")
