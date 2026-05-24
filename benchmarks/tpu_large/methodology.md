# RunuX-AI TPU v5e Benchmark — Full Methodology

> **Version**: 1.1 · **Last Updated**: May 2026
> **Author**: Xavier Callens · Socrate AI Lab

---

## 1. Hardware Specifications

### TPU v5e Configuration

| Component | Specification |
|:----------|:-------------|
| **Accelerator** | Google Cloud TPU v5e (v5litepod-1, single chip) |
| **Compute (BF16)** | 197 TFLOPS |
| **Compute (INT8)** | 393 TOPS |
| **HBM Capacity** | 16 GB |
| **HBM Bandwidth** | 819.2 GB/s |
| **ICI Bandwidth** | 1,600 GB/s (inter-chip, not used for single-chip) |
| **TDP** | 200 W |
| **Host CPU** | Intel Xeon (Emerald Rapids) |
| **Host RAM** | 48 GB DDR5 |
| **OS** | Ubuntu 22.04 (tpu-ubuntu2204-base) |
| **Region** | us-west4-a |
| **Pricing** | $1.20/chip-hr (on-demand) · $0.40/chip-hr (spot) |

### Software Stack

| Component | Version |
|:----------|:--------|
| Python | 3.10.6 |
| PyTorch | 2.4.0 |
| torch_xla | 2.4.0 (PJRT backend) |
| libtpu | 0.1.dev20240612 |
| Transformers | 4.44.2 |
| Accelerate | 1.13.0 |
| JAX | Latest pip (for JAX/Flax baselines) |

---

## 2. Models Under Test

### 2.1 Qwen 2.5 0.5B

| Property | Value |
|:---------|:------|
| **HuggingFace ID** | [`Qwen/Qwen2.5-0.5B`](https://huggingface.co/Qwen/Qwen2.5-0.5B) |
| **Parameters** | 0.49B |
| **Architecture** | Qwen2 (decoder-only transformer) |
| **Hidden size** | 896 |
| **Layers** | 24 |
| **Attention heads** | 14 (GQA: 2 KV heads) |
| **Context window** | 32,768 tokens |
| **Vocabulary** | 151,936 tokens |
| **BF16 memory footprint** | ~1.0 GB |
| **Fits on TPU v5e (16 GB)** | ✅ Yes (BS=1, 8, 32) |
| **Tokenizer** | SentencePiece (byte-level BPE) |
| **License** | Apache 2.0 |
| **Notes** | Smallest model; validates overhead measurement. Fast decode allows high iteration count for statistical significance. |

### 2.2 DeepSeek R1 Distill 1.5B

| Property | Value |
|:---------|:------|
| **HuggingFace ID** | [`deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B) |
| **Parameters** | 1.5B |
| **Architecture** | Qwen2-based (distilled from DeepSeek R1) |
| **Hidden size** | 1,536 |
| **Layers** | 28 |
| **Attention heads** | 12 (GQA: 2 KV heads) |
| **Context window** | 32,768 tokens |
| **Vocabulary** | 151,936 tokens |
| **BF16 memory footprint** | ~3.0 GB |
| **Fits on TPU v5e (16 GB)** | ✅ Yes (BS=1, 8, 32) |
| **Tokenizer** | SentencePiece (byte-level BPE) |
| **License** | MIT |
| **Notes** | Reasoning-optimized distillation. Tests performance on dense 1.5B architecture with longer chain-of-thought outputs. |

### 2.3 Gemma 2 2B

| Property | Value |
|:---------|:------|
| **HuggingFace ID** | [`google/gemma-2-2b`](https://huggingface.co/google/gemma-2-2b) |
| **Parameters** | 2.6B |
| **Architecture** | Gemma2 (decoder-only, sliding window attention) |
| **Hidden size** | 2,304 |
| **Layers** | 26 |
| **Attention heads** | 8 (GQA: 4 KV heads) |
| **Context window** | 8,192 tokens |
| **Vocabulary** | 256,000 tokens |
| **BF16 memory footprint** | ~5.2 GB |
| **Fits on TPU v5e (16 GB)** | ✅ Yes (BS=1, 8) |
| **Tokenizer** | SentencePiece |
| **License** | Gemma License |
| **Notes** | Google's native TPU model. Uses sliding window + global attention alternation. Best-case scenario for TPU XLA optimization. |

### 2.4 Mistral 7B v0.3

| Property | Value |
|:---------|:------|
| **HuggingFace ID** | [`mistralai/Mistral-7B-v0.3`](https://huggingface.co/mistralai/Mistral-7B-v0.3) |
| **Parameters** | 7.2B |
| **Architecture** | Mistral (decoder-only, sliding window attention) |
| **Hidden size** | 4,096 |
| **Layers** | 32 |
| **Attention heads** | 32 (GQA: 8 KV heads) |
| **Context window** | 32,768 tokens |
| **Vocabulary** | 32,768 tokens |
| **BF16 memory footprint** | ~14.4 GB |
| **Fits on TPU v5e (16 GB)** | ⚠️ Tight (BS=1 only, ~14.4/16 GB) |
| **Tokenizer** | SentencePiece (BPE) |
| **License** | Apache 2.0 |
| **Notes** | Tests near-capacity operation where memory management is critical. Sliding window KV-cache reduces memory pressure for longer sequences. Relevant for Mistral AI's European datacenter strategy. |

### 2.5 Gemma 2 9B

| Property | Value |
|:---------|:------|
| **HuggingFace ID** | [`google/gemma-2-9b`](https://huggingface.co/google/gemma-2-9b) |
| **Parameters** | 9.2B |
| **Architecture** | Gemma2 (decoder-only, sliding + global attention) |
| **Hidden size** | 3,584 |
| **Layers** | 42 |
| **Attention heads** | 16 (GQA: 8 KV heads) |
| **Context window** | 8,192 tokens |
| **Vocabulary** | 256,000 tokens |
| **BF16 memory footprint** | ~18.4 GB |
| **Fits on TPU v5e (16 GB)** | ❌ Requires multi-chip or quantization |
| **Tokenizer** | SentencePiece |
| **License** | Gemma License |
| **Notes** | Exceeds single v5e HBM. Benchmarked on simulated metrics using validated per-layer scaling from smaller models. Validates RunuX-AI's memory-efficient paged KV-cache at scale. |

### 2.6 Gemma 2 27B

| Property | Value |
|:---------|:------|
| **HuggingFace ID** | [`google/gemma-2-27b`](https://huggingface.co/google/gemma-2-27b) |
| **Parameters** | 27.2B |
| **Architecture** | Gemma2 (decoder-only, sliding + global attention) |
| **Hidden size** | 4,608 |
| **Layers** | 46 |
| **Attention heads** | 32 (GQA: 16 KV heads) |
| **Context window** | 8,192 tokens |
| **Vocabulary** | 256,000 tokens |
| **BF16 memory footprint** | ~54.4 GB |
| **Fits on TPU v5e (16 GB)** | ❌ Requires 4+ chips or INT4 quantization |
| **Tokenizer** | SentencePiece |
| **License** | Gemma License |
| **Notes** | Multi-chip extrapolation based on validated scaling laws. Included to project datacenter-scale economics. |

### Model Selection Rationale

| Category | Models | Rationale |
|:---------|:-------|:----------|
| **Small (< 1B)** | Qwen 2.5 0.5B | Validates overhead measurement, high throughput regime |
| **Medium (1–3B)** | DeepSeek R1 1.5B, Gemma 2 2B | Sweet spot for edge/mobile inference |
| **Large (7B)** | Mistral 7B v0.3 | Industry standard, European AI relevance |
| **XL (9B+)** | Gemma 2 9B, Gemma 2 27B | Scaling validation, multi-chip projection |

---

## 3. Benchmark Protocol

### 3.1 MLPerf-Aligned Methodology

This benchmark follows the [MLPerf Inference](https://mlcommons.org/benchmarks/inference-datacenter/) 
methodology for autoregressive text generation:

1. **Warm-up Phase** (3 iterations)
   - Full inference pass with identical input
   - Ensures JIT compilation, XLA graph tracing, and memory allocation complete
   - Results discarded

2. **Measurement Phase** (10 iterations)
   - Identical input and generation parameters
   - Wall-clock latency measured with `time.perf_counter()` (nanosecond resolution)
   - **Median** reported as primary metric (robust to GC/preemption outliers)
   - Full distribution retained in JSON results for statistical analysis

3. **Token Counting**
   - Only **newly decoded tokens** count toward throughput
   - Prefill tokens are excluded
   - Throughput = `new_tokens / decode_latency`

### 3.2 Input Configuration

| Parameter | Value | Rationale |
|:----------|:------|:----------|
| **Prompt** | "The future of artificial intelligence in sustainable computing is" | Neutral topic, avoids model-specific behavior |
| **Input length** | 12 tokens (natural) | Short prompt tests decode-dominated workload |
| **Decode length** | 128 tokens | Standard benchmark length per MLPerf |
| **Decoding strategy** | Greedy (`do_sample=False`) | Deterministic, reproducible |
| **KV-cache** | Enabled (`use_cache=True`) | Standard inference configuration |
| **Precision** | BF16 (bfloat16) | Native TPU precision, no quantization |
| **Batch sizes** | 1, 8 | BS=1 for latency, BS=8 for throughput |

### 3.3 Framework Configurations

#### PyTorch/XLA (Primary Baseline)

```python
import torch
import torch_xla.core.xla_model as xm
from transformers import AutoModelForCausalLM, AutoTokenizer

device = xm.xla_device()
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.bfloat16
).to(device)
model.eval()

# Synchronize before timing
xm.mark_step()
t0 = time.perf_counter()
with torch.no_grad():
    output = model.generate(input_ids, max_new_tokens=128, do_sample=False)
    xm.mark_step()  # Force XLA execution
t1 = time.perf_counter()
```

- **No** custom kernels or `torch.compile`
- **No** manual XLA graph manipulation
- **Default** PJRT backend (auto-detected)
- **Unmodified** HuggingFace model weights

#### JAX/Flax (Secondary Baseline)

```python
import jax
import jax.numpy as jnp
from transformers import FlaxAutoModelForCausalLM

model = FlaxAutoModelForCausalLM.from_pretrained(
    model_id, dtype=jnp.bfloat16
)
# Default JIT compilation, no custom PALLAS kernels
```

#### JetStream (Google Optimized)

- Google's optimized serving engine for TPU
- Uses fused attention kernels and optimized KV-cache
- Represents the best publicly-available TPU optimization

#### vLLM (TPU Backend)

- vLLM with TPU backend via torch_xla
- PagedAttention for KV-cache management
- Standard deployment configuration

#### RunuX-AI (This Work)

- Results only — proprietary runtime not disclosed
- Same model weights, same input, same decoding parameters
- Measured using identical wall-clock methodology

---

## 4. Derived Metrics

### 4.1 Energy per Token

```
energy_joules_per_token = TDP_watts / throughput_tokens_per_second
```

- Uses **TDP** (200W) as a conservative upper bound
- Actual draw may be 120–180W during memory-bound decode phases
- This means our energy numbers are **pessimistic** (real savings may be higher)

### 4.2 CO₂ per 1,000 Tokens

```
gCO2_per_1k_tokens = energy_joules_per_token × 1000 / 3_600_000 × grid_intensity_gCO2_per_kWh
```

Regional carbon intensities (source: IEA World Energy Outlook 2024 + national grid operators):

| Region | gCO₂/kWh | Source |
|:-------|:--------:|:-------|
| 🇸🇪 Sweden | 20 | Energimyndigheten (Swedish Energy Agency) |
| 🇫🇷 France | 56 | RTE (Réseau de Transport d'Électricité) |
| 🇳🇴 Norway | 8 | NVE (Norwegian Water Resources & Energy Directorate) |
| 🇫🇮 Finland | 73 | Fingrid |
| 🇩🇪 Germany | 350 | Bundesnetzagentur |
| 🇺🇸 USA (avg) | 386 | EIA (U.S. Energy Information Administration) |
| 🇬🇧 UK | 180 | National Grid ESO |
| 🇯🇵 Japan | 450 | METI |
| 🇨🇳 China | 540 | China Electricity Council |
| 🇮🇳 India | 632 | Central Electricity Authority |

See [`carbon_factors.json`](carbon_factors.json) for complete data.

### 4.3 Cost per Million Tokens

```
cost_per_M_tokens_USD = (1_000_000 / throughput_tps) / 3600 × tpu_price_per_hour
```

| Pricing Tier | Rate | Notes |
|:-------------|:-----|:------|
| On-demand | $1.20/chip-hr | Primary benchmark price |
| Spot / Preemptible | ~$0.40/chip-hr | Subject to availability |
| 1-year CUD | ~$0.77/chip-hr | 36% discount |
| 3-year CUD | ~$0.54/chip-hr | 55% discount |

### 4.4 MXU Utilization

```
mxu_utilization = actual_tflops / peak_bf16_tflops × 100
actual_tflops = (2 × model_flops_per_token × throughput_tps) / 1e12
```

Where `model_flops_per_token ≈ 2 × num_parameters` for dense transformer decode.

---

## 5. Reproducibility Notes

### What You Can Reproduce

- ✅ **PyTorch/XLA baseline** measurements (using `benchmark_baselines.py`)
- ✅ **Energy/CO₂/cost** derivations (using published throughput + formulas above)
- ✅ **Statistical analysis** (all individual run times included in JSON)

### What You Cannot Reproduce (IP Protected)

- ❌ RunuX-AI internal runtime algorithms
- ❌ MXU optimization techniques (patent pending)
- ❌ Memory management and scheduling strategies

### Expected Variance

| Source | Impact | Mitigation |
|:-------|:-------|:-----------|
| XLA compilation | ±2% first-run | 3-iteration warmup |
| Thermal throttling | ±1% | Median over 10 runs |
| HBM temperature | ±0.5% | Short benchmark window |
| Spot preemption | N/A | On-demand instances used |
| Model download | 0% | Pre-cached before timing |

### Validation Checklist

```bash
# 1. Verify TPU device is detected
python3 -c "import torch_xla.core.xla_model as xm; print(xm.xla_device())"
# Expected: xla:0

# 2. Verify BF16 support
python3 -c "import torch; import torch_xla.core.xla_model as xm; \
  t = torch.ones(2,2, dtype=torch.bfloat16, device=xm.xla_device()); \
  print(t.dtype)"
# Expected: torch.bfloat16

# 3. Run benchmark
python3 benchmark_baselines.py
```

---

## 6. Citation

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

*© 2026 Xavier Callens / Socrate AI Lab · Apache-2.0*
