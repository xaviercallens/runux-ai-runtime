<![CDATA[# Benchmark Methodology

## Hardware Specifications

| Component | Specification |
|-----------|--------------|
| Accelerator | Google Cloud TPU v5e (single chip) |
| Compute (BF16) | 197 TFLOPS |
| HBM Capacity | 16 GB |
| HBM Bandwidth | 819.2 GB/s |
| TDP | 200W |
| Host CPU | Intel Xeon (Emerald Rapids) |
| Host RAM | 48 GB DDR5 |
| Software | tpu-ubuntu2204-base |

## Benchmark Protocol

### MLPerf-Aligned Methodology
This benchmark follows the [MLPerf Inference](https://mlcommons.org/benchmarks/inference-datacenter/) 
methodology for autoregressive text generation:

1. **Warm-up Phase**: 3 full inference passes are executed and discarded to ensure
   JIT compilation, memory allocation, and XLA graph tracing are complete.

2. **Measurement Phase**: 10 inference passes are executed. The **median latency** is
   reported as the primary metric to eliminate outlier effects from garbage collection
   or preemption events.

3. **Token Counting**: Only newly generated (decoded) tokens are counted for throughput.
   Prefill tokens are excluded from the throughput calculation.

### Input Configuration
- **Input prompt**: 512 tokens (padded/truncated to exact length)
- **Decode length**: 128 tokens (greedy decoding, `do_sample=False`)
- **Precision**: BF16 (brain floating-point 16) throughout
- **Batch sizes**: 1, 8, 32

### Framework Configurations

#### PyTorch (torch_xla)
```python
import torch
import torch_xla.core.xla_model as xm
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.bfloat16
).to(xm.xla_device())
```
- torch_xla version: latest pip release
- XLA optimization: default (no manual graph manipulation)
- No custom kernels or torch.compile applied

#### JAX/Flax
```python
import jax.numpy as jnp
model = FlaxAutoModelForCausalLM.from_pretrained(
    model_id, dtype=jnp.bfloat16
)
```
- JAX version: latest pip release
- XLA optimization: default JIT compilation
- No custom PALLAS kernels

### Energy Calculation

Energy per token is derived from:
```
joules_per_token = TDP_watts / throughput_tokens_per_second
```

This uses TDP (Thermal Design Power) as a conservative upper bound.
Actual power draw may be lower during memory-bound decode phases.

### CO₂ Calculation

Regional CO₂ per 1,000 tokens:
```
gCO2_per_1k = joules_per_token × 1000 / 3600 × grid_carbon_intensity_gCO2_per_kWh / 1000
```

Grid carbon intensities are sourced from the IEA World Energy Outlook 2024
and national grid operators (see `carbon_factors.json`).

### Cost Calculation

Cost per million tokens:
```
cost_per_M_tokens = (1_000_000 / throughput_tps) / 3600 × tpu_price_per_hour
```

TPU v5e pricing: $1.20/chip-hr (on-demand), $0.40/chip-hr (spot).

## Reproducibility Notes

- All models are loaded from HuggingFace Hub without modification
- No quantization, pruning, or distillation is applied
- Results may vary by ±5% due to spot instance scheduling and thermal conditions
- The `benchmark_baselines.py` script is self-contained and requires only 
  `torch`, `torch_xla[tpu]`, `transformers`, and `accelerate`
]]>
