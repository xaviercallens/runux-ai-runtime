# RunuX RISC-V AI/ML Runtime

## Overview

RunuX provides kernel-level AI/ML acceleration on RISC-V hardware through five dedicated crates. These enable native LLM inference (DeepSeek, Qwen), KV-cache compression (TurboQuant), and federated learning across edge clusters.

## Supported Hardware

| Platform | SoC | AI Performance | RAM | GPU | Price |
|---|---|---|---|---|---|
| **Banana Pi BPI-F3** | SpacemiT K1 (8× X60) | 2.0 TOPS | 4–8 GB | — | ~€80 |
| **Firefly AIBOX-K3** | SpacemiT K3 (8× X100 + 8× A100) | **60 TOPS** | 8–32 GB | PowerVR BXM-4-64 | ~€524 |

## Crate Architecture

```
                    ┌────────────────────────┐
                    │  User Applications     │
                    │  (Python, C++, Rust)    │
                    └────────┬───────────────┘
                             │
                    ┌────────▼───────────────┐
                    │    ai_bridge           │
                    │    C FFI Layer         │
                    └──┬─────────────┬──────┘
                       │             │
            ┌──────────▼───┐  ┌─────▼──────────┐
            │  rvv_simd    │  │  turbo_quant    │
            │  Vector ops  │  │  KV compression │
            └──────────┬───┘  └─────┬──────────┘
                       │            │
                    ┌──▼────────────▼──┐
                    │   ai_runtime     │
                    │   Tensor types   │
                    └──────────────────┘
                             │
                    ┌────────▼───────────────┐
                    │    federated           │
                    │    Distributed ML      │
                    └────────────────────────┘
```

## Crates

### `ai_runtime` — Core Tensor Types

Foundational data structures for AI/ML on RISC-V:

- **DataType**: FP32, FP16, BF16, FP8 (K3 native), INT8, INT4, Binary
- **DeviceType**: CPU, A100AiCore, PowerVrGpu, NvmeOffload
- **TensorDescriptor**: C-compatible, zero-copy tensor metadata
- **ModelConfig**: GGUF model loading with RAM estimation
- **HardwareCaps**: Runtime hardware detection (K1 vs K3)
- **ModelRegistry**: Pre-configured profiles for DeepSeek R1, Qwen 2.5

### `rvv_simd` — Vectorized Kernels

RISC-V Vector (RVV 1.0) accelerated ML operations:

- Matrix multiplication (FP32 scalar + tiled RVV)
- INT4 dequantization with Q4_K_M GGUF blocks
- Softmax (numerically stable)
- Layer normalization / RMS normalization
- SiLU / GELU activation functions
- Rotary Positional Embeddings (RoPE)
- All operations have scalar fallbacks for any architecture

### `turbo_quant` — KV-Cache Compression

TurboQuant (Google, ICLR 2026) for extending context windows:

- **PolarQuant**: Orthogonal rotation for variance equalization
- **QJL**: Johnson-Lindenstrauss error correction
- Compresses KV cache from 16-bit to ~3-bit (5× savings)
- Enables 32K+ context on 32GB AIBOX-K3

### `ai_bridge` — FFI Layer

C-compatible API for Python/C++ framework integration:

- `runux_tensor_create/free` — Tensor lifecycle
- `runux_matmul` — Hardware-dispatched matrix multiply
- `runux_softmax`, `runux_silu` — In-place activations
- `runux_detect_hardware` — Hardware capability detection

### `federated` — Distributed ML

Federated learning support for RISC-V edge clusters:

- **Aggregation**: FedAvg, FedProx, FedYogi, FedAdam
- **Privacy**: Differential privacy (ε-budget), gradient clipping
- **LoRA**: Low-rank adaptation config with param estimation
- **Cluster management**: Node roles, status tracking, TOPS estimation

## Model Compatibility

| Model | Params | BPI-F3 (8GB) | AIBOX-K3 (32GB) |
|---|---|---|---|
| Qwen 2.5 0.5B | 0.5B | ✅ Fast | ✅ Very Fast |
| DeepSeek R1 1.5B | 1.5B | ✅ Good | ✅ Fast |
| Qwen 2.5 3B | 3B | ✅ Usable | ✅ Fast |
| DeepSeek R1 7B | 7B | ❌ OOM | ✅ Good (FP8) |
| Qwen 2.5 14B | 14B | ❌ | ✅ Good (INT4) |

## Building

```bash
# Cross-compile all AI crates for RISC-V
cargo check -p ai_runtime -p rvv_simd -p turbo_quant -p ai_bridge -p federated \
    --target riscv64gc-unknown-none-elf

# Run tests (host)
cargo test -p ai_runtime -p rvv_simd -p turbo_quant -p ai_bridge -p federated
```
