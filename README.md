# RunuX AI Runtime

> **Confidential** — © 2026 Xavier Callens / Socrate AI. All rights reserved.  
> Licensed under LicenseRef-RunuX-Commercial. Unauthorized distribution prohibited.

## Overview

RunuX is a **no_std Rust AI inference and training runtime** optimized for RISC-V edge hardware. It delivers extreme energy efficiency for running open-weight LLMs (DeepSeek, Qwen, Mistral) on SpacemiT K1/K3 RISC-V processors.

### Key Differentiators

| Feature | RunuX | llama.cpp | ONNX Runtime |
|---------|-------|-----------|--------------|
| Language | `no_std` Rust | C++ | C++ |
| RISC-V RVV native | ✅ 256/1024-bit | ❌ Scalar only | ❌ |
| FP8 inference (K3) | ✅ Zero overhead | ❌ | ❌ |
| KV cache compression | ✅ TurboQuant 3-bit | ❌ | ❌ |
| FlashAttention | ✅ Tiled O(N) | ❌ | ❌ |
| Federated learning | ✅ FedAvg + DP | ❌ | ❌ |
| LoRA training | ✅ AdamW, cosine LR | ❌ | ❌ |
| Memory safety | ✅ Rust guarantees | ❌ | ❌ |
| CO2 tracking | ✅ Per-token | ❌ | ❌ |

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    runux-report (CLI binary)                     │
├─────────┬──────────┬──────────┬──────────┬──────────┬──────────┤
│sim_infer│sim_train │sim_bench │perf_model│ai_bridge │federated │
├─────────┴──────────┴──────────┴──────────┴──────────┴──────────┤
│                    Optimization Layer                           │
│  flash_attention │ turbo_quant │ speculative │ arena_mem        │
├──────────────────┴─────────────┴─────────────┴─────────────────┤
│                    Model Pipeline                               │
│  transformer │ gguf_loader │ tokenizer │ power_monitor          │
├──────────────┴─────────────┴───────────┴───────────────────────┤
│                    Core Runtime                                 │
│  ai_runtime │ rvv_simd │ k3_a100 │ gpu_compute                 │
└─────────────┴──────────┴─────────┴─────────────────────────────┘
                    ↓ Cross-compile ↓
         riscv64gc-unknown-linux-gnu (QEMU / Hardware)
```

## Crate Summary (18 crates, ~10K lines)

| Crate | Lines | Purpose |
|-------|-------|---------|
| `ai_runtime` | 774 | Core types, InferenceEngine trait, ModelConfig |
| `rvv_simd` | 615 | RVV vectorized kernels (gemv, softmax, RMSNorm) |
| `turbo_quant` | 617 | PolarQuant KV-cache compression (3-bit) |
| `gguf_loader` | 834 | GGUF file parser and weight loader |
| `transformer` | 725 | RoPE, GQA attention, SwiGLU FFN |
| `flash_attention` | 556 | Tiled FlashAttention (O(N) memory) |
| `arena_mem` | 566 | Bump allocator + Paged KV cache |
| `speculative` | 533 | Draft-verify speculative decoding |
| `tokenizer` | 478 | BPE encoder/decoder |
| `power_monitor` | 471 | Energy estimation + CO2 tracking |
| `perf_model` | 640 | Roofline analytical performance model |
| `sim_inference` | 561 | End-to-end inference simulation |
| `sim_train` | 1032 | LoRA training + federated learning simulation |
| `sim_bench` | 455 | Benchmark runner |
| `federated` | 758 | Federated learning (FedAvg, DP, secure aggregation) |
| `ai_bridge` | 455 | Python FFI bridge |
| `k3_a100` | 86 | K3 A100 AI core driver |
| `gpu_compute` | 54 | PowerVR BXM GPU compute |

## Target Hardware

| Board | SoC | CPU | RVV | AI | RAM | Price |
|-------|-----|-----|-----|-----|-----|-------|
| **BPI-F3** | SpacemiT K1 | 8× X60 | 256-bit | 2 TOPS | 4-16 GB | ~€80 |
| **AIBOX-K3** | SpacemiT K3 | 8× X100 + 8× A100 | 1024-bit | 60 TOPS | 8-32 GB | ~€524 |

## Supported Models

| Model | Params | Quant | K1 (4GB) | K3 (8GB) | K3 (32GB) |
|-------|--------|-------|----------|----------|-----------|
| Qwen 2.5 0.5B | 0.5B | Q4_K_M | ✅ ~12 tok/s | ✅ ~45 tok/s | ✅ |
| DeepSeek R1 1.5B | 1.5B | Q4_K_M | ✅ ~8 tok/s | ✅ ~30 tok/s | ✅ |
| Qwen 2.5 7B | 7B | FP8 | ❌ OOM | ⚠️ ~15 tok/s | ✅ ~20 tok/s |
| DeepSeek R1 14B | 14B | Q4_K_M | ❌ OOM | ❌ OOM | ✅ ~10 tok/s |

## Quick Start

```bash
# Build (host)
cargo check --workspace

# Cross-compile for RISC-V
cargo build --target riscv64gc-unknown-linux-gnu --release

# Run simulation report
cargo run --bin runux-report

# Run tests
cargo test --workspace -- --test-threads=1
```

## Project Structure

```
runux-ai-runtime/
├── Cargo.toml              # Workspace + binary manifest
├── src/main.rs             # runux-report CLI binary
├── crates/
│   ├── ai_runtime/         # Core types and traits
│   ├── rvv_simd/           # RISC-V vector kernels
│   ├── turbo_quant/        # KV-cache compression
│   ├── flash_attention/    # Tiled attention
│   ├── arena_mem/          # Memory allocator
│   ├── speculative/        # Speculative decoding
│   ├── transformer/        # Transformer layers
│   ├── gguf_loader/        # GGUF parser
│   ├── tokenizer/          # BPE tokenizer
│   ├── power_monitor/      # Energy/CO2 tracking
│   ├── perf_model/         # Performance estimator
│   ├── sim_inference/      # Inference simulation
│   ├── sim_train/          # Training simulation
│   ├── sim_bench/          # Benchmarks
│   ├── federated/          # Federated learning
│   ├── ai_bridge/          # Python FFI
│   ├── k3_a100/            # K3 AI core driver
│   └── gpu_compute/        # GPU compute
├── benchmarks/             # Benchmark harness
└── .github/workflows/      # CI (QEMU RISC-V)
```

## License

**LicenseRef-RunuX-Commercial** — Proprietary license.  
Contact: xavier.callens@socrate.ai
