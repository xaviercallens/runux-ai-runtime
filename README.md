# RunuX AI Runtime

> **PROPRIETARY & CONFIDENTIAL** — This repository contains trade secrets and proprietary technology.

## Overview

RunuX AI Runtime is a kernel-level AI acceleration stack for RISC-V processors, delivering **2–5× inference performance gains** over userspace solutions.

### Key Capabilities

| Feature | Description |
|---|---|
| **Kernel-Level Tensors** | Zero-copy, zero-syscall tensor operations |
| **RVV SIMD Kernels** | RISC-V Vector 1.0 optimized (256/1024-bit) |
| **TurboQuant** | KV-cache compression: 16-bit → 3-bit (5× savings) |
| **Native FP8** | Zero dequantization overhead on SpacemiT K3 A100 cores |
| **Federated Learning** | Privacy-preserving distributed training (DP + LoRA) |
| **C FFI Bridge** | Python/TensorFlow/PyTorch interop |

### Supported Hardware

| Platform | SoC | AI TOPS | Status |
|---|---|---|---|
| Banana Pi BPI-F3 | SpacemiT K1 | 2.0 | ✅ Validated |
| Firefly AIBOX-K3 | SpacemiT K3 | 60.0 | 🔧 In progress |

### Supported Models

| Model | K1 (8GB) | K3 (32GB) |
|---|---|---|
| Qwen 2.5 0.5B–3B | ✅ | ✅ |
| DeepSeek R1 1.5B–7B | ⚠️–❌ | ✅ FP8 native |
| DeepSeek R1 14B | ❌ | ✅ INT4 |
| Mistral 7B | ❌ | ✅ FP8 native |

## License

**Proprietary**. See [LICENSE](./LICENSE).

For licensing inquiries: xavier.callens@socrate.ai

## Patents

Certain aspects of this software are covered by pending patent applications.

---

*© 2026 Xavier Callens / Socrate AI. All Rights Reserved.*
