# RunuX AI Runtime — Partner One-Pager

## The Problem

Edge AI inference on RISC-V hardware is **40–60% slower** than it needs
to be. Current solutions (llama.cpp, ONNX Runtime) run entirely in
userspace, wasting cycles on syscalls, memory copies, and context switches.

## The Solution

**RunuX AI Runtime** is the first kernel-level AI acceleration stack for
RISC-V. By integrating tensor operations directly into the OS kernel, we
eliminate the abstraction overhead and unlock the full potential of
next-generation RISC-V AI silicon.

## Key Results

| Metric | Improvement |
|---|---|
| Inference throughput | **2–5× faster** |
| Memory overhead | **10× less** |
| Context window | **5× longer** (TurboQuant) |
| Security | **Zero CVE surface** (Rust) |

## Supported Platforms

| SoC | AI TOPS | Status |
|---|---|---|
| SpacemiT K1 (BPI-F3) | 2.0 | ✅ Validated |
| SpacemiT K3 (AIBOX-K3) | 60.0 | 🔧 In progress |

## Supported Models

DeepSeek R1 (1.5B–14B) • Qwen 2.5 (0.5B–14B) • Mistral 7B

## Technology

- **Language**: Rust (memory-safe, zero buffer overflows)
- **Architecture**: Kernel module (no userspace overhead)
- **Quantization**: Native FP8, INT4 (GGUF Q4_K_M compatible)
- **Privacy**: Federated learning with differential privacy
- **License**: Commercial (dual-license model)

## Partnership Opportunities

| Model | Description |
|---|---|
| **Silicon Partner** | Pre-install RunuX on your RISC-V boards |
| **Model Partner** | Optimized inference for your LLM on RISC-V |
| **Enterprise License** | Deploy RunuX in your AI edge products |
| **Joint Development** | Co-develop kernel AI features |

## Contact

**Xavier Callens** — Founder & CTO
- Email: xavier.callens@socrate.ai
- LinkedIn: linkedin.com/in/xaviercallens
- GitHub: github.com/xaviercallens

---

*© 2026 Xavier Callens / Socrate AI. All Rights Reserved.*
*Patents pending (EU, China, US).*
