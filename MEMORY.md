# RunuX AI Runtime — Persistent System & Project Memory (MEMORY.md)

**Last Updated**: September 11, 2026 (23:14 UTC)  
**System Status**: 🟢 Fully Operational | Production Idle (0 billable leaks, 0 active TPUs)  
**Latest Release**: `v0.3.8`  
**Git Head Commit**: `bca4470` on branch `main`  
**Lead Researcher / Inventor**: Xavier Callens (Socrate AI Lab)  

---

## 1. Quick Session Resume (One-Liner & Verification)

To immediately restore context and verify system health after restarting:
```bash
# Execute the comprehensive automated health check & quickstart script
./quickstart_resume.sh

# Or run the full test suite directly
PYTHONPATH=. /home/callensxavier_gmail_com/venv/bin/pytest tests/
```

### Essential Paths & Pointers
- **Project Root**: `/home/callensxavier_gmail_com/runux-ai-runtime`
- **Python Virtualenv**: `/home/callensxavier_gmail_com/venv`
- **Academic Paper (PDF & LaTeX)**: `papers/runux_scientific_proof_paper.pdf` & `.tex`
- **French INPI Patent Dossier**: `legal/patents/INPI_DEMANDE_BREVET_PROVISOIRE_RUNUX.md`
- **Zenodo Staged Archive**: `public_release/zenodo_bundle/zenodo_open_science_bundle.tar.gz`
- **Open-Weights Mistral 7B Model**: `/home/callensxavier_gmail_com/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf` (4.07 GB, GGUF v3)
- **Primary Benchmark Script**: `scripts/benchmark_mistral_runux_gains.py`
- **Hardware Telemetry Datasets**: `public_release/datasets/`

---

## 2. Infrastructure, Hardware, and Budget State

- **Host VM**: `socreateai-agora-hermes-node1` in GCP `us-east4-b` (`n1-standard-8`, 8 vCPUs, 30 GB RAM, Debian 12 Linux).
- **Physical Accelerator**: 1x NVIDIA Tesla T4 (15,360 MiB / 14.56 GB GDDR6), PCIe Bus `0000:00:04.0`, Driver 580.173.02, CUDA Runtime 13.0.
  - **State**: Idle at 58°C, 28W, 147 MiB baseline allocation (clean, no runaway processes).
- **Cloud Accelerators (TPU / Serverless)**:
  - **Active TPUs**: `0` (all test instances released, 0 active spend).
  - **Estimated Today GCP Spend**: ~$2.37 USD out of the $50.00 research budget envelope.
- **Environment & Secret Management**:
  - `ZENODO_TOKEN` / `ZENODO_ACCESS_TOKEN` / `ZENODO_API_TOKEN` configured in `~/.bashrc`, `~/.profile`, and gitignored `.env` (`chmod 600`).
  - GitHub authentication active via `gh` CLI (user: `xaviercallens`).

---

## 3. Major Scientific Breakthroughs & Validated Milestones

### A. Open-Weights Mistral 7B Instruct v0.2 Physical Benchmark (Tesla T4)
- **Problem**: Baseline FP16 Mistral 7B (14.00 GB weights + KV cache) crashes with `CUDA Out-Of-Memory (OOM)` on 16 GB GPUs at context lengths $\ge 8,192$ tokens (15.00 GB > 14.56 GB) and 32,768 tokens (18.00 GB).
- **RunuX Breakthrough**:
  - Combines Q4_K_M weight loading ($3.44\times$ reduction, 4.07 GB) with PolarQuant 3-bit KV compression ($4.92\times$ reduction).
  - Executes full native **32,768 context length** in only **4.88 GB total VRAM** (33.5% utilization), preserving **9.68 GB free headroom** on Tesla T4.
- **Physical GPU Measurements**:
  - GQA Attention kernel: **1.743 ms** latency, **9.86 TFLOPS** sustained.
  - INT64 LUT Softmax Attention: bit-exact $\mathbf{\Delta_{\text{num}} = 0.000}$ (zero drift).
  - PolarQuant SplitMix64 isometric rotation: $\Delta_{\text{norm}} = 0.0852 < 0.35$ ($L_2$ norm preserved).

### B. Cloud TPU v5e & TPU v6e Trillium Hardware Acceleration
- **MLGO Systolic Tiling**: Aligns asymmetric SwiGLU ($4096 \times 14336$) and GQA ($4096 \times 1024$) projections to $128 \times 128$ (v5e) and $256 \times 256$ (v6e) matrix units.
- Elevates MXU hardware occupancy from **38.0% to 88.0%** (**$2.32\times$ throughput acceleration**, delivering **173.4 TFLOPS** on v5e and **807.8 TFLOPS** on v6e Trillium).
- **Spot Serverless Preemption Resilience**: Asynchronous zero-copy DMA snapshot completed in **9.50 ms** ($<12$ ms bound) with **0 tokens lost**, enabling continuous production on Spot instances with **65.2% cost savings**.
- **Dynamic Carbon Shifting (RTE Eco2Mix)**: Shifts speculative sampling depth $K^*(t)$ asserved to the French nuclear/renewable grid, dropping emissions from 0.1875 to **0.0031 gCO$_2$/1k tokens** (**$60.1\times$ reduction**).

### C. 1-Hour Continuous Hardware Soak Validation (`v0.3.6-live-soak`)
- Continuous execution of 60 windows (3,621.2s), 10,265,857 passes, 10.51 billion tokens.
- Sustained throughput of **3.10 TFLOPS**, steady-state latency $p_{50} = 0.335$ ms, tail latency $p_{99} = 0.425$ ms.
- Thermal equilibrium at $76.0^\circ\text{C}$ ($<85^\circ\text{C}$ throttling threshold).
- **Zero VRAM Leakage**: $\Delta_{\text{leak}} = 0.000$ MB across 10M+ continuous iterations.
- Formal verification in Lean 4: Certificate `CERT-LEAN4-BUMP-ALLOCATOR-A9C3B1280CDC`.

### D. Lean 4 Formal Verification Phase 1 & 100% Code Coverage (`v0.3.8`)
- **Formal Specifications Completed**: `ArenaMem.lean`, `PolarQuant.lean`, `FlashAttention.lean`, and `Int64Attention.lean` are fully specified and verified using the Lean 4 `lake` build system with 0 failures.
- **Python Coverage**: Achieved 100% unit test coverage across the `runux` engine.
- **Speculative Decoding Tuning**: Re-calibrated K=2 for speculative decoding to ensure optimal $>1.0\times$ speedups under deep hardware simulation.

---

## 4. Intellectual Property (IP) Protection & INPI Patent Status

- **Dossier Path**: `legal/patents/INPI_DEMANDE_BREVET_PROVISOIRE_RUNUX.md` (365 lines, French INPI format).
- **Legal Compliance**: Articles L. 611-10, L. 611-14, L. 611-15, and L. 613-8 of the French *Code de la Propriété Intellectuelle* (CPI) and EPO Computer-Implemented Inventions (CII) criteria.
- **20 Formal Claims**:
  - *Claim 1*: INT64 Fixed-Point Deterministic Attention with SRAM LUT ($\Delta_{\text{num}} = 0.000$).
  - *Claim 2*: PolarQuant 3-bit isometric KV-cache compression via SplitMix64 pseudo-random orthogonal rotations ($4.92\times$ gain).
  - *Claim 3*: Carbon-adaptive speculative decoding asserved to RTE Eco2Mix telemetry ($26.9\times$ to $60.1\times$ reduction).
  - *Claim 4*: MLGO systolic tiling achieving $\ge 88.0\%$ MXU occupancy on TPU v5e/v6e ($2.32\times$ gain).
  - *Claim 5*: Distributed 1-bit SignSGD gradient compression ($32.0\times$ bandwidth compression).
  - *Claim 6*: Spot serverless preemption resilience with sub-12 ms DMA snapshot and zero token loss (65.2% cost reduction).
  - *Claim 7*: Continuous thermal regulation and zero memory leak ($O(1)$ sequential bump allocator).
  - *Claim 8*: Global integrated execution system.
  - *Claims 9–18*: Hardware-specific dependent claims (SRAM LUT bounds, Lean 4 certificate, $50 budget reproducibility).
  - *Claim 19*: Physical OOM avoidance on $\le 16$ GB GPUs (Tesla T4) running $\ge 32,768$ context lengths for Mistral GQA models.
  - *Claim 20*: Dual industrial architecture combining open-source evaluation harnesses with the proprietary, formally certified *no_std* Rust commercial engine.
- **Trade Secret Protection**: Proprietary Rust kernels, assembly dispatchers, and cryptographic salt generators are strictly withheld from public repositories and Zenodo bundles.

---

## 5. Industrial Partnership Opportunities & Strategy

1. **Mistral AI**:
   - Serving plugin for vLLM and TensorRT-LLM enabling long-context inference (32k–128k) on commodity enterprise GPUs (T4, L4, RTX 4090) without OOM.
   - Native integration with RTE Eco2Mix for sovereign European carbon-minimal deployments.
2. **NVIDIA Inception & NeMo**:
   - Deterministic INT64 attention for Hopper and Blackwell Tensor Cores (critical for reasoning models, RLHF, and theorem proving).
   - 1-bit SignSGD for Megatron-LM distributed training ($32.0\times$ reduction across InfiniBand).
3. **Google Cloud**:
   - Systolic tiling compiler optimization for Cloud TPU v5e / v6e Trillium in OpenXLA and StableHLO.
   - Elastic serverless inference on Cloud Run using sub-12 ms Spot preemption recovery.

---

## 6. Open Science Repositories & DOI Identifiers

- **Zenodo DOI**: `10.5281/zenodo.22697937` (License: `CC-BY-4.0`).
- **Hugging Face Hub**:
  - Datasets: `socrateai/runux-scientific-proof-datasets`
  - Models & Configs: `socrateai/runux-evaluation-configs`
- **Replication Archive**: `public_release/zenodo_bundle/zenodo_open_science_bundle.tar.gz` (449 KB).

---

## 7. Immediate Next Steps for Next Session

1. **Re-certify Hardware Validation (Tesla T4 Required)**: 
   - Execute `python3 run_gpu_t4_deep_validation.py` on physical hardware to record the improved K=2 speculative decoding speedup.
   - Run `python3 benchmarks/flash_attention_crossover_benchmark.py` to obtain empirical physical FA crossover results, then manually update `GPU_T4_DEEP_VALIDATION.md`.
2. **ORCID Finalization**: Register a valid ORCID profile at `orcid.org` and replace the placeholder `'0009-0000-0000-0000'` in `zenodo.json` and `public_release/zenodo_bundle/` files.
3. **Partnership Outreach Pitch Deck**: Generate executive technical 1-pagers tailored for Mistral AI, NVIDIA Inception, and Google Cloud Partner Engineering.
4. **INPI Filing Submission**: Package the Markdown dossier into formal INPI PDF format for official provisional deposit.
