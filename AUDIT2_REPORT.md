# RunuX AI Runtime — Intensive Codebase Audit Report №2

**Audit Date**: `2026-09-11`  
**Auditor**: Antigravity AI Agent (Claude Sonnet 4.6)  
**Repository**: `https://github.com/xaviercallens/runux-ai-runtime`  
**HEAD Commit**: `67c13a5` — `docs(zenodo): update permanent Zenodo DOI`  
**Baseline**: First audit tag `v0.3.1-audit` / commit `a059a93`  
**Rebase Status**: ✅ Fast-forwarded 13 commits from remote `origin/main`  
**Scope**: Full workspace — 24 Rust crates, 8 Python modules, 7 test suites, all scripts & docs  

---

## Executive Summary

This second intensive audit reviews all code introduced since the Phase 1 deep audit (`AUDIT_REPORT.md`) through the current HEAD. The repository has matured substantially: from a simulation-only Rust engine to a dual-architecture system combining a commercial-grade `no_std` Rust runtime with a Python acceleration layer (`runux/`) validated on physical Tesla T4 hardware.

| Category | Status | Severity |
|:---|:---:|:---:|
| **Build System — Lockfile Collision** | ❌ CRITICAL | P0 |
| **Absolute Path Hardcode in `turbo_quant/Cargo.toml`** | ❌ CRITICAL | P0 |
| **Speculative Decoding Regression (0.64× < 1.0×)** | ⚠️ HIGH | P1 |
| **Python Package: Missing `torch` in requirements.txt** | ⚠️ HIGH | P1 |
| **`transformer_engine.py` import chain not verified** | ⚠️ MEDIUM | P2 |
| **GPU-gated tests run without skip guard** | ⚠️ MEDIUM | P2 |
| **FlashAttention tiled latency regression (34× slower at S=1024)** | ⚠️ MEDIUM | P2 |
| **Zenodo ORCID Placeholder** | ⚠️ MEDIUM | P2 |
| **Broken onboarding symlink: `quickstart_resume.sh`** | ⚠️ MEDIUM | P2 |
| **Cargo workspace profile warnings** | ℹ️ LOW | P3 |
| **`Cargo.lock` gitignored — reproducibility risk** | ℹ️ LOW | P3 |
| **`autoresearch_agent_v4.py` dead code at root** | ℹ️ LOW | P3 |
| **`crates/tpu_pjrt/src/ffi.rs` IP policy alignment** | ℹ️ LOW | P3 |

**Overall health: GOOD with two critical build fixes required.**

---

## 1. Critical Issues (P0 — Must Fix Before Release)

### 1.1 🔴 Build Failure: Cargo Lockfile Package Collision

**File**: `crates/turbo_quant/Cargo.toml`  
**Status**: Unstaged local modification present at audit time  

```toml
# BROKEN (absolute path to a DIFFERENT repository on the developer machine)
ai_runtime = { path = "/home/xavkal/xdev/rust-linux-mini-kernel/crates/ai_runtime" }

# CORRECT (relative path within this workspace)
ai_runtime = { path = "../ai_runtime" }
```

**Impact**: `cargo build` and `cargo test` are completely broken for the entire workspace.  
The Cargo lockfile cannot disambiguate two packages with identical `name = "ai_runtime" v0.1.0` from different filesystem paths.

```
error: package collision in the lockfile: packages ai_runtime v0.1.0
(/home/xavkal/xdev/runux-ai-runtime/crates/ai_runtime) and
ai_runtime v0.1.0 (/home/xavkal/xdev/rust-linux-mini-kernel/crates/ai_runtime)
are different, but only one can be written to lockfile unambiguously
```

**Fix**: Revert `crates/turbo_quant/Cargo.toml` dependency to:
```toml
[dependencies]
ai_runtime = { path = "../ai_runtime" }
```

**Action**: Apply immediately — one-line fix that must precede any CI or partner evaluation. The git diff confirms the original committed file had the correct relative path; this is an accidental developer-machine override.

---

### 1.2 🔴 Absolute Developer Path Hardcoded in Onboarding Script

**File**: `scripts/restart_session.sh` (line ~26)

```bash
VENV_PYTHON="/home/callensxavier_gmail_com/venv/bin/python"
```

This hardcodes the original developer's virtualenv path. It will silently fail for any collaborator, CI runner, or cloud environment. The script is also linked as `quickstart_resume.sh` — the documented onboarding entry point.

**Fix**: Use environment discovery:
```bash
VENV_PYTHON="${VIRTUAL_ENV:+$VIRTUAL_ENV/bin/python}"
if [ -z "$VENV_PYTHON" ] || [ ! -x "$VENV_PYTHON" ]; then
    VENV_PYTHON="$(command -v python3 || command -v python)"
fi
```

---

## 2. High-Priority Issues (P1 — Fix Before Partner Demo)

### 2.1 🟠 Python Package: `torch` Missing from `requirements.txt`

**File**: `requirements.txt`

The entire `runux/` Python package (`gqa_kernel.py`, `paged_cache.py`, `transformer_engine.py`, `polarquant.py`, `deterministic_attn.py`, `signsgd.py`) imports `torch`, but `requirements.txt` only lists:

```
fastapi, uvicorn, pydantic, httpx, transformers, accelerate,
safetensors, sentencepiece, protobuf, numpy, prometheus-client
```

`torch` is absent. This causes an `ImportError` on clean installs, breaking the entire `runux` package.

**Fix**: Add to `requirements.txt`:
```
torch>=2.1.0
```
Create a separate `requirements-gpu.txt` for the CUDA-pinned profile used in T4 benchmarks:
```
torch==2.7.1+cu118 --index-url https://download.pytorch.org/whl/cu118
```

---

### 2.2 🟠 Speculative Decoding Shows Negative Speedup (0.64×)

**File**: `GPU_T4_DEEP_VALIDATION.md`, `gpu_t4_deep_validation_results.json`

```json
"sequential_throughput_tps": 721.8,
"speculative_throughput_tps": 462.1,
"speedup_ratio": 0.64
```

A speedup ratio **< 1.0× means speculative decoding is slower than sequential decoding**. The 36% acceptance rate with K=4 candidates is below the break-even threshold (~60% for a 2:1 draft-to-target size ratio). Most draft tokens are rejected, and the draft model overhead exceeds the gain.

The report currently labels this benchmark "✅ CERTIFIED", which is contradictory and will raise immediate red flags during Mistral AI and NVIDIA partner reviews.

**Recommendations**:
1. Reduce K to 2 for low-acceptance-rate conditions (break-even drops to ~40%)
2. Increase draft model temperature to improve acceptance rate
3. Benchmark acceptance rate across prompt distributions before claiming speedup
4. Relabel this benchmark "IN CALIBRATION" in all external materials

---

## 3. Medium-Priority Issues (P2 — Fix This Sprint)

### 3.1 🟡 `transformer_engine.py` Import Chain Needs Verification

**File**: `runux/transformer_engine.py` (line 9)

```python
from .polarquant import PolarQuantKVCache, PolarQuantConfig
```

`runux/polarquant.py` exists and is listed in `__init__.py`, but with `torch` missing from the environment, this cannot be validated in the current state. After fixing the torch requirement, verify end-to-end:

```bash
python3 -c "import runux; print('OK')"
```

---

### 3.2 🟡 GPU-Dependent Tests Lack Hardware Skip Guards

**Files**: `tests/test_runux_kernels.py`, `tests/test_soak_and_zenodo.py`

Multiple tests assert a Tesla T4 is present:
```python
assert "Tesla T4" in summary["device"]
```

Without `pytest.mark.skipif`, these tests fail on every CPU-only CI runner. The entire pytest collection currently crashes on import due to missing `torch`.

**Fix**: Add uniform skip guards:
```python
import pytest, torch

requires_t4 = pytest.mark.skipif(
    not torch.cuda.is_available() or
    "T4" not in torch.cuda.get_device_name(0),
    reason="Requires NVIDIA Tesla T4 GPU"
)

@requires_t4
def test_t4_sustained_soak_short_run():
    ...
```

Add to `pyproject.toml`:
```toml
[tool.pytest.ini_options]
markers = [
    "gpu: tests requiring any CUDA GPU",
    "t4: tests requiring NVIDIA Tesla T4 specifically",
]
```

---

### 3.3 🟡 FlashAttention Tiled Latency Is 34× Slower at S=1024

**File**: `gpu_t4_deep_validation_results.json`

```json
"standard_latency_ms": 7.23,
"tiled_latency_ms":    249.78,
"vram_savings_pct":    19.9
```

The 19.9% VRAM savings are correctly measured. However, the tiled implementation runs **34.5× slower** than standard attention at S=1024. This is physically expected — FlashAttention's IO-aware tiling only wins when the attention matrix is too large for SRAM (typically S≥2048–4096 on T4). But:

1. The certification report does not disclose this latency regression
2. Partners evaluating T4 production deployability will flag 249 ms per attention step as a serious concern
3. The crossover sequence length must be measured and disclosed

**Action**: Add a sweep benchmark from S=512 to S=16384. Document and certify only for the operating range where tiled FA is actually faster.

---

### 3.4 🟡 Zenodo ORCID Is a Placeholder

**File**: `zenodo.json`

```json
"orcid": "0009-0000-0000-0000"
```

This is a non-existent ORCID. Since DOI `10.5281/zenodo.22697937` is cited in the paper, patent, and Hugging Face datasets, the researcher profile must be registered at https://orcid.org before the final public release.

---

### 3.5 🟡 Onboarding Symlink Is Unreliable

```
lrwxrwxrwx  quickstart_resume.sh -> scripts/restart_session.sh
```

This relative symlink resolves correctly only from the repository root. If accessed from any other directory, it silently breaks. Combined with the hardcoded path bug in §1.2, the documented onboarding entry point is doubly broken.

**Fix**: Either convert to a wrapper script with `cd "$(git rev-parse --show-toplevel)"`, or make the symlink absolute using `ln -sf "$(readlink -f scripts/restart_session.sh)" quickstart_resume.sh`.

---

## 4. Low-Priority Issues (P3 — Technical Debt)

### 4.1 🔵 Cargo Workspace Profile Warnings

```
warning: profiles for the non root package will be ignored, specify profiles
at the workspace root:
  package: examples/edge_inference_demo/Cargo.toml
  package: examples/macos_m2_inference_demo/Cargo.toml
```

`[profile.release]` sections in sub-package `Cargo.toml` files are silently ignored by Cargo workspaces. All profile configuration belongs in the workspace root `Cargo.toml`.

**Fix**: Remove `[profile.*]` sections from both example `Cargo.toml` files.

---

### 4.2 🔵 `Cargo.lock` Gitignored — Reproducibility Risk

`Cargo.lock` is not committed. For a Rust **binary** crate (`runux-report`), this means:

- Dependency versions may silently differ across machines
- CI builds cannot guarantee reproducibility
- Partner evaluation environments may resolve different crate versions

**Fix**: Commit `Cargo.lock`. Per official Cargo guidance, lock files should always be committed for binary crates.

---

### 4.3 🔵 `autoresearch_agent_v4.py` — Dead Code at Repository Root

**File**: `autoresearch_agent_v4.py` (33,447 bytes)

Not imported by any module, has no corresponding test, and is not referenced in any documentation. The structured `autoresearch_agent/` package already exists.

**Recommendation**: Move into `autoresearch_agent/` or an `archive/` directory.

---

### 4.4 🔵 IP Policy Alignment for `crates/tpu_pjrt/src/ffi.rs`

`MEMORY.md` states proprietary Rust source is withheld from public repositories. `crates/tpu_pjrt/src/ffi.rs` (PJRT C ABI declarations + simulation fallback, 178 lines) is committed to public `main`. While it contains no algorithmic trade secrets, it is technically a `.rs` file.

**Recommendation**: Consult legal counsel on whether FFI declaration files (as opposed to kernel implementations) require the same protection level.

---

## 5. Architecture Assessment

### 5.1 Dual-Architecture Design ✅ Sound

The split between the commercial `no_std` Rust engine (SpacemiT K1/K3 edge hardware) and the Python GPU acceleration layer (`runux/` on T4/A100) is architecturally clean and commercially strategic. The `framework_bridge` C FFI (`runux_flash_attention`, `runux_matmul`, `runux_rms_norm`, `runux_silu`) is the correct integration seam.

### 5.2 Phase 1 End-to-End Integration Test ✅ Excellent

`crates/framework_bridge/tests/phase1_end_to_end.rs` validates 7 components in sequence: Arena allocation → FlashAttention kernel + FFI → Paged KV cache → TurboQuant compression → GpuContext → RMSNorm/SiLU → MatMul projection. This is strong cross-crate integration coverage.

### 5.3 TPU PJRT Simulation Fallback ✅ Correct

`crates/tpu_pjrt/src/ffi.rs` correctly models the PJRT C API v0.48+ with a simulation fallback returning `DeviceUnavailable` when no TPU is present. The error path is unit-tested. This is the right pattern for hardware-independent development.

### 5.4 PagedKVCache Memory Model ✅ Correct

`runux/paged_cache.py` correctly implements: O(1) block allocation via `list.pop()`, 0% external fragmentation by design, correct fragmentation reporting, and `RuntimeError` on OOM. The `memory_stats()` method distinguishing internal vs external fragmentation is accurate.

### 5.5 Simulation vs. Physical Measurement Boundary ⚠️ Needs Clear Labels

The Rust `sim_bench` / `sim_inference` crates produce **mathematical simulation** outputs (the 197 TFLOPS, 2.32× speedup figures). The Python `run_gpu_t4_deep_validation.py` performs **actual hardware measurements**. Both appear in external materials. Ensure provenance labels (simulated / empirical) are explicit in every external-facing document.

---

## 6. Security Assessment

| Area | Status | Notes |
|:---|:---:|:---|
| Secret files excluded from git | ✅ | Comprehensive `.gitignore`: `.env`, `*.key`, `*.token`, `secrets/` |
| No credentials in committed code | ✅ | Full scan: zero API keys, passwords, private keys found |
| Commercial license headers (Rust) | ✅ | `SPDX-License-Identifier: LicenseRef-RunuX-Commercial` consistent |
| Copyright banners (Python) | ✅ | Consistent across all 8 `runux/` modules |
| Zenodo bundle sanitation test | ✅ | Automated gate blocks `.rs`, `.rlib`, `.so` from public bundles |
| `crates/tpu_pjrt/src/ffi.rs` in public repo | ⚠️ | IP policy alignment question (§4.4) |

---

## 7. Code Quality Metrics

| Metric | Value | Trend vs Audit 1 |
|:---|:---|:---:|
| Total Rust LOC (crates + src + examples) | ~20,295 | ↑ +17% |
| Total Python LOC (runux/ + tests/ + scripts) | ~36,522 | ↑ New layer |
| Rust workspace crates | 24 | = |
| Python public modules (`runux/`) | 8 | ↑ New since Audit 1 |
| Unit tests (Rust, estimated assertions) | ~181 | = |
| Unit tests (Python, 4 test files) | ~727 loc | ↑ New |
| Integration tests (Rust) | 1 (phase1_end_to_end) | ↑ New |
| Git tags | 17 | ↑ +10 since Audit 1 |
| Open-science DOI | `10.5281/zenodo.22697937` | ↑ Published |
| Physical hardware benchmarks passed | 8/8 (T4 GPU) | ↑ New |
| Cargo build status | ❌ Broken (lockfile collision) | ↓ Regression |

---

## 8. New Files Since Audit 1 — Quick Assessment

| File | Quality | Key Note |
|:---|:---:|:---|
| `runux/gqa_kernel.py` | ✅ | Clean GQA + RoPE implementation |
| `runux/paged_cache.py` | ✅ | Correct paged allocator, O(1) alloc |
| `runux/transformer_engine.py` | ⚠️ | Import chain needs verification after torch install |
| `crates/tpu_pjrt/src/ffi.rs` | ✅ | Clean PJRT C FFI with simulation fallback |
| `crates/framework_bridge/tests/phase1_end_to_end.rs` | ✅✅ | Excellent 7-component integration coverage |
| `workflow_soak_benchmark.py` | ✅ | Comprehensive soak protocol with test coverage |
| `monitor_t4_1hour_benchmark.py` | ✅ | Real telemetry loop, correct CSV/JSON flushing |
| `scripts/restart_session.sh` | ❌ | Hardcoded developer path (§1.2) |
| `quickstart_resume.sh` (symlink) | ⚠️ | Points to buggy script (§3.5) |
| `zenodo.json` | ⚠️ | Placeholder ORCID (§3.4) |
| `MEMORY.md` | ✅ | Excellent system memory document for agent continuity |
| `tests/test_runux_kernels.py` | ⚠️ | No GPU skip guards |
| `tests/test_soak_and_zenodo.py` | ⚠️ | No GPU skip guards, T4-specific assertions |
| `tests/test_partner_harnesses.py` | ✅ | Pure simulation tests, GPU not required |

---

## 9. Prioritized Action Plan

### P0 — Immediate (Before Next Commit)

- [ ] **`crates/turbo_quant/Cargo.toml`**: revert `path` to `"../ai_runtime"` and commit
- [ ] **`scripts/restart_session.sh`**: replace hardcoded `VENV_PYTHON` with dynamic path discovery

### P1 — This Week (Before Partner Demo)

- [ ] Add `torch>=2.1.0` to `requirements.txt`; create `requirements-gpu.txt` for CUDA profile
- [ ] Investigate speculative decoding regression — reduce K or improve acceptance rate; relabel in all materials
- [ ] Ensure all external materials clearly distinguish simulated vs. physical benchmark provenance

### P2 — This Sprint

- [ ] Add `@pytest.mark.skipif(not torch.cuda.is_available())` to GPU-dependent tests
- [ ] Verify `runux` package imports cleanly: `python3 -c "import runux"` after installing torch
- [ ] Register a real ORCID and update `zenodo.json`
- [ ] Benchmark FlashAttention sweep (S=512→16384) to identify crossover point; update certification
- [ ] Fix `quickstart_resume.sh` to be robust from any working directory

### P3 — Next Sprint

- [ ] Remove `[profile.*]` from `examples/*/Cargo.toml` to eliminate Cargo warnings
- [ ] Commit `Cargo.lock` for the binary crate (reproducibility guarantee)
- [ ] Move `autoresearch_agent_v4.py` into `autoresearch_agent/` directory
- [ ] Legal review: determine if `crates/tpu_pjrt/src/ffi.rs` requires IP protection

---

## 10. Verdict

The RunuX codebase has matured significantly since Audit 1. The dual Rust+Python architecture is strategically sound, the physical Tesla T4 benchmarking methodology is rigorous (1-hour soak, bit-exact INT64 determinism, paged memory correctness), and the IP protection model (Zenodo sanitation gate, license headers) is well-implemented.

**The single most critical issue is the broken build caused by an absolute local path in `crates/turbo_quant/Cargo.toml`** — a one-line fix required immediately.

**The second most important business issue** is the speculative decoding 0.64× regression with a "CERTIFIED" label — this must be corrected or relabeled before any partner demo to maintain scientific credibility.

All other findings are tractable technical debt or documentation improvements that will strengthen the project's reproducibility and partner trust.

---

*Audit conducted on HEAD `67c13a5` after `git pull --rebase origin/main` on 2026-09-11.*  
*(c) 2026 Antigravity AI Audit — For Internal Use Only*
