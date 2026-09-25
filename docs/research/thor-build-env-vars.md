# vLLM on Thor — Build & Runtime Environment Variables (Audit)

> **Status:** Complete audit summary · **Date:** 2026-08-24 · **Scope:** Build-time environment variables for building vLLM (v0.28.0rc2) from source for Jetson AGX Thor (SM110a) on the aarch64 CUDA-13 base image (`vllm/vllm-openai:v0.27.1`); audited against every working Thor build recipe then available. This knowledge base informs the shipped overlay in `docker/vllm-thor/` (Dockerfile + patches); the current stack is documented in [`docs/thor-stack/`](../thor-stack/).

## TL;DR

- The audited Dockerfile (`Dockerfile.dflash2` — vLLM v0.28.0rc2 from source for Thor) is **partially incomplete** for a successful from-source build on Thor: it sets `TORCH_CUDA_ARCH_LIST=11.0a`, `CUDA_HOME=/usr/local/cuda`, and `MAX_JOBS=4`, but omits `TRITON_PTXAS_PATH`, `VLLM_TARGET_DEVICE`, `PATH`, and `LD_LIBRARY_PATH` adjustments that are universally present in every working Thor build recipe examined.
- **`TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas` is mandatory** — without it, Triton's bundled ptxas rejects `sm_110a` and the build halts with `ptxas fatal : Value 'sm_110a' is not defined for option 'gpu-name'`.
- **`VLLM_TARGET_DEVICE=cuda` is required** for explicit CUDA variant selection at compile time.
- `PATH`/`LD_LIBRARY_PATH` including `/usr/local/cuda/bin` / `/usr/local/cuda/lib64` are strongly recommended; `FLASHINFER_CUDA_ARCH_LIST=11.0a` is conditionally required (flash-attn's flashinfer JIT step).
- Verdict: without the missing variables the build **will almost certainly fail** at the flash-attn compilation step (or silently compile Triton kernels for the wrong architecture).

## Missing ENV lines — recommended additions

### 1. `TRITON_PTXAS_PATH` — REQUIRED

```dockerfile
ENV TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
```

**Verdict:** Mandatory. Without this, Triton's bundled ptxas refuses to recognise `sm_110a` and the build halts with:

```
ptxas fatal : Value 'sm_110a' is not defined for option 'gpu-name'
```

**Evidence:**

| Source | Quote | Confidence |
|--------|-------|------------|
| johnny_nv (NVIDIA engineer), "Run VLLM in Thor" forum | `export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas` | **High** — NVIDIA staff, verified working recipe |
| vLLM issue #37060 (sm110 illegal instruction) | `export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas` | **High** — reproduction steps |
| HackMD SGLang multi-node Thor/Spark cluster | `export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas` | **High** — operational cluster |
| saskia.hold, "vLLM service on Thor JP7.2" | `Environment=TRITON_PTXAS_BLACKWELL_PATH=/usr/local/cuda/bin/ptxas` | **Medium** — uses `BLACKWELL_PATH` suffix variant |
| Sage Attention / DGX Spark forum | `export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas` | **Medium** — secondary source |

**Path note:** All working Thor builds use `/usr/local/cuda/bin/ptxas`. The base image `vllm/vllm-openai:v0.27.1` is built from `nvidia/cuda:13.0.3-base-ubuntu24.04`, which creates the standard `/usr/local/cuda` → versioned-directory symlink.

### 2. `VLLM_TARGET_DEVICE` — REQUIRED

```dockerfile
ENV VLLM_TARGET_DEVICE=cuda
```

**Verdict:** Mandatory for explicit CUDA variant selection during `pip install -e .` or `setup.py bdist_wheel`. While the default is `cuda`, the vLLM build system auto-detects based on installed PyTorch; explicitly setting it avoids mis-detection edge cases (especially when mixing pre-built torch with from-source vLLM).

**Evidence:**

| Source | Quote | Confidence |
|--------|-------|------------|
| vLLM official Dockerfile (line ~771) | `ENV VLLM_TARGET_DEVICE=${vllm_target_device}` where `ARG vllm_target_device="cuda"` | **High** — official source |
| RobotFlow-Labs'anima-vllm-thor Dockerfile | `ENV VLLM_TARGET_DEVICE=cuda` | **High** — working Thor recipe |
| DGX Spark build guide (troy.e.davis) | `ENV VLLM_TARGET_DEVICE=cuda` | **High** — verified SM121 build |
| vLLM DeepWiki build variants page | Confirms `VLLM_TARGET_DEVICE` determines build variant at compile time | **High** — architectural documentation |

### 3. `PATH` — RECOMMENDED

```dockerfile
ENV PATH=/usr/local/cuda/bin:${PATH}
```

**Verdict:** Strongly recommended. Ensures `nvcc` and `ptxas` are discoverable by pip/setuptools during the flash-attn and vLLM builds. Several Thor builds rely on this.

**Evidence:**

| Source | Quote | Confidence |
|--------|-------|------------|
| johnny_nv (NVIDIA engineer) | `export PATH=/usr/local/cuda/bin:$PATH` | **High** |
| vLLM issue #37060 | `export PATH="${CUDA_HOME}/bin:$PATH"` | **High** |
| saskia.hold Thor systemd unit | `Environment=PATH="${CUDA_HOME}/bin:/usr/bin:/usr/sbin:/usr/local/bin:$PATH"` | **High** |
| vLLM official GPU docs | Recommends `export PATH="${CUDA_HOME}/bin:$PATH"` for from-source builds | **High** |
| DGX Spark build guide | `ENV CMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc` (alternative approach) | **Medium** |

### 4. `LD_LIBRARY_PATH` — RECOMMENDED

```dockerfile
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
```

**Verdict:** Strongly recommended for build-time linking. The vLLM official Dockerfile explicitly sets this at multiple stages. Without it, the linker may not find `libcublas.so`, `libcudart.so`, etc. during CUDA extension compilation.

**Evidence:**

| Source | Quote | Confidence |
|--------|-------|------------|
| vLLM official Dockerfile (vllm-base stage) | `ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH` | **High** |
| vLLM official Dockerfile (test stage) | Same directive | **High** |
| PSA: FP4/NVFP4 support for DGX Spark (forum) | `export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/lib:${LD_LIBRARY_PATH}` | **Medium** |

### 5. `FLASHINFER_CUDA_ARCH_LIST` — CONDITIONAL

```dockerfile
ENV FLASHINFER_CUDA_ARCH_LIST=11.0a
```

**Verdict:** Conditionally required. The Dockerfile installs the `flashinfer_cubin` pre-compiled wheel (which is architecture-independent), but if flash-attn's `setup.py` internally invokes FlashInfer JIT compilation, the arch list must be set. RobotFlow-Labs sets this explicitly.

**Evidence:**

| Source | Quote | Confidence |
|--------|-------|------------|
| RobotFlow-Labs'anima-vllm-thor Dockerfile | `ENV FLASHINFER_CUDA_ARCH_LIST=11.0a` | **High** — working Thor recipe |
| vLLM issue #37060 (flashinfer build step) | User sets `FLASHINFER_CUDA_ARCH_LIST="11.0a"` for flashinfer source build | **High** |

## Existing lines — assessment

| Current line | Status | Notes |
|-------------|--------|-------|
| `TORCH_CUDA_ARCH_LIST=11.0a` | **Correct** | Universally agreed upon for Thor. Matches sm_110a compute capability. |
| `CUDA_HOME=/usr/local/cuda` | **Likely correct** | Consistent with the vLLM official Dockerfile and most Thor recipes. Some Thor users report `CUDA_HOME=/usr/local/cuda-13`. The `/usr/local/cuda` symlink in the v0.27.1 base image should resolve correctly, but if the build encounters `nvcc` not found, switch to `/usr/local/cuda-13`. |
| `MAX_JOBS=4` | **Acceptable** | Conservative for Thor's limited memory. RobotFlow-Labs uses `MAX_JOBS=12`. Consider increasing to 8 if memory permits (≥64 GB free). |

**Questionable line:** `CUDA_HOME=/usr/local/cuda` — canonical in the vLLM Dockerfile, but Thor-specific forum posts occasionally use `/usr/local/cuda-13`. The base image `vllm/vllm-openai:v0.27.1` is derived from `nvidia/cuda:13.0.3-base-ubuntu24.04`, which maintains the `/usr/local/cuda` symlink convention. **Recommendation:** keep as-is, with a fallback plan to change to `/usr/local/cuda-13` if `nvcc` is not found during build.

## Verdict — buildability assessment

### Current audited Dockerfile: LIKELY WILL FAIL

Without the missing environment variables, the build will almost certainly fail at the flash-attn compilation step with:

```
ptxas fatal : Value 'sm_110a' is not defined for option 'gpu-name'
```

This is the **same error** reported by multiple Thor builders who forgot `TRITON_PTXAS_PATH`. Even if flash-attn succeeds accidentally, the vLLM build may silently compile Triton kernels for the wrong architecture.

### Minimum ENV addition required

At minimum, add these lines to the existing ENV block:

```dockerfile
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    CUDA_HOME=/usr/local/cuda \
    TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas \
    VLLM_TARGET_DEVICE=cuda \
    MAX_JOBS=4 \
    PATH=/usr/local/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
```

Optionally add `FLASHINFER_CUDA_ARCH_LIST=11.0a` for extra safety.

### Complete recommended ENV block

Based on cross-referencing all working Thor build recipes, the full recommended ENV block is:

```dockerfile
ARG TORCH_CUDA_ARCH_LIST=11.0a
ARG FLASH_ATTN_VERSION=2.8.3
ARG VLLM_RC_TAG=v0.28.0rc2

ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    CUDA_HOME=/usr/local/cuda \
    TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas \
    VLLM_TARGET_DEVICE=cuda \
    MAX_JOBS=4 \
    PATH=/usr/local/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH} \
    FLASHINFER_CUDA_ARCH_LIST=11.0a
```

## Additional recommendations beyond ENV variables

1. **flash-attn build invocation.** The audited Dockerfile runs `python3 setup.py bdist_wheel`, which relies on flash-attn's default `FLASH_ATTN_CUDA_ARCHS` (defaults to `"80;90;100;110;120"`). The `110` in that list maps to `sm_110` via `setup.py`'s `add_cuda_gencodes()` (which handles the Thor rename: CUDA 13.0+ → `sm_110f`). **This should work**, but explicitly setting `FLASH_ATTN_CUDA_ARCHS=110` is safer:
   ```dockerfile
   RUN FLASH_ATTN_CUDA_ARCHS=110 python3 setup.py bdist_wheel
   ```
2. **flash-attn build with nvcc threads.** Consider adding `ENV NVCC_THREADS=2` to control parallelism during flash-attn compilation (the most memory-intensive step).
3. **Post-flash-attn cleanup.** The flash-attn `git clone` leaves source in the image; the audited Dockerfile cleans it up (`cd .. && rm -rf flash-attention`). Ensure the same for the vLLM clone.
4. **Runtime env vars.** The audited Dockerfile's footer already documents runtime env vars (`VLLM_USE_FLASHINFER_MOE_FP4=0`, etc.). No changes needed there.

## Source breakdown — evidence mapping

### Claim: `TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas` is required for Thor builds

- **Primary sources:**
  - johnny_nv (NVIDIA), "Run VLLM in Thor from VLLM Repository" forum thread (Oct 2025) — <https://forums.developer.nvidia.com/t/run-vllm-in-thor-from-vllm-repository/348804>
  - vLLM issue #37060 (Mar 2026) — <https://github.com/vllm-project/vllm/issues/37060>
  - HackMD SGLang multi-node Thor/Spark cluster — <https://hackmd.io/@johnnynunez/S19A_Keqbe>
- **Secondary sources:**
  - sage-attention DGX Spark forum — <https://forums.developer.nvidia.com/t/sage-attention-with-comfyui/350423>
  - Run VLLM in Spark forum page 4 — <https://forums.developer.nvidia.com/t/run-vllm-in-spark/348862?page=4>
- **Confidence:** High — unanimous across all Thor-specific sources

### Claim: `VLLM_TARGET_DEVICE=cuda` is required for CUDA variant builds

- **Primary sources:**
  - vLLM official Dockerfile (main branch) — <https://github.com/vllm-project/vllm/blob/main/docker/Dockerfile>
  - RobotFlow-Labs'anima-vllm-thor Dockerfile — <https://github.com/RobotFlow-Labs/anima-vllm-thor/blob/main/Dockerfile>
  - DGX Spark SM121 build guide — <https://forums.developer.nvidia.com/t/dgx-spark-13-49-tok-s-with-qwen3-5-35b-native-sm121-kernel-build-guide/365083>
- **Confidence:** High — officially mandated in vLLM's own Dockerfile

### Claim: `PATH` and `LD_LIBRARY_PATH` must include CUDA directories

- **Primary sources:**
  - vLLM official GPU installation docs — <https://docs.vllm.ai/en/stable/getting_started/installation/gpu/>
  - vLLM official Dockerfile — <https://github.com/vllm-project/vllm/blob/main/docker/Dockerfile>
  - Multiple Thor forum threads (johnny_nv, saskia.hold, issue #37060)
- **Confidence:** High — consistent across official and community sources

### Claim: flash-attn supports `sm_110a` via `FLASH_ATTN_CUDA_ARCHS=110`

- **Primary source:**
  - flash-attn `setup.py` (main branch) — `add_cuda_gencodes()`, mapping `"110"` → `compute_110f`/`code=sm_110` for CUDA 13.0+ — <https://github.com/Dao-AILab/flash-attention/blob/main/setup.py>
- **Confidence:** High — verified in source code

### Claim: Base image `vllm/vllm-openai:v0.27.1` provides the CUDA toolkit at `/usr/local/cuda`

- **Primary source:**
  - vLLM official Dockerfile builds from `nvidia/cuda:13.0.3-base-ubuntu24.04` — <https://github.com/vllm-project/vllm/blob/main/docker/Dockerfile>; NVIDIA CUDA base images maintain the `/usr/local/cuda` symlink convention
- **Confidence:** Medium-High — inferred from base image lineage; not directly inspected

## Methodology

- **Searches performed (sequential, refined iteratively):**
  1. `vLLM build from source Jetson ARM64 environment variables CUDA`
  2. `TRITON_PTXAS_PATH environment variable Triton build arm64 CUDA`
  3. `site:github.com RobotFlow-Labs anima-vllm-thor Dockerfile`
  4. `patrickbdevaney qwen nvfp4 thor docker vllm blackwell`
  5. `vLLM build from source required environment variables TORCH_CUDA_ARCH_LIST VLLM_TARGET_DEVICE`
  6. `flash-attn setup.py sm_110a GPU_ARCHS build environment variable jetson thor`
  7. `"vllm" "jetson thor" "CUDA_HOME" OR "TRITON_PTXAS_PATH" OR "LD_LIBRARY_PATH" docker build environment`
  8. `"vllm/vllm-openai:v0.27.1" docker image /usr/local/cuda path jetson thor`
- **Pages fully read:** [RobotFlow-Labs'anima-vllm-thor Dockerfile](https://github.com/RobotFlow-Labs/anima-vllm-thor/blob/main/Dockerfile); [RobotFlow-Labs'anima-vllm-thor README](https://github.com/RobotFlow-Labs/anima-vllm-thor/blob/main/README.md); [patrickbdevaney's DOCKERFILE](https://github.com/patrickbdevaney/qwen-3.5-122b-a10b-jetson-thor/blob/main/DOCKERFILE); [patrickbdevaney's README](https://github.com/patrickbdevaney/qwen-3.5-122b-a10b-jetson-thor/blob/main/README.MD); [vLLM official GPU installation docs](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/); [vLLM build variants (DeepWiki)](https://deepwiki.com/vllm-project/vllm/11.3-build-variants-and-configuration); [DGX Spark SM121 build guide forum](https://forums.developer.nvidia.com/t/dgx-spark-13-49-tok-s-with-qwen3-5-35b-native-sm121-kernel-build-guide/365083); [flash-attn setup.py](https://github.com/Dao-AILab/flash-attention/blob/main/setup.py); [vLLM official Dockerfile](https://github.com/vllm-project/vllm/blob/main/docker/Dockerfile); [Run VLLM in Thor (johnny_nv)](https://forums.developer.nvidia.com/t/run-vllm-in-thor-from-vllm-repository/348804); [vLLM issue #37060](https://github.com/vllm-project/vllm/issues/37060); [Running vLLM as a service on Thor JP7.2](https://forums.developer.nvidia.com/t/running-vllm-as-a-service-on-thor-jp7-2/373029); [HackMD SGLang multi-node Thor/Spark cluster](https://hackmd.io/@johnnynunez/S19A_Keqbe); [Run vllm fail forum (sm_110a ptxas error)](https://forums.developer.nvidia.com/t/run-vllm-fail/344574).
- **GitHub tools used:** `github_get_file_contents` ×6 (Dockerfiles, READMEs from the RobotFlow-Labs and patrickbdevaney repos); `github_list_branches` ×1 (attempted `patrickschae/nvfp4-thor` — repo not found).
- **Pages skimmed but not fully read:** NVIDIA developer forum "How to install pytorch in thor" (content collapsed to navigation chrome; insufficient detail); vLLM issue #26791 (sm110 compatibility — referenced but not critical to the env-var audit).

## Gaps and caveats

1. **Base image CUDA path not directly verified.** The assumption that `/usr/local/cuda` resolves correctly in `vllm/vllm-openai:v0.27.1` is based on the image's base-image lineage (`nvidia/cuda:13.0.3-base-ubuntu24.04`). Direct inspection (`docker run vllm/vllm-openai:v0.27.1 ls -la /usr/local/cuda`) would confirm.
2. **`TRITON_PTXAS_BLACKWELL_PATH` variant.** saskia.hold's systemd unit uses `TRITON_PTXAS_BLACKWELL_PATH` instead of `TRITON_PTXAS_PATH`. This may indicate a newer Triton API distinguishing generic and Blackwell-specific ptxas paths. If `TRITON_PTXAS_PATH` alone fails, try `TRITON_PTXAS_BLACKWELL_PATH` as a backup.
3. **CUDA 13.0 vs 13.3+.** The v0.27.1 image ships CUDA 13.0.3. If a later Thor system has CUDA 13.3+, the ptxas path may differ slightly; the `/usr/local/cuda` symlink abstraction shields against this.
4. **flash-attn 2.8.3 vs 2.8.4.** The audited Dockerfile pins 2.8.3. Later versions (2.8.4+) may have different default `FLASH_ATTN_CUDA_ARCHS`. The `setup.py` read confirms the 110 → sm_110 mapping exists in the current main branch.
5. **Cross-build vs on-board build.** The audited Dockerfile is intended to build on Thor (aarch64). Cross-compilation from x86_64 is explicitly unsupported (per patrickbdevaney's README). The build instructions do not declare `--platform linux/arm64`, which may cause issues if built on an x86 host.
6. **Runtime vs build-time distinction.** Some env vars (like `LD_PRELOAD`) are runtime-only and intentionally not set at build time (as noted in the audited Dockerfile's own comments). This audit covers build-time variables only.
