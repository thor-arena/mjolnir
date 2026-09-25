# NVFP4/FP8 on sm_110/sm_12x — Field Notes

> **Status:** Consolidated field notes (compiled from 8 research captures: 7 NVIDIA developer-forum threads + 1 community playbook; captured 2026-09-07, content spans Nov 2025 – Jun 2026) · **Date:** 2026-09-07 · **Scope:** FP8/NVFP4 behavior of vLLM on Jetson AGX Thor (sm_110a) and DGX Spark / GB10 (sm_121), plus cross-architecture CUTLASS verification. Each section: TL;DR + evidence + links. These notes underpin the quantization decisions in the shipped overlay (`docker/vllm-thor/`; see also [`docs/thor-stack/`](../thor-stack/)).

## TL;DR

- FP8 on Thor (sm_110) is **fragile in stock images**: FP8 checkpoints have produced garbled output (official 25.09 NGC image) and hard crashes (`This kernel only supports sm100f` → `CUBLAS_STATUS_INTERNAL_ERROR`) — several kernel families are compiled for sm100/sm100f only and do not run on sm_110.
- CUTLASS's own sparse-INT8 profiler kernels **do not run** when compiled for sm_110; the sm_80 fallback runs at ~6.9 TFLOP/s — claimed Thor TOPS cannot be verified with that kernel on that platform.
- On sm_121 (DGX Spark) there is **no native FP4 compute** (no `tcgen05`); the CUTLASS FP4 GEMM crashes, and the working path is **Marlin** (W4A16, dequantizes FP4→BF16 on the fly): 50 tok/s vs 42.6, 32 GB vs 39 GB, 16 % faster, 7 GB less.
- MoE on Thor in stock vLLM 26.02 runs **far below reference** (Qwen3-30B-A3B: ~34 vs ~61 tok/s at C1) — no Thor fused-MoE config exists, and the unquantized MoE falls back to Triton.
- A full from-source Thor build recipe for NVFP4 (Nemotron-3-Super-120B-A12B) is documented, including the sm_110a env-var set and a ~10.5 tok/s single-stream measurement.
- Community consensus on DGX Spark: pin NGC `26.04-py3` (vLLM 0.19.0) or a known nightly; on 0.23+ use `--moe-backend marlin` (the old `VLLM_USE_FLASHINFER_MOE_FP4` env vars are deprecated); `!!!!!`/empty output is the tell for a broken FP4 kernel path.

---

## Garbled FP8 output on Thor (official 25.09 vLLM image)

**TL;DR:** FP8 models hosted with the official `nvcr.io/nvidia/vllm:25.09-py3` image on Jetson Linux 38.2 produce **garbled output**, while their BF16 counterparts work. Two Qwen3 FP8 checkpoints were affected.

**Evidence** (forum thread, opened Nov 13, 2025 by changtimwu; closed Dec 16, 2025):

- Affected FP8 models (bench fine with `vllm bench`, garbled at serve time):
  - [Qwen/Qwen3-30B-A3B-Instruct-2507-FP8](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507-FP8)
  - [Qwen/Qwen3-0.6B-FP8](https://huggingface.co/Qwen/Qwen3-0.6B-FP8)
- Unaffected BF16 counterparts: [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B), [Qwen/Qwen3-30B-A3B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507)
- Environment: Jetson Linux 38.2, `nvcr.io/nvidia/vllm:25.09-py3`, OpenWebUI frontend.
- Context: NVIDIA had just released the official vLLM image and [Jetson benchmark results](https://developer.nvidia.com/embedded/jetson-benchmarks) at the time.
- The thread links to the sibling sm100f-kernel failure: "vLLM FP8 models unusable on AGX Thor (SM 11.0): kernels compiled for sm100f only — This kernel only supports sm100f. → CUBLAS_STATUS_INTERNAL_ERROR".

**Links:** <https://forums.developer.nvidia.com/t/fp8-series-models-hosting-with-the-official-2509-vllm-consistently-produces-garbled-output/351272> · related: <https://forums.developer.nvidia.com/t/vllm-fp8-models-unusable-on-agx-thor-sm-11-0-kernels-compiled-for-sm100f-only-this-kernel-only-supports-sm100f-cublas-status-internal-error/375912> · <https://forums.developer.nvidia.com/t/announcing-new-vllm-container-3-5x-increase-in-gen-ai-performance-in-just-5-weeks-of-jetson-agx-thor-launch/346634> · <https://developer.nvidia.com/embedded/jetson-benchmarks>

---

## CUTLASS SM110 verification — claimed TOPS not reproducible

**TL;DR:** The sparse-INT8 CUTLASS kernel that verified claimed TOPS on Orin **does not run at all** when compiled for Thor (sm_110); forcing sm_80 makes it run at ~6.9 TFLOP/s — orders of magnitude below advertised Thor sparse-INT8 TOPS. The thread stayed open through Dec 2025 with CUTLASS maintainers (AastaLLL) responding.

**Evidence** (forum thread, opened Nov 20, 2025 by john_c; 23 posts, active through Dec 2025):

- Kernel under test: `cutlass_tensorop_i88128xorgemm_b1_256x128_512x2_tn_align128` (the sparse-INT8 kernel that gave best results on Orin per the [Orin INT8 verification thread](https://forums.developer.nvidia.com/t/discrepancy-between-claimed-and-actual-sparse-int8-performance-of-tensor-cores-on-jetson-agx-orin/303133)).
- `cmake .. -DCUTLASS_NVCC_ARCHS=110 -DCUTLASS_LIBRARY_KERNELS=cutlass_tensorop_i88128xorgemm_b1_256x128_512x2_tn_align128` → cutlass_profiler builds, but the kernel **is not executed at all** ("No results").
- `cmake .. -DCUTLASS_NVCC_ARCHS=80 …` (Ampere fallback) → compiles and runs. Profiler command: 16384×16384×16384 sparse-INT8 TN GEMM, `--split_k_mode=serial --op_class=tensorop --accum=s32 --cta_m=256 --cta_n=128 --cta_k=512 --cluster_m=1 --cluster_n=1 --cluster_k=1 --stages=2 --warps_m=4 --warps_n=2 --warps_k=1 --inst_m=8 --inst_n=8 --inst_k=128 --min_cc=75 --max_cc=1024`.
- Result: `Status: Success`, `Runtime: 1272.44 ms`, `Math: 6913.18 GFLOP/s` (~6.9 TFLOP/s), FLOPs 8796629893120, Bytes 1140850688 — "orders of magnitude below the marketed sparse INT8 TOPS of Jetson Thor".
- Original questions: (1) correct compute capability / NVCC arch flag for Thor in CUTLASS today — is SM110 supported? (2) which kernel reaches peak sparse-INT8 TOPS on Thor? (3) official/recommended way to measure and verify claimed sparse-INT8 TOPS?

**Links:** <https://forums.developer.nvidia.com/t/verifying-claimed-tops-performance-on-jetson-thor-cutlass-kernel-for-sm110-does-not-run-sm80-gives-very-low-performance-6-9-tflop-s/352063> · related: <https://forums.developer.nvidia.com/t/how-to-benchmark-on-thor-to-get-the-real-fp4-fp8-performance-tfops/353640/3> · <https://forums.developer.nvidia.com/t/1pflop-how/365583/3> · <https://forums.developer.nvidia.com/t/question-on-reproducing-dgx-spark-gb10-fp4-1-pflops-performance-using-cutlass-profiler/357249/5>

---

## Marlin fix — NVFP4 on sm_121 (DGX Spark)

**TL;DR:** NVFP4 on DGX Spark (sm_121) silently runs on **broken CUTLASS kernels** — vLLM auto-selects `FLASHINFER_CUTLASS` because sm_121 has capability ≥ 100, but SM121 lacks the `tcgen05` tensor-core instructions (the CUTLASS FP4 kernels emit `cvt .e2m1x2` PTX sm_121 can't run). The autotuner skips the broken tactics and falls back to a slower, higher-memory path. Setting three env vars forces **Marlin** and takes NVFP4 from broken/slow to **50 tok/s**: 16 % faster and 7 GB less memory than the default FlashInfer path.

**Evidence** (forum thread, Mar 30, 2026 by sggin1; 17 posts, closed Apr 26, 2026; tested Mar 26, 2026 — DGX Spark GB10, CUDA 13.2, Driver 580.142, vLLM 0.18.1rc1 eugr build):

- The tell in vLLM logs when affected:
  ```
  [Autotuner]: Skipping tactic … due to failure while profiling:
  [TensorRT-LLM][ERROR] Failed to initialize cutlass TMA WS grouped gemm
  ```
- Root cause — backend auto-selection (vLLM source, `nvfp4_utils.py` lines 59-64):
  ```python
  if current_platform.has_device_capability(100) and has_flashinfer():
      backend = NvFp4LinearBackend.FLASHINFER_CUTLASS  # ← broken on sm_121
  ```
- The fix — three environment variables:
  ```bash
  VLLM_USE_FLASHINFER_MOE_FP4=0
  VLLM_NVFP4_GEMM_BACKEND=marlin
  VLLM_TEST_FORCE_FP8_MARLIN=1
  ```
  | Variable | Value | Purpose |
  |----------|-------|---------|
  | `VLLM_USE_FLASHINFER_MOE_FP4` | `0` | Disables FlashInfer's FP4 MoE kernel path |
  | `VLLM_NVFP4_GEMM_BACKEND` | `marlin` | Forces Marlin for all NVFP4 linear layers |
  | `VLLM_TEST_FORCE_FP8_MARLIN` | `1` | Also routes FP8 operations through Marlin |

  Marlin dequantizes FP4 to BF16 on the fly using operations valid on sm_121; it only needs capability ≥ 75 (Turing). Expected post-fix log lines:
  ```
  Using NvFp4LinearBackend.MARLIN for NVFP4 GEMM
  Using 'MARLIN' NvFp4 MoE backend out of potential backends: ['VLLM_CUTLASS', 'MARLIN']
  ```
- Benchmark (DGX Spark GB10, Nemotron-3-Nano-30B-A3B-NVFP4, 19 GB model; identical settings except backend):

  | Backend | Memory | tok/s | Notes |
  |---------|:------:|:-----:|-------|
  | **Marlin** | **32 GB** | **50.0** | Clean, no errors |
  | FlashInfer (default) | 39 GB | 42.6 | CUTLASS errors in log, falls back |

- Full launch command (flags as in the thread):
  ```bash
  docker run -d --runtime=nvidia --name nemotron-nvfp4 \
    -v /path/to/hf-cache:/root/.cache/huggingface \
    -p 8000:8000 \
    -e VLLM_USE_FLASHINFER_MOE_FP4=0 \
    -e VLLM_NVFP4_GEMM_BACKEND=marlin \
    -e VLLM_TEST_FORCE_FP8_MARLIN=1 \
    vllm-node:latest \
    python3 -m vllm.entrypoints.openai.api_server \
      --host 0.0.0.0 --port 8000 \
      --model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4 \
      --enforce-eager --gpu-memory-utilization 0.2 \
      --max-model-len 8192 --kv-cache-dtype fp8 --trust-remote-code
  ```
- Applies to any NVFP4/ModelOpt FP4 model on sm_121 (DGX Spark) or sm_120 (RTX 5090, RTX PRO 6000) — all consumer Blackwell lacking `tcgen05`. Reported affected: Nemotron-3-Nano-30B-A3B-NVFP4, Nemotron-3-Super-120B-A12B-NVFP4, Qwen3-VL-235B-A22B-NVFP4, Qwen3.5-122B-A10B-NVFP4, GLM-4.7-Flash-NVFP4.
- Native FP4 on sm_121: no NVIDIA timeline at the time. Active upstream work: CUTLASS #3038 (SM121-gated MXFP4 kernel wiring), vLLM #35947 (software E2M1 conversion for SM12x), vLLM #38126 (architecture suffix preservation, merged). Until native support lands, Marlin is the recommended path — it does not use native FP4 tensor cores, but is faster than the broken CUTLASS fallback and keeps the full NVFP4 memory savings.
- Credit: Marlin discovery came from the DGX Spark community — ["We unlocked NVFP4 on the DGX Spark: 20% faster than AWQ!"](https://forums.developer.nvidia.com/t/we-unlocked-nvfp4-on-the-dgx-spark-20-faster-than-awq/361163).

**Links:** <https://forums.developer.nvidia.com/t/marlin-fix-nvfp4-actually-works-on-sm121-dgx-spark/365119> · related: <https://forums.developer.nvidia.com/t/nvfp4-on-dgx-spark-gb10-is-broken-i-bought-9-of-these-for-this-feature-requesting-nvidias-official-roadmap-and-response/367082> · <https://forums.developer.nvidia.com/t/your-gpu-does-not-have-native-support-for-fp4-computation-but-fp4-quantization-is-being-used/355494> · <https://forums.developer.nvidia.com/t/fp4-on-dgx-spark-why-it-doesnt-scale-like-youd-expect/360142>

---

## MoE performance below reference on Thor (stock vLLM 26.02)

**TL;DR:** MoE models on stock `nvcr.io/nvidia/vllm:26.02-py3` (vLLM 0.15.1) on Thor run **significantly below published reference**, while dense models meet or beat it. The server logs show **no Thor fused-MoE config** exists and unquantized MoE falls back to Triton — the missing fused-MoE config is the suspected cause.

**Evidence** (forum thread, opened Mar 25, 2026 by waycore; 10 posts, closed Jul 8, 2026):

- Environment: Jetson AGX Thor Dev Kit, Ubuntu 24.04.4 LTS, kernel 6.8.12-tegra, CUDA 13.0, driver 580, container `nvcr.io/nvidia/vllm:26.02-py3` (vLLM 0.15.1), power mode MAXN, clocks locked (`jetson_clocks`).
- Benchmark setup: input ~2048 tokens, output ~128 tokens, random synthetic prompts, C1 (single request) and C8 (8 concurrent); repeated until stable (~5 % variance).
- Results (stable runs):

  | Model | C1 tok/s | C8 tok/s | Reference |
  |---|---|---|---|
  | Llama 3.1 8B (dense) | ~45 | ~270 | consistent with / better than published Thor references |
  | Qwen3-30B-A3B (MoE) | ~34 | ~96 | ~61 (C1), ~226 (C8) |
  | Mixtral-8x7B (MoE) | ~7 | ~14 | — |

- Observations: warmup alone insufficient (multiple runs to stabilize); dense models scale well under concurrency; MoE models show reduced throughput and higher latency, especially under concurrency.
- Relevant server-log output:
  ```
  Using default MoE config. Performance might be sub-optimal!
  Config file not found at:
  .../fused_moe/configs/E=128,N=768,device_name=NVIDIA_Thor.json

  Not enough SMs to use max_autotune_gemm mode

  Using TRITON backend for Unquantized MoE
  ```
- Questions raised: (1) are Thor-specific fused-MoE config files expected in the container (26.02+)? (2) is Triton the intended MoE backend on Thor? (3) recommended runtime flags/tuning for MoE on Thor? (4) known MoE vs dense limitations/best practices?
- Attachments in the thread: [thor_vllm_benchmark_playbook_fixed_v3_with_results_add.docx](https://forums.developer.nvidia.com/uploads/short-url/7Xqa8RcJPpBD7CEuIoO4iZqof41.docx) (207.9 KB) and [benchresult.txt](https://forums.developer.nvidia.com/uploads/short-url/6xsqNuF46ry2qefhpDSSVo5ZHkC.txt) (1.5 KB).

**Links:** <https://forums.developer.nvidia.com/t/jetson-agx-thor-vllm-26-02-moe-performance-significantly-below-reference-missing-fused-moe-config/364663> · related: <https://forums.developer.nvidia.com/t/nemotron-3-nano-on-jetson-thor-vllm-itl-degrades-4-7x-with-concurrency-mtp-rejected/370044> · <https://forums.developer.nvidia.com/t/benchmark-report-qwen3-6-35b-a3b-nvfp4-on-nvidia-dgx-spark-jetson-thor-blackwell-6000-pro/371810> · <https://forums.developer.nvidia.com/t/performance-comparison-of-qwen3-30b-a3b-awq-on-jetson-thor-vs-orin-agx-64gb/345449> · <https://forums.developer.nvidia.com/t/recipes-to-run-qwen3-5-models-on-thor/363532>

---

## Nemotron NVFP4 on Thor — forum findings

**TL;DR:** A full working recipe for **NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4** on a single Jetson AGX Thor (128 GB unified) was posted: from-source vLLM + from-source FlashInfer (AOT) with the sm_110a env-var set, served with `--attention-backend TRITON_ATTN`, fp8 KV, chunked prefill, and capped cudagraph capture. Measured single-stream generation throughput: ~10.5 tok/s.

**Evidence** (forum thread, opened Mar 14, 2026 by shahizat; 11 posts, closed Apr 22, 2026):

- Build environment for sm_110a:
  ```bash
  export TORCH_CUDA_ARCH_LIST=11.0a
  export CUDA_HOME=/usr/local/cuda-13
  export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
  export PATH="${CUDA_HOME}/bin:$PATH"
  ```
  (Note the `CUDA_HOME=/usr/local/cuda-13` variant — the versioned path, cf. the env-var audit.)
- Steps: fresh venv (uv, python 3.12); torch from `https://download.pytorch.org/whl/cu130`; build vLLM from source (`python3 use_existing_torch.py`, `MAX_JOBS=$(nproc) python3 setup.py bdist_wheel`); build FlashInfer from source with `FLASHINFER_CUDA_ARCH_LIST="11.0a"` (`python -m flashinfer.aot`, `python3 -m build --no-isolation --wheel`); build `flashinfer-cubin`; build `flashinfer-jit-cache` (edit `pyproject.toml` to remove the `nvidia-nvshmem-cu12` dep first); clear `~/.cache/flashinfer/` and `~/.cache/vllm/`.
- Serve command (abridged to the load-bearing flags):
  ```bash
  vllm serve nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 \
    --async-scheduling --dtype auto --kv-cache-dtype fp8 \
    --tensor-parallel-size 1 --pipeline-parallel-size 1 --data-parallel-size 1 \
    --trust-remote-code --attention-backend TRITON_ATTN \
    --gpu-memory-utilization 0.8 \
    --max-cudagraph-capture-size 32 --max-num-seqs 32 \
    --enable-chunked-prefill \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder \
    --reasoning-parser-plugin "./super_v3_reasoning_parser.py" --reasoning-parser super_v3
  ```
  Without `--max-cudagraph-capture-size`, vLLM captures CUDA graphs for every batch size in [1, 2, 4, 8, 16, … up to 512] — the flag caps it at 32.
- Measured (from the thread's server logs, single request): "Avg generation throughput: 10.5 tokens/s, Running: 1 reqs, GPU KV cache usage: 0.8 %" (first minute 7.4 tok/s while the warmup/precompute settled; stable at ~10.4-10.7 tok/s).
- Related failure documented in the thread's links: NIM `qwen3.5-35b-a3b:1.7.0-variant` fails on Thor — "Triton ptxas-blackwell does not recognize sm_110a".

**Links:** <https://forums.developer.nvidia.com/t/running-nvidia-nemotron-3-super-120b-a12b-nvfp4-on-the-nvidia-jetson-thor/363485> · related: <https://forums.developer.nvidia.com/t/title-nim-qwen3-5-35b-a3b-1-7-0-variant-fails-on-jetson-agx-thor-triton-ptxas-blackwell-does-not-recognize-sm-110a/364963/7> · <https://forums.developer.nvidia.com/t/run-vllm-in-thor-from-vllm-repository/348804> · <https://forums.developer.nvidia.com/t/jetson-agx-thor-cuda13-vllm0-11-0/350739>

---

## Qwen3-VL-8B FP8 crash on Thor (official Jetson Thor image)

**TL;DR:** On the official Jetson Thor vLLM container, **Qwen3-VL-8B-Instruct (BF16) serves fine, but Qwen3-VL-8B-Instruct-FP8 fails** — the poster's read: "It seems that vllm does not support sm110?" The failure is part of the same sm100f-kernel family as the garbled-output and CUBLAS_STATUS_INTERNAL_ERROR threads.

**Evidence** (forum thread, opened Apr 9, 2026 by steven6_wang; 9 posts, closed May 6, 2026):

- Setup: Jetson AGX Thor settings screenshot; image from NVIDIA — [ghcr.io/nvidia-ai-iot/vllm:latest-jetson-thor](http://ghcr.io/nvidia-ai-ot/vllm:latest-jetson-thor).
- Image package inventory (relevant to the crash): `flash_attn==2.8.4`, `flashinfer-cubin==0.6.6`, `flashinfer-jit-cache==0.6.6+cu130`, `flashinfer-python==0.6.6`, `nvidia-cutlass==4.2.1.0`, `nvidia-cutlass-dsl==4.4.2`, `cuda-bindings==13.0.1`, plus the full vLLM stack (fastapi 0.135.3, grpcio 1.80.0, …).
- Behavior: "I can normally run Qwen3-VL-8B-Instruct on Jetson AGX Thor using vllm. But it failed on Qwen3-VL-8B-Instruct-FP8." (screenshots of the failure attached in the thread).
- Thread cross-links the same failure family: <https://forums.developer.nvidia.com/t/vllm-fp8-models-unusable-on-agx-thor-sm-11-0-kernels-compiled-for-sm100f-only-this-kernel-only-supports-sm100f-cublas-status-internal-error/375912> ("This kernel only supports sm100f." → `CUBLAS_STATUS_INTERNAL_ERROR`) and the garbled-output thread (351272).

**Links:** <https://forums.developer.nvidia.com/t/can-not-run-qwen3-vl-8b-instruct-fp8-on-jetson-agx-thor-using-vllm/366086>

---

## DGX Spark: state of FP4/NVFP4 (PSA + community playbook)

**TL;DR:** The PSA thread (page 8, late-March 2026) shows the FP4-on-Spark situation was still churning: the "Spark Expert" community image author had **switched NVFP4 recipes from Marlin back to `VLLM_CUTLASS`** because of model-quality degradation reports with Marlin ("seems to be very stable, unlike FlashInfer"), while FlashInfer-0.6.7-nightly builds still **crashed** on long decode and the pending CUTLASS fix proved insufficient. The mid-2026 community playbook consolidates the durable state: sm_121 has **no native FP4**, the CUTLASS FP4 GEMM crashes on sm120/sm121, Marlin is the working path (with the quality caveat noted), and the container→vLLM-version mapping is what actually matters.

### PSA thread (page 8, Mar 29-30, 2026)

Evidence (posts 139-147 of [PSA: State of FP4/NVFP4 Support for DGX Spark in VLLM](https://forums.developer.nvidia.com/t/psa-state-of-fp4-nvfp4-support-for-dgx-spark-in-vllm/353069)):

- eugr ("Spark Expert"): "Actually, I switched NVFP4 recipes to VLLM_CUTLASS last week because people were experiencing model quality degradation with Marlin. Seems to be very stable, unlike Flashinfer."
- johnny_nv: "could you try with my latest commit? I added fix for nemotron and nvfp4 accuracy" — with the flashinfer nightly wheels (`nightly-v0.6.7-20260328`: `flashinfer_python-0.6.7.dev20260328` + `flashinfer_cubin-0.6.7.dev20260328` from the [flashinfer releases page](https://github.com/flashinfer-ai/flashinfer/releases/download/nightly-v0.6.7-20260328/flashinfer_python-0.6.7.dev20260328-py3-none-any.whl)); "yes, until 0.6.8 yes…"
- eugr: "I'm still getting crashes. Tested with Flashinfer built from the source from main during my nightly build."
- trystan1: installs from [johnnynunez/vllm](https://github.com/johnnynunez/vllm/tree/main) + flashinfer nightlies on top; raises that if the tile-shape mismatch causes the crash during long decode sequences, `vllm_cutlass` should exhibit the same (it is "ultimately a cutlass bug").
- eugr (edit): "the only way to avoid the crash currently is: 1. Rebuild both vLLM and Flashinfer with CUTLASS PR applied; 2. or use CUDA_LAUNCH_BLOCKING=1? — EDIT: looks like CUTLASS fix is not enough since there is an issue with that PR as well. I guess, we'll just have to wait a little bit longer."

### Community playbook (mid-2026, "Running vLLM on NVIDIA DGX Spark: The Complete Playbook", vlaicu.io, 2026-06-20)

Durable state and rules worth carrying over:

- **Hardware facts:** GB10 Grace Blackwell SoC, 128 GB unified CPU+GPU memory, consumer Blackwell `sm_121`, ~273 GB/s bandwidth, aarch64 host. Decode is bandwidth-bound; keep `--max-num-seqs` low (1-4); NVFP4 MoE with ~3-13 B active params is the sweet spot.
- **The sm121 catch:** GB10 has **no native FP4 compute**; the CUTLASS FP4 GEMM crashes on sm120/sm121 (`[FP4 gemm Runner] Failed to run cutlass FP4 gemm on sm120`), and FlashInfer-TRTLLM MoE is "SM100+ only". The working path is **Marlin (W4A16)**: force `--moe-backend marlin` (NVFP4) or `VLLM_MXFP4_BACKEND=marlin` (MXFP4). "If any FP4 model emits `!!!!!` or empty output on Spark, it's an sm121 FP4-kernel problem, not your prompt."
- **vLLM 0.23 deprecated the FlashInfer-MoE env vars** (`VLLM_USE_FLASHINFER_MOE_FP4`, `VLLM_USE_FLASHINFER_MOE_FP8`, …) in favor of the `--moe-backend` flag (`marlin`, `flashinfer_cutlass`, `flashinfer_trtllm`, `flashinfer_cutedsl`); on 0.23+ images they log "Unknown vLLM environment variable detected" and do nothing. They still work on the NGC 0.19 image.
- **Container image → vLLM version (known-good mappings, mid-2026):**

  | Image | vLLM | Notes |
  |---|---|---|
  | `nvcr.io/nvidia/vllm:26.04-py3` | **0.19.0** | CUDA 13.2.1, PyTorch 2.12; the de-facto stable Spark NGC image. Ships no Ray. |
  | `nvcr.io/nvidia/vllm:26.03-py3` | 0.17.1 | previous NGC |
  | `nvcr.io/nvidia/vllm:26.02-py3` | 0.15.1 | **rejects MIXED_PRECISION checkpoints** (mixed FP8+NVFP4 MoE) |
  | `nvcr.io/nvidia/vllm:25.12.post1-py3` | (Dec-2025) | the Spark/Jetson image NVIDIA points to for Nemotron-3-Nano |
  | `vllm/vllm-openai:nightly` / `:cu130-nightly` | 0.23.x | upstream; newest archs + DFlash. **Moves** — pin a digest |
  | `vllm/vllm-openai:gemma4-cu130` | Gemma-4 build | **required** for Gemma 4 on GB10 — the bare `:gemma4` tag is v0.18.2-dev and **crashes on sm121** (`FP4 gemm Runner ... sm120/sm121`) |
  | `vllm/vllm-openai:gemma` | diffusion build | official aarch64-cu130 image with `diffusion_gemma` baked in for GB10 |

- **Two entrypoint conventions** (#1 first-run mistake): `vllm/vllm-openai:*` images already `ENTRYPOINT vllm serve` (pass only the model + flags); `nvcr.io/nvidia/vllm:*` are pass-through (write the full `vllm serve <model>`).
- **Quant-flag rule:** `--quantization modelopt` for `nvidia/...` ModelOpt checkpoints; **omit** it for compressed-tensors (Unsloth/RedHat — auto-detect).
- **Measured GB10 performance (single DGX Spark, this playbook's bench script):** Nemotron-3-Nano-30B-A3B ~56 tok/s, Gemma-4-26B-A4B ~52, dense Gemma-4-31B ~6, Qwen3-Coder-Next ~43, Qwen3.5-122B-A10B ~16, Nemotron-Nano-Omni ~50. Qwen3.6-35B-A3B (MTP-3): answer-only 102.0 / thinking 124.8 tok/s (NVFP4, 3-run medians) vs Gemma-4-26B-A4B (no draft) 49.6 / 51.1. Concurrency: Gemma-4-31B 6 tok/s solo → ~92 tok/s aggregate at 16 concurrent; Qwen2.5-3B 26 → 477 tok/s at 16 concurrent (≈1,460 at 64).
- **Speculative decoding on Spark:** DFlash on Qwen3.5-27B NVFP4 — baseline 12.2 tok/s → 33.2 tok/s (n=15, short prompt) / 26.3 (n=15, long prompt): up to 2.2-2.7×, content-dependent. DFlash requires vLLM ≥0.21 + `--attention-backend triton_attn`. **DFlash gotcha:** on quantized (NVFP4/FP8) weights under **stock** vLLM, acceptance collapses to ~4 of 15 tokens — use the sm121-patched community build (AEON) or a BF16 target.
- **Prefill is compute-bound and Marlin FP4 is slower for it:** ~90 B dense model at large context took **~133 s** to first token; keep prompts tight and use `--enable-prefix-caching`.
- **Unified-memory OOM valve:** the Linux page cache can hold memory CUDA can't reclaim — an "OOM" well under 128 GB; flush with `sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'`.

**Links:** PSA thread <https://forums.developer.nvidia.com/t/psa-state-of-fp4-nvfp4-support-for-dgx-spark-in-vllm/353069> · playbook <https://vlaicu.io/posts/dgx-vllm/> · [NVIDIA official Spark vLLM instructions](https://build.nvidia.com/spark/vllm/instructions) · [NVIDIA/dgx-spark-playbooks](https://github.com/NVIDIA/dgx-spark-playbooks) · [vLLM team blog: vLLM on DGX Spark (2026-06-01)](https://vllm.ai/blog/2026-06-01-vllm-dgx-spark) · [AEON-7/vllm-dflash](https://github.com/AEON-7/vllm-dflash) · [NGC vLLM container tags](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/vllm) · [recipes.vllm.ai](https://recipes.vllm.ai/) · related Spark FP4 threads: <https://forums.developer.nvidia.com/t/new-bleeding-edge-vllm-docker-image-avarok-vllm-nvfp4-gb10-sm120/354231> · <https://forums.developer.nvidia.com/t/help-running-nvfp4-model-on-2x-dgx-spark-with-vllm-ray-multi-node/353723> · <https://forums.developer.nvidia.com/t/qwen3-8-27b-on-dgx-spark-using-vllm-nvfp4-vs-fp8-performance/380258>
