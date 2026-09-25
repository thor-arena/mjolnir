# c1 Regression Fix Study — FlashInfer CUDA-Graph Capture on Thor

> **Status:** Concluded — verdict: the forensics mechanism does not hold; FA2 `batch_prefill` is replay-safe by construction · **Date:** 2026-09-19 · **Scope:** Static design study (no code changes) of the c1 (single-request) regression introduced by granting the DSpark draft a non-causal CUDA-graph capture path on sm_110a; produced the design for the draft-CG gate patch. This study informed the draft-cudagraph gate patch shipped in the 13-patch stack — see [`docker/vllm-thor/PATCHES.md`](../../docker/vllm-thor/PATCHES.md).

## TL;DR

- The root-cause hypothesis from the earlier forensics pass ("FlashInfer prefill grid + split-KV **baked** at capture from dummy short `seq_lens`") **does not hold** in this exact code/platform combination, and the evidence is static and airtight.
- On Thor (sm_110a) every FlashInfer attention path in this deployment resolves to the **classic FA2 `batch_prefill` JIT module** — not fmha_v2, not FA3, not cuDNN. The fmha_v2 grid quoted by the forensics belongs to the **TRT-LLM SM120-only** kernel family and can never be selected on sm_110a.
- The FA2 prefill CUDA-graph design (FlashInfer 0.6.18) is **replay-safe by construction**: the launch grid is fixed at capture at the SM cap, but the *per-CTA work assignment* (tile tables, `kv_chunk_size`, `merge_indptr`, `block_valid_mask`, device `total_num_rows`) is **re-derived every runtime step** by the eager `plan()`/`fast_decode_plan()` and H2D-copied into a persistent workspace. Long KV at replay still gets split-KV up to the SM cap.
- JIT ground truth: forcing a `plan()`+`run()` on the host GPU produced only `batch_prefill_with_kv_cache_*` modules for 110a — no `trtllm_fmha_v2_*`, no `batch_decode_*` modules. The tensor-core decode wrapper (target spec-verify) rides the **same** `batch_prefill` module.
- The c1 delta (−7…−16 % at N=1, K=7, long context; draft-side only, target identical between the compared configs) must therefore be one of: (a) graph-replay overhead cancelling the launch savings on a tiny 5-layer draft; (c) per-step eager `plan()` cost (shared with the no-graph config, so it cancels — unless workspace sizes differ); (e) in-graph inductor GEMM config vs autotune.
- Ranked recommendation: (1) run the confirm-first nsys experiment to localize the delta; (2) **ship the shape-gated draft-CG replay** (gate: replay the graph only where launch savings beat replay overhead; fall back to eager otherwise) as the production fix. The resulting patch is described in the final section of this document.

---

## 1. Deployment facts

| Fact | Value | Source |
|---|---|---|
| GPU | NVIDIA Thor, sm_110a (cc 11.0), **20 SMs**, 128 GB HBM, aarch64, CUDA 13 | `nvidia-smi` on host |
| vLLM tree | patched vLLM tree, 11 patches applied (50885, 49652, 54165, 52244, 55390, 55519, thor-fused, thor-dspark, thor-gdn, thor-b12x, thor-trtllm, thor-fa4) | `docker/vllm-thor/PATCHES.md` |
| FlashInfer | 0.6.18.post1 | `vllm/vllm-openai:nightly-aarch64` image, `pip show` |
| cudagraph mode | `FULL_AND_PIECEWISE`; capture sizes [1,2,4,8,…,160]; block_size 880; `max_model_len` 204800 | forensics `configs.md` |
| Draft (DSpark) | 40 q / 8 kv heads (GQA 5:1), head_dim 128, K=7 (`num_query_per_req=7`), 5 non-causal full-attn layers, hidden 5120 | `docker/vllm-thor/PATCHES.md` |
| Target (Qwen3.8-27B) | 24:4 GQA, head_dim 256, NVFP4 weights, fp8 KV | `docker/vllm-thor/PATCHES.md` |
| FI autotune | `enable_flashinfer_autotune=true` → `flashinfer_autotune()` in warmup, **before** capture | `vllm/model_executor/warmup/kernel_warmup.py:249-256` |
| Fixed split sizes | **disabled by default**: `decode_fixed_split_size=-1`, `prefill_fixed_split_size=-1`, `disable_split_kv=False` (only `VLLM_BATCH_INVARIANT` sets 2048/4096 + disable) | `vllm/v1/attention/backends/flashinfer.py:711-717` |

All FlashInfer source anchors below are in the `vllm/vllm-openai:nightly-aarch64` image, `flashinfer==0.6.18.post1` (abbreviated `$FI`); all vLLM anchors are in the patched vLLM tree above.

Derived constants (SM cap for split-KV): `max_batch_size_if_split = 2·num_SM/num_kv_heads` → **draft: 40/8 = 5**, **target: 40/4 = 10** (`scheduler.cuh` `PrefillSplitQOKVIndptr`).

## 2. Per-path fix-site map

Three capture paths exist. "Baked verdict" = what is frozen at graph capture vs re-derived at each replay step, in FlashInfer 0.6.18 on sm_110a.

| # | Path | Capture site (dummy inputs) | Runtime re-plan site | Baked at capture | Re-derived every step | Verdict |
|---|---|---|---|---|---|---|
| 1a | **DRAFT non-causal prefill** (dspark patch; the c1 suspect) | `v1/worker/gpu/spec_decode/dflash/speculator.py:148-164` `capture()` → `dflash/cudagraph.py:24-72` `_prepare_dflash_inputs_to_capture` → dummy `seq_lens = num_tokens//num_reqs` (`v1/worker/gpu/input_batch.py:121-213`, seq_lens at :152-153 ⇒ **N=1 ⇒ [7]**, 1 dummy page) | `speculator.py:450-459` — comment: "Rebuild the draft attention metadata even when replaying the FULL graph" → FI `build()` → per-bucket non-causal wrapper `plan()` (`v1/attention/backends/flashinfer.py:1279-1295` `_get_noncausal_prefill_wrapper_cudagraph`, wrapper per bucket at :755) | grid.x = `padded_batch_size` = max(SM cap 5, dummy tiles) = **5**; `split_kv=true`; `cta_tile_q` template (from dummy QO len 35 → 32); NUM_MMA_KV (head_dim); workspace layout (f(padded, cta_tile_q) — **not** f(kv_len)) | request/qo-tile/kv-tile tables, `kv_chunk_size` (binary-search, tiles ≤ SM cap), `merge_indptr`, `block_valid_mask`, device `total_num_rows` — H2D every step (`scheduler.cuh:820-900`) | **Not under-parallelized at replay.** c1: eager grid (4 q-tiles × kv-chunks…) vs replay grid (5,1,8)=40 CTAs with ≤5 active — same or better parallelism. |
| 1b | **TARGET spec-verify decode** (50885 grant; q_len_per_req = 1+K = 8, tensor-core decode) | `v1/worker/gpu/cudagraph_utils.py:592` `ModelCudaGraphManager.capture` → `prepare_inputs_to_capture` → `InputBatch.make_dummy` (max_query_len from descriptor); `model_states/default.py:194-197`: capture uses `max_seq_len = max_model_len` (204800) — but FA2 plan tiles from **dummy indptr values**, not `max_seq_len` | FI `build()` every step; TC decode `plan()`/`fast_decode_plan()` (`$FI/decode.py:3826-4040`) re-runs the full host plan with real `indptr_host` (`$FI/decode.py:4003-4030`) | grid.x = padded = max(SM cap 10, dummy tiles) = **10**; `split_kv=true`; `q_len_per_req=8` **frozen per wrapper** ("frozen cudagraph shape", `$FI/decode.py:3866-3871`); `cta_tile_q` from QO=8 (workload-invariant) | same tile-table set as 1a (same `batch_prefill` module — confirmed by the JIT cache containing no `batch_decode_*`) | **Not under-parallelized at replay.** Identical kernel in both compared configs → cannot explain the delta anyway. |
| 1c | **TARGET non-spec decode** (single token) | same as 1b with q_len_per_req=1 | same | same, q_len_per_req=1 frozen | same | **Not under-parallelized.** Identical in both compared configs. |

### 2.1 What actually happens at replay (generated host code)

Materialized module in the JIT cache ground truth: `…/0.6.18.post1/110a/generated/batch_prefill_with_kv_cache_dtype_q_bf16_…/batch_prefill.cu` (373 lines; host `plan`/`run` mirror of `$FI/data/csrc/batch_prefill.cu`):

- `run` (lines 109-225): builds `params` from the persistent workspace: `request_indices / qo_tile_indices / kv_tile_indices / o_indptr / kv_chunk_size_ptr / merge_indptr / block_valid_mask / total_num_rows` (device ptr) all via `GetPtrFromBaseOffset` into the captured int/float workspace; the **only by-value baked scalars** are `params.padded_batch_size` (→ grid.x) and `params.max_total_num_rows` (merge bookkeeping).
- Kernel: `DISPATCH_CTA_TILE_Q(plan_info.cta_tile_q, …)` then `BatchPrefillWithPagedKVCacheDispatched` (`$FI/data/include/flashinfer/attention/prefill.cuh:4335`) with `grid = (padded_batch_size, 1, num_kv_heads)`; graph mode adds the `block_valid_mask` early-out for idle padded CTAs and a **persistent-grid merge kernel** for split-KV reduction.

So the forensics expectation ("graph should show a smaller grid / no split-KV for long KV") is **inverted** in 0.6.18: the grid is the *largest* it will ever be (SM cap), and split-KV for long KV is *enabled* at replay via the dynamic tables.

### 2.2 Per-step CPU cost shared by both configs

Even when the FULL graph is replayed, every step still runs eagerly: `prepare_dflash_inputs` + `precompute_and_store_context_kv` (`speculator.py:379-427`) + FI `build()` → `plan()` host grid compute + **full int-workspace H2D** (`scheduler.cuh:894-897`, `cudaMemcpyAsync` of `int_workspace_size_out`). The no-graph (eager-draft) config paid the same `plan()` cost per step — this term **cancels** between the two configs and is not the delta (it is forensics hypothesis (a)/(c), which the forensics ranked second only because (b) looked stronger on paper).

## 3. The FlashInfer decision function (plan → grid → split-KV)

Chain, all in 0.6.18:

1. **Backend resolution** — `BatchPrefillWithPagedKVCacheWrapper.plan` (`$FI/prefill.py:3443+`), "auto" branch at :3804: `determine_attention_backend` (`$FI/utils.py:545-600`): sm90a+supported → "fa3"; **else "fa2"** (Thor lands here). The fmha_v2 override (:3816-3845) additionally requires `_should_use_fmha_v2_sm120` (`$FI/utils.py:523`) → SM120 only. ⇒ **Thor always gets the classic FA2 module.**
2. **Host plan** — `$FI/data/csrc/batch_prefill.cu` → `PrefillPlan` / `PrefillPlanImpl` (`$FI/data/include/flashinfer/attention/scheduler.cuh`):
   - `cta_tile_q`: per-bucket QO tile (32/64/128…); **template**, chosen from QO lengths only — KV-length-independent.
   - `PrefillSplitQOKVIndptr`: `padded_batch_size = max(max_batch_size_if_split, total_num_tiles_q)`; `max_batch_size_if_split = 2·num_SM/num_kv_heads` (draft 5 / target 10).
   - `PrefillBinarySearchKVChunkSize`: `split_kv` forced **true** in cuda-graph mode; picks the largest `kv_chunk_size` (page multiples) such that total tiles ≤ SM cap → long KV *does* get split at replay.
   - Materialize (:820-900): tile tables, `kv_chunk_size`, `merge_indptr`, `block_valid_mask`, `total_num_rows` → pinned buffer → **H2D every `plan()`**.
3. **Replay launch** — generated `batch_prefill.cu` `run` + `prefill.cuh:4335` (grid = `(padded_batch_size, 1, kv_heads)`), persistent merge kernel.

**Existing knobs (all already checked in this deployment):**

| Knob | Where | Value here | Effect if changed |
|---|---|---|---|
| `fixed_split_size` | `flashinfer.py:711-717`, passed to plan | −1 (auto) | pinning a small fixed chunk *could* change split granularity, but the binary search already maximizes parallelism ≤ SM cap; pinning larger = worse |
| `disable_split_kv` | same | False | disabling = the actual §5(b)-style regression; do not |
| `q_len_per_req` / `uniform_q_len` | plan arg; frozen per wrapper in graph mode (`$FI/decode.py:3866`) | 7 (draft) / 8, 1 (target) | workload-invariant anyway |
| `VLLM_BATCH_INVARIANT` | env | off | would force fixed splits + disable_split_kv — leave off |
| `enable_flashinfer_autotune` | `kernel_warmup.py:249-256` | true | tunes **inductor GEMM/MoE** kernels (captured into graphs); FI attention `plan()` is deterministic — **no** attention tactic autotune exists |

**There is no replay-time re-plan mode to enable** — per-step eager `plan()` with a persistent workspace *is* the CUDA-graph design of this module.

## 4. Options A–E (re-ranked under the new evidence)

| Opt | Description | Exact site(s) | Verdict |
|---|---|---|---|
| **A** | Re-capture draft with long dummy `seq_lens` (e.g. 8192) — the forensics §5(b) fix | `dflash/cudagraph.py:24-72` (dummy seq_lens/indptr), `input_batch.py:121-213` `make_dummy` | **Predicted null** for 0.6.18: grid already = SM cap; tile tables re-planned every step; `cta_tile_q` is QO-based (unchanged). Long dummies only change the *initial* `kv_chunk_size`/tables, overwritten at the first runtime `plan()`. **Keep as a 30-min falsification A/B, not as the fix.** |
| **B** | **Gate draft CG replay by batch/seq shape** — replay the FULL graph only where launch savings beat replay overhead; fall back to eager otherwise | Gate: `speculator.py:122-146` (`init_cudagraph_manager`) and/or at dispatch `dp_utils.py:213` `dispatch_cg_and_sync_dp` (the eager `NONE` fallback path already exists: `speculator.py:469-480`). Heuristic seed from §6 data (e.g. N≤2 & ctx>4k → eager; else CG) | **Top fix candidate.** Small, contained in the dspark patch, reversible, keeps the win at larger N / shorter ctx. |
| **C** | Revert the non-causal CG grant (draft fully eager) | `flashinfer.py:1079-1086` (thor branch of `get_cudagraph_support`, comment at :1071) or `speculator.py:128-139` | **Isolation baseline, not a fix** — the no-graph config in the matrix *is* this state; use only to re-confirm the delta is 100% draft-CG (it already is, since the target is unchanged). |
| **D** | Add replay-time split-KV re-derivation | — | **Already the behavior** in 0.6.18 (§3, step 2: per-step `plan()` re-materializes split-KV tables). Document so the team doesn't re-derive it. |
| **E** | Multi-shape capture with per-shape long dummies | superset of A | **Rejected** — same null prediction as A, strictly more complexity (workspace/VRAM per shape). |
| **F** | Instrument + in-graph inductor audit (forensics (a)/(c)/(e)) | nsys/ncu on a single c1 step, graph vs eager; compare (i) FI prefill kernel grid + time inside graph vs eager, (ii) `cudaGraphLaunch` CPU cost, (iii) `plan()` int-workspace H2D bytes, (iv) inductor GEMM kernel/config chosen under CG vs eager | **Confirm-first experiment** (§6). |

## 5. Ranked recommendation

1. **Run F (confirm-first, §6) before shipping any code.** It falsifies or confirms the forensics hypothesis end-to-end and localizes the c1 delta to one of: {replay overhead, in-graph GEMM config, `plan()`/H2D}.
2. **Ship B (gated draft CG)** as the production fix, threshold from step-1 data. Fallback (eager) already exists in the dspark code path (`speculator.py:469-480`), so the change is a dispatch-site condition, not new infrastructure.
3. Run **A as a null-check A/B** (30 min) — expected no change at c1; if it *does* change c1, the deployment's effective FlashInfer differs from 0.6.18 and the forensics hypothesis revives → re-audit that build.
4. Reject D (no-op) and E (superset of A).

**What would falsify this study:** nsys showing the in-graph FA2 kernel with a *smaller* grid or *no* split-KV than the eager kernel for the same real `seq_lens` (i.e., the dynamic tables not being updated) — check the deployed container's actual `flashinfer` version and its JIT cache first (below).

## 6. Confirm-first experiment

```
# c1, K=7, one decode step, ctx≈8k — nsys single-step, graph config vs no-graph config
docker exec -it <container> nsys profile --trace=cuda,nvtx -o /tmp/c1 \
  -- python -m vllm.entrypoints.openai.api_server <c1 one-step workload>
# In the draft window (5 non-causal attn layers + GEMMs + precompute + sampling):
#   1) FA2 batch_prefill kernel: compare grid dims + kernel time, graph vs eager
#      (expect: grid (5,1,8) with block_valid_mask vs (4·chunks,1,8) eager; same/less time)
#   2) cudaGraphLaunch CPU cost vs eager launch total for the draft module
#   3) plan() int-workspace H2D bytes (expect identical between configs)
#   4) inductor GEMM kernel names/times inside graph vs eager
nsys stats --report cuda_gpu_kern_sum --format csv /tmp/c1.nsys-rep
# Null-check A (falsify the forensics hypothesis end-to-end): patch dummy seq_lens=8192 in
# dflash/cudagraph.py, re-run c1 → expect 0 Δ (30 min)
```

Decision rule: (1) equal or better in-graph attention ⇒ forensics hypothesis dead; delta then sits in (2)/(4) ⇒ ship B with threshold from (2). If (1) shows a frozen under-parallelized grid ⇒ the deployed flashinfer ≠ 0.6.18 ⇒ check `pip show flashinfer` + `~/.cache/flashinfer` in the deployed container and re-run this study against that build.

## 7. Caveats / open questions

1. **Nightly drift:** the forensics was written Sep 10-11 against whatever nightly was current then; this study verifies the *current* image (0.6.18.post1, `nightly-aarch64`). The forensics fmha_v2 grid citation (`generator_utils.py:457/831/984/1373` → actual 0.6.18 lines 465/524) is the SM120 TRT-LLM path and **cannot execute on sm_110a**. If the deployed container pins a different flashinfer, re-verify: `pip show flashinfer`, list `~/.cache/flashinfer/*/generated/` (expect `batch_prefill_with_kv_cache_*_…_110a*` dirs, no `trtllm_fmha_v2_*`).
2. **20-SM device makes the SM cap tiny** (draft 5, target 10): the padded grid is small in absolute terms, so "under-parallelization" would be visible as *tile* under-utilization, which the dynamic tables prevent — but it does mean per-CTA work at long ctx is large (4096-token kv chunks), and the persistent merge adds a second pass. If the instrumentation shows merge time dominating at c1, Option B's threshold should weight seq_len heavily.
3. **Autotune-vs-capture (forensics (e)) is still open:** `flashinfer_autotune()` runs during warmup *before* capture; if inductor picks different GEMM kernels under graph capture vs the autotuned eager path, that is a legitimate c1 regression source independent of FI attention.
4. `update_draft_decode_metadata` (`flashinfer.py:1924`) advances only the KV plan buffers (indptr/last_page_len/indices via Triton) — confirmed it does not touch grid/split-KV, so the draft's between-step metadata update is consistent with the per-step full `plan()` described above.

## 8. Source inventory

- Forensics: `codepath.md` §5 (hypotheses (a)-(e), §5(b) detail, evidence plan); `configs.md`, `scripts.md` (matrix runs) — internal forensics notes.
- `docker/vllm-thor/PATCHES.md` (patch inventory; dspark = thor-dspark).
- Patched vLLM tree: `v1/worker/gpu/spec_decode/dflash/{speculator.py,cudagraph.py}`, `v1/worker/gpu/{input_batch.py,cudagraph_utils.py,model_runner.py,dp_utils.py,model_states/default.py}`, `v1/attention/backends/flashinfer.py`, `model_executor/warmup/{kernel_warmup.py,flashinfer_autotune_cache.py}`.
- FlashInfer 0.6.18: `$FI/prefill.py` (plan 3443, auto-branch 3804-3892), `$FI/decode.py` (fast_decode_plan 3826-4040), `$FI/utils.py` (545 backend, 523 sm120 gate), `$FI/jit/attention/modules.py` (gen_fmha_v2_module 2082-2160), `$FI/jit/attention/fmha_v2/generator_utils.py` (465 — the quoted grid), `$FI/data/include/flashinfer/attention/scheduler.cuh` (820-900 materialize), `prefill.cuh:4335` (kernel grid), `$FI/data/csrc/batch_prefill.cu`.
- Materialized JIT cache ground truth: `~/.cache/flashinfer/0.6.18.post1/110a/generated/batch_prefill_with_kv_cache_*` (generated `batch_prefill.cu` = replay-time host code).

---

## The resulting patch (E)

### Overview

**Patch:** `docker/vllm-thor/patches/thor-draft-cg-gate-sm110.patch` (12th entry in `PATCH_ORDER` at the time of writing; the stack has since grown to 13 — see [`docker/vllm-thor/PATCHES.md`](../../docker/vllm-thor/PATCHES.md)).
**Base:** pristine vLLM `0.29.1rc1.dev347+gdee37d891`; the diff is relative to the 11-patch reference state, where the only file touched is byte-identical to pristine.
**Scope:** exactly one vLLM file plus the four wiring files. No capture-logic changes (`dflash/cudagraph.py` untouched), no 50885/49652/thor-dspark grant changes, no adjacent cleanups.

### Gate site

**Exact gate site (patched tree):** `speculator.py:557` — `need_eager = is_profile or _thor_draft_cg_gate_wants_eager(num_reqs)`, feeding `dispatch_cg_and_sync_dp(..., need_eager=need_eager, ...)` at `:583-594` (`need_eager=need_eager` at `:565`).

**Why this site, not the `cg_mode == FULL` branch.** The branch at `:583` is where "replay FULL graph vs run eager" is *executed*, but the descriptor (and its `num_tokens` padding + `cg_mode`) is chosen one call earlier. The only correct way to take the *existing* eager fallback is the existing `need_eager` channel of `dispatch_cg_and_sync_dp` (`v1/worker/gpu/dp_utils.py:213`, `need_eager` branch at `:269-275`): it returns the `CUDAGraphMode.NONE` descriptor *unpadded*, which the existing else-branch then runs eagerly (`_generate_draft` with `cudagraph_runtime_mode=NONE`). Routing at the branch instead would call the eager forward with the bucket-**padded** `num_tokens_padded` and a `FULL` runtime mode — a behavior bug (e.g. N=9 would eagerly process 70 padded tokens instead of 63). The `need_eager` channel is the same path the draft already uses for profile runs and for the no-CG-support fallback, so no new eager code was written.

**Capture interaction (49652).** `DFlashCudaGraphManager.capture` (`dflash/cudagraph.py`) and `CudaGraphManager._init_candidates` (the 49652 guard in `cudagraph_utils.py`) are untouched: every bucket is still captured; only the per-step dispatch decision changed. `is_profile` short-circuits the `or`, so the gate function never runs during profile/capture; capture buckets therefore remain a superset of what gets replayed.

### New module-level units

| Unit | Line | Role |
|---|---|---|
| `_THOR_DRAFT_CG_*` state vars (min_batch cache, capability cache, 4 one-shot log flags) | 35-46 | lazy/one-shot state |
| `_thor_draft_cg_use_cuda_graph(num_reqs, capability, min_batch) -> bool` | 49-62 | **pure** decision helper (capability passed in; no CUDA/torch state) |
| `_thor_draft_cg_min_batch() -> int` | 65-89 | lazy once-parse of the env knob |
| `_thor_draft_cg_capability() -> tuple[int,int]` | 92-97 | one read of `torch.cuda.get_device_capability()` |
| `_thor_draft_cg_gate_wants_eager(num_reqs) -> bool` | 100-150 | driver: evaluate, one-shot logging, try/except → CG on error |

Helper rules (as specified): `capability[0] != 11` → `True` (CG; structural no-op off sm_110); else `num_reqs >= min_batch`.

### Env knob

`VLLM_THOR_DRAFT_CG_MIN_BATCH` — int, **default 2, minimum 1** (`1` ⇒ CG for every batch size = pre-gate behavior, the A/B control). Parsed lazily once (first draft step); cached thereafter. Invalid values (non-integer or `< 1`) → default 2 with a **one-time** `logger.warning` ("THOR draft-CG gate: invalid VLLM_THOR_DRAFT_CG_MIN_BATCH %r, defaulting to 2"). Read via `os.environ` in the dflash speculator module (no `envs.py` change — the spec calls for lazy once-parse, which `vllm/envs.py` eager loading would not give).

### THOR log lines (all `logger.info`, one-shot per kind via module flags)

| When | Line |
|---|---|
| first dispatch evaluation, sm_110 | `THOR draft-CG gate: active (sm_110, min_batch=<t>)` |
| first dispatch evaluation, other arch | `THOR draft-CG gate: inactive (arch <maj>.<min>)` |
| first eager dispatch | `THOR draft-CG gate: N=<n> -> eager (<n> < min_batch)` |
| first CG dispatch | `THOR draft-CG gate: N=<n> -> cudagraph` |
| any exception in gate evaluation (then never again) | `THOR draft-CG gate: error -> defaulting to cudagraph (<exc>)` |

Error semantics: on the first evaluation error the gate logs once and permanently degrades to pre-gate (CG) behavior — serving can never break (e.g. a future refactor moving the call into a context without a device).

### Sentinel

`THOR_DRAFT_CG_GATE` — 3 occurrences in the patched file: helper docstring (:52), driver docstring (:101), dispatch-site comment (:552). The spec requires the helper docstring + gate-site comment; the driver docstring is the same literal serving the driver (kept identical for grep-ability).

### Diffstat

```
vllm/v1/worker/gpu/spec_decode/dflash/speculator.py | 125 ++++++++++++++-
  4 hunks, +124 / -1   (the -1: need_eager=is_profile → need_eager=need_eager)
```

### Wiring changes (mirrors the A–D pattern)

- **`apply_patches.py`** — `PATCH_ORDER`: `thor-draft-cg-gate-sm110.patch` appended as the **12th** entry (after `thor-dspark` and after `thor-fa4-fp8kv`); comment explains it must come after thor-dspark (it edits the code the dspark patch's ecosystem introduced). Note: the only file it touches (`dflash/speculator.py`) is touched by no other patch, so its position is mechanically safe; the ordering is contractual.
- **`verify_patches.py`** — new FIXES entry `"thor sm110 draft-CG gate: shape-gated draft cudagraph replay (c1 fix)"` with **two** sentinels (52244 lesson — "used, not just defined"): 1. `def _thor_draft_cg_use_cuda_graph` (the helper, sentinel in docstring), 2. `need_eager = is_profile or _thor_draft_cg_gate_wants_eager(num_reqs)` (the actual dispatch call site — a partial apply landing only the helper would leave the c1 regression in place and now fails the build).
- **`PATCHES.md`** — header counts updated (Eleven→Twelve total; six→seven Thor-authored); new row in the main table; new bullet in "Files touched"; new section **"Thor sm_110 draft-CG shape gate (2026-09-19)"** (purpose = c1 fix, files, env knob, all four THOR log lines, sentinel, status "probe-gated, inert on non-sm_110, default N=1→eager"); canary paragraph now notes T10 runs unconditionally without a GPU.
- **`kernel_test_sm110_gates.py`** — new **T10** section: imports `_thor_draft_cg_use_cuda_graph` from the installed (patched) vllm — same in-function import mechanism as T6–T9; the helper takes `capability` as an argument, so the import chain needs no CUDA. Asserts the four spec cases: `(11,x)+N=1+min=2 → False`; `(11,x)+N=2+min=2 → True`; `(9,0)+N=1+min=2 → True` (arch no-op); `(11,x)+N=1+min=1 → True` (knob). `main()` reworked: without a CUDA device, T6–T9 print SKIP (as before) and T10 still executes; exit 0 iff T10 passes (plus T6–T9 pass when run with a GPU).

### Verification results (no GPU available — no-GPU gate only)

Environment: host box (no torch for the host python); canary run inside `vllm/vllm-openai:nightly-aarch64` (the 0.29.1rc1.dev347 base image), no GPU (`CUDA_VISIBLE_DEVICES=` → `torch.cuda.is_available() == False`), patched tree on `PYTHONPATH` (so `import vllm` resolves to the 12-patch tree).

| Check | Result |
|---|---|
| Fresh apply: copy pristine vLLM to a clean tree → `python3 apply_patches.py` | **12/12 applied, zero .rej** — last line `[thor-draft-cg-gate-sm110.patch] patching file vllm/v1/worker/gpu/spec_decode/dflash/speculator.py`; built-in gate: `all 12 fixes verified` |
| Standalone `verify_patches.py` | `all 12 fixes present` (rc 0) |
| Patch applies to the 11-patch reference state | `patch -p1 --forward` → clean; resulting `speculator.py` byte-identical to the full-stack tree's |
| Canary T1–T10 (`kernel_test_sm110_gates.py`, no GPU) | T6 GDN prefill **SKIP**, T7 B12x NVFP4 **SKIP**, T8 XQA **SKIP**, T9 FA4 FP8-KV **SKIP** (no CUDA device — the green no-GPU state), **T10 PASS (4/4 cases)**, `kernel test sm110 gates: DONE (exit 0)` |
| Canary negative control (same run, `PYTHONPATH` = tree without the 12th patch) | **FAIL T10** (`ImportError: cannot import name '_thor_draft_cg_use_cuda_graph'`), exit 1 — the canary has teeth |
| `compile()` (python 3.12, in image) on every touched file | 5/5 OK: patched speculator (both trees), `apply_patches.py`, `verify_patches.py`, `kernel_test_sm110_gates.py` |
| Tree delta check | clean tree vs 11-patch state: only `dflash/speculator.py` differs (see caveat below) |

Verbatim canary output (patched tree, no GPU):

```
SKIP T6 GDN prefill: no CUDA device visible (re-run with --gpus all on a Thor)
SKIP T7 B12x NVFP4: no CUDA device visible (re-run with --gpus all on a Thor)
SKIP T8 XQA/TRTLLM decode: no CUDA device visible (re-run with --gpus all on a Thor)
SKIP T9 FA4 FP8-KV: no CUDA device visible (re-run with --gpus all on a Thor)
PASS T10 draft-CG gate: pure decision logic (4/4 cases)
kernel test sm110 gates: DONE (exit 0)
```

### A/B at serve time

The gate is **on by default** on sm_110 (N=1 → eager; N≥2 → CG). On the gated image + this 12-patch stack:

| Scenario | How |
|---|---|
| Default (c1 fix active) | no env set. Startup log shows `THOR draft-CG gate: active (sm_110, min_batch=2)`, then `THOR draft-CG gate: N=1 -> eager (1 < min_batch)` on the first decode step; N≥2 steps log `... N=<n> -> cudagraph` once |
| A/B control = pre-gate behavior (CG for all N) | `VLLM_THOR_DRAFT_CG_MIN_BATCH=1` → `active (sm_110, min_batch=1)`; every step replays the graph (the "graph" arm of the c1 matrix) |
| Widen/shrink the eager region | `VLLM_THOR_DRAFT_CG_MIN_BATCH=3` (N≤2 eager) etc.; invalid values warn once and fall back to 2 |
| Non-Thor box | gate is a structural no-op: first step logs `THOR draft-CG gate: inactive (arch <maj>.<min>)`, every step replays as pre-gate |

So the c1 A/B is: run the c1 workload on the same image with `VLLM_THOR_DRAFT_CG_MIN_BATCH=1` (regression reproduced, pre-gate behavior) vs unset (gate active) — expect the 7–16 % c1 delta to close, N=2/N=4 unchanged (N=2 is already CG under the default; N=4 likewise).

### Caveats

- **DP > 1 (out of scope, noted):** the gate evaluates per rank on local N. At dp_size=1 (the Thor deployment) there is no cross-rank contract. At dp>1 the reuse path of `dispatch_cg_and_sync_dp` carries per-rank descriptors without a collective, so a small-N rank would run eager while a large-N rank replays — same property as the existing reuse path, but the per-rank `num_tokens_across_dp` would then diverge. The non-reuse path already has the "any rank eager ⇒ all eager" all-reduce rule. If dp>1 dspark ever ships, the gate must move to an agreed value; flagged here, not fixed (spec: minimal, no refactors).
- **Threshold is batch-size only** (per spec), not seq-len-weighted. Study §7.2 hints seq_len may matter (20-SM device); if the §6 confirm-first data (replay overhead vs in-graph GEMM config) says so, the pure helper is the single place to extend (e.g. an extra `seq_len` arg) — the canary T10 asserts its contract.
- **GPU-side verification pending** (no GPU available at authoring time): on a Thor with `--gpus all`, expect the four THOR log lines at startup + the c1 speedup; T6–T9 in the canary also re-run (INERT-OK/PASS as before — the gate does not touch any probe path).

### Known inconsistency in the reference tree (for the team)

The 11-patch reference tree used for verification was found to be missing the `thor-fa4-fp8kv-sm110` patch: `verify_patches.py` on it reported exactly two missing fixes — `thor sm110 FA4 FP8-KV` and (expected) this new 12th. Concretely, `fa_utils.py` had no `# THOR_SM110_FA4_PROBE` and `vllm_flash_attn/cute/interface.py:910` still asserted `arch // 10 == 10` (not `in (10, 11)`). **No effect on this patch** — its only target file is byte-identical in pristine / reference / full-stack trees, and the patch was verified against both — but any consumer citing that tree as the 11-patch ground truth will mis-cite the FA4 state; the reference tree should be rebuilt (pristine + all 11 patches) or relabeled as 10-patch.
