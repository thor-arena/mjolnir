# FA4 hd256 FP8 Descale — Upstream Status

> **Status:** Research complete — GO · **Date:** 2026-09-21 · **Scope:** upstream status of FP8 (descaled KV) support in the `head_dim=256` FA4 (CuTe-DSL) forward kernel, verified against `Dao-AILab/flash-attention` `main` at that date

Research target: porting FP8 (descaled KV) support into the `head_dim=256` FA4 (CuTe-DSL) forward kernel for Jetson Thor (`sm_110a`, CC 11.0, 20 SMs). Local tree: `vllm 0.30.0` (the pristine vLLM 0.30.0 package root, since the 2026-09-26 base bump). Raw fetched artifacts: `docs/fa4-hd256-fp8/raw/`.

---

## 2026-09-26 update — hd256 SplitKV merged upstream (#2916 + #2917)

- [PR #2916](https://github.com/Dao-AILab/flash-attention/pull/2916) "[CuTe, SM100] SplitKV for the hd256 2CTA forward kernel" + [PR #2917](https://github.com/Dao-AILab/flash-attention/pull/2917) "[CuTe, SM100] hd256 fwd: derive lengths and KV ranges from BlockInfo/SeqlenInfoQK" (self-described pure refactor; the base of the stack) — both merged by @drisspg on 2026-09-25, landing on `main` as a single squash commit `e9cf2c1`. Cross-linked into the #2456 tracker the same evening.
- **What changed.** `sm100_hd256_2cta_fmha_forward.py` now supports **static** SplitKV: the categorical `assert not is_split_kv` becomes `assert seqlen_k_per_split is None` (dynamic split metadata still rejected; the CLC scheduler is rejected); LSE partials are required; empty splits (zero-KV-block ranges from ceil-divided causal tiles) are skipped by every warp role via the new `BlockInfo.has_kv_work` predicate (softmax writes only `LSE = -inf`); the varlen scheduler folds the split into grid dim y. `interface.py` `_get_fwd_config` gains a CTA-counting heuristic gated `arch // 10 in [10, 11] and head_dim == head_dim_v == 256` — **sm_110 is explicitly in scope** (`get_num_sms_for_selection` reads the real device → 20 on Thor). Author's GB300 (152-SM) numbers: 2.0×–3.9× decode speedup at b=1, L=4k→128k (ns=1 within ±0.2% of pre-merge).
- **This is NOT the FP8-descale work.** `assert descale_tensors is None` is still on `main` today; the #2456 tracker (updated minutes after these merges) still lists FP8-for-hd256 as 🔨 "code complete, perf to be improved", private. This workstream's plan (wait for the upstream FP8 branch, then port per the AGENTS.md plan) is unchanged.
- **Impact on our sm_110 GEMV stack: none, by the arithmetic.** For Qwen3.8-27B decode (b=1, GQA 24:4, M≤8 → one 256-row m-block) the heuristic computes `total_mblocks = 2·1·4·6·1 = 48` clusters vs `num_SMs = 20` → `num_splits = min(20 // 48, 128, num_n_blocks) = 0` → **auto-SplitKV stays disabled on Thor** (the GPU is already CTA-oversubscribed: 48 CTAs > 20 SMs). This independently corroborates our in-tree NO-WIN measurement (`docs/fa4-hd256-fp8/real-splitkv-bench.md`, 2026-09-24) one day before it.
- **Base-bump consequence: zero fa4 patch rework.** The vLLM 0.30.0 vendored FA4 is **pre-#2916** (its `sm100_hd256_2cta_fmha_forward.py` still carries `assert not is_split_kv`; no `has_kv_work`, no CTA-count heuristic), so the fa4 patch group (hd256-fp8, 1cta-decode, gemv-decode) applied verbatim on the 0.30.0 bump — only two unrelated docstring-style hunks (55390/55519) needed re-adapting (`docker/vllm-thor/PATCHES.md`).
- **When this becomes relevant to us:** large-batch × long-context on Thor (or once the persistent-cluster scheduler lands). #2916 now provides a *maintained* kernel-side SplitKV implementation (empty-split handling, varlen scheduler, paged+split tests) that our inert A1a/A1b plumbing pre-duplicated — if we ever want 1CTA+splits, swap to the upstream machinery and re-run the gated A/B rather than trusting either side's earlier verdict a priori.

---

## TL;DR

- **The RFC (#2456) body is stale.** Three of the four v2 branch features it lists as "Ready — PR pending" have since merged into `main`; the persistent-cluster scheduler is still unmerged.
- **No public FP8-hd256 kernel exists.** No descale code in `main`, in any of the 8 candidate public fork branches, in any PR/commit/issue — the RFC's "code complete" FP8 work is private. The only public reference implementation is the *generic* kernel `flash_fwd_sm100.py`.
- **The hd256 kernel on `main` still rejects descales** (`assert descale_tensors is None`); TMA paged KV (page_size = tile_n = 128) and `seqused` *are* merged.
- **Silicon is not the blocker.** CC 11.0 is Blackwell-class (tcgen05 + tmem, 256 KB/SM), upstream already gates 2CTA hd256 features as `arch // 10 in [10, 11]` (PR #2590), and the in-house patch-D probe ran the FA4 2CTA kernel (full-tmem, FP8, paged/varlen/GQA) correctly on sm_110a.
- **What actually gates the port:** four policy asserts (FP8 `arch//10==10`, hd256 `descale_tensors is None`, vLLM "quantized KV cache dtype", `block_size % 128 != 0`), the absence of any public FP8-hd256 kernel to copy, and the production `block_size=16` vs 128 paged-cache mismatch.

---

## (a) RFC Dao-AILab/flash-attention #2456 — status table

[RFC #2456 "[RFC] FA4 — head_dim=256 & head_dim=512"](https://github.com/Dao-AILab/flash-attention/issues/2456) (open, created 2026-04-13 by @Johnsonms, assigned @Johnsonms + @tzadouri, last activity 2026-09-18). **Important: the RFC body is stale.** It still lists the v2 branches as "Ready — PR pending" although three of the four listed features have since been **merged into `main`** (verified per-PR on 2026-09-21).

| RFC item (Phase 1/2/3) | RFC body says | **Verified actual status (2026-09-21)** | Where |
|---|---|---|---|
| hd256 fwd+bwd 2CTA (SM100, bf16) | ✅ merged #2412 | ✅ Merged (Apr 2026, commit `27b4eb9`) + cleanup #2487 (`b21e204`) | [PR #2412](https://github.com/Dao-AILab/flash-attention/pull/2412), [PR #2487](https://github.com/Dao-AILab/flash-attention/pull/2487) |
| exp2 FMA emulation (softmax SFU→FMA) | 🚢 branch `exp2-emu-hd256-v2` | ✅ **MERGED** as [PR #2488](https://github.com/Dao-AILab/flash-attention/pull/2488) (2026-04-28, commit `b97ca5d`, +2.3% MHA / +1.5% GQA fwd; tuned to freq=14/res=6 ≈43% emulation) | in `main` |
| TMA paged KV (page_size = tile_n = 128) | 🚢 branch `paged-kv-hd256-v2` | ✅ **MERGED** as [PR #2489](https://github.com/Dao-AILab/flash-attention/pull/2489) (2026-05-01, commits `0b2c01b`+`fe44ca8`; paged overhead vs dense cut from +3.5% to +1.2%) | in `main` |
| `seqused_k` (+ `seqused_q`) & decoder paged-KV shape | 🚢 branch `seqused-k-hd256-v2` (stacked) | ✅ **MERGED** as [PR #2810](https://github.com/Dao-AILab/flash-attention/pull/2810) by @kzos (2026-09-11) — superset: seqused_q/k overrides at 4 work-tile sites + interface normalization of (max_seqlen_k, page_table) for the continuous-batching decoder shape. Validated in the `flash-attn-4==4.0.0b31` PyPI wheel ([torchtitan#4737](https://github.com/pytorch/torchtitan/pull/4737) cross-ref) | in `main` |
| Persistent + cluster-aware tile scheduler | 🚢 branch `persistent-cluster-hd256-v2` (stacked) | ❌ **NOT merged, no PR opened** (searched all PR titles; none found 2026-09-21). Branch still exists in [Johnsonms/flash-attention](https://github.com/Johnsonms/flash-attention) (`persistent-cluster-hd256-v2` @ `e911031`) | unmerged |
| varlen/packed support (hd256) | 📋 planned, "not yet branched" | ⚠️ Partially: #2810 dropped the hd256 skip from the varlen test matrix (seqused modes run) and rejects varlen-K-without-varlen-Q loudly; a full dedicated varlen mode is still planned | [PR #2810](https://github.com/Dao-AILab/flash-attention/pull/2810) |
| **FP8 input (FP8 tensor-core MMA, descaled KV)** | 🔨 "Code complete, perf to be improved", *not on a named branch* | ❌ **NO PUBLIC CODE EXISTS** (verified 2026-09-21): no descale code in any of the 8 candidate public branches of Johnsonms/flash-attention (all 0 descale mentions), no PR, no commit, no issue in Dao-AILab/flash-attention. The only public FP8-descale reference implementation is the **generic** kernel `flash_fwd_sm100.py` (see (c)). The "code complete" claim is unverifiable / private. | — |
| `pack_gqa` | 🔨 (notes in `AI/PACK_GQA_NOTES.md`) | ❌ Not on main (the `AI/` dir no longer contains that notes file; still no PR) | — |
| hd=512 (SM100) | 📋 | Open PR [#2877](https://github.com/Dao-AILab/flash-attention/pull/2877) (symmetric D512 fwd+bwd) | open |
| causal-left local attention (hd256) | — (post-RFC) | Open PR [#2749](https://github.com/Dao-AILab/flash-attention/pull/2749); composes with #2810 (trial merge clean per that PR's description) | open |
| hd256 backward + seqused_q/k | — | Open PR [#2891](https://github.com/Dao-AILab/flash-attention/pull/2891) (2026-09-16, cross-listed in #2456 comment 3); forward only for now — hd256 **backward still rejects seqused** | open |
| sm_110 arch gating | — | ✅ **MERGED** [PR #2590](https://github.com/Dao-AILab/flash-attention/pull/2590) (2026-05-25): `arch // 10 in [10, 11]` convention now covers the `use_dedicated_hd256_kernel` fwd+bwd gates; sm110 2CTA dQ-postprocess bug fixed here (absorbing [#2491](https://github.com/Dao-AILab/flash-attention/pull/2491)) | in `main` |

Issue thread (3 comments, read in full):

1. 2026-05-20: link to [issue #2576](https://github.com/Dao-AILab/flash-attention/issues/2576) (hd256 fwd perf analysis).
2. 2026-09-14: link to [PR #2810](https://github.com/Dao-AILab/flash-attention/pull/2810).
3. 2026-09-18: @sudhakarsingh27 pings @Johnsonms on [PR #2891](https://github.com/Dao-AILab/flash-attention/pull/2891) (seqused for hd256 backward).

sm_110 / CC 11.0 / Jetson mentions inside #2456: the RFC overview says it "targets Blackwell (SM100/SM110) GPUs using 2CTA instructions", and the merged [#2590](https://github.com/Dao-AILab/flash-attention/pull/2590) is the concrete arch-gate fix (see (e)). No GB10/Thor-specific discussion in the thread.

---

## (b) Exact upstream file state for the hd256 kernel (FP8)

**Fetched on 2026-09-21 from `main`:** [`flash_attn/cute/sm100_hd256_2cta_fmha_forward.py`](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/cute/sm100_hd256_2cta_fmha_forward.py) (local copy: `docs/fa4-hd256-fp8/raw/sm100_hd256_2cta_fmha_forward_main.py`, 1993 lines).

- **FP8 descale: STILL ABSENT.** The kernel signature takes the parameter and rejects it — lines 195/221-222:
  ```python
  descale_tensors: Optional[DescaleTensors] = None,
  ...
  assert descale_tensors is None, (
      "SM100 forward with head_dim=256 does not support descale_tensors")
  ```
  Same shape as the in-tree vendored kernel (local L191/L211) — the vendored kernel is only ~200 lines *ahead* of the upstream state here, in the same way.
- Paged KV **is** in main now (TMA only): L77-78 asserts `"SM100 hd256 2CTA supports TMA paged KV only (page_size must equal tile_n=128)"`; L330-335 consume rank-4 paged `(page_size, d, h_k, num_pages)` K / `(d, page_size, h_k, num_pages)` V with `mPageTable` + `max_seqlen_k`.
- seqused is in main: L910 `"# seqused overrides lengths, not cu_seqlens packing offsets."` (four `has_seqused` override sites, from #2810).
- Hard constraints still in effect: `score_mod`/`mask_mod`/aux/pack_gqa/SplitKV forbidden (L74-82); `q_subtile_factor == 1`, `kv_subtile_factor == 1` (L82-85); `m/n_block == 128`, `mma_tiler == (128,128,256)` (L88-102); `is_local` / `window_size_*` forbidden (L215-218, i.e. **no sliding-window until #2749 merges**); `learnable_sink` forbidden (L203); dedicated kernel forces `scheduler_metadata = None` in the interface (no dynamic-persistent scheduler).
- Structure (relevant to the arch question): `cluster_shape_mn = (2,1)` (L121), `tmem_alloc_cols = SM100_TMEM_CAPACITY_COLUMNS` = **512** (L136, constant at `tile_scheduler.py:2154`), `tmem_s_offset=0 / tmem_o_offset=256 / tmem_p_offset=0` (L154-156), `_tune_key = (True, is_causal, 256, False)  # always 2cta, no sm103 variant` (L158).

**Branch check (all public branches of the active fork):** `sm100_hd256_2cta_fmha_forward.py` fetched from 8 branches of [Johnsonms/flash-attention](https://github.com/Johnsonms/flash-attention) (`paged-kv-hd256-v2`, `exp2-emu-hd256-v2`, `persistent-cluster-hd256-v2`, `seqused-k-hd256-v2`, `tune-hd256-and-clock`, `bench-hd256-paged`, `feature/hd256-paged-tma`, `backup/main-before-sync-2026-09-16`) → **0 descale mentions in every branch**. Commit search `repo:Dao-AILab/flash-attention "fp8 hd256"` returns exactly one commit: the #2590 arch-gating commit. → **The FP8 hd256 kernel exists only privately (if at all).**

**vLLM side (vllm-project/vllm, checked 2026-09-21):**

- No PR/issue in vLLM ports FP8/descale into the hd256 kernel. Related vLLM work:
  - [#52050](https://github.com/vllm-project/vllm/pull/52050) (merged 2026-08-16) — *temporarily disabled* FA4 hd256 on Blackwell (seqused rejection broke decoders).
  - [#52980](https://github.com/vllm-project/vllm/pull/52980) (merged 2026-08-26, @simon-veitner-redhat) — **re-enabled** FA4 hd256: bumps the vendored FA4, and the backend now *advertises/forces KV block size 128 for hd256* (`FA4_HD256_PAGE_SIZE = 128` in `fa_utils.py`; `get_supported_kernel_block_sizes` returns [128]; a user-pinned `--block-size` not a multiple of 128 makes FLASH_ATTN ineligible for hd256).
  - [#55366](https://github.com/vllm-project/vllm/pull/55366) (open) — include SM110 in FA4 **auto-selection** (currently `device_capability.major == 10` only); explicitly notes hd256 stays blocked on sm_110 by the #52050 guard at that time.
  - [#51363](https://github.com/vllm-project/vllm/pull/51363) (merged 2026-08-11) — "Forward per-head FP8 descales through FA4" (generic kernel path; this plumbing is already in the in-tree tree, incl. the `fp8_kv_dequant` identity-q_descale helper at `interface.py:891-898`).
  - [#54705](https://github.com/vllm-project/vllm/pull/54705) (open) — "Inkling: fp8 e4m3 KV cache with per-tensor scales on SM100" (not hd256-specific).
  - [Gemma-4 FA4 FP8 kernel #48666](https://github.com/vllm-project/vllm/pull/48666) was merged 8/14 and **reverted** 8/19 ([#52987](https://github.com/vllm-project/vllm/pull/52987)); resubmission [#53175](https://github.com/vllm-project/vllm/pull/53175) is open (head_dim=512 backend selection).
- Current vLLM main [`v1/attention/backends/fa_utils.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/fa_utils.py): `_fa4_hd256_fallback_reason()` still returns `"quantized KV cache dtype {dtype}"` for FP8 KV, **and** `"a KV cache block size of {bs}"` when `block_size % 128 != 0` (the in-tree copy is identical — `FA4_HD256_PAGE_SIZE = 128` at L14).
- The vLLM backend already routes **sm_110** into the hd256 dispatch: [`flash_attn.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/flash_attn.py) L172-175: `major in (10, 11) and uses_fa4_hd256_kernel(head_dim) and page_size == FA4_HD256_PAGE_SIZE`.
- Note: in current vLLM main the FA4 CuTe sources are **not tracked in the repo** (`vllm/vllm_flash_attn/cute/` absent from git; the wheel ships them resolved from the flash-attention source tree / `VLLM_FLASH_ATTN_SRC_DIR` symlink mode — see `vllm/vllm_flash_attn/__init__.py` symlink handling and setup.py's `flash_attn_files_to_skip`). The in-tree dev tree still has them vendored in-tree (divergent — see (c) note and Risks).

---

## (c) The descale plumbing to transcribe

Reference: upstream `main` [`flash_attn/cute/flash_fwd_sm100.py`](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/cute/flash_fwd_sm100.py) (`docs/fa4-hd256-fp8/raw/flash_fwd_sm100_main.py`) + [`interface.py`](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/cute/interface.py) (`docs/fa4-hd256-fp8/raw/interface_main.py`). **The same plumbing already exists in the in-tree `vllm_flash_attn/cute/flash_fwd_sm100.py`** (class at L127, loader at L2011-2027, application sites L2187-2196 / L2626-2628 / L2759) — so transcription is against code the tree already owns.

**1. Representation (per-batch/per-KV-head scalar — NOT block-wise NVFP4 style):**

- `q_descale / k_descale / v_descale`: optional `torch.float32` tensors, shape `(batch_size, num_head_kv)`, validated in `interface.py` L764-772 (`_validate_tensor(..., (batch_size, num_head_kv), torch.float32, ...)`). Semantics: FA3 "descale" convention — QK descale = `q_descale[b,h]*k_descale[b,h]`, V descale = `v_descale[b,h]` (comment at `flash_fwd_sm100.py` L2004: "Map query-head tile index -> KV-head index (FA3 descale semantics)").
- Kernel-side container: `DescaleTensors` NamedTuple (`flash_fwd_sm100.py` L132-138).
- FP8 inputs only: `interface.py` L775-776 asserts descales are `None` for non-FP8; L781: `assert arch // 10 == 10, "FP8 is only supported on SM100 (compute capability 10.x) for FA4 CuTe."` (**the arch gate that must be widened for sm_110** — #2590 explicitly left it untouched as "deliberate").
- vLLM addition (in-tree only): `fp8_kv_dequant` mode materializes an identity `q_descale` for bf16-Q runs (`interface.py:891-898`).

**2. Where the descale is applied (two folds, both in the existing online-softmax numerics — no new pass, no separate rescale):**

- QK side (softmax warp), `flash_fwd_sm100.py` L2186-2196: `qk_descale, _ = self._load_effective_descales(...)` then `softmax_scale_log2_eff = softmax_scale_log2 * qk_descale` (i.e. the descale is folded into the exp2-domain softmax scale; the `score_mod` variant folds into `softmax_scale_eff` instead).
- PV side (correction/epilogue warps), L2620-2622 + L2753-2759: `scale = cute.arch.rcp_approx(row_sum) ...; scale = scale * v_descale`.
- `_load_effective_descales` (L2010-2026): each descale tensor guarded by `cutlass.const_expr` → **compile-time specialization**: presence of each of q/k/v descale is part of the kernel variant (compile key L1181-1183: `q_descale is not None`, etc.).
- FP8-specific numerics that travel with it: `max_offset = 8` when `q_dtype.width == 8` (L2190, L2624), `max_offset_scale = 256.0` (L2626), `rescale_threshold = 8.0 if 16-bit else 0.0` (L2193) — keeps P inside e4m3 range for the FP8 PV MMA. FP8 Q/K/V go in as `uint8` views on torch<2.11 (`interface.py` L1513-1514).

**3. What that means for the hd256 port — two viable routes:**

- **Route A (true FP8 MMA, matches upstream RFC scope):** the hd256 kernel must also switch its QK and PV MMA operands to FP8 (today `mma_tiler[2]==256` bf16), add P→e4m3 conversion in the softmax warp, adopt `max_offset=8` / `rescale_threshold`, and take the two descale folds. This is the work upstream calls "code complete, perf to be improved" — and it is **not public**, so it would have to be written by mirroring the generic kernel's FP8 branch. Estimate: ~300-600 LOC kernel changes + ~150 LOC interface/vLLM plumbing + tests, plus numerical/perf tuning (exp2-emu knobs are B200-tuned).
- **Route B (dequant-outside-kernel, vLLM's existing `fp8_kv_dequant` pattern):** dequantize fp8 KV→bf16 before the (unchanged) bf16 hd256 kernel. Zero kernel descale work; cost = dequant bandwidth in the KV load path on a 20-SM GPU, and it forfeits FP8 tensor-core MMA throughput (Q would also need dequantizing, since tcgen05 MMAs can't mix bf16×fp8 operands). The in-kernel descale plumbing itself (if Q stays FP8 and KV is dequantized, or for completeness) is only ~60-100 LOC: parameter + assert removal, the `_load_effective_descales` helper (17 LOC, verbatim copy), 2 fold sites, interface compile-key/arg plumbing (~20 LOC).

**Divergence warning:** the in-tree `sm100_hd256_2cta_fmha_forward.py` (2204 lines) has a **different structure** from upstream main (1993 lines) — the in-tree copy uses `is_varlen_b1` / `l2_swizzle` / `mask_residual` / `use_2cta=True` params and a different tile-scheduler import (`TileSchedulerArguments`), vs upstream's `is_varlen_q` / `use_clc_scheduler` / `q_stage` / `is_static_persistent` (~1400 diff lines). Any transcription must target the **local** kernel's warp structure; the generic-kernel descale code (identical in both trees) is the safe source.

---

## (d) Paged-KV / page-size status for hd256

- **TMA paged KV is MERGED** (#2489) and **page_size == tile_n == 128 is a hard constraint** (kernel assert, upstream main L77-78; interface requires `page_table` width ≥ `ceil(max_seqlen_k/page_size)` and, since #2810, rounds `max_seqlen_k` up to the page and narrows the table, with `seqused_k` masking the rounded tail). **No non-TMA paged fallback exists for hd256** — `paged_kv_non_tma` is asserted off (the generic kernel has a non-TMA paged path; the hd256 kernel does not).
- Supported page sizes for hd256: **128 only** (16/64 → not supported; that's why vLLM's `_fa4_hd256_fallback_reason` rejects `block_size % 128 != 0`).
- vLLM policy (post-#52980, in current main and in the in-tree tree): for an hd256 model on FA4, the backend **auto-selects `kv_cache_block_size = 128`** (`_get_fa4_hd256_block_size` → `FA4_HD256_PAGE_SIZE = 128`); a user-pinned `--block-size` not a multiple of 128 makes FLASH_ATTN ineligible for hd256.
- **Consequence: a production `block_size=16` + fp8_e4m3 KV cache cannot directly feed the hd256 TMA path.** Options: (i) switch the model's KV cache to block_size=128 (vLLM will do it automatically once FA4 hd256 is chosen — but see the FP8 fallback reasons in (a)/(e)); (ii) port a non-TMA paged path (big kernel work); (iii) dense/contiguous KV (impractical for decode).

---

## (e) sm_110 arch-compatibility assessment — 2CTA / tmem / clusters on CC 11.0

**Verdict: hardware supports everything the hd256 kernel needs. The blockers are software gates, not silicon.** Decisive evidence chain:

1. **CC 11.0 is Blackwell-class silicon with 5th-gen tensor cores + tmem.**
   - Thor spec writeup (2026-03): "Jetson AGX Thor … GPU Architecture: **Blackwell SM 11.0 (20 SMs, tcgen05 + tmem)**" ([ajeetraina.com comparison](https://www.ajeetraina.com/nvidia-dgx-spark-vs-jetson-agx-thor-which-one-should-you-buy/)).
   - llama.cpp runtime output on-device: "NVIDIA Thor, compute capability 11.0" + "Blackwell GPU, **5th-gen Tensor Cores**" ([ggml-org/llama.cpp discussion #16578](https://github.com/ggml-org/llama.cpp/discussions/16578)).
   - NVIDIA developer forum (GB10/SM121 thread, citing SM100/110): "On SM100/110, the CUDA cores don't touch the **256KB** [tmem] allocated per Tensor core" ([NVIDIA forum 357663](https://forums.developer.nvidia.com/t/dgx-spark-sm121-software-support-is-severely-lacking-official-roadmap-needed/357663)) — i.e. SM110 tmem = 256 KB/SM, same budget as SM100's 512×128-bit tmem columns.
2. **Upstream flash-attention treats 11.x as 2CTA-capable Blackwell family.** Merged [PR #2590](https://github.com/Dao-AILab/flash-attention/pull/2590) (commit `59cf537`, 2026-05-25): the codebase convention is `arch // 10 in [10, 11]` for Blackwell-family 2CTA features; #2590 fixed the last three `== 10`-only gates, including **both `use_dedicated_hd256_kernel` sites** (interface L579 fwd, L1335 bwd). Its commit message states the fix absorbs [#2491 "[CuTe,Sm110] Fix sm110 2cta dQ postprocess"](https://github.com/Dao-AILab/flash-attention/pull/2491) — i.e. the 2CTA kernels **actually ran on sm_110** and only a postprocess bug (software) needed fixing.
3. **Empirical (this workstream, on the Thor):** the FA4 2CTA kernel — the same family, allocating the full tmem via `get_max_tmem_alloc_cols("sm_100")` — compiles via CuTe-DSL JIT and runs correctly on `sm_110a` with full FP8 (dense + paged + varlen + GQA + causal), matching the fp32 reference within FP8 quantization error ([docs/thor-stack/fa4-fp8kv-sm110.md](../thor-stack/fa4-fp8kv-sm110.md) §1-2; probe result recorded with the [thor-fa4-fp8kv-sm110](../../docker/vllm-thor/patches/thor-fa4-fp8kv-sm110.patch) patch: `maxerr=0.01965` at hd128, page 16). This exercises tcgen05 2-CTA MMA, tmem alloc/dealloc, clusters, and mbarrier pipelines on sm_110a — the exact hardware features the hd256 kernel uses (`cluster_shape_mn=(2,1)`, `tmem_alloc_cols=512`, `tcgen05.OperandSource.TMEM`).
4. **vLLM already routes sm_110 into the hd256 dispatch path** (backend `major in (10, 11)` in [flash_attn.py](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/flash_attn.py) L172-175), and [PR #55366](https://github.com/vllm-project/vllm/pull/55366) documents that "the vendored FA4 interface validates and dispatches compute capability 11 throughout".
5. **Remaining software gates for the FP8 hd256 port on sm_110** (all policy asserts, none capability checks): (i) `interface.py` L781 `assert arch // 10 == 10` for FP8 (upstream) / L910 (in-tree copy); (ii) the hd256 kernel's `descale_tensors is None` assert; (iii) vLLM `_fa4_hd256_fallback_reason` "quantized KV cache dtype" + `block_size % 128`; (iv) FA4 auto-selection `major == 10` only (until #55366 lands). The patch-D probe already proved the pattern: widen (i) with a fam(110) term and probe-gate it.

**tmem sizing nuance:** the hd256 kernel wants the *full* 512 tmem columns (S@0, O@256, P sharing S). The generic kernel also allocates the full 512 on Thor and works — so sm_110 tmem capacity is demonstrated adequate for the worst-case FA4 allocation. (If Thor's per-SM tmem were smaller, the generic kernel would have failed first.)

---

## (f) Open risks

1. **No public FP8-hd256 kernel to borrow from.** The RFC's "code complete" FP8 work is private; expect to implement Route A ourselves (~300-600 LOC kernel + ~150 LOC plumbing) by mirroring `flash_fwd_sm100.py`'s FP8 branch into the *divergent in-tree* hd256 kernel. Or take Route B (dequant KV→bf16 outside the kernel, vLLM's existing `fp8_kv_dequant` pattern) at a perf cost.
2. **block_size=128 forced for hd256 TMA paged.** The production cache is `block_size=16` + fp8_e4m3. Either re-block the model to 128 (vLLM auto-selects it; KV memory layout change) or invest in a non-TMA paged path (none exists for hd256; the generic kernel's `paged_kv_non_tma` would need a hd256 port — large).
3. **Numerical risk (Route A):** FP8 P-quantization in a 256-wide tile with the exp2-emulation knobs tuned on B200 (freq=14/res=6); `max_offset=8`/e4m3 range margins may need Thor re-tuning. Upstream itself lists open FP8 numerical issues for the generic kernel ([#2577](https://github.com/Dao-AILab/flash-attention/issues/2577) precision at KV tile boundaries; [#2694](https://github.com/Dao-AILab/flash-attention/pull/2694)/[#2713](https://github.com/Dao-AILab/flash-attention/pull/2713) P-saturation fixes) — expect the same class of issues in the hd256 tile.
4. **20-SM scheduler behavior.** The unmerged persistent-cluster scheduler work (RFC: +10-25% at small batch) targets exactly the 20-SM regime; until it lands, hd256 fwd perf on Thor may be scheduler-bound (cf. [#2576](https://github.com/Dao-AILab/flash-attention/issues/2576): hd256 already trails hd128 by up to 28% on B200 at 4K due to tile scheduling).
5. **Model-feature coverage gaps in the dedicated hd256 kernel:** no sliding-window (`window_size_*`/`is_local` — blocked until [#2749](https://github.com/Dao-AILab/flash-attention/pull/2749) merges), no score_mod/softcap, no learnable_sink, no pack_gqa, no SplitKV — all already fallback triggers in vLLM's `_fa4_hd256_fallback_reason`. Qwen3.8-27B's 16 full-attention layers must be plain causal for this to apply.
6. **vLLM auto-selection won't pick FA4 on sm_110 by default** (major==10 gate; [#55366](https://github.com/vllm-project/vllm/pull/55366) open). The serve config keeps forcing `flash_attn_version=4` (as the patch-D probe does) or backports #55366.
7. **Kernel-tree divergence:** the in-tree vendored hd256 kernel ≠ upstream main (different params/scheduler). Upstream PRs (#2810 etc.) won't cherry-pick cleanly into the in-tree tree; port by feature, not by diff.
8. **FP8 KV on GDN hybrids** has an open vLLM RFC questioning whether the memory win is real for this exact model class ([vllm#55196](https://github.com/vllm-project/vllm/issues/55196), 2026-09-03) — worth reading before investing in Route A.

---

## Methodology

- **Structured GitHub reads (API-level):** full body + all 3 comments of [Dao-AILab/flash-attention#2456](https://github.com/Dao-AILab/flash-attention/issues/2456); status sweeps over Dao-AILab/flash-attention (PRs #2488/#2489/#2810/#2891/#2590/#2491, "fp8" titles, "fp8 hd256" commits, `AI/` dir, Johnsonms fork branches) and vllm-project/vllm (PRs #52050/#52980/#55366/#51363/#54705/#53175, "hd256" titles, commit history of `vllm/vllm_flash_attn`), via PR/issue/commit search and branch/file listing.
- **Pages read in full:** PR #2810, #2489, #2488, #2590, #2491 (flash-attention); PR #52050, #52980, #55366 (vllm); [ajeetraina Thor-vs-Spark comparison](https://www.ajeetraina.com/nvidia-dgx-spark-vs-jetson-agx-thor-which-one-should-you-buy/) (SM110 tcgen05+tmem claim). NVIDIA forum 357663 p2 (SM110 tmem 256KB claim) — the full-text fetch returned partial content; the quote is cited from the search snippet, treated as **medium confidence**.
- **Raw file fetches (saved under `docs/fa4-hd256-fp8/raw/`):** upstream `main` `sm100_hd256_2cta_fmha_forward.py`, `flash_fwd_sm100.py`, `interface.py`, `tile_scheduler.py` (grep); 8 Johnsonms-fork branch copies of the hd256 kernel (descale-grepped, all 0); vLLM `main` `fa_utils.py`, `flash_attn.py`, `flash_attn_interface.py`, `vllm_flash_attn/__init__.py`, setup.py greps.
- **In-tree artifacts re-verified:** the pristine vLLM 0.29.1 package root (descale assert L211, `FA4_HD256_PAGE_SIZE` L14, FP8 arch assert L910, `fp8_kv_dequant` L891), plus the workstream doc `docs/thor-stack/fa4-fp8kv-sm110.md` and the patch probe `docker/vllm-thor/patches/thor-fa4-fp8kv-sm110.patch` (patch D).
- **Skipped/failed:** the NVIDIA forum p2 full-text fetch returned empty (fell back to snippet citation); filename-restricted code search is not available via the search API (worked around with per-branch raw fetches); no arXiv (out of scope).

## GO / NO-GO

**GO — the sm_110 port is feasible; silicon is not the blocker.** The decisive evidence: (1) CC 11.0 is documented as Blackwell "SM 11.0 (20 SMs, **tcgen05+tmem**)" with 256 KB tmem/SM (same class as SM100); (2) upstream flash-attention already ships `arch // 10 in [10, 11]` 2CTA gating incl. both `use_dedicated_hd256_kernel` sites (merged [#2590](https://github.com/Dao-AILab/flash-attention/pull/2590)), and the sm110-specific 2CTA bug ([#2491](https://github.com/Dao-AILab/flash-attention/pull/2491)) was a software postprocess fix, not a hardware gap; (3) the in-house patch-D probe already ran the FA4 2CTA kernel (full-tmem, FP8, paged/varlen/GQA) correctly on sm_110a. What actually gates the port is four policy asserts (FP8 `arch//10==10`, hd256 `descale_tensors is None`, vLLM "quantized KV cache dtype", `block_size%128!=0`) plus the **absence of any public FP8-hd256 kernel to copy** (the RFC's "code complete" work is not on any public branch/PR/commit) and the **block_size=16 vs 128** paged-cache mismatch.
