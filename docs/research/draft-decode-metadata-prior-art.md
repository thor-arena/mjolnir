# vLLM Prior-Art Survey — Fused Draft-Decode Metadata Advancement & the DSpark/dflash Path

> **Status:** Complete research summary · **Date:** 2026-09-09 · **Scope:** Upstream survey of `vllm-project/vllm` (main @ `3fb676bfad0f1c7099af6296983e739be3fe29cc`) to determine whether extending the FlashInfer in-place decode-plan advance (`update_draft_decode_metadata`) to the DSpark/dflash non-causal draft path is net-new work or already upstream.

## TL;DR

- **PR #46849** (merged 2026-08-11) is the upstream origin of the fused multi-step draft-decode infrastructure: the `supports_draft_decode_metadata_update` flag, the `update_draft_decode_metadata()` hook, and the `_fused_multi_step_decode` path in the autoregressive speculator. Measured there: −58.1 % multi-step draft-decode CPU span and −49.8 % complete `propose` CPU span (H100 ×8, DeepSeek-V4-Flash MTP-3, Qwen3.5-9B MTP-7).
- **FlashInfer does not opt in upstream** — its native decode path keeps `supports_draft_decode_metadata_update = False` and still rebuilds metadata per step.
- FlashInfer's structural limitation explains the non-participation: its native decode path caps at `AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE` (issue #49547), below the `UNIFORM_BATCH` level required for FULL decode graphs under speculative decoding.
- **DSpark/DSpark draft attention is non-causal**, and FlashInfer rejects non-causal attention entirely (issue #41559) — so the DSpark draft never runs on FlashInfer (dense targets use `FLASH_ATTN`).
- The dflash speculator does **not** consume `supports_draft_decode_metadata_update` at all; its CUDA-graph capture is gated on its own `attn_cg_support` / `UNIFORM_BATCH` mechanism.
- **Conclusion: extending the in-place plan advance to the dflash/DSpark non-causal draft path is definitively net-new work** — different backend, different consumer, different (non-causal) geometry, and a separate capture boundary.

## Findings

### Finding 1 — Upstream PR #46849 introduces the core infrastructure

**PR #46849** — "[MRV2][Spec] Fuse AR speculator multi-step decodes back into one CUDA graph" (opened 2026-06-26, merged 2026-08-11 by `yiz-liu`) is the upstream origin of `supports_draft_decode_metadata_update` and `update_draft_decode_metadata()`.

- The PR defines `AttentionMetadataBuilder.supports_draft_decode_metadata_update: bool = False` (default disabled) and `update_draft_decode_metadata(metadata)` (default `raise NotImplementedError`) in `v1/attention/backend.py`.
- The autoregressive speculator (`v1/worker/gpu/spec_decode/autoregressive/speculator.py`) gained `use_fused_multi_step_decode` and chooses between `_multi_step_decode` (per-step rebuild) and `_fused_multi_step_decode` (in-place update + single graph) based on backend capability.
- When all attention groups declare `supports_draft_decode_metadata_update = False`, the speculator logs: *"Fused multi-step draft decode is not supported by attention backend(s) %s; falling back to rebuilding attention metadata"* — exactly the log observed in the local deployment.
- Tested on H100 ×8 with DeepSeek-V4-Flash MTP-3 and Qwen3.5-9B MTP-7: −58.1 % multi-step draft-decode CPU span and −49.8 % complete `propose` CPU span in the fused path.

**Source:** [PR #46849](https://github.com/vllm-project/vllm/pull/46849) · [vLLM Daily Digest 2026-08-11](https://github.com/vllm-project/vllm-daily/blob/main/2026/08/2026-08-11.md#21-mrv2spec-fuse-ar-speculator-multi-step-decodes-back-into-one-cuda-graph)
**Confidence:** HIGH — full PR description and merged PR page read.

### Finding 2 — FlashInfer does not opt in to the fused path upstream

Across the code search, `flashinfer.py` does **not** set `supports_draft_decode_metadata_update = True`; the default `False` persists. Consequently, upstream FlashInfer on the native decode path still falls back to per-step rebuild.

By contrast, the following backends **do** opt in:

| Backend | `supports_draft_decode_metadata_update` | Handling |
|---------|----------------------------------------|----------|
| FlashAttention / FA3 (no DCP) | `True` (when `dcp_world_size == 1`) | Regenerates FA3 scheduler metadata into persistent storage |
| DeepSeek V4 sparse SWA | `True` (no DCP) | Recomputes SWA lengths/indices, refreshes tile schedulers |
| Triton Attention | `True` | Step-dependent fields already reference persistent buffers |
| Triton MLA (no DCP) | `True` | No additional materialized metadata to update |
| **FlashInfer (FI native)** | **`False` (unchanged)** | Falls back to per-step rebuild |
| DeepSeek V4 ROCm sparse SWA | `False` | Ragged SWA indices still require adaptation |

**Source:** [Code search: `supports_draft_decode_metadata_update vllm-project/vllm`](https://github.com/search?q=repo%3Avllm-project%2Fvllm+supports_draft_decode_metadata_update&type=code) · [PR #46849 backend table](https://github.com/vllm-project/vllm/pull/46849)
**Confidence:** HIGH — merged PR description's backend-handling table read and verified via code search.

### Finding 3 — FlashInfer has a structural limitation for FULL CUDA graphs under spec-decode

**ISSUE #49547** — "FlashInfer + spec-decode silently downgrades to PIECEWISE cudagraphs (−16 % measured)" documents that FlashInfer's native decode path caps at `AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE`, below the `UNIFORM_BATCH` level required to keep FULL decode graphs under speculative decoding. The resolver silently downgrades `cudagraph_mode` from `FULL_AND_PIECEWISE` to `PIECEWISE`, costing ~16 % throughput on bandwidth-bound hardware (GB10/sm_121).

This structural limitation explains **why** FlashInfer does not participate in the fused multi-step path upstream: the FI native decode wrapper is built at `UNIFORM_SINGLE_TOKEN_DECODE` resolution, whereas the fused path requires `UNIFORM_BATCH`-level capture.

**Source:** [Issue #49547](https://github.com/vllm-project/vllm/issues/49547)
**Confidence:** HIGH — full issue body read with measured benchmark tables.

### Finding 4 — DSpark uses non-causal sliding-window attention via SparseMLA

**PR #46995** — "[Spec Decode] DSpark" (merged early July 2026, benhislett/NVIDIA) establishes that DSpark uses non-causal sliding-window attention. The PR description states:

> "DSpark uses non-causal sliding-window attention. To implement this, instead of manually reimplementing the MLA attention, we instead utilize the existing SparseMLA backends with an expanded topk size…"

For **dense targets** (the Qwen3.8 deployment), the draft attention runs on `FLASH_ATTN` (not FlashInfer). The PR example command explicitly sets `"attention_backend":"FLASH_ATTN"`.

**Source:** [PR #46995](https://github.com/vllm-project/vllm/pull/46995)
**Confidence:** HIGH — PR #46995 body read and corroborated with DSpark speculator docs.

### Finding 5 — DFlash requires non-causal attention; FLASH_INFER rejects it entirely

**ISSUE #41559** — "DFlash speculative decoding fundamentally incompatible with all KV cache quantization" maps the compatibility matrix on v0.20.0:

| Backend | Non-causal support |
|---------|-------------------|
| FLASH_ATTN | Yes |
| **FLASHINFER** | **No (rejects non-causal entirely)** |
| TRITON | No |
| FLEX_ATTENTION | Yes (but rejects FP8 KV) |

Both DFlash and DSpark mandate `causal=False` for their draft cross-attention. FlashInfer rejects this outright. Therefore, DFlash/DSpark draft attention **never** runs on FlashInfer — it uses FLASH_ATTN or FLEX_ATTENTION.

**Source:** [Issue #41559](https://github.com/vllm-project/vllm/issues/41559)
**Confidence:** HIGH — full issue body read with verified compatibility matrix.

### Finding 6 — The dflash speculator uses `attn_cg_support` / `UNIFORM_BATCH`, independent of `supports_draft_decode_metadata_update`

The dflash speculator (`v1/worker/gpu/spec_decode/dflash/speculator.py`) has its own CUDA-graph capture mechanism governed by `attn_cg_support` (checked via `UNIFORM_BATCH`). The warning *"does not support full CUDA graphs; running the draft eagerly"* is emitted from `dflash/speculator.py:126` and is gated on `attn_cg_support`, **not** on `supports_draft_decode_metadata_update`.

The autoregressive speculator (`autoregressive/speculator.py:115`) is the sole consumer of `supports_draft_decode_metadata_update` in the spec-decode path. The dflash speculator does not check or consume this flag. This was independently confirmed in internal handoff notes (§5.4, "where the flag is consumed"):

> "NOT consumed by the DSpark/dflash speculator (`v1/worker/gpu/spec_decode/dflash/speculator.py:126`): its 'does not support full CUDA graphs; running the draft eagerly' warning is gated on `attn_cg_support` (`UNIFORM_BATCH`), independent of our flag."

**Confidence:** HIGH — confirmed via code search and internal handoff notes.

### Finding 7 — Non-causal FlashInfer plans exist but are not addressed for in-place advance

**ISSUE #50707** — "DFlash on SM121 (GB10 / DGX Spark): attention autoselect picks FLASH_ATTN for non-causal draft attention and device-asserts" documents that non-causal draft attention on SM12x causes FlashAttention to assert in `_vllm_fa2_C.varlen_fwd`; the SM100 workaround in `vllm/platforms/cuda.py` does not cover SM12x.

While this issue concerns FLASH_ATTN (not FlashInfer), it illustrates the broader challenge of non-causal attention on modern architectures. Critically, **no upstream PR or issue discusses adapting `update_draft_decode_metadata` for non-causal attention plans** — the existing implementations (FlashAttention, DeepSeek SWA) operate on causal decode paths.

**Source:** [Issue #50707](https://github.com/vllm-project/vllm/issues/50707)
**Confidence:** MEDIUM — the issue exists but does not directly address plan-advance for non-causal; the absence of such discussion is itself informative.

### Finding 8 — DFlash draft-loop capture is a separate concern

**ISSUE #53031** — "DFlash speculator `capture()` logs 'Capturing model...' even when nothing is captured" documents that the dflash speculator's capture path is observable/unobservable only via a log line. This is a UX concern, not a functional gap.

More importantly, **ISSUE #45258** — "[RFC]: FULL cudagraph support for spec-decode drafter chain steps" proposes FIRST-CLASS FULL-cudagraph support for drafter chain steps (currently PIECEWISE-only), noting that drafter chain steps pay ~15–18 % of decode step time in eager attention + per-layer metadata dispatch. The proposed solution involves `FULL_AND_PIECEWISE` keys with `uniform_decode_query_len=1` and a `CUDAGraphWrapper(runtime_mode=FULL)` around the drafter model.

This RFC is specifically about the **MTP-style drafter chain** (autoregressive, per-token), not the DFlash/DSpark block-parallel draft loop. It is a separate concern from the `update_draft_decode_metadata` hook.

**Sources:** [Issue #45258](https://github.com/vllm-project/vllm/issues/45258) · [Issue #53031](https://github.com/vllm-project/vllm/issues/53031)
**Confidence:** HIGH — both issues read in full.

## Conclusion

### Is any of this already upstream?

**Yes — partially.** The foundational infrastructure (`supports_draft_decode_metadata_update` flag + `update_draft_decode_metadata()` hook + `_fused_multi_step_decode` path) was merged upstream in **PR #46849** (2026-08-11). Several backends (FlashAttention, DeepSeek V4 sparse SWA, Triton Attention, Triton MLA) opt in and provide working in-place metadata advancement.

However:

1. **FlashInfer does not opt in.** The FI native decode path remains at `False`, still rebuilding metadata per step.
2. **The Thor-authored patch** (shipped as `thor-fused` in the overlay — see [`docker/vllm-thor/PATCHES.md`](../../docker/vllm-thor/PATCHES.md)) fills this gap for FlashInfer's **causal** native decode plan — this is net-new work, not upstream.
3. **Neither that patch nor any upstream PR addresses the DSpark/dflash non-causal draft path.**

### Is extending the in-place plan advance to the dflash/DSpark non-causal draft path net-new work?

**Yes — definitively net-new.** Why:

- **Different backend:** DSpark's draft attention runs on `FLASH_ATTN` (not FlashInfer) because FlashInfer rejects non-causal attention entirely (issue #41559). The FLASH_ATTN backend already has `update_draft_decode_metadata` from upstream PR #46849.
- **Different consumer:** the dflash speculator does not consume `supports_draft_decode_metadata_update` at all — it uses its own `attn_cg_support` / `UNIFORM_BATCH`-based capture mechanism (issue #45258; `dflash/speculator.py:126`).
- **Different geometry:** the dflash/DSpark draft attention is **non-causal** (bidirectional). The in-place plan advance for FlashInfer was designed around causal decode (`use_non_causal=False`). Whether a plan-advance works for non-causal plans is **unverified** — the kernel computes `paged_kv_indptr`, `paged_kv_last_page_len`, and `paged_kv_indices` from `seq_lens` + block table, but the non-causal FlashInfer plan has different masking semantics and may require different bookkeeping.
- **Separate capture boundary:** the dflash draft loop's CUDA-graph capture is managed by `DFlashCudaGraphManager` with `decode_query_len=self.num_query_per_req`, which is structurally different from the autoregressive speculator's `_fused_multi_step_decode` path.

**Bottom line:** extending the FlashInfer in-place plan advance to the dflash/DSpark non-causal draft path requires:

1. A new `update_draft_decode_metadata` implementation tailored to the **non-causal** FlashInfer plan (or the FLASH_ATTN plan if DSpark uses that backend).
2. Integration into the dflash speculator's draft loop — a consumer path that does not currently reference `supports_draft_decode_metadata_update`.
3. Verification that the in-place advance is valid for non-causal attention semantics.

This is **net-new engineering effort**, not a matter of enabling an existing upstream feature.

## Source breakdown

### Major claims

| Claim | Source | Confidence | Contradictions |
|-------|--------|------------|---------------|
| PR #46849 introduces `supports_draft_decode_metadata_update` + fused multi-step | [PR #46849](https://github.com/vllm-project/vllm/pull/46849) | HIGH | None |
| FlashInfer does NOT opt in to fused path | [Code search: `supports_draft_decode_metadata_update`](https://github.com/search?q=repo%3Avllm-project%2Fvllm+supports_draft_decode_metadata_update&type=code) + PR #46849 backend table | HIGH | None |
| DSpark uses non-causal SLWA via SparseMLA; dense targets use FLASH_ATTN | [PR #46995](https://github.com/vllm-project/vllm/pull/46995) body | HIGH | None |
| FlashInfer rejects non-causal attention entirely | [Issue #41559](https://github.com/vllm-project/vllm/issues/41559) | HIGH | None |
| DSpark draft loop gated on `attn_cg_support`, not `supports_draft_decode_metadata_update` | Internal handoff notes §5.4; `dflash/speculator.py:126` | HIGH | None |
| FI native decode capped at `UNIFORM_SINGLE_TOKEN_DECODE` | [Issue #49547](https://github.com/vllm-project/vllm/issues/49547) | HIGH | None |
| DFlash drafter chain steps are PIECEWISE-only (FULL cudagraph RFC) | [Issue #45258](https://github.com/vllm-project/vllm/issues/45258) | HIGH | None |

### Methodology

- **Searches performed (sequential, refined iteratively):**
  1. `"update_draft_decode_metadata FlashInfer in-place plan advance"` → 28 issues; revealed the dflash/DSpark ecosystem
  2. `"supports_draft_decode_metadata_update fused multi-step draft decode"` → 92 issues; pointed to PR #46849
  3. `"dflash dspark speculator cuda graph capture draft loop attention metadata"` → 119 issues; revealed DFlash bugs and non-causal challenges
  4. `"FlashInfer non-causal use_non_causal decode plan requires_non_causal"` → 13 issues; confirmed FlashInfer rejects non-causal
  5. `"dflash speculator draft loop rebuild attention metadata between steps"` → 30 issues; revealed DFlash operational bugs
  6. `"fused multi-step draft decode MRV2 spec decode metadata update"` → 30 issues; confirmed PR #46849 as the origin
  7. **Code search:** `supports_draft_decode_metadata_update vllm-project/vllm` → confirmed FlashInfer does not opt in
  8. **Code search:** `use_fused_multi_step_decode vllm-project/vllm` → confirmed only the autoregressive speculator consumes it
  9. **Code search:** `update_draft_decode_metadata vllm-project/vllm` → identified which backends implement it
- **Pages fully read:** [PR #46849](https://github.com/vllm-project/vllm/pull/46849) (full description, diagrams, benchmark tables, backend handling); [Issue #49547](https://github.com/vllm-project/vllm/issues/49547); [Issue #41559](https://github.com/vllm-project/vllm/issues/41559); [Issue #50707](https://github.com/vllm-project/vllm/issues/50707); [Issue #45258](https://github.com/vllm-project/vllm/issues/45258); [Issue #53031](https://github.com/vllm-project/vllm/issues/53031); [PR #46995](https://github.com/vllm-project/vllm/pull/46995); [PR #47808](https://github.com/vllm-project/vllm/pull/47808) (DSpark confidence-scheduled verification); [vLLM Daily Digest 2026-08-11](https://github.com/vllm-project/vllm-daily/blob/main/2026/08/2026-08-11.md) (PR #51145 and #46849 descriptions); internal handoff notes (Thor-authored patch context); [DFlash spec docs](https://docs.vllm.ai/projects/speculators/en/latest/user_guide/algorithms/dflash/) (non-causal attention requirement); [DSpark spec docs](https://docs.vllm.ai/projects/speculators/en/latest/user_guide/algorithms/dspark/) (Markov/confidence heads).
- **Issues read in full:** 45258, 55312, 50707, 41559, 53031, 13264, 49547.
- **Pages skipped and why:** ASCEND-ported PRs referencing #46849 (irrelevant to CUDA/FlashInfer); DFlash2 Ampere/OOB/sliding-window issues (#53873, #55279, #55800) — ARM/Ampere-specific, not SM110/FlashInfer; FlashInfer MLA workspace overflow (#50781) — different problem domain (workspace buffers, not metadata advance); ROCm/AMD-specific issues — not applicable to Thor/aarch64/SM110.

## Gaps and caveats

1. **No upstream PR explicitly discusses plan-advance validity for non-causal FlashInfer plans.** The existing `update_draft_decode_metadata` implementations all target causal decode paths. Whether the same mathematical transformation (computing `indptr`/`last_page_len`/`page_indices` from `seq_lens` + block table) holds for non-causal plans is unverified.
2. **The dflash speculator's draft attention may use FLASH_ATTN rather than FlashInfer** depending on the target model's capabilities. For Qwen3.8 (hybrid GDN/Mamba), the draft attention runs on `FLASH_ATTN` — which already has `update_draft_decode_metadata` from upstream. The question is whether the dflash speculator **hooks into** this existing upstream capability.
3. **No empirical validation exists for DSpark on Jetson Thor (SM 11.0a).** All public DSpark data is on H200/H100/GB10. The SM110 architecture may exhibit different behavior for non-causal attention plans.
4. **Internal handoff notes report that the Thor-authored FlashInfer kernel works correctly but shows no measurable benefit** at K≤3 MTP on the then-current workload. Extending to DSpark (K=7) may reveal a benefit, but this is unmeasured.
5. **Version alignment:** PR #46849 was merged in v0.27.x (before v0.28.0). Ensure the target vLLM version includes this PR; the `daily-aarch64` images at v0.28.0+ include it.
