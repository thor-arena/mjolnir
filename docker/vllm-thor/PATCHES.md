# vLLM Thor PR Patches

Five vLLM PR fixes, **hand-adapted** onto the vLLM nightly (not a raw
`git apply` of the upstream PR diffs, because the nightly code has diverged
from each PR's base), plus nine **Thor-authored** patches: two for a
capability the upstream backends don't yet have, two `sm_110`
gate probes (see "Thor sm_110 gate probes" below), one dispatch-side
`sm_110` draft-CG shape gate (see "Thor sm_110 draft-CG shape gate" below),
one **flashinfer-side** sm_110 enablement hunk (see "Thor sm_110 GDN
prefill enablement (flashinfer side)" below), one hd256 FP8-KV kernel +
policy patch (the "N-1" workstream — probe-gated, T11 canary in
`kernel_test_sm110_gates.py`), one hd256 1CTA decode carve-out (the
"N-2 decode" workstream — see "Thor hd256 1CTA decode carve-out (v11,
2026-09-24)" below), and one hd256 GEMV decode kernel (the "N-2 decode"
completion — a **default-on** build patch; see "Thor hd256 GEMV decode
kernel" below). Fourteen patches in total, applied by
`apply_patches.py` onto the installed vllm package **and** the installed
flashinfer package; verified by `verify_patches.py`.

Base this was verified against: **vllm 0.30.0**
(`vllm/vllm-openai:v0.30.0-ubuntu2404`, 2026-09-26), FlashInfer 0.6.18.post1.

| File | PR | What it fixes |
|------|----|---------------|
| `50885-flashinfer-full-cudagraph.patch` | [#50885](https://github.com/vllm-project/vllm/pull/50885) | Capture **FULL** decode cudagraphs for spec-decode on the FlashInfer **native** (non-trtllm) path — uniform multi-token verify batches. Previously only trtllm-gen (SM100) / dedicated-XQA (SM12x) could FULL-capture spec decode; elsewhere it fell back to PIECEWISE, hurting Thor (SM110) TTFT. |
| `49652-draft-decode-capture.patch` | [#49652](https://github.com/vllm-project/vllm/pull/49652) | Autoregressive (MTP) **draft-decode** cudagraph capture under **Dynamic SD**. The draft decoder always processes 1 token/request, so its graphs must stay at fixed query length 1 instead of following the target's dynamic-SD verify schedule. |
| `54165-mamba-spec-kv-restore.patch` | [#54165](https://github.com/vllm-project/vllm/pull/54165) | Restore **hybrid mamba-align cache hits under spec decode** with a KV connector. Gates the EAGLE last-block drop on drafters that actually *preserve* (pollute) the target KV cache — DFlash/DSpark don't, so they must not back off `last_cache_position`. |
| `55390-mtp-draft-group-annotation.patch` | [#55390](https://github.com/vllm-project/vllm/pull/55390) | **Annotate the MTP draft KV group positionally on the hybrid path.** The draft-group detector (`_annotate_eagle_groups`) only recognized DSpark's `non_causal_multi_token_decode` marker, so for Qwen hybrid MTP (whose MTP layer is spec-identical to the target's) no group was flagged → the coordinator's fallback treated *every* group as a draft group → Mamba groups got a widened lookup window that align-mode never satisfies → the startup WARNING and crippled cross-request prefix-cache reuse. The fix generalizes the trailing-layer positional rule from DeepSeekV4-only to **any `method=="mtp"` model** (MTP blocks always register their layers last), guarded by an exact-layer-partition check. |
| `55519-no-warn-when-block-drop-off.patch` | [#55519](https://github.com/vllm-project/vllm/pull/55519) | **Don't warn "prefix reuse is disabled" when `disable_eagle_block_drop` is set.** All flag-all fallbacks are gated on `use_eagle_block_drop()`, so with the drop disabled reuse is intact — the old `use_eagle()` gate made the warning a false positive. Also rewords the scheduler's "drop disabled" warning to document that reuse works in that mode. |
| `thor-fused-draft-decode-fi-native-update.patch` | — (Thor-authored) | **Implement `update_draft_decode_metadata` for the FlashInfer backend** so the spec-decode loop can use the **fused multi-step draft decode** instead of the "rebuild attention metadata between draft steps" fallback. The upstream contract (backend.py `update_draft_decode_metadata`) is implemented for triton/triton_mla (a `pass`, they read live GPU buffers) and flash_attn (re-plan the scheduler), but the FlashInfer native decode wrapper plans from host buffers and had no in-place advance. Adds `_advance_fi_decode_plan_kernel`, which re-derives the FI decode plan (`paged_kv_indptr`/`last_page_len`/`page_indices`) **on the GPU** from the advancing `seq_lens` + block table — the same formula as `build()`, but capture-safe (no host sync, no D2D index copy) so it replays correctly inside the draft CUDA graph. Gated to full-cudagraph, single-rank, non-DCP, native-FI (non-trtllm) builders. Pays off only at **K ≥ 3** (MTP3 / DSpark K≥2); it is a deliberate no-op at K=2. |
| `thor-dspark-draft-fi-noncausal-cudagraph.patch` | — (Thor-authored) | **Capture the DSpark non-causal multi-token draft under a FULL CUDA graph** instead of running the draft eagerly (removes the `speculator.py:124` "does not support full CUDA graphs; running the draft eagerly" warning). A non-causal drafter (DSpark/DSpark) routes its draft through the FI *prefill* wrapper — the FI *decode* wrapper cannot express non-causal multi-token (its fa2 kernel is block-causal only for `q_len_per_req > 1`). In cuda-graph mode that prefill wrapper freezes its batch size at creation and reads its KV plan from **persistent** buffers, so one wrapper per captured batch size is needed (`_noncausal_prefill_wrappers_cudagraph`, mirroring `_decode_wrappers_cudagraph`); the draft plans eagerly outside the captured graph and the captured `run()` replays safely across ragged re-plans. Adds the `_get_noncausal_prefill_wrapper_cudagraph` factory, grants a non-causal `UNIFORM_BATCH` in `get_cudagraph_support` (gated on a multi-token drafter, non-SM12x, single-rank DCP), and dispatches the prefill batch to the cudagraph wrapper in `build()`. Lands **last** in `PATCH_ORDER` (its `flashinfer.py` hunks sit on top of the fused-draft-decode patch's changes). Most bump-fragile patch — see below. |
| `thor-draft-cg-gate-sm110.patch` | — (Thor-authored) | **Shape-gate the DSpark draft's FULL cudagraph replay on sm_110 (the c1 fix).** The forensics showed the graphed draft is 7–16% *slower* than the eager draft at N=1 (replay overhead eats the launch savings on a tiny 5-layer draft). The gate routes per-step draft batches below `VLLM_THOR_DRAFT_CG_MIN_BATCH` requests (default **2**; min 1, where 1 = today's behavior) through the **existing eager fallback** via the dispatch's `need_eager` channel — no new eager path, no capture-logic changes. sm_110 only: off sm_110 the decision helper is a structural no-op (always CG). One-shot `THOR draft-CG gate: …` log lines; any gate evaluation error degrades to the CG behavior (logged once). Probe-gated, inert on non-sm_110, default N=1→eager. |
| `thor-fa4-hd256-1cta-decode-sm110.patch` | — (Thor-authored) | **The 1CTA decode carve-out for the FA4 hd256 kernel (the v11 "N-2 decode" workstream).** Decode-shaped hd256 calls — `max_seqlen_q <= 8` (decode M 1..8, incl. MTP verify; prefill is 1024+), non-local, dense or paged-128 TMA (the kernel's only legal paged mode) — now OR into the `hd256_use_2cta` carve-out in `vllm_flash_attn/cute/interface.py` (one hunk), so the kernel drops its 2CTA cluster to a single CTA (`cluster (1,1)`, `CtaGroup.ONE`): a **1.56× decode win** at L=8192 M=1 (clean micro-bench 2CTA 323 µs → 1CTA 207 µs), with 1CTA == 2CTA == fp32-ref within fp8 tolerance; the kernel file is untouched and the 2CTA prefill path is unchanged (separate compile keys coexist in one process). The in-tree hd256 SplitKV machinery was investigated and is a no-win on **both** the 2CTA and 1CTA paths — kept inert (see "Thor hd256 1CTA decode carve-out (v11, 2026-09-24)" below). |
| `thor-fa4-hd256-gemv-decode-sm110.patch` | — (Thor-authored) | **The M=1 hd256 GEMV decode kernel (the "N-2 decode" completion).** Four `vllm_flash_attn/cute/interface.py` hunks: the `BlackwellHd256DecodeGEMV` import, the `_gemv_auto_num_splits` helper (FlashInfer Alg.1 split plan — fill `4 × #SM` CTAs), a relaxed mixed Q/KV dtype check (`_gemv_dtype_ok`, M=1 hd256 only), and the `use_gemv_hd256` dispatch gate that returns `out, lse, None, None` **before** the tcgen05/1CTA forward. The kernel itself (`vllm_flash_attn/cute/sm100_hd256_decode_gemv.py`, a CuTe-DSL pure-FMA GEMV) is a NEW file upstream has none of, so it is installed by a **separate `COPY`** in the `Dockerfile` (from `fa4-gemv-kernel/`) rather than embedded as a `/dev/null`→file new-file diff in this patch — this patch edits only the existing `interface.py`. **DEFAULT-ON** (opt out with `VLLM_FA4_HD256_GEMV=0`) — unlike the rest of the fa4 group (passive-by-default), this changes in-scope M=1 hd256 decode behavior the moment FA4 hd256 is selected; scope gates confine it to M=1, `head_dim==256`, arch 10/11, dense or paged-16/128, `seqused_q` None, no softcap/block-sparse/learnable-sink. Within ~10% of FlashInfer at L=8192 (222.8 µs vs 202.05 µs, same gated window); see "Thor hd256 GEMV decode kernel" below. |
| `thor-gdn-prefill-sm110.patch` | — (Thor-authored) | GDN **prefill** grant in `_resolve_gdn_prefill_backend()` (the fast FlashInfer kernel for the model's GDN layers) — probe-gated at import; see "Thor sm_110 gate probes". |
| `thor-fa4-fp8kv-sm110.patch` | — (Thor-authored) | **FA4 + FP8-KV** eligibility gate (probe-gated fam(110) OR-term in `flash_attn_supports_kv_cache_dtype`) + the FA4 arch assert widened to `(10, 11)` — **inert until a config selects FA4** (default stays FlashInfer); see "Thor sm_110 gate probes". |
| `thor-fa4-hd256-fp8-sm110.patch` | — (Thor-authored) | hd256 **FP8-KV kernel + policy** patch (the "N-1" workstream — enable the FA4 hd256 kernel on sm_110, probe-gated, T11 canary in `kernel_test_sm110_gates.py`); the decode path is the 1CTA carve-out above + the GEMV decode below. |
| `thor-gdn-prefill-fi-sm110.patch` | — (FlashInfer) | **FlashInfer-side** GDN prefill non-CP dispatch allowlist widened major 10 → (10, 11) so the SM100 CuTe-DSL kernel is reachable on sm_110 (behavior unchanged on other devices); see "Thor sm_110 GDN prefill enablement (flashinfer side)". |
  
## Why "hand-adapted" and not the raw diff

The four PRs were written against vllm bases that predate the current
nightly. Two kinds of drift force manual adaptation:

1. **The nightly already contains *newer* concepts.** The scheduler/coordinator
   now use `use_eagle_block_drop()` / `use_eagle_block_drop()` gating, which the
   PRs (written against `use_eagle`) don't know about. For **54165** this means
   its "introduce `drop_last_prefix_cache_block`" hunk is *superseded* — the
   correct adaptation is to make the existing `use_eagle_block_drop()` return
   `use_eagle_preserves_target_kv_cache()` instead of `use_eagle()`. For
   **52244**, the new `state_position` stop is gated on `use_eagle_block_drop`
   (the nightly's equivalent of the PR's `use_eagle`).

2. **Signatures grew a trailing parameter.** The nightly's
   `reachable_block_mask(...)` added `dcp_world_size`/`final_segment_end_block`
   params after `reachable_boundaries`; the PRs insert
   `unreachable_boundaries` at that same slot, so the whole signature (and the
   `SlidingWindowManager` override) must be extended, not just the base.

3. **A PR hunk is already done by the nightly.** 54165's
    "SlidingWindow `find_longest_cache_hit` assert → block-aligned fallback"
    hunk is already present in the nightly (it has the dense-fallback comment),
    so that hunk is omitted from the adapted patch.

Note: **55390/55519** (opened 2026-09-04/06, i.e. ~2 days after this nightly)
were adapted with zero context drift — only the `tests/` hunks were dropped
(the wheel ships no test dir). Their source hunks applied verbatim at offset
only, so these two should survive nightly bumps longest.

## Files touched (per patch)

- **50885**: `v1/attention/backends/flashinfer.py`
  - new `flashinfer_supports_uniform_multi_token_decode()` (@cache) probing
    FlashInfer's `fast_decode_plan` for `q_len_per_req`;
  - `_decode_wrappers_cudagraph` keyed by `(batch_size, q_len_per_req)`;
  - `supports_spec_as_decode` / `get_cudagraph_support` extended to return
    `UNIFORM_BATCH` when the FI native multi-token path is available;
  - `build()` computes `decode_q_len = num_decode_tokens // num_decodes`,
    per-request indptr/last_page_len, passes `q_len_per_req` through to
    `_get_decode_wrapper` and `fast_plan_decode`.
- **49652**: base `CudaGraphManager._init_candidates` (rebased 2026-09-18)
  - one-line guard: skip Dynamic-SD candidate expansion when
    `decode_query_len <= 1` — replaces the old
    `SpeculatorCudaGraphManager.__init__` workaround in
    `v1/worker/gpu/spec_decode/autoregressive/cudagraph_utils.py`.
- **54165**: `config/speculative.py`
  - new `use_eagle_preserves_target_kv_cache()` (eagle/eagle3/mtp only);
  - `use_eagle_block_drop()` now returns
    `use_eagle_preserves_target_kv_cache() and not disable_eagle_block_drop`.
- **52244**: `v1/core/kv_cache_utils.py`, `v1/core/sched/scheduler.py`,
  `v1/core/single_type_kv_cache_manager.py`, `v1/core/kv_cache_coordinator.py`
  - new `mamba_state_cache_position()` — where a replay of a prompt lands
    (one hash unit below the deepest boundary under the `num_tokens - 1` cap);
  - scheduler adds that position to the mamba split `stops` (gated on
    `mamba_partial_cache_hit and use_eagle_block_drop`);
  - base `cache_blocks`/`reachable_block_mask` gain `unreachable_boundaries` +
    `_unreachable_boundaries`; `FullAttentionManager._cache_partial_tail_block`
    also registers the replay-tail boundary under EAGLE;
  - `MambaManager.reachable_block_mask` drops unreachable blocks in *every*
    retention mode; new `_state_boundary`/`_unreachable_boundaries` (gated on
    `engine_uses_eagle`); Mamba partial-tail uses `_state_boundary`;
  - coordinator sets `manager.engine_uses_eagle = use_eagle` on every manager.
- **55390**: `v1/core/kv_cache_utils.py`
  - `_is_deepseek_v4_eagle()` replaced by `_uses_trailing_mtp_layers()` — the
    trailing-layer positional rule now applies to **any `method=="mtp"`**
    model (MTP blocks always register their layers after the target's), not
    just DeepSeekV4;
  - new `_groups_partition_layers_exactly()` guards the rule: it only fires
    when the groups cover every layer of `kv_cache_spec` exactly once;
  - `_annotate_eagle_groups()` param renamed
    `use_deepseek_v4_fallback` → `use_trailing_layer_fallback`, partition check
    added before the flag.
- **55519**: `v1/core/kv_cache_utils.py`, `v1/core/sched/scheduler.py`
  - `_warn_if_unannotated_eagle_mamba()` gate changed from `use_eagle()` to
    `use_eagle_block_drop()` — when `disable_eagle_block_drop` is set, every
    flag-all consumer fallback is already off, so the "reuse disabled" warning
    was a false positive;
  - scheduler's "drop disabled" warning reworded to document that the trailing
    block stays cache-eligible in that mode.
- **thor fused draft-decode** (Thor-authored, no upstream PR):
  `v1/attention/backends/flashinfer.py`
  - `FlashInferMetadataBuilder.__init__` sets
    `supports_draft_decode_metadata_update = enable_cuda_graph and
    dcp_world_size==1 and not use_trtllm_decode_attention` (the trtllm flag
    identifies the *non*-native path) and allocates the live references
    `_draft_seq_lens_gpu` / `_draft_block_table`;
  - `build()` captures `common_attn_metadata.seq_lens` / `.block_table_tensor`
    (the advancing GPU buffers) into those fields;
  - new `update_draft_decode_metadata()` — returns for the trtllm/XQA
    `decode` (already stable), guards to the pure uniform single-token decode,
    and launches `_advance_fi_decode_plan_kernel`;
  - new module-level Triton `_advance_fi_decode_plan_kernel` — one program per
    request re-derives `paged_kv_indptr` (prefix block count, recomputed from
    the read-only `seq_lens` so programs are independent), `last_page_len`
    (promoted to `page_size` when a non-empty request lands on a boundary), and
    `page_indices` (copied from the live block table). Same formula as
    `_compute_flashinfer_kv_metadata`.
- **thor draft-CG gate** (Thor-authored, no upstream PR):
  `v1/worker/gpu/spec_decode/dflash/speculator.py`
  - pure decision helper `_thor_draft_cg_use_cuda_graph(num_reqs,
    capability, min_batch)` — `True` = replay the FULL draft graph; a
    structural no-op (always `True`) off sm_110;
  - lazy once-parsed `VLLM_THOR_DRAFT_CG_MIN_BATCH` env knob (default 2,
    minimum 1; invalid → 2 + one-time warning) and lazy
    `torch.cuda.get_device_capability()` read;
  - the per-step dispatch (`dispatch_cg_and_sync_dp` call in `propose`) now
    passes `need_eager = is_profile or _thor_draft_cg_gate_wants_eager(num_reqs)`
    — below the threshold the existing `NONE`-descriptor eager path runs;
  - four one-shot `logger.info` lines (state / eager / cudagraph / error).

## Behavioral gate (`functional_check_55390_55519.py`)

The sentinel check in `verify_patches.py` proves the code *landed*; the Dockerfile
additionally runs `functional_check_55390_55519.py`, which mirrors the PRs'
unit tests (the wheel ships no `tests/` dir, so the specs are constructed
directly): a Qwen3.5-shaped hybrid `[GDN×3, full-attn] × 2` + trailing MTP
full-attn layer must flag **exactly one** group (the MTP one), never the
Mamba groups; non-MTP methods stay unannotated; a non-partitioning group set
must not trigger the rule; and the warning path must be gated on
`use_eagle_block_drop()`. Run standalone inside any container with a patched
vllm: `docker run --rm --entrypoint python3 <image> /path/to/functional_check_55390_55519.py`.

## Verifying a new nightly before re-adapting

```
docker pull vllm/vllm-openai:nightly-aarch64
# quick check: which fixes are already merged upstream?
docker run --rm --entrypoint sh vllm/vllm-openai:nightly-aarch64 \
  python3 - <<'PY'
import importlib.util, inspect, pathlib
r = pathlib.Path(importlib.util.find_spec("vllm").origin).parent
def has(rel, s): return s in (r/rel).read_text()
print("50885", has("v1/attention/backends/flashinfer.py",
                    "flashinfer_supports_uniform_multi_token_decode"))
print("54165", has("config/speculative.py",
                    "use_eagle_preserves_target_kv_cache"))
print("52244", has("v1/core/kv_cache_utils.py",
                    "mamba_state_cache_position"))
print("55390", has("v1/core/kv_cache_utils.py",
                    "_uses_trailing_mtp_layers"))
print("55519", has("v1/core/kv_cache_utils.py",
                    'not spec_config.use_eagle_block_drop():'))
print("fi-update", has("v1/attention/backends/flashinfer.py",
                       "def _advance_fi_decode_plan_kernel"))
PY
```
If a fix is already `True`, delete that `.patch` file and its entry in
`apply_patches.py::PATCH_ORDER` + `verify_patches.py::FIXES` (the build's
`--forward` would skip it anyway, but keeping the list accurate keeps
verification meaningful). If a patch's `--forward` run leaves a missing fix,
`verify_patches.py` names it — re-adapt that one patch against the new source.

## The Thor fused-draft-decode patch — verification & fragility

This patch is **not** an upstream PR; it is the first of the six-plus-one where
the overlay *adds a capability* to vLLM rather than backporting a merged fix.
Consequences:

- **Most fragile under bumps.** It inserts into the FI native-decode branch of
  `build()` and the `__init__` buffer block — both churned by #50885 and any
  future FlashInfer-backend work. If it fails to apply after a nightly bump,
  re-adapt it against the new `flashinfer.py` (the other six are unaffected).
- **Kernel is GPU-verified, not just present.** `verify_patches.py` only proves
  the symbols landed. The actual Triton kernel is checked by
  `kernel_test_fi_update.py`, which runs `_advance_fi_decode_plan_kernel`
  against a numpy reference over random seq_lens, page-boundary, empty
  (padded) requests, multi-page, and a 20× determinism repeat. The test
  **self-skips (exit 0)** when no CUDA device is visible or the GPU is
  saturated (a live engine holds most of the 128GB and can fragment the free
  pool) — so a plain no-`--gpus` build still passes. Build with
  `docker build --gpus all …` on a Thor (or run the test in a `--gpus all`
  container) to actually exercise the kernel.
- **Correctness guardrails baked in.** The `enable_cuda_graph` gate is load-
  bearing: without full-cudagraph the decode wrapper is built with `paged_kv_*
  = None` (`_get_decode_wrapper`), so the kernel would have nothing valid to
  advance — enabling the fused path in that case would be a real bug. The
  `not use_trtllm_decode_attention` gate keeps the trtllm/XQA path (which is
  already stable and needs a `pass`) out of the GPU-advance path. The method
  also refuses to advance anything that is not a pure uniform single-token
  decode, so a future non-uniform draft query degrades to the fallback rather
  than reading a stale plan.
- **No K=2 regression.** At `num_speculative_tokens=2` the fused loop runs a
  single draft step and the inter-step update is guarded off by
  `step < num_speculative_steps - 1`, so the K=2 MTP config is byte-for-byte
  unchanged in behavior; the patch only activates the fused path for K ≥ 3
  (MTP3 / DSpark K≥2), which is where the per-step metadata rebuild is avoided.

### Production verification (2026-09-09)

The 7-patch image was built, caches cleared, and benched (pp2048/tg128,
Qwen3.8-27B NVFP4, MTP K=2, MTP K=3, DSpark K=7) against the 6-patch
baseline. Outcome:

- **Working correctly.** The autoregressive speculator's fallback line
  (`INFO … speculator.py:120 Fused multi-step draft decode is not supported by
  attention backend(s) FLASHINFER; falling back…`) is **present in the 6-patch
  log and absent in the 7-patch logs** — the fused path is active. No runtime
  errors and no "will not be advanced in place" guard warnings in any log.
- **No K=2 regression (empirical).** MTP K=2 tg deltas vs the 6-patch baseline
  are ±7% (d0: 25.2→23.4; d8192c4: 69.3→70.7) — indistinguishable from
  run-to-run noise, confirming the by-design no-op.
- **No measurable gain at K=3 in this benchmark.** MTP K=3 is the only MTP
  config that executes the update (once per decode), and it saves exactly one
  Python `build()`+`plan()` (H2D) per decode — below the ±5–7% noise floor.
  The favorable deep-context **TTFT** deltas (d4096c4 −300ms, d8192 −245ms)
  are prefill-side and not attributable to this decode-path kernel (treated as
  variance, not a claim).
- **DSpark K=7 is unaffected by design.** DSpark uses the *dflash* speculator
  (`v1/worker/gpu/spec_decode/dflash/speculator.py`), which gates on
  `attn_cg_support` (`UNIFORM_BATCH`), **not** on
  `supports_draft_decode_metadata_update`; its "does not support full CUDA
  graphs; running the draft eagerly" warning appears identically in both
  baseline and patched logs. It is a control group, not a regression.
  (The only consumers of the flag are
  `autoregressive/speculator.py:115` and `v1/worker/utils.py:318`.)

Implication for measuring the benefit: the current workloads can't isolate
it. Options: MTP at K ≥ 4–5 (check how many MTP layers the model exposes —
`num_nextn_predict_layers`), or port the same plan-advance to the dflash
speculator — a **separate** patch (its draft attention is non-causal, so the
kernel's assumptions need re-derivation, not a copy-paste).

## Nightly bump to `dev580+g385dce36b` (2026-09-09) — 52244 re-adaptation

Bumping from `dev516+g9ea8f3ffc` to `dev580+g385dce36b` broke 52244's
application (the other six patches were unaffected): the nightly restructured
the scheduler's mamba-split region and `MambaManager.cache_partial_tail_block`.
Three 52244 hunks needed re-adapting:

- **scheduler `stops` tuple** — the nightly inserted a whole junction-stop
  block between the `tail_boundary`/`use_eagle_block_drop` shift and
  `stops = (`, so the `state_position` definition and its insertion into the
  tuple (one hunk now) anchor on the junction block's tail.
- **base `__init__` `engine_uses_eagle`** — the nightly added a
  `fine_grained_prefix_cache` field after `self.use_eagle = False`, so the
  hunk now inserts the flag right after the `use_eagle` line.
- **`cache_partial_tail_block`** — the nightly had added its *own* inline
  `use_eagle` one-unit shift **and** a fine-grained junction check. The adapted
  hunk drops the nightly's inline shift (superseded by 52244's
  `_state_boundary(request)`, which fires on the Mamba/target manager via
  `engine_uses_eagle`) but **keeps** the new junction check, so the method now
  reads `if num_tokens != self._state_boundary(request) and not (…junction…)`.

Because the old `verify_patches.py` only checked `mamba_state_cache_position`
+ `manager.engine_uses_eagle` (which apply independently of the three hunks
above), a partial 52244 apply reported `all 7 fixes verified` and built a
**silently broken** image (the scheduler computed `state_position` but never
used it). The 52244 fix now carries four extra sentinels — `engine_uses_eagle`
init, the `state_position,` line in the scheduler `stops` tuple, the
`_state_boundary` method, and its use in `cache_partial_tail_block` — so the
same partial state now fails the build gate.

## Nightly bump to `0.29.1rc1.dev347+gdee37d891` (2026-09-18) — 52244 upstreamed

**52244 dropped** (see table above): its fix (hybrid GDN/Mamba prefix-cache
hits under MTP) upstreamed in 0.29.1, restructured as the replay-boundaries
architecture (`get_replay_boundaries`/`reachable_boundaries`). The other seven
patches were rebased onto the new base; the fresh-tree gate (apply all 7 +
`verify_patches.py` + `functional_check_55390_55519.py`) passed.

- Rebased 2026-09-18 onto 0.29.1rc1.dev347: the flashinfer trio coexists with
  the new trtllm/XQA decode machinery (the FI-native grants fire only when that
  path is off — it is always off on Thor SM110); **49652** is now a base
  `CudaGraphManager._init_candidates` guard instead of the old
  autoregressive-speculator `__init__` workaround.

## Nightly bump to `0.29.1rc1.dev452+g3df4ae153` (2026-09-21) — clean, no re-adaptation

Bumping from `dev347+gdee37d891` to `dev452+g3df4ae153` (105 commits) required
**zero hunk changes**: all 10 vllm + 1 flashinfer patches applied verbatim
(fresh-tree container test + the real build, identical result). The only
movement was line offsets in **55519** (hunks landed at +20 and +12 lines,
context intact). flashinfer-python moved 0.6.18 → 0.6.18.post1; the GDN
allowlist hunk (`gdn_prefill.py`) still applies at the same line.

- **52244-drop re-validated**: the replay-boundaries architecture that
  superseded it (`get_replay_boundaries`/`reachable_boundaries` in
  `v1/core/single_type_kv_cache_manager.py`) is present in the new base.
- **49652's target moved upstream**: it now lands in
  `v1/worker/gpu/cudagraph_utils.py` (the file the patch names); the old
  `v1/worker/gpu/spec_decode/autoregressive/cudagraph_utils.py` still exists
  alongside it, and the Dockerfile smoke-test list covers both, so no
  Dockerfile change was needed there.
- Build gates: `all 10 fixes verified` + `all 1 flashinfer fixes verified`,
  compileall 7 OK, functional check 55390/55519 ALL PASS; GPU canary
  (`kernel_test_sm110_gates.py`, `--gpus all`): **T6 GDN prefill PASS**
  (FI kernel enabled on sm_110, FI-vs-FLA A/B match), **T9 FA4 FP8-KV PASS**
  (probe enabled, passive-by-default), **T10 draft-CG gate PASS** (4/4).
- Image: `mjolnir/vllm-thor:qwen38-sm110-v9`
  (`sha256:321f8f618b28ce7b6842a2e70c40b68b0f427168bc8c26b3a92e00993968d2c1`).

## The DSpark non-causal cudagraph patch — fragility

`thor-dspark-draft-fi-noncausal-cudagraph.patch` is the **most bump-fragile**
patch in the set (more than the fused-draft-decode patch). It inserts into two
of the most churned regions of `v1/attention/backends/flashinfer.py` at once:
the `get_cudagraph_support` tail (where the non-causal `UNIFORM_BATCH` grant
lands) and the prefill branch of `build()` (where the routing call site
replaces the `_get_prefill_wrapper` call). Both are rewritten by #50885 and
further by the fused-draft-decode patch, so a nightly that moves either region
breaks this patch's context — if it fails to apply after a bump, re-adapt it
against the new `flashinfer.py` (the other six are unaffected). It is kept
**last** in `PATCH_ORDER` for exactly this reason.

Because both hunks target hot code, the `verify_patches.py` entry asserts the
*used* lines, not just the definitions (the 52244 lesson): the wrapper factory
def, the `_noncausal_prefill_wrappers_cudagraph` dict in `__init__`, the gate
line in `get_cudagraph_support`, and — load-bearing — the `build()` routing call
site `self._get_noncausal_prefill_wrapper_cudagraph(num_prefills)` plus the
`use_nc_cudagraph =` condition. A partial apply that lands the definition but
not the routing must fail the gate. The captured behavior itself (gate, routing,
non-causal run vs. reference, capture-replay across ragged re-plans,
determinism) is checked by `kernel_test_dspark_draft_nonncausal.py` (Dockerfile
gate), which **self-skips (exit 0)** when no CUDA device is visible or the GPU
is saturated, so a plain no-`--gpus` build still passes. Build with
`docker build --gpus all …` on a Thor (or run the test in a `--gpus all`
container) to actually exercise the graph.

## Thor sm_110 gate probes (2026-09-18)

Two Thor-authored patches that turn *static upstream arch gates* into
**one-shot runtime probes** for Thor (SM 110 / sm_110a). Each adds an SM110
grant to an existing gate, but the grant fires only if a small
deterministic kernel A/B test passes on the current device. The verdict is
process-cached (zero per-forward overhead), any failure degrades to
today's exact behavior, and — the point of the probes — each grant
**auto-activates with zero vLLM changes** when the underlying kernel
support lands in a future CuTe-DSL / flashinfer release.

| Patch | What it gates | Probe behavior | Current Thor status |
|-------|---------------|----------------|---------------------|
| `thor-gdn-prefill-sm110.patch` | The FlashInfer **GDN prefill** grant in `_resolve_gdn_prefill_backend()` (`qwen_gdn_linear_attn.py`) — the fast prefill kernel for the model's GDN layers | FlashInfer's `fi_chunk_gated_delta_rule` vs. the Triton/FLA reference on a deterministic, state-carrying 2-chunk input (B=1, T=256, H=2, K=V=128, seeded; pre-L2-normalized q/k); output **and** final state must match within tight bf16 tolerances | **Enablement (probe-gated, live-verified 2026-09-21 on this Thor)**: the flashinfer-side hunk (`thor-gdn-prefill-fi-sm110.patch`) widens the FI GDN prefill dispatch to sm_110, so the probe now *exercises the real FI kernel path*; live T6 canary: probe **PASS** (`THOR GDN prefill probe: FI kernel enabled on sm_110`, `resolved ('auto', 'flashinfer')`); the grant fires only if the probe passes, otherwise Triton prefill, byte-identical to today |
| `thor-fa4-fp8kv-sm110.patch` | The **FA4 + FP8-KV** eligibility gate: probe-gated fam(110) OR-term in `flash_attn_supports_kv_cache_dtype` (`fa_utils.py`) — the *only* policy gate that rejects FA4+FP8 on sm_110 — plus the FA4 kernel's arch assert widened `arch//10 == 10` → `in (10, 11)` (`cute/interface.py`) | One Qwen3.8-shaped full-FP8 paged GQA decode (B=2, 8q/4kv, head_dim 128, page 16, causal, seqused 96/64; Q+K+V all e4m3, mirroring vLLM's quantize-Q-for-FP8-KV production path) through the real `fa_version=4` dispatcher vs. an fp32 reference (rtol 2e-2 / atol 1e-2; live evidence on this shape class: maxerr ≈ 0.015) | **INERT today until config explicitly selects FA4** (passive-by-default): sm_110's default FA version is 2, so the probe-gated term is unreachable and the patch changes nothing for every existing config; enabling requires `--attention-config '{"backend":"FLASH_ATTN","flash_attn_version":4}'` + fp8 KV, and even then the grant fires only while the probe passes on the device |

> **REMOVED 2026-09-21:** the other two probe-gated sm_110 grants —
> `thor-b12x-nvfp4-sm110.patch` (NVFP4 B12x GEMM) and
> `thor-trtllm-xqa-decode-sm110.patch` (XQA/TRTLLM decode) — are out of the
> stack: probe-gated sm_110 grants whose underlying kernels are
> arch-incompatible (B12x targets sm_120a–121f; XQA whitelist
> majors [9, 10, 12]) — measured INERT on Thor, no improvement over stock;
> per the no-improvement rule. Re-add trigger: a flashinfer release adding
> sm_110 B12x/XQA support (code preserved in git history +
> `task/nvfp4-thor/patchB-b12x-nvfp4.md` / `patchC-trtllm-xqa.md`).

`thor-gdn-prefill` and `thor-fa4-fp8kv` are diffed against the pristine
base (they touch files that no other patch touches), so their position in
`PATCH_ORDER` is immaterial.

Runtime check on Thor: `kernel_test_sm110_gates.py` (manual, `--gpus all`)
exercises each probe against the installed patched vllm and reports
**PASS** (kernel support found and the consuming gate/selection agrees) /
**INERT-OK** (probe correctly disabled — the *expected* state on the
current stack, not a failure) / **SKIP** (no CUDA device); it exits 0
unless a probe verdict disagrees with the gate or kernel selection that
consumes it. **T10** (the draft-CG gate's pure decision logic) needs no
GPU and runs unconditionally — without a device it is the only section
that executes (T6/T9 print SKIP).

## Thor sm_110 draft-CG shape gate (2026-09-19)

`thor-draft-cg-gate-sm110.patch` — purpose: fix the **c1 regression**
(V4 draft-cudagraph measured 7–16% *slower* at N=1, neutral at N=2, −5%
at N=4; study: `task/dspark-nc-cudagraph/cudagraph-c1-fix-study.md` Option
B). It does not touch capture: the draft graphs are still captured for
every bucket (the 49652 `_init_candidates` guard is undisturbed); only the
per-step *dispatch* decision changes.

- **Files touched:** `vllm/v1/worker/gpu/spec_decode/dflash/speculator.py`
  (only file — no other patch touches it, so its position in `PATCH_ORDER`
  is safe as long as it lands after `thor-dspark`).
- **Env knob:** `VLLM_THOR_DRAFT_CG_MIN_BATCH` — int, default **2**,
  minimum 1. `1` = CG for every batch size (today's behavior, for A/B).
  Invalid values → default 2 + one-time warning. Parsed lazily once.
- **Gate site:** the per-step draft dispatch in `propose`
  (`dispatch_cg_and_sync_dp(..., need_eager=...)`, dflash/speculator.py):
  `need_eager = is_profile or _thor_draft_cg_gate_wants_eager(num_reqs)`.
  Below the threshold the existing `NONE`-descriptor path runs (the same
  eager fallback the draft already uses for profile runs / no-CG-support
  — no new eager code). `is_profile` short-circuits, so the gate never
  runs during capture.
- **THOR log lines** (one-shot each, `logger.info`):
  `THOR draft-CG gate: active (sm_110, min_batch=<t>)` /
  `THOR draft-CG gate: inactive (arch <maj>.<min>)` (first evaluation);
  `THOR draft-CG gate: N=<n> -> eager (<n> < min_batch)` (first eager
  dispatch); `THOR draft-CG gate: N=<n> -> cudagraph` (first CG dispatch);
  `THOR draft-CG gate: error -> defaulting to cudagraph (<exc>)` (on any
  gate evaluation error — logged once, then today's CG behavior).
- **Sentinel:** `THOR_DRAFT_CG_GATE` (helper docstring + dispatch-site
  comment); `verify_patches.py` additionally asserts the dispatch call
  site (a partial apply that lands the helper but never routes through it
  must fail the build).
- **Status:** probe-gated, inert on non-sm_110 (the helper is a structural
  no-op there), default N=1→eager.

## Thor sm_110 GDN prefill enablement (flashinfer side) (2026-09-21)

`thor-gdn-prefill-fi-sm110.patch` — a **one-line hunk in the installed
flashinfer package** (a different site-packages tree from vllm) that
widens the GDN prefill dispatch allowlist so the SM100 CuTe-DSL kernel is
reachable on sm_110. This is the *enablement* half of the GDN prefill
probe; the vLLM-side grant (`thor-gdn-prefill-sm110`) stays exactly as
before (probe-gated), so the hunk alone changes no behavior on any
device.

- **The wall it opens** (`flashinfer/gdn_prefill.py`, flashinfer
  0.6.18.post1): `chunk_gated_delta_rule`'s non-CP dispatch is
  `if _arch_major == 10 / elif 12 / elif 9 / else` (L460/L519/L546) —
  major 11 falls into the final `else` and raises
  `NotImplementedError("GDN prefill DSL kernel is unavailable")` (L576).
  The kernel behind the SM100 branch (`gdn_kernels/blackwell/
  gated_delta_net_chunked.py`) is **device-JIT-compiled CuTe-DSL**
  (FA4-type wall, not the B12x type): its `cute.compile` carries no
  `GPUArch` option, so the DSL targets the current device; its single
  sm_100-specific construct is the `arch = "sm_100"` class attribute,
  used only as a TMEM-column sizing lookup
  (`cute.arch.get_max_tmem_alloc_cols`, L193/L280). The CP kernels by
  contrast are arch-pinned (the SM100 CP prefill hard-rejects
  `major != 10`; the SM90/SM120 delta-rule DSL kernels pin
  `GPUArch("sm_90a")` / `sm_12xa`), which is why the CP heuristic
  (`_arch_major in (9, 10, 12)`, L359) is deliberately **not** widened —
  sm_110 simply never routes to CP and falls through to the non-CP
  branch.
- **The hunk** (`gdn_prefill.py:460`, one line, sentinel comment on the
  changed line):
  `if _arch_major == 10:` →
  `if _arch_major in (10, 11):  # THOR_SM110_GDN_ALLOWLIST`.
- **Mechanism change** (`apply_patches.py` / `verify_patches.py`):
  `FLASHINFER_PATCH_ORDER` applies the hunk to the installed flashinfer
  package (`flashinfer_package_root()`, `importlib`-resolved, same
  `patch -p1 --forward` semantics as the vllm root);
  `verify_patches.py` gains a parallel `FLASHINFER_FIXES` table checked
  by `verify_flashinfer()`. Both scripts take an optional explicit root
  (vllm root arg 1, flashinfer root arg 2) — the verdict is about the
  given trees, matching the existing explicit-root convention. The
  Dockerfile is unchanged: it already runs both scripts against the
  installed packages.
- **Verification:** no-GPU fresh-tree gate (apply 12+1, verify 12+1
  fixes, py_compile, import smoke, canary T10) and the live T6 canary —
  see `task/nvfp4-thor/gdn-prefill-fi-enable.md`.

## Thor hd256 1CTA decode carve-out (v11, 2026-09-24)

`thor-fa4-hd256-1cta-decode-sm110.patch` — the "N-2 decode" workstream: a
decode carve-out in the FA4 hd256 kernel's interface that drops the 2CTA
cluster to a single CTA for decode-shaped calls. One hunk in
`vllm/vllm_flash_attn/cute/interface.py` (`_flash_attn_fwd`, the
`hd256_use_2cta` site); the kernel file is untouched. v11 = v10 + this patch
only — the live tree's inert SplitKV kernel plumbing is deliberately not
shipped (see below).

- **What it does.** The kernel's `use_2cta` flag (a compile-key slot, so the
  two cluster forms are separate cached binaries) is now driven by an OR'd
  decode condition:

      # decode-shaped hd256 calls (small M) take the 1CTA form
      hd256_decode_1cta = (
          max_seqlen_q is not None
          and max_seqlen_q <= 8                       # decode M (1..8); prefill is 1024+
          and not local
          and (page_size in (None, tile_n))           # dense or paged-128 TMA (only legal hd256 paged mode)
      )

  i.e. decode M (1..8 — covers the MTP verify batch; prefill is 1024+),
  non-local, and dense or paged-128 TMA (the hd256 kernel's only legal
  paged mode — its own `paged_kv_non_tma` assert makes page-128-or-dense
  the only legal KV layout, so the last term is a defensive tautology).
  The 1CTA form (cluster (1,1), `CtaGroup.ONE`, shallower k/v rings
  2/3 vs 4/4) has roughly half the 2CTA per-KV-token scan slope (0.016 vs
  0.032 µs/tok on the 20-SM Thor), so the win grows with L. Prefill
  (2CTA) and decode (1CTA) coexist in one process under separate compile
  keys; a serving-like prefill→decode compile sequence was verified to
  route decode to the 1CTA binary (compile key `(varlen_b1, l2_swizzle,
  mask_residual, use_2cta) = (True, False, True, False)`). Like the rest
  of the fa4 group this only runs on the FA4 hd256 path, which is still
  passive-by-default on sm_110 (default FA version 2), so no existing
  config changes behavior.
- **Evidence (clean micro-bench, server 0/0-gated, 2026-09-24).** L=8192,
  M=1: 2CTA 323.0 µs → 1CTA 207.1 µs median — a **1.56× decode win**
  (M=4: 1.55×); the ratio grows 1.20× (L=256) → 1.56× (L=8192) for M=1.
  The clean 2CTA number reproduces the prior clean 2CTA (324.1 µs) to
  0.3%. Correctness: full-FP8 A/B 12/12 (probe2a-fp8 — all prefill shapes
  stay on the unchanged 2CTA path, all decode shapes run the 1CTA binary);
  1CTA == 2CTA == fp32-ref within fp8 tolerance (1CTA-vs-2CTA max diff
  9.77e-4 @ L=2048 / 4.88e-4 @ L=8192 vs tolerances 1.52e-2 / 1.35e-2 —
  both cluster forms equally correct; the residual is fp8/softmax
  numerics, not a cluster-form difference). Depth:
  `docs/fa4-hd256-fp8/decode-kernel-feasibility.md` (DK-0 go/no-go),
  `dk1a-1cta-paged.md` (DK-1a trigger + production wiring),
  `decode-1cta-clean.md` (DK-1/DK-2 clean validation),
  `v11-1cta-patch.md` (patch provenance).
- **Status.** Built into the v11 image (`mjolnir/vllm-thor:
  qwen38-sm110-v11`, same pinned base digest, no Dockerfile change). Full
  v11 flow simulation on the expanded v10 site-packages: all v10 patches
  idempotently skipped, this patch applied cleanly, gate **`all 11 fixes
  verified`** (+ `all 1 flashinfer fixes verified`), and the resulting
  `interface.py` is **byte-identical** to the live 1CTA tree the clean
  micro-bench measured (`cmp` clean). No `verify_patches.py` sentinel was
  added — the hunk re-wraps an existing expression (adding only the
  `hd256_decode_1cta` local), and the gate verifies the existing fixes,
  which all remain present (`v11-1cta-patch.md` §Registration).
- **Residual gap.** 1CTA is still ~2.9× FlashInfer at L=8192, M=1 (207.1
  µs vs the isolated-clean FI reference 72.5 µs; 2.25× in-window) — the
  remaining lever is a FlashInfer-inspired GEMV-style decode kernel (the
  active next task in `AGENTS.md`), not more cluster/split tuning.

### hd256 SplitKV — investigated, no win (kept inert)

The hd256 kernel's SplitKV machinery was benchmarked on **both** cluster
forms and found **not** to be a performance win, so it is kept inert in
the tree: the v11 patch ships the 1CTA carve-out only (the live tree's
A1a/A1b kernel plumbing — `num_splits` param, per-split gO/LSE staging,
FP8-descale-correct LSE, test-only `A1B_TEST_NUM_SPLITS` knob — is
deliberately excluded from v11, and the shipped interface never enables
`num_splits > 1` on hd256; the machinery stays in the live tree as a
correctness-verified harness for future kernel work):

- **2CTA path** (`docs/fa4-hd256-fp8/splitkv-sweep.md`, 2026-09-23): ns>1
  is slower at every L — 40/40 cells (L∈{256…8192} × M∈{1,4} ×
  ns∈{2,4,8,16}) above serial; the best case (ns=2, L=8192) is still
  1.24× *slower*, and the per-split penalty shows no amortization knee as
  L grows.
- **1CTA path** (`docs/fa4-hd256-fp8/real-splitkv-bench.md`, 2026-09-24):
  real SplitKV was enabled on the 1CTA path (num_splits clamp lifted,
  per-ns compile key, `num_splits` as a constructor constant — a trailing
  FFI argument breaks CuTe DSL IR verification) and verified correct
  (every leg within fp8 tolerance; per-split LSE + combine match the fp32
  logsumexp to ~1e-6; empty-split paths exercised) — but still no win:
  clear loss at L=2048/4096 (ns=2 +18.6% / +6.4%), and at L=8192 at most a
  marginal ns=2 "win" inside the inter-run noise band (−1.5…−2.4% across
  four measurements), with ns=4/ns=8 clear losses.
- **Why (confirmed on both paths):** the per-CTA 512-tmem/MMA machinery
  cost is replicated ns× over shrinking KV chunks — splitting trades
  per-KV-token work for ns× the fixed per-CTA cost, so it never pays on
  either cluster form. The shipped decode optimization is the 1CTA
  carve-out (this section); the real long-context SplitKV win is the
  FlashInfer path (1.78–2.8× faster than FA4 1CTA serial at L=8192),
  which is also the reference architecture for the next GEMV-style kernel.
  — *superseded as the decode answer by the GEMV kernel packaged below.*

## Thor hd256 GEMV decode kernel (2026-09-25) — `patches/thor-fa4-hd256-gemv-decode-sm110.patch`, default-on

The "N-2 decode" completion: a **build patch**
(`patches/thor-fa4-hd256-gemv-decode-sm110.patch`) applied by
`apply_patches.py` (lands after the 1CTA-decode patch; both touch
`interface.py` in disjoint regions). The `fa4-gemv-kernel/` package remains
the shippable source of truth for the kernel + bench docs. The **patch edits
only the existing `interface.py`** (the 4 dispatch hunks, rebased onto the
current interface); the **kernel `.py` is installed by a separate `COPY`** in
the `Dockerfile` (from `fa4-gemv-kernel/sm100_hd256_decode_gemv.py`) — a NEW
file upstream has none of, so it is not a patch hunk.
The working copies in `docs/fa4-hd256-fp8/` remain the measurement record.

- **`sm100_hd256_decode_gemv.py`** (separate `COPY` in the `Dockerfile`, from
  `fa4-gemv-kernel/`) → installs at
  `vllm/vllm_flash_attn/cute/sm100_hd256_decode_gemv.py`
  (`from vllm.vllm_flash_attn.cute.sm100_hd256_decode_gemv import
  BlackwellHd256DecodeGEMV`). CuTe-DSL pure-FMA GEMV decode: M=1, hd=256,
  GQA, dense **or** paged (16/128), bf16/fp16/fp8-e4m3 Q/KV + per-(batch,
  kv_head) descales, SplitKV via the in-tree `_flash_attn_fwd_combine`
  (chunk-normalized partials + LSE ⊕-merge; write-through at ns=1),
  cp.async KV ring (S=16 stages — the ring-depth fix is correctness-neutral,
  see `fa4-gemv-kernel/docs/gemv-ring-fix-bench.md`).
- **The 4 `interface.py` hunks** — (1) the `BlackwellHd256DecodeGEMV` import,
  (2) `_gemv_auto_num_splits` (FlashInfer Alg.1 split plan: fill `4 × #SM`
  CTAs), (3) the `_gemv_dtype_ok` mixed-Q/KV-dtype allowance (GEMV on, M=1
  hd256 only) + the relaxed dtype assert, (4) the `use_gemv_hd256` dispatch
  block (scope gates, `gemv_ns` precedence env-knob > caller > auto, 128-row
  floor + 64 cap, compile cache, write-through vs in-tree combine) inserted
  before the `tcgen05`/1CTA forward so in-scope M=1 hd256 calls short-circuit
   to the GEMV kernel and return before it. Round-trip verified: applying the
   patch to the current interface reproduces the measured interface
   byte-exactly (the kernel file is installed by the separate `COPY`, not the
   patch); the patched interface `py_compile`s.
- **Knobs:** **DEFAULT-ON** — the dispatch is active unless
  `VLLM_FA4_HD256_GEMV=0` (opt out); `VLLM_FA4_HD256_GEMV_NUM_SPLITS=N`
  overrides the split plan; auto at L=8192 GQA-4 → ns=20 → 80 CTAs.
  Because it is default-on, this is the one fa4 patch that changes in-scope
  M=1 hd256 decode behavior out of the box (it still only runs when FA4 hd256
  is actually selected, which on sm_110 stays passive-by-default, so no
  existing FA2 config is affected).
- **Results** (gated clean window, L=8192 M=1 GQA 24/4, bf16 Q + fp8 KV):
  GEMV auto/ns=20 **222.8 µs** vs FlashInfer FA2-tc **202.05 µs** in the same
  window → within ~10% of FI, vs FA4-1CTA's 207 µs clean reference; both
  window numbers depressed by ungatable desktop Xorg co-tenancy (same FI
  kernel: 73 µs fully-clean Sept 22). Kernel achieved BW (ncu, L2-fabric):
  ns 1→20 = ×8.7 (13.5→118.3 GB/s; roofline 273 GB/s) — the SplitKV/CTA-count
  lever; stages 2→16 = +12.9% at ns=20 only, wall wash at L=8192.
- **Methodology** (reproducible, in `fa4-gemv-kernel/README.md` §Methodology):
  clean-window gate on the live vLLM server's `num_requests_running/waiting`
  (consecutive 0/0 samples before timing; never stop/restart the server),
  ncu achieved-BW preferred over wall-clock under desktop co-tenancy,
  same-window relative ratios (ns-vs-ns, st16-vs-st2 via compile-cache swap),
  two-lever decomposition (CTA-count × in-flight depth), null results
  documented (not LDGSTS-vs-LDG — both lower to plain LDG on sm_110a; not
  occupancy).

## Bump to vLLM 0.30.0 (2026-09-26) — two docstring-style hunks re-adapted

Base moved from `0.29.1rc1.dev452+g3df4ae153` (nightly-aarch64, 2026-09-21)
to the **release** `vllm 0.30.0`
(`vllm/vllm-openai:v0.30.0-ubuntu2404`, digest
`sha256:439c19d48db36401abc914b9842d060fe610a54bc3ac29bb02505f0b69af1baf`;
torch 2.13.0+cu130, flashinfer 0.6.18.post1 unchanged). ~1203 of 2712 vllm
`.py` files differ between the two bases.

Only **two hunks** needed re-adapting, both in `v1/core/kv_cache_utils.py`,
and both were **context-only** breaks from an upstream repo-wide docstring
reformat (no semantic drift in the touched regions):

- **55390 hunk #3** (the `use_deepseek_v4_fallback` →
  `use_trailing_layer_fallback` docstring rename): 0.30.0 re-indented the
  `_annotate_eagle_groups` docstring `Args:` block from 4/8 to **8/12 spaces**
  and dropped the trailing blank line before the closing `"""`. Context
  re-adapted to the new indentation; lands with fuzz 1; post-patch docstring
  verified byte-exact against the intended state.
- **55519 hunk #1** (the warning gate `use_eagle()` →
  `use_eagle_block_drop()` + docstring paragraph): 0.30.0 dropped the
  trailing blank line in `_warn_if_unannotated_eagle_mamba`'s docstring.
  Context re-adapted; lands clean.

  (Both functions' *code* is unchanged across bases: `_annotate_eagle_groups`
  already gates on `use_eagle_block_drop()` in both, so the semantic intent of
  the two patches is exactly preserved.)

Everything else applied verbatim: 54165/55519-scheduler at line offsets only;
50885 hunk #11 with fuzz 2 (as on the previous bump); all three
`flashinfer.py`-touching patches, the two fa_utils patches, the hd256 forward
kernel patch, the `interface.py` 1CTA + GEMV hunks, the dflash speculator
gate, and the flashinfer GDN hunk clean.

**FA4 side note:** the 0.30.0 vendored `vllm_flash_attn/cute/` tree is
**pre-#2916/#2917** (flash-attention's hd256 2CTA SplitKV, merged to FA
`main` 2026-09-25, after the 0.30.0 cut — the vendored kernel still asserts
`not is_split_kv`), so the fa4 patch group is unaffected; see
`docs/research/fa4-hd256-upstream-status.md` (2026-09-26 update) for the
impact analysis (auto-SplitKV disables itself on the 20-SM Thor by the
upstream heuristic's own arithmetic, corroborating the in-tree NO-WIN).

Build gates (no-GPU build on Thor; the two kernel tests self-skip, T10/T1
execute): `all 12 vllm fixes verified` + `all 1 flashinfer fix verified`,
compileall 7 OK, import smoke `vllm 0.30.0`, functional check 55390/55519
**ALL PASS**, dspark non-causal **T1 gate PASS**. Image:
`mjolnir/vllm-thor:qwen38-sm110-v13`
(`sha256:ce760f9ad00e87c6fa67b6fbe85273c5ba03a1cec460eb29977b7fd3ed488cc6`).
GPU canaries (T6 GDN prefill / T9 FA4 FP8-KV / kernel tests) still need a
`--gpus all` re-run on a clean window before the new image is served.
- Docs: `fa4-gemv-kernel/README.md` (install/knobs/results/how-to-run),
  `fa4-gemv-kernel/docs/gemv-decode-design.md` (kernel design),
  `docs/gemv-fi-gap-final.md` (ncu gap analysis vs FI),
  `docs/gemv-ring-fix-bench.md` + `.json` (final gated numbers).
