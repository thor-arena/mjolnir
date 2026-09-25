# FA4 hd256 FP8: formal v11 patch (1CTA decode carve-out)

## Goal

Promote the in-place "deeper A" decode win (1CTA decode carve-out, applied during
investigation to the live tree `<vfa-tree>/`) into a formal v11 build
patch, in the same format/order as the v10 patch set consumed by
`docker/vllm-thor/apply_patches.py`. v11 = v10 + 1CTA decode only; the inert
SplitKV plumbing stays out (minimal, low-risk).

## Deliverable

- `docker/vllm-thor/patches/thor-fa4-hd256-1cta-decode-sm110.patch` — the 1CTA
  decode carve-out ONLY (`vllm/vllm_flash_attn/cute/interface.py`, one hunk at
  line 1034 in `_flash_attn_fwd`), formatted to match the v10 patch style
  (`diff --git` + `index <sha>..<sha> 100644` + `a/vllm/...`/`b/vllm/...` paths).
  Blob SHAs in the `index` line are real: `91c1ffb..3819188` (git hash-object of
  the clean v10-base and live interface.py).

## v10 patch format (what the new patch had to match)

- `git diff`/`diff --git` style: per-file `diff --git a/vllm/<p> b/vllm/<p>` +
  `index <sha>..<sha> 100644` + `--- a/vllm/<p>` / `+++ b/vllm/<p>` + standard
  `@@ -l,c +l,c @@ <func>` hunks.
- Paths are relative to site-packages (vllm package root = `vllm/`), so
  `vllm_flash_attn` files carry the `vllm/vllm_flash_attn/` prefix.
- Applied by `apply_patches.py` with `patch -p1 --forward` (patch(1), NOT
  `git apply`), cwd = site-packages (= `root.parent`); `-p1` strips `a/`.
  rc 0 = applied, rc 1 = hunks skipped/already present (OK), rc > 1 = build fail.
  After application `verify_patches.verify` + `verify_flashinfer` is the
  authoritative sentinel gate (missing fix => exit 1).
- Base: pinned nightly `vllm/vllm-openai@sha256:a17c15e3...` (vllm
  0.29.1rc1.dev452+g3df4ae153, 2026-09-21). Image tag is chosen at build time:
  `docker build -t mjolnir/vllm-thor:qwen38-sm110-vNN docker/vllm-thor/`
  (v10 tag = `...-v10`). No version constant in the Dockerfile.
- Build order (PATCH_ORDER): 54165, 49652, 50885, 55390, 55519,
  thor-gdn-prefill-sm110, thor-fused-draft-decode-fi-native-update,
  thor-dspark-draft-fi-noncausal-cudagraph, thor-fa4-fp8kv-sm110,
  thor-fa4-hd256-fp8-sm110, thor-draft-cg-gate-sm110; plus flashinfer-side
  thor-gdn-prefill-fi-sm110. The hd256 patch must land after fa4-fp8kv
  (fa_utils.py); the new 1cta patch touches only `cute/interface.py` at a
  region disjoint from the fp8kv interface hunk (line ~1034 vs ~907), so it
  is order-independent — placed after the hd256 patch to keep the fa4 group
  together.

## Live-tree inventory (clean v10 base -> <vfa-tree>/)

Clean v10 base taken from the v10 image
(`mjolnir/vllm-thor:qwen38-sm110-v10`,
`site-packages/vllm/vllm_flash_attn`). Recursive diff (`diff -rq`, excluding
`__pycache__`/`.orig`/`.diff`) shows exactly two differing files:

1. `cute/interface.py` — **the 1CTA decode carve-out (the win)**: new
   `hd256_decode_1cta` gate (`max_seqlen_q <= 8`, not local, dense or
   paged-128 TMA) OR'd into `hd256_use_2cta` so decode-shaped hd256 calls take
   the 1CTA form. => this is the v11 patch.
2. `cute/sm100_hd256_2cta_fmha_forward.py` — **inert SplitKV plumbing
   (A1a/A1b)**: `num_splits` param + `is_split_kv` storage (categorical assert
   removed, replaced by varlen-path-only scheduler asserts), `_split_kv_clamp`
   KV-bounds narrowing in the load/MMA/softmax/correction loops, per-split gO
   + LSE staging (LSE b-mode = split axis), FP8 descale-correct LSE
   (`lse_max_offset_ln2`, removed the old "no LSE with descales" assert), and
   the A1b TEST-ONLY `A1B_TEST_NUM_SPLITS` env knob. Inert: the interface never
   passes `num_splits > 1`/`is_split_kv=True`, so all of it is behind
   `cutlass.const_expr(is_split_kv)` gates => excluded from v11 on purpose.

No other files in the live tree differ from the v10 base.

## Verification

1. `patch -p1 --forward --dry-run` (the exact command `apply_one` runs), cwd =
   a tree with the clean v10-base `vllm/vllm_flash_attn/cute/interface.py`:
   rc 0, "checking file vllm/vllm_flash_attn/cute/interface.py", no conflicts.
2. Applying for real: result byte-identical to the live
   `<vfa-tree>/cute/interface.py` (`cmp` clean).
3. `git apply --check` against a repo containing the v10-base blob: rc 0
   (also validates the `index` SHAs).
4. Full v11 flow simulation: a copy of `apply_patches.py` with the new patch
   appended to PATCH_ORDER, run against the full extracted v10 site-packages
   (vllm + flashinfer): all v10 patches idempotently skipped, the new patch
   applied cleanly ("patching file vllm/vllm_flash_attn/cute/interface.py",
   no fuzz/rejects), `verify_patches` gate passed ("all 11 fixes verified",
   "all 1 flashinfer fixes verified"), exit 0; interface.py in the result is
   byte-identical to the live 1CTA tree.
   - Simulation-only artifact (NOT a v11 concern): re-running
     `thor-fa4-fp8kv-sm110.patch`'s fa_utils probe-function hunk on an
     already-patched tree re-inserted the `_thor_fa4_fp8kv_probe` function once
     more with fuzz (the hd256 patch, landing after fp8kv, changed that hunk's
     trailing context, defeating patch's "previously applied" detection). In
     real builds the patch set is applied once, in order, on a clean nightly
     base — the v10 image itself is consistent — so this cannot occur for v11.

## Registration (stated, not applied)

- `docker/vllm-thor/apply_patches.py`, PATCH_ORDER — add one line after
  `"thor-fa4-hd256-fp8-sm110.patch",`:
  `    "thor-fa4-hd256-1cta-decode-sm110.patch",`
- `verify_patches.py` needs no change (no interface.py sentinels exist there;
  the gate only checks existing fixes, which all remain present).
- Build the v11 image: `docker build -t mjolnir/vllm-thor:qwen38-sm110-v11
  docker/vllm-thor/` (same pinned base digest; no Dockerfile change required).
- Optional doc upkeep (not required for the build): the Dockerfile header
  comment lists the Thor patches, and PATCHES.md carries the per-patch
  inventory — add a `thor-fa4-hd256-1cta-decode-sm110` entry there.

## Status

COMPLETE — patch written, verified clean; registration left to the user.
