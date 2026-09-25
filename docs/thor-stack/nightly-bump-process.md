# Bumping the vLLM Nightly Base

> **Status:** Process reference — worked example verified 2026-09-21 (dev347 → dev452) · **Date:** 2026-09-21 · **Scope:** how to move `docker/vllm-thor/` to a newer `vllm/vllm-openai:nightly-aarch64` base, what can break, and the gates that prove the 13-patch stack still holds

This document is the narrative behind the "BUMPING TO A NEWER NIGHTLY" comment block in [`docker/vllm-thor/Dockerfile`](../../docker/vllm-thor/Dockerfile). The build itself is the authoritative gate: it **fails** if a patch no longer applies or a verified fix is missing after application — that is the signal that the nightly moved past the assumptions one of the patches was written against, and that patch needs re-adapting.

---

## TL;DR

- A bump is: pull the nightly, compare the vllm/flashinfer/torch versions, update `BASE_IMAGE_DIGEST`, rebuild. The patch stack either applies (with at worst line offsets) or a hunk rejects and must be re-adapted.
- Patch application is idempotent (`patch -p1 --forward`): a fix the base image already merged upstream is skipped, not re-applied.
- The usual drift is benign — line offsets (e.g. 55519's known 20-line offset). The dangerous drifts are file moves, superseded concepts, and newly-merged upstream fixes.
- Two layers of verification: no-GPU build gates (apply+verify, functional check, compile+import smoke) and the GPU canary (`kernel_test_sm110_gates.py` in a fresh `--gpus all` container).
- Worked example (2026-09-21): dev347 → dev452 (105-commit delta) — all 10 vllm + 1 flashinfer patches applied clean, **0 rejects**, no hunk re-adaptation needed; v9 became the canonical image.

---

## 1. What a Bump Is

The overlay builds on a pinned digest of the public `vllm/vllm-openai:nightly-aarch64` image. Bumping means moving to the current nightly and re-proving the stack:

1. `docker pull vllm/vllm-openai:nightly-aarch64`
2. `docker image inspect vllm/vllm-openai:nightly-aarch64 --format '{{index .RepoDigests 0}}'`
3. Update the default `BASE_IMAGE_DIGEST` in `docker/vllm-thor/Dockerfile` (keep the floating tag too), and the "Verified against base" comment + `org.opencontainers.image.description` label.
4. `docker build -t <tag> docker/vllm-thor/`

## 2. What Can Break

The 13 patches are hand-adapted (not raw `git apply` of upstream diffs), so the failure modes are:

- **Line offsets** — hunks apply at a different line number. Benign; `apply_patches.py` reports `Hunk #N succeeded at L (offset N lines)`.
- **Hunk rejects** — the context the hunk was written against no longer exists. The patch must be re-adapted against the new base (see [docker/vllm-thor/PATCHES.md](../../docker/vllm-thor/PATCHES.md), "Why 'hand-adapted' and not the raw diff", for the drift classes: concepts superseded upstream, signatures that grew, hunks already done by the nightly).
- **File moves** — a patch targets a path that moved (e.g. `spec_decode/autoregressive/cudagraph_utils.py` → `v1/worker/gpu/cudagraph_utils.py`). The patch path and any Dockerfile smoke-test `touched` list must be updated.
- **Superseded patches** — an upstream change absorbs what a patch did (or the patch's original sentinels disappear by design). Such a patch is re-validated (drop stays a drop) rather than blindly re-applied.
- **Dependency bumps inside the nightly** (flashinfer/torch) — a second patch root (flashinfer) must be re-checked; the functional check and canary re-run.

## 3. Verification Gates

**No-GPU (the build itself, `docker build` without `--gpus`):**

1. `apply_patches.py` against the installed vllm **and** flashinfer packages — all patches apply; all fixes verified in both roots.
2. `verify_patches.py` — authoritative sentinel check of every fix (both roots).
3. `functional_check_55390_55519.py` — functional check of the upstream-ported behavior.
4. `compileall` of patched files + import smoke (reports the resolved vllm version).
5. GPU-only kernel tests self-skip when no CUDA device is visible to the build container.

**GPU (post-build canary, fresh container `--gpus all`):** `kernel_test_sm110_gates.py` — the sm_110 gate-probe suite (GDN prefill T6, B12x T7, XQA/TRTLLM T8, FA4 FP8-KV T9, draft-CG T10; see [docker/vllm-thor/PATCHES.md](../../docker/vllm-thor/PATCHES.md), "Thor sm_110 gate probes").

A bump is complete when all no-GPU gates are green **and** the canary exits 0. The new image becomes canonical; the previous one is kept for A/B reference.

---

## 4. Worked Example — dev347 → dev452 (2026-09-21)

### 4.1 Result: NEWER — bump applied, no hunk re-adaptation needed

| | old base (v8) | new nightly (v9) |
|---|---|---|
| vllm | `0.29.1rc1.dev347+gdee37d891` (2026-09-18) | `0.29.1rc1.dev452+g3df4ae153` (2026-09-21) |
| digest | `sha256:c27ab158…fb90981` | `sha256:a17c15e30254f83adadcf1d9b85db18c59ea1e76a1aca1db8db20c3a12b48ba2` |
| flashinfer-python | 0.6.18 | 0.6.18.post1 |
| torch | 2.13.0+cu130 | 2.13.0+cu130 (unchanged) |

### 4.2 Patch application (fresh nightly container, `apply_patches.py`)

All 10 vllm + 1 flashinfer patches applied clean — **0 rejects**. Only movement: 55519 two hunks at line offsets (+20 in `kv_cache_utils.py`, +12 in `scheduler.py`). Verbatim tail:

```
[55519-no-warn-when-block-drop-off.patch] Hunk #1 succeeded at 2216 (offset 20 lines).
[55519-no-warn-when-block-drop-off.patch] Hunk #1 succeeded at 296 (offset 12 lines).
[thor-gdn-prefill-fi-sm110.patch] patching file flashinfer/gdn_prefill.py
all 10 fixes verified in /usr/local/lib/python3.12/dist-packages/vllm
all 1 flashinfer fixes verified in /usr/local/lib/python3.12/dist-packages/flashinfer
EXIT=0
```

Notes:

- **49652** lands in `v1/worker/gpu/cudagraph_utils.py` (its patch path) — the file the earlier rebase already targeted; the older `spec_decode/autoregressive/cudagraph_utils.py` still exists alongside. Every file in the Dockerfile smoke-test `touched` list exists in the new base, so no Dockerfile list change was needed.
- **52244-drop re-validated**: the upstream replay-boundaries architecture that superseded it (`get_replay_boundaries`/`reachable_boundaries`, 12 occurrences in `single_type_kv_cache_manager.py`) is present in the new base. (Its original sentinels — `mamba_state_cache_position` etc. — are absent by design; the upstream version is restructured. Same state as dev347: the drop was intentional, 2026-09-18.)

### 4.3 Dockerfile changes

- `BASE_IMAGE_DIGEST` → `a17c15e3…`
- "Verified against base" comment → dev452 / flashinfer 0.6.18.post1
- Header comment block de-staled: was "six vLLM PR fixes … plus two Thor-authored" (pre-cleanup state) — now lists the actual 10+1 stack (5 upstream + 5 Thor vllm + 1 flashinfer), with the 52244 drop noted.
- `org.opencontainers.image.description` label updated to match.

### 4.4 v9 build (plain build, no-GPU gates)

`docker build -t mjolnir/vllm-thor:qwen38-sm110-v9 docker/vllm-thor/` — **rc 0** (build log):

```
all 10 fixes verified in /usr/local/lib/python3.12/dist-packages/vllm
all 1 flashinfer fixes verified in /usr/local/lib/python3.12/dist-packages/flashinfer
compileall: 7 patched files OK
imports OK, vllm 0.29.1rc1.dev452+g3df4ae153
functional check 55390/55519: ALL PASS
kernel test fi-update: SKIPPED (no CUDA device visible to this build container)
PASS T1 gate (non-causal+drafter->UNIFORM_BATCH; no-drafter->SINGLE_TOKEN)
kernel test dspark-noncausal: SKIPPED (no GPU / OOM-saturated)
```

### 4.5 GPU canary (fresh container, `--gpus all`)

`kernel_test_sm110_gates.py` mounted into `mjolnir/vllm-thor:qwen38-sm110-v9`:

```
device capability: (11, 0)
INFO 09-21 19:10:53 [qwen_gdn_linear_attn.py:190] THOR GDN prefill probe: FI kernel enabled on sm_110
T6 A/B re-check: FI vs FLA output+state still match
PASS T6 GDN prefill: FI kernel verified on this device; resolved ('auto', 'flashinfer')
T9 passive-by-default: default config leaves the probe untriggered, FP8-KV gate closed, auto-selection -> FlashInferBackend
INFO 09-21 19:10:57 [fa_utils.py:312] THOR FA4 FP8-KV probe: enabled on sm_110 (maxerr=0.01965 vs fp32 reference)
PASS T9 FA4 FP8-KV: probe enabled on sm_110; gate open for the explicit FA4 config; default config stays passive (FlashInferBackend)
PASS T10 draft-CG gate: pure decision logic (4/4 cases)
kernel test sm110 gates: DONE (exit 0)
```

**All green.**

### 4.6 Verdict

v9 (`sha256:321f8f618b28ce7b6842a2e70c40b68b0f427168bc8c26b3a92e00993968d2c1`, tag `mjolnir/vllm-thor:qwen38-sm110-v9`) is the new canonical image; v8 is superseded (kept for A/B reference). The 105-commit nightly delta broke nothing in the stack — worth benching v9 vs v2 (NVFP4_9) the same way v8 was.
