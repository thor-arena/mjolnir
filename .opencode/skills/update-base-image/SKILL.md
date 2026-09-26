---
name: update-base-image
description: Bumps the pinned vllm/vllm-openai:nightly-aarch64 base image of the docker/vllm-thor overlay — new digest, patch-stack re-verification, canary gates, default-tag switch. Use when a new vLLM nightly is published, or when the patch stack must be re-validated against a newer base.
compatibility: opencode
---

# Update the base image (vllm-thor overlay)

The overlay in `docker/vllm-thor/` is a thin layer on
`vllm/vllm-openai:nightly-aarch64`. Bumping the base is a **gated process**: the
image build itself is the gate — it applies the 14 patches
(`apply_patches.py`) and verifies them (`verify_patches.py`), and **fails** if a
patch stops applying. A build failure means the base moved past that patch's
assumptions; the fix is to hand re-adapt the patch, not to weaken the gate.

Background: `docs/thor-stack/nightly-bump-process.md` (worked example with
logs). The Dockerfile's "BUMPING TO A NEWER NIGHTLY" section is the short form.

## Procedure

### 1. Fetch the new base + digest (on the aarch64 box)
```bash
docker pull vllm/vllm-openai:nightly-aarch64
docker image inspect vllm/vllm-openai:nightly-aarch64 --format '{{index .RepoDigests 0}}'
# vLLM version inside:
docker run --rm vllm/vllm-openai:nightly-aarch64 python -c 'import vllm; print(vllm.__version__)'
```

### 2. Point the Dockerfile at it
`docker/vllm-thor/Dockerfile`:
- `ARG BASE_IMAGE_DIGEST=vllm/vllm-openai@sha256:<new-digest>` (the floating
  `ARG BASE_IMAGE=vllm/vllm-openai:nightly-aarch64` stays untouched).
- Update the "Verified against base:" header block (vLLM version,
  flashinfer-python/cubin, torch).

### 3. Build — the patch gate
```bash
mjolnir image build --tag mjolnir/vllm-thor:qwen38-sm110-v<N>
```
- Exit 0 → all 14 patches applied + verified. Continue.
- **Failure → a patch drifted.** Read the build log, hand re-adapt that
  patch in `docker/vllm-thor/patches/` (never a raw `git apply` of upstream
  diffs), keep its `apply_patches.py` / `verify_patches.py` / `PATCHES.md`
  entries current, and re-run the build. Log what changed in
  `PATCHES.md` for that patch.

### 4. Canaries (fresh container — no server needed)
```bash
mjolnir image gates                        # sm_110 gate probes: T6 GDN, T9 FA4, T10 draft-CG, T11 hd256
mjolnir bench kernel functional            # no-GPU functional check
```
Every gate must PASS before the tag is considered real.

### 5. Serve + gated e2e smoke (needs the GPU — clean-window rules apply)
- **Never** stop/restart the live vLLM server to make room (the GPU is shared —
  `AGENTS.md`, critical rules). If the bump must be validated end-to-end,
  schedule it in a clean window via `mjolnir gate`, or use `--dry-run` to
  inspect the command first:
```bash
mjolnir serve up --image mjolnir/vllm-thor:qwen38-sm110-v<N> --dry-run
mjolnir gate --once
mjolnir bench perf --image mjolnir/vllm-thor:qwen38-sm110-v<N> --runs 2 --repeat 1
```
- Compare the smoke sweep against recent rows in `benchmarks/history.jsonl`
  (`mjolnir history`). A bump that shifts numbers starts a **new comparison
  epoch** — old rows keep their old image tag; don't mix them in charts.

### 6. Switch the default
- `src/mjolnir/config.py`: `DEFAULT_IMAGE = "mjolnir/vllm-thor:qwen38-sm110-v<N>"`.
- Update the tag references in `README.md` and the Current-state section of
  `AGENTS.md`.

### 7. Record + commit
- `AGENTS.md` current state: one line — date, old→new base (dev tags), which
  patches needed re-adaptation, canary result.
- If any patch was re-adapted: `docs/thor-stack/nightly-bump-<YYYY-MM-DD>.md`
  (old→new base, per-patch deltas, gate outputs — mirror
  `docs/thor-stack/nightly-bump-process.md` style).
- Commits grouped by concern: patch re-adaptation(s) / Dockerfile digest /
  default + docs.

## Hard rules
- The gate is `verify_patches.py` inside the build — never bypass it
  (`--no-verify` style shortcuts don't exist; don't add them).
- Patch re-adaptation is hand-edited inside `docker/vllm-thor/patches/`,
  registered in `apply_patches.py` `PATCH_ORDER`, with a `verify_patches.py`
  entry and a `PATCHES.md` entry (the probe-gated pattern is the norm).
- Any GPU measurement goes through the clean-window gate
  (`src/mjolnir/gate.py`); wall-clock comparisons only in the same gated
  window (see `docs/methodology/benchmarking.md`).
