# Upgrading the base image and re-adapting the patch stack

**Story:** "A new `vllm/vllm-openai` release/nightly is out. I want to move
the image base to it, and when the 14-patch stack stops applying, I want to
know *which* hunks are invalidated and how to re-adapt them."

## Why the build fails on purpose

`docker/vllm-thor/` is a **thin overlay**: the prebuilt aarch64 wheel in the
base image carries the sm_110a kernels; on top, `apply_patches.py`
hand-adapts and applies the 14 patches, and `verify_patches.py` asserts every
fix is present. The `RUN apply_patches.py && verify_patches.py` step in the
Dockerfile **fails the build** if a patch no longer applies or a fix is
missing — that failure *is* the invalidation signal: the base moved past
the assumptions a patch was written against.

## The bump, step by step

```bash
# 1. Pull the new base and note its version + digest
docker pull --platform linux/arm64 vllm/vllm-openai:v0.31.0-ubuntu2404
docker image inspect vllm/vllm-openai:v0.31.0-ubuntu2404 \
       --format '{{index .RepoDigests 0}}'

# 2. Point the Dockerfile at it (ARG BASE_IMAGE + BASE_IMAGE_DIGEST,
#    the "Verified against base" comment, the image label)

# 3. Build the candidate tag — the build is the gate
mjolnir image build --tag mjolnir/vllm-thor:qwen38-sm110-v14
```

Outcomes of step 3:

- **Green** (patches applied, maybe with line offsets — benign;
  `--forward` skips hunks the base already merged upstream): run the canary,
  tag it canonical, keep the old tag for A/B.
  ```bash
  mjolnir image gates --image mjolnir/vllm-thor:qwen38-sm110-v14
  ```
- **Red** (a patch rejects / a fix is missing): don't force it. Re-adapt the
  invalidated hunk(s) — below — then rebuild.

## Re-adapting invalidated hunks

Each patch in `patches/` is a **hand-maintained diff** (not a raw `git
apply` of upstream PRs), so "re-adapt" = edit the patch file to match the
new base:

1. **Reproduce without the build.** Inside a fresh container of the *new*
   base, run the applier directly — it stops at the first rejected patch and
   tells you the file + hunk:
   ```bash
   docker run --rm -it --entrypoint bash vllm/vllm-openai:v0.31.0-ubuntu2404
   # inside: locate site-packages/vllm (importlib), then
   cp /path/from/host/docker/vllm-thor/patches/… /tmp/
   patch -p1 --forward -i /tmp/<patch>.patch
   # rc=0 applied · rc=1 some hunks already present (ok) · rc>1 REJECTED
   ```
   The `.rej` file is the ground truth of what didn't match.
2. **Diff the context, not the lines.** Open the current file in the new
   base next to the hunk's expected context. Typical drift classes
   (see `PATCHES.md`): line offsets (benign, usually auto), signatures that
   grew, a concept superseded upstream, or a hunk the base already did
   (the fix is *already there* — re-validate and keep skipping).
3. **Edit the hunk in `patches/<name>.patch`** (context lines + line counts)
   until `patch -p1 --forward` applies it clean on the new base. If a file
   moved, update the path in the patch **and** the Dockerfile smoke-test
   `touched` list; if the patch is now absorbed upstream, keep the
   `verify_patches.py` entry or consciously drop both (a drop stays a drop —
   re-validate why).
4. **Re-run the full gate:** `apply_patches.py` (all 14, in `PATCH_ORDER`,
   plus the flashinfer root) → `verify_patches.py` all fixes present.
5. **Close the loop:** `mjolnir image build --tag …` → `mjolnir image gates`
   (GPU canary: GDN prefill, FA4 FP8-KV, draft-CG, hd256) → `mjolnir image
   use <tag>`.

## Record it

Every bump that needed re-adaptation gets: a `PATCHES.md` note on the
affected patch, a depth doc under `docs/thor-stack/` (per-patch outcomes +
canary transcript — worked examples:
[`base-bump-2026-09-26.md`](../thor-stack/base-bump-2026-09-26.md),
[`nightly-bump-process.md`](../thor-stack/nightly-bump-process.md)), and a
commit that groups the Dockerfile + patch + doc changes. The process
reference is in the Dockerfile's "BUMPING TO A NEWER vLLM" block.
