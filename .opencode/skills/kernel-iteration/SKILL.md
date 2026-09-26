---
name: kernel-iteration
description: Iterates on the FA4 GEMV hd256 decode kernel — vfa-tree workflow, correctness-first (mjolnir verify), clean-window-gated benches, NCU achieved-BW, and the ship path (kernel file → image → docs). Use when changing sm100_hd256_decode_gemv.py or its dispatch, or when producing kernel numbers for the repo.
compatibility: opencode
---

# Iterate on the GEMV decode kernel

The shippable kernel is `docker/vllm-thor/fa4-gemv-kernel/sm100_hd256_decode_gemv.py`
(CuTe-DSL, pure-FMA GEMV for M=1 hd256 decode) + the dispatch carve-out in
`fa4-gemv-kernel/thor-fa4-hd256-gemv-decode-sm110.patch`. Fast iteration happens
on a **vfa tree** (an installed-tree extract of `vllm_flash_attn` with the
GEMV patch applied); the winner is copied back to the repo and rebuilt into
the image. The measurement discipline is normative:
`docs/methodology/benchmarking.md` (every number must follow it).

## Procedure

### 1. Prepare the iteration tree
```bash
mjolnir vfa prepare            # default image; tree at <repo>/vfa-tree/vllm_flash_attn
```
- Edits target `vfa-tree/vllm_flash_attn/cute/sm100_hd256_decode_gemv.py`
  (and `interface.py` for dispatch changes).
- `mjolnir vfa prepare --out <dir>` for parallel trees; `--image <tag>` to
  iterate on a different image (the GEMV-baked image is detected and the copy
  is skipped — the tree already carries the current kernel).
- After an edit, nothing rebuilds: the next `--vfa-tree` bench run picks the
  file up live.

### 2. Correctness first — never skip
```bash
mjolnir verify                  # 3 gated suites: Phase-A 15 + B1 14 + stages 6
mjolnir verify --image <tag>    # verify inside a specific image
```
A kernel change is not a kernel change until it's **GO** on all three suites
(fx64 reference, bf16 + e4m3-descale, dense + paged, L sweep, SplitKV
partially-softmax exactness). Suites gate themselves on the clean window and
preflight the server.

### 3. Gated perf bench (wall-clock, relative only)
```bash
mjolnir bench kernel gemv-bench --mode gemv_dense --out /tmp/r.json
mjolnir bench kernel gemv-ringfix --out /tmp/r.json   # the ns/stages sweep
```
- Canonical shape (from the methodology): L=8192, M=1, GQA 24:4, bf16 Q +
  e4m3 paged KV, head_dim 256, page 16; median over 300 iters, back-to-back
  comparison set (new kernel vs FA4 1-CTA vs FlashInfer) in one gated window.
- Raw JSON → grep the cells; never cite a median that lacks `clean: true`.
- Wall-clock is **relative ratios within the same window only**.

### 4. NCU achieved-BW (absolute numbers)
```bash
mjolnir bench kernel gemv-ncu --out /tmp/r.json   # env knobs: NCU_NS/NCU_STAGES/NCU_L/NCU_ITERS
```
- CC 11.0 has no `dram__bytes.sum`; achieved BW is measured at the L2-fabric
  level: `lts__t_sectors.sum × 32 B / gpu__time_duration.sum` (median of 4
  profiled launches). The 273 GB/s DRAM roofline is reference only.
- Read NCU and wall-clock **together**; disagreements are findings (the
  ring-depth study: +12.9% kernel BW, 0.6% wall — launch/combine overhead).

### 5. Ship the winner
1. Copy the kernel (and any dispatch change) back to
   `docker/vllm-thor/fa4-gemv-kernel/` (kernel file; patch if dispatch changed
   — the patch is the build-time source of truth, see
   `fa4-gemv-kernel/interface-gemv-dispatch.diff` format).
2. `mjolnir image build --tag mjolnir/vllm-thor:qwen38-sm110-v<N>` — build gate
   (14 patches apply + verify).
3. `mjolnir verify --image <vN>` — must be GO inside the image.
4. Update the results tables in `docker/vllm-thor/fa4-gemv-kernel/README.md`
   + the workstream report; record provenance (image tag, gate status, raw
   path).
5. If the default image moved: `src/mjolnir/config.py` `DEFAULT_IMAGE` +
   state-file check (`mjolnir image use`) — per the update-base-image skill.

## Hard rules
- **Never** stop/restart/kill the live vLLM server to "clean" the GPU. The
  clean-window gate (`src/mjolnir/gate.py`) is the only way; import it, never
  re-implement it.
- No prose-only results: every number links to a raw artifact in
  `benchmarks/raw/` (or an NCU table) with image tag + gate status.
- Wall-clock claims are same-window ratios; absolute claims use NCU
  achieved-BW.
- A base bump invalidates old comparisons — new epoch, old numbers keep
  their tag.
