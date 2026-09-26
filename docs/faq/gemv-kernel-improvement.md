# Improving the GEMV decode kernel: methodology, validation, benchmarking

**Story:** "I want to make the FA4 hd256 GEMV decode kernel faster on Thor —
and I want my numbers to be trustworthy: correctness proven, measured under
the clean-window discipline, compared against the right baselines."

## Where the kernel lives (three pieces)

| Piece | Location | How it ships |
|---|---|---|
| Kernel source (CuTe-DSL, pure-FMA) | `docker/vllm-thor/fa4-gemv-kernel/sm100_hd256_decode_gemv.py` | `COPY` into the image (`Dockerfile`), imported as `vllm.vllm_flash_attn.cute.sm100_hd256_decode_gemv` |
| Interface dispatch (import, auto split-plan, dtype gate, `use_gemv_hd256` block) | `patches/thor-fa4-hd256-gemv-decode-sm110.patch` (the authoritative copy; `fa4-gemv-kernel/interface-gemv-dispatch.diff` is the reference cut) | build patch, applied by `apply_patches.py` |
| Knobs | env vars | `VLLM_FA4_HD256_GEMV` (default `1`), `VLLM_FA4_HD256_GEMV_NUM_SPLITS`, `VLLM_FA4_HD256_GEMV_DEBUG` |

Scope: M=1, `head_dim=256`, non-local, GQA any ratio, dense or paged KV,
bf16/fp16/fp8-e4m3 Q+KV (mixed dtypes + descales), SplitKV with in-tree
LSE-merge combine. Design docs:
[`fa4-gemv-kernel/README.md`](../../docker/vllm-thor/fa4-gemv-kernel/README.md)
and [`gemv-decode-design.md`](../fa4-hd256-fp8/gemv-decode-design.md).

## The iteration loop (no image rebuild per change)

```bash
mjolnir vfa prepare                       # → vfa-tree/ = image tree + kernel + dispatch
vim vfa-tree/vllm_flash_attn/cute/sm100_hd256_decode_gemv.py
mjolnir bench kernel gemv-bench --mode all        # gated 4-mode bench against the tree
```

`vfa prepare` copies the *image's* `vllm_flash_attn` out, drops your kernel
file into `cute/`, and applies the dispatch hunks — bench containers mount
that tree over the in-image package, so edits are instant. (For
GEMV-baked images v12+ it detects the dispatch is already in-tree and skips
the patch step.) The tree is the measurement record; the repo's
`fa4-gemv-kernel/` package is the shippable source of truth — keep them in
sync when you ship.

## Methodology (how every number must be produced)

1. **Correctness before speed.** `mjolnir verify` runs the gated suites
   against an fp32 reference — all three must PASS before you look at a
   wall number:
   - `gemv-verify` — Phase A: dense/varlen, mixed dtypes, GEMV on/off (15 cases)
   - `gemv-verify-b1` — SplitKV partials, paged, combine
   - `gemv-stages` — stages 2 vs 16 bit-identical (the ring fix is correctness-neutral)
2. **The clean-window gate.** The Thor GPU is shared with the live server.
   Every bench task waits for `vllm:num_requests_running/waiting == 0/0` for
   N consecutive samples; a dirty sample aborts the sweep. See
   [clean-window-gate.md](clean-window-gate.md). **Never** restart the
   server to get a clean GPU.
3. **Wall-clock for ratios, ncu for absolutes.** The desktop Xorg session
   can't be gated, so absolute bandwidth claims rest on **ncu achieved-BW**
   (`lts__t_sectors.sum × 32 B / gpu__time_duration.sum` — L2-fabric
   convention; `dram__bytes` is n/a on CC 11.0), and wall-clock is only
   used for **relative ratios measured inside the same gated window**
   (co-tenancy cancels).
4. **Decompose two levers, vary one at a time.** CTA count (split plan,
   `ns`) × per-CTA in-flight depth (ring `stages`). The kernel compile cache
   keys on shape/dtype/split/paged — *not* `stages` — which is why the
   stages contrast swaps compiled variants in the cache between blocks.
5. **Raw JSON, grep-able, committed.** `--out` writes the raw result; the
   history log gets the row; no prose-only results.

## Benchmark commands

```bash
# 4-mode: gemv_dense / gemv_paged / fa4_1cta / flashinfer, gated
mjolnir bench kernel gemv-bench --mode all --out /p/b.json

# the strict one: ns sweep + stages contrast + FI baseline, ONE gated
# window (CONFIRM=6 + per-burst re-check) — the headline comparison
mjolnir bench kernel gemv-ringfix --out /p/ring.json
grep -E '"(median|p95|bw_gbs_nominal_kv|clean)"' /p/ring.json
# (the workdir is mounted at /p in the container → the file lands in
#  docker/vllm-thor/fa4-gemv-kernel/ on the host)

# kernel counters under ncu (run inside the image with counter perms)
mjolnir bench kernel gemv-ncu          # NCU_L / NCU_NS / NCU_STAGES / NCU_ITERS env knobs

# no GPU at all? the functional check still runs:
mjolnir bench kernel functional
```

Preview any of them without touching the GPU: `mjolnir bench kernel <task>
--dry-run`.

## What counts as "improvement" here (current open work)

- **B3 — mma.sync tile-shape GEMV for exact FI parity:** close the last
  ~10% (222.8 → ~202 µs at L=8192) by matching FlashInfer FA2-tc's tile
  shape (192 KB 384-row K+V tiles, 128-thread blocks). A kernel redesign,
  not a knob — only if exact parity is required.
- **varlen-M GEMV:** decode with MTP verify (M ≤ 8) through the GEMV path.
- **Large-L validation:** the stages=16 ring depth is a wash at L=8192
  (KV still fits the 32 MiB L2) but is the right direction once KV leaves
  L2 — validate at L=32K/64K before touching tiling.
- Already ruled out (don't re-litigate): LDGSTS-vs-LDG (both lower to
  plain LDG on sm_110a), occupancy (smem-bound at st16, not the limit).

## Shipping an improvement

1. Land the change in `fa4-gemv-kernel/sm100_hd256_decode_gemv.py` (and the
   dispatch patch if dispatch changed) — the repo package is the source of
   truth the Dockerfile `COPY`s.
2. `mjolnir verify` (gated, all suites) → `mjolnir bench kernel
   gemv-ringfix` (in-window vs FI baseline).
3. `mjolnir image build --tag <new>` → `mjolnir image gates` (canary, e.g.
   T11 hd256) → `mjolnir bench ab` end-to-end against the previous image
   ([ab-image.md](ab-image.md)).
4. Record: raw JSON in `benchmarks/raw/` (or the package's `docs/`), a
   report under `docs/fa4-hd256-fp8/`, `mjolnir plot` re-renders the charts.
