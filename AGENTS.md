# AGENTS.md — working on the Mjolnir repo

This repo is a **custom vLLM overlay for the NVIDIA Jetson AGX Thor**
(sm_110a) serving Qwen3.8-27B, driven by the `mjolnir` Python package: one
CLI for serving the patched vLLM image, the FA4 GEMV decode kernel package,
and the clean-window-gated benchmarks. This file is the single operational
doc: critical rules, current state, open work. Read it before doing any work
here.

## Current state

- **14-patch stack** (13 vllm + 1 flashinfer) on vLLM `0.30.0`
  (`vllm/vllm-openai:v0.30.0-ubuntu2404`, bumped 2026-09-26; image
  `mjolnir/vllm-thor:qwen38-sm110-v13`). The bump needed only two
  context-only re-adaptations (55390 hunk #3, 55519 hunk #1 — an upstream
  docstring reformat, no semantic drift); canaries: T11 FA4 hd256 FP8-KV
  **PASS** (kernels ran on sm_110a), T10 PASS, T6/T9 INERT-OK (shared-GPU
  OOM). Depth: `docs/thor-stack/base-bump-2026-09-26.md`. Build gate:
  `apply_patches.py` + `verify_patches.py` (the image build fails if a patch
  stops applying). Inventory: `docker/vllm-thor/PATCHES.md`.
- **v11 (2026-09-24): FA4 hd256 1CTA decode** —
  `thor-fa4-hd256-1cta-decode-sm110`: decode-shaped hd256 calls (M≤8,
  non-local, dense or paged-128 TMA) drop the 2CTA cluster to a single CTA →
  **1.56×** decode at L=8192 M=1 (323 µs → 207 µs clean micro-bench). In-tree
  hd256 SplitKV investigated, NO-WIN on both paths, kept inert (num_splits=1).
- **GEMV decode kernel (2026-09-25) — default-on build patch.**
  `thor-fa4-hd256-gemv-decode-sm110` (+ kernel `.py` via `COPY` in the
  Dockerfile; opt out with `VLLM_FA4_HD256_GEMV=0`). Pure-FMA CuTe-DSL kernel
  for M=1 hd256 decode (GQA, dense/paged, fp8 + descales, SplitKV +
  LSE-merge combine). Gated clean window, L=8192 M=1 GQA 24/4: GEMV auto
  (ns=20) = **222.8 µs** vs FlashInfer **202.05 µs** in the same window —
  within ~10% of FI, vs ~2.9× slower before the kernel work. Depth:
  `docker/vllm-thor/fa4-gemv-kernel/README.md`.
- **The `mjolnir` CLI is shippable**: `serve`, `bench perf|ab|kernel`,
  `model` / `image` (bare = arrow-key picker over `configs/` / local docker
  images), `vfa prepare`, `verify`, `plot`, `history` — defaults bake in
  FA4 + GEMV-on for `Qwen/Qwen3.8-27B` / `NVFP4_FA4hd256`, port 6001.
  Selections persist to `$MJOLNIR_STATE` (default `~/.mjolnir-state.json`:
  model + quant + image); precedence: CLI flag > state > `$MJOLNIR_*` >
  baked-in default.

## Critical rules (read first)

### The GPU is shared — never kill the server, wait for idle
The Thor GPU is shared with the **live vLLM server that serves the LLM the
subagents run on**. When benchmarking anything on the GPU:
- **NEVER** stop / restart / kill / rebuild the vLLM server to "clean" the
  GPU.
- **Wait for a "clean window"** — vLLM running+waiting requests = 0 for
  several consecutive samples — BEFORE measuring. The gate lives in
  `src/mjolnir/gate.py` (`CleanWindow`, `wait_for_idle`, `server_load`);
  **use it, don't re-implement it.** CLI: `mjolnir gate --once` /
  `mjolnir bench kernel <task>` (gated tasks gate themselves).
- **ncu achieved-BW** is more robust than wall-clock under the desktop Xorg
  co-tenancy (which can't be gated at request level). Wall-clock is only for
  **relative ratios measured in the same gated window**.

### The clean-window gate
`src/mjolnir/gate.py` is the single source of truth. It polls
`vllm:num_requests_running` / `vllm:num_requests_waiting` at
`http://127.0.0.1:6001/metrics` (the co-located server's own port). Import it
or run it as a CLI; do not copy its logic into a new bench.

## Repo layout
- `src/mjolnir/` — **the CLI + package** (typer). `cli.py` (commands),
  `dockerctl.py` (serve up/down/status — docker orchestration), `gate.py`
  (the gate), `tasks.py` (kernel test/bench registry + docker launcher),
  `benchy.py` (llama-benchy wrapper: raw JSON + history), `history.py`
  (the append-only log), `plots.py` + `theme.py` (charts + design system),
  `vfa.py` (build a live GEMV'd `vllm_flash_attn` tree for kernel iteration).
- `docker/vllm-thor/` — **the build.** `Dockerfile` (thin overlay on the vllm
  aarch64 nightly) + `patches/` (14 hand-adapted patches) +
  `apply_patches.py` + `verify_patches.py` (the authoritative build gate) +
  `PATCHES.md` (the per-patch inventory) + canary scripts.
- `docker/vllm-thor/fa4-gemv-kernel/` — the **GEMV decode kernel package**
  (kernel source + `interface-gemv-dispatch.diff` reference + bench/verify
  scripts + docs). Ships **default-on** in the image (see Current state).
- `benchmarks/` — `raw/` (committed raw JSONs; perf raws land here at
  runtime) + `history.jsonl` (append-only e2e bench history — the substrate
  for the README charts).
- `assets/` — logo/hero/arch SVGs + generated benchmark charts
  (`assets/benchmarks/*.png`, rendered by `mjolnir plot`).
- `configs/<vendor>/<model>/<quant>.yaml` — model configs
  (default: `configs/Qwen/Qwen3.8-27B/NVFP4_FA4hd256.yaml`). User-local
  configs also load from `~/.local/share/mjolnir/configs/` (`$MJOLNIR_CONFIGS_DIR`);
  on a `(model, quant)` clash the repo copy wins. The served model name a
  bench targets comes from the config's `served-model-name` (else its
  `model:` field).
- `docs/` — the formalized research + workstream reports
  (index at `docs/README.md`):
  - `docs/methodology/` — the measurement discipline (clean-window gate,
    NCU achieved-BW vs wall-clock) every number must follow.
  - `docs/fa4-hd256-fp8/` — the kernel workstream (1CTA, SplitKV, GEMV design
    + benches).
  - `docs/thor-stack/` — why the patches exist (hd256 route trace, FP8-on-sm_110
    proof, GDN prefill, nightly-bump process, e2e forensics).
  - `docs/draft-cudagraph/` — the c1 regression study + gate patch.
  - `docs/research/` — formal research notes (upstream status, prior-art,
    methodology literature review, field notes).
  - `docs/faq/` — user-story guides for the `mjolnir` CLI (serve, defaults,
    new model configs, A/B benches, base bumps, GEMV kernel work; index at
    `docs/faq/index.md`).

## Running kernel tests / benchmarks — use `mjolnir`, don't hand-roll `docker run`
`mjolnir bench kernel --help-list` shows every task. Examples:
- `mjolnir image gates` — the sm_110 gate-probe canary (fresh container, no server).
- `mjolnir bench kernel functional` — the no-GPU functional check.
- `mjolnir bench kernel gemv-ringfix --out /p/r.json` — a gated GEMV bench (server must be up).
- `mjolnir bench kernel gemv-bench --dry-run` — preview the docker command (no container).

**Arg model:** launcher flags (`--image`, `--vfa-tree`, `--dry-run`, ...) are
parsed wherever they appear; the **task's own** flags (`--out`, `--mode`,
`--no-gate`, ...) pass through to the script verbatim.

**Getting raw numbers for grep:** pass `--out <path>` — the bench writes the
raw JSON there; then grep it:
```
mjolnir bench kernel gemv-ringfix --out /p/gemv.json
grep -E '"(median|p95|bw_gbs_nominal_kv|clean)"' /p/gemv.json
```

**BENCH tasks need the vLLM server running** (the gate polls it). Start it
with `mjolnir serve up` first; the launcher preflights the metrics endpoint
and fails fast with a pointer if it's down. It **never** restarts the server.

**End-to-end perf:** `mjolnir bench perf` — gated llama-benchy sweeps
(`--runs 5` measured/cell, `--repeat 3` independent windows, `--exact-tg`
pins output length). Raw JSON → `benchmarks/raw/perf-<ts>/`, one row per
sweep → `benchmarks/history.jsonl`, charts auto-renders. `mjolnir bench ab`
does the headline A/B (default legs: the default config vs the baseline
config on `--image`; each leg is `'<config>'` or `'<image>:<config>'` —
the image carries the backend, the config the serving params). Restarts
the server per leg — do it on purpose.

## Building the image
```
mjolnir image build --tag <tag>        # = docker build -t <tag> docker/vllm-thor/
mjolnir image gates                   # sm_110 gate-probe canaries (fresh container)
```
The build applies the 14 patches (`apply_patches.py`) and verifies them
(`verify_patches.py`); it **fails** if a patch no longer applies — that's the
signal the base nightly moved past a patch's assumptions. Bumping the base:
see the "BUMPING TO A NEWER NIGHTLY" section in the `Dockerfile` and
`docs/thor-stack/nightly-bump-process.md`. The active image tag is a CLI
default (`mjolnir serve up --image …` / `$MJOLNIR_IMAGE`); defaults live in
`src/mjolnir/config.py`.

## Open work

### Bench (needs the live GPU)
- [ ] **First gated end-to-end A/B**: `mjolnir bench ab`
      (default legs: `NVFP4_FA4hd256` vs `NVFP4` on the default image)
      — produces the first FA4-GEMV row in `benchmarks/history.jsonl` and
      re-renders the README charts. (Restarts the live server per leg —
      schedule deliberately.)
- [ ] Re-measure the seeded history rows under the `--runs 5` / `--repeat 3`
      protocol and re-baseline the charts.

### GEMV kernel
- [ ] **B3 — mma.sync tile-shape GEMV for exact FI parity**: close the
      remaining ~10% (222.8 → ~202 µs in-window) by matching FI FA2-tc's tile
      shape (192 KB 384-row K+V tiles, 128-thread blocks, 40 CTAs at L=8192).
      A kernel redesign, not a tuning knob — pursue only if exact parity is
      required.
- [ ] **varlen-M GEMV**: decode with MTP verify (M ≤ 8) through the GEMV path.
- [ ] **Large-L validation**: the stages=16 ring depth is a wash at L=8192 but
      the right direction once KV stops fitting in the 32 MiB L2 — validate at
      L=32K/64K.

### FA4 + hd256 + FP8 KV (prefill) — blocked upstream, plan to port later
FA4's dedicated hd256 kernel is bf16-only upstream (asserts
`descale_tensors is None`), and vLLM downgrades FA4→FA2 for hd256 + any
quantized KV — so **FlashInfer remains the only FP8-KV path for the hd256
layers** (FA4+FP8 for hd≤128 *is* proven on Thor —
`docs/thor-stack/fa4-fp8kv-sm110.md`). Depth:
`docs/thor-stack/fa4-hd256-route-trace.md` (kernel/policy trace),
`docs/research/fa4-hd256-upstream-status.md` (upstream status).

Plan (as probe-gated patches, same pattern as the shipped ones):
1. Wait for (or request early access to) the upstream FP8-descale branch for
   the hd256 kernel + the paged-kv (TMA, page-128) branch; varlen is the long
   pole — check RFC #2456 status before investing.
2. Port the FP8-descale plumbing into the vendored
   `sm100_hd256_2cta_fmha_forward.py` (transcribable from the generic kernel
   `flash_fwd_sm100.py` `DescaleTensors` path, ~200–400 LOC) + relax the
   `descale_tensors is None` assert + widen the arch assert.
3. Widen vLLM's policy gates (`fa_utils.py` "quantized KV" reason, the
   `family(110)` term in `flash_attn_supports_kv_cache_dtype`).
4. sm_110 bring-up: TMEM table entry, register/tile tuning on real Thor
   silicon (all upstream tuning is B200-derived), fp8 numerics validation.
5. Config side: `block_size=128` (TMA path) for the hd256 kernel; verify the
   kernel's other forbids (no pack_gqa, no SplitKV>1, no score_mod) against
   GQA 24:4 + paged.
6. Probe-gated patch + end-to-end A/B vs the FlashInfer baseline before any
   workload switch.

## Upstream tracking

- Dao-AILab/flash-attention RFC [#2456](https://github.com/Dao-AILab/flash-attention/issues/2456)
  (owners @Johnsonms, @tzadouri):
  - FA4 hd256 fwd+bwd merged ([#2412](https://github.com/Dao-AILab/flash-attention/pull/2412), Apr 2026).
  - **FP8 input (descaled KV) for hd256: "🔨 code complete, perf to be improved"** —
    written, unmerged, not public on any branch/PR (verified 2026-09-21).
    Est. merge: Q4 2026.
   - Merged since the RFC: exp2-emu (#2488), paged-kv TMA (#2489),
     seqused_k/q (#2810), sm_110 arch gating (#2590 — the 2CTA kernels *do* run
     on sm_110; only a software postprocess bug, #2491).
   - **hd256 SplitKV fwd merged** ([#2916](https://github.com/Dao-AILab/flash-attention/pull/2916)
     + refactor [#2917](https://github.com/Dao-AILab/flash-attention/pull/2917),
     2026-09-25, squash `e9cf2c1`): static splits on the dedicated 2CTA fwd
     kernel (LSE partials, empty-split safe, varlen scheduler, paged+split
     tests; 2.0–3.9× decode on GB300). The new CTA-count heuristic is gated
     `arch // 10 in [10, 11]` (sm_110 in scope) but auto-disables on the
     20-SM Thor for our decode shapes (48 clusters > 20 SMs → 0 splits) —
     corroborates our in-tree NO-WIN. NOT the FP8-descale work (still private
     per #2456), and NOT in vLLM 0.30.0's vendored FA4 (pre-#2916), so the fa4
     patch group is unaffected. Depth:
     `docs/research/fa4-hd256-upstream-status.md` (2026-09-26 update).
   - Open: persistent-cluster scheduler (the 20-SM small-batch lever),
     sliding-window (#2749), hd256 backward + seqused (#2891), hd512 (#2877).
- FlashInfer: no hd256+FP8 work; XQA sm_110 gap open, unassigned
  ([#2522](https://github.com/flashinfer-ai/flashinfer/issues/2522)).
- vLLM: [#55366](https://github.com/vllm-project/vllm/pull/55366) (include SM110
  in FA4 auto-selection — we currently force `flash_attn_version=4`),
  [#54705](https://github.com/vllm-project/vllm/pull/54705) (fp8 e4m3 KV cache
  with per-tensor scales on SM100), [#55196](https://github.com/vllm-project/vllm/issues/55196)
  (open RFC questioning the FP8-KV memory win on GDN hybrids — our exact model
  class).
- Draft-decode cudagraphs:
  - vLLM [PR #53383](https://github.com/vllm-project/vllm/pull/53383) (open):
    triton verify builders bake `seq_lens=1` at cudagraph capture — same bug
    class as our c1 regression, different kernel. Watch for merge.
  - vLLM [#55581](https://github.com/vllm-project/vllm/issues/55581) +
    [PR #56289](https://github.com/vllm-project/vllm/pull/56289) (in progress):
    `get_cudagraph_support()` head-count mismatch can kill/limit draft
    cudagraphs — directly relevant to the DSpark draft CG grant.
  - vLLM [#49547](https://github.com/vllm-project/vllm/issues/49547): FlashInfer
    native decode caps at UNIFORM_SINGLE_TOKEN_DECODE → PIECEWISE downgrade
    under spec-decode.
  - FlashInfer [PR #3871](https://github.com/flashinfer-ai/flashinfer/pull/3871)
    (merged Jul 2026): graph-safe uniform multi-token decode; design statement
    confirms frozen plan_info depends only on (batch, total_rows, uniform_q_len)
    — the property our c1 regression study relies on
    (`docs/draft-cudagraph/c1-regression-study.md`).

## Conventions
- **New Thor patches** go in `docker/vllm-thor/patches/` (hand-adapted,
  **not** a raw `git apply`), are registered in `apply_patches.py`
  `PATCH_ORDER`, get a `verify_patches.py` entry, and a `PATCHES.md` entry.
  The **probe-gated pattern** (granted only on sm_110, passive by default
  until a config opts in) is the norm.
- **Kernel research** lives in `docs/<workstream>/` (reports); the
  **shippable** kernel code lives under `docker/vllm-thor/`.
- **Bench discipline:** every new bench uses `src/mjolnir/gate.py`, emits raw
  JSON (grep-able, no prose-only results), and appends to
  `benchmarks/history.jsonl` through `src/mjolnir/history.py`. Subagents
  spend cycles on the kernel, not the plumbing.
- **Charts** only go through `src/mjolnir/plots.py` + the `theme.py` design
  system (one visual language for every number in the repo).
- **Commit style:** short, imperative, grouped by concern (see `git log`).

## Pointers
- The patch inventory + per-patch analysis: `docker/vllm-thor/PATCHES.md`.
- The GEMV decode kernel (design + results + methodology):
  `docker/vllm-thor/fa4-gemv-kernel/README.md`.
- The measurement discipline behind every number:
  `docs/methodology/benchmarking.md`.
- The docs index: `docs/README.md`.
