<p align="center">
  <img src="assets/header.png" alt="Mjolnir" width="100%">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/Jetson-AGX%20Thor-76b900" alt="NVIDIA Jetson AGX Thor">
  <img src="https://img.shields.io/badge/vLLM-0.30.0-black" alt="vLLM 0.30.0">
  <img src="https://img.shields.io/badge/FA4-GEMV%20decode%20kernel-orange" alt="FA4 GEMV decode kernel">
  <img src="https://img.shields.io/badge/license-Apache--2.0-green" alt="Apache-2.0">
</p>

<p align="center">
  <b><em>Mjölnir</em></b>
   — <em>serve LLM models on a <b>Jetson AGX Thor</b>,<br/>with a dedicated <b>FlashAttention 4</b> kernel that makes decode lightning fast.</em>
</p>

## Introduction

Mjolnir is a patched vLLM image and a single CLI for serving LLM models on the NVIDIA Jetson AGX Thor (`sm_110a`) with a custom FlashAttention decode kernel.

Why a custom kernel: at every decode step the model re-reads the **entire KV cache** — at `L=8192` that is `16 MiB` per `head-dim-256` attention layer per token. Thor's DRAM roofline is `~273 GB/s`, so decode speed is set by how many CTAs can be made to read that cache in parallel. Upstream has no fast `hd256` decode path for `sm_110`: vLLM's native FlashAttention route ends in a 2-CTA-cluster kernel that, for this shape, is ~4.5× slower than the FlashInfer production kernel (measured, same window — [below](#benchmarks)), and vLLM downgrades `hd256` layers with quantized `KV` to `FA2` with a `bf16` KV cache (2× the memory traffic).

**Mjolnir** ships a **GEMV decode kernel** for exactly this shape — pure FMA, paged/dense, `fp16`/`bf16`/`e4m3-KV` with `descales`, `SplitKV` auto-planned for the
`20-SM` device — default-on in the image, plus the 14-patch vLLM/FlashInfer overlay that makes the rest of the stack work on Thor (spec-decode cudagraphs, GDN prefill enablement, FP8-KV policy gates).

Measured where it matters (`L=8192`, `M=1`, `GQA 24:4`, `e4m3 KV`, gated window): the GEMV kernel is **within ~10% of the FlashInfer production baseline** and the 1-CTA carve-out is **1.56×** over the stock 2-CTA kernel. The gated end-to-end A/B that turns kernel wins into serving numbers is the next measurement ([roadmap](#roadmap)).

## What's in the image

### The kernel
Kernel code sits in the `sm100_hd256_decode_gemv.py` (CuTe-DSL): a pure-FMA GEMV decode kernel for `M=1`, `head_dim=256`, GQA, dense or paged KV, `fp16`/`bf16` Q with `fp16`/`bf16`/`e4m3` KV (+ descale tensors), SplitKV + LSE-merge combine. It replaces the tcgen05/tensor-core path for the decode shape (at `M=1` the tensor core buys nothing; the cost is the KV gather).

### Backported upstream vLLM PRs
#### (`sm_110`-correct, not in 0.30.0's tree):

- [#50885](https://github.com/vllm-project/vllm/pull/50885) — FULL cudagraph mode for speculative decode on the FlashInfer-native backend.
- [#49652](https://github.com/vllm-project/vllm/pull/49652) — draft-decode cudagraphs under Dynamic speculative decoding.
- [#54165](https://github.com/vllm-project/vllm/pull/54165) — Mamba/GDN
  `copy_kv_cache` restore under speculative decode (correctness fix).
- [#55390](https://github.com/vllm-project/vllm/pull/55390) — MTP `draft_spec_decode` annotation for the draft group.
- [#55519](https://github.com/vllm-project/vllm/pull/55519) — suppress the false-positive KV-reuse warning on the GDN hybrid.

### Thor-authored patches
#### (the `thor-*` set — probe-gated on `sm_110`, passive
by default until a config opts in):

- **GEMV decode kernel** — the kernel above, default-on, opt-out `VLLM_FA4_HD256_GEMV=0`.
- **1-CTA decode carve-out** — decode-shaped `hd256` calls (`M≤8`) drop the 2-CTA cluster
- **hd256 FP8-KV kernel + policy** — FA4 hd256 with e4m3 KV cache (upstream asserts `descale_tensors is None`); gated, canary-proven on Thor.
- **FA4 + FP8-KV eligibility** — widens vLLM's KV-dtype policy for `fa_version=4` on `sm_110` (`hd≤128` proven).
- **GDN prefill enablement** — FlashInfer prefill for the GDN hybrid layers (3.9–5.2× kernel-level, live canary).
- **Fused draft-decode FI plan advance** — one plan per decode step instead of per-token.
- **DSpark non-causal draft cudagraph** — draft model runs non-causal attention under cudagraph.
- **Draft-cudagraph shape gate** — the c1 regression fix (graph shapes must match capture shapes).

Inventory + per-patch analysis: [`docker/vllm-thor/PATCHES.md`](docker/vllm-thor/PATCHES.md).

## Benchmarks

All numbers from committed raw artifacts; every table is **one gated window**
— wall-clock deltas are only meaningful within a window
([methodology](docs/methodology/benchmarking.md)). Colored deltas are relative
to the window's baseline row; <b style="color:#57606a">grey</b> = |Δ| < 5%.

### Kernel — L=8192, M=1, GQA 24:4, bf16 Q + e4m3 paged KV

**The native FlashAttention route is the gap** (exclusive window, 100 iters):

| decode path | µs (median) | Δ vs FlashInfer |
|---|---:|---:|
| FlashInfer FA2-tc, e4m3 KV — production baseline | 72.53 | <b style="color:#57606a">0%</b> |
| FA4 hd256 2-CTA, e4m3 KV (stock FA4 path) | 324.08 | <b style="color:#cf222e">+346.8%</b> |
| FA4 hd256 2-CTA, bf16 KV (stock vLLM fallback class) | 323.57 | <b style="color:#cf222e">+344.7%</b> |

**1-CTA carve-out** (clean micro-bench, both kernels back-to-back):

| decode path | µs (median) | Δ vs 2-CTA |
|---|---:|---:|
| FA4 hd256 2-CTA (stock) | 323.0 | <b style="color:#57606a">0%</b> |
| FA4 hd256 1-CTA (carve-out) | 207.1 | <b style="color:#1a7f37">−35.9%</b> |

**GEMV decode kernel** (gated window, 300 iters):

| decode path | µs (median) | Δ vs FlashInfer |
|---|---:|---:|
| FlashInfer FA2-tc — production baseline | 202.05 | <b style="color:#57606a">0%</b> |
| FA4 GEMV, stages=16, ns=20 (auto) | 222.80 | <b style="color:#cf222e">+10.3%</b> |
| FA4 GEMV, stages=2, ns=20 | 221.42 | <b style="color:#cf222e">+9.5%</b> |

Window quality is data, not noise: the *same* FlashInfer kernel measured
72.5 µs in the exclusive window above and 202 µs in the gated one below —
desktop co-tenancy inflates wall-clock ~2.8×. That is why absolute claims in
this repo come from **NCU achieved bandwidth**, and wall-clock only as
same-window ratios. GEMV's absolute position:

| NCU achieved BW (L=8192 M=1) | ns=1 | ns=20 (auto) | ns=64 |
|---|---:|---:|---:|
| GEMV, GB/s (L2-fabric, ×32 B/sector) | 13.55 | 118.30 | 168.60 |

ns=1→20 = **×8.7** achieved bandwidth for the same kernel — the SplitKV
parallelism lever that the stock hd256 kernel cannot use.

Raw: [`gemv-ring-fix-bench.json`](docker/vllm-thor/fa4-gemv-kernel/gemv-ring-fix-bench.json)
(the ns sweep), [decode micro-bench results](docs/fa4-hd256-fp8/decode-microbench.md).

### End-to-end — llama-benchy, Qwen3.8-27B NVFP4 (token-generation t/s)

Rows below are the **seeded** history (3 runs, no exact-tg, pre-gate protocol)
— the first gated A/B under `--runs 6` re-baselines them
([roadmap](#roadmap)). Deltas vs the stock row.

| backend (image, config) | c1 / no ctx | c4 / no ctx | c1 / 4K ctx | c4 / 4K ctx | c1 / 8K ctx | c4 / 8K ctx |
|---|---|---|---|---|---|---|
| stock vLLM (unpatched), NVFP4 | 24.1 — | 82.1 — | 25.1 — | 83.2 — | 24.4 — | 70.7 — |
| v9, NVFP4 (FlashInfer) | 24.6 <b style="color:#57606a">+2.1%</b> | 83.0 <b style="color:#57606a">+1.1%</b> | 24.3 <b style="color:#57606a">−3.2%</b> | 81.6 <b style="color:#57606a">−1.9%</b> | 25.3 <b style="color:#57606a">+3.7%</b> | 68.5 <b style="color:#57606a">−3.1%</b> |
| v10, NVFP4_FA4hd256 (FA4 for hd256) | 27.0 <b style="color:#1a7f37">+12.0%</b> | 84.0 <b style="color:#57606a">+2.3%</b> | 24.7 <b style="color:#57606a">−1.6%</b> | 76.0 <b style="color:#cf222e">−8.7%</b> | 22.6 <b style="color:#cf222e">−7.4%</b> | 56.0 <b style="color:#cf222e">−20.8%</b> |

The v10 row is the 2-CTA FA4 path: it wins where KV re-read is small and
loses where it dominates — the exact shape the GEMV kernel targets. The GEMV
image (v13) e2e row lands with the first gated A/B.

**Prompt-processing t/s** (same runs):

| backend | c1 / no ctx | c4 / no ctx | c1 / 4K ctx | c4 / 4K ctx | c1 / 8K ctx | c4 / 8K ctx |
|---|---|---|---|---|---|---|
| stock vLLM | 2520 — | 2940 — | 2709 — | 2507 — | 2155 — | 2213 — |
| v9 (FlashInfer) | 2791 <b style="color:#1a7f37">+10.8%</b> | 3326 <b style="color:#1a7f37">+13.1%</b> | 3038 <b style="color:#1a7f37">+12.1%</b> | 3113 <b style="color:#1a7f37">+24.2%</b> | 2813 <b style="color:#1a7f37">+30.5%</b> | 2813 <b style="color:#1a7f37">+27.1%</b> |
| v10 (FA4 hd256) | 2781 <b style="color:#1a7f37">+10.4%</b> | 3492 <b style="color:#1a7f37">+18.8%</b> | 3559 <b style="color:#1a7f37">+31.4%</b> | 3471 <b style="color:#1a7f37">+38.5%</b> | 3209 <b style="color:#1a7f37">+48.9%</b> | 3332 <b style="color:#1a7f37">+50.6%</b> |

Prefill is the GDN-hybrid story — the GDN prefill enablement (FlashInfer over
the Triton/FLA reference) shows up at every context depth.

Log: [`benchmarks/history.jsonl`](benchmarks/history.jsonl) ·
charts: [`assets/benchmarks/`](assets/benchmarks) (rendered by `mjolnir plot`).

## Going deeper

**Decode is a memory-gather problem, not a FLOPs problem.** One decode step
re-reads the whole KV cache of every attention layer. For the hd256 layers at
L=8192 (GQA 24:4, e4m3) that is 16 MiB per layer per step; at the 273 GB/s
DRAM roofline the step is bandwidth-bound by construction. Two levers follow:
read more in parallel (more CTAs over the KV), and make each CTA's scan
cheaper.

- **Parallelism (SplitKV / ns).** The GEMV kernel splits the KV range over
  `ns` CTAs (LSE partials + exact merge). NCU achieved-BW at L=8192 M=1:
  ns=1 → 13.55 GB/s, ns=20 → 118.3 GB/s (**×8.7**), ns=64 → 168.6 GB/s but
  wall-clock turns down (233.5 µs vs 222.8 µs at ns=20) — merge/recompute
  overhead eats the tail. The auto-plan picks ns=20 = 80 CTAs on the 20-SM
  device.
- **CTA width (1-CTA vs 2-CTA).** The dedicated hd256 FA4 kernel runs a
  2-CTA cluster; for decode shapes (M≤8) the second CTA halves the
  per-token KV scan rate. Dropping to 1 CTA is 1.56× at L=8192 (323 → 207 µs).
- **Ring depth (stages).** KV loads pipeline through a cp.async ring:
  stages=2→16 buys **+12.9% kernel bandwidth** at ns=20 (NCU: 104.7 →
  118.3 GB/s) but 0.6% wall-clock at L=8192 — combine/launch overhead and
  non-KV L2 traffic swallow it while the KV still fits the 32 MiB L2. That is
  why stages=16 ships as default anyway: it is the right direction once KV
  stops fitting L2 (L=32K/64K validation is open work).
- **Why GEMV at all.** At M=1 the GEMV (vector-load + FMA) class is what the
  incumbent baseline (FlashInfer) uses; the tensor-core path adds pipeline
  cost it can't spend. The kernel matches that class and wins back the gap on
  the memory side — hence "within ~10%" in-window, with the remaining gap
  attributed to FI's wider tile shape (192 KB 384-row K+V tiles, 40 CTAs) —
  a tile-shape redesign, not a tuning knob.

**Measurement discipline** (normative for every number in the repo): the
GPU is shared with the live server and the desktop, so nothing is ever
killed to "clean" the GPU; benches run only inside **clean windows** (the
live server's queue reads 0/0 for N consecutive samples) and discard
dirty runs. Wall-clock is same-window ratios only; absolute kernel claims
use **NCU achieved bandwidth** — and on CC 11.0 `dram__bytes.sum` is
unavailable, so achieved BW is measured at the L2-fabric level
(`lts__t_sectors × 32 B / time`). Full rules:
[`docs/methodology/benchmarking.md`](docs/methodology/benchmarking.md).

**Architecture decisions, each with a number attached:**

- *Probe-gated sm_110 grants.* Every Thor patch is granted only on
  `capability == (11, 0)`, passive by default, and opt-in per config — so a
  patch can sit in the image at zero cost until its feature is switched on,
  and upstream landing the feature makes the grant a no-op. Canary-proven:
  the FA4 hd256 FP8-KV probe fires kernels on real Thor silicon while the
  GDN/draft-CG probes stay inert-OK under shared-GPU OOM.
- *Default-on GEMV, scope-narrow.* The kernel dispatches only on in-scope
  shapes (M=1, hd256, GQA, dense/paged, e4m3 or bf16 KV); anything else falls
  through to the FA4 path. Out of scope: varlen-M (MTP verify, M≤8) — the
  next dispatch surface.
- *Cudagraphs on FA4.* Graph-captured FA4 decode runs 43 µs faster per step
  than eager (275.3 vs 318.4 µs median, same window) — the c1 regression
  study fixed what broke it, and the draft-cudagraph shape gate keeps
  capture and replay shapes identical.

Workstream reports (kernel design + benches, route traces, canary logs):
[`docs/fa4-hd256-fp8/`](docs/fa4-hd256-fp8) ·
[`docs/thor-stack/`](docs/thor-stack/) ·
[`docs/draft-cudagraph/`](docs/draft-cudagraph/)

## Quickstart

Requirements: Jetson AGX Thor (JetPack 7.x.x, `sm_110a`), Docker + NVIDIA
Container Toolkit, [`uv`](https://docs.astral.sh/uv/), model weights on the
HF hub (fetched on first serve).

```bash
# one-time: install the CLI from source
uv tool install --from git+https://github.com/<owner>/mjolnir.git mjolnir

# the CLI guides the next steps
mjolnir

# serve — baked-in defaults: Qwen/Qwen3.8-27B, NVFP4 + FA4 + GEMV, port 6001
mjolnir serve up

# another model? drop a config at
# ~/.local/share/mjolnir/configs/<vendor>/<model>/<quant>.yaml and:
mjolnir model
```

```bash
curl -s http://localhost:6001/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model": "Qwen/Qwen3.8-27B", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 64}'
```

## CLI

| command | what it does |
|---|---|
| `mjolnir` | help + next-steps guide |
| `mjolnir serve up\|down\|status\|logs` | drive the vLLM container |
| `mjolnir model [use\|list]` | active model/config (bare = picker over repo `configs/` + user-local `~/.local/share/mjolnir/configs/`) |
| `mjolnir image [use\|list\|build\|gates]` | active image (bare = picker) / build the patch stack / run the canary gates |
| `mjolnir bench perf` | gated e2e throughput sweep → raw JSON → history → charts |
| `mjolnir bench ab` | headline A/B (restarts the server per leg — schedule it) |
| `mjolnir bench kernel <task>` | gated kernel benches (`--help-list`; `--dry-run` previews) |
| `mjolnir verify` | GEMV kernel correctness suites (gated, vs fp32 reference) |
| `mjolnir gate` | the clean-window gate (`--once` = check now) |
| `mjolnir vfa prepare` | build the GEMV'd `vllm_flash_attn` tree (kernel iteration) |
| `mjolnir plot` / `mjolnir history` | render the README charts / browse the bench log |

Selections persist to `~/.mjolnir-state.json`; precedence: CLI flag > state
> `$MJOLNIR_*` > baked-in default. Raw numbers for grep: any bench takes
`--out <path>` — e.g. `mjolnir bench kernel gemv-ringfix --out /p/r.json`,
then `grep -E '"(median|p95|bw_gbs_nominal_kv|clean)"' /p/r.json`.

## Repository layout

```
├── src/mjolnir/               the CLI + package (typer): cli, dockerctl, gate,
│                              tasks, benchy, history, plots, vfa
├── docker/vllm-thor/          the build: Dockerfile (thin overlay) +
│   │                          apply/verify gate + PATCHES.md + canaries
│   ├── patches/               the 14 hand-adapted patches
│   └── fa4-gemv-kernel/       the GEMV kernel package (source of truth,
│                              verify + bench scripts, design doc)
├── .opencode/skills/          agent skills: update-base-image, kernel-iteration
├── configs/Qwen/Qwen3.8-27B/  model configs (NVFP4_FA4hd256 default + NVFP4);
│                              user-local: ~/.local/share/mjolnir/configs/
├── benchmarks/
│   ├── raw/                   committed raw JSONs (grep-able)
│   └── history.jsonl          append-only e2e bench log (the chart substrate)
├── assets/                    header / logo / arch + generated benchmark charts
└── docs/
    ├── methodology/           the measurement discipline (normative)
    ├── fa4-hd256-fp8/         kernel workstream reports
    ├── thor-stack/            patch-stack rationale + base-bump records
    ├── draft-cudagraph/       the c1 regression study
    ├── research/              upstream status, prior art
    └── faq/                   user-story guides (serve, configs, A/B, bumps)
```

## Development

**Bumping the base image** (new vLLM nightly → new digest → patch re-verify →
canaries → default-tag switch): the image build *is* the gate — it applies
and verifies all 14 patches and fails if one drifts. Follow
[`.opencode/skills/update-base-image/SKILL.md`](.opencode/skills/update-base-image/SKILL.md);
worked examples: [`nightly-bump-process.md`](docs/thor-stack/nightly-bump-process.md)
and [`base-bump-2026-09-26.md`](docs/thor-stack/base-bump-2026-09-26.md)
(the vLLM 0.30.0 bump: 14/14 patches, 2 context-only re-adaptations).

**Kernel work** (the GEMV decode kernel and its dispatch): correctness first
(`mjolnir verify` must be GO), then clean-window-gated benches, NCU for
absolute claims, then ship path (repo kernel file → image rebuild → verify in
image). Follow [`.opencode/skills/kernel-iteration/SKILL.md`](.opencode/skills/kernel-iteration/SKILL.md);
the normative rules live in
[`docs/methodology/benchmarking.md`](docs/methodology/benchmarking.md)
and the kernel design doc in
[`docker/vllm-thor/fa4-gemv-kernel/README.md`](docker/vllm-thor/fa4-gemv-kernel/README.md).

**User guides** (serve, defaults, new model configs, A/B benches, base bumps,
kernel work): [`docs/faq/index.md`](docs/faq/index.md).

## Roadmap

- **First gated end-to-end A/B** — `mjolnir bench ab` (NVFP4_FA4hd256 vs
  NVFP4 on v13): the GEMV row in `history.jsonl` + re-rendered charts; then
  re-measure the seeded rows under `--runs 6` and re-baseline.
- **GEMV kernel** — mma.sync tile-shape variant for exact FlashInfer parity
  (the remaining ~10%, in-window); varlen-M dispatch (MTP verify, M≤8);
  large-L validation of the stages=16 ring (L=32K/64K, where KV leaves L2).
- **FA4 hd256 FP8-descale prefill** — upstream is code-complete but
  unmerged (Dao-AILab RFC [#2456](https://github.com/Dao-AILab/flash-attention/issues/2456));
  the port plan is tracked in
  [`docs/thor-stack/fa4-hd256-route-trace.md`](docs/thor-stack/fa4-hd256-route-trace.md).
- **CI** — aarch64 image-build + functional canary on push.

## License

[Apache-2.0](LICENSE)

<p align="center"><sub>⚡ Mjölnir</sub></p>
