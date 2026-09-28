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
  <b><em>Mjölnir</em></b> <sup>(MYOL-nir)</sup>
   — <em>serve LLM models on a <b>Jetson AGX Thor</b> with a decode kernel built for exactly that hardware.</em>
</p>

<p align="center">
  <a href="#quickstart">Quickstart</a> •
  <a href="#documentation">Documentation</a> •
  <a href="#benchmarks">Benchmarks</a> •
  <a href="#roadmap">Roadmap</a> •
  <a href="https://github.com/thor-arena/mjolnir/issues">Issues</a>
</p>

## What is ⚡ Mjolnir?

*Mjölnir* is Old Norse for "Thor's hammer" — the project name is
pronounced **"MYOL-nir"**, stress on the first syllable.

It is a ready-to-run vLLM image and a single CLI for serving LLM models on
the NVIDIA Jetson AGX Thor (`sm_110a`). One command starts an
OpenAI-compatible server, another benchmarks it, another verifies the
custom kernel against a reference implementation. The defaults are baked
in for `Qwen/Qwen3.8-27B`, so the first command already does the right
thing.

**Why it exists.** Serving a large model on a single GPU is mostly one
bottleneck: every generated token re-reads the model's entire memory of
the conversation, and that read is what sets your tokens-per-second.
Out of the box, vLLM has no fast path for the biggest attention layers of
models like Qwen3.8-27B on this GPU. Measured on Thor (first gated A/B,
2026-09-26 · kernel bench, 2026-09-27 — [Benchmarks](#benchmarks)):

| | ⚡ Mjolnir FlashInfer | ⚡ Mjolnir FlashAttention 4 (GEMV) |
|---|---|---|
| token generation (e2e, c=1, 8K ctx) | <b>25.4 t/s</b> <b style="color:#1a7f37">+4.1%</b> | <b>25.0 t/s</b> <b style="color:#1a7f37">+2.2%</b> |
| prompt processing (e2e, c=1, 8K ctx) | 2 938 t/s <b style="color:#1a7f37">+19%</b> | 3 557 t/s <b style="color:#1a7f37">+44%</b> |
| decode kernel (L=8192) | <b>190.6 µs</b> — the in-window baseline | 223.4 µs (**+17%** in-window)<br />209.8 µs 1-CTA carve-out (directional) |
| model memory (KV cache) | 8-bit | 8-bit |

Deltas vs stock vLLM on Thor (24.4 t/s tg · 2 469 t/s pp · 190.6 µs decode
kernel, same gated window) — the full tables are in
[Benchmarks](#benchmarks).

So: the stock path leaves bandwidth on the table, and Mjolnir's kernels
close the gap — while the rest of the stack (speculative decoding,
quantized caches, CUDA graphs) is made correct on Thor by a 14-patch
overlay. Every number in this repo is a gated, reproducible measurement
with a committed raw artifact — [how we measure](docs/methodology/benchmarking.md).
The mechanics, with the kernel-level detail, live in the
[deep dive](#going-deeper).

## Quickstart

Requirements: Jetson AGX Thor (JetPack 7.x.x, `sm_110a`), Docker + NVIDIA
Container Toolkit, [`uv`](https://docs.astral.sh/uv/), model weights on the
HF hub (fetched on first serve).

```bash
# one-time: install the CLI from source
uv tool install --from git+https://github.com/thor-arena/mjolnir.git mjolnir

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

## The CLI

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

## Documentation

| you want to… | read |
|---|---|
| Serve, reconfigure, or add a model | [User guides (FAQ)](docs/faq/index.md) — one story per CLI task |
| Trust or reproduce any number | [Measurement methodology](docs/methodology/benchmarking.md) |
| Understand the GEMV decode kernel | [Kernel design doc](docker/vllm-thor/fa4-gemv-kernel/README.md) |
| Follow the kernel workstream | [fa4-hd256-fp8/ reports](docs/fa4-hd256-fp8/) |
| See why the 14 patches exist | [Thor-stack rationale](docs/thor-stack/) + [patch inventory](docker/vllm-thor/PATCHES.md) |
| Track upstream (FP8-descale, SplitKV) | [Upstream status](docs/research/fa4-hd256-upstream-status.md) |
| Study the draft-cudagraph regression | [draft-cudagraph/](docs/draft-cudagraph/) |
| Browse every report | [Docs index](docs/README.md) |
| See the results behind the charts | [Bench history](benchmarks/history.jsonl) + [charts](assets/benchmarks/) |

## What's in the image

### The kernel
Kernel code sits in the `sm100_hd256_decode_gemv.py` (CuTe-DSL): a pure-FMA GEMV decode kernel for `M=1`, `head_dim=256`, GQA, dense or paged KV, `fp16`/`bf16` Q with `fp16`/`bf16`/`e4m3` KV (+ descale tensors), SplitKV + LSE-merge combine. It replaces the tcgen05/tensor-core path for the decode shape (at `M=1` the tensor core buys nothing; the cost is the KV gather).

### Backported upstream vLLM PRs
(`sm_110`-correct, not in 0.30.0's tree):

- [#50885](https://github.com/vllm-project/vllm/pull/50885) — FULL cudagraph mode for speculative decode on the FlashInfer-native backend.
- [#49652](https://github.com/vllm-project/vllm/pull/49652) — draft-decode cudagraphs under Dynamic speculative decoding.
- [#54165](https://github.com/vllm-project/vllm/pull/54165) — Mamba/GDN
  `copy_kv_cache` restore under speculative decode (correctness fix).
- [#55390](https://github.com/vllm-project/vllm/pull/55390) — MTP `draft_spec_decode` annotation for the draft group.
- [#55519](https://github.com/vllm-project/vllm/pull/55519) — suppress the false-positive KV-reuse warning on the GDN hybrid.

### Thor-authored patches
(the `thor-*` set — probe-gated on `sm_110`, passive by default until a
config opts in):

- **GEMV decode kernel** — the kernel above, default-on, opt-out `VLLM_FA4_HD256_GEMV=0`.
- **1-CTA decode carve-out** — decode-shaped `hd256` calls (`M≤8`) drop the 2-CTA cluster
- **hd256 FP8-KV kernel + policy** — FA4 hd256 with e4m3 KV cache (upstream asserts `descale_tensors is None`); gated, canary-proven on Thor.
- **FA4 + FP8-KV eligibility** — widens vLLM's KV-dtype policy for `fa_version=4` on `sm_110` (`hd≤128` proven).
- **GDN prefill enablement** — FlashInfer prefill for the GDN hybrid layers (canary-verified on Thor).
- **Fused draft-decode FI plan advance** — one plan per decode step instead of per-token.
- **DSpark non-causal draft cudagraph** — draft model runs non-causal attention under cudagraph.
- **Draft-cudagraph shape gate** — the c1 regression fix (graph shapes must match capture shapes).

Inventory + per-patch analysis: [`docker/vllm-thor/PATCHES.md`](docker/vllm-thor/PATCHES.md).

## Benchmarks

All numbers from committed raw artifacts in [`benchmarks/`](benchmarks/).
Kernel tables: **one gated window** each — except the multi-L session, where
each kernel runs in its own window (marked). Wall-clock deltas are only
meaningful within a window ([methodology](docs/methodology/benchmarking.md)).
Colored deltas are relative to the table's baseline row;
<b style="color:#57606a">grey</b> = |Δ| < 5%.

### Kernel — M=1 decode, GQA 24:4, quantized KV + descales

**GEMV SplitKV sweep — one gated window** (300 iters; the nominal KV-BW
column is the raw's own `kv_bytes ÷ wall-clock`):

| decode path | µs (median) | nominal KV BW (GB/s) | Δ vs FlashInfer |
|---|---:|---:|---:|
| FlashInfer FA2-tc, e4m3 KV — production baseline | 190.58 | 88.03 | <b style="color:#57606a">0%</b> |
| FA4 GEMV, stages=16, ns=1 | 1 304.91 | 12.86 | <b style="color:#cf222e">+584.7%</b> |
| FA4 GEMV, stages=16, ns=4 | 367.74 | 45.62 | <b style="color:#cf222e">+93.0%</b> |
| FA4 GEMV, stages=16, ns=8 | 315.30 | 53.21 | <b style="color:#cf222e">+65.4%</b> |
| FA4 GEMV, stages=16, ns=20 | 223.42 | 75.09 | <b style="color:#cf222e">+17.2%</b> |
| FA4 GEMV, stages=16, ns=64 | 235.79 | 71.15 | <b style="color:#cf222e">+23.7%</b> |
| FA4 GEMV, stages=16, auto | 226.88 | 73.95 | <b style="color:#cf222e">+19.0%</b> |
| FA4 GEMV, stages=2, ns=1 | 1 275.65 | 13.15 | <b style="color:#cf222e">+569.4%</b> |
| FA4 GEMV, stages=2, ns=20 | 222.46 | 75.42 | <b style="color:#cf222e">+16.7%</b> |

The `ns` sweep is the dominant lever — ×5.8 wall-clock from ns=1 to ns=20
(12.86 → 75.09 GB/s nominal) — and ns=64 turns down (235.8 µs): merge/recompute
overhead eats the tail. stages=2 ≈ stages=16 at ns=20 (222.5 vs 223.4 µs).

**Multi-L session — each kernel in its own clean window** (300 iters, same
session; rows are separate windows, so cross-kernel ratios are directional
only):

| kernel | L=2048 | L=4096 | L=8192 |
|---|---:|---:|---:|
| FlashInfer | 229.50 | 237.09 | 258.91 |
| FA4 hd256 1-CTA (carve-out) | 103.82 | 139.17 | 209.82 |
| FA4 GEMV paged (auto) | 108.10 | 140.51 | 224.48 |
| FA4 GEMV dense (auto) | 108.00 | 139.81 | 224.10 |

In this session's raws the 1-CTA carve-out is the fastest decode kernel —
directional, per the caveat above.

Window quality is data, not noise: the *same* FlashInfer kernel measured
190.6 µs in the sweep window and 258.9 µs in the multi-L session — ×1.36
from desktop co-tenancy. That is why absolute claims in this repo come from
**NCU achieved bandwidth**, and wall-clock only as same-window ratios
([methodology](docs/methodology/benchmarking.md)).

Raw: [`gemv-ring-fix-bench.json`](benchmarks/raw/gemv-ring-fix-bench.json) ·
[`gemv-decode-bench-flashinfer.json`](benchmarks/raw/gemv-decode-bench-flashinfer.json) ·
[`gemv-decode-bench-fa4_1cta.json`](benchmarks/raw/gemv-decode-bench-fa4_1cta.json) ·
[`gemv-decode-bench-gemv_paged.json`](benchmarks/raw/gemv-decode-bench-gemv_paged.json) ·
[`gemv-decode-bench-gemv_dense.json`](benchmarks/raw/gemv-decode-bench-gemv_dense.json)

### End-to-end — llama-benchy, Qwen3.8-27B NVFP4 (token-generation t/s)

The first gated A/B under `--runs 5 --repeat 3 --exact-tg` (2026-09-26;
prompt=2048, gen=128, c=1/2/4, ctx=0/4K/8K). Deltas vs the stock row; the
c=2 cells are in the raw.

| backend (image, config) | c1 / no ctx | c4 / no ctx | c1 / 4K ctx | c4 / 4K ctx | c1 / 8K ctx | c4 / 8K ctx |
|---|---|---|---|---|---|---|
| stock vLLM 0.30.0, NVFP4 † | 22.50 — | 88.54 — | 24.31 — | 81.00 — | 24.43 — | 68.24 — |
| mjolnir v14, NVFP4_FA4hd256 (FA4 + GEMV) | 23.67 <b style="color:#1a7f37">+5.2%</b> | 83.70 <b style="color:#cf222e">−5.5%</b> | 25.39 <b style="color:#57606a">+4.4%</b> | 75.63 <b style="color:#cf222e">−6.6%</b> | 24.96 <b style="color:#57606a">+2.2%</b> | 57.74 <b style="color:#cf222e">−15.4%</b> |
| mjolnir v14, NVFP4 (FlashInfer) | 25.72 <b style="color:#1a7f37">+14.3%</b> | 84.46 <b style="color:#57606a">−4.6%</b> | 24.54 <b style="color:#57606a">+0.9%</b> | 83.84 <b style="color:#57606a">+3.5%</b> | 25.42 <b style="color:#57606a">+4.1%</b> | 70.69 <b style="color:#57606a">+3.6%</b> |

† the stock row was rescued from a window the gate discarded
(`gate.clean=false`, server idle at the time) — treat it as a lower bound
on the baseline.

The GEMV leg wins at c=1 and regresses at c=4 (−15.4% at 8K ctx) — under
investigation ([roadmap](#roadmap)).

**Prompt-processing t/s** (same runs, the raw's `pp_throughput`):

| backend | c1 / no ctx | c4 / no ctx | c1 / 4K ctx | c4 / 4K ctx | c1 / 8K ctx | c4 / 8K ctx |
|---|---|---|---|---|---|---|
| stock vLLM | 2839 — | 2892 — | 2894 — | 2737 — | 2469 — | 2457 — |
| mjolnir v14 (FA4 + GEMV) | 3493 <b style="color:#1a7f37">+23.0%</b> | 3521 <b style="color:#1a7f37">+21.7%</b> | 3941 <b style="color:#1a7f37">+36.2%</b> | 3460 <b style="color:#1a7f37">+26.4%</b> | 3557 <b style="color:#1a7f37">+44.1%</b> | 3307 <b style="color:#1a7f37">+34.6%</b> |
| mjolnir v14 (FlashInfer) | 3522 <b style="color:#1a7f37">+24.0%</b> | 3325 <b style="color:#1a7f37">+15.0%</b> | 3290 <b style="color:#1a7f37">+13.7%</b> | 3104 <b style="color:#1a7f37">+13.4%</b> | 2938 <b style="color:#1a7f37">+19.0%</b> | 2800 <b style="color:#1a7f37">+14.0%</b> |

Prefill is the GDN-hybrid story — the GDN prefill enablement (FlashInfer over
the Triton/FLA reference) shows up at every context depth.

Log: [`benchmarks/history.jsonl`](benchmarks/history.jsonl) · per-run raws:
[`benchmarks/raw/perf-20260926T*Z/`](benchmarks/raw/) ·
charts: [`assets/benchmarks/`](assets/benchmarks) (rendered by `mjolnir plot`).

## Going deeper

*(The kernel-level detail, for people who want the mechanics. The
[Introduction](#what-is-mjolnir) is the human version of this section.)*

**Decode is a memory-gather problem, not a FLOPs problem.** One decode step
re-reads the whole KV cache of every attention layer. For the hd256 layers at
L=8192 (GQA 24:4, e4m3) that is 16 MiB per layer per step; at the 273 GB/s
DRAM roofline the step is bandwidth-bound by construction. Two levers follow:
read more in parallel (more CTAs over the KV), and make each CTA's scan
cheaper.

**Why the stock path is slow.** Upstream has no fast `hd256` decode path for
`sm_110`: vLLM's native FlashAttention route ends in a 2-CTA-cluster kernel
— the slow path for this shape — and vLLM downgrades `hd256` layers with
quantized KV to `FA2` with a `bf16` KV cache (2× the memory traffic).
Mjolnir ships a GEMV decode kernel for exactly this shape —
pure FMA, dense or paged KV, `fp16`/`bf16`/`e4m3` KV with descales, SplitKV
auto-planned for the 20-SM device — default-on in the image, plus the
14-patch vLLM/FlashInfer overlay that makes the rest of the stack work on
Thor (spec-decode cudagraphs, GDN prefill enablement, FP8-KV policy gates).

- **Parallelism (SplitKV / ns).** The GEMV kernel splits the KV range over
  `ns` CTAs (LSE partials + exact merge). Same gated window, L=8192 M=1:
  ns=1 → 12.86 GB/s nominal, 1 304.9 µs; ns=20 → 75.09 GB/s, 223.4 µs
  (**×5.8**); ns=64 → 71.15 GB/s but wall-clock turns down (235.8 µs) —
  merge/recompute overhead eats the tail. The auto-plan lands ≈ ns=20 in the
  same window (226.9 µs).
- **CTA width (1-CTA vs 2-CTA).** The dedicated hd256 FA4 kernel runs a
  2-CTA cluster; for decode shapes (M≤8) the second CTA halves the
  per-token KV scan rate. Dropping to 1 CTA (the carve-out) makes it the
  fastest decode kernel in the latest multi-L session (209.8 µs at L=8192,
  its own clean window).
- **Ring depth (stages).** KV loads pipeline through a cp.async ring:
  stages=2 vs 16 at ns=20, same window — 222.5 µs (75.42 GB/s) vs 223.4 µs
  (75.09 GB/s): a wash at L=8192 while the KV still fits the 32 MiB L2. That
  is why stages=16 ships as default anyway: it is the right direction once KV
  stops fitting L2 (L=32K/64K validation is open work).
- **Why GEMV at all.** At M=1 the GEMV (vector-load + FMA) class is what the
  incumbent baseline (FlashInfer) uses; the tensor-core path adds pipeline
  cost it can't spend. The kernel matches that class and wins back the gap on
  the memory side — hence "within ~17%" in-window (223.4 vs 190.6 µs), with
  the remaining gap attributed to FI's wider tile shape (192 KB 384-row
  K+V tiles, 40 CTAs) — a tile-shape redesign, not a tuning knob.

**Measurement discipline** (normative for every number in the repo): the
GPU is shared with the live server and the desktop, so nothing is ever
killed to "clean" the GPU; benches run only inside **clean windows** (the
live server's queue reads 0/0 for N consecutive samples) and discard
dirty runs. Wall-clock is same-window ratios only; absolute kernel claims
use **NCU achieved bandwidth** — and on CC 11.0 `dram__bytes.sum` is
unavailable, so achieved BW is measured at the L2-fabric level
(`lts__t_sectors × 32 B / time`). Full rules:
[`docs/methodology/benchmarking.md`](docs/methodology/benchmarking.md).

**Architecture decisions:**

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
- *Cudagraphs on FA4.* Graph-captured FA4 decode runs faster per step than
  eager — the c1 regression study fixed what broke it, and the
  draft-cudagraph shape gate keeps capture and replay shapes identical.

Workstream reports (kernel design + benches, route traces, canary logs):
[`docs/fa4-hd256-fp8/`](docs/fa4-hd256-fp8) ·
[`docs/thor-stack/`](docs/thor-stack/) ·
[`docs/draft-cudagraph/`](docs/draft-cudagraph/)

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

## Contributing

The short version: the GPU is shared, so benches wait for a **clean window**
and never restart the server — and every number ships with its raw artifact.
Everything else — patch conventions, kernel workflow, base-image bumps,
commit style — is in [CONTRIBUTING.md](CONTRIBUTING.md).

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
  (the remaining ~17%, in-window); varlen-M dispatch (MTP verify, M≤8);
  large-L validation of the stages=16 ring (L=32K/64K, where KV leaves L2).
- **FA4 hd256 FP8-descale prefill** — upstream is code-complete but
  unmerged (Dao-AILab RFC [#2456](https://github.com/Dao-AILab/flash-attention/issues/2456));
  the port plan is tracked in
  [`docs/thor-stack/fa4-hd256-route-trace.md`](docs/thor-stack/fa4-hd256-route-trace.md).
- **CI** — aarch64 image-build + functional canary on push.

## Security

See [SECURITY.md](SECURITY.md) — in particular, the served API is
unauthenticated; expose it deliberately.

## License

[Apache-2.0](LICENSE)

<p align="center"><sub>⚡ Mjölnir</sub></p>
