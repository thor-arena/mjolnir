# Contributing to Mjolnir

Mjolnir is an open kernel case study: a decode-attention kernel for the NVIDIA
Jetson AGX Thor (`sm_110`), the vLLM image it ships in, and the `mjolnir` CLI
that measures it. Every chapter of the case study is rendered from raw data —
so **a contribution can be as small as one benchmark run from your Thor**.

## The two hard rules

1. **The GPU is shared — never kill the server.** A live vLLM server may be
   serving the people (and agents) working on the box. Never stop, restart or
   rebuild it to "clean" the GPU. Every GPU measurement runs inside a **clean
   window** (the server's running + waiting requests read 0 for N consecutive
   samples); dirty runs are discarded. The gate is
   [`mjolnir_cli/gate.py`](../src/mjolnir-cli/mjolnir_cli/gate.py) — use it,
   don't re-implement it. Every `mjolnir verify` stage and `mjolnir bench` waits
   for it by itself; `mjolnir gate --once` shows the current state.
2. **Every number ships with its raw artifact.** Kernel results are
   `mjolnir.verify/1` JSON in [`benchmarks/raw/verify/`](../benchmarks/raw/verify/);
   end-to-end rows go to [`benchmarks/history.jsonl`](../benchmarks/history.jsonl).
   Charts are rendered only by
   [`charts/`](../src/mjolnir-cli/mjolnir_cli/charts/) with the
   [`theme.py`](../src/mjolnir-cli/mjolnir_cli/theme.py) design system, and the
   README / kernel chapters only from their `.md.j2` templates — CI fails if
   `README.md` doesn't reproduce from the committed data. No prose-only results.
   Full rules: [`docs/methodology/benchmarking.md`](../docs/methodology/benchmarking.md).

## Opening a PR

Pick the matching template from the dropdown when creating the PR — it
collects the provenance CI can't check (run ids, calibration, the stage that
isolates a change, the patch registration). `default.md` covers everything
else; dependabot PRs keep their own body.

## Ways to contribute

### 1. Run the benchmarks on your Thor

The most useful contribution: independent numbers.

```bash
uv tool install mjolnir-cli
git clone https://github.com/thor-arena/mjolnir && cd mjolnir
mjolnir hw setup                     # once: clocks, power mode, docker runtime
mjolnir verify all                   # correctness + every benchmark stage (~1–2 h)
uv run mjolnir plot --readme
```

Open a PR with `benchmarks/raw/verify/**` and the re-rendered `README.md`,
`docs/kernel/` and `assets/benchmarks/` — or attach the JSON files to an issue
if you only want to report them. Mention your JetPack / L4T version and power
mode. ([docs/faq/verify-kernel.md](../docs/faq/verify-kernel.md))

### 2. Improve the kernel

`src/mjolnir/mjolnir/t5_decode.py` is the kernel, its harness and the
bit-faithful emulator. The order is fixed: correctness first, then a same-run
A/B, then the defaults, then the vLLM side.

- each chapter in [`docs/kernel/`](../docs/kernel/README.md) ends with **open
  ideas** — pick one;
- new paths go behind a harness flag; flip a default only with an A/B win
  across L (`mjolnir verify ab`);
- update the chapter's template (`docs/kernel/<stage>.md.j2`) with what you
  learned — negative results included.

Workflow: [`.opencode/skills/kernel-iteration/SKILL.md`](../.opencode/skills/kernel-iteration/SKILL.md).

Good first projects: a vLLM cache-write path for fp6 KV (the kernel already
reads it — see [`kvfmt`](../docs/kernel/kvfmt.md)); fusing the split-KV merge
into the last CTA; closing the fp6 byte-vs-time gap
([`fp6prof`](../docs/kernel/fp6prof.md)).

### 3. Run (or share) an autoresearch session

Let agents run kernel experiments overnight with every attempt on a ledger:
[`research/README.md`](../research/README.md). Share the exported
`research/runs/<tag>/` in a PR — a ledger of 40 discarded ideas is a real
contribution: it tells everyone what not to try. Improvements to the orchestrator
(`.opencode/agent/autoresearch.md`) or the program (`research/program.md`) are welcome too.

### 4. Add a Thor patch, or bump the base image

- Patches are **hand-adapted** diffs in
  [`docker/vllm-thor/patches/`](../docker/vllm-thor/patches/), registered in
  `apply_patches.py` `PATCH_ORDER`, with a `verify_patches.py` sentinel and a
  [`PATCHES.md`](../docker/vllm-thor/PATCHES.md) entry. Prefer the probe-gated
  pattern (active only on `capability == (11, 0)`, passive otherwise).
- **The image build is the gate**: `mjolnir image build` applies and verifies
  the stack and fails if a patch stops applying.
- Base bumps: [`.opencode/skills/update-base-image/SKILL.md`](../.opencode/skills/update-base-image/SKILL.md).
  The image tag follows the vLLM version (`narandill/mjolnir-vllm:v0.31.0`).

### 5. Docs, configs and the CLI

- Task guides live in [`docs/faq/`](../docs/faq/index.md) — one "I want to do X"
  per file, with the exact commands.
- New model configs: [`docs/faq/new-model-config.md`](../docs/faq/new-model-config.md).
- CLI conventions: selections persist to `~/.mjolnir-state.json` with
  precedence *flag > state > `$MJOLNIR_*` > default*; anything that touches
  docker or the GPU has `--dry-run`; anything that emits data has `--json` or
  writes raw JSON.

### Report a problem

Open an [issue](https://github.com/thor-arena/mjolnir/issues) with the image tag
(`mjolnir image`), the config, and the raw JSON or log path.

## Development setup

```bash
uv sync                      # both packages, editable, + pytest
uv run pytest -q             # CLI tests (kernel tests skip without CUDA)
pre-commit install           # ruff check --fix + format on src/ and tests/
```

CI ([`ci.yml`](workflows/ci.yml)) runs ruff, pytest, the README reproducibility
check, both package builds and the vLLM patch check on every PR.

## Releases (maintainers)

| artifact | trigger | version from |
|---|---|---|
| `mjolnir-kernel` on PyPI | tag `mjolnir-kernel-v<version>` | `src/mjolnir/mjolnir/__init__.py` |
| `mjolnir-cli` on PyPI | tag `mjolnir-cli-v<version>` | `src/mjolnir-cli/mjolnir_cli/__init__.py` |
| `narandill/mjolnir-vllm` on Docker Hub | push to `main` touching the image | the vLLM base (`v0.31.0`, `v0.31.0-<sha>`, `latest`) |

The three are versioned independently.

## Commit style

Short, imperative, grouped by concern (see `git log`).

## License

By contributing, you agree that your contributions are licensed under the
[Apache License 2.0](../LICENSE) (its section 5) — except changes to
`src/mjolnir/mjolnir/t5_decode.py`, which stays under its BSD-3-Clause license
(NVIDIA CUTLASS-derived; keep NVIDIA's notice at the top intact). New
third-party code needs an entry in [NOTICE](../NOTICE) and its license kept
with it. No per-file license headers in our own files: the repo `LICENSE`
covers them.
