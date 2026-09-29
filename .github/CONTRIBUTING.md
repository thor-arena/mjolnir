# Contributing to Mjolnir

Mjolnir is a vLLM overlay for the NVIDIA Jetson AGX Thor (`sm_110a`): a
patched image, a GEMV decode kernel, and the `mjolnir` CLI. Read this before
opening a PR — most of it is about the two rules that protect every number in
the repo.

## The two hard rules

1. **The GPU is shared — never kill the server.** The live vLLM server runs
   the LLM the development agents work on. No stopping, restarting, or
   rebuilding it to "clean" the GPU. Every GPU bench runs only inside a
   **clean window** (the server's running + waiting requests read 0 for N
   consecutive samples), and dirty runs are discarded. The gate lives in
   [`src/mjolnir/gate.py`](src/mjolnir/gate.py) — **use it, don't re-implement
   it**. `mjolnir gate --once` checks the current state; `mjolnir bench kernel
   <task>` gates itself.
2. **Every number ships with its raw artifact.** Benches emit grep-able JSON
   (`--out <path>`), append e2e rows to
   [`benchmarks/history.jsonl`](benchmarks/history.jsonl) through
   [`src/mjolnir/history.py`](src/mjolnir/history.py), and render charts only
   through [`src/mjolnir/plots.py`](src/mjolnir/plots.py) +
   [`src/mjolnir/theme.py`](src/mjolnir/theme.py) — one visual language for
   every number in the repo. No prose-only results. Wall-clock claims are
   same-window ratios only; absolute kernel claims use **NCU achieved
   bandwidth**. Full rules: [`docs/methodology/benchmarking.md`](docs/methodology/benchmarking.md).

## Ways to contribute

### Report a problem / request a feature

Open an [issue](https://github.com/thor-arena/mjolnir/issues). For anything
measured, include the image tag (`mjolnir image`), the model + quant config,
and the raw JSON (`--out <path>` from the bench).

### Add a Thor patch

The patch stack is the heart of the image:

- Patches are **hand-adapted** (not a raw `git apply`) and live in
  [`docker/vllm-thor/patches/`](docker/vllm-thor/patches/).
- Each new patch gets: a registration in `apply_patches.py` `PATCH_ORDER`, an
  entry in `verify_patches.py`, and an entry in
  [`docker/vllm-thor/PATCHES.md`](docker/vllm-thor/PATCHES.md).
- The **probe-gated pattern is the norm**: the grant fires only on
  `capability == (11, 0)` (sm_110), the patch is passive by default until a
  config opts in — so it sits in the image at zero cost until needed, and
  upstream landing the feature makes the grant a no-op.
- **The image build is the gate.** `mjolnir image build` applies and
  verifies the whole stack and *fails* if a patch stops applying; canaries
  (`mjolnir image gates`, `mjolnir bench kernel …`) prove the change on real
  Thor silicon.

### Improve the GEMV kernel

Kernel work follows a fixed order: correctness first (`mjolnir verify` must
be GO), then clean-window-gated benches, then NCU for absolute claims, then
the ship path (repo kernel file → image rebuild → verify in image). Follow
[`.opencode/skills/kernel-iteration/SKILL.md`](.opencode/skills/kernel-iteration/SKILL.md);
the design doc is
[`docker/vllm-thor/fa4-gemv-kernel/README.md`](docker/vllm-thor/fa4-gemv-kernel/README.md).

### Bump the base image

New vLLM digest → patch re-verify → canaries → default-tag switch. The build
is the gate; re-adapt any hunk whose context drifted. Follow
[`.opencode/skills/update-base-image/SKILL.md`](.opencode/skills/update-base-image/SKILL.md);
worked examples:
[`docs/thor-stack/nightly-bump-process.md`](docs/thor-stack/nightly-bump-process.md)
and [`docs/thor-stack/base-bump-2026-09-26.md`](docs/thor-stack/base-bump-2026-09-26.md).

### Docs and the CLI

- User-story guides live in [`docs/faq/`](docs/faq/index.md) — one file per
  CLI task ("I want to do X" with the exact commands).
- The README charts are **rendered, not hand-edited**: `mjolnir plot` reads
  `benchmarks/history.jsonl` and writes `assets/benchmarks/*.png`.
- CLI conventions: launcher flags parse wherever they appear; task-specific
  flags pass through verbatim; selections persist to
  `~/.mjolnir-state.json` with precedence *CLI flag > state > `$MJOLNIR_*` >
  baked-in default*.

## Pre-commit gate (one-time setup)

`pre-commit install` — ruff (`check --fix` + `format`) runs on every commit
over `src/` (the shippable package = the wheel contents). The docker research
scripts and the CuTe-DSL kernel are deliberately out of scope: kernel naming
follows FA conventions (e.g. the running log-sum-exp accumulator `l`). The
hook version is pinned in `.pre-commit-config.yaml` — the CI in
[.github/workflows/ci.yml](.github/workflows/ci.yml) runs the same checks on
every PR, so the gate you see locally is the gate that blocks the release.

## Commit style

Short, imperative, grouped by concern (see `git log`).

## License

By contributing, you agree that your contributions are made under the
[MIT](LICENSE) license.
