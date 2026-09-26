# FAQ — User Stories

Task-oriented guides for the `mjolnir` CLI: each file is one story
("I want to do X") with the exact commands and the methodology behind them.
Read [`../README.md`](../README.md) first for what the repo is; read
[`../methodology/benchmarking.md`](../methodology/benchmarking.md) before you
trust or publish any number.

## Serving & defaults

| Story | File |
|---|---|
| Start / stop / inspect the vLLM server | [serve-server.md](serve-server.md) |
| Change the default vLLM image (and build a new one) | [default-image.md](default-image.md) |
| Change the default model / quant config | [default-model-quant.md](default-model-quant.md) |
| Add a new model config (repo or user-local, persisted at `~/`) | [new-model-config.md](new-model-config.md) |
| What wins: flag, state, env var, or baked-in default? | [settings-precedence.md](settings-precedence.md) |

## Benchmarking

| Story | File |
|---|---|
| The clean-window gate — why every bench waits, and never kills the server | [clean-window-gate.md](clean-window-gate.md) |
| A/B a custom image against the default image | [ab-image.md](ab-image.md) |
| A/B different models against each other | [ab-models.md](ab-models.md) |
| Run kernel benches / correctness suites (with or without the server) | [kernel-bench-tasks.md](kernel-bench-tasks.md) |
| Where results land, and how to compare runs | [results-history-charts.md](results-history-charts.md) |

## Image & kernel work

| Story | File |
|---|---|
| Upgrade the Dockerfile base image and re-adapt the patch stack | [base-image-bump.md](base-image-bump.md) |
| Improve the GEMV kernel: methodology, validation, benchmarking | [gemv-kernel-improvement.md](gemv-kernel-improvement.md) |
