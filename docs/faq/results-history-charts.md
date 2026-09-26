# Where results land, and how to compare runs

**Story:** "I ran some benches. Where is the data, how do I compare two
runs, and how do the charts get updated?"

## The three layers

| Layer | Path | What it is |
|---|---|---|
| **raw** | `benchmarks/raw/perf-<ts>/benchy-r*.json` | llama-benchy's full raw JSON per sweep — per-run values, stddev. Grep-able, committed. |
| **history** | `benchmarks/history.jsonl` | append-only log: **one line per gated sweep** (ts, image, model, config, backend, gate summary, per-cell `tg_tps`/`pp_tps` mean/std/values, raw path). The substrate for every chart. |
| **charts** | `assets/benchmarks/*.png` | `kernel.png`, `fa4-vs-fi.png`, `history.png` — rendered from the raw data by `mjolnir plot` (always via `plots.py` + the `theme.py` design system — one visual language for every number in the repo). |

## Compare runs

```bash
mjolnir history                # recent runs as tg t/s tables (context × concurrency)
mjolnir history --limit 10
mjolnir history --json         # raw records for jq / scripts
```

Each run is labeled `backend — image`, e.g.
`FA4-GEMV — mjolnir/vllm-thor:qwen38-sm110-v13`. For a same-cell
comparison (e.g. context 8K, c=1) pull the mean + std from the records or
the raw JSONs; a single-run delta smaller than the cell's std is noise —
the protocol (3 repeats × 5 runs/cell, each in its own clean window) exists
so the spread is visible.

Kernel-level raw results (GEMV benches) follow the same contract: `--out`
JSON in `docker/vllm-thor/fa4-gemv-kernel/` (+ `docs/` for the published
tables), with the methodology recorded next to the numbers.

## Refresh the charts

```bash
mjolnir plot                   # re-renders all charts from history + raw JSONs
mjolnir plot --concurrency 1 --context 8192
```

`bench perf` / `bench ab` already re-render at the end (skipped with a
warning if plotting deps are missing). The rule: **every number gets raw
JSON first**, the chart comes second. Provenance — which image, which
config, which gate window, which raw file — travels with the row in
`history.jsonl`, so a chart spike is always traceable to an artifact.
