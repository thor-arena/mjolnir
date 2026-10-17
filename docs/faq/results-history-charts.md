# Where results land, and how to compare runs

**Story:** "I ran some benches. Where is the data, how do I compare two
runs, and how do the charts get updated?"

## The three layers

| Layer | Path | What it is |
|---|---|---|
| **raw** | `benchmarks/raw/perf-<ts>/benchy-r*.json` | llama-benchy's full raw JSON per sweep — per-run values, stddev. Grep-able, committed. |
| **server log** | `benchmarks/raw/perf-<ts>/vllm-server.log` | the vLLM container's whole log (startup kernel dispatch + the bench window) — the kernel-debugging artifact. |
| **history** | `benchmarks/history.jsonl` | append-only log: **one line per gated run** (ts, image, model, config, backend, gate summary, per-cell `tg_tps`/`pp_tps` mean/std/values, raw path). The substrate for every chart. |
| **kernel raw** | `benchmarks/raw/verify/<stage>/<run_id>.json` | one file per `mjolnir verify` run (schema `mjolnir.verify/1`): commit, image, parameters, every leg's per-L numbers; `"default": true` marks headline runs. See [verify-kernel.md](verify-kernel.md). |
| **charts** | `assets/benchmarks/*.png` (+ `-dark.png`) | rendered from the raw data by `mjolnir plot` — always via the `charts/` package + the `theme.py` design system: one color per backend everywhere (Mjolnir orange, FlashInfer blue, FA2 aqua), light and dark variants, `manifest.json` listing what was rendered from which run. |

## Compare runs

```bash
mjolnir history                # recent runs as tg t/s tables (context × concurrency)
mjolnir history --limit 10
mjolnir history --json         # raw records for jq / scripts
```

Each run is labeled `backend — image`, e.g.
`Mjolnir — narandill/mjolnir-vllm:v0.31.0`. For a same-cell
comparison (e.g. context 8K, c=1) pull the mean + std from the records or
the raw JSONs; a single-run delta smaller than the cell's std is noise —
the protocol (6 measured runs/cell in one clean window, `--repeat` adds
independent windows) exists so the spread is visible. `mjolnir bench perf`
also prints llama-benchy's own terminal table (pooled over the run) beside
the raw dump, and saves the server log next to the raw JSON.

Kernel results follow the same contract: `mjolnir verify show <stage>` prints
the latest run, and two runs of a stage compare cell by cell from their JSON.

## Refresh the charts

```bash
mjolnir plot                   # re-renders all charts from history + raw JSONs
mjolnir plot --readme          # + README.md and docs/kernel/*.md from their .md.j2 templates
mjolnir plot --site _site      # the interactive Plotly page (published by pages.yml)
```

`bench perf` / `bench ab` already re-render at the end (skipped with a
warning if plotting deps are missing). The rule: **every number gets raw
JSON first**, the chart comes second. Provenance — which image, which
config, which gate window, which raw file — travels with the row in
`history.jsonl`, so a chart spike is always traceable to an artifact.

The README and the kernel chapters are **templates** (`README.md.j2`,
`docs/kernel/*.md.j2`): every number and sentence about results comes from the
raw data, and CI fails if the committed README differs from a fresh render. Never
hand-edit `README.md` — edit the template or commit new data.
