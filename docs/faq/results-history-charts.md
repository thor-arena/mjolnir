# Where results land, and how to compare runs

**Story:** "I ran some benches. Where is the data, how do I compare two
runs, and how do the charts get updated?"

## The three layers

| Layer | Path | What it is |
|---|---|---|
| **raw** | `benchmarks/raw/perf-<ts>/benchy-r*.json` | llama-benchy's full raw JSON per sweep — per-run values, stddev. Grep-able, committed. |
| **server log** | `benchmarks/raw/perf-<ts>/vllm-server.log` | the vLLM container's whole log (startup kernel dispatch + the bench window) — the kernel-debugging artifact. |
| **history** | `benchmarks/history.jsonl` | append-only log: **one line per gated run** (ts, image, model, config, backend, gate summary, per-cell `tg_tps`/`pp_tps` mean/std/values, raw path). The substrate for every chart. |
| **charts** | `assets/benchmarks/*.png` | `kernel-microbench.png`, `vllm-vs-mjolnir-image.png`, `bench-compare.png`, `tg-variability.png`, `ttfr-by-context.png` — rendered from the raw data by `mjolnir plot` (always via `plots.py` + the `theme.py` design system — one visual language for every number in the repo; one color per label, everywhere). The red dotted line on `kernel-microbench.png` is the Jetson Thor DRAM roofline: one hd256 layer's KV at L=8192 (16.8 MiB) ÷ 273 GB/s ≈ 61 µs — a pure bytes÷BW floor from the kernel shape (`roofline_us()` in `plots.py`). No t/s ceiling lines are drawn: a weight-bytes + MTP-acceptance limit is workload-dependent and can't be reliably proven. |

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
the protocol (6 measured runs/cell in one clean window, `--repeat` adds
independent windows) exists so the spread is visible. `mjolnir bench perf`
also prints llama-benchy's own terminal table (pooled over the run) beside
the raw dump, and saves the server log next to the raw JSON.

Kernel-level raw results (GEMV benches) follow the same contract: `--out`
JSON in `docker/vllm-thor/fa4-gemv-kernel/` (+ `docs/` for the published
tables), with the methodology recorded next to the numbers.

## Refresh the charts

```bash
mjolnir plot                   # re-renders all charts from history + raw JSONs
```

`bench perf` / `bench ab` already re-render at the end (skipped with a
warning if plotting deps are missing). The rule: **every number gets raw
JSON first**, the chart comes second. Provenance — which image, which
config, which gate window, which raw file — travels with the row in
`history.jsonl`, so a chart spike is always traceable to an artifact.
