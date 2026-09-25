# Tool-Eval-Bench Perf Runs — What the Throughput Path Actually Measures

> **Status:** Final · **Date:** 2026-09-25

Research notes on the performance path of [SeraphimSerapis/tool-eval-bench](https://github.com/SeraphimSerapis/tool-eval-bench):
what it measures, its run/repeat model, how it aggregates results, and what happens
to the per-run raw data. Sources: the full source of tool-eval-bench (local clone,
commit `bd35ba9`, 2026-09-24), of its perf engine
[eugr/llama-benchy](https://github.com/eugr/llama-benchy) (local clone, commit
`e9be344`, 2026-07-10), and vLLM's `vllm bench serve` source on `main`.

## TL;DR

- The perf path is **not a loop in tool-eval-bench** — the CLI shells out to
  **llama-benchy** once per test point, and llama-benchy itself repeats each test point
  `--runs` times. The flag to turn up is **`--benchy-runs`** (default **3**). There is no
  `--perf-runs` / `--repeat` / `--n-runs` flag and no environment variable.
- `--benchy-runs 3` is the default, so the stock command is not a single run: it performs
  **3 measured runs per test point** (6 test points for `--depth 0,4096,8192
  --concurrency 1,2,4` → 18 measured runs). The single table row per cell is the **mean**
  of those runs.
- Aggregation is **mean ± std** (numpy) inside llama-benchy; tool-eval-bench keeps only
  the **mean**. No median, no min. Per-run raw values exist in llama-benchy's JSON
  (`benchmarks[i].<metric>.values`), but tool-eval-bench writes that JSON to a
  **temp file and deletes it** — nothing per-run is persisted.
- For 5 stable runs per cell: `--benchy-runs 5`, plus `--no-warmup
  --benchy-args="--warmup-runs 2 --exact-tg"` to hand warmup to llama-benchy (see the
  warmup inversion below).
- Computing stddev across runs requires llama-benchy's raw JSON: either a ~3-line patch
  to keep it, or running llama-benchy directly with `--save-result` — which is what
  mjolnir does (Section "Why mjolnir wraps llama-benchy directly").

## What the Perf Path Measures

The CLI argument group `"throughput benchmark (llama-benchy)"`
(`src/tool_eval_bench/cli/legacy_parser.py` lines 288–336), documented in the project's
`docs/benchmarks.md` and `docs/cli-reference.md`:

| Flag | Type / default | Exact help text | Effect |
|---|---|---|---|
| `--perf` | store_true | "Run throughput benchmark before tool-call scenarios" | run the llama-benchy sweep, then scenarios |
| `--perf-only` | store_true | "Run ONLY throughput benchmark (skip tool-call scenarios)" | writes a standalone Markdown report and exits |
| `--pp` | int, 2048 | "Prompt tokens (default: 2048)" | prompt size |
| `--tg` | int, 128 | "Generation tokens (default: 128)" | generation size |
| `--depth` | str, `"0,4096,8192"` | "Context depths, comma separated (default: '0,4096,8192')" | context sizes |
| `--concurrency` | str, `"1,2,4"` | "Concurrency levels (default: '1,2,4')" | concurrency levels |
| **`--benchy-runs`** | **int, 3** | **"Measurement runs per test point (default: 3)"** | **→ llama-benchy `--runs`: measured repeats per (depth × pp × tg × concurrency) cell** |
| `--benchy-latency-mode` | `"api"` / `"generation"` / `"none"`, default `"generation"` | "Latency measurement mode (default: generation)" | how latency is estimated |
| `--benchy-args` | str, None | "Pass-through args for llama-benchy (quoted string)" | arbitrary llama-benchy flags, e.g. `--warmup-runs`, `--exact-tg` |
| `--tokenizer` | str, None | "Path to a local tokenizer.json … Used to override auto-detection" | tokenizer for prompt construction |
| `--skip-coherence` | store_true | "Deprecated: llama-benchy coherence check is now always skipped (retained for compatibility)" | no-op |
| `--no-warmup` | store_true | "Skip server warm-up request" | inverts *which side* warms — see below |
| `--trials` | int, 1 | "Number of trial runs for statistical rigor (default: 1)" | **tool-call scenarios only** — the perf-only path returns before the trials loop (`dispatch.py` lines 921–923) |

No repetition-related environment variables exist (only
`TOOL_EVAL_API_KEY/BACKEND/BASE_URL/HEADERS/HOST/MODEL/PORT/PROVIDER/SESSION_HEADER`).

llama-benchy's own flags (reachable via `--benchy-args`), from `src/llama_benchy/config.py`:

| Flag | Type / default | Exact help text |
|---|---|---|
| `--runs` | int, 3 | "Number of runs per test - default: 3" |
| `--warmup-runs` | int, 1 | "Number of discarded warmup runs per test shape - default: 1. Also controls discarded probes for --latency-mode generation. Does not affect the initial prompt-adaptation warmup." |
| `--no-warmup` | store_true | "Skip warmup phase" |
| `--exact-tg` | store_true | "Force output length to match --tg by sending min_tokens=<tg> and ignore_eos=true in benchmark requests." |
| `--save-result` | str, None | "File to save results to" |
| `--format` | `md` / `json` / `csv`, `md` | "Output format" |
| `--save-total-throughput-timeseries` / `--save-all-throughput-timeseries` | store_true | per-second / per-request throughput series in JSON |
| `--emit-progress` | str, None | "Emit benchmark progress events as JSONL to PATH (or '-' for stdout). Schema: docs/progress-schema.md." |
| `--post-run-cmd` | str, None | "Command to execute after each test run" |
| `--no-cache` | store_true | "Ensure unique requests to avoid prefix caching and send cache_prompt=false to the server" |

tool-eval-bench's command builder (`_build_command`,
`src/tool_eval_bench/runner/llama_benchy.py` lines 172–277) always appends:
`--runs <benchy-runs>`, `--no-cache`, `--skip-coherence`, `--no-adapt-prompt`,
`--format json`, `--save-result <tempfile>`, `--emit-progress -` (unless the user already
set it), then the user's `--benchy-args` verbatim **last**.

## Run/Repeat Model

`--benchy-runs` controls the **number of measured (counted) runs per test point** — i.e.
per (depth, pp, tg, concurrency) cell — passed verbatim as llama-benchy's `--runs`
(`dispatch.py` line 372: `runs=args.benchy_runs` → `llama_benchy.py` line 236:
`cmd.extend(["--runs", str(runs)])`).

In llama-benchy's main loop (`runner.py` lines 96–116): `total_runs = num_runs +
warmup_runs`; the first `warmup_runs` iterations are discarded, and each measured
iteration is one **batch of `concurrency` concurrent requests**. It is *not* "run the
whole sweep N times" — no such flag exists; a full-sweep repetition has to be a shell
loop around the CLI invocation.

## Warmup Layers and the Inversion

There are two warmup layers:

1. tool-eval-bench's own: one small request via `warmup_server` (`dispatch.py`
   lines 492–502, `cli/probe.py` line 110).
2. llama-benchy's own: an initial `client.warmup()` probe, discarded latency probes, and
   per-test-shape discarded runs, governed by `--warmup-runs` (default 1) unless
   `--no-warmup` forces them all to 0 (`runner.py` lines 63–87, 108–116).

The inversion (`dispatch.py` line 378: `skip_warmup=not args.no_warmup`, comment: "We
already warmed the server, so skip llama-benchy's own warm-up and save two requests"):

- **Default** (no `--no-warmup`): tool-eval-bench does the one server warmup, and
  llama-benchy receives `--no-warmup` → **0** per-shape warmup runs → exactly
  `--benchy-runs` measured runs per cell.
- **With `--no-warmup`**: tool-eval-bench skips its warmup, and llama-benchy runs its
  own warmup phase plus `--warmup-runs` (default 1, raise via
  `--benchy-args="--warmup-runs 2"`) discarded runs per cell.

Consequence: passing `--benchy-args="--warmup-runs 2"` **without** `--no-warmup` is a
no-op (llama-benchy still gets `--no-warmup`).

## Aggregation

Per cell, llama-benchy builds `BenchmarkMetric(mean=np.mean, std=np.std,
values=[raw per-run numbers])` over the measured runs (`results.py` lines 74–82,
439–442); its Markdown table prints `mean ± std`. tool-eval-bench's parser keeps **only
the mean** (`_stat_mean`, `llama_benchy.py` lines 336–340) into `ThroughputSample`, which
is all that reaches the Markdown report (`storage/reports/throughput.py`) and the SQLite
row (`persist_run` — counts/config only). **No median, no min, no per-run values in any
tool-eval-bench output.**

## Per-Run Raw Numbers (JSON) — and the Deletion

llama-benchy's `--format json` output (`results.py` lines 490–505): top level
`version, timestamp, latency_mode, latency_ms, model, prefix_caching_enabled,
max_concurrency` plus `benchmarks[]` (one entry per test cell) with
`concurrency, context_size, prompt_size, response_size, is_context_prefill_phase`, and
each metric `pp_throughput / pp_req_throughput / tg_throughput / tg_req_throughput /
peak_throughput / peak_req_throughput / ttfr / est_ppt / e2e_ttft` =
`{"mean":…, "std":…, "values":[…]}` where **`values` is one entry per measured run** —
exactly what cross-run stddev needs. Optional `throughput_over_time` /
`requests_throughput_over_time` time series when
`--save-total-throughput-timeseries` / `--save-all-throughput-timeseries` are set.

**But tool-eval-bench deletes it.** `_build_command` points llama-benchy's
`--save-result` at a `NamedTemporaryFile` (`llama_benchy.py` lines 439–458); after
parsing, the `finally` block deletes it (lines 625–630). The full raw JSON does survive
in-memory as `LlamaBenchyResult.raw_json`, but nothing in the CLI consumes or persists
that field. The `--benchy-args="--save-result /path.json"` trick **breaks the run**: the
duplicate flag makes llama-benchy write to `/path.json`, the temp file is never created,
and tool-eval-bench raises "llama-benchy did not produce JSON output" (check at lines
598–603). `--json` / `--json-file` only cover tool-call runs — `--perf-only` returns
before the JSON emission path (`dispatch.py` lines 921–923 vs 1597–1628).

Ways to get per-run raw data:

1. **Patch** (~3 lines, recommended for tool-eval-bench users): in `run_llama_benchy`,
   copy the temp JSON to the output dir before deletion (e.g.
   `<output_dir>/llama-benchy-raw-<ts>.json`, or thread an `output_dir` param into
   `cli/perf.py → runner.llama_benchy.run_llama_benchy` and do
   `shutil.copyfile(output_file, raw_path)` right after `raw_data = json.loads(...)` at
   line 605).
2. **Run llama-benchy directly** (loses tool-eval-bench's tokenizer resolution, warmup,
   and progress UX):

   ```bash
   uvx llama-benchy --base-url http://localhost:8000/v1 --model <model> \
     --pp 2048 --tg 128 --depth 0 4096 8192 --concurrency 1 2 4 \
     --runs 5 --warmup-runs 2 --latency-mode generation \
     --no-cache --no-adapt-prompt --format json --save-result runs/llama-benchy-raw.json
   ```

## Recommended Command — 5 Stable Runs per Cell

```bash
tool-eval-bench \
  --seed 42 --backend litellm --api-key ... --base-url http://localhost:8000/v1 --model ... \
  --perf --perf-only \
  --pp 2048 --tg 128 --depth "0,4096,8192" --concurrency "1,2,4" \
  --benchy-runs 5 \
  --no-warmup --benchy-args="--warmup-runs 2 --exact-tg" \
  --output-dir ./runs
```

- `--benchy-runs 5`: 5 measured runs per cell (6 cells → 30 measured runs).
- `--no-warmup` (tool-eval-bench): hands warmup to llama-benchy so the per-shape
  discarded runs actually take effect (see the inversion above).
- `--benchy-args="--warmup-runs 2"`: 2 discarded warmup runs per cell before the 5
  measured ones (keeps JIT/warm-cache state constant).
- `--exact-tg` (optional, vLLM-compatible): pins output length to `--tg` via
  `min_tokens`/`ignore_eos`, removing EOS-early-stop variance from tg t/s.
- To repeat the whole sweep independently instead of / in addition to per-cell repeats:
  wrap the invocation in a shell loop (each iteration gets its own run id + Markdown
  report under `--output-dir`).
- Drop `--no-warmup --benchy-args` to keep the original behavior (one tool-eval-bench
  warmup request, 0 per-shape warmups).

## vLLM Upstream Note

`vllm bench serve` (`vllm/benchmarks/serve.py` on `main`) has **no "repeat the whole
benchmark N times" flag** — a run sends `--num-prompts` (default 1000) requests at
`--request-rate` / `--max-concurrency` and reports mean + percentile
(p50/p90/… of ttft/tpot/itl/e2el via `--percentile-metrics`) metrics *within that one
run*, optionally written to JSON with `--save-result` (`+ --save-detailed` for
per-request data); for N independent repetitions you loop the command in a shell, or use
`vllm bench sweep serve` for multi-configuration sweeps in one call. It also cannot be
called through tool-eval-bench, so switching means giving up the unified
tool-call+perf report/SQLite storage.

## Method

- Shallow clones: tool-eval-bench @ `bd35ba9` and llama-benchy @ `e9be344` (both
  2026-09-25). Files read in full: `legacy_parser.py`, `cli/perf.py`,
  `runner/llama_benchy.py`, `dispatch.py` (perf path + warmup + JSON),
  `runner/throughput.py` (`ThroughputSample`), `storage/reports/throughput.py`,
  `run_io.py`, `api.py`, `cli/helpers.py`, `cli/probe.py`, `docs/cli-reference.md`,
  `docs/benchmarks.md`, `README.md`; llama-benchy `config.py`, `results.py`,
  `runner.py`, `README.md`.
- vLLM: fetched `vllm/benchmarks/serve.py` from `main` (argument parser read);
  `benchmarks/benchmark_serving.py` is now a deprecation shim pointing to
  `vllm bench serve`.
- Cross-references: tool-eval-bench source ↔ its docs (`benchmarks.md` documents
  `--benchy-runs` default 3, "Measurement iterations per test point") ↔ llama-benchy
  source/README all agree; no contradictions found.
- Gaps: `vllm bench serve` `--num-prompts` default confirmed only from the module
  docstring (the flag is defined in a dataset-args helper, not read in full);
  llama-benchy `client.warmup()` internals not read line-by-line (behavior inferred
  from `runner.py` call sites).

## Why mjolnir Wraps llama-benchy Directly

The findings above define mjolnir's benchmarking design. tool-eval-bench's perf path is
useful for its unified tool-call + throughput reporting, but it is a lossy funnel for
raw data: it keeps only the per-cell mean, discards stddev, and actively **deletes**
llama-benchy's raw JSON (temp file, removed in a `finally` block), with the
`--save-result` override breaking the run rather than working.

Therefore mjolnir's perf runner ([`src/mjolnir/benchy.py`](../../src/mjolnir/benchy.py))
**wraps llama-benchy directly** instead of going through tool-eval-bench: it invokes
llama-benchy with the same protocol as the recommended command above — `--runs 5`
measured and `--warmup-runs 2` discarded per (depth × concurrency) cell,
`--latency-mode generation`, `--no-cache --exact-tg` — inside a single clean-window gate,
and persists llama-benchy's full JSON (per-run `values` for every metric) to
[`benchmarks/raw/`](../../benchmarks/raw/) so that per-run stddev, regressions, and
charts are computed from raw data, with normalized summary rows written to
[`benchmarks/history.jsonl`](../../benchmarks/history.jsonl).
