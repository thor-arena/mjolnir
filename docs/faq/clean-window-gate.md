# The clean-window gate

**Story:** "Why do my benchmarks wait, what does 'clean window' mean, and
how do I check GPU availability myself?"

## The rule

The Thor GPU is **shared with the live vLLM server** (the one serving this
repo's agents) plus a desktop Xorg session. Two consequences:

1. **Never stop/restart/kill the server to "clean" the GPU.** Instead,
   wait for a clean window — the gate polls and the bench proceeds when the
   GPU is provably idle.
2. **The desktop Xorg load can't be gated** (no request-level metric), so
   absolute numbers rest on NCU kernel counters; wall-clock is only for
   **relative ratios inside the same gated window**.

Full discipline: [`../methodology/benchmarking.md`](../methodology/benchmarking.md).

## What "clean" means

`src/mjolnir/gate.py` (the single source of truth — import it or run it,
never re-implement it) polls the server's Prometheus endpoint
`http://127.0.0.1:6001/metrics` every 2 s and reads
`vllm:num_requests_running` / `vllm:num_requests_waiting`. The window is
clean after **N consecutive 0/0 samples** (`--confirm`, default 6 for the
gate CLI, 3 for perf sweeps, 6 for the strict ring-fix bench). A sweep that
sees any non-zero sample mid-window is **aborted (DIRTY) and discarded** —
never averaged in. BENCH tasks re-check 0/0 immediately before each timed
burst.

## Using it

```bash
mjolnir gate --once                  # check now: CLEAN (rc 0) / BUSY (rc 1)
mjolnir gate                         # block until a window opens (confirm 6, 1 h timeout)
mjolnir gate --confirm 6 --timeout 3600
```

Every gated task uses it transparently:

- `mjolnir bench perf` — each `--repeat` sweep runs inside a `CleanWindow`.
- `mjolnir bench kernel <bench-task>` — preflights the metrics endpoint
  (fails fast with a pointer if the server is down; **never** restarts it),
  then waits for its own window inside the container.
- `mjolnir verify` — same, per suite.

## Interpreting results

- A record's `gate` field in `benchmarks/history.jsonl` tells you the
  window's confirm count and whether it stayed clean.
- If a sweep came back DIRTY, you get no history row — re-run rather than
  patching the number.
- Under co-tenancy, cross-window wall-clock comparisons are suspect; the
  ncu achieved-BW tables (L2-fabric BW, roofline-relative) are the
  absolute reference. When in doubt: same window, relative ratio.
