"""End-to-end perf bench — wraps ``llama-benchy`` directly.

We drive llama-benchy (the perf engine behind tool-eval-bench) instead of
tool-eval-bench itself: tool-eval-bench writes llama-benchy's raw JSON to a
temp file and deletes it, keeping only the per-cell mean. Calling
llama-benchy directly gives us the full raw data — per-run values, stddev —
which is what the charts and the history log need.

Protocol per ``--repeat`` sweep (all inside one clean-window gate):
  llama-benchy --base-url ... --model ... \
    --pp 2048 --tg 128 --depth 0 4096 8192 --concurrency 1 2 4 \
    --runs 5 --warmup-runs 2 --latency-mode generation \
    --no-cache --exact-tg --format json --save-result benchmarks/raw/<...>/benchy.json

* ``--runs N``     — N measured runs per (depth × pp × tg × concurrency) cell
* ``--warmup-runs``— discarded warmup runs per shape (JIT/warm-cache state)
* ``--exact-tg``   — pin output length (min_tokens + ignore_eos) so
                      EOS-early-stop can't skew the tg t/s variance
* raw JSON per sweep → benchmarks/raw/<ts>/benchy-r<i>.json (git-trackable)
* normalized rows   → benchmarks/history.jsonl (one line per sweep)
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from mjolnir.config import Settings
from mjolnir import dockerctl
from mjolnir.gate import server_load, wait_for_idle
from mjolnir import history as hist


class BenchError(RuntimeError):
    pass


def find_benchy() -> list[str]:
    """Resolve the llama-benchy invocation: installed binary on PATH first,
    then a uv-managed tool, then ``uvx`` (auto-download)."""
    if shutil.which("llama-benchy"):
        return ["llama-benchy"]
    for cand in (Path.home() / ".local" / "bin" / "llama-benchy",
                 Path.home() / ".cargo" / "bin" / "llama-benchy"):
        if cand.exists():
            return [str(cand)]
    if shutil.which("uv"):
        return ["uvx", "llama-benchy"]
    raise BenchError(
        "llama-benchy not found. Install it once:\n"
        "  uv tool install llama-benchy\n"
        "(or: pipx install llama-benchy)")


def build_command(benchy: list[str], s: Settings, runs: int, warmup_runs: int,
                  exact_tg: bool, depths: list[int], concurrency: list[int],
                  pp: int, tg: int, out_json: Path, api_key: str | None
                  ) -> list[str]:
    cmd = [*benchy,
           "--base-url", s.base_url, "--model", s.served_model,
           "--pp", str(pp), "--tg", str(tg),
           "--depth", *map(str, depths),
           "--concurrency", *map(str, concurrency),
           "--runs", str(runs),
           "--latency-mode", "generation",
           "--no-cache", "--format", "json",
           "--save-result", str(out_json)]
    if exact_tg:
        cmd += ["--exact-tg"]
    if api_key:
        cmd += ["--api-key", api_key]
    return cmd


# Metrics that carry per-run ``values`` (pooled across sweeps for aggregation).
_METRICS = ("tg_tps", "pp_tps", "peak_tps", "ttfr_ms")


def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals)


def _std(vals: list[float]) -> float:
    if len(vals) < 2:
        return 0.0
    m = _mean(vals)
    return (sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5


def _p95(vals: list[float]) -> float:
    """Linear-interpolated 95th percentile."""
    if not vals:
        return None
    s = sorted(vals)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * 0.95
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def _aggregate_cells(sweep_cells: list[list[dict]]) -> list[dict]:
    """Pool the per-run samples of every cell across the ``repeat`` sweeps.

    One logical bench run is a single record: its cells carry the pooled
    per-run ``values`` with ``mean``/``std``/``p95`` recomputed over all
    samples (``repeat × runs``). The history chart then plots one point per
    run instead of one per sweep."""
    from collections import defaultdict
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for cells in sweep_cells:
        for c in cells:
            groups[(c.get("concurrency"), c.get("context"))].append(c)
    out = []
    for (conc, ctx), clist in groups.items():
        base = {k: v for k, v in clist[0].items()
                if k not in _METRICS}
        base["concurrency"], base["context"] = conc, ctx
        for metric in _METRICS:
            pooled: list[float] = []
            for c in clist:
                m = c.get(metric)
                if m and m.get("values"):
                    pooled.extend(m["values"])
            if pooled:
                base[metric] = {"mean": _mean(pooled), "std": _std(pooled),
                                "p95": _p95(pooled), "values": pooled}
        out.append(base)
    out.sort(key=lambda c: (c.get("context") or 0, c.get("concurrency") or 0))
    return out


def _aggregate_gates(gates: list[dict | None]) -> dict | None:
    """Collapse the per-sweep gate summaries into one for the run record."""
    if all(g is None for g in gates):
        return None
    opens = [g["open_ts"] for g in gates
             if g and g.get("open_ts") is not None]
    return {
        "clean": all(g and g.get("clean") for g in gates),
        "confirm": (gates[0] or {}).get("confirm"),
        "repeat": len(gates),
        "open_ts": min(opens) if opens else None,
    }


def run_perf(s: Settings, layout, runs: int = 5, warmup_runs: int = 2,
             exact_tg: bool = True, repeat: int = 3,
             depths: list[int] | None = None,
             concurrency: list[int] | None = None, pp: int = 2048,
             tg: int = 128, gate: bool = True, confirm: int = 3,
             gate_timeout_s: float = 3600, api_key: str | None = None,
             label: str | None = None) -> list[dict]:
    """Run ``repeat`` gated perf sweeps; persist raw + history; return the
    records."""
    depths = depths or [0, 4096, 8192]
    concurrency = concurrency or [1, 2, 4]
    benchy = find_benchy()
    if label:
        s.label = label

    layout.raw_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out_dir = layout.raw_dir / f"perf-{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[mjolnir] perf bench: {repeat} sweep(s), {runs} runs/cell, "
          f"{warmup_runs} warmups/cell, exact-tg={exact_tg}")
    print(f"[mjolnir]   server: {s.base_url}  ({s.backend_label})")
    print(f"[mjolnir]   raw →  {out_dir}")

    records = []
    sweep_cells: list[list[dict]] = []
    sweep_gates: list[dict | None] = []
    for i in range(repeat):
        out_json = out_dir / f"benchy-r{i + 1}.json"
        cmd = build_command(benchy, s, runs, warmup_runs, exact_tg, depths,
                            concurrency, pp, tg, out_json, api_key)
        print(f"\n[mjolnir] sweep {i + 1}/{repeat}:\n  {' '.join(cmd)}")
        gate_summary = None
        if gate:
            if server_load(s.metrics_url) is None:
                raise BenchError(
                    f"vLLM metrics {s.metrics_url} unreachable — is the "
                    f"server up? (mjolnir serve up)")
            # One-shot clean-window check on the server under test (the
            # --port). Wait until its queue is 0/0 for `confirm` samples,
            # THEN start the bench. We do NOT monitor during the run: the
            # bench's own in-flight requests would register as "dirty".
            print(f"[mjolnir] waiting for a clean window "
                  f"({confirm}× 0/0, up to {gate_timeout_s:.0f}s) …")
            open_ts = wait_for_idle(s.metrics_url, confirm=confirm,
                                    timeout_s=gate_timeout_s)
            if open_ts is None:
                print(f"[mjolnir] WARNING: timed out after "
                      f"{gate_timeout_s:.0f}s waiting for a clean window — "
                      f"sweep {i + 1} skipped", file=sys.stderr)
                continue
            gate_summary = {"clean": True, "confirm": confirm,
                            "open_ts": open_ts}
        sweep_start = time.time()
        rc = subprocess.call(cmd)
        if rc != 0:
            raise BenchError(f"llama-benchy exited {rc} — see raw output")
        if not out_json.exists():
            raise BenchError(f"llama-benchy produced no JSON at {out_json}")

        log_path = out_dir / f"vllm-r{i + 1}.log"
        if dockerctl.dump_server_log(sweep_start, log_path):
            print(f"[mjolnir]   server log → {log_path}")
        else:
            print(f"[mjolnir]   (vLLM log not captured — container "
                  f"'{dockerctl.CONTAINER_NAME}' not found/running)",
                  file=sys.stderr)

        raw = json.loads(out_json.read_text())
        cells = hist.parse_benchy_json(raw)
        sweep_cells.append(cells)
        sweep_gates.append(gate_summary)
        print(f"[mjolnir] sweep {i + 1}/{repeat} done ({len(cells)} cells)")

    if not sweep_cells:
        return []
    # One logical run = one history record: pool the per-run samples across
    # all ``repeat`` sweeps so the chart plots a single point per run.
    rec = hist.make_record(s, _aggregate_gates(sweep_gates), runs,
                           warmup_runs, exact_tg, _aggregate_cells(sweep_cells),
                           str(out_dir.relative_to(layout.benchmarks_dir)))
    rec["repeat"] = len(sweep_cells)
    hist.append_record(layout.history_file, rec)
    records.append(rec)
    _print_summary(rec)
    return records


def _print_summary(rec: dict) -> None:
    n = rec.get("repeat", 1) * rec["runs"]
    print(f"\n{'=' * 66}\n  {rec['backend']} — tg t/s "
          f"(pooled over {rec.get('repeat', 1)} sweep(s) × {rec['runs']} runs = "
          f"{n} samples/cell)\n{'=' * 66}")
    print(f"  {'context':>8} {'c=1':>9} {'c=2':>9} {'c=4':>9}")
    rows: dict[int, dict] = {}
    for c in rec["cells"]:
        m = c.get("tg_tps")
        if not m or m.get("mean") is None:
            continue
        ctx = c.get("context")
        rows.setdefault(ctx, {})[c.get("concurrency")] = m["mean"]
    for ctx in sorted(rows):
        row = rows[ctx]
        print(f"  {ctx:>8} " + " ".join(
            f"{row.get(cc, float('nan')):>9.1f}" for cc in (1, 2, 4)))
    print(f"{'=' * 66}")
    print(f"  history: {rec['raw']}  (+1 row to history.jsonl)")
