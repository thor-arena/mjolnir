"""End-to-end perf bench — wraps ``llama-benchy`` directly.

We drive llama-benchy (the perf engine behind tool-eval-bench) instead of
tool-eval-bench itself: tool-eval-bench writes llama-benchy's raw JSON to a
temp file and deletes it, keeping only the per-cell mean. Calling
llama-benchy directly gives us the full raw data — per-run values, stddev —
which is what the charts and the history log need.

Protocol per ``--repeat`` sweep (all inside one clean-window gate):
  llama-benchy --base-url ... --model ... \
    --pp 2048 --tg 128 --depth 0 4096 8192 --concurrency 1 2 4 \
    --runs 6 --warmup-runs 2 --latency-mode generation \
    --no-cache --exact-tg --format json --save-result benchmarks/raw/<...>/benchy.json

* ``--runs N``     — N measured runs per (depth × pp × tg × concurrency) cell
* ``--warmup-runs``— discarded warmup runs per shape (JIT/warm-cache state)
* ``--exact-tg``   — pin output length (min_tokens + ignore_eos) so
                      EOS-early-stop can't skew the tg t/s variance
* raw JSON per sweep → benchmarks/raw/<ts>/benchy-r<i>.json (git-trackable)
* the vLLM server log (container start → bench end) →
  benchmarks/raw/<ts>/vllm-server.log (kernel-dispatch debugging)
* normalized rows   → benchmarks/history.jsonl (one line per run)
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


def run_perf(s: Settings, layout, runs: int = 6, warmup_runs: int = 2,
             exact_tg: bool = True, repeat: int = 1,
             depths: list[int] | None = None,
             concurrency: list[int] | None = None, pp: int = 2048,
             tg: int = 128, gate: bool = True, confirm: int = 3,
             gate_timeout_s: float = 3600, api_key: str | None = None,
             label: str | None = None) -> list[dict]:
    """Run ``repeat`` gated perf sweeps (default: one, 6 measured runs/cell);
    persist raw + the vLLM server log + history; print llama-benchy's
    terminal table; return the records."""
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
        rc = subprocess.call(cmd)
        if rc != 0:
            raise BenchError(f"llama-benchy exited {rc} — see raw output")
        if not out_json.exists():
            raise BenchError(f"llama-benchy produced no JSON at {out_json}")

        raw = json.loads(out_json.read_text())
        cells = hist.parse_benchy_json(raw)
        sweep_cells.append(cells)
        sweep_gates.append(gate_summary)
        print(f"[mjolnir] sweep {i + 1}/{repeat} done ({len(cells)} cells)")

    # The vLLM server log for kernel debugging: the WHOLE container log
    # (startup kernel dispatch + every request of the bench), not just the
    # bench window — what the attention backend got picked as is only in
    # the startup lines.
    full_log = out_dir / "vllm-server.log"
    if dockerctl.dump_server_log(None, full_log):
        print(f"[mjolnir]   server log (container start → now) → {full_log}")
    else:
        print(f"[mjolnir]   (vLLM log not captured — container "
              f"'{dockerctl.CONTAINER_NAME}' not found/running)",
              file=sys.stderr)

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
    print_benchy_table(out_dir, rec)
    return records


# ── the llama-benchy terminal table (the one the tool renders natively) ─────
#
# Same layout as llama-benchy's own md table: one row per test (pp rows +
# tg rows per context × concurrency), cells "mean ± std" over the pooled
# per-run values. Rebuilt from the raw benchy-r*.json files so the terminal
# shows the run's numbers next to the raw dump.

# raw-JSON metric fields, in table order.
_BENNY_METRICS = ("pp_throughput", "pp_req_throughput", "tg_throughput",
                  "tg_req_throughput", "peak_throughput",
                  "peak_req_throughput", "ttfr", "est_ppt", "e2e_ttft")

# Table columns: (header, metric key). Full layout when >1 concurrency;
# the req-split columns collapse when concurrency is single.
_COLS_FULL = (("t/s (total)", "total"), ("t/s (req)", "req"),
              ("peak t/s", "peak"), ("peak t/s (req)", "peak_req"),
              ("ttfr (ms)", "ttfr"), ("est_ppt (ms)", "est_ppt"),
              ("e2e_ttft (ms)", "e2e_ttft"))
_COLS_C1 = (("t/s", "total"), ("peak t/s", "peak"),
            ("ttfr (ms)", "ttfr"), ("est_ppt (ms)", "est_ppt"),
            ("e2e_ttft (ms)", "e2e_ttft"))


def _pool_benchy_files(out_dir: Path) -> tuple[str, int, list[dict]]:
    """Pool the per-run values across every sweep's raw JSON. Returns
    (model, max_concurrency, pooled entries) in producer order: context
    depth outermost (ctx-phase rows first, then the standard run),
    concurrency innermost."""
    groups: dict[tuple, list[dict]] = {}
    model = ""
    max_c = 1
    for f in sorted(out_dir.glob("benchy-r*.json")):
        try:
            raw = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        model = raw.get("model", model)
        for b in raw.get("benchmarks", []):
            key = (b.get("context_size") or 0,
                   not bool(b.get("is_context_prefill_phase")),
                   b.get("concurrency") or 1)
            groups.setdefault(key, []).append(b)
            max_c = max(max_c, b.get("concurrency") or 1)
    entries = []
    for key in sorted(groups):
        blist = groups[key]
        first = blist[0]
        base = {"concurrency": first.get("concurrency"),
                "context_size": first.get("context_size") or 0,
                "prompt_size": first.get("prompt_size"),
                "response_size": first.get("response_size"),
                "is_context_prefill_phase":
                    bool(first.get("is_context_prefill_phase"))}
        for name in _BENNY_METRICS:
            vals: list[float] = []
            for b in blist:
                m = b.get(name)
                if isinstance(m, dict) and m.get("values"):
                    vals.extend(float(v) for v in m["values"])
            if vals:
                base[name] = {"mean": _mean(vals), "std": _std(vals)}
        entries.append(base)
    return model, max_c, entries


def _entry_rows(model: str, max_c: int, e: dict) -> list[tuple[str, list]]:
    """One pooled entry → its pp row + tg row (the ctx variants when it is a
    prefix-caching prefill-phase run). Cells = metric dicts (or None)."""
    ctx = e["context_size"] or 0
    conc = e["concurrency"] or 1
    d = f" @ d{ctx}" if ctx else ""
    c = f" (c{conc})" if max_c > 1 else ""
    if e["is_context_prefill_phase"]:
        pp_lbl, tg_lbl = f"ctx_pp{d}{c}", f"ctx_tg{d}{c}"
    else:
        pp_lbl = f"pp{e['prompt_size']}{d}{c}"
        tg_lbl = f"tg{e['response_size']}{d}{c}"
    pp_vals = {"total": e.get("pp_throughput"),
               "req": e.get("pp_req_throughput"),
               "peak": None, "peak_req": None,
               "ttfr": e.get("ttfr"), "est_ppt": e.get("est_ppt"),
               "e2e_ttft": e.get("e2e_ttft")}
    tg_vals = {"total": e.get("tg_throughput"),
               "req": e.get("tg_req_throughput"),
               "peak": e.get("peak_throughput"),
               "peak_req": e.get("peak_req_throughput"),
               "ttfr": None, "est_ppt": None, "e2e_ttft": None}
    cols = _COLS_FULL if max_c > 1 else _COLS_C1
    out = []
    for lbl, vals in ((pp_lbl, pp_vals), (tg_lbl, tg_vals)):
        cells = [vals[k] for _, k in cols]
        if any(m is not None for m in cells):  # skip all-null rows
            out.append((lbl, cells))
    return out


def _pipe_table(headers: list[str], rows: list[list[str]]) -> str:
    """A GitHub-pipe table (llama-benchy's md format): first column
    left-aligned, metric columns right-aligned."""
    aligns = ["<"] + [">"] * (len(headers) - 1)
    widths = [max([len(h)] + [len(r[i]) for r in rows])
              for i, h in enumerate(headers)] if rows \
        else [len(h) for h in headers]

    def data_row(cells: list[str]) -> str:
        return "| " + " | ".join(
            c.rjust(w) if a == ">" else c.ljust(w)
            for c, w, a in zip(cells, widths, aligns)) + " |"

    def sep_cell(a: str, w: int) -> str:
        return ":" + "-" * (w - 1) if a == "<" else "-" * (w - 1) + ":"
    sep = "|" + "|".join(sep_cell(a, w) for w, a in zip(widths, aligns)) + "|"
    return "\n".join([data_row(headers), sep,
                      *map(data_row, rows)])


def print_benchy_table(out_dir: Path, rec: dict) -> None:
    """Print llama-benchy's terminal table for the run (pooled over its
    sweeps), beside the raw JSON dump."""
    model, max_c, entries = _pool_benchy_files(out_dir)
    model = model or rec.get("model", "")
    rows = [[model] + [lbl] +
            [f"{m['mean']:.2f} ± {m['std']:.2f}" if m else ""
             for m in cells]
            for e in entries for lbl, cells in _entry_rows(model, max_c, e)]
    cols = _COLS_FULL if max_c > 1 else _COLS_C1
    headers = ["model", "test"] + [h for h, _ in cols]
    n = rec.get("repeat", 1) * rec["runs"]
    print(f"\n{'=' * 100}\n  llama-benchy — {rec.get('backend')} "
          f"({rec.get('image')})\n  pooled over {rec.get('repeat', 1)} "
          f"sweep(s) × {rec['runs']} runs = {n} samples/cell\n"
          f"{'=' * 100}")
    if rows:
        print(_pipe_table(headers, rows))
    else:
        print("  (no results)")
    print(f"{'=' * 100}")
    print(f"  raw: {out_dir}  (+1 row to history.jsonl)")
