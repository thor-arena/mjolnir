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
from mjolnir.gate import CleanWindow, PreflightError
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
           "--runs", str(runs), "--warmup-runs", str(warmup_runs),
           "--latency-mode", "generation",
           "--no-cache", "--format", "json",
           "--save-result", str(out_json)]
    if exact_tg:
        cmd += ["--exact-tg"]
    if api_key:
        cmd += ["--api-key", api_key]
    return cmd


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
        s.backend_label = label

    layout.raw_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out_dir = layout.raw_dir / f"perf-{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[mjolnir] perf bench: {repeat} sweep(s), {runs} runs/cell, "
          f"{warmup_runs} warmups/cell, exact-tg={exact_tg}")
    print(f"[mjolnir]   server: {s.base_url}  ({s.backend_label})")
    print(f"[mjolnir]   raw →  {out_dir}")

    records = []
    for i in range(repeat):
        out_json = out_dir / f"benchy-r{i + 1}.json"
        cmd = build_command(benchy, s, runs, warmup_runs, exact_tg, depths,
                            concurrency, pp, tg, out_json, api_key)
        print(f"\n[mjolnir] sweep {i + 1}/{repeat}:\n  {' '.join(cmd)}")
        gate_summary = None
        try:
            if gate:
                print(f"[mjolnir] waiting for a clean window "
                      f"({confirm}× 0/0, up to {gate_timeout_s:.0f}s) …")
                with CleanWindow(s.metrics_url, confirm=confirm,
                                 timeout_s=gate_timeout_s) as gw:
                    rc = subprocess.call(cmd)
                    gate_summary = gw.summary()
                if not gw.clean:
                    print(f"[mjolnir] WARNING: window went DIRTY "
                          f"({len(gw.dirty_samples)} sample(s)) — sweep "
                          f"discarded from the history log", file=sys.stderr)
                    continue
            else:
                rc = subprocess.call(cmd)
        except (PreflightError, TimeoutError) as e:
            raise BenchError(str(e)) from e
        if rc != 0:
            raise BenchError(f"llama-benchy exited {rc} — see raw output")
        if not out_json.exists():
            raise BenchError(f"llama-benchy produced no JSON at {out_json}")

        raw = json.loads(out_json.read_text())
        cells = hist.parse_benchy_json(raw)
        rec = hist.make_record(s, gate_summary, runs, warmup_runs, exact_tg,
                               cells, str(out_json.relative_to(layout.benchmarks_dir)))
        hist.append_record(layout.history_file, rec)
        records.append(rec)
        print(f"[mjolnir] sweep {i + 1}/{repeat} done → {rec['raw']} "
              f"({len(cells)} cells)")

    if records:
        _print_summary(records[-1])
    return records


def _print_summary(rec: dict) -> None:
    print(f"\n{'=' * 66}\n  {rec['backend']} — tg t/s (mean of {rec['runs']} runs/cell)\n{'=' * 66}")
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
