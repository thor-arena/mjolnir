"""Append-only benchmark history (``benchmarks/history.jsonl``).

One line per perf sweep repeat:

    {
      "ts": "2026-09-25T14:03:11+00:00", "epoch": 1758801791.123,
      "host": "thor-01", "image": "mjolnir/vllm-thor:qwen38-sm110-v11",
      "model": "Qwen/Qwen3.8-27B", "config": "NVFP4_14_FA4hd256",
      "backend": "FA4-GEMV",
      "gate": {"clean": true, "confirm": 3, "open_ts": 1758801600.0},
      "runs": 5, "warmup_runs": 2, "exact_tg": true,
      "cells": [
        {"concurrency": 1, "context": 8192, "prompt": 2048, "gen": 128,
         "tg_tps": {"mean": 41.2, "std": 0.7, "values": [...]},
         "pp_tps": {"mean": 9830.0, "std": 120.0, "values": [...]}},
        ...
      ]
    }

The file is the raw, git-trackable substrate for the charts and for
"progress over time" — every nightly bump / kernel change adds a row.
"""
from __future__ import annotations

import json
import platform
import time
from pathlib import Path
from typing import Any


def append_record(history_file: Path, record: dict[str, Any]) -> None:
    history_file.parent.mkdir(parents=True, exist_ok=True)
    with history_file.open("a") as f:
        f.write(json.dumps(record) + "\n")


def load_records(history_file: Path) -> list[dict[str, Any]]:
    if not history_file.exists():
        return []
    out = []
    with history_file.open() as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def make_record(s, gate_summary: dict | None, runs: int, warmup_runs: int,
                exact_tg: bool, cells: list[dict[str, Any]],
                raw_relpath: str) -> dict[str, Any]:
    return {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.gmtime())
              .replace("+0000", "+00:00"),
        "epoch": time.time(),
        "host": platform.node() or "unknown",
        "image": s.image,
        "model": s.model,
        "config": s.quant,
        "backend": s.backend_label,
        "gate": gate_summary,
        "runs": runs,
        "warmup_runs": warmup_runs,
        "exact_tg": exact_tg,
        "cells": cells,
        "raw": raw_relpath,
    }


def parse_benchy_json(raw: dict[str, Any]) -> list[dict[str, Any]]:
    """llama-benchy JSON → normalized per-cell records with the metrics we
    chart (tg / pp generation + prefill throughput, each with mean/std/values
    from the per-run samples)."""
    cells = []
    for b in raw.get("benchmarks", []):
        def metric(name: str) -> dict | None:
            m = b.get(name)
            if not isinstance(m, dict):
                return None
            vals = m.get("values")
            out = {"mean": m.get("mean"), "std": m.get("std")}
            if vals is not None:
                out["values"] = [float(v) for v in vals]
            return out

        cell = {
            "concurrency": b.get("concurrency"),
            "context": b.get("context_size"),
            "prompt": b.get("prompt_size"),
            "gen": b.get("response_size"),
            "prefill_phase": bool(b.get("is_context_prefill_phase", False)),
        }
        for src, dst in (("tg_throughput", "tg_tps"),
                         ("pp_throughput", "pp_tps"),
                         ("peak_throughput", "peak_tps"),
                         ("ttfr", "ttfr_ms")):
            m = metric(src)
            if m is not None:
                cell[dst] = m
        cells.append(cell)
    return cells
