"""Report ``mjolnir bench`` results: the accuracy table, the official vs
Mjolnir comparison, and re-rendering the charts and README after a run.

Used only by ``commands.bench``; it runs no benchmark."""

from __future__ import annotations

import json
from typing import Optional

import typer
from rich import box
from rich.console import Console
from rich.table import Table

from mjolnir_cli import charts, readme


def print_accuracy(doc: dict) -> None:
    from mjolnir_cli import accuracy

    rows = doc.get("legs", [])
    suites = accuracy.suites_in([{"suites": r.get("suites", {})} for r in rows])
    t = Table(box=box.SIMPLE_HEAD, title=f"accuracy · run {doc['run_id']} · tool-eval-bench {' '.join(doc['args'])}")
    t.add_column("config")
    for n in suites:
        t.add_column(f"{accuracy.suite_label(n)} %", justify="right")
    t.add_column("completion", justify="right")
    bases = accuracy.baselines(rows)  # a run may hold several models: each vs its own
    for r, base in zip(rows, bases):
        cells = []
        for n in suites:
            v = (r.get("suites", {}).get(n) or {}).get("score")
            cells.append("—" if v is None else f"{v:.1f}{accuracy.fmt_delta(accuracy.delta(r, base, n))}")
        comp = [x.get("completion_rate") for x in r.get("suites", {}).values() if x.get("completion_rate") is not None]
        t.add_row(r.get("label", r.get("id", "")), *cells, f"{min(comp):.0f}%" if comp else "—")
    if accuracy.any_excluded(rows, bases, suites):
        t.caption = accuracy.EXCLUDED_NOTE
    Console().print(t)


def rerender(layout) -> None:
    try:
        charts.render_all(layout)
    except RuntimeError as e:
        typer.secho(f"(plots skipped: {e})", fg=typer.colors.YELLOW, err=True)
    try:
        readme.render_readme(layout)
    except (RuntimeError, FileNotFoundError) as e:
        typer.secho(f"(readme skipped: {e})", fg=typer.colors.YELLOW, err=True)


def print_compare(
    layout,
    mjolnir_config: Optional[str] = None,
    as_json: bool = False,
    official_config: Optional[str] = None,
    model: Optional[str] = None,
) -> int:
    from mjolnir_cli import e2e

    off, mj = e2e.latest_pair(layout.history_file, mjolnir_config, official_config, model)
    if not off or not mj:
        missing = "the official vLLM image" if not off else "a Mjolnir image"
        typer.secho(f"no run on {missing} in {layout.history_file} — run: mjolnir bench e2e", fg=typer.colors.YELLOW)
        return 4
    rows = e2e.compare(off, mj)
    if as_json:
        typer.echo(
            json.dumps({"official": e2e.side_label(off), "mjolnir": e2e.side_label(mj), "cells": rows}, indent=2)
        )
        return 0
    con = Console()
    t = Table(
        box=box.SIMPLE_HEAD,
        title="official vLLM image → Mjolnir image (gain: + = Mjolnir better)",
        caption=f"official: {e2e.side_label(off)}\nmjolnir:  {e2e.side_label(mj)}\nboth: {e2e.spec_label(off, mj)}"
        + e2e.acceptance_line(off, mj),
    )
    t.add_column("ctx", justify="right")
    t.add_column("c", justify="right")
    for m in e2e.METRICS:
        t.add_column(
            f"{m.label} {m.unit} {'↑' if m.higher_better else '↓'} ({'throughput' if m.higher_better else 'latency'} change)",
            justify="right",
        )
    for r in rows:
        cells = []
        for m in e2e.METRICS:
            a, b, g = r[m.key]
            if a is None or b is None:
                cells.append("—")
                continue
            color = "green" if g >= e2e.E2E_NOISE_PCT else ("red" if g <= -e2e.E2E_NOISE_PCT else "grey50")
            cells.append(
                f"{m.fmt.format(a)} → {m.fmt.format(b)} [{color}]{e2e.pct(g if m.higher_better else -g)}[/{color}]"
            )
        t.add_row(e2e.ctx_label(r["ctx"]), str(r["conc"]), *cells)
    con.print(t)
    h = e2e.hero(off, mj)
    if h:
        con.print(
            f"[bold]1 request @ {e2e.ctx_label(h['ctx'])}:[/bold] decode {e2e.pct(h['tg_tps'][2])} · "
            f"prefill {e2e.pct(h['pp_tps'][2])} · TTFT {e2e.pct(-h['ttft_ms'][2])} · ITL {e2e.pct(-h['itl_ms'][2])} "
            "(throughput: % more tokens/s; latency: % change in ms)"
        )
    return 0
