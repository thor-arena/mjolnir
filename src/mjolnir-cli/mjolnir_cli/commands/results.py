"""Root commands over the committed raw data: ``mjolnir plot`` (charts,
README, docs, the site) and ``mjolnir history`` (recent bench runs)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich import box
from rich.table import Table

from mjolnir_cli import charts, clidoc, readme
from mjolnir_cli.commands._common import console
from mjolnir_cli.config import load_layout

# ── plot / history ──────────────────────────────────────────────────────────


def make_plot(app: typer.Typer):
    """The ``mjolnir plot`` command, closed over the root ``app``: ``--readme``
    writes that app's reference to docs/cli.md. The composition root injects
    ``app`` (no command module imports the cli)."""

    def plot(
        render_readme: bool = typer.Option(
            False,
            "--readme",
            help="also render README.md from the README.md.j2 template "
            "(every table + number computed from the committed raw data)",
        ),
        readme_template: Optional[Path] = typer.Option(
            None, "--readme-template", help="template override (default: <repo>/README.md.j2)"
        ),
        readme_out: Optional[Path] = typer.Option(
            None, "--readme-out", help="output override (default: <repo>/README.md)"
        ),
        site: Optional[Path] = typer.Option(
            None, "--site", help="also build the interactive benchmark site (index.html) into this directory"
        ),
    ):
        """Render every chart (light + dark PNGs in assets/benchmarks/, from the
        latest default `mjolnir verify` run of each stage + the e2e history).
        --readme also renders README.md and docs/**/*.md from their .md.j2
        templates; --site builds the interactive GitHub Pages site. Everything is
        computed from the committed raw data."""
        layout = load_layout()
        try:
            charts.render_all(layout)
        except RuntimeError as e:
            typer.secho(str(e), fg=typer.colors.RED, err=True)
            raise typer.Exit(1)
        if layout.repo_root:  # research/progress.png: the live ledger, else the newest exported run
            from mjolnir_cli import autoresearch

            autoresearch.render_progress(layout.repo_root, autoresearch.latest_rows(layout.repo_root))
        if render_readme:
            try:
                readme.render_readme(layout, template=readme_template, out=readme_out)
                print(f"[mjolnir] wrote {clidoc.write_reference(layout.repo_root, app)}")
            except (RuntimeError, FileNotFoundError, OSError) as e:
                typer.secho(str(e), fg=typer.colors.RED, err=True)
                raise typer.Exit(1)
        if site is not None:
            from mjolnir_cli import interactive

            interactive.build(layout, site)

    return plot


def _ctx_label(ctx: int | None) -> str:
    if not ctx:
        return "0"
    return f"{ctx // 1024}K" if ctx % 1024 == 0 else str(ctx)


def _history_record_table(r: dict) -> Table:
    """One history record → a small tg-t/s (decode throughput) table."""
    by: dict[int | None, dict] = {}
    for c in r.get("cells", []):
        tg = (c.get("tg_tps") or {}).get("mean")
        if tg is None:
            continue
        by.setdefault(c.get("context"), {})[c.get("concurrency")] = tg
    concs = sorted({cc for v in by.values() for cc in v}, key=lambda x: (x is None, x or 0))
    t = Table(box=box.SIMPLE_HEAD, pad_edge=False)
    t.add_column("ctx")
    for cc in concs:
        t.add_column(f"c={cc} · t/s")
    for ctx in sorted(by, key=lambda x: (x is None, x or 0)):
        t.add_row(_ctx_label(ctx), *[f"{by[ctx][cc]:.1f}" if cc in by[ctx] else "—" for cc in concs])
    return t


def history(
    limit: int = typer.Option(5, "--limit", help="how many recent runs to show"),
    as_json: bool = typer.Option(False, "--json", "-j", help="emit the raw JSON records (for scripts)"),
):
    """Bench results at a glance: decode throughput (t/s) per recent run.

    The raw substrate (every sweep, every cell) is committed in
    benchmarks/history.jsonl — ``--json`` dumps the records."""
    layout = load_layout()
    from mjolnir_cli.history import load_records

    recs = load_records(layout.history_file)[-limit:]
    if not recs:
        typer.secho("no history yet — run: mjolnir bench perf", fg=typer.colors.YELLOW)
        return
    if as_json:
        typer.echo(json.dumps(recs, indent=2))
        return
    for r in recs:
        ts = str(r.get("ts", ""))[:16].replace("T", " ")
        console.print(f"  {ts}   {r.get('backend', '')} — {r.get('image', '')}")
        console.print(_history_record_table(r))
        console.print()
