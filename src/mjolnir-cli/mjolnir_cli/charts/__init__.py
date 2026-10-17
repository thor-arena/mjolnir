"""Every README / docs chart, rendered from raw data only.

Inputs: the latest *default* run of each ``mjolnir verify`` stage
(``benchmarks/raw/verify/<stage>/*.json``) and the e2e bench history
(``benchmarks/history.jsonl``). Outputs: ``assets/benchmarks/<name>.png``
(light) + ``<name>-dark.png`` (dark) per chart, plus ``manifest.json`` —
the list of charts that had data, which the README/docs templates read to
decide what to embed. A chart whose stage has no run yet is simply skipped.

``mjolnir plot`` (and ``mjolnir verify ... --plot``) re-renders everything;
the interactive versions live in ``interactive.py`` (GitHub Pages).

This package has one job per module (``tests/cli/test_module_boundaries.py``
pins the layering): ``_common`` holds the shared plumbing (lazy ``pyplot``,
``L_label``, the display font, ``save_png``); one module per chart family
builds that family's figures and imports only ``_common`` from ``charts`` —
``kernel`` (verify stages), ``e2e`` (bench history), ``hero`` (README hero),
``infographic`` (opening infographic), ``accuracy`` (bench accuracy) and
``autoresearch`` (the research progress chart); and this ``__init__`` renders
them all (``render_all``) and exports the API callers outside ``charts`` use.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from mjolnir_cli import theme
from mjolnir_cli.charts import accuracy, e2e, infographic, kernel
from mjolnir_cli.charts import hero as hero_image  # render_all keeps a local named hero
from mjolnir_cli.charts._common import pyplot, save_png
from mjolnir_cli.charts.autoresearch import chart_autoresearch
from mjolnir_cli.charts.e2e import e2e_rows
from mjolnir_cli.config import RepoLayout
from mjolnir_cli.verify.latest import load_verify

__all__ = ["chart_autoresearch", "e2e_rows", "load_verify", "pyplot", "render_all", "save_png"]


def render_all(layout: RepoLayout, data: dict | None = None) -> dict[str, dict]:
    """Render every chart that has data, in both modes; write the manifest.
    Returns ``{chart: {"stage": ..., "light": path, "dark": path}}``."""
    plt = pyplot()
    data = data if data is not None else load_verify(layout)
    out_dir = layout.charts_dir
    manifest: dict[str, dict] = {}
    jobs: list[tuple[str, str, Callable]] = [
        (n, st, (lambda f: lambda m: f(data, m))(fn)) for n, (st, fn) in kernel.KERNEL_CHARTS.items()
    ]
    jobs += [(n, "e2e", (lambda f: lambda m: f(layout, m))(fn)) for n, fn in e2e.E2E_CHARTS.items()]
    jobs += [(n, "accuracy", (lambda f: lambda m: f(layout, m))(fn)) for n, fn in accuracy.ACCURACY_CHARTS.items()]
    for name, stage, build in jobs:
        entry = {"stage": stage}
        for mode in theme.MODES:
            fig = build(mode)
            if fig is None:
                break
            p = save_png(fig, out_dir / f"{name}{mode.suffix}.png")
            plt.close(fig)
            entry[mode.name] = str(p.relative_to(layout.repo_root) if layout.repo_root else p)
        if "light" in entry:
            manifest[name] = entry
    info = infographic.render_infographic(layout)
    hero = hero_image.render_hero(layout)
    # stale renders (data no longer qualifies) must not linger next to the manifest
    keep = {Path(v).name for e in manifest.values() for k, v in e.items() if k in ("light", "dark")}
    for png in out_dir.glob("*.png"):
        if png.name not in keep:
            png.unlink()
    if info is None and layout.repo_root is not None:
        (layout.repo_root / "assets" / infographic.INFOGRAPHIC).unlink(missing_ok=True)
    if hero is not None:
        manifest["hero"] = {
            "stage": "readme",
            "light": str(hero.relative_to(layout.repo_root)),
            "dark": str(hero.relative_to(layout.repo_root)),
        }
    if info is not None:
        manifest["opening-infographic"] = {
            "stage": "e2e",
            "light": str(info.relative_to(layout.repo_root)),
            "dark": str(info.relative_to(layout.repo_root)),
        }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    print(f"[mjolnir] {len(manifest)} charts (light + dark) -> {out_dir}")
    return manifest
