"""Accuracy charts (``mjolnir bench accuracy``: per-suite scores and the
tool-call category grid) and their ``ACCURACY_CHARTS`` registry.
"""

from __future__ import annotations

from itertools import groupby
from typing import Callable

from mjolnir_cli import theme
from mjolnir_cli.charts._common import pyplot
from mjolnir_cli.config import RepoLayout
from mjolnir_cli.theme import Mode


def _accuracy_series(r: dict) -> str:
    """Theme series for an accuracy row: official FlashInfer blue, FlashInfer
    on the Mjolnir image hollow blue, Mjolnir fp8 KV hollow orange, the fp6
    Mjolnir configs (and their experiments) filled orange."""
    from mjolnir_cli import e2e

    cfg = str(r.get("config", ""))
    if "MJOLNIR" not in cfg:
        return "NVFP4" if not e2e.is_mjolnir({"image": r.get("image", "")}) else "fi-hnd"
    if cfg in theme.SERIES_STYLE:
        return cfg
    return "NVFP4_MJOLNIR_E2M3"


def _accuracy_rows(layout: RepoLayout) -> tuple[list[dict], list[dict | None], list[str]]:
    from mjolnir_cli import accuracy

    rows = accuracy.latest_legs(accuracy.load_runs(layout))
    return rows, accuracy.baselines(rows), accuracy.suites_in(rows)


def _accuracy_footer(fig, mode: Mode, rows: list[dict]) -> None:
    from mjolnir_cli import accuracy

    runs = sorted({v["run_id"] for r in rows for v in r["runs"].values()})
    theme.footer(
        fig,
        mode,
        f"mjolnir bench accuracy · tool-eval-bench {accuracy.TOOL_COMMIT[:7]} · run{'s' if len(runs) > 1 else ''} "
        f"{', '.join(runs)}",
    )


def chart_accuracy_suites(layout: RepoLayout, mode: Mode):
    """One panel per suite: each config's score as a dot, the baseline as a
    dashed line, the difference printed next to the dot."""
    from mjolnir_cli import accuracy

    rows, bases, names = _accuracy_rows(layout)
    if not rows:
        return None
    plt = pyplot()
    n = len(names)
    fig, axes = plt.subplots(1, n, figsize=(max(7.5, 3.4 * n + 3.2), 1.9 + 0.55 * len(rows)), sharey=True)
    axes = list(axes) if n > 1 else [axes]
    theme.apply(fig, axes, mode)
    ys = list(range(len(rows)))[::-1]
    for ax, name in zip(axes, names):
        # each row's baseline score: its own model's baseline row
        bs = [(base["suites"].get(name) or {}).get("score") if base else None for base in bases]
        vals = [(y, r, (r["suites"].get(name) or {}).get("score"), b) for y, r, b in zip(ys, rows, bs)]
        got = [v for _, _, v, _ in vals if v is not None]
        if not got:
            ax.set_visible(False)
            continue
        refs = [b for b in bs if b is not None]
        lo, hi = min(got + refs), max(got + refs)
        pad = max(4.0, (hi - lo) * 0.35)
        ax.set_xlim(max(0, lo - pad), min(100, hi + pad) + pad * 0.6)
        # one dashed mark per model, spanning its rows (latest_legs keeps them together)
        for _, group in groupby(zip(ys, bs, bases), key=lambda t: id(t[2])):
            span = list(group)
            if span[0][1] is not None:
                ax.vlines(
                    span[0][1],
                    span[-1][0] - 0.4,
                    span[0][0] + 0.4,
                    color=mode.text_dim,
                    lw=1.2,
                    ls=(0, (4, 3)),
                    zorder=2,
                )
        for (y, r, v, b), base in zip(vals, bases):
            if v is None:
                continue
            sid = _accuracy_series(r)
            kw = theme.line_kw(mode, sid)
            if b is not None and r is not base:
                ax.plot([b, v], [y, y], color=kw["color"], lw=2, zorder=3, solid_capstyle="round")
            ax.plot(
                [v], [y], ls="none", **{k: kw[k] for k in ("color", "marker", "mfc", "mec", "mew")}, ms=10, zorder=5
            )
            txt = f"{v:.1f}{accuracy.fmt_delta(accuracy.delta(r, base, name))}"
            ax.annotate(
                txt,
                (v if b is None else max(v, b), y),
                textcoords="offset points",
                xytext=(9, 0),
                va="center",
                fontsize=9.5,
                color=mode.text,
            )
        ax.set_xlabel(f"{accuracy.suite_label(name)} score, %")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([theme.wrap(r["label"], 24) for r in rows], color=mode.text, fontsize=10)
    axes[0].set_ylim(-0.7, len(rows) - 0.3)
    theme.title(
        axes[0],
        mode,
        "Model quality per serving config",
        "tool-eval-bench over each config's OpenAI endpoint · dashed = "
        + (accuracy.baseline_caption(bases, ", ") or "baseline")
        + " · (±) = points vs it"
        + (f" · {accuracy.EXCLUDED_NOTE_SHORT}" if accuracy.any_excluded(rows, bases, names) else ""),
    )
    _accuracy_footer(fig, mode, rows)
    return fig


# a category holds 2–10 scenarios: one partial in a 10-scenario category moves it
# 5 points, so up to that is "on par"; red from 30 points behind
ACC_PAR_POINTS = 5.0
ACC_RED_POINTS = 30.0


def chart_accuracy_categories(layout: RepoLayout, mode: Mode):
    """Tool-call categories × configs: the share of points per cell, colored
    by how far the config falls behind the baseline — green on par (or
    better), then orange, then red the more points it loses."""
    from mjolnir_cli import accuracy

    rows, bases, names = _accuracy_rows(layout)
    pairs = [(r, b) for r, b in zip(rows, bases) if r.get("categories")]
    if not pairs:
        return None
    rows, bases = (list(t) for t in zip(*pairs))
    import numpy as np
    from matplotlib.colors import Normalize

    cats = sorted({c for r in rows for c in r["categories"]})
    pct = np.array([[r["categories"].get(c, {}).get("percent", np.nan) for c in cats] for r in rows], dtype=float)
    ref = np.array(  # per row: its own model's baseline
        [
            [b["categories"].get(c, {}).get("percent", np.nan) for c in cats]
            if b and b.get("categories")
            else [np.nan] * len(cats)
            for b in bases
        ],
        dtype=float,
    )
    delta = pct - ref
    loss = np.where(np.isfinite(delta), np.clip(-delta, 0, None), 0.0)  # points behind the baseline; ahead = 0
    span = max(ACC_RED_POINTS, float(loss.max()) if loss.size else ACC_RED_POINTS)
    plt = pyplot()
    fig, ax = plt.subplots(figsize=(max(9.0, 0.78 * len(cats) + 4.5), 1.8 + 0.75 * len(rows)))
    theme.apply(fig, ax, mode)
    ax.grid(False)
    ax.imshow(
        loss,
        cmap=theme.status_cmap(mode, ACC_PAR_POINTS, span),
        norm=Normalize(vmin=0.0, vmax=span),
        aspect="auto",
        interpolation="nearest",
    )
    for i in range(len(rows) + 1):
        ax.axhline(i - 0.5, color=mode.panel, lw=2)
    for j in range(len(cats) + 1):
        ax.axvline(j - 0.5, color=mode.panel, lw=2)
    for i in range(len(rows)):
        for j in range(len(cats)):
            if np.isfinite(pct[i, j]):
                ax.text(j, i, f"{pct[i, j]:.0f}", ha="center", va="center", fontsize=9.5, color=mode.text)
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels(cats, color=mode.text)
    ax.set_yticks(range(len(rows)))
    marked = [accuracy.leg_excluded(r, b, names) for r, b in zip(rows, bases)]
    ax.set_yticklabels(
        [theme.wrap(r["label"] + (" ‡" if m else ""), 24) for r, m in zip(rows, marked)], color=mode.text
    )
    ax.set_xlabel("tool-call category (names: the table below)")
    theme.title(
        ax,
        mode,
        "Tool calls by category",
        "cell = share of points, % · green = on par with "
        + (accuracy.baseline_caption(bases, ", ") or "the first row")
        + f" (at most {ACC_PAR_POINTS:g} points behind, or ahead) · orange → red = further behind"
        + (f" · {accuracy.CATEGORY_EXCLUDED_NOTE}" if any(marked) else ""),
    )
    _accuracy_footer(fig, mode, rows)
    return fig


ACCURACY_CHARTS: dict[str, Callable] = {
    "accuracy-suites": chart_accuracy_suites,
    "accuracy-categories": chart_accuracy_categories,
}
