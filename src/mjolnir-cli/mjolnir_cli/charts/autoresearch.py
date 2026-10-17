"""The autoresearch progress chart (``research/progress.png``): every
experiment on one chart, after karpathy/autoresearch. ``mjolnir_cli.autoresearch``
renders it through the ``charts`` facade.
"""

from __future__ import annotations

import re

from mjolnir_cli import theme
from mjolnir_cli.charts._common import L_label, pyplot
from mjolnir_cli.theme import Mode


def chart_autoresearch(rows: list[dict], mode: Mode):
    """Top: the score of every experiment, the kept ones labelled, the running
    best as a staircase. Bottom: the per-L split — each experiment's
    FlashInfer÷Mjolnir ratio at every L over the baseline's — so a gain that
    lives at one L (or comes from a FlashInfer outlier) is visible at a glance."""
    import numpy as np
    from matplotlib.colors import TwoSlopeNorm

    plt = pyplot()
    if not rows:  # before the first exported run: say so instead of a broken image
        fig, ax = plt.subplots(figsize=(13.5, 0.4))
        theme.apply(fig, ax, mode)
        ax.set_axis_off()
        theme.title(
            ax,
            mode,
            "Autoresearch progress: no run exported yet",
            "mjolnir autoresearch export → research/runs/<tag>/, then mjolnir plot --readme",
        )
        return fig
    base = next((r for r in rows if r.get("status") == "baseline" and r.get("per_L")), None)
    Ls = sorted({int(L) for r in rows for L in (r.get("per_L") or {})})
    if Ls:
        fig, (ax, hx) = plt.subplots(
            2, 1, figsize=(13.5, 8.2), sharex=True, gridspec_kw={"height_ratios": [3, 1.35], "hspace": 0.08}
        )
    else:  # rows without per-L numbers (an imported ledger): the score panel only
        fig, ax = plt.subplots(figsize=(13.5, 5.6))
        hx = None
    theme.apply(fig, [a for a in (ax, hx) if a is not None], mode)
    ids = [r["id"] for r in rows]
    scored = [r for r in rows if r.get("score")]
    kept = [r for r in scored if r.get("status") in ("keep", "baseline")]
    rest = [r for r in scored if r.get("status") not in ("keep", "baseline")]
    lo = min((r["score"] for r in scored), default=1.0)
    hi = max((r["score"] for r in scored), default=1.0)
    pad = max(0.02, (hi - lo) * 0.12)
    floor = lo - pad
    ax.scatter(
        [r["id"] for r in rest if r["status"] == "discard"],
        [r["score"] for r in rest if r["status"] == "discard"],
        s=22,
        color=mode.text_dim,
        alpha=0.45,
        zorder=3,
        label="discarded",
        linewidths=0,
    )
    noisy = [r for r in rest if r["status"] == "noisy"]
    if noisy:
        ax.scatter(
            [r["id"] for r in noisy],
            [r["score"] for r in noisy],
            s=30,
            facecolor=mode.panel,
            edgecolor=mode.text_dim,
            linewidths=1.2,
            zorder=3,
            label="noisy (re-evaluated)",
        )
    crashes = [r["id"] for r in rows if r.get("status") == "crash" or not r.get("score")]
    if crashes:
        ax.scatter(
            crashes,
            [floor + pad * 0.25] * len(crashes),
            marker="x",
            s=34,
            color=mode.text_dim,
            zorder=3,
            label="crashed",
        )
    if kept:
        xs, ys, best = [], [], None
        for r in rows:  # the running best, as a staircase across every experiment
            if r in kept and (best is None or r["score"] > best):
                best = r["score"]
            if best is not None:
                xs.append(r["id"])
                ys.append(best)
        ax.step(xs, ys, where="post", color=mode.mjolnir, lw=2.0, zorder=4, label="running best")
        ax.scatter(
            [r["id"] for r in kept],
            [r["score"] for r in kept],
            s=64,
            color=mode.mjolnir,
            edgecolor=mode.panel,
            linewidths=1.5,
            zorder=5,
            label="kept",
        )
        for r in kept:
            h = r.get("hypothesis", "")
            ax.annotate(
                h if len(h) <= 48 else h[:47] + "…",
                (r["id"], r["score"]),
                xytext=(6, 6),
                textcoords="offset points",
                rotation=22,
                fontsize=8.5,
                color=mode.text,
                zorder=6,
            )
    ax.set_ylim(floor, hi + pad * 2.2)
    ax.set_ylabel("FlashInfer ÷ Mjolnir\n(geomean over L, higher is better)")
    ax.yaxis.set_major_formatter(_fmt_x())
    theme.legend(ax, mode, loc="lower right", fontsize=9)
    n_keep = sum(r.get("status") == "keep" for r in rows)
    theme.title(
        ax,
        mode,
        f"Autoresearch progress: {len(rows)} experiments, {n_keep} kept",
        "score = FlashInfer time ÷ Mjolnir time (fp6 e2m3 KV), both legs in the same ncu run"
        + (" · below: each L against the baseline" if Ls else ""),
    )
    if hx is not None and not base:
        hx.set_visible(False)
    if hx is not None and base:
        vals = np.full((len(Ls), len(ids)), np.nan)
        for j, r in enumerate(rows):
            for i, L in enumerate(Ls):
                v, b = (r.get("per_L") or {}).get(str(L)), base["per_L"].get(str(L))
                if v and b:
                    vals[i, j] = v / b
        finite = vals[np.isfinite(vals)]
        span = min(0.5, max(0.1, float(np.nanmax(np.abs(np.log2(finite)))) if finite.size else 0.1))
        norm = TwoSlopeNorm(vmin=2**-span, vcenter=1.0, vmax=2**span)
        hx.grid(False)
        im = hx.imshow(
            vals,
            cmap=theme.diverging_cmap(mode),
            norm=norm,
            aspect="auto",
            interpolation="nearest",
            extent=(min(ids) - 0.5, max(ids) + 0.5, len(Ls) - 0.5, -0.5),
        )
        for j in ids:  # 2px surface gaps between experiments and between L
            hx.axvline(j - 0.5, color=mode.panel, lw=2)
        for i in range(len(Ls) + 1):
            hx.axhline(i - 0.5, color=mode.panel, lw=2)
        for j, r in enumerate(rows):  # L set aside: the FlashInfer reference jumped there
            m = re.search(r"reference jumped: ([^)]*)\)", r.get("reason") or "")
            for lab in m.group(1).split(", ") if m else []:
                if lab in [L_label(L) for L in Ls]:
                    hx.text(
                        r["id"],
                        [L_label(L) for L in Ls].index(lab),
                        "ref?",
                        ha="center",
                        va="center",
                        fontsize=7.5,
                        color=mode.text,
                    )
        hx.set_yticks(range(len(Ls)))
        hx.set_yticklabels([L_label(L) for L in Ls], fontsize=8.5)
        hx.set_ylabel("L, vs baseline")
        cb = fig.colorbar(im, ax=[ax, hx], fraction=0.015, pad=0.01, aspect=12, anchor=(0.0, 0.0), shrink=0.3)
        cb.ax.tick_params(labelsize=8, colors=mode.text_dim)
        cb.outline.set_visible(False)
        cb.ax.yaxis.set_major_formatter(_fmt_x())
    bottom = hx if hx is not None and hx.get_visible() else ax
    bottom.set_xlabel("experiment #")
    bottom.xaxis.set_major_locator(_int_locator())
    ax.set_xlim(min(ids) - 0.8, max(ids) + 0.8)
    theme.footer(
        fig,
        mode,
        f"mjolnir autoresearch · {rows[-1].get('branch', '')}"
        + (" · ref? = FlashInfer reference jumped at that L: set aside by the judge" if hx is not None else ""),
        below_in=0.75 if hx is not None else 0.62,
    )
    return fig


def _fmt_x():
    from matplotlib.ticker import FuncFormatter

    return FuncFormatter(lambda v, _: f"{v:.2f}×")


def _int_locator():
    from matplotlib.ticker import MaxNLocator

    return MaxNLocator(integer=True)
