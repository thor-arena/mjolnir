"""The Mjolnir chart design system — dark dashboard, one visual language.

Every chart (README heroes, bench reports) uses this palette so the repo's
numbers look like one product. Light text on charcoal, one amber accent
(the hammer), cool neutrals for baseline series.

Color discipline: **one color per label, everywhere.** ``color_for(label)``
is deterministic — the same label gets the same color in every figure, so a
series can be tracked panel to panel and plot to plot. No per-figure
"first-come" assignment.
"""

from __future__ import annotations

import hashlib

# ── palette ──────────────────────────────────────────────────────────────────
BG = "#0D1117"  # figure background (GitHub-dark charcoal)
PANEL = "#151B26"  # axes background
TEXT = "#E6EDF3"  # primary text
TEXT_DIM = "#8B949E"  # secondary text
GRID = "#21262D"  # gridlines
ACCENT = "#FFB224"  # Mjolnir amber — "ours"
TEAL = "#2DD4BF"  # Mjolnir image, non-hero backend
SLATE = "#7C93C4"  # baseline (stock vLLM / the FI kernel)
ROOFLINE = "#F85149"  # red — the Jetson Thor HW limit lines
SERIES_FALLBACK = ["#A78BFA", "#F85149", "#3FB950", "#F0883E"]

# Exact-label registry (the canonical labels of this repo's data).
LABEL_COLORS = {
    # e2e backends (image-level rows in history.jsonl)
    "Mjolnir FA4_GEMV": ACCENT,
    "FA4-GEMV": ACCENT,
    "Mjolnir FlashInfer": TEAL,
    "Native FlashInfer": SLATE,
    # kernel microbench series
    "FA4-1CTA": TEAL,
    "FlashInfer": SLATE,
}


def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def shade(color: str, toward: str, t: float) -> str:
    """Linearly blend ``color`` toward ``toward`` by ``t`` ∈ [0, 1]."""
    a, b = _hex_to_rgb(color), _hex_to_rgb(toward)
    return "#%02X%02X%02X" % tuple(int(av + (bv - av) * t) for av, bv in zip(a, b))


def context_shades(base: str, n: int, max_darken: float = 0.45) -> list[str]:
    """``n`` shades of a bench's label color, darkening toward the figure
    background: shade 0 = the label color itself, later shades (later
    context positions) progressively darker. Every bench keeps its own hue,
    so a group is recognizable by color AND by position."""
    return [shade(base, BG, max_darken * j / max(n - 1, 1)) for j in range(n)]


# Display order (baseline/stock first, "ours" in the middle): used for x
# positions, legends and bar groups.
BACKEND_ORDER = ["Native FlashInfer", "Mjolnir FA4_GEMV", "Mjolnir FlashInfer", "FA4-GEMV", "FA4-1CTA", "FlashInfer"]


def _stable_hash(label: str) -> int:
    return int(hashlib.md5(label.encode()).hexdigest(), 16)


def color_for(label: str) -> str:
    """The one color of a label — deterministic across every chart."""
    if label in LABEL_COLORS:
        return LABEL_COLORS[label]
    low = label.lower()
    if "gemv" in low:
        return ACCENT
    if "native" in low or "stock" in low or "baseline" in low or "vanilla" in low:
        return SLATE
    if "flashinfer" in low:
        return TEAL
    if "fa4" in low:
        return TEAL
    return SERIES_FALLBACK[_stable_hash(label) % len(SERIES_FALLBACK)]


def order_labels(labels: list[str]) -> list[str]:
    """Stable display order: known backends first (in BACKEND_ORDER), then
    unknowns alphabetically."""
    known = [b for b in BACKEND_ORDER if b in labels]
    rest = sorted(set(labels) - set(known))
    return known + rest


def image_first_order(labels: list[str]) -> list[str]:
    """Group order with GEMV first and native/baseline last: the kernel
    work is the hero, the stock image is the floor."""

    def rank(label: str) -> int:
        s = label.lower()
        if "gemv" in s:
            return 0
        if any(k in s for k in ("native", "stock", "baseline", "vanilla")):
            return 99
        return 1

    return sorted(labels, key=lambda label: (rank(label), label))


def apply_theme(fig, ax):
    """Style a figure + one axes with the Mjolnir theme."""
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(PANEL)
    for s in ax.spines.values():
        s.set_visible(False)
    # which="both": minor ticks/labels default to BLACK on dark themes —
    # that's the "dark-on-dark" the eye catches before the data.
    ax.tick_params(colors=TEXT_DIM, labelsize=10, which="both")
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(TEXT_DIM) if ax.xaxis.label is not None else None
    ax.yaxis.label.set_color(TEXT_DIM) if ax.yaxis.label is not None else None
    if ax.title is not None:
        ax.title.set_color(TEXT)


def style_title(ax, title: str, subtitle: str | None = None, fontsize: int = 16):
    """Title (may span two lines via ``\\n``) + one dim subtitle line above
    the axes. The block grows upward; the figure is saved bbox-tight."""
    nlines = title.count("\n") + 1
    pad = (24 if subtitle else 10) + (fontsize + 5) * (nlines - 1)
    # NB: never pass ``loc="left"`` to set_title — matplotlib <=3.6 wipes the
    # title text when loc is given. Set the text via the safe path, then
    # left-align manually.
    t = ax.set_title(title, pad=pad)
    t.set_color(TEXT)
    t.set_fontsize(fontsize)
    t.set_fontweight("bold")
    _, y = t.get_position()
    t.set_position((0.0, y))
    t.set_horizontalalignment("left")
    if subtitle:
        ax.text(0, 1.015, subtitle, transform=ax.transAxes, color=TEXT_DIM, fontsize=10.5, ha="left", va="bottom")


def watermark(fig, label: str = "MJOLNIR"):
    fig.text(
        0.995,
        0.008,
        f"{label}  ·  Jetson Thor  ·  sm_110",
        ha="right",
        va="bottom",
        color=TEXT_DIM,
        fontsize=8,
        alpha=0.75,
        family="monospace",
    )


def legend(ax, handles, labels, ncols: int = 1, loc: str = "best"):
    leg = ax.legend(
        handles,
        labels,
        ncols=ncols,
        loc=loc,
        facecolor=PANEL,
        edgecolor=GRID,
        labelcolor=TEXT,
        fontsize=10,
        framealpha=1.0,
    )
    for lh in leg.get_lines():
        lh.set_alpha(1.0)
    return leg


def errorbar_kwargs(color: str) -> dict:
    # No zorder here — the call site sets it explicitly (avoids a duplicate
    # kwarg).
    return dict(ecolor=color, elinewidth=1.2, capsize=4, capthick=1.2, alpha=0.9)
