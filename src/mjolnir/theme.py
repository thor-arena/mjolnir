"""The Mjolnir chart design system — dark dashboard, one visual language.

Every chart (README heroes, bench reports) uses this palette so the repo's
numbers look like one product. Light text on charcoal, one amber accent
(the hammer), cool neutrals for baseline series.
"""
from __future__ import annotations

# ── palette ──────────────────────────────────────────────────────────────────
BG = "#0D1117"            # figure background (GitHub-dark charcoal)
PANEL = "#151B26"         # axes background
TEXT = "#E6EDF3"          # primary text
TEXT_DIM = "#8B949E"      # secondary text
GRID = "#21262D"         # gridlines
ACCENT = "#FFB224"       # Mjolnir amber — "ours"
SERIES = {
    "FA4-GEMV": "#FFB224",       # amber — the hero series
    "FA4-1CTA": "#2DD4BF",       # teal
    "FlashInfer": "#7C93C4",     # slate blue — baseline
}
SERIES_FALLBACK = ["#A78BFA", "#F85149", "#3FB950", "#F0883E"]

# Per-backend style order for legends (ours first).
BACKEND_ORDER = ["FA4-GEMV", "FA4-1CTA", "FlashInfer"]


def color_for(backend: str, used: set[str] | None = None) -> str:
    used = used or set()
    if backend in SERIES:
        return SERIES[backend]
    for i, c in enumerate(SERIES_FALLBACK):
        if c not in used:
            return c
    return TEXT


def apply_theme(fig, ax):
    """Style a figure + one axes with the Mjolnir theme."""
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(PANEL)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors=TEXT_DIM, labelsize=10)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(TEXT_DIM) if ax.xaxis.label is not None else None
    ax.yaxis.label.set_color(TEXT_DIM) if ax.yaxis.label is not None else None
    if ax.title is not None:
        ax.title.set_color(TEXT)


def style_title(ax, title: str, subtitle: str | None = None):
    ax.set_title(title, color=TEXT, fontsize=16, fontweight="bold",
                 loc="left", pad=24 if subtitle else 10)
    if subtitle:
        ax.text(0, 1.015, subtitle, transform=ax.transAxes,
                color=TEXT_DIM, fontsize=10.5, ha="left", va="bottom")


def watermark(fig, label: str = "MJOLNIR"):
    fig.text(0.995, 0.008, f"{label}  ·  jetson thor · sm_110",
             ha="right", va="bottom", color=TEXT_DIM, fontsize=8,
             alpha=0.75, family="monospace")


def legend(ax, handles, labels, ncols: int = 1):
    leg = ax.legend(handles, labels, ncols=ncols,
                    facecolor=PANEL, edgecolor=GRID,
                    labelcolor=TEXT, fontsize=10, framealpha=1.0)
    for lh in leg.get_lines():
        lh.set_alpha(1.0)
    return leg


def errorbar_kwargs(color: str) -> dict:
    return dict(ecolor=color, elinewidth=1.2, capsize=4, capthick=1.2,
                alpha=0.9, zorder=4)
