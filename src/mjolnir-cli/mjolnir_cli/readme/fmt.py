"""Pure markdown formatting for the README / docs: context-length labels,
µs/ms times, colored ratios and percentages (LaTeX GitHub renders), and the
markdown table. No domain data, no labels for configs — that is ``labels``."""

from __future__ import annotations

from mjolnir_cli import e2e
from mjolnir_cli.verify.stages import NOISE_PCT

_GREEN = "#16a34a"
_RED = "#dc2626"
_GREY = "grey"
_MINUS = "−"


def L_label(L: int) -> str:
    return f"{L // 1024}K" if L >= 1024 and L % 1024 == 0 else str(L)


def fmt_us(t: float | None) -> str:
    if t is None:
        return "—"
    return f"{t / 1000:.2f}&nbsp;ms" if t >= 1000 else f"{t:.1f}&nbsp;µs"


def ratio_md(r: float | None, bold_wins: bool = True) -> str:
    """``1.25×`` colored: green = faster beyond the ±NOISE_PCT noise band, grey =
    within noise, red = slower beyond noise (GitHub renders the LaTeX color)."""
    if r is None:
        return "—"
    noise = NOISE_PCT / 100
    color = _GREEN if r >= 1 + noise else (_GREY if r >= 1 - noise else _RED)
    text = f"{r:.2f}\\times"
    if bold_wins and r >= 1 + noise:
        text = f"\\mathbf{{{r:.2f}\\times}}"
    return f"$\\color{{{color}}}{{{text}}}$"


def color(text: str, color: str = "grey") -> str:
    """Render grey LaTeX text, escaping whitespace for math mode."""
    return rf"$\color{{{color}}}\scriptstyle\text{{{text}}}$"


def pct_md(p: float | None, lower_is_better: bool = True) -> str:
    """A colored % delta: green = better beyond the noise band, grey = within
    noise, red = worse beyond noise. Times: lower is better; throughput: higher."""
    if p is None:
        return "—"
    good = -p if lower_is_better else p
    color = _GREEN if good >= NOISE_PCT else (_RED if good <= -NOISE_PCT else _GREY)
    sign = "+" if p > 0 else (_MINUS if p < 0 else "")
    # We need to double-escape the `\\\%` as Github refuses to render the `%` symbol otherwise
    return f"$\\color{{{color}}}{{\\text{{{sign}{abs(p):.1f}\\\\%}}}}$"


def md_table(header: list[str], rows: list[list[str]], align: list[str] | None = None) -> str:
    align = align or ["---"] + ["--:"] * (len(header) - 1)
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(align) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def ratio_pct(g: float | None, higher_better: bool = True) -> str:
    """The relative change of the measured value, colored by whether it is
    better: ``g`` is the signed gain (positive = Mjolnir better), shown as the
    change of the value itself — +x% throughput, −x% latency (a latency
    reduction, never the reciprocal speedup). Green = better beyond the
    ±E2E_NOISE_PCT band, grey = within it, red = worse beyond it."""
    if g is None:
        return "—"
    noise = e2e.E2E_NOISE_PCT
    color = _GREEN if g >= noise else (_RED if g <= -noise else _GREY)
    shown = g if higher_better else -g
    sign = "+" if shown > 0 else (_MINUS if shown < 0 else "")
    g = shown
    # We need to double-escape the `\\\%` as Github refuses to render the `%` symbol otherwise
    return f"$\\color{{{color}}}{{\\text{{{sign}{abs(g):.1f}\\\\%}}}}$"
