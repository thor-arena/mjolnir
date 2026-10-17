"""Shared chart plumbing, used by every chart family: the lazy matplotlib
handle (``pyplot``), the context-length tick label (``L_label``), the bundled
Inter Display font (``display_family``) and the PNG writer (``save_png``).

Nothing else in ``charts`` is imported here, so every family can depend on it.
"""

from __future__ import annotations

from pathlib import Path

from mjolnir_cli import theme

DPI = 160


def L_label(L: int) -> str:
    return f"{L // 1024}K" if L >= 1024 and L % 1024 == 0 else str(L)


def pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "matplotlib missing — install the plot extra: uv sync (or pip install 'mjolnir-cli[plot]')"
        ) from e
    plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans", "savefig.dpi": DPI})
    return plt


FONT_DIR = Path(__file__).parent.parent / "fonts"  # Inter Display (SIL OFL 1.1, fonts/OFL.txt)
_FONTS_REGISTERED = False


def _register_fonts() -> None:
    """Make the bundled Inter Display weights known to matplotlib, so the
    hero and infographic render the same on every host (Thor ships Noto Sans
    without its medium / semibold weights)."""
    global _FONTS_REGISTERED
    if _FONTS_REGISTERED:
        return
    from matplotlib import font_manager

    for f in sorted(FONT_DIR.glob("*.ttf")):
        font_manager.fontManager.addfont(str(f))
    _FONTS_REGISTERED = True


def display_family() -> str:
    """A clean display face for the hero numbers: the first static-weight
    sans available (variable fonts render thin in matplotlib)."""
    from matplotlib import font_manager

    _register_fonts()
    for fam in ("Inter Display", "Inter", "Noto Sans", "Nimbus Sans", "Liberation Sans", "DejaVu Sans"):
        try:
            path = font_manager.findfont(fam, fallback_to_default=False)
        except Exception:  # noqa: BLE001
            continue
        if path and (fam == "DejaVu Sans" or "dejavu" not in Path(path).name.lower()):
            return fam
    return "DejaVu Sans"


def save_png(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    theme.fit_footer(fig)
    fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.12, dpi=DPI)
    return path
