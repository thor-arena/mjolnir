"""The README opening infographic (``assets/opening-infographic.png``): the
official vLLM image vs the Mjolnir image on the latest e2e history pair, drawn
over ``assets/background.jpeg``.
"""

from __future__ import annotations

from pathlib import Path

from mjolnir_cli import theme
from mjolnir_cli.charts._common import display_family, pyplot
from mjolnir_cli.config import RepoLayout, model_short

INFOGRAPHIC = "opening-infographic.png"


def multiple(gain_pct: float) -> str:
    """A gain of 100% or more as a multiple: +200% -> "3×", +150% -> "2.5×"."""
    m = f"{1 + gain_pct / 100:.1f}"
    return f"{m[:-2] if m.endswith('.0') else m}×"


def render_infographic(layout: RepoLayout) -> Path | None:
    """``assets/opening-infographic.png`` — the README hero.

    ``assets/background.jpeg`` (the Thor-as-Mjolnir art on the left) stays
    untouched. Four stat blocks fill the dark right half — decode, prefill,
    time to first token, inter-token latency — each with the gain as the big
    number and the two measured values as bars; the claim, the scenario and
    the provenance sit bottom-left, under the hammer (a bottom scrim keeps
    them readable over the handle). Every number comes from the latest
    history pair (:func:`mjolnir_cli.e2e.latest_pair`): native FlashInfer on
    the official vLLM image vs the Mjolnir backend on the Mjolnir image, one
    request at the longest context both measured."""
    import numpy as np
    from matplotlib.patches import FancyBboxPatch

    from mjolnir_cli import e2e

    if layout.repo_root is None:
        return None
    off, mj = e2e.latest_pair(layout.history_file)
    h = e2e.hero(off, mj) if off and mj else None
    bg_path = layout.repo_root / "assets" / "background.jpeg"
    if not h or not bg_path.exists():
        print("[mjolnir] no official-vLLM / Mjolnir e2e pair in history — infographic skipped (run: mjolnir bench e2e)")
        return None

    plt = pyplot()
    bg = plt.imread(str(bg_path))
    H, W = bg.shape[:2]
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.imshow(bg, extent=(0, W, 0, H), zorder=0)
    # scrims: navy on the right (the stats), and up from the bottom-left
    # corner (the title block over the hammer's handle); clear elsewhere
    gx, gy = np.meshgrid(np.linspace(0, 1, 512), np.linspace(1, 0, 256))  # row 0 = top
    right = np.clip((gx - 0.42) / 0.22, 0, 1) ** 1.5 * 0.76
    bottom = np.clip((0.58 - gy) / 0.30, 0, 1) ** 0.9 * 0.93 * np.clip((0.66 - gx) / 0.16, 0, 1)
    scrim = np.zeros((256, 512, 4))
    scrim[..., :3] = np.array([3, 7, 16]) / 255
    scrim[..., 3] = np.maximum(right, bottom)
    ax.imshow(scrim, extent=(0, W, 0, H), aspect="auto", zorder=1)

    disp = display_family()
    white, dim, faint = "#ffffff", "#b4bfcc", "#7f8a97"
    c_off, c_mj = theme.DARK.flashinfer, theme.LIGHT.mjolnir
    ver = str(off.get("image", "")).split(":")[-1].split("-")[0].lstrip("v")
    model = model_short(mj)
    quant = str(mj.get("config", "")).split("_")[0]
    kv = "fp6 KV" if "E2M3" in str(mj.get("config", "")) else "fp8 KV"

    from matplotlib import patheffects

    halo = [patheffects.withStroke(linewidth=3, foreground=(0.01, 0.03, 0.06, 0.55))]

    def T(xf, yf, text, size, color=white, weight="normal", ha="left", **kw):
        return ax.text(
            xf * W, yf * H, text, fontsize=size, color=color, weight=weight, ha=ha, family=disp, zorder=6, **kw
        )

    # ── bottom-left: claim, scenario, provenance ──────────────────────────
    BL, BY = 0.035, 0.05  # left margin; lift of the whole block off the bottom edge
    T(BL, BY + 0.255, "JETSON AGX THOR   ·   END-TO-END SERVING", 10, dim, "semibold", path_effects=halo)
    title = T(BL, BY + 0.145, "Mjolnir", 34, white, "bold", path_effects=halo)
    x_after = title.get_window_extent(renderer=fig.canvas.get_renderer()).x1
    ax.text(
        x_after + 14,
        (BY + 0.145) * H,
        f"vs vLLM {ver}",
        fontsize=34,
        color=dim,
        family=disp,
        zorder=6,
        path_effects=halo,
    )
    T(
        BL,
        BY + 0.085,
        f"1 request  ·  {e2e.ctx_label(h['ctx'])} context  ·  {model} {quant}  ·  {kv}  ·  {e2e.spec_short(off, mj)}"
        "  ·  vs the official image's native FlashInfer",
        10.5,
        dim,
        path_effects=halo,
    )
    T(
        BL,
        BY + 0.035,
        f"{off.get('image', '')}  vs  {mj.get('image', '')}  ·  llama-benchy, gated clean windows",
        8.5,
        faint,
    )

    # ── right: four stat blocks ────────────────────────────────────────────
    blocks = (
        ("tg_tps", "faster decode", "slower decode", "tokens / s", "{:.1f}"),
        ("pp_tps", "faster prefill", "slower prefill", "prompt tokens / s", "{:,.0f}"),
        ("ttft_ms", "lower TTFT", "higher TTFT", "time to first token", None),
        ("itl_ms", "lower ITL", "higher ITL", "inter-token latency", "{:.1f} ms"),
    )
    L, R, gap = 0.56, 0.96, 0.045
    cw = (R - L - gap) / 2
    tops = (0.93, 0.48)

    def fmt_ttft(v: float) -> str:
        return f"{v / 1000:.2f} s" if v >= 1000 else f"{v:,.0f} ms"

    for i, (key, better, worse, unit, fmt) in enumerate(blocks):
        va, vb, gain = h.get(key) or (None, None, None)
        if va is None or vb is None or gain is None:
            continue
        x0 = L + (i % 2) * (cw + gap)
        y0 = tops[i // 2]
        # throughput: the change in tokens/s, a doubling or more as a multiple
        # (+200% reads "3×"); latency: the change in ms (−27% = 27% fewer ms —
        # a reduction, never the reciprocal speedup)
        if key in ("ttft_ms", "itl_ms"):
            big = e2e.pct(-gain).replace(".0%", "%")
        else:
            big = multiple(gain) if gain >= 100 else e2e.pct(gain)
        T(x0, y0 - 0.115, big, 50, white, "bold")
        T(x0, y0 - 0.178, better if gain >= 0 else worse, 16.5, white)
        T(x0 + cw, y0 - 0.178, unit, 10.5, faint, ha="right")
        X, BW = x0 * W, cw * W
        vmax = max(va, vb)
        for j, (lbl, v, col) in enumerate((("vLLM", va, c_off), ("Mjolnir", vb, c_mj))):
            y = (y0 - 0.255 - j * 0.085) * H
            ax.text(X, y + 9, lbl, fontsize=9.5, color=dim, family=disp, zorder=6)
            ax.text(
                X + BW,
                y + 9,
                fmt_ttft(v) if fmt is None else fmt.format(v),
                fontsize=11,
                color=white,
                weight="bold",
                ha="right",
                family=disp,
                zorder=6,
            )
            bh = 0.014 * H
            for width, face, z in ((BW, (1, 1, 1, 0.10), 5), (max(BW * v / vmax, bh), col, 6)):
                ax.add_patch(
                    FancyBboxPatch(
                        (X, y - bh),
                        width,
                        bh,
                        boxstyle=f"round,pad=0,rounding_size={bh / 2:.1f}",
                        facecolor=face,
                        edgecolor="none",
                        zorder=z,
                    )
                )

    out = layout.repo_root / "assets" / INFOGRAPHIC
    fig.savefig(out, dpi=100)
    plt.close(fig)
    print(f"[mjolnir] wrote {out}")
    return out
