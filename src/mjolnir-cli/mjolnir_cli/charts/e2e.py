"""End-to-end serving charts (decode, prefill) from the bench history
(``benchmarks/history.jsonl``, ``mjolnir bench ab`` / ``bench e2e``), the
per-config site rows the interactive page shares (``e2e_rows``), and their
``E2E_CHARTS`` registry.
"""

from __future__ import annotations

from typing import Callable

from mjolnir_cli import theme
from mjolnir_cli.charts._common import L_label, pyplot
from mjolnir_cli.config import DEFAULT_MODEL, RepoLayout, model_of
from mjolnir_cli.history import load_records
from mjolnir_cli.theme import Mode


# (series id, label): native vLLM backends come ONLY from the official image;
# Mjolnir only from a Mjolnir image (see mjolnir_cli.e2e).
E2E_SERIES = (
    ("NVFP4", "FlashInfer · official vLLM"),
    ("NVFP4_FA2", "FlashAttention 2 · official vLLM"),
    ("NVFP4_MJOLNIR", "Mjolnir · fp8 KV"),
    ("NVFP4_MJOLNIR_E2M3", "Mjolnir · fp6 KV"),
    ("NVFP4_MJOLNIR_E2M3_LMH8_VT64K", "Mjolnir · fp6 KV + fp8 lm_head + 64K draft vocab"),
)
MJOLNIR_E2E_CONFIGS = ("NVFP4_MJOLNIR", "NVFP4_MJOLNIR_E2M3", "NVFP4_MJOLNIR_E2M3_LMH8_VT64K")


def e2e_rows(layout: RepoLayout, model: str = DEFAULT_MODEL) -> dict[str, dict]:
    """Latest history record per series: stock configs on the official vLLM
    image, the Mjolnir config on a Mjolnir image. Stock backends measured on
    the Mjolnir image are ignored (that image patches FlashInfer too). One
    model's runs: a series is one config of one model."""
    from mjolnir_cli import e2e

    latest: dict[str, dict] = {}
    for r in load_records(layout.history_file):
        cfg = r.get("config")
        if not r.get("cells") or model_of(r) != model:
            continue
        ok = (cfg in MJOLNIR_E2E_CONFIGS and e2e.is_mjolnir(r)) or (
            cfg in ("NVFP4", "NVFP4_FA2") and e2e.is_official(r)
        )
        if ok and (cfg not in latest or r.get("epoch", 0) >= latest[cfg].get("epoch", 0)):
            latest[cfg] = r
    return latest


def _e2e_series(rec: dict) -> tuple[str, str]:
    """(theme series id, legend label) for one side of the e2e pair."""
    from mjolnir_cli.config import BACKEND_LABELS

    cfg = str(rec.get("config", ""))
    label = dict(E2E_SERIES).get(cfg) or BACKEND_LABELS.get(cfg, cfg)
    if cfg in theme.SERIES_STYLE:
        return cfg, label
    return ("NVFP4_MJOLNIR_E2M3" if "MJOLNIR" in cfg else "NVFP4"), label


def _e2e_chart(layout: RepoLayout, mode: Mode, metric: str, ylabel: str, ttl: str, sub: str):
    """The README table as bars: the same pair (official vLLM vs the Mjolnir
    config, :func:`mjolnir_cli.e2e.latest_pair` — each side's runs merged cell
    by cell), every (requests, context) cell either side measured, one panel
    per request count; a cell one side lacks is marked "not measured" instead
    of being dropped."""
    from mjolnir_cli import e2e

    off, mj = e2e.latest_pair(layout.history_file)
    if not off or not mj:
        return None
    cells = e2e.all_cells(off, mj)
    concs = sorted({c for c, _ in cells})
    if not concs:
        return None
    plt = pyplot()
    fig, axes = plt.subplots(1, len(concs), figsize=(4.6 * len(concs) + 1.2, 4.8), sharey=True, squeeze=False)
    axes = list(axes[0])
    theme.apply(fig, axes, mode)
    sides = [(rec, *_e2e_series(rec)) for rec in (off, mj)]
    top = max((e2e.value(e2e.cell(rec, c, x), metric) or 0) for rec in (off, mj) for c, x in cells)
    for ax, conc in zip(axes, concs):
        ctxs = sorted({x for c, x in cells if c == conc and (x or metric != "pp_tps")})  # no ctx: no prefill
        w = 0.8 / len(sides)
        for i, (rec, sid, lbl) in enumerate(sides):
            hollow = not theme.filled(sid)
            col = theme.color(mode, sid)
            labelled = ax is not axes[0]  # legend entry: the side's first drawn bar
            for j, ctx in enumerate(ctxs):
                x = j + (i - (len(sides) - 1) / 2) * w
                v = e2e.value(e2e.cell(rec, conc, ctx), metric)
                if v is None:
                    ax.text(
                        x,
                        top * 0.02,
                        "n/a",
                        rotation=90,
                        ha="center",
                        va="bottom",
                        fontsize=7.5,
                        color=mode.text_dim,
                        zorder=4,
                    )
                    continue
                ax.bar(
                    x,
                    v,
                    width=w * 0.92,
                    color=mode.panel if hollow else col,
                    edgecolor=col if hollow or not theme.hatch(sid) else mode.panel,
                    linewidth=1.6 if hollow else 0,
                    hatch=theme.hatch(sid),
                    zorder=3,
                    label=None if labelled else lbl,
                )
                labelled = True
        ax.set_xticks(range(len(ctxs)))
        ax.set_xticklabels([f"{L_label(c)}" if c else "0" for c in ctxs])
        ax.set_xlabel(f"context · {conc} concurrent request{'s' if conc > 1 else ''}")
    axes[0].set_ylabel(ylabel)
    axes[0].set_ylim(0, top * 1.08)
    # the legend sits in the title row, right-aligned: no panel is wide enough for it
    h, lbl = axes[0].get_legend_handles_labels()
    fig.legend(
        h,
        lbl,
        loc="lower right",
        bbox_to_anchor=(axes[-1].get_position().x1, axes[-1].get_position().y1 + 0.01),
        ncol=len(lbl),
        facecolor=mode.panel,
        edgecolor=mode.grid,
        labelcolor=mode.text,
        framealpha=1.0,
        fontsize=9,
    )
    theme.title(axes[0], mode, ttl, sub)
    theme.footer(
        fig,
        mode,
        f"mjolnir bench e2e · {mj.get('model', '')} · {e2e.spec_short(off, mj)} · "
        f"{off.get('image', '')} vs {mj.get('image', '')} · llama-benchy",
    )
    return fig


def chart_e2e_decode(layout: RepoLayout, mode: Mode):
    return _e2e_chart(
        layout,
        mode,
        "tg_tps",
        "decode tokens/s (higher is better)",
        "End-to-end decode throughput",
        "official image (native FlashInfer) vs the Mjolnir image · the README table's cells",
    )


def chart_e2e_prefill(layout: RepoLayout, mode: Mode):
    return _e2e_chart(
        layout,
        mode,
        "pp_tps",
        "prompt tokens/s (higher is better)",
        "End-to-end prompt processing",
        "official image (native FlashInfer) vs the Mjolnir image · the README table's cells",
    )


E2E_CHARTS: dict[str, Callable] = {"e2e-decode": chart_e2e_decode, "e2e-prefill": chart_e2e_prefill}
