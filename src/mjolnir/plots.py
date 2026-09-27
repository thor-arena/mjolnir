"""Mjolnir charts — the repo's face on GitHub.

Three figures, one design system (``mjolnir.theme``), all dark-dashboard
style so the numbers read like a product:

 * ``assets/benchmarks/kernel.png``    — kernel-level hero: the GEMV split-KV
   sweep in one clean window + FA4-vs-FlashInfer decode µs vs context. Fed by
   the committed raw JSONs in ``benchmarks/raw/`` (reproducible, greppable).
 * ``assets/benchmarks/kernel-length.png`` — decode kernel wall-µs vs context
   length across GEMV / FA4-1CTA / FlashInfer (best config each) + the
   KV-bandwidth roofline. Fed by the ``gemv-decode-bench-*.json`` raws.
 * ``assets/benchmarks/fa4-vs-fi.png`` — end-to-end throughput per backend
  (decode + prefill panels) from ``benchmarks/history.jsonl``.
* ``assets/benchmarks/history.png``   — progress over time: tg t/s per
  backend across the dates (the living chart; every nightly bump adds a point).

Run ``mjolnir plot`` after a bench (or any time) to (re)render them.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from mjolnir.config import RepoLayout
from mjolnir.history import load_records
from mjolnir import theme


def _matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "matplotlib is required for charts:  uv tool install --with plot "
            "mjolnir   (or: pip install matplotlib)") from e
    return plt


def _fig(figsize=(13.0, 5.6)):
    plt = _matplotlib()
    plt.rcParams.update({
        "figure.facecolor": theme.BG, "axes.facecolor": theme.PANEL,
        "text.color": theme.TEXT, "axes.edgecolor": theme.GRID,
        "font.size": 11, "axes.unicode_minus": False,
        "figure.dpi": 160, "savefig.dpi": 160,
    })
    return plt


def _footer(fig, text: str):
    fig.text(0.01, 0.008, text, ha="left", va="bottom",
             color=theme.TEXT_DIM, fontsize=8, alpha=0.85, family="monospace")


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=fig.get_facecolor(),
                bbox_inches="tight", pad_inches=0.18)
    print(f"[mjolnir] wrote {path}")
    return path


# ── kernel.png (raw JSON fed) ────────────────────────────────────────────────

def _load_json(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:  # noqa: BLE001
        return None


def render_kernel_charts(layout: RepoLayout) -> list[Path]:
    raw = layout.raw_dir
    ringfix = None
    for cand in sorted(raw.glob("gemv-ring-fix-bench*.json")):
        ringfix = _load_json(cand)
        if ringfix:
            break
    micro = None
    for cand in sorted(raw.glob("decode-microbench*.json")):
        micro = _load_json(cand)
        if micro:
            break
    if not ringfix and not micro:
        print(f"[mjolnir] no kernel raw JSON in {raw} — skipping kernel chart "
              f"(run: mjolnir bench kernel gemv-ringfix)")
        return []

    plt = _fig()
    if ringfix is not None and micro is not None:
        fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.6))
        left_ax, right_ax = axes[0], axes[1]
    elif ringfix is not None:
        fig, left_ax = plt.subplots(figsize=(13.0, 5.6))
        right_ax = None
    else:
        fig, right_ax = plt.subplots(figsize=(13.0, 5.6))
        left_ax = None

    if ringfix is not None and left_ax is not None:
        ax = left_ax
        res = ringfix.get("results", {})
        ns_pts, ns_us = [], []
        for tag, d in res.items():
            if tag.startswith("st16_ns") and d.get("median"):
                try:
                    ns = int(tag.split("_ns")[1])
                except (IndexError, ValueError):
                    continue
                ns_pts.append(ns)
                ns_us.append(d["median"])
        order = sorted(range(len(ns_pts)), key=lambda i: ns_pts[i])
        ns_pts = [ns_pts[i] for i in order]
        ns_us = [ns_us[i] for i in order]

        fi = res.get("flashinfer", {}).get("median")
        theme.apply_theme(fig, ax)
        ax.set_yscale("log")
        gemv_line = None
        if ns_pts:
            gemv_line = ax.plot(ns_pts, ns_us, "-o",
                                color=theme.SERIES["FA4-GEMV"],
                                lw=2.2, ms=6.5, zorder=5,
                                label="GEMV (st16 ring)")[0]
            best_i = min(range(len(ns_us)), key=lambda i: ns_us[i])
            ax.annotate("auto\nns=20", xy=(ns_pts[best_i], ns_us[best_i]),
                        xytext=(ns_pts[best_i] * 1.6, ns_us[best_i] * 1.15),
                        color=theme.TEXT_DIM, fontsize=9,
                        arrowprops=dict(arrowstyle="->", color=theme.TEXT_DIM,
                                        lw=0.8))
        if fi:
            ax.axhline(fi, color=theme.SERIES["FlashInfer"], ls="--", lw=1.4,
                       zorder=3, alpha=0.9)
            ax.text(0.85, fi * 1.06, f"FlashInfer  {fi:.0f} µs",
                    color=theme.SERIES["FlashInfer"], fontsize=9.5,
                    ha="left", va="bottom")
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 2, 4, 8, 16, 32, 64])
        ax.set_xticklabels(["1", "2", "4", "8", "16", "32", "64"])
        ax.set_xlim(0.8, 96)
        ax.set_xlabel("SplitKV plan — CTAs = 4 × ns (ns=1 → 4, ns=64 → 256)")
        ax.set_ylabel("wall µs (median of 300, CUDA events)")
        theme.style_title(ax, "GEMV decode — split-KV sweep, one clean window",
                          "L=8192 · M=1 · GQA 24/4 · bf16 Q + FP8 KV · "
                          "nvfp4 weights · sm_110a")
        if gemv_line is not None:
            theme.legend(ax, [gemv_line], ["FA4-native GEMV (st16)"], ncols=1)
        ax.set_xlim(left=0.8)

    if micro is not None and right_ax is not None:
        ax = right_ax
        res = micro.get("results", {})
        by_backend: dict[str, dict[int, float]] = {}
        for tag, d in res.items():
            if not d.get("median"):
                continue
            parts = tag.split("|")
            backend = parts[0].replace("B1_", "").replace("B2_", "FA4-") \
                .replace("B3_", "")
            if "fa4_fp8" in backend:
                backend = "FA4-1CTA (fp8)"
            elif "fa4_bf16" in backend:
                continue  # keep the panel focused: fp8 is the served path
            m = parts[1].lstrip("M")
            if m != "1":
                continue
            L = int(parts[2].lstrip("L"))
            by_backend.setdefault(backend, {})[L] = d["median"]

        labels = {"FA4-fp8": "FA4-hd256 (fp8 KV)",
                  "FA4-fa4_fp8": "FA4-hd256 (fp8 KV)",
                  "flashinfer": "FlashInfer (fa2-tc)"}
        theme.apply_theme(fig, ax)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        legend_items = []
        for backend, pts in by_backend.items():
            if not pts:
                continue
            xs = sorted(pts)
            ys = [pts[x] for x in xs]
            label = labels.get(backend, backend)
            color = theme.SERIES["FlashInfer"] if "flashinfer" in backend \
                else theme.SERIES["FA4-1CTA"]
            line = ax.plot(xs, ys, "-o", color=color, lw=2.2, ms=6, zorder=5,
                           label=label)[0]
            legend_items.append((line, label))
        ax.set_xticks([256, 1024, 4096, 16384])
        ax.set_xticklabels(["256", "1K", "4K", "16K"])
        ax.set_xlabel("context length (tokens)")
        ax.set_ylabel("wall µs (median of 100)")
        theme.style_title(ax, "FA4 vs FlashInfer — decode kernel",
                         "M=1 · GQA 24/4 · block-128 paged · one clean window")
        if legend_items:
            theme.legend(ax, [h for h, _ in legend_items],
                         [lb for _, lb in legend_items])

    _footer(fig, "mjolnir · clean-window gated (vLLM queue 0/0 × N samples) · "
                "wall-clock relative ratios; ncu achieved-BW for absolutes")
    theme.watermark(fig)
    return [_save(fig, layout.charts_dir / "kernel.png")]


# ── kernel-length.png (raw JSON fed) ─────────────────────────────────────────

# gemv-decode-bench --mode values → theme series key. GEMV (dense + paged)
# collapses to one series (best split plan / KV layout wins).
_MULTIL_BACKEND = {
    "gemv_dense": "FA4-GEMV",
    "gemv_paged": "FA4-GEMV",
    "fa4_1cta": "FA4-1CTA",
    "flashinfer": "FlashInfer",
}


def _multil_L(tag: str):
    """L<LEN> |… → the context length, or None."""
    head = tag.split("|", 1)[0]
    if head.startswith("L"):
        try:
            return int(head[1:])
        except ValueError:
            return None
    return None


def _load_decode_multil(layout: RepoLayout):
    """Per-backend, per-L best (min) median µs from the gemv-decode-bench raws.

    Returns ``({series_key: {L: best_median_us}}, roofline_dict_or_None)``.
    """
    per: dict[str, dict[int, float]] = {}
    roofline: dict | None = None
    for cand in sorted(layout.raw_dir.glob("gemv-decode-bench-*.json")):
        data = _load_json(cand)
        if not data:
            continue
        mode = data.get("env", {}).get("mode") or \
            cand.stem.replace("gemv-decode-bench-", "")
        slot = per.setdefault(_MULTIL_BACKEND.get(mode, mode), {})
        for tag, d in data.get("results", {}).items():
            if not d.get("median"):
                continue
            L = _multil_L(tag)
            if L is None:
                continue
            prev = slot.get(L)
            if prev is None or d["median"] < prev:
                slot[L] = d["median"]
        if roofline is None:
            roofline = data.get("roofline_us")
    return per, roofline


def render_kernel_length(layout: RepoLayout) -> list[Path]:
    per, roofline = _load_decode_multil(layout)
    if not per:
        print(f"[mjolnir] no gemv-decode-bench raws in {layout.raw_dir} — "
              f"skipping kernel-length chart "
              f"(run: mjolnir bench kernel gemv-bench --mode <m>)")
        return []

    plt = _fig()
    fig, ax = plt.subplots(figsize=(13.0, 5.6))
    theme.apply_theme(fig, ax)

    all_L = sorted({L for pts in per.values() for L in pts})
    if not all_L:
        return []

    order = [b for b in theme.BACKEND_ORDER if b in per] + \
        [b for b in per if b not in theme.BACKEND_ORDER]
    items: list[tuple] = []
    for b in order:
        pts = per[b]
        xs = sorted(pts)
        ys = [pts[x] for x in xs]
        h = ax.plot(xs, ys, "-o", color=theme.color_for(b), lw=2.2, ms=6,
                    zorder=5, label=b)[0]
        items.append((h, b))
    if roofline:
        rx = sorted(int(k[1:]) for k in roofline)
        ry = [roofline[f"L{r}"] for r in rx]
        h = ax.plot(rx, ry, ":", color=theme.TEXT_DIM, lw=1.4, zorder=2,
                    label="roofline (KV @ BW)")[0]
        items.append((h, "roofline (KV @ BW)"))

    ax.set_xscale("log", base=2)
    ax.set_xticks(all_L)
    ax.set_xticklabels([f"{L // 1024}K" for L in all_L])
    ax.set_xlim(min(all_L) * 0.9, max(all_L) * 1.12)
    ax.set_xlabel("context length (tokens)")
    ax.set_ylabel("wall µs (median of 300, CUDA events)")
    theme.style_title(ax, "Decode kernel vs context — GEMV / FA4-1CTA / FlashInfer",
                      "M=1 · GQA 24/4 · bf16 Q + FP8 KV · nvfp4 weights · sm_110a · "
                      "clean-window gated · GEMV = best split plan")
    if items:
        theme.legend(ax, [h for h, _ in items], [lb for _, lb in items], ncols=2)

    _footer(fig, "mjolnir · gemv-decode-bench (dense/paged/FA4-1CTA/FlashInfer) · "
                 "wall-clock, one clean window each")
    theme.watermark(fig)
    return [_save(fig, layout.charts_dir / "kernel-length.png")]


# ── history.jsonl fed charts ─────────────────────────────────────────────────

def _records_by_backend(records: list[dict]) -> dict[str, dict]:
    """Latest record per backend."""
    latest: dict[str, dict] = {}
    for r in records:
        b = r.get("backend")
        if b and r.get("cells"):
            latest[b] = r
    return latest


def _cell_series(record: dict, concurrency: int, metric: str
                 ) -> tuple[list[int], list[float], list[float | None]]:
    xs, ys, errs = [], [], []
    for c in record["cells"]:
        if c.get("concurrency") != concurrency:
            continue
        m = c.get(metric)
        if not m or m.get("mean") is None:
            continue
        xs.append(c.get("context") or 0)
        ys.append(m["mean"])
        errs.append(m.get("std"))
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    return ([xs[i] for i in order], [ys[i] for i in order],
            [errs[i] for i in order])


def render_fa4_vs_fi(layout: RepoLayout) -> list[Path]:
    records = load_records(layout.history_file)
    latest = _records_by_backend(records)
    if not latest:
        print(f"[mjolnir] no bench history yet — run: mjolnir bench perf")
        return []

    plt = _fig()
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.6))
    backends = [b for b in theme.BACKEND_ORDER if b in latest]
    for b in latest:
        if b not in backends:
            backends.append(b)

    used_colors: set[str] = set()
    handles, labels = [], []
    for ax, (metric, title, sub) in zip(
            axes,
            [("tg_tps", "Decode — generation throughput",
              "tokens/s · concurrency 1 · pp=2048 tg=128"),
             ("pp_tps", "Prefill — prompt throughput",
              "tokens/s · concurrency 1 · pp=2048 tg=128")]):
        theme.apply_theme(fig, ax)
        for b in backends:
            rec = latest[b]
            xs, ys, errs = _cell_series(rec, 1, metric)
            if not xs:
                continue
            color = theme.color_for(b, used_colors)
            used_colors.add(color)
            errs = [e for e in errs if e is not None]
            if errs and len(errs) == len(ys):
                ax.errorbar(xs, ys, yerr=errs, **theme.errorbar_kwargs(color),
                            marker="o", ms=6, lw=2.2, color=color, zorder=5,
                            label=b)
            else:
                ax.plot(xs, ys, "-o", color=color, lw=2.2, ms=6, zorder=5,
                        label=b)
        ax.set_xticks([0, 4096, 8192])
        ax.set_xticklabels(["0", "4K", "8K"])
        ax.set_xlabel("context length (tokens)")
        ax.set_ylabel("tokens/s")
        theme.style_title(ax, title, sub)
        if not ax.get_legend():
            ax.legend(facecolor=theme.PANEL, edgecolor=theme.GRID,
                      labelcolor=theme.TEXT, fontsize=10)

    img = next(iter(latest.values())).get("image", "")
    _footer(fig, f"mjolnir · {img} · {len(records)} bench sweeps on record · "
                f"llama-benchy, {latest[backends[0]].get('runs', '?')} runs/cell, "
                f"clean-window gated")
    theme.watermark(fig)
    return [_save(fig, layout.charts_dir / "fa4-vs-fi.png")]


def render_history(layout: RepoLayout, concurrency: int = 1,
                   context: int = 8192) -> list[Path]:
    records = load_records(layout.history_file)
    if not records:
        print(f"[mjolnir] no bench history yet — run: mjolnir bench perf")
        return []

    plt = _fig()
    fig, ax = plt.subplots(figsize=(13.0, 5.6))
    theme.apply_theme(fig, ax)

    used_colors: set[str] = set()
    for b in [x for x in theme.BACKEND_ORDER if
              any(r.get("backend") == x for r in records)] \
            + [r["backend"] for r in records
               if r.get("backend") not in theme.BACKEND_ORDER]:
        pts = []
        for r in records:
            if r.get("backend") != b:
                continue
            for c in r.get("cells", []):
                if c.get("concurrency") != concurrency or \
                        c.get("context") != context:
                    continue
                m = c.get("tg_tps")
                if not m or m.get("mean") is None:
                    continue
                pts.append((datetime.fromisoformat(
                    r["ts"].replace("Z", "+00:00")
                    ) if "T" in r.get("ts", "") else datetime.fromtimestamp(
                        r["epoch"]), m["mean"], r.get("config", "")))
        if not pts:
            continue
        pts.sort(key=lambda p: p[0])
        color = theme.color_for(b, used_colors)
        used_colors.add(color)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "-o", color=color,
                lw=2.0, ms=7, zorder=5, label=b)
        for ts, y, cfg in pts:
            short = cfg.replace("NVFP4_", "")
            ax.annotate(short, xy=(ts, y), xytext=(0, 10),
                        textcoords="offset points", ha="center",
                        color=theme.TEXT_DIM, fontsize=8.5)
    ax.set_xlabel("")
    ax.set_ylabel("tg t/s (decode)")
    theme.style_title(ax, "Progress over time — tg t/s",
                      f"concurrency {concurrency} · context {context} · "
                      "one point per bench sweep")
    ax.legend(facecolor=theme.PANEL, edgecolor=theme.GRID,
              labelcolor=theme.TEXT, fontsize=10, loc="upper left")
    _footer(fig, "mjolnir · history.jsonl — every `mjolnir bench perf` "
                "appends a row; this chart grows with the repo")
    theme.watermark(fig)
    return [_save(fig, layout.charts_dir / "history.png")]


def render_all(layout: RepoLayout, concurrency: int = 1,
               context: int = 8192) -> list[Path]:
    out: list[Path] = []
    out += render_kernel_charts(layout)
    out += render_kernel_length(layout)
    out += render_fa4_vs_fi(layout)
    out += render_history(layout, concurrency, context)
    return out
