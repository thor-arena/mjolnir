"""Mjolnir charts — the repo's face on GitHub.

Six figures, one design system (``mjolnir.theme``), all dark-dashboard
style so the numbers read like a product. Two data sources: the committed
raw JSONs (kernel microbenches) and ``benchmarks/history.jsonl`` (e2e).

* ``kernel-microbench.png``   — kernel vs kernel (microbench data only, never
  e2e): the GEMV SplitKV ns sweep in one clean window + FA4-vs-FlashInfer
  decode µs vs context.
* ``kernel-length.png``       — decode kernel wall-µs vs context length across
  GEMV / FA4-1CTA / FlashInfer (best config each) + the KV-bandwidth
  roofline, fed by the ``gemv-decode-bench-*.json`` raws.
* ``vllm-vs-mjolnir-image.png`` — e2e, image vs image: the baseline vLLM
  docker image vs the Mjolnir docker image (decode + prefill panels), per
  backend row of ``history.jsonl``.
* ``bench-compare.png``       — bar chart: every unique bench (latest record
  per backend+image+config), decode tg t/s at c=1 across contexts.
* ``tg-variability.png``      — per-run tg t/s sample spread per bench
  (boxplot, c=1) — the noise floor kernel changes have to beat.
* ``ttfr-by-context.png``     — time-to-first-token per bench × context
  (log) — prefill latency, what the decode kernel does *not* move.

Run ``mjolnir plot`` after a bench (or any time) to (re)render them.
"""
from __future__ import annotations

import json
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


def _load_json(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:  # noqa: BLE001
        return None


def _find_raw(layout: RepoLayout, pattern: str):
    """Raw kernel JSON: committed under ``benchmarks/raw/`` first, then the
    kernel package dir (where the GEMV benches are versioned with the code)."""
    dirs = [layout.raw_dir,
            layout.repo_root / "docker" / "vllm-thor" / "fa4-gemv-kernel"]
    for d in dirs:
        if not d.is_dir():
            continue
        for cand in sorted(d.glob(pattern)):
            data = _load_json(cand)
            if data:
                return data
    return None


def _ctx_label(ctx: int | None) -> str:
    if not ctx:
        return "0"
    return f"{ctx // 1024}K" if ctx % 1024 == 0 else str(ctx)


def _log_axis_ticks(lo: float, hi: float) -> tuple[list[float], list[str]]:
    """Value ticks for a log axis. A <1-decade span has exactly ONE decade
    tick (10^3), which leaves the axis unlabeled — so use every integer
    multiple (700, 800, …, 4k) while that stays ≤8 ticks, and plain decades
    once the span widens."""
    import math
    lo_e, hi_e = math.floor(math.log10(lo)), math.ceil(math.log10(hi))
    fine = [d * 10**e for e in range(lo_e, hi_e + 1) for d in range(1, 10)
            if lo <= d * 10**e <= hi]
    if len(fine) <= 8:
        ticks = fine
    else:
        ticks = [10**e for e in range(lo_e, hi_e + 1)
                 if lo <= 10**e <= hi]
    labels = [f"{t / 1000:g}k" if t >= 1000 else f"{t:g}" for t in ticks]
    return ticks, labels


def _samples_per_cell(record: dict) -> int:
    """Pooled measured runs per cell (the llama-benchy ``--runs`` × the
    independent gated sweeps)."""
    return int(record.get("runs", 0)) * int(record.get("repeat", 1) or 1)


# ── Jetson Thor DRAM roofline (the kernel charts' red line) ──────────────────
# LPDDR5X roofline 273 GB/s (docs/fa4-hd256-fp8/gemv-ring-fix-bench.md).
# Only the per-kernel floor is drawn: it's a pure bytes÷BW computation from
# the kernel shape (no model-weight or MTP-acceptance assumptions, which are
# workload-dependent and can't be reliably proven as a limit).
THOR_DRAM_GBS = 273.0


def roofline_us(L: int) -> float:
    """Single hd256-layer decode floor (µs) at length L: that layer's fp8 KV
    (GQA 24/4 → 4 kv heads × 256 dim × K+V) streamed at the DRAM roofline."""
    return L * 4 * 256 * 2 / (THOR_DRAM_GBS * 1e9) * 1e6


def _unique_benches(records: list[dict]) -> dict[str, dict]:
    """Latest record per unique bench (backend + image + config)."""
    latest: dict[tuple, dict] = {}
    for r in records:
        if r.get("backend") and r.get("cells"):
            key = (r.get("backend"), r.get("image", ""), r.get("config", ""))
            if key not in latest or r.get("epoch", 0) > latest[key].get("epoch", 0):
                latest[key] = r
    return latest


def _cell_metric(record: dict, concurrency: int, context: int, metric: str
                 ) -> tuple[float | None, float | None]:
    for c in record.get("cells", []):
        if c.get("concurrency") != concurrency or c.get("context") != context:
            continue
        m = c.get(metric)
        if m and m.get("mean") is not None:
            return m["mean"], m.get("std")
    return None, None


# ── kernel-microbench.png (raw kernel JSON fed — kernel vs kernel) ──────────

def render_kernel_charts(layout: RepoLayout) -> list[Path]:
    ringfix = _find_raw(layout, "gemv-ring-fix-bench*.json")
    micro = _find_raw(layout, "decode-microbench*.json")
    if not ringfix and not micro:
        print(f"[mjolnir] no kernel raw JSON in {layout.raw_dir} — skipping "
              f"kernel chart (run: mjolnir bench kernel gemv-ringfix)")
        return []

    plt = _fig()
    if ringfix and micro:
        fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(13.0, 5.6))
    elif ringfix:
        fig, ax_left = plt.subplots(1, 1, figsize=(13.0, 5.6))
        ax_right = None
    else:
        fig, ax_right = plt.subplots(1, 1, figsize=(13.0, 5.6))
        ax_left = None

    if ringfix is not None and ax_left is not None:
        ax = ax_left
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
            color = theme.color_for("FA4-GEMV")
            gemv_line = ax.plot(ns_pts, ns_us, "-o", color=color,
                                lw=2.2, ms=6.5, zorder=5,
                                label="FA4-GEMV (st16 ring)")[0]
            best_i = min(range(len(ns_us)), key=lambda i: ns_us[i])
            ax.annotate("auto\nns=20", xy=(ns_pts[best_i], ns_us[best_i]),
                        xytext=(ns_pts[best_i] * 1.6, ns_us[best_i] * 1.15),
                        color=theme.TEXT_DIM, fontsize=9,
                        arrowprops=dict(arrowstyle="->", color=theme.TEXT_DIM,
                                        lw=0.8))
        if fi:
            fi_color = theme.color_for("FlashInfer")
            ax.axhline(fi, color=fi_color, ls="--", lw=1.4,
                       zorder=3, alpha=0.9)
            ax.text(0.85, fi * 1.06, f"FlashInfer  {fi:.0f} µs",
                    color=fi_color, fontsize=9.5,
                    ha="left", va="bottom")
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 2, 4, 8, 16, 32, 64])
        ax.set_xticklabels(["1", "2", "4", "8", "16", "32", "64"])
        ax.set_xlim(0.8, 96)
        ax.set_xlabel("SplitKV plan — CTAs = 4 × ns (ns=1 → 4, ns=64 → 256)")
        ax.set_ylabel("wall µs (median of 300, CUDA events)")
        theme.style_title(
            ax, "GEMV decode — split-KV sweep\none clean window",
            "L=8192 · M=1 · GQA 24/4 · FP8 KV · nvfp4 weights · sm_110a")
        if gemv_line is not None:
            theme.legend(ax, [gemv_line], ["FA4-GEMV (st16)"], ncols=1)
        # Red HW limit: one hd256 layer's KV (16 MiB at L=8192) at DRAM BW.
        rl = roofline_us(8192)
        ax.axhline(rl, color=theme.ROOFLINE, ls=":", lw=1.6, zorder=4)
        ax.text(96, rl * 1.12, f"DRAM roofline {rl:.0f} µs "
                "(16 MiB KV @ 273 GB/s)",
                color=theme.ROOFLINE, fontsize=9, ha="right", va="bottom")
        ax.set_xlim(left=0.8)
        ax.set_ylim(bottom=rl * 0.82)

    if micro is not None and ax_right is not None:
        ax = ax_right
        res = micro.get("results", {})
        by_backend: dict[str, dict[int, float]] = {}
        for tag, d in res.items():
            if not d.get("median"):
                continue
            parts = tag.split("|")
            backend = parts[0].replace("B1_", "").replace("B2_", "FA4-") \
                .replace("B3_", "")
            if "fa4_fp8" in backend:
                backend = "FA4-1CTA (fp8 KV)"
            elif "fa4_bf16" in backend:
                continue  # keep the panel focused: fp8 is the served path
            m = parts[1].lstrip("M")
            if m != "1":
                continue
            L = int(parts[2].lstrip("L"))
            by_backend.setdefault(backend, {})[L] = d["median"]

        theme.apply_theme(fig, ax)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        legend_items = []
        for backend in theme.order_labels(list(by_backend)):
            pts = by_backend[backend]
            if not pts:
                continue
            xs = sorted(pts)
            ys = [pts[x] for x in xs]
            color = theme.color_for(backend)
            line = ax.plot(xs, ys, "-o", color=color, lw=2.2, ms=6, zorder=5,
                           label=backend)[0]
            legend_items.append((line, backend))
        ax.set_xticks([256, 1024, 4096, 16384])
        ax.set_xticklabels(["256", "1K", "4K", "16K"])
        ax.set_xlabel("context length (tokens)")
        ax.set_ylabel("wall µs (median of 100)")
        theme.style_title(
            ax, "FA4 vs FlashInfer —\ndecode kernel",
            "M=1 · GQA 24/4 · block-128 paged · one clean window")
        # Red HW limit: the per-layer KV floor at each L (DRAM BW).
        Ls = sorted({L for v in by_backend.values() for L in v})
        if Ls:
            import numpy as np
            xs = np.logspace(np.log10(Ls[0]), np.log10(Ls[-1]), 40)
            rl_line = ax.plot(xs, [roofline_us(int(l)) for l in xs],
                              color=theme.ROOFLINE, ls=":", lw=1.6, zorder=4,
                              label="DRAM roofline (273 GB/s)")[0]
            legend_items.append((rl_line, "DRAM roofline (273 GB/s)"))
        if legend_items:
            theme.legend(ax, [h for h, _ in legend_items],
                         [lb for _, lb in legend_items])

    _footer(fig, "mjolnir · kernel vs kernel (ns sweep + decode micro-bench)")
    theme.watermark(fig)
    plt.close(fig)
    return [_save(fig, layout.charts_dir / "kernel-microbench.png")]


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


# ── e2e + compare charts (history.jsonl fed) ─────────────────────────────────

def render_e2e_images(layout: RepoLayout) -> list[Path]:
    """Baseline vLLM image vs Mjolnir image — per-backend e2e throughput
    (image-level: the image carries the kernel stack; one line per panel
    per backend row in history.jsonl)."""
    records = load_records(layout.history_file)
    latest = {r["backend"]: r for r in _unique_benches(records).values()}
    if not latest:
        print(f"[mjolnir] no bench history yet — run: mjolnir bench perf")
        return []

    plt = _fig()
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.6))
    backends = theme.order_labels(list(latest))

    for ax, (metric, title, sub) in zip(
            axes,
            [("tg_tps", "Decode — generation throughput",
              "tokens/s · concurrency 1 · pp=2048 tg=128"),
             ("pp_tps", "Prefill — prompt throughput",
              "tokens/s · concurrency 1 · pp=2048 tg=128")]):
        theme.apply_theme(fig, ax)
        for b in backends:
            rec = latest[b]
            pts = []
            for c in rec["cells"]:
                if c.get("concurrency") != 1:
                    continue
                m = c.get(metric)
                if not m or m.get("mean") is None:
                    continue
                pts.append((c.get("context") or 0, m["mean"], m.get("std")))
            if not pts:
                continue
            pts.sort(key=lambda p: p[0])
            color = theme.color_for(b)
            errs = [e for e in (p[2] for p in pts) if e is not None]
            if errs and len(errs) == len(pts):
                ax.errorbar([p[0] for p in pts], [p[1] for p in pts],
                            yerr=errs, **theme.errorbar_kwargs(color),
                            marker="o", ms=6, lw=2.2, color=color, zorder=5,
                            label=b)
            else:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], "-o",
                        color=color, lw=2.2, ms=6, zorder=5, label=b)
        ax.set_xticks([0, 4096, 8192])
        ax.set_xticklabels(["0", "4K", "8K"])
        ax.set_xlabel("context length (tokens)")
        ax.set_ylabel("tokens/s")
        theme.style_title(ax, title, sub)
        if not ax.get_legend():
            ax.legend(facecolor=theme.PANEL, edgecolor=theme.GRID,
                      labelcolor=theme.TEXT, fontsize=10)

    rec0 = latest[backends[0]]
    _footer(fig, f"mjolnir image VS vllm image · {rec0.get('model', '')} · "
                 f"llama-benchy {_samples_per_cell(rec0)} runs/cell")
    theme.watermark(fig)
    plt.close(fig)
    return [_save(fig, layout.charts_dir / "vllm-vs-mjolnir-image.png")]


def render_bench_compare(layout: RepoLayout) -> list[Path]:
    """Bar chart over ALL unique benches (latest record per
    backend+image+config): decode tg t/s at c=1, one bar per context."""
    records = load_records(layout.history_file)
    benches = {r["backend"]: r for r in _unique_benches(records).values()}
    if not benches:
        print(f"[mjolnir] no bench history yet — run: mjolnir bench perf")
        return []

    plt = _fig()
    fig, ax = plt.subplots(figsize=(15.0, 5.6))
    theme.apply_theme(fig, ax)
    # Extra bottom margin: room for the per-bar context sub-labels AND the
    # per-group bench name (the legend), both above the fig footer.
    fig.subplots_adjust(bottom=0.16)

    # Contexts present in the data (sorted), encoded by a fixed ramp.
    contexts: list[int] = sorted({
        c.get("context") or 0
        for r in benches.values()
        for c in r.get("cells", [])
        if c.get("concurrency") == 1 and (c.get("tg_tps") or {}).get("mean")
    })
    if not contexts:
        print(f"[mjolnir] no c=1 tg cells in history — skipping compare chart")
        return []

    bench_names = theme.image_first_order(list(benches))
    n_ctx = len(contexts)
    BAR_PITCH = 1.5        # x units between bars inside a group (wide enough
                          # for the two-line labels — see figsize math above)
    GROUP_GAP = 4       # extra x units between groups (wide separation)
    GROUP_PITCH = (n_ctx - 1) * BAR_PITCH + GROUP_GAP
    max_mean = 0.0
    for i, name in enumerate(bench_names):
        rec = benches[name]
        shades = theme.context_shades(theme.color_for(name), n_ctx)
        color = theme.color_for(name)
        for j, ctx in enumerate(contexts):
            mean, _std = _cell_metric(rec, 1, ctx, "tg_tps")
            if mean is None:
                continue
            x = i * GROUP_PITCH + j * BAR_PITCH
            ax.bar(x, mean, width=1.1, color=shades[j], alpha=0.95,
                   edgecolor=theme.BG, linewidth=0.8, zorder=3)
            ax.text(x, mean * 1.03, f"{mean:.1f}", ha="center",
                    va="bottom", color=theme.TEXT_DIM, fontsize=9)
            max_mean = max(max_mean, mean)
            # Per-bar context sub-label (dim, small) — separate from the
            # group's bench name below it.
            ax.text(x, -0.03, f"{_ctx_label(ctx)} ctx",
                    transform=ax.get_xaxis_transform(), ha="center",
                    va="top", fontsize=9, color=theme.TEXT_DIM)
        # The bench name once per group, centered under its bars, in the
        # group's color (doubles as the legend).
        ax.text(i * GROUP_PITCH + (n_ctx - 1) * BAR_PITCH / 2, -0.1, name,
                transform=ax.get_xaxis_transform(), ha="center",
                va="top", fontsize=10.5, fontweight="bold", color=color)
    ax.set_xticks([])
    ax.set_xlim(-1.6, (len(bench_names) - 1) * GROUP_PITCH
                + (n_ctx - 1) * BAR_PITCH + 1.6)
    # Baseline: the native image at 0 ctx — the floor every group is read
    # against. Dashed grey, behind the bars (zorder 2 < bars' 3).
    base_name = next((n for n in bench_names
                      if "native" in n.lower()), None)
    base_ctx = 0 if 0 in contexts else contexts[0]
    if base_name:
        base_val, _ = _cell_metric(benches[base_name], 1, base_ctx,
                                   "tg_tps")
        if base_val:
            x0, x1 = ax.get_xlim()
            ax.hlines(base_val, x0, x1, color=theme.TEXT_DIM,
                      linestyle="--", linewidth=1.2, zorder=2)
    ax.set_ylabel("tg t/s (decode)")
    ax.set_ylim(0, max_mean * 1.16)
    theme.style_title(ax, "All unique benches — decode tg t/s",
                      "concurrency 1 · latest record per bench "
                      "(backend + image + config) · bar = mean · "
                      "shades darken with context")
    _footer(fig, "mjolnir · history.jsonl — every `mjolnir bench perf` "
                  "appends a row; a new bench adds a bar group")
    theme.watermark(fig)
    plt.close(fig)
    return [_save(fig, layout.charts_dir / "bench-compare.png")]


def render_tg_variability(layout: RepoLayout) -> list[Path]:
    """Per-run tg t/s sample spread (c=1), one box per bench × context —
    the noise floor a kernel change has to beat to be real."""
    records = load_records(layout.history_file)
    if not records:
        print(f"[mjolnir] no bench history yet — run: mjolnir bench perf")
        return []
    # Pool every record per bench (full sampling history, not just latest).
    by_bench: dict[str, dict[int, list[float]]] = {}
    for r in records:
        b = r.get("backend")
        if not b:
            continue
        for c in r.get("cells", []):
            if c.get("concurrency") != 1:
                continue
            m = c.get("tg_tps")
            if not m or not m.get("values"):
                continue
            by_bench.setdefault(b, {}).setdefault(c.get("context") or 0,
                                                 []).extend(m["values"])
    if not by_bench:
        print("[mjolnir] no c=1 tg sample values in history — skipping "
              "variability chart")
        return []

    plt = _fig()
    fig, ax = plt.subplots(figsize=(13.0, 5.6))
    theme.apply_theme(fig, ax)

    bench_names = theme.image_first_order(list(by_bench))
    contexts = sorted({ctx for v in by_bench.values() for ctx in v})
    n_ctx = len(contexts)
    handles, labels = [], []
    for i, name in enumerate(bench_names):
        shades = theme.context_shades(theme.color_for(name), n_ctx)
        handles.append(plt.Rectangle((0, 0), 1, 1,
                     color=theme.color_for(name)))
        labels.append(name)
        for j, ctx in enumerate(contexts):
            vals = by_bench[name].get(ctx, [])
            if not vals:
                continue
            x = i + (j - (n_ctx - 1) / 2) * (0.8 / n_ctx)
            ax.boxplot([vals], positions=[x], widths=0.8 / n_ctx * 0.9,
                       patch_artist=True, showfliers=False, zorder=3,
                       medianprops=dict(color=theme.TEXT, lw=1.6),
                       whiskerprops=dict(color=theme.TEXT_DIM, lw=1.0),
                       capprops=dict(color=theme.TEXT_DIM, lw=1.0),
                       boxprops=dict(facecolor=shades[j], alpha=0.85,
                                     edgecolor=theme.TEXT_DIM, lw=0.8))
    _rot = 0 if len(bench_names) <= 4 else 20
    ax.set_xticks(range(len(bench_names)))
    ax.set_xticklabels(bench_names, rotation=_rot,
                       ha="right" if _rot else "center")
    ax.set_ylabel("tg t/s (decode, per-run sample)")
    ax.set_ylim(bottom=0)
    theme.legend(ax, handles, labels, ncols=len(bench_names),
                 loc="lower center")
    theme.style_title(ax, "tg t/s per-run spread — c=1",
                      "box = IQR, line = median · every pooled per-run "
                      "sample in history, per bench · shades darken "
                      "with context")
    _footer(fig, "mjolnir · history.jsonl per-run values — a kernel change "
                 "must move the median by more than this box to be real")
    theme.watermark(fig)
    plt.close(fig)
    return [_save(fig, layout.charts_dir / "tg-variability.png")]


def render_ttfr(layout: RepoLayout) -> list[Path]:
    """Time-to-first-token per bench × context (log) — the prefill-latency
    story: the decode kernel does not move TTFR, the image (GDN prefill,
    patched stacks) does."""
    records = load_records(layout.history_file)
    latest = {r["backend"]: r for r in _unique_benches(records).values()}
    if not latest:
        print(f"[mjolnir] no bench history yet — run: mjolnir bench perf")
        return []

    plt = _fig()
    fig, ax = plt.subplots(figsize=(13.0, 5.6))
    theme.apply_theme(fig, ax)

    contexts: list[int] = []
    for r in latest.values():
        for c in r.get("cells", []):
            if c.get("concurrency") == 1 and \
                    (c.get("ttfr_ms") or {}).get("mean"):
                ctx = c.get("context") or 0
                if ctx not in contexts:
                    contexts.append(ctx)
    contexts.sort()
    if not contexts:
        print("[mjolnir] no ttfr cells in history — skipping ttfr chart")
        return []

    for b in theme.order_labels(list(latest)):
        rec = latest[b]
        pts = []
        for c in rec.get("cells", []):
            if c.get("concurrency") != 1:
                continue
            m = c.get("ttfr_ms")
            if not m or m.get("mean") is None:
                continue
            pts.append((c.get("context") or 0, m["mean"], m.get("std")))
        if not pts:
            continue
        pts.sort(key=lambda p: p[0])
        color = theme.color_for(b)
        errs = [e for e in (p[2] for p in pts) if e is not None]
        if errs and len(errs) == len(pts):
            ax.errorbar([p[0] for p in pts], [p[1] for p in pts], yerr=errs,
                        **theme.errorbar_kwargs(color), marker="o", ms=6,
                        lw=2.2, color=color, zorder=5, label=b)
        else:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "-o",
                    color=color, lw=2.2, ms=6, zorder=5, label=b)
    ax.set_yscale("log")
    import matplotlib.ticker as mticker
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    # Real value ticks (a <1-decade log span would otherwise show only 10^3).
    lo, hi = ax.get_ylim()
    ticks, tick_labels = _log_axis_ticks(lo, hi)
    ax.set_yticks(ticks)
    ax.set_yticklabels(tick_labels)
    ax.set_xticks(contexts)
    ax.set_xticklabels([_ctx_label(c) for c in contexts])
    ax.set_xlabel("context length (tokens)")
    ax.set_ylabel("TTFR (ms, log)")
    theme.style_title(ax, "Time-to-first-token by context",
                      "concurrency 1 · prefill latency — the decode kernel "
                      "does not move it; the image does")
    if not ax.get_legend():
        ax.legend(facecolor=theme.PANEL, edgecolor=theme.GRID,
                  labelcolor=theme.TEXT, fontsize=10)
    rec0 = latest[theme.order_labels(list(latest))[0]]
    _footer(fig, f"mjolnir image VS vllm image · {rec0.get('model', '')} · "
                 f"llama-benchy {_samples_per_cell(rec0)} runs/cell")
    theme.watermark(fig)
    plt.close(fig)
    return [_save(fig, layout.charts_dir / "ttfr-by-context.png")]


def render_all(layout: RepoLayout) -> list[Path]:
    out: list[Path] = []
    out += render_kernel_charts(layout)
    out += render_kernel_length(layout)
    out += render_e2e_images(layout)
    out += render_bench_compare(layout)
    out += render_tg_variability(layout)
    out += render_ttfr(layout)
    return out
