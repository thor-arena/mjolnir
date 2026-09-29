"""README generation — README.md rendered from README.md.j2, fully
raw-data-derived.

``mjolnir plot --readme`` (and the auto-render after every gated bench)
re-renders the repo README from the Jinja2 template at the repo root
(``README.md.j2``): every table and every number is computed from the
committed raw artifacts (``benchmarks/`` + the vLLM base tag in
``docker/vllm-thor/Dockerfile``); the prose is the template's own text.
Re-rendering is idempotent — the generated README only changes when the
raw data does.

Context contract: the template references exactly one top-level variable,
``ctx`` — a ``ReadmeContext`` instance. Every data value is a field there
(grep the template for ``ctx.`` to audit them all; a misspelled ``ctx.x``
fails loud at render time via StrictUndefined; a new data value is a new
field on ``ReadmeContext``):

* badges/labels — ``vllm_version``, ``model``, ``kv_bits``
* pre-computed markdown tables (with the repo's color math, see
  ``_delta``) — ``table_intro``, ``table_gemv_sweep``, ``table_multil``,
  ``table_e2e_tg``, ``table_e2e_pp`` (``None`` when the source data is
  missing; the template guards each block)
* sparse deep-dive values (pre-formatted strings) — ``hero_L``,
  ``kv_mib``, ``roofline_gbs``, ``ns1_bw``/``ns1_us``, ``ns20_bw``/
  ``ns20_us``, ``ns_speedup``, ``ns64_bw``/``ns64_us``, ``auto_us``,
  ``st2_us``/``st2_bw``, ``st16_us``/``st16_bw``, ``gemv_us``, ``fi_us``,
  ``gemv_vs_fi``, ``cta1_us``, ``cta1_L``
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import jinja2

from mjolnir.config import BASELINE_QUANT, DEFAULT_MODEL, DEFAULT_QUANT, RepoLayout
from mjolnir.history import load_records
from mjolnir import theme

# The repo's delta colors (GitHub markdown supports these): green for a win
# over the baseline, red for a regression, grey for a flat 0%.
_GREEN = "#16a34a"
_RED = "red"
_GREY = "grey"
_MINUS = "\u2212"  # the README's typeset minus (U+2212), not an ASCII hyphen

# Config → the short parenthetical used in the e2e table row labels.
CONFIG_DESCRIPTIONS = {
    BASELINE_QUANT: "FlashInfer",
    DEFAULT_QUANT: "FA4 + GEMV",
}

# Which concurrencies the e2e tables show (c1 = single user, c4 = saturation).
E2E_CONCURRENCIES = (1, 4)

# Deltas below this (in %) are gray, not red/green: the pooled per-run means
# carry ~2% SE (n=15, per-run std 2-3%), so a smaller gap is within the noise
# floor a kernel change has to beat (see the tg-variability chart).
E2E_GREY_BAND = 3.0


# ── formatting ──────────────────────────────────────────────────────────────


def _thousands(v: float, dec: int) -> str:
    """``1304.912`` → ``'1 304.91'`` (space group separator, the README's
    typography)."""
    return f"{v:,.{dec}f}".replace(",", " ")


def _delta(value: float, base: float, dec: int, higher_is_better: bool, grey_band: float = 0.0) -> str:
    """A colored LaTeX delta against ``base``, e.g.
    ``$\\color{#16a34a}{\\text{+4.1\\%}}$`` (green) / red / grey (0% — or a
    non-zero delta inside ``grey_band``, i.e. within the measurement noise
    floor)."""
    pct = (value - base) / base * 100.0 if base else 0.0
    mag = f"{abs(pct):.{dec}f}"
    if float(mag) == 0.0:
        color, text = _GREY, "0\\\\%"
    else:
        sign = "+" if pct > 0 else _MINUS
        text = f"{sign}{mag}\\\\%"
        if 0 < abs(pct) < grey_band:
            color = _GREY
        else:
            good = pct > 0 if higher_is_better else pct < 0
            color = _GREEN if good else _RED
    return f"$\\color{{{color}}}{{\\text{{{text}}}}}$"


def _ctx_label(ctx: int) -> str:
    if not ctx:
        return "no ctx"
    return f"{ctx // 1024}K ctx" if ctx % 1024 == 0 else f"{ctx} ctx"


def _load_json(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:  # noqa: BLE001
        return None


# ── raw data sources ────────────────────────────────────────────────────────


def _find_ringfix(layout: RepoLayout) -> dict | None:
    """The GEMV SplitKV sweep raw (one clean window, L=8192 M=1)."""
    for cand in sorted(layout.raw_dir.glob("gemv-ring-fix-bench*.json")):
        data = _load_json(cand)
        if data and data.get("results"):
            return data
    return None


def _vllm_base_version(layout: RepoLayout) -> str:
    """The vLLM version the image is built on: the Dockerfile's BASE_IMAGE
    tag, else the stock baseline image in the bench history."""
    dockerfile = layout.repo_root / "docker" / "vllm-thor" / "Dockerfile"
    if dockerfile.exists():
        m = re.search(r"vllm/vllm-openai:v(\d+\.\d+\.\d+)", dockerfile.read_text())
        if m:
            return m.group(1)
    for r in load_records(layout.history_file):
        img = r.get("image", "")
        if img.startswith("vllm/"):
            m = re.search(r"v(\d+\.\d+\.\d+)", img.rsplit(":", 1)[-1])
            if m:
                return m.group(1)
    return ""


def _is_stock_image(image: str) -> bool:
    """The upstream vLLM image (the baseline leg of the A/B)."""
    return image.startswith("vllm/")


def _vllm_version_of_image(image: str) -> str:
    m = re.search(r"v(\d+\.\d+\.\d+)", image.rsplit(":", 1)[-1])
    return m.group(1) if m else ""


_MULTIL_LABELS = {
    "flashinfer": "FlashInfer",
    "fa4_1cta": "FA4 hd256 1-CTA (carve-out)",
    "gemv_paged": "FA4 GEMV paged (auto)",
    "gemv_dense": "FA4 GEMV dense (auto)",
}
_MULTIL_ORDER = ["flashinfer", "fa4_1cta", "gemv_paged", "gemv_dense"]


def _load_multil(layout: RepoLayout) -> dict[str, dict[int, float]]:
    """Per kernel, per-L decode µs (median) from the gemv-decode-bench raws:
    the auto-plan row when present (the served dispatch), else the best
    (min) split plan."""
    per: dict[str, dict[int, float]] = {}
    for cand in sorted(layout.raw_dir.glob("gemv-decode-bench-*.json")):
        data = _load_json(cand)
        if not data:
            continue
        mode = data.get("env", {}).get("mode") or cand.stem.replace("gemv-decode-bench-", "")
        if mode not in _MULTIL_LABELS:
            continue
        slot: dict[int, float] = {}
        by_L: dict[int, list[tuple[str, float]]] = {}
        for tag, d in data.get("results", {}).items():
            if not d.get("median"):
                continue
            head = tag.split("|", 1)[0]
            if not head.startswith("L"):
                continue
            try:
                L = int(head[1:])
            except ValueError:
                continue
            by_L.setdefault(L, []).append((tag, d["median"]))
        for L, rows in by_L.items():
            autos = [m for t, m in rows if t.endswith("(auto)")]
            slot[L] = min(autos) if autos else min(m for _, m in rows)
        if slot:
            per[mode] = slot
    return per


def _e2e_latest(records: list[dict]) -> list[dict]:
    """Latest record per (image, config) — one e2e table row each. Stock
    baseline first, then the Mjolnir legs in hero order (GEMV, FlashInfer)."""
    latest: dict[tuple, dict] = {}
    for r in records:
        if not r.get("cells"):
            continue
        key = (r.get("image", ""), r.get("config", ""))
        if key not in latest or r.get("epoch", 0) > latest[key].get("epoch", 0):
            latest[key] = r
    stock = [r for r in latest.values() if _is_stock_image(r.get("image", ""))]
    mj = [r for r in latest.values() if not _is_stock_image(r.get("image", ""))]
    mj.sort(key=lambda r: theme.image_first_order([r.get("backend", "")])[0])
    return stock + mj


def _cell_mean(rec: dict, conc: int, ctx: int, metric: str) -> float | None:
    for c in rec.get("cells", []):
        if c.get("concurrency") == conc and c.get("context") == ctx:
            m = c.get(metric)
            if m and m.get("mean") is not None:
                return m["mean"]
    return None


# ── table builders (pre-computed markdown, color math included) ────────────


def _e2e_row_label(rec: dict, label_style: str = "full") -> str:
    cfg = rec.get("config", "")
    if _is_stock_image(rec.get("image", "")):
        if label_style == "short":
            return "stock vLLM"
        ver = _vllm_version_of_image(rec.get("image", ""))
        return f"stock vLLM {ver}, {cfg}" if ver else f"stock vLLM, {cfg}"
    desc = CONFIG_DESCRIPTIONS.get(cfg)
    if label_style == "short":
        return f"mjolnir ({desc})" if desc else f"mjolnir ({cfg})"
    return f"mjolnir, {cfg} ({desc})" if desc else f"mjolnir, {cfg}"


def _e2e_table(
    rows: list[dict],
    metric: str,
    header_first: str,
    value_fmt: str,
    delta_dec: int,
    higher_is_better: bool,
    label_style: str = "full",
) -> str | None:
    """One e2e table (tg or pp): rows = the A/B legs, columns = context ×
    the selected concurrencies (c1, c4), in that order."""
    concs = [c for c in E2E_CONCURRENCIES if any(_cell_mean(r, c, 0, metric) is not None for r in rows)]
    ctxs: list[int] = []
    for r in rows:
        for c in r.get("cells", []):
            cc = c.get("concurrency")
            ctx = c.get("context") or 0
            if cc in concs and (c.get(metric) or {}).get("mean") is not None and ctx not in ctxs:
                ctxs.append(ctx)
    if not concs or not ctxs:
        return None
    ctxs.sort()

    base = next((r for r in rows if _is_stock_image(r.get("image", ""))), rows[0])
    lines = [
        f"| {header_first} | " + " | ".join(f"c{c} / {_ctx_label(ctx)}" for ctx in ctxs for c in concs) + " |",
        "|" + "---|" * (1 + len(concs) * len(ctxs)),
    ]
    for r in rows:
        label = _e2e_row_label(r, label_style)
        cells = []
        for ctx in ctxs:
            for c in concs:
                v = _cell_mean(r, c, ctx, metric)
                if v is None:
                    cells.append("—")
                    continue
                text = value_fmt.format(v)
                if r is not base:
                    bv = _cell_mean(base, c, ctx, metric)
                    if bv:
                        text += " " + _delta(v, bv, delta_dec, higher_is_better, grey_band=E2E_GREY_BAND)
                cells.append(text)
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _intro_table(rows: list[dict], rf: dict | None, multi: dict[str, dict[int, float]], kv_bits: str) -> str | None:
    """The headline 3-leg table (stock FlashInfer / Mjolnir FlashInfer /
    Mjolnir GEMV): e2e tg + pp at the hero context, the L=hero decode
    kernel, the KV cache width."""

    def leg(config: str, stock: bool) -> dict | None:
        cands = [r for r in rows if r.get("config") == config and _is_stock_image(r.get("image", "")) == stock]
        if not cands:
            return None
        return max(cands, key=lambda r: r.get("epoch", 0))

    stock = leg(BASELINE_QUANT, True)
    mj_fi = leg(BASELINE_QUANT, False)
    mj_g = leg(DEFAULT_QUANT, False)
    if not (stock and mj_fi and mj_g):
        print("[mjolnir] intro table needs all three A/B legs (stock + both Mjolnir configs) — skipping")
        return None
    if not rf:
        print("[mjolnir] intro table needs the ring-fix kernel raw — skipping")
        return None

    ctxs = sorted(
        {
            c.get("context") or 0
            for r in (stock, mj_fi, mj_g)
            for c in r.get("cells", [])
            if _cell_mean(r, 1, c.get("context") or 0, "tg_tps") is not None
        }
    )
    hero = max(ctxs)
    fi_med = rf["results"].get("flashinfer", {}).get("median")
    gemv_med = (
        rf["results"].get("st16_ns20", {}).get("median")
        or rf["results"].get("st16_auto", {}).get("median")
        or min((d["median"] for k, d in rf["results"].items() if k.startswith("st16_")), default=None)
    )
    cta1 = (multi.get("fa4_1cta") or {}).get(hero)
    if None in (fi_med, gemv_med, cta1):
        print("[mjolnir] intro table needs the flashinfer + GEMV ns=20 kernel rows and the FA4 1-CTA row — skipping")
        return None

    def tsv(r, m):
        v = _cell_mean(r, 1, hero, m)
        return v

    tg_s, tg_fi, tg_g = tsv(stock, "tg_tps"), tsv(mj_fi, "tg_tps"), tsv(mj_g, "tg_tps")
    pp_s, pp_fi, pp_g = tsv(stock, "pp_tps"), tsv(mj_fi, "pp_tps"), tsv(mj_g, "pp_tps")
    if None in (tg_s, tg_fi, tg_g, pp_s, pp_fi, pp_g):
        print(f"[mjolnir] intro table: no c=1 ctx={hero} tg/pp cells on all three legs — skipping")
        return None

    lines = [
        "| | stock vLLM FlashInfer | \u26a1 Mjolnir FlashInfer | \u26a1 Mjolnir FlashAttention 4 GEMV |",
        "|---|---|---|---|",
        f"| token generation (e2e, c=1, {hero // 1024}K ctx) | "
        f"{tg_s:.1f} t/s | "
        f"{tg_fi:.1f} t/s {_delta(tg_fi, tg_s, 1, True)} | "
        f"{tg_g:.1f} t/s {_delta(tg_g, tg_s, 1, True)} |",
        f"| prompt processing (e2e, c=1, {hero // 1024}K ctx) | "
        f"{_thousands(pp_s, 0)} t/s | "
        f"{_thousands(pp_fi, 0)} t/s {_delta(pp_fi, pp_s, 0, True)} | "
        f"{_thousands(pp_g, 0)} t/s {_delta(pp_g, pp_s, 0, True)} |",
        f"| decode kernel (L={hero}) | "
        f"{fi_med:.1f} \u00b5s | "
        f"{fi_med:.1f} \u00b5s {_delta(fi_med, fi_med, 0, False)} | "
        f"{gemv_med:.1f} \u00b5s {_delta(gemv_med, fi_med, 0, False)}"
        f"<br />{cta1:.1f} \u00b5s 1-CTA carve-out |",
        f"| model memory (KV cache) | {kv_bits} | {kv_bits} | {kv_bits} |",
    ]
    return "\n".join(lines)


def _sweep_table(rf: dict) -> str:
    """The GEMV SplitKV sweep (one clean window): every row of the raw,
    Δ vs the FlashInfer baseline."""
    res = rf["results"]
    fi = res.get("flashinfer", {}).get("median")
    if fi is None:
        return ""

    def label(key: str) -> str:
        m = re.match(r"st(\d+)_(ns\d+|auto)$", key)
        if m:
            part = m.group(2)
            part = f"ns={part[2:]}" if part.startswith("ns") else "auto"
            return f"FA4 GEMV, stages={m.group(1)}, {part}"
        return key

    rows = []
    for key, d in res.items():
        if not d.get("median") or key == "flashinfer":
            continue
        m = re.match(r"st(\d+)_(ns\d+|auto)$", key)
        stages = int(m.group(1)) if m else -1
        ns = int(key.split("_ns")[1]) if "_ns" in key else None
        rows.append((stages, ns is None, ns or 0, key, d))
    # stages desc, numeric ns asc, auto last within a stage.
    rows.sort(key=lambda t: (-t[0], t[1], t[2]))

    lines = [
        "| decode path | \u00b5s (median) | nominal KV BW (GB/s) | \u0394 vs FlashInfer |",
        "|---|---:|---:|---:|",
        f"| FlashInfer FA2-tc, e4m3 KV \u2014 production baseline | "
        f"{_thousands(fi, 2)} | {res['flashinfer'].get('bw_gbs_nominal_kv', 0):.2f} | "
        f"{_delta(fi, fi, 1, False)} |",
    ]
    for _s, _a, _n, key, d in rows:
        lines.append(
            f"| {label(key)} | {_thousands(d['median'], 2)} | "
            f"{d.get('bw_gbs_nominal_kv', 0):.2f} | "
            f"{_delta(d['median'], fi, 1, False)} |"
        )
    return "\n".join(lines)


def _multil_table(multi: dict[str, dict[int, float]]) -> str | None:
    """The multi-L kernel session: per-kernel decode µs at every L (the
    auto-plan row for GEMV, best plan otherwise)."""
    if not multi:
        return None
    Ls = sorted({L for pts in multi.values() for L in pts})
    order = [m for m in _MULTIL_ORDER if m in multi]
    order += sorted(set(multi) - set(order))
    lines = [
        "| kernel | " + " | ".join(f"L={L}" for L in Ls) + " |",
        "|" + "---|" + "---:|" * len(Ls),
    ]
    for m in order:
        pts = multi[m]
        lines.append(
            f"| {_MULTIL_LABELS.get(m, m)} | "
            + " | ".join(f"{pts.get(L, float('nan')):.2f}" if L in pts else "—" for L in Ls)
            + " |"
        )
    return "\n".join(lines)


# ── context ─────────────────────────────────────────────────────────────────


@dataclass
class ReadmeContext:
    """Everything the template references — one namespaced root: the
    template only ever touches ``ctx.<field>`` (grep-friendly, no name
    collisions with prose, and a misspelled ``ctx.x`` fails loud at render
    time via StrictUndefined). A new data value = a new field here.

    Tables are pre-computed markdown blocks (color math included); ``None``
    when the source raws are absent (the template guards each block).
    Sparse values are pre-formatted strings; ``""`` when the raws are
    absent."""

    # badges / labels
    vllm_version: str = ""
    model: str = ""
    kv_bits: str = "8-bit"
    # pre-computed markdown tables
    table_intro: str | None = None
    table_gemv_sweep: str | None = None
    table_multil: str | None = None
    table_e2e_tg: str | None = None
    table_e2e_pp: str | None = None
    # sparse deep-dive values (the ring-fix window + the multi-L session)
    hero_L: str = ""
    kv_mib: str = ""
    roofline_gbs: str = ""
    ns1_bw: str = ""
    ns1_us: str = ""
    ns20_bw: str = ""
    ns20_us: str = ""
    ns_speedup: str = ""
    ns64_bw: str = ""
    ns64_us: str = ""
    auto_us: str = ""
    st2_us: str = ""
    st2_bw: str = ""
    st16_us: str = ""
    st16_bw: str = ""
    gemv_us: str = ""
    fi_us: str = ""
    gemv_vs_fi: str = ""
    cta1_us: str = ""
    cta1_L: str = ""


def collect_context(layout: RepoLayout) -> ReadmeContext:
    """Every value the README template may reference, computed from the
    committed raw data."""
    records = load_records(layout.history_file)
    rows = _e2e_latest(records)
    rf = _find_ringfix(layout)
    multi = _load_multil(layout)

    ctx = ReadmeContext(vllm_version=_vllm_base_version(layout), model=DEFAULT_MODEL)

    if rows:
        ctx.table_e2e_tg = _e2e_table(rows, "tg_tps", "backend (image, config)", "{:.2f}", 1, True)
        ctx.table_e2e_pp = _e2e_table(rows, "pp_tps", "backend", "{:.0f}", 1, True, label_style="short")
        if rf:
            ctx.table_intro = _intro_table(rows, rf, multi, ctx.kv_bits)

    if rf:
        ctx.table_gemv_sweep = _sweep_table(rf) or None
        env = rf.get("env", {})
        res = rf["results"]

        def g(key: str, field: str):
            return (res.get(key) or {}).get(field)

        kv_bytes = env.get("kv_bytes")
        if kv_bytes:
            ctx.kv_mib = f"{kv_bytes / 2**20:.0f}"
        bw = env.get("roofline_bw_bytes_per_s")
        if bw:
            ctx.roofline_gbs = f"{bw / 1e9:.0f}"
        L = env.get("L")
        hero = L or 8192
        ctx.hero_L = str(hero)
        fi = g("flashinfer", "median")
        ns1 = g("st16_ns1", "median")
        ns20 = g("st16_ns20", "median") or g("st16_auto", "median")
        ns64 = g("st16_ns64", "median")
        auto = g("st16_auto", "median")
        st2 = g("st2_ns20", "median")
        if fi:
            ctx.fi_us = f"{fi:.1f}"
        if ns1:
            ctx.ns1_bw = f"{g('st16_ns1', 'bw_gbs_nominal_kv'):.2f}"
            ctx.ns1_us = _thousands(ns1, 1)
        if ns20:
            ctx.ns20_bw = f"{g('st16_ns20', 'bw_gbs_nominal_kv'):.2f}"
            ctx.ns20_us = f"{ns20:.1f}"
            ctx.st16_us = f"{ns20:.1f}"
            ctx.st16_bw = f"{g('st16_ns20', 'bw_gbs_nominal_kv'):.2f}"
            ctx.gemv_us = f"{ns20:.1f}"
        if ns1 and ns20:
            ctx.ns_speedup = f"{ns1 / ns20:.1f}"
        if fi and ns20:
            ctx.gemv_vs_fi = f"{abs(ns20 - fi) / fi * 100:.0f}"
        if ns64:
            ctx.ns64_bw = f"{g('st16_ns64', 'bw_gbs_nominal_kv'):.2f}"
            ctx.ns64_us = f"{ns64:.1f}"
        if auto:
            ctx.auto_us = f"{auto:.1f}"
        if st2:
            ctx.st2_us = f"{st2:.1f}"
            ctx.st2_bw = f"{g('st2_ns20', 'bw_gbs_nominal_kv'):.2f}"
        cta1 = (multi.get("fa4_1cta") or {}).get(hero)
        if cta1:
            ctx.cta1_us = f"{cta1:.1f}"
            ctx.cta1_L = str(hero)

    if multi:
        ctx.table_multil = _multil_table(multi)
    return ctx


def render_readme(layout: RepoLayout, template: Path | None = None, out: Path | None = None) -> Path:
    """Render README.md from the Jinja2 template (repo root by default)."""
    tpl = Path(template) if template else layout.repo_root / "README.md.j2"
    dest = Path(out) if out else layout.repo_root / "README.md"
    if not tpl.exists():
        raise FileNotFoundError(
            f"README template not found: {tpl} (git-managed — restore it or pass --readme-template)"
        )
    ctx = collect_context(layout)
    env = jinja2.Environment(
        undefined=jinja2.StrictUndefined,
        keep_trailing_newline=True,
        autoescape=False,
    )
    rendered = env.from_string(tpl.read_text()).render(ctx=ctx)
    dest.write_text(rendered)
    print(f"[mjolnir] wrote {dest}")
    return dest
