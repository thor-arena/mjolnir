"""Everything a README / docs template reads: ``collect_context`` gathers the
versions, headline numbers, per-stage tables and findings, the official-vs-
Mjolnir e2e comparison, the accuracy tables and the patch inventory into one
``ReadmeContext``."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from mjolnir_cli import accuracy, config, e2e
from mjolnir_cli.config import DEFAULT_MODEL, IMAGE_REPO, RepoLayout, by_model
from mjolnir_cli.readme.findings import findings
from mjolnir_cli.readme.fmt import L_label, fmt_us, md_table
from mjolnir_cli.readme.tables import TABLES, vs_vllm_table, wall_table
from mjolnir_cli.verify import results
from mjolnir_cli.verify.latest import load_verify
from mjolnir_cli.verify.stages import ALL_LS, NOISE_PCT, STAGES


@dataclass
class ReadmeContext:
    vllm_version: str = ""
    model: str = DEFAULT_MODEL
    image: str = ""
    kernel_version: str = ""
    cli_version: str = ""
    ls: str = " ".join(L_label(L) for L in ALL_LS)
    noise_pct: str = f"{NOISE_PCT:.0f}"
    e2e_noise_pct: str = field(default_factory=lambda: f"{e2e.E2E_NOISE_PCT:.0f}")
    # headline (sweep)
    has_sweep: bool = False
    geo_vs_fi: str = ""
    geo_vs_fi_fp8: str = ""
    geo_vs_fa2: str = ""
    min_vs_fi: str = ""
    max_vs_fi: str = ""
    min_vs_fi_L: str = ""
    max_vs_fi_L: str = ""
    long_vs_fi: str = ""  # at the largest L
    long_L: str = ""
    beats_fi: str = ""  # "8 of 8"
    us_64k_mjolnir: str = ""
    us_64k_fi: str = ""
    us_64k_fa2: str = ""
    correctness: str = ""  # "96 / 96"
    sweep_run: str = ""
    tables: dict = field(default_factory=dict)
    findings: dict = field(default_factory=dict)
    runs: dict = field(default_factory=dict)  # stage -> run_id
    pending: list = field(default_factory=list)
    # official vLLM image vs the Mjolnir image (mjolnir_cli.e2e)
    vs_vllm: str | None = None  # the comparison table
    vs_vllm_hero: dict = field(default_factory=dict)  # c=1 headline gains, formatted
    vs_vllm_official: str = ""
    vs_vllm_mjolnir: str = ""
    vs_vllm_spec: str = ""  # speculative decoding both sides ran with, e.g. "MTP, 3 speculative tokens"
    vs_vllm_missing: str = ""
    e2e_example: dict = field(default_factory=dict)
    e2e_max_ctx: str = field(default_factory=lambda: L_label(e2e.MAX_CTX))
    # the README's Key Features grid: measured numbers, each None when its run is missing
    feat: dict = field(default_factory=dict)
    default_config: str = field(
        default_factory=lambda: config.DEFAULT_QUANT
    )  # contexts only one side measured (the command that completes the table)
    # model quality per serving config (mjolnir bench accuracy)
    accuracy_table: str = ""
    accuracy_categories: str = ""
    accuracy_findings: list = field(default_factory=list)
    accuracy_runs: list = field(default_factory=list)  # run ids the table draws on
    accuracy_args: list = field(default_factory=list)  # (suite label, tool-eval-bench args)
    accuracy_baseline: str = ""
    accuracy_excluded_note: str = ""
    # the image's patch inventory (docker/vllm-thor/PATCHES.md) — verbatim table + row counts
    patches_table: str = ""
    patches_total: int = 0
    patches_pr: int = 0


def _vllm_base_version(layout: RepoLayout) -> str:
    if layout.repo_root is None:
        return ""
    dockerfile = layout.repo_root / "docker" / "vllm-thor" / "Dockerfile"
    if dockerfile.exists():
        m = re.search(r"vllm/vllm-openai:v(\d+\.\d+\.\d+)", dockerfile.read_text())
        if m:
            return m.group(1)
    return ""


def _patches(layout: RepoLayout) -> tuple[str, int, int]:
    """The patch inventory table from docker/vllm-thor/PATCHES.md, re-emitted
    verbatim — it is the single source of truth the build consumes. Returns
    (table, total, pr_backed); pr_backed = rows patching a vLLM PR."""
    p = layout.repo_root / "docker" / "vllm-thor" / "PATCHES.md"
    if not p.exists():
        return "", 0, 0
    lines = p.read_text().splitlines()
    start = next((i for i, ln in enumerate(lines) if ln.startswith("| File |")), None)
    if start is None or start + 1 >= len(lines) or not lines[start + 1].startswith("|-"):
        raise ValueError(f"{p}: the '| File | PR | What it fixes |' inventory table is missing")
    i = start + 2
    rows = []
    while i < len(lines) and lines[i].startswith("|"):
        rows.append(lines[i])
        i += 1
    if not rows:
        raise ValueError(f"{p}: the inventory table has no rows")
    return (
        "\n".join([lines[start], lines[start + 1], *rows]),
        len(rows),
        sum(1 for r in rows if "vllm-project/vllm/pull/" in r),
    )


def _version_from(path: Path) -> str:
    try:
        m = re.search(r'__version__\s*=\s*"([^"]+)"', path.read_text())
        return m.group(1) if m else ""
    except OSError:
        return ""


def collect_context(layout: RepoLayout, data: dict | None = None) -> ReadmeContext:
    data = data if data is not None else load_verify(layout)
    ctx = ReadmeContext(vllm_version=_vllm_base_version(layout))
    ctx.image = f"{IMAGE_REPO}:v{ctx.vllm_version}" if ctx.vllm_version else IMAGE_REPO
    if layout.repo_root is not None:
        ctx.kernel_version = _version_from(layout.repo_root / "src" / "mjolnir" / "mjolnir" / "__init__.py")
        ctx.cli_version = _version_from(layout.repo_root / "src" / "mjolnir-cli" / "mjolnir_cli" / "__init__.py")
        ctx.patches_table, ctx.patches_total, ctx.patches_pr = _patches(layout)
    for stage, doc in data.items():
        ctx.runs[stage] = doc["run_id"]
        if stage in TABLES:
            try:
                ctx.tables[stage] = TABLES[stage](doc)
            except (KeyError, ZeroDivisionError, TypeError, ValueError) as e:
                print(f"[mjolnir] table {stage} skipped: {e!r}")
        ctx.findings[stage] = findings(stage, doc)
        if stage in ("kvfmt", "ab"):
            w = wall_table(doc)
            if w:
                ctx.tables[f"{stage}_wall"] = w
    ctx.pending = [s for s in STAGES if s not in data and s not in ("timing", "vllm", "profile")]

    sweep = data.get("sweep")
    if sweep:
        by = results.series_by_id(sweep)
        ours = by.get("mjolnir") or by.get("mjolnir-fp8")
        if ours and "fi" in by:
            ctx.has_sweep = True
            ctx.sweep_run = sweep["run_id"]
            sp = results.speedup(by["fi"], ours)
            ctx.geo_vs_fi = f"{results.geomean(sp.values()):.2f}"
            lo, hi = min(sp, key=sp.get), max(sp, key=sp.get)
            ctx.min_vs_fi, ctx.min_vs_fi_L = f"{sp[lo]:.2f}", L_label(lo)
            ctx.max_vs_fi, ctx.max_vs_fi_L = f"{sp[hi]:.2f}", L_label(hi)
            last = max(sp)
            ctx.long_vs_fi, ctx.long_L = f"{sp[last]:.2f}", L_label(last)
            ctx.beats_fi = f"{sum(1 for v in sp.values() if v >= 1 + NOISE_PCT / 100)} of {len(sp)}"
            if "mjolnir-fp8" in by:
                ctx.geo_vs_fi_fp8 = f"{results.geomean(results.speedup(by['fi'], by['mjolnir-fp8']).values()):.2f}"
            if "fa2" in by:
                ctx.geo_vs_fa2 = f"{results.geomean(results.speedup(by['fa2'], ours).values()):.2f}"
            pm = results.point_map(ours)
            if 65536 in pm:
                ctx.us_64k_mjolnir = fmt_us(pm[65536])
                ctx.us_64k_fi = fmt_us(results.point_map(by["fi"]).get(65536))
                if "fa2" in by:
                    ctx.us_64k_fa2 = fmt_us(results.point_map(by["fa2"]).get(65536))
    corr = data.get("correctness")
    if corr:
        cs = corr.get("cases", [])
        ctx.correctness = f"{sum(1 for c in cs if c.get('passed'))} / {len(cs)}"
    off, mj = e2e.latest_pair(layout.history_file)
    if off and mj:
        ctx.vs_vllm = vs_vllm_table(off, mj)
        ctx.vs_vllm_official, ctx.vs_vllm_mjolnir = e2e.side_label(off), e2e.side_label(mj)
        ctx.vs_vllm_spec = e2e.spec_label(off, mj)
        gaps = e2e.missing_contexts(off, mj)
        notes = [
            f"the {'Mjolnir' if side == 'mjolnir' else 'official'} side has no run at "
            f"{', '.join(e2e.ctx_label(c) for c in ctxs)} context yet (`mjolnir bench e2e --only {side} --depths "
            f"{','.join(str(c) for c in ctxs)}`)"
            for side, ctxs in gaps.items()
            if ctxs
        ]
        ctx.vs_vllm_missing = "; ".join(notes)
        h = e2e.hero(off, mj)
        if h and h["ttft_ms"][2] is not None:
            a, b = h["ttft_ms"][0], h["ttft_ms"][1]
            ctx.e2e_example = {
                "ctx": e2e.ctx_label(h["ctx"]),
                "a": a,
                "b": b,
                "change": e2e.change_pct(a, b),
                "ratio": a / b,
            }
        if h:
            ctx.vs_vllm_hero = {
                "ctx": e2e.ctx_label(h["ctx"]),
                "tg_tps": e2e.pct(h["tg_tps"][2]),
                "pp_tps": e2e.pct(h["pp_tps"][2]),
                "ttft_ms": f"{h['ttft_ms'][0] / 1000:.2f} s → {h['ttft_ms'][1] / 1000:.2f} s "
                f"({e2e.pct(-h['ttft_ms'][2])})"
                if h["ttft_ms"][2] is not None
                else "—",
                "itl_ms": f"{h['itl_ms'][0]:.1f} ms → {h['itl_ms'][1]:.1f} ms ({e2e.pct(-h['itl_ms'][2])})"
                if h["itl_ms"][2] is not None
                else "—",
            }
    acc_rows = accuracy.latest_legs(accuracy.load_runs(layout))  # read once, for the page and the hero
    _accuracy_context(ctx, acc_rows)
    _features(ctx, data, layout, acc_rows)
    return ctx


def _features(ctx: ReadmeContext, data: dict, layout: RepoLayout, acc_rows: list[dict]) -> None:
    """The numbers the Key Features grid quotes — from the same raw results as
    the tables (the grid falls back to plain text where a run is missing)."""
    f: dict = {"geo": ctx.geo_vs_fi or None}
    off, mj = e2e.latest_pair(layout.history_file)
    h = e2e.hero(off, mj) if off and mj else None
    if h:
        f.update(
            ctx=e2e.ctx_label(h["ctx"]),
            decode=f"+{h['tg_tps'][2]:.0f}%",
            prefill=f"+{h['pp_tps'][2]:.0f}%",
            ttft=f"−{h['ttft_ms'][2]:.0f}%" if h["ttft_ms"][2] is not None else None,
        )
    kv = data.get("kv-cache-size")
    if kv:
        rows = [r for r in kv.get("rows", []) if r.get("kv_tokens")]
        base = next((r for r in rows if r.get("side") == "official"), None)
        fp6 = max(
            (r for r in rows if str(r.get("kv_dtype", "")).startswith("fp6")),
            key=lambda r: r["kv_tokens"],
            default=None,
        )
        if base and fp6:
            f["kv"] = f"+{(fp6['kv_tokens'] / base['kv_tokens'] - 1) * 100:.0f}%"
    rows = by_model(acc_rows).get(DEFAULT_MODEL, [])  # the hero names one model's baseline
    b = accuracy.baseline(rows)
    if b:
        mine = [r for r in rows if r is not b and str(r.get("config", "")).startswith("NVFP4_MJOLNIR")]
        deltas = [abs(d[0]) for r in mine for n in ("tools", "hardmode") if (d := accuracy.delta(r, b, n))]
        if deltas:
            f["acc"] = f"{max(deltas):.1f}"
    ctx.feat = f


def _accuracy_context(ctx: ReadmeContext, rows: list[dict]) -> None:
    if not rows:
        return
    bases = accuracy.baselines(rows)
    names = accuracy.suites_in(rows)
    ctx.accuracy_baseline = accuracy.baseline_caption(bases)
    if accuracy.any_excluded(rows, bases, names):
        ctx.accuracy_excluded_note = accuracy.EXCLUDED_NOTE
    ctx.accuracy_runs = sorted({v["run_id"] for r in rows for v in r["runs"].values()})
    args = {}
    for r in rows:
        for n, v in r["runs"].items():
            args.setdefault(n, " ".join(v.get("args") or []))
    ctx.accuracy_args = [(accuracy.suite_label(n), args[n]) for n in names if n in args]

    def cell(r: dict, base: dict | None, n: str) -> str:
        v = (r["suites"].get(n) or {}).get("score")
        if v is None:
            return "—"
        out = f"{v:.1f}"
        if (r.get("suites", {}).get(n) or {}).get("n", 1) > 1:
            out += f" <sub>n={r['suites'][n]['n']}</sub>"
        return out + accuracy.fmt_delta(accuracy.delta(r, base, n))

    head = ["config", "KV", *[f"{accuracy.suite_label(n)} %" for n in names], "completion"]
    body = []
    for r, base in zip(rows, bases):
        comp = [x.get("completion_rate") for x in r["suites"].values() if x.get("completion_rate") is not None]
        label = f"**{r['label']}**" if r is base else r["label"]
        kv = str(r.get("kv_cache_dtype") or "auto").replace("fp6_e2m3", "fp6 e2m3").replace("auto", "bf16")
        body.append([label, kv, *[cell(r, base, n) for n in names], f"{min(comp):.0f}%" if comp else "—"])
    ctx.accuracy_table = md_table(head, body, ["---", "---"] + ["--:"] * (len(head) - 2))

    cats = sorted({c for r in rows for c in r.get("categories", {})})
    if cats:
        labels = {c: next(r["categories"][c]["label"] for r in rows if c in r.get("categories", {})) for c in cats}
        marked = [accuracy.leg_excluded(r, b, names) for r, b in zip(rows, bases)]
        chead = ["category", *[r["label"] + (" ‡" if m else "") for r, m in zip(rows, marked)]]
        cbody = []
        for c in cats:
            cbody.append(
                [f"{c} · {labels[c]}"]
                + [f"{r['categories'][c]['percent']:.0f}" if c in r.get("categories", {}) else "—" for r in rows]
            )
        ctx.accuracy_categories = md_table(chead, cbody)
        if any(marked):
            ctx.accuracy_categories += f"\n\n<sub>{accuracy.CATEGORY_EXCLUDED_NOTE}.</sub>"

    for r, base in zip(rows, bases):
        if base is None or r is base:
            continue
        deltas = [
            f"{accuracy.suite_label(n)} {d[0]:+.1f}{'‡' if d[1] else ''}"
            for n in names
            if (d := accuracy.delta(r, base, n))
        ]
        if deltas:
            changed = []
            if r.get("scenarios") and base.get("scenarios"):
                # a scenario one leg excluded counts as zero there, as in the ‡ delta
                rs, bs = r["scenarios"], base["scenarios"]
                every = rs.keys() | bs.keys()
                worse = sum(1 for k in every if rs.get(k, 0.0) < bs.get(k, 0.0))
                better = sum(1 for k in every if rs.get(k, 0.0) > bs.get(k, 0.0))
                gone = sorted(every - (rs.keys() & bs.keys()))
                changed = [
                    f"{worse} scenarios scored lower, {better} higher of {len(every)}"
                    + (f"; excluded {', '.join(gone)} counted as zero" if gone else "")
                ]
            ctx.accuracy_findings.append(
                f"**{r['label']}** vs {base['label']}: {', '.join(deltas)} points"
                + (f" ({changed[0]})" if changed else "")
                + "."
            )
