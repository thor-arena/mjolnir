"""The interactive benchmark site (GitHub Pages): one self-contained HTML page
built from the same raw verify documents as the PNG charts.

``mjolnir plot --site <dir>`` writes ``<dir>/index.html`` (+ ``data.json``);
the ``pages`` workflow publishes it. Plotly (pinned, jsdelivr) draws the charts
with hover tooltips; colors and the light/dark switch follow ``theme.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

from mjolnir_cli import theme
from mjolnir_cli.config import RepoLayout

PLOTLY = "https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"


def _slim(doc: dict) -> dict:
    """Drop per-kernel detail and full ncu detail pages to keep the page small."""
    d = {k: v for k, v in doc.items() if not k.startswith("_")}
    if "series" in d:
        d["series"] = [
            {**s, "points": [{k: v for k, v in p.items() if k != "kernels"} for p in s.get("points", [])]}
            for s in d["series"]
        ]
    if "profiles" in d:
        d["profiles"] = [{k: v for k, v in p.items() if k not in ("details", "pipes")} for p in d["profiles"]]
    if "cases" in d:
        d["cases"] = [{k: c.get(k) for k in ("label", "q", "passed", "check")} for c in d["cases"]]
    return d


def build(layout: RepoLayout, out_dir: Path, data: dict | None = None) -> Path:
    from mjolnir_cli.charts import e2e_rows
    from mjolnir_cli.verify.latest import load_verify
    from mjolnir_cli.verify.stages import STAGES

    data = data if data is not None else load_verify(layout)
    payload = {
        "stages": {k: _slim(v) for k, v in data.items()},
        "titles": {k: {"title": s.title, "summary": s.summary} for k, s in STAGES.items()},
        "e2e": {
            k: {
                "config": v.get("config"),
                "image": v.get("image"),
                "cells": [
                    {
                        "concurrency": c.get("concurrency"),
                        "context": c.get("context") or 0,
                        "tg": (c.get("tg_tps") or {}).get("mean"),
                        "pp": (c.get("pp_tps") or {}).get("mean") if c.get("context") else None,  # no ctx: no prefill
                    }
                    for c in v.get("cells", [])
                ],
            }
            for k, v in e2e_rows(layout).items()
        },
        "colors": {
            m.name: {
                "bg": m.bg,
                "panel": m.panel,
                "text": m.text,
                "dim": m.text_dim,
                "grid": m.grid,
                "mjolnir": m.mjolnir,
                "fi": m.flashinfer,
                "fa2": m.fa2,
                "neutral": m.neutral,
            }
            for m in theme.MODES
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "data.json").write_text(json.dumps(payload, separators=(",", ":")))
    from mjolnir_cli.verify.stages import NOISE_PCT

    html = (
        _PAGE.replace("__PLOTLY__", PLOTLY)
        .replace("__NOISE_PCT__", f"{NOISE_PCT:g}")
        .replace("__DATA__", json.dumps(payload, separators=(",", ":")))
    )
    p = out_dir / "index.html"
    p.write_text(html)
    print(f"[mjolnir] interactive site -> {p}")
    return p


_PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Mjolnir Benchmarks</title>
<meta name="description" content="Mjolnir decode-attention kernel on Jetson AGX Thor vs FlashInfer and FlashAttention 2 — interactive, generated from the repo's raw benchmark data.">
<script src="__PLOTLY__"></script>
<style>
:root {
  color-scheme: light;
  --bg: #ffffff; --panel: #fcfcfb; --text: #0b0b0b; --dim: #52514e; --grid: #e6e5e1;
  --accent: #eb6834; --link: #2a78d6;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --bg: #0d1117; --panel: #151b26; --text: #e6edf3; --dim: #8b949e; --grid: #21262d;
    --accent: #d95926; --link: #3987e5;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --bg: #0d1117; --panel: #151b26; --text: #e6edf3; --dim: #8b949e; --grid: #21262d;
  --accent: #d95926; --link: #3987e5;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--text);
  font: 15px/1.55 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
a { color: var(--link); }
header, main, footer { max-width: 1180px; margin: 0 auto; padding: 0 16px; }
header { padding-top: 32px; display: flex; gap: 16px; align-items: flex-start; justify-content: space-between; flex-wrap: wrap; }
h1 { font-size: 30px; margin: 0 0 6px; letter-spacing: -0.01em; }
h1 span { color: var(--accent); }
.sub { color: var(--dim); margin: 0; max-width: 760px; }
.toggle { border: 1px solid var(--grid); background: var(--panel); color: var(--text); border-radius: 8px;
  padding: 6px 12px; font: inherit; cursor: pointer; }
nav { position: sticky; top: 0; background: var(--bg); z-index: 2; border-bottom: 1px solid var(--grid); }
nav div { max-width: 1180px; margin: 0 auto; padding: 8px 16px; display: flex; gap: 6px; overflow-x: auto; }
nav a { white-space: nowrap; text-decoration: none; color: var(--dim); padding: 4px 10px; border-radius: 6px; font-size: 14px; }
nav a:hover { color: var(--text); background: var(--panel); }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; margin: 24px 0 8px; }
.tile { background: var(--panel); border: 1px solid var(--grid); border-radius: 10px; padding: 14px 16px; }
.tile b { display: block; font-size: 28px; color: var(--text); }
.tile span { color: var(--dim); font-size: 13px; }
section { margin: 36px 0; }
section h2 { font-size: 21px; margin: 0 0 4px; }
section p.what { color: var(--dim); margin: 0 0 10px; }
.chart { background: var(--panel); border: 1px solid var(--grid); border-radius: 10px; min-height: 380px; }
details { margin-top: 8px; }
summary { cursor: pointer; color: var(--dim); font-size: 14px; }
table { border-collapse: collapse; width: 100%; font-size: 13px; margin-top: 8px; font-variant-numeric: tabular-nums; }
th, td { border-bottom: 1px solid var(--grid); padding: 5px 8px; text-align: right; white-space: nowrap; }
th:first-child, td:first-child { text-align: left; }
.tablewrap { overflow-x: auto; }
.empty { color: var(--dim); padding: 24px; }
footer { color: var(--dim); font-size: 13px; padding: 32px 16px 48px; }
code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.92em; }
</style>
</head>
<body>
<header>
  <div>
    <h1>⚡ <span>Mjolnir</span> benchmarks</h1>
    <p class="sub">A decode-attention kernel for the Jetson AGX Thor (<code>sm_110</code>) against FlashInfer and
    FlashAttention 2, at every context length from 128 to 256K tokens. Every chart here is generated from the
    raw results in <a href="https://github.com/thor-arena/mjolnir/tree/main/benchmarks/raw/verify">benchmarks/raw/verify</a>
    — hover any point for its value.</p>
  </div>
  <button class="toggle" id="toggle" type="button">Toggle theme</button>
</header>
<nav><div id="nav"></div></nav>
<main>
  <div class="tiles" id="tiles"></div>
  <div id="sections"></div>
</main>
<footer>
  Generated by <code>mjolnir plot --site</code> from <a href="https://github.com/thor-arena/mjolnir">thor-arena/mjolnir</a>.
  Reproduce any number with <code>mjolnir verify &lt;stage&gt;</code>; ncu kernel time, cold L2, base clocks;
  ±__NOISE_PCT__% is the run-to-run noise of a same-run ratio.
</footer>
<script>
const D = __DATA__;
const NOISE = __NOISE_PCT__ / 100;
const LS_LABEL = L => L >= 1024 && L % 1024 === 0 ? (L / 1024) + "K" : String(L);
const NAMES = {"mjolnir": "Mjolnir · e2m3 KV", "mjolnir-fp8": "Mjolnir · fp8 KV", "fi": "FlashInfer · fp8 KV",
               "fi-hnd": "FlashInfer · HND", "fa2": "FlashAttention 2 · bf16 KV"};
const STYLE = {"mjolnir": ["mjolnir", "circle", true], "mjolnir-fp8": ["mjolnir", "square-open", false],
               "fi": ["fi", "diamond", true], "fi-hnd": ["fi", "diamond-open", false], "fa2": ["fa2", "triangle-up", true]};
function mode() {
  const t = document.documentElement.dataset.theme;
  if (t) return t;
  return matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}
function C() { return D.colors[mode()]; }
function byId(doc) { const o = {}; (doc.series || []).forEach(s => o[s.id] = s); return o; }
function pm(s, f) { const o = {}; (s.points || []).forEach(p => { if (p[f || "total_us"] != null) o[p.L] = p[f || "total_us"]; }); return o; }
function geomean(xs) { xs = xs.filter(x => x > 0); return xs.length ? Math.exp(xs.reduce((a, x) => a + Math.log(x), 0) / xs.length) : null; }
function fmtUs(t) { return t == null ? "—" : t >= 1000 ? (t / 1000).toFixed(2) + " ms" : t.toFixed(1) + " µs"; }
function layout(extra) {
  const c = C();
  return Object.assign({
    paper_bgcolor: c.panel, plot_bgcolor: c.panel, font: {color: c.text, family: "system-ui, sans-serif", size: 13},
    margin: {l: 64, r: 24, t: 24, b: 56}, hovermode: "x unified",
    hoverlabel: {bgcolor: c.bg, bordercolor: c.grid, font: {color: c.text}},
    legend: {orientation: "h", y: -0.2, font: {color: c.text}},
    xaxis: {gridcolor: c.grid, zeroline: false, color: c.dim},
    yaxis: {gridcolor: c.grid, zeroline: false, color: c.dim},
  }, extra || {});
}
function lAxis(Ls) { return {type: "log", tickvals: Ls, ticktext: Ls.map(LS_LABEL), title: "context length L (tokens)", gridcolor: C().grid, color: C().dim}; }
function line(sid, xs, ys, name, hover) {
  const c = C(), st = STYLE[sid] || ["mjolnir", "circle", true];
  return {x: xs, y: ys, name: name || NAMES[sid] || sid, mode: "lines+markers", type: "scatter",
          line: {color: c[st[0]], width: 2}, marker: {symbol: st[1], size: 9, color: st[2] ? c[st[0]] : c.panel, line: {color: c[st[0]], width: 2}},
          hovertemplate: hover || "%{y:.2f}<extra>%{fullData.name}</extra>"};
}
function noiseShapes(c) { return [{type: "rect", xref: "paper", x0: 0, x1: 1, y0: 1 - NOISE, y1: 1 + NOISE, fillcolor: c.grid, line: {width: 0}, layer: "below"},
                                   {type: "line", xref: "paper", x0: 0, x1: 1, y0: 1, y1: 1, line: {color: c.fi, width: 1.5}}]; }
function table(header, rows) {
  return '<div class="tablewrap"><table><thead><tr>' + header.map(h => "<th>" + h + "</th>").join("") + "</tr></thead><tbody>" +
    rows.map(r => "<tr>" + r.map(x => "<td>" + x + "</td>").join("") + "</tr>").join("") + "</tbody></table></div>";
}
const CHARTS = [];
function section(id, title, what, draw, tableHtml) {
  const el = document.createElement("section");
  el.id = id;
  el.innerHTML = "<h2>" + title + "</h2><p class='what'>" + what + "</p><div class='chart' id='c-" + id + "'></div>" +
    (tableHtml ? "<details><summary>data table</summary>" + tableHtml + "</details>" : "");
  document.getElementById("sections").appendChild(el);
  const a = document.createElement("a"); a.href = "#" + id; a.textContent = title;
  document.getElementById("nav").appendChild(a);
  CHARTS.push(() => draw("c-" + id));
}
function speedupSection() {
  const doc = D.stages.sweep; if (!doc) return;
  const by = byId(doc); if (!by.fi) return;
  const fi = pm(by.fi), Ls = Object.keys(fi).map(Number).sort((a, b) => a - b);
  section("sweep", "Mjolnir vs FlashInfer vs FlashAttention 2", "Speedup over FlashInfer at every context length (ncu kernel time; gray band = ±__NOISE_PCT__% noise).",
    id => {
      const traces = ["mjolnir", "mjolnir-fp8", "fa2"].filter(s => by[s]).map(s => {
        const o = pm(by[s]); return line(s, Ls.filter(L => o[L]), Ls.filter(L => o[L]).map(L => fi[L] / o[L]), null, "%{y:.2f}×<extra>%{fullData.name}</extra>");
      });
      Plotly.react(id, traces, layout({xaxis: lAxis(Ls), yaxis: {title: "speedup over FlashInfer (×)", gridcolor: C().grid, color: C().dim, rangemode: "tozero"}, shapes: noiseShapes(C())}), {responsive: true, displaylogo: false});
    },
    table(["L"].concat((doc.series || []).map(s => s.label)), Ls.map(L => [LS_LABEL(L)].concat((doc.series || []).map(s => fmtUs(pm(s)[L]))))));
  section("latency", "Decode latency per step", "Time for one decode step over the whole KV cache (log scale, lower is faster).",
    id => {
      const traces = ["fa2", "fi", "mjolnir-fp8", "mjolnir"].filter(s => by[s]).map(s => {
        const o = pm(by[s]); const xs = Object.keys(o).map(Number).sort((a, b) => a - b);
        return line(s, xs, xs.map(L => o[L]), null, "%{y:.1f} µs<extra>%{fullData.name}</extra>");
      });
      Plotly.react(id, traces, layout({xaxis: lAxis(Ls), yaxis: {type: "log", title: "µs per step", gridcolor: C().grid, color: C().dim}}), {responsive: true, displaylogo: false});
    });
  section("bandwidth", "Effective KV bandwidth", "Each leg's real KV bytes ÷ decode time, against the 273 GB/s LPDDR5X peak.",
    id => {
      const traces = ["fa2", "fi", "mjolnir-fp8", "mjolnir"].filter(s => by[s]).map(s => {
        const o = pm(by[s], "gbps"); const xs = Object.keys(o).map(Number).sort((a, b) => a - b);
        return line(s, xs, xs.map(L => o[L]), null, "%{y:.0f} GB/s<extra>%{fullData.name}</extra>");
      });
      Plotly.react(id, traces, layout({xaxis: lAxis(Ls), yaxis: {title: "GB/s", range: [0, 300], gridcolor: C().grid, color: C().dim},
        shapes: [{type: "line", xref: "paper", x0: 0, x1: 1, y0: 273, y1: 273, line: {color: C().dim, width: 1.2}}]}), {responsive: true, displaylogo: false});
    });
}
function ratioLines(stage, ids, title, what) {
  const doc = D.stages[stage]; if (!doc) return;
  const by = byId(doc); if (!by.fi) return;
  const fi = pm(by.fi), Ls = Object.keys(fi).map(Number).sort((a, b) => a - b);
  section(stage, title, what, id => {
    const traces = ids.filter(s => by[s]).map(s => { const o = pm(by[s]); return line(s, Ls.filter(L => o[L]), Ls.filter(L => o[L]).map(L => fi[L] / o[L]), by[s].label, "%{y:.2f}×<extra>%{fullData.name}</extra>"); });
    Plotly.react(id, traces, layout({xaxis: lAxis(Ls), yaxis: {title: "speedup over FlashInfer NHD (×)", gridcolor: C().grid, color: C().dim}, shapes: noiseShapes(C())}), {responsive: true, displaylogo: false});
  });
}
function heatmap(stage, title, what) {
  const doc = D.stages[stage]; if (!doc) return;
  const by = byId(doc); if (!by.fi) return;
  const fi = pm(by.fi), Ls = Object.keys(fi).map(Number).sort((a, b) => a - b);
  const vs = (doc.series || []).filter(s => s.kind === "t5");
  if (!vs.length) return;
  const z = vs.map(s => { const o = pm(s); return Ls.map(L => o[L] ? fi[L] / o[L] : null); });
  const zlog = z.map(r => r.map(v => v ? Math.max(-1, Math.min(1, Math.log2(v))) : null));
  section(stage, title, what, id => {
    const c = C();
    Plotly.react(id, [{type: "heatmap", x: Ls.map(LS_LABEL), y: vs.map(s => s.label), z: zlog, customdata: z, zmin: -1, zmax: 1,
      colorscale: [[0, c.fi], [0.5, c.neutral], [1, c.mjolnir]], xgap: 2, ygap: 2, showscale: false,
      text: z.map(r => r.map(v => v ? v.toFixed(2) + "×" : "")), texttemplate: "%{text}", textfont: {color: c.text},
      hovertemplate: "%{y} @ L=%{x}: %{customdata:.3f}× vs FlashInfer<extra></extra>"}],
      layout({margin: {l: 150, r: 24, t: 16, b: 48}, height: 120 + 44 * vs.length, xaxis: {color: c.dim, title: "context length L"}, yaxis: {color: c.text, autorange: "reversed"}, hovermode: "closest"}), {responsive: true, displaylogo: false});
  }, table(["variant", "flags"].concat(Ls.map(LS_LABEL)), vs.map((s, i) => [s.label, "<code>" + (s.flags || "(default)") + "</code>"].concat(z[i].map(v => v ? v.toFixed(2) + "×" : "—")))));
}
function quantSection() {
  const doc = D.stages.quant; if (!doc) return;
  const by = {}, bits = {};
  (doc.rows || []).forEach(r => { if (r.rel_l2 != null) { (by[r.format] = by[r.format] || []).push(r); bits[r.format] = r.bits; } });
  const fmts = Object.keys(by).sort((a, b) => (bits[b] || 0) - (bits[a] || 0));
  if (!fmts.length) return;
  const mean = f => by[f].reduce((a, r) => a + r.rel_l2, 0) / by[f].length;
  section("quant", "KV format accuracy", "Relative L2 error of the kernel output vs attention over unquantized KV — bar = mean over L, whisker = min–max over L.", id => {
    const c = C();
    const m = fmts.map(mean);
    Plotly.react(id, [{type: "bar", x: fmts.map(f => f + " · " + bits[f] + " bit"), y: m, marker: {color: c.mjolnir},
      error_y: {type: "data", symmetric: false, array: fmts.map((f, i) => Math.max(...by[f].map(r => r.rel_l2)) - m[i]),
                arrayminus: fmts.map((f, i) => m[i] - Math.min(...by[f].map(r => r.rel_l2))), color: c.dim, thickness: 1.5, width: 6},
      customdata: fmts.map(f => by[f].map(r => LS_LABEL(r.L) + ": " + r.rel_l2.toFixed(4)).join("<br>")),
      hovertemplate: "%{x}<br>mean %{y:.4f}<br>%{customdata}<extra></extra>"}],
      layout({yaxis: {title: "relative L2 error (lower is better)", gridcolor: c.grid, color: c.dim}, xaxis: {color: c.text}, hovermode: "closest", showlegend: false}),
      {responsive: true, displaylogo: false});
  }, table(["format", "bits"].concat(by[fmts[0]].map(r => LS_LABEL(r.L))), fmts.map(f => [f, bits[f]].concat(by[f].map(r => r.rel_l2.toFixed(4))))));
}
function pdlSection() {
  const doc = D.stages.pdl; if (!doc) return;
  const rows = (doc.rows || []).filter(r => r.delta_pct != null);
  const bs = [...new Set(rows.map(r => r.batches))];
  section("pdl", "Programmatic dependent launch", "Wall-clock change with PDL on — orange = faster, blue = slower; hatched = batch 8; gray band = ±__NOISE_PCT__% noise.", id => {
    const c = C();
    Plotly.react(id, bs.map((b, i) => { const rs = rows.filter(r => r.batches === b);
      return {type: "bar", name: "batch " + b, x: rs.map(r => LS_LABEL(r.L)), y: rs.map(r => r.delta_pct),
              marker: {color: rs.map(r => r.delta_pct < 0 ? c.mjolnir : c.fi), pattern: {shape: i ? "/" : "", fgcolor: c.panel, solidity: 0.35}},
              hovertemplate: "batch " + b + " @ L=%{x}: %{y:+.1f}%<extra></extra>"}; }),
      layout({barmode: "group", yaxis: {title: "Δ wall clock (%)", gridcolor: c.grid, color: c.dim}, xaxis: {color: c.dim}, hovermode: "closest",
        shapes: [{type: "rect", xref: "paper", x0: 0, x1: 1, y0: -7, y1: 7, fillcolor: c.grid, line: {width: 0}, layer: "below"}]}), {responsive: true, displaylogo: false});
  });
}
function profSection() {
  const doc = D.stages.fp6prof; if (!doc) return;
  const ps = (doc.profiles || []).filter(p => p.ok);
  if (!ps.length) return;
  const labs = [...new Set(ps.map(p => p.label))], base = labs[0];
  const Ls = [...new Set(ps.map(p => p.L))].sort((a, b) => a - b);
  const get = (lab, L) => ps.find(p => p.label === lab && p.L === L);
  section("fp6prof", "Bytes vs time per KV format", "Full ncu profiles: L2 bytes read and kernel time relative to " + base + " (the ideal is the bits ratio).", id => {
    const c = C(), traces = [];
    labs.slice(1).forEach((lab, i) => {
      const t = Ls.map(L => { const b = get(base, L), p = get(lab, L); return b && p ? p.metrics["gpu__time_duration.sum"] / b.metrics["gpu__time_duration.sum"] : null; });
      const s = Ls.map(L => { const b = get(base, L), p = get(lab, L); const k = "lts__t_sectors_op_read.sum"; return b && p && b.metrics[k] ? p.metrics[k] / b.metrics[k] : null; });
      traces.push({type: "bar", name: lab + " · kernel time", x: Ls.map(LS_LABEL), y: t, marker: {color: c.mjolnir, opacity: i ? 0.55 : 1}});
      traces.push({type: "bar", name: lab + " · L2 bytes", x: Ls.map(LS_LABEL), y: s, marker: {color: c.dim, opacity: i ? 0.55 : 1}});
    });
    Plotly.react(id, traces, layout({barmode: "group", yaxis: {title: "ratio vs " + base, range: [0, 1.15], gridcolor: c.grid, color: c.dim}, xaxis: {color: c.dim}, hovermode: "x unified"}), {responsive: true, displaylogo: false});
  });
}
function e2eSection() {
  const rows = D.e2e; const keys = ["NVFP4", "NVFP4_FA2", "NVFP4_MJOLNIR", "NVFP4_MJOLNIR_E2M3"].filter(k => rows[k]);
  if (!(rows.NVFP4_MJOLNIR || rows.NVFP4_MJOLNIR_E2M3) || keys.length < 2) return;
  const label = {"NVFP4_MJOLNIR": "Mjolnir · fp8 KV", "NVFP4_MJOLNIR_E2M3": "Mjolnir · fp6 KV", "NVFP4": "FlashInfer · official vLLM", "NVFP4_FA2": "FlashAttention 2 · official vLLM"};
  const col = {"NVFP4_MJOLNIR": "mjolnir", "NVFP4_MJOLNIR_E2M3": "mjolnir", "NVFP4": "fi", "NVFP4_FA2": "fa2"};
  section("e2e", "End-to-end decode throughput", "The official vLLM image (native backends) vs the Mjolnir image, same model and config (llama-benchy, gated clean windows).", id => {
    const c = C();
    Plotly.react(id, keys.map(k => { const cs = rows[k].cells.filter(x => x.tg != null);
      return {type: "bar", name: label[k] || k, x: cs.map(x => "c" + x.concurrency + " · " + (x.context ? LS_LABEL(x.context) + " ctx" : "no ctx")), y: cs.map(x => x.tg),
              marker: {color: c[col[k]] || c.dim, opacity: k === "NVFP4_MJOLNIR" ? 0.55 : 1}, hovertemplate: "%{y:.1f} tok/s<extra>%{fullData.name}</extra>"}; }),
      layout({barmode: "group", yaxis: {title: "decode tokens/s", gridcolor: c.grid, color: c.dim}, xaxis: {color: c.dim}}), {responsive: true, displaylogo: false});
  });
}
function tiles() {
  const doc = D.stages.sweep, el = document.getElementById("tiles"), out = [];
  if (doc) {
    const by = byId(doc), ours = by.mjolnir || by["mjolnir-fp8"];
    if (ours && by.fi) {
      const fi = pm(by.fi), o = pm(ours), sp = Object.keys(o).filter(L => fi[L]).map(L => fi[L] / o[L]);
      out.push(["" + geomean(sp).toFixed(2) + "×", "geomean speedup over FlashInfer (all L)"]);
      out.push(["" + Math.max(...sp).toFixed(2) + "×", "best speedup over FlashInfer"]);
      if (by.fa2) { const fa = pm(by.fa2); const s2 = Object.keys(o).filter(L => fa[L]).map(L => fa[L] / o[L]); out.push([geomean(s2).toFixed(2) + "×", "geomean speedup over FlashAttention 2"]); }
    }
  }
  const corr = D.stages.correctness;
  if (corr) { const cs = corr.cases || []; out.push([cs.filter(c => c.passed).length + " / " + cs.length, "correctness checks passing"]); }
  if (!out.length) out.push(["—", "no verify runs yet — run mjolnir verify bench"]);
  el.innerHTML = out.map(t => "<div class='tile'><b>" + t[0] + "</b><span>" + t[1] + "</span></div>").join("");
}
function build() {
  tiles();
  speedupSection();
  ratioLines("hndsweep", ["fi-hnd", "mjolnir-fp8"], "Head-major KV pages", "Same data on NHD vs HND pages — FlashInfer gains a little, Mjolnir more.");
  heatmap("kvfmt", "KV layouts × formats", "Speedup over FlashInfer per KV format and page layout.");
  quantSection();
  heatmap("ring", "KV ring depth", "TMA → MMA ring depth (--kv_stages).");
  heatmap("prefetch", "L2 prefetch", "TMA L2 prefetch distance (--l2_prefetch).");
  pdlSection();
  heatmap("ab", "Flag A/B", "Harness flag sets (default: evict-first × PDL).");
  profSection();
  e2eSection();
  if (!document.getElementById("sections").children.length)
    document.getElementById("sections").innerHTML = "<p class='empty'>No benchmark data yet — run <code>mjolnir verify bench</code> and re-publish.</p>";
}
function redraw() { CHARTS.forEach(f => f()); }
document.getElementById("toggle").addEventListener("click", () => {
  document.documentElement.dataset.theme = mode() === "dark" ? "light" : "dark"; redraw();
});
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", redraw);
build(); redraw();
</script>
</body>
</html>
"""
