"""Pins what the mjolnir_cli.readme package exposes (each name from the module
that owns it, the facade only the render API) and what its table builders /
findings produce, so a module split cannot change either unnoticed."""

from __future__ import annotations

import pytest

from mjolnir_cli import readme
from mjolnir_cli.labels import kvcache_label, spec_label
from mjolnir_cli.readme.findings import findings
from mjolnir_cli.readme.fmt import L_label, color, fmt_us, pct_md, ratio_md, ratio_pct
from mjolnir_cli.readme.tables import TABLES

from verify_fixtures import all_docs
from test_kvcache import _doc as _kv_doc, _row as _kv_row


def test_readme_facade_exports_exactly_the_public_render_api():
    assert sorted(readme.__all__) == ["render_file", "render_readme"]


def _spec_doc():
    def row(cfg, c, k, tps, acc):
        return {
            "config": cfg,
            "concurrency": c,
            "k": k,
            "decode_tps": tps,
            "agg_tps": tps * c,
            "ttft_ms": 1500,
            "acceptance_length": acc,
            "per_pos": [0.9, 0.7, 0.5][:k],
        }

    return {
        "stage": "spec",
        "rows": [
            row("NVFP4", 1, 1, 40.0, 1.8),
            row("NVFP4", 1, 3, 55.0, 2.5),
            row("NVFP4_MJOLNIR_E2M3", 1, 3, 60.0, 2.75),
            row("NVFP4_MJOLNIR_E2M3", 1, 2, 50.0, 2.2),
        ],
    }


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _docs():
    d = dict(all_docs())
    d["spec"] = _spec_doc()
    d["kv-cache-size"] = _kv_doc(
        [_kv_row("official", "NVFP4", "fp8", 2000000), _kv_row("mjolnir", "NVFP4_MJOLNIR_E2M3", "fp6_e2m3", 2656033)]
    )
    return d


# stage -> (first header cells, number of body rows)
EXPECTED = {
    "sweep": (["L", "FlashInfer", "FA2", "Mjolnir fp8", "Mjolnir e2m3"], 8 + 1),  # 8 contexts + geomean
    "hndsweep": (["L", "FlashInfer", "FlashInfer HND", "Mjolnir fp8"], 8 + 1),
    "kvfmt": (["variant", "flags"], 7),
    "ring": (["variant", "flags"], 6),
    "prefetch": (["variant", "flags"], 4),
    "ab": (["variant", "flags"], 4),
    "quant": (["format", "bits"], 5),
    "pdl": (["Q", "batch"], 2),
    "fp6prof": (["L", "variant", "bits", "kernel time"], 3 * 8),
    "correctness": (["case group", "passed", "Q dtypes"], None),
    "kv-cache-size": (["config", "serving", "KV memory", "KV tokens"], 2),
    "spec": (["config", "c", "drafts", "tokens/step"], 4),
}


def test_tables_registry_covers_exactly_the_expected_stages():
    assert set(TABLES) == set(EXPECTED)


@pytest.mark.parametrize("stage", sorted(EXPECTED))
def test_table_builder_renders_markdown_table_with_header_and_rows_when_given_fixture(stage):
    head, nrows = EXPECTED[stage]
    table = TABLES[stage](_docs()[stage])
    lines = table.splitlines()
    assert all(ln.startswith("| ") or ln.startswith("|-") or ln.startswith("|:") for ln in lines)
    assert _cells(lines[0])[: len(head)] == head
    assert set(lines[1].replace("|", "")) <= set("-:")
    assert len(_cells(lines[1])) == len(_cells(lines[0]))
    body = lines[2:]
    if nrows is not None:
        assert len(body) == nrows
    assert all(len(_cells(ln)) == len(_cells(lines[0])) for ln in body)


def test_variant_table_lists_one_row_per_series_by_label_when_stage_is_kvfmt():
    doc = all_docs()["kvfmt"]
    body = TABLES["kvfmt"](doc).splitlines()[2:]
    labels = [s["label"] for s in doc["series"] if s["kind"] == "t5"]
    assert [_cells(ln)[0] for ln in body] == [f"`{lab}`" for lab in labels]


def test_variant_table_is_empty_when_reference_series_is_missing():
    doc = all_docs()["ring"]
    doc = {**doc, "series": [s for s in doc["series"] if s["id"] != "fi"]}
    assert TABLES["ring"](doc) == ""


def test_series_time_table_ends_with_geomean_row_and_bolds_fastest_time():
    lines = TABLES["sweep"](all_docs()["sweep"]).splitlines()
    assert _cells(lines[-1])[0] == "geomean"
    assert "**" in _cells(lines[2])[1] + _cells(lines[2])[2] + _cells(lines[2])[3] + _cells(lines[2])[4]


def test_correctness_table_counts_every_case_when_grouped():
    doc = all_docs()["correctness"]
    body = TABLES["correctness"](doc).splitlines()[2:]
    total = sum(int(_cells(ln)[1].split()[1].split("/")[1]) for ln in body)
    assert total == len(doc["cases"])
    assert all("✅" in _cells(ln)[1] for ln in body)


def test_correctness_table_marks_failing_group_when_a_case_fails():
    doc = all_docs()["correctness"]
    doc = {**doc, "cases": [{**c, "passed": False} if i == 0 else c for i, c in enumerate(doc["cases"])]}
    assert "❌" in TABLES["correctness"](doc)


def test_spec_table_stars_the_best_draft_count_per_config_and_concurrency():
    body = TABLES["spec"](_spec_doc()).splitlines()[2:]
    starred = [_cells(ln) for ln in body if "★" in ln]
    assert sorted((c[0], c[2]) for c in starred) == [
        ("Mjolnir · fp6 KV", "3 **★**"),
        ("official vLLM · FlashInfer fp8", "3 **★**"),
    ]


def test_spec_label_maps_known_configs_and_passes_unknown_through():
    assert spec_label("NVFP4") == "official vLLM · FlashInfer fp8"
    assert spec_label("NVFP4_MJOLNIR") == "Mjolnir · fp8 KV"
    assert spec_label("custom_cfg") == "custom_cfg"


def test_kvcache_label_names_official_and_mjolnir_sides_with_extras():
    assert kvcache_label({"side": "official", "backend": "flashinfer", "kv_dtype": "fp8"}) == (
        "official vLLM · FlashInfer · fp8 KV"
    )
    assert kvcache_label({"side": "mjolnir", "kv_dtype": "fp6_e2m3", "config": "A_LMH8_VT64K"}) == (
        "Mjolnir · fp6 e2m3 KV + fp8 lm_head + 64K draft vocab"
    )


def test_ratio_pct_colors_by_noise_band_and_flips_sign_for_latency():
    assert "#16a34a" in ratio_pct(10.0) and "+10.0" in ratio_pct(10.0)
    assert "#dc2626" in ratio_pct(-10.0) and "−10.0" in ratio_pct(-10.0)
    assert "−10.0" in ratio_pct(10.0, higher_better=False)
    assert ratio_pct(None) == "—"


# stage -> substrings that must each appear in some finding sentence
FINDINGS = {
    "sweep": ["geomean", "Mjolnir (fp8 KV, HND)** vs FlashInfer"],
    "hndsweep": ["geomean"],
    "kvfmt": ["Fastest variant at the most context lengths", "Best geomean vs FlashInfer"],
    "ring": ["Fastest variant at the most context lengths"],
    "prefetch": ["Best geomean vs FlashInfer"],
    "ab": ["Best geomean vs FlashInfer"],
    "quant": ["mean relative L2 error"],
    "pdl": ["move by less than", "noise band"],
    "fp6prof": ["of the byte saving becomes speed"],
    "correctness": ["checks pass"],
    "kv-cache-size": ["+33%", "tokens**"],
    "spec": ["best with **3 drafts**", "Acceptance, 3 drafts, 1 request"],
}


@pytest.mark.parametrize("stage", sorted(FINDINGS))
def test_findings_returns_sentences_for_fixture_run(stage):
    found = findings(stage, _docs()[stage])
    assert found and all(isinstance(f, str) and f.endswith((".", ")")) for f in found)
    for needle in FINDINGS[stage]:
        assert any(needle in f for f in found), (stage, needle, found)


def test_findings_is_empty_when_stage_has_no_data():
    assert findings("quant", {"rows": []}) == []
    assert findings("unknown-stage", {}) == []


def test_fmt_us_formats_none_submillisecond_and_millisecond_values_exactly():
    assert fmt_us(None) == "—"
    assert fmt_us(0.5) == "0.5&nbsp;µs"
    assert fmt_us(12.34) == "12.3&nbsp;µs"
    assert fmt_us(999.9) == "999.9&nbsp;µs"
    assert fmt_us(1234.5) == "1.23&nbsp;ms"
    assert fmt_us(64200) == "64.20&nbsp;ms"


def test_l_label_uses_k_suffix_only_for_whole_kibibyte_values():
    assert L_label(128) == "128"
    assert L_label(1500) == "1500"
    assert L_label(1024) == "1K"
    assert L_label(131072) == "128K"


def test_ratio_md_renders_exact_latex_for_win_noise_and_loss():
    assert ratio_md(None) == "—"
    assert ratio_md(1.5) == "$\\color{#16a34a}{\\mathbf{1.50\\times}}$"
    assert ratio_md(1.5, bold_wins=False) == "$\\color{#16a34a}{1.50\\times}$"
    assert ratio_md(1.0) == "$\\color{grey}{1.00\\times}$"
    assert ratio_md(0.9) == "$\\color{#dc2626}{0.90\\times}$"


def test_pct_md_renders_exact_latex_and_flips_color_with_direction():
    assert pct_md(None) == "—"
    assert pct_md(-3.0) == "$\\color{#16a34a}{\\text{−3.0\\\\%}}$"
    assert pct_md(3.0) == "$\\color{#dc2626}{\\text{+3.0\\\\%}}$"
    assert pct_md(3.0, lower_is_better=False) == "$\\color{#16a34a}{\\text{+3.0\\\\%}}$"
    assert pct_md(0.5) == "$\\color{grey}{\\text{+0.5\\\\%}}$"


def test_color_wraps_text_in_grey_scriptstyle_latex_by_default():
    assert color("x") == "$\\color{grey}\\scriptstyle\\text{x}$"
    assert color("x", "red") == "$\\color{red}\\scriptstyle\\text{x}$"


def test_series_time_table_renders_exact_markdown_for_hndsweep_fixture():
    expected = "| L | FlashInfer | FlashInfer HND | Mjolnir fp8 | fp8 vs FI | fp8 vs FI-HND |\n|---|--:|--:|--:|--:|--:|\n| **128** | 25.4&nbsp;µs | 25.3&nbsp;µs | **13.6&nbsp;µs** | $\\color{#16a34a}{\\mathbf{1.87\\times}}$ | $\\color{#16a34a}{\\mathbf{1.86\\times}}$ |\n| **4K** | 64.2&nbsp;µs | 61.0&nbsp;µs | **55.1&nbsp;µs** | $\\color{#16a34a}{\\mathbf{1.17\\times}}$ | $\\color{#16a34a}{\\mathbf{1.11\\times}}$ |\n| **8K** | 103.0&nbsp;µs | 96.0&nbsp;µs | **87.7&nbsp;µs** | $\\color{#16a34a}{\\mathbf{1.17\\times}}$ | $\\color{#16a34a}{\\mathbf{1.09\\times}}$ |\n| **16K** | 191.5&nbsp;µs | 175.4&nbsp;µs | **169.6&nbsp;µs** | $\\color{#16a34a}{\\mathbf{1.13\\times}}$ | $\\color{#16a34a}{\\mathbf{1.03\\times}}$ |\n| **32K** | 371.9&nbsp;µs | **344.8&nbsp;µs** | 354.7&nbsp;µs | $\\color{#16a34a}{\\mathbf{1.05\\times}}$ | $\\color{grey}{0.97\\times}$ |\n| **64K** | 763.4&nbsp;µs | 716.9&nbsp;µs | **676.3&nbsp;µs** | $\\color{#16a34a}{\\mathbf{1.13\\times}}$ | $\\color{#16a34a}{\\mathbf{1.06\\times}}$ |\n| **128K** | 1.37&nbsp;ms | 1.31&nbsp;ms | **1.27&nbsp;ms** | $\\color{#16a34a}{\\mathbf{1.08\\times}}$ | $\\color{#16a34a}{\\mathbf{1.03\\times}}$ |\n| **256K** | 2.56&nbsp;ms | 2.53&nbsp;ms | **2.40&nbsp;ms** | $\\color{#16a34a}{\\mathbf{1.07\\times}}$ | $\\color{#16a34a}{\\mathbf{1.05\\times}}$ |\n| geomean |  |  |  | $\\color{#16a34a}{\\mathbf{1.19\\times}}$ | $\\color{#16a34a}{\\mathbf{1.13\\times}}$ |"
    assert TABLES["hndsweep"](all_docs()["hndsweep"]) == expected
