"""Official vLLM image vs Mjolnir image: pairing, gains, ITL/TTFT fallbacks."""

from __future__ import annotations

import json

from mjolnir_cli import e2e
from mjolnir_cli.history import parse_benchy_json


def _cell(conc, ctx, tg, pp, ttfr):
    return {
        "concurrency": conc,
        "context": ctx,
        "tg_tps": {"mean": tg},
        "pp_tps": {"mean": pp},
        "ttfr_ms": {"mean": ttfr},
    }


def _rec(image, config, epoch, cells, model="Qwen/Qwen3.8-27B"):
    return {"model": model, "image": image, "config": config, "epoch": epoch, "backend": "x", "cells": cells}


def test_pair_prefers_mjolnir_backend_and_latest(tmp_path):
    h = tmp_path / "history.jsonl"
    rows = [
        _rec("vllm/vllm-openai:v0.30.0-ubuntu2404", "NVFP4", 1, [_cell(1, 8192, 20, 2000, 4000)]),
        _rec("vllm/vllm-openai:v0.30.0-ubuntu2404", "NVFP4", 3, [_cell(1, 8192, 22, 2100, 3900)]),
        _rec("narandill/mjolnir-vllm:v0.30.0", "NVFP4", 5, [_cell(1, 8192, 23, 2500, 3500)]),
        _rec("narandill/mjolnir-vllm:v0.30.0", "NVFP4_MJOLNIR", 4, [_cell(1, 8192, 30, 2600, 3200)]),
    ]
    h.write_text("".join(json.dumps(r) + "\n" for r in rows))
    off, mj = e2e.latest_pair(h, "NVFP4_MJOLNIR")
    assert off["epoch"] == 3 and mj["config"] == "NVFP4_MJOLNIR"  # backend config beats a newer FI run
    hero = e2e.hero(off, mj)
    assert hero["ctx"] == 8192
    assert round(hero["tg_tps"][2], 1) == round((30 / 22 - 1) * 100, 1)
    # lower-is-better metrics: positive gain = the latency REDUCTION (official − Mjolnir) ÷ official,
    # not the reciprocal speedup official ÷ Mjolnir − 1; TTFT falls back to ttfr, ITL to 1000/tg at c=1
    assert round(hero["ttft_ms"][2], 1) == round((3900 - 3200) / 3900 * 100, 1) == 17.9
    assert round(e2e.change_pct(3900, 3200), 1) == -17.9
    assert round(hero["itl_ms"][1], 3) == round(1000 / 30, 3)


def test_benchy_cells_carry_ttft_and_itl():
    raw = {
        "benchmarks": [
            {
                "concurrency": 2,
                "context_size": 4096,
                "prompt_size": 2048,
                "response_size": 128,
                "tg_throughput": {"mean": 50, "std": 1, "values": [49, 51]},
                "tg_req_throughput": {"mean": 25, "std": 1, "values": [20, 25]},
                "e2e_ttft": {"mean": 900, "std": 5, "values": [895, 905]},
            }
        ]
    }
    c = parse_benchy_json(raw)[0]
    assert c["ttft_ms"]["mean"] == 900
    assert c["itl_ms"]["values"] == [50.0, 40.0]  # per request: 1000 / tg_req
    assert e2e.value(c, "itl_ms") == 45.0


def test_native_backends_on_the_mjolnir_image_never_count(tmp_path):
    """FlashInfer measured on the Mjolnir image is not native vLLM: it must
    neither stand in for the official side nor for Mjolnir."""
    h = tmp_path / "history.jsonl"
    rows = [
        _rec("vllm/vllm-openai:v0.30.0-ubuntu2404", "NVFP4", 1, [_cell(1, 8192, 20, 2000, 4000)]),
        _rec("narandill/mjolnir-vllm:v14", "NVFP4", 2, [_cell(1, 8192, 25, 2900, 3500)]),
    ]
    h.write_text("".join(json.dumps(r) + "\n" for r in rows))
    off, mj = e2e.latest_pair(h, "NVFP4_MJOLNIR")
    assert off["image"].startswith("vllm/") and mj is None


def test_bench_e2e_only_runs_one_leg(monkeypatch):
    from mjolnir_cli import cli, dockerctl
    from mjolnir_cli import legs
    from typer.testing import CliRunner

    served = []
    monkeypatch.setattr(dockerctl, "serve_up", lambda s, **kw: served.append((s.image, s.quant)))
    runner = CliRunner()
    mj, off = legs.side_configs("bench.e2e", legs.MJOLNIR), legs.side_configs("bench.e2e", legs.OFFICIAL)
    for side, want in (("mjolnir", mj), ("official", off[:1])):
        served.clear()
        r = runner.invoke(cli.app, ["bench", "e2e", "--dry-run", "--only", side])
        assert r.exit_code == 0, r.output
        assert [q for _, q in served] == want
    served.clear()
    assert runner.invoke(cli.app, ["bench", "e2e", "--dry-run"]).exit_code == 0
    assert [q for _, q in served] == [off[0], *mj]  # the pair, then --also
    served.clear()
    assert runner.invoke(cli.app, ["bench", "e2e", "--dry-run", "--also", ""]).exit_code == 0
    assert len(served) == 2
    assert runner.invoke(cli.app, ["bench", "e2e", "--dry-run", "--only", "fi"]).exit_code == 2


def test_spec_label_from_records_and_configs():
    mtp3 = {"spec": {"method": "mtp", "tokens": 3}}
    assert e2e.spec_label(mtp3, dict(mtp3)) == "MTP, 3 speculative tokens"
    assert e2e.spec_label(mtp3, {"spec": {}}) == "MTP, 3 speculative tokens vs no speculative decoding"
    # older records carry no "spec": it is read from the config they name
    old = {"model": "Qwen/Qwen3.8-27B", "config": "NVFP4_MJOLNIR"}
    assert e2e.spec_label(old) == "MTP, 3 speculative tokens"


def test_e2e_cells_are_colored_by_the_band():
    from mjolnir_cli.readme.fmt import ratio_pct

    assert e2e.E2E_NOISE_PCT == 3.0
    assert "#16a34a" in ratio_pct(3.5) and "+3.5" in ratio_pct(3.5)
    assert "grey" in ratio_pct(2.9) and "grey" in ratio_pct(-2.9)
    assert "#dc2626" in ratio_pct(-3.1) and "−3.1" in ratio_pct(-3.1)
    assert not any(ch in ratio_pct(10) for ch in "✅🟰🔻")


def test_each_leg_stops_its_server(monkeypatch):
    from mjolnir_cli import benchy, dockerctl
    from mjolnir_cli.commands import _common, bench

    calls = []
    monkeypatch.setattr(dockerctl, "serve_down", lambda s, **kw: calls.append(("down", s.quant)))
    monkeypatch.setattr(dockerctl, "serve_up", lambda s, **kw: calls.append(("up", s.quant)))
    monkeypatch.setattr(benchy, "run_perf", lambda s, layout, **kw: calls.append(("bench", s.quant)))
    s = _common.settings(None, None, None, None)
    bench.run_legs(s, None, [(s.model, "a", "NVFP4", None), (s.model, "b", "NVFP4_MJOLNIR", None)])
    assert calls == [
        ("down", "NVFP4"),
        ("up", "NVFP4"),
        ("bench", "NVFP4"),
        ("down", "NVFP4"),
        ("down", "NVFP4_MJOLNIR"),
        ("up", "NVFP4_MJOLNIR"),
        ("bench", "NVFP4_MJOLNIR"),
        ("down", "NVFP4_MJOLNIR"),
    ]

    def boom(s, layout, **kw):
        raise benchy.BenchError("llama-benchy failed")

    calls.clear()
    monkeypatch.setattr(benchy, "run_perf", boom)
    import pytest
    import typer

    with pytest.raises(typer.Exit):
        bench.run_legs(s, None, [(s.model, "a", "NVFP4", None)])
    assert calls[-1] == ("down", "NVFP4")  # stopped even when the bench fails


def test_runs_merge_cell_by_cell(tmp_path):
    """A later run at extra depths adds rows; the newest run wins shared cells."""
    h = tmp_path / "history.jsonl"
    off = "vllm/vllm-openai:v0.31.0-ubuntu2404"
    mjim = "narandill/mjolnir-vllm:v0.31.0"
    rows = [
        _rec(off, "NVFP4", 1, [_cell(1, 8192, 20, 2000, 4000), _cell(1, 32768, 18, 1700, 9000)]),
        _rec(mjim, "NVFP4_MJOLNIR", 2, [_cell(1, 8192, 22, 4000, 3000), _cell(1, 32768, 19, 3400, 5000)]),
        _rec(off, "NVFP4", 3, [_cell(1, 131072, 12, 900, 90000)]),
        _rec(mjim, "NVFP4_MJOLNIR", 4, [_cell(1, 131072, 14, 1800, 50000), _cell(1, 8192, 23, 4100, 2900)]),
    ]
    h.write_text("".join(json.dumps(r) + "\n" for r in rows))
    o, m = e2e.latest_pair(h, "NVFP4_MJOLNIR")
    assert sorted(c["context"] for c in o["cells"]) == [8192, 32768, 131072]
    assert e2e.cell(m, 1, 8192)["tg_tps"]["mean"] == 23  # newest run wins
    assert e2e.cell(m, 1, 32768)["tg_tps"]["mean"] == 19  # kept from the older run
    # the hero: 1 request, up to MAX_CTX, the cell with the highest decode gain (128K is past the cap)
    assert e2e.hero(o, m)["ctx"] == 8192
    assert e2e.hero(o, m, max_ctx=None)["ctx"] in (8192, 131072)


def test_cells_only_one_side_measured_are_kept_and_flagged():
    def rec(ctxs):
        return {"cells": [{"concurrency": 1, "context": c, "tg_tps": {"mean": 20.0}} for c in ctxs]}

    off, mj = rec([0, 8192, 131072]), rec([0, 8192])
    assert [r["ctx"] for r in e2e.compare(off, mj)] == [0, 8192]  # the comparison: shared cells only
    full = e2e.compare(off, mj, every=True)
    assert [r["ctx"] for r in full] == [0, 8192]  # 128K is past MAX_CTX: not in the table
    assert e2e.missing_contexts(off, mj) == {"official": [], "mjolnir": []}
    full = e2e.compare(off, mj, every=True, max_ctx=None)
    assert [r["ctx"] for r in full] == [0, 8192, 131072] and full[-1]["tg_tps"][1] is None
    assert e2e.missing_contexts(off, mj, max_ctx=None) == {"official": [], "mjolnir": [131072]}


def test_history_survives_a_missing_final_newline(tmp_path):
    from mjolnir_cli import history

    h = tmp_path / "history.jsonl"
    h.write_text(json.dumps({"a": 1}))  # hand-edited: no final newline
    history.append_record(h, {"b": 2})
    assert h.read_text().count("\n") == 2
    h.write_text(json.dumps({"a": 1}) + json.dumps({"b": 2}) + "\n" + '{"c": ')  # glued + truncated
    assert history.load_records(h) == [{"a": 1}, {"b": 2}]


def test_no_context_has_no_prefill():
    from mjolnir_cli import e2e

    assert e2e.value({"context": 0, "pp_tps": {"mean": 2929.0}}, "pp_tps") is None
    assert e2e.value({"context": 4096, "pp_tps": {"mean": 2929.0}}, "pp_tps") == 2929.0
    assert e2e.value({"context": 0, "tg_tps": {"mean": 22.5}}, "tg_tps") == 22.5


def test_optional_legs_run_only_when_asked(monkeypatch):
    from mjolnir_cli import cli, dockerctl
    from mjolnir_cli import legs
    from mjolnir_cli.config import STOCK_IMAGE
    from typer.testing import CliRunner

    served = []
    monkeypatch.setattr(dockerctl, "serve_up", lambda s, **kw: served.append((s.image, s.quant)))
    runner = CliRunner()
    assert runner.invoke(cli.app, ["bench", "e2e", "--dry-run", "--also", ""]).exit_code == 0
    assert len(served) == 2  # no optional leg by default
    served.clear()
    r = runner.invoke(cli.app, ["bench", "e2e", "--dry-run", "--also", "", "--optional"])
    assert r.exit_code == 0, r.output
    opt = legs.OPTIONAL["bench.e2e"]
    assert [q for _, q in served[2:]] == [cfg for _, cfg in opt]
    for (img, _), (side, _) in zip(served[2:], opt):
        assert (img == STOCK_IMAGE) == (side == legs.OFFICIAL)  # official legs stay on the upstream image
    served.clear()
    r = runner.invoke(cli.app, ["bench", "e2e", "--dry-run", "--only", "official", "--optional", "--also", ""])
    assert [q for _, q in served] == ["NVFP4", "NVFP4_DSPARK", "NVFP4_DFLASH2"]
    served.clear()
    r = runner.invoke(cli.app, ["bench", "e2e", "--dry-run", "--also", "official:NVFP4_DSPARK"])
    assert served[-1] == (STOCK_IMAGE, "NVFP4_DSPARK")


def test_optional_configs_exist_and_carry_their_drafter():
    from mjolnir_cli import legs
    from mjolnir_cli.config import config_speculative, configs_root

    want = {"DSPARK": "dspark", "DFLASH2": "dflash"}
    for _, cfg in legs.OPTIONAL["bench.e2e"]:
        path = configs_root() / "Qwen" / "Qwen3.8-27B" / cfg / "config.yaml"
        assert path.exists(), cfg
        assert config_speculative(path) == {"method": want[cfg.rsplit("_", 1)[-1]], "tokens": 7}
        assert not (path.parent / ".env").exists()  # DFlash2 shares the target's lm_head: no LMH8 / VT knobs
    assert legs.parse_leg("official:NVFP4_DSPARK") == ("official", "NVFP4_DSPARK")
    assert legs.parse_leg("NVFP4_MJOLNIR_E2M3_DSPARK") == ("mjolnir", "NVFP4_MJOLNIR_E2M3_DSPARK")


def test_a_newer_optional_official_leg_never_replaces_the_baseline(tmp_path):
    h = tmp_path / "history.jsonl"
    rows = [
        _rec("vllm/vllm-openai:v0.31.0-ubuntu2404", "NVFP4", 1, [_cell(1, 8192, 20, 2000, 4000)]),
        _rec("vllm/vllm-openai:v0.31.0-ubuntu2404", "NVFP4_DSPARK", 5, [_cell(1, 8192, 35, 2000, 4000)]),
        _rec("narandill/mjolnir-vllm:v0.31.0", "NVFP4_MJOLNIR", 4, [_cell(1, 8192, 30, 2600, 3200)]),
    ]
    h.write_text("".join(json.dumps(r) + "\n" for r in rows))
    off, _ = e2e.latest_pair(h, "NVFP4_MJOLNIR")
    assert off["config"] == "NVFP4"
    off, _ = e2e.latest_pair(h, "NVFP4_MJOLNIR", "NVFP4_DSPARK")
    assert off["config"] == "NVFP4_DSPARK"


def test_pair_and_site_rows_never_mix_models(tmp_path):
    """Another model's run under a Mjolnir config name (Qwen3.6's NVFP4_MJOLNIR_E2M3)
    never lands next to the default model's runs: not in the README pair, not
    on the interactive site."""
    from types import SimpleNamespace

    from mjolnir_cli.charts.e2e import e2e_rows

    h = tmp_path / "history.jsonl"
    q36 = {"model": "nvidia/Qwen3.6-35B-A3B"}
    rows = [
        _rec("vllm/vllm-openai:v0.31.0-ubuntu2404", "NVFP4", 1, [_cell(1, 0, 20, 0, 100)]),
        _rec("narandill/mjolnir-vllm:v0.31.0", "NVFP4_MJOLNIR_E2M3", 2, [_cell(1, 0, 30, 0, 90)]),
        {**_rec("vllm/vllm-openai:v0.31.0-ubuntu2404", "NVFP4", 3, [_cell(1, 0, 80, 0, 50)]), **q36},
        {**_rec("narandill/mjolnir-vllm:v0.31.0", "NVFP4_MJOLNIR_E2M3", 4, [_cell(1, 0, 130, 0, 40)]), **q36},
    ]
    h.write_text("".join(json.dumps(r) + "\n" for r in rows))
    off, mj = e2e.latest_pair(h, "NVFP4_MJOLNIR_E2M3")
    assert off["epoch"] == 1 and mj["epoch"] == 2  # the default model's runs only
    off, mj = e2e.latest_pair(h, "NVFP4_MJOLNIR_E2M3", model="nvidia/Qwen3.6-35B-A3B")
    assert off["epoch"] == 3 and mj["epoch"] == 4
    site = e2e_rows(SimpleNamespace(history_file=h))
    assert {k: v["epoch"] for k, v in site.items()} == {"NVFP4": 1, "NVFP4_MJOLNIR_E2M3": 2}


def test_compare_pairs_the_requested_models_runs(tmp_path, monkeypatch, capsys):
    """`bench compare --model` (and `bench e2e`, which passes its own model) show
    that model's pair, not the default model's."""
    from mjolnir_cli.commands import bench_report

    h = tmp_path / "history.jsonl"
    q36 = {"model": "nvidia/Qwen3.6-35B-A3B"}
    rows = [
        _rec("vllm/vllm-openai:v0.31.0-ubuntu2404", "NVFP4", 1, [_cell(1, 0, 20, 0, 100)]),
        _rec("narandill/mjolnir-vllm:v0.31.0", "NVFP4_MJOLNIR_E2M3", 2, [_cell(1, 0, 30, 0, 90)]),
        {**_rec("vllm/vllm-openai:v0.31.0-ubuntu2404", "NVFP4_DFLASH", 3, [_cell(1, 0, 80, 0, 50)]), **q36},
        {**_rec("narandill/mjolnir-vllm:v0.31.0", "NVFP4_MJOLNIR_E2M3", 4, [_cell(1, 0, 130, 0, 40)]), **q36},
    ]
    h.write_text("".join(json.dumps(r) + "\n" for r in rows))
    layout = type("L", (), {"history_file": h})()
    assert bench_report.print_compare(layout, "NVFP4_MJOLNIR_E2M3", True, "NVFP4_DFLASH", q36["model"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["cells"][0]["tg_tps"][:2] == [80.0, 130.0]  # Qwen3.6's pair, not Qwen3.8's 20 vs 30
    assert "NVFP4_DFLASH" in out["official"]


def test_a_run_without_model_is_never_the_default_models(tmp_path):
    """No default model: a history record without `model` pairs with nothing."""
    h = tmp_path / "history.jsonl"
    rows = [
        _rec("vllm/vllm-openai:v0.31.0-ubuntu2404", "NVFP4", 1, [_cell(1, 0, 20, 0, 100)]),
        _rec("narandill/mjolnir-vllm:v0.31.0", "NVFP4_MJOLNIR_E2M3", 2, [_cell(1, 0, 30, 0, 90)], model=None),
    ]
    h.write_text("".join(json.dumps(r) + "\n" for r in rows))
    off, mj = e2e.latest_pair(h, "NVFP4_MJOLNIR_E2M3")
    assert off["epoch"] == 1 and mj is None


def test_compare_defaults_to_the_models_own_mjolnir_config(tmp_path):
    """Without --config, a model that has no DEFAULT_QUANT runs pairs its newest
    Mjolnir-image config instead of reporting no Mjolnir run."""
    h = tmp_path / "history.jsonl"
    q36 = "nvidia/Qwen3.6-35B-A3B"
    rows = [
        _rec("vllm/vllm-openai:v0.31.0-ubuntu2404", "NVFP4_DFLASH", 1, [_cell(1, 0, 80, 0, 50)], model=q36),
        _rec("narandill/mjolnir-vllm:v0.31.0", "NVFP4_MJOLNIR_E2M3", 2, [_cell(1, 0, 130, 0, 40)], model=q36),
    ]
    h.write_text("".join(json.dumps(r) + "\n" for r in rows))
    off, mj = e2e.latest_pair(h, model=q36)
    assert off["config"] == "NVFP4_DFLASH" and mj["config"] == "NVFP4_MJOLNIR_E2M3"
