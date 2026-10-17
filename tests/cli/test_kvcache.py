"""verify kv-cache-size: log parsing, capacity math, docs table, chart; and
the speculative-decoding counters the e2e bench records."""

from __future__ import annotations

from mjolnir_cli import gate
from mjolnir_cli.verify import kvcache

LOG = """
(EngineCore pid=123) INFO 10-11 21:51:30 [gpu_worker.py:420] Available KV cache memory: 38.12 GiB
(EngineCore pid=123) INFO 10-11 21:51:34 [kv_cache_utils.py:2468] GPU KV cache size: 2,656,033 tokens
(EngineCore pid=123) INFO 10-11 21:51:34 [kv_cache_utils.py:2472] GPU KV cache size: 2,656,033 tokens, Maximum concurrency for 204,800 tokens per request: 12.97x
"""


def _doc(rows):
    return {
        "stage": "kv-cache-size",
        "run_id": "20990101-000000",
        "params": {"e2e_pp": 2048, "e2e_tg": 128},
        "kernel": {},
        "rows": rows,
    }


def _row(side, cfg, kv, tokens):
    return {
        "id": f"{side}-{cfg}",
        "side": side,
        "config": cfg,
        "backend": "flashinfer" if side == "official" else "MJOLNIR",
        "kv_dtype": kv,
        "kv_gib": 38.1,
        "kv_tokens": tokens,
        "max_model_len": 204800,
        "max_concurrency": tokens / 204800,
        "model": "Qwen/Qwen3.8-27B",
    }


def test_parse_log():
    r = kvcache.parse_log(LOG)
    assert {k: r[k] for k in ("kv_gib", "kv_tokens", "max_model_len", "max_concurrency")} == {
        "kv_gib": 38.12,
        "kv_tokens": 2656033,
        "max_model_len": 204800,
        "max_concurrency": 12.97,
    }
    mem = kvcache.parse_log(
        "Model loading took 21.30 GiB memory and 30.1 seconds\n"
        "Memory profiling takes 9.1 seconds ... model weights take 21.30GiB; non_torch_memory takes 0.42GiB; "
        "PyTorch activation peak memory takes 3.25GiB; the rest of the memory reserved for KV Cache is 38.12GiB."
    )
    assert (mem["load_gib"], mem["weights_gib"], mem["activation_gib"], mem["non_torch_gib"]) == (
        21.3,
        21.3,
        3.25,
        0.42,
    )
    assert kvcache.parse_log("nothing here")["kv_tokens"] is None


def test_fits_and_legs():
    assert kvcache.fits(2656033, 32768 + 2048 + 128) == 76
    assert kvcache._legs(kvcache.DEFAULT_LEGS)[0] == ("official", "NVFP4")
    try:
        kvcache._legs("nope")
    except ValueError:
        pass
    else:
        raise AssertionError("a leg without side:config must be rejected")


def test_docs_table_and_findings():
    from mjolnir_cli.readme.findings import findings
    from mjolnir_cli.readme.tables import TABLES

    doc = _doc(
        [
            _row("official", "NVFP4", "fp8", 2000000),
            _row("mjolnir", "NVFP4_MJOLNIR_E2M3", "fp6_e2m3", 2656033),
        ]
    )
    table = TABLES["kv-cache-size"](doc)
    assert "2,656,033" in table and "1.33×" in table and "| 76 |" in table
    found = findings("kv-cache-size", doc)
    assert any("+33%" in f for f in found)


def test_chart_renders():
    from mjolnir_cli import theme
    from mjolnir_cli.charts.kernel import chart_kvcache

    doc = _doc([_row("official", "NVFP4", "fp8", 2000000), _row("mjolnir", "NVFP4_MJOLNIR", "fp8", 1990000)])
    fig = chart_kvcache({"kv-cache-size": doc}, theme.LIGHT)
    assert fig is not None
    assert chart_kvcache({}, theme.LIGHT) is None


def test_spec_delta():
    before = {"drafts": 100, "draft_tokens": 300, "accepted": 150, "preemptions": 1, "per_pos": {0: 80, 1: 50, 2: 20}}
    after = {"drafts": 200, "draft_tokens": 600, "accepted": 330, "preemptions": 1, "per_pos": {0: 170, 1: 110, 2: 50}}
    d = gate.spec_delta(before, after)
    assert d["acceptance_length"] == 2.8 and d["acceptance_rate"] == 0.6
    assert d["per_pos"] == [0.9, 0.6, 0.3] and d["preemptions"] == 0
    assert gate.spec_delta(None, after) is None
