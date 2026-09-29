"""The history substrate: append-only round-trip, llama-benchy raw parsing,
and record assembly (what the charts + the CI substrate gate consume)."""

from __future__ import annotations

from mjolnir.config import Settings
from mjolnir.history import append_record, load_records, make_record, parse_benchy_json


def test_append_then_load_roundtrip(tmp_path):
    f = tmp_path / "history.jsonl"
    append_record(f, {"a": 1})
    append_record(f, {"a": 2})
    assert load_records(f) == [{"a": 1}, {"a": 2}]


def test_load_missing_file_is_empty(tmp_path):
    assert load_records(tmp_path / "nope.jsonl") == []


def test_parse_benchy_json_maps_cells():
    raw = {
        "benchmarks": [
            {
                "concurrency": 1,
                "context_size": 8192,
                "prompt_size": 2048,
                "response_size": 128,
                "is_context_prefill_phase": False,
                "tg_throughput": {"mean": 41.2, "std": 0.7, "values": [40.0, 42.4]},
                "pp_throughput": {"mean": 9830.0, "std": 120.0, "values": [9800.0, 9860.0]},
            },
            {
                "concurrency": 4,
                "context_size": 0,
                "prompt_size": 2048,
                "response_size": 128,
                "is_context_prefill_phase": True,
                "tg_throughput": {"mean": 1.0, "std": 0.0},  # no 'values' -> omitted
            },
        ]
    }
    cells = parse_benchy_json(raw)
    assert len(cells) == 2
    c0, c1 = cells
    assert (c0["concurrency"], c0["context"], c0["prompt"], c0["gen"]) == (1, 8192, 2048, 128)
    assert c0["prefill_phase"] is False
    assert c0["tg_tps"] == {"mean": 41.2, "std": 0.7, "values": [40.0, 42.4]}
    assert c0["pp_tps"]["mean"] == 9830.0
    assert c1["prefill_phase"] is True
    assert c1["tg_tps"] == {"mean": 1.0, "std": 0.0}  # no 'values' key at all
    assert "pp_tps" not in c1  # absent source metric -> absent cell field


def test_parse_benchy_json_empty():
    assert parse_benchy_json({"benchmarks": []}) == []
    assert parse_benchy_json({}) == []


def test_make_record_maps_settings():
    s = Settings(image="img:v1", model="Qwen/Qwen3.8-27B", quant="NVFP4")
    rec = make_record(
        s, gate_summary={"clean": True, "confirm": 3}, runs=5, warmup_runs=2,
        exact_tg=True, cells=[{"concurrency": 1, "context": 0, "prompt": 2048, "gen": 128}],
        raw_relpath="raw/perf-x/",
    )
    assert rec["image"] == "img:v1"
    assert rec["model"] == "Qwen/Qwen3.8-27B"
    assert rec["config"] == "NVFP4"
    assert rec["backend"] == "FlashInfer"  # backend_label via BACKEND_LABELS
    assert rec["gate"] == {"clean": True, "confirm": 3}
    assert rec["runs"] == 5 and rec["warmup_runs"] == 2 and rec["exact_tg"] is True
    assert rec["raw"] == "raw/perf-x/"
    assert "ts" in rec and "epoch" in rec and "host" in rec
