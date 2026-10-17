"""``mjolnir bench accuracy``: tool-eval-bench's SQLite -> accuracy.json ->
docs table + charts, and the container cleanup helpers it relies on."""

from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from mjolnir_cli import accuracy, charts, dockerctl, readme
from mjolnir_cli.config import load_layout

REPO = Path(__file__).resolve().parents[2]
OFFICIAL = "vllm/vllm-openai:v0.31.0-ubuntu2404"
MJ = "narandill/mjolnir-vllm:v0.31.0"


def _db(path: Path, runs: list[tuple[str, dict]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.execute(
        "CREATE TABLE scenario_runs (run_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, status TEXT NOT NULL, "
        "model TEXT NOT NULL, config_json TEXT NOT NULL, scores_json TEXT, metadata_json TEXT, "
        "run_type TEXT NOT NULL DEFAULT 'tool_eval', report_path TEXT)"
    )
    for i, (run_type, scores) in enumerate(runs):
        con.execute(
            "INSERT INTO scenario_runs VALUES (?, ?, 'completed', 'm', '{}', ?, '{}', ?, NULL)",
            (f"r{i}", f"2026-10-12T10:0{i}:00", json.dumps(scores), run_type),
        )
    con.commit()
    con.close()
    return path


def _tool_scores(a_earned: int, p_earned: int) -> dict:
    return {
        "final_score": 80,
        "category_scores": [
            {"category": "A", "label": "Tool selection", "earned": a_earned, "max": 10, "pass_count": 4},
            {"category": "B", "label": "Parameters", "earned": 8, "max": 10, "pass_count": 4},
            {"category": "P", "label": "Hard Mode", "earned": p_earned, "max": 8, "pass_count": 2},
        ],
        "scenario_results": [
            {"scenario_id": "TC-01", "points": 2},
            {"scenario_id": "TC-02", "points": a_earned // 5},
        ],
    }


def test_summarize_db_splits_standard_and_hard_mode(tmp_path):
    db = _db(
        tmp_path / "data" / "benchmarks.sqlite",
        [("tool_eval", _tool_scores(10, 4)), ("gsm8k", {"accuracy": 91.5, "correct": 183, "total": 200})],
    )
    s = accuracy.summarize_db(db)
    assert s["suites"]["tools"]["score"] == 90.0  # (10 + 8) / 20
    assert s["suites"]["hardmode"]["score"] == 50.0
    assert s["suites"]["gsm8k"]["score"] == 91.5 and s["suites"]["gsm8k"]["total"] == 200
    assert s["categories"]["A"]["percent"] == 100.0 and s["scenarios"]["TC-02"] == 2.0


def test_summarize_db_averages_trials(tmp_path):
    db = _db(tmp_path / "b.sqlite", [("tool_eval", _tool_scores(10, 4)), ("tool_eval", _tool_scores(6, 8))])
    t = accuracy.summarize_db(db)["suites"]
    assert t["tools"]["n"] == 2 and t["tools"]["scores"] == [90.0, 70.0] and t["tools"]["score"] == 80.0
    assert accuracy.summarize_db(tmp_path / "missing.sqlite")["suites"] == {}


def test_tool_args_default_and_passthrough():
    assert accuracy.tool_args([], "http://h:8000/v1") == [*accuracy.DEFAULT_ARGS, "--base-url", "http://h:8000/v1"]
    fwd = accuracy.tool_args(["--gsm8k-only", "--gsm8k-limit", "50"], "http://h/v1")
    assert fwd[:3] == ["--gsm8k-only", "--gsm8k-limit", "50"] and fwd[-2:] == ["--base-url", "http://h/v1"]
    assert accuracy.tool_args(["run", "--base-url=http://x"], "http://h/v1") == ["run", "--base-url=http://x"]


def _leg(image: str, config: str, tools: float, hard: float, cats: dict, extra: dict | None = None) -> dict:
    rec = {"id": config, "model": "Qwen/Qwen3.8-27B", "image": image, "config": config}
    rec["label"] = accuracy.leg_label(rec)
    rec["suites"] = {"tools": {"score": tools, "n": 1, "completion_rate": 100.0}, "hardmode": {"score": hard, "n": 1}}
    rec["suites"].update(extra or {})
    rec["categories"] = {c: {"label": f"cat {c}", "percent": v} for c, v in cats.items()}
    rec["scenarios"] = {"TC-01": 2.0, "TC-02": 1.0 if config != "NVFP4" else 2.0}
    return rec


def _write_runs(raw: Path) -> None:
    first = {
        "format": accuracy.FORMAT,
        "run_id": "20261012-100000",
        "args": list(accuracy.DEFAULT_ARGS),
        "legs": [
            _leg(OFFICIAL, "NVFP4", 88.0, 60.0, {"A": 90, "P": 60}),
            _leg(MJ, "NVFP4_MJOLNIR_E2M3", 86.5, 62.0, {"A": 85, "P": 62}),
        ],
    }
    later = {
        "format": accuracy.FORMAT,
        "run_id": "20261013-100000",
        "args": ["--gsm8k-only"],
        "legs": [
            {
                **_leg(MJ, "NVFP4_MJOLNIR_E2M3", 0, 0, {}),
                "suites": {"gsm8k": {"score": 90.0, "n": 1}},
                "categories": {},
            }
        ],
    }
    for d in (first, later):
        p = raw / "accuracy" / d["run_id"] / "accuracy.json"
        p.parent.mkdir(parents=True)
        p.write_text(json.dumps(d))


def test_latest_legs_merge_suites_across_runs(tmp_path):
    _write_runs(tmp_path)
    rows = accuracy.latest_legs(accuracy.load_runs(SimpleNamespace(raw_dir=tmp_path)))
    assert [r["config"] for r in rows] == ["NVFP4", "NVFP4_MJOLNIR_E2M3"]
    mj = rows[1]
    assert set(mj["suites"]) == {"tools", "hardmode", "gsm8k"}  # the GSM8K run joins the A/B row
    assert mj["categories"]["A"]["percent"] == 85  # categories stay from the tool-call run
    assert accuracy.baseline(rows)["config"] == "NVFP4"
    assert accuracy.suites_in(rows) == ["tools", "hardmode", "gsm8k"]
    assert rows[0]["label"].startswith("official vLLM")


def test_baseline_is_per_model(tmp_path):
    """A second model's run never borrows the first model's baseline: each row is
    compared with its own model's official-image row, and labels name the model."""
    _write_runs(tmp_path)
    other = {
        "format": accuracy.FORMAT,
        "run_id": "20261017-100000",
        "args": list(accuracy.DEFAULT_ARGS),
        "legs": [
            {**_leg(OFFICIAL, "NVFP4_DFLASH", 91.3, 76.8, {"A": 100}), "model": "nvidia/Qwen3.6-35B-A3B"},
            {**_leg(MJ, "NVFP4_MJOLNIR_E2M3", 90.4, 80.4, {"A": 100}), "model": "nvidia/Qwen3.6-35B-A3B"},
        ],
    }
    p = tmp_path / "accuracy" / other["run_id"] / "accuracy.json"
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps(other))
    rows = accuracy.latest_legs(accuracy.load_runs(SimpleNamespace(raw_dir=tmp_path)))
    by = {(r["model"], r["config"]): r for r in rows}
    q36_mj = by[("nvidia/Qwen3.6-35B-A3B", "NVFP4_MJOLNIR_E2M3")]
    base = dict(zip(map(id, rows), accuracy.baselines(rows)))
    assert base[id(q36_mj)]["config"] == "NVFP4_DFLASH"  # no NVFP4 row: first official one
    assert base[id(by[("Qwen/Qwen3.8-27B", "NVFP4_MJOLNIR_E2M3")])]["config"] == "NVFP4"
    assert q36_mj["label"].startswith("Qwen3.6-35B-A3B · Mjolnir")


def test_print_accuracy_compares_within_model(capsys):
    """The end-of-run table: a run with legs of two models never prints a delta
    across models."""
    from mjolnir_cli import cli

    q36 = {"model": "nvidia/Qwen3.6-35B-A3B"}
    doc = {
        "run_id": "r",
        "args": ["run"],
        "legs": [
            _leg(OFFICIAL, "NVFP4", 97.0, 82.0, {}),
            {**_leg(OFFICIAL, "NVFP4_DFLASH", 91.0, 77.0, {}), **q36, "label": "q36 official"},
            {**_leg(MJ, "NVFP4_MJOLNIR_E2M3", 90.0, 80.0, {}), **q36, "label": "q36 mjolnir"},
        ],
    }
    cli._print_accuracy(doc)
    out = capsys.readouterr().out
    assert "(-1.0)" in out and "(+3.0)" in out  # vs the Qwen3.6 official leg
    assert "(-7.0)" not in out  # never vs Qwen3.8's NVFP4


def _write_two_model_run(raw: Path) -> None:
    """One run whose legs interleave two models (Qwen3.8, Qwen3.6, Qwen3.8, Qwen3.6)."""
    q36 = {"model": "nvidia/Qwen3.6-35B-A3B"}
    doc = {
        "format": accuracy.FORMAT,
        "run_id": "20261017-100000",
        "args": list(accuracy.DEFAULT_ARGS),
        "legs": [
            _leg(OFFICIAL, "NVFP4", 88.0, 60.0, {"A": 90}),
            {**_leg(OFFICIAL, "NVFP4_DFLASH", 91.3, 76.8, {"A": 95}), **q36},
            _leg(MJ, "NVFP4_MJOLNIR_E2M3", 86.5, 62.0, {"A": 85}),
            {**_leg(MJ, "NVFP4_MJOLNIR_E2M3", 90.4, 80.4, {"A": 94}), **q36},
        ],
    }
    p = raw / "accuracy" / doc["run_id"] / "accuracy.json"
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps(doc))


def test_rows_of_one_model_stay_together(tmp_path):
    """Interleaved legs come back grouped by model (first-seen order), so each
    model's dashed baseline mark spans only its own rows."""
    _write_two_model_run(tmp_path)
    rows = accuracy.latest_legs(accuracy.load_runs(SimpleNamespace(raw_dir=tmp_path)))
    assert [r["model"] for r in rows] == ["Qwen/Qwen3.8-27B"] * 2 + ["nvidia/Qwen3.6-35B-A3B"] * 2
    assert [b["config"] for b in accuracy.baselines(rows)] == ["NVFP4", "NVFP4", "NVFP4_DFLASH", "NVFP4_DFLASH"]


@pytest.fixture
def repo(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    (root / "docker" / "vllm-thor").mkdir(parents=True)
    shutil.copy(REPO / "docker" / "vllm-thor" / "Dockerfile", root / "docker" / "vllm-thor" / "Dockerfile")
    shutil.copy(REPO / "README.md.j2", root / "README.md.j2")
    shutil.copytree(REPO / "docs" / "kernel", root / "docs" / "kernel", ignore=shutil.ignore_patterns("*.md"))
    for pkg in ("mjolnir/mjolnir", "mjolnir-cli/mjolnir_cli"):
        (root / "src" / pkg).mkdir(parents=True)
        shutil.copy(REPO / "src" / pkg / "__init__.py", root / "src" / pkg / "__init__.py")
    (root / "benchmarks").mkdir()
    (root / "benchmarks" / "history.jsonl").write_text("")
    monkeypatch.setenv("MJOLNIR_REPO", str(root))
    for v in ("MJOLNIR_BENCHMARKS", "MJOLNIR_HISTORY"):
        monkeypatch.delenv(v, raising=False)
    return root


def test_accuracy_page_and_charts_render(repo):
    pytest.importorskip("matplotlib")
    layout = load_layout()
    _write_runs(layout.raw_dir)
    manifest = charts.render_all(layout)
    for name in ("accuracy-suites", "accuracy-categories"):
        assert name in manifest and (repo / manifest[name]["dark"]).exists()
    readme.render_readme(layout)
    page = (repo / "docs" / "kernel" / "accuracy.md").read_text()
    assert "{{" not in page and "{%" not in page
    assert "| **official vLLM · FlashInfer · bf16 KV** |" in page
    assert "86.5 (-1.5)" in page and "62.0 (+2.0)" in page and "90.0" in page
    assert "1 scenarios scored lower, 0 higher of 2" in page
    assert "accuracy-suites-dark.png" in page and "20261013-100000/accuracy.json" in page
    assert "[Accuracy](docs/kernel/accuracy.md)" in (repo / "README.md").read_text()


def test_two_model_page_compares_within_each_model(repo):
    pytest.importorskip("matplotlib")
    layout = load_layout()
    _write_two_model_run(layout.raw_dir)
    manifest = charts.render_all(layout)
    assert "accuracy-suites" in manifest and "accuracy-categories" in manifest
    readme.render_readme(layout)
    page = (repo / "docs" / "kernel" / "accuracy.md").read_text()
    assert "| **Qwen3.8-27B · official vLLM · FlashInfer · bf16 KV** |" in page
    assert "| **Qwen3.6-35B-A3B · official vLLM · FlashInfer + DFlash · " in page
    assert "86.5 (-1.5)" in page  # Qwen3.8 Mjolnir vs Qwen3.8 FlashInfer
    assert "90.4 (-0.9)" in page and "80.4 (+3.6)" in page  # Qwen3.6 Mjolnir vs Qwen3.6 DFlash
    assert "90.4 (+2.4)" not in page  # never vs the other model
    assert "never across models" in page


def _excluded_pair() -> tuple[dict, dict]:
    """The same-run Qwen3.6 pair: official 125/138, Mjolnir 123/136 (TC-45 excluded)."""
    off = _leg(OFFICIAL, "NVFP4_DFLASH", 90.58, 76.79, {"H": 100})
    mj = _leg(MJ, "NVFP4_MJOLNIR_E2M3", 90.44, 80.36, {"H": 100})
    off["scenarios"]["TC-45"] = 2.0  # the Mjolnir leg has no TC-45: excluded
    off["suites"]["tools"].update(earned=125, max=138)
    mj["suites"]["tools"].update(earned=123, max=136, completion_rate=99.0)
    off["suites"]["hardmode"].update(earned=43, max=56)
    mj["suites"]["hardmode"].update(earned=45, max=56)
    return off, mj


def test_an_excluded_scenario_never_raises_the_delta():
    """A leg whose endpoint had a scenario excluded scores on a smaller max; the
    delta counts the excluded points as zero (123/138 vs 125/138), and says so."""
    off, mj = _excluded_pair()
    d, flagged = accuracy.delta(mj, off, "tools")
    assert flagged and round(d, 2) == round((123 - 125) / 138 * 100, 2)  # -1.45, not -0.14
    assert accuracy.delta(mj, off, "hardmode") == pytest.approx((80.36 - 76.79, False))
    assert accuracy.delta(off, off, "tools") is None
    assert accuracy.fmt_delta((d, flagged)) == " (-1.4‡)"
    # the other way round: an excluded scenario on the baseline does not flatter it either
    d2, flagged2 = accuracy.delta(off, mj, "tools")
    assert flagged2 and round(d2, 2) == round((125 - 123) / 138 * 100, 2)


def test_different_scenario_sets_give_no_delta():
    """Different maxes at full completion are not an exclusion (another run's
    arguments, a subset): no rescaled delta, and no plain one either."""
    off, mj = _excluded_pair()
    mj["suites"]["tools"].update(earned=38, max=40, score=95.0, completion_rate=100.0)
    assert accuracy.delta(mj, off, "tools") is None


def test_averaged_trials_rescale_every_trial():
    """--trials: each trial's own points over the full max, so an exclusion in
    any trial counts, not just in the last one."""
    off, mj = _excluded_pair()
    # trials 1-2 excluded TC-45 (max 136), trial 3 did not (max 138)
    mj["suites"]["tools"].update(
        score=round((123 / 136 + 123 / 136 + 125 / 138) / 3 * 100, 2),
        n=3,
        earned=125,
        max=138,
        earneds=[123, 123, 125],
        maxes=[136, 136, 138],
        completion_rate=99.0,
    )
    d, flagged = accuracy.delta(mj, off, "tools")
    assert flagged and round(d, 3) == round(((123 + 123 + 125) / 3 - 125) / 138 * 100, 3)


def test_print_accuracy_marks_an_excluded_scenario(capsys):
    from mjolnir_cli import cli

    off, mj = _excluded_pair()
    cli._print_accuracy({"run_id": "r", "args": ["run"], "legs": [off, mj]})
    out = capsys.readouterr().out
    assert "(-1.4‡)" in out and "(-0.1)" not in out
    assert "excluded points counting as zero" in out


def test_page_marks_an_excluded_scenario(repo):
    layout = load_layout()
    off, mj = _excluded_pair()
    p = layout.raw_dir / "accuracy" / "20261017-131320" / "accuracy.json"
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps({"format": accuracy.FORMAT, "run_id": "20261017-131320", "args": [], "legs": [off, mj]}))
    readme.render_readme(layout)
    page = (repo / "docs" / "kernel" / "accuracy.md").read_text()
    assert "90.4 (-1.4‡)" in page and "80.4 (+3.6)" in page
    assert "Tool calls -1.4‡" in page
    assert "excluded points counting as zero" in page
    assert "excluded TC-45 counted as zero" in page  # the tally counts what the ‡ delta counts
    assert "Mjolnir · bf16 KV ‡ |" in page and "its category % divides by the smaller max" in page


def test_accuracy_page_pending_without_runs(repo):
    layout = load_layout()
    readme.render_readme(layout)
    assert "Pending — run `mjolnir bench accuracy`" in (repo / "docs" / "kernel" / "accuracy.md").read_text()


def test_infographic_multiple():
    from mjolnir_cli.charts.infographic import multiple

    assert multiple(200) == "3×" and multiple(150) == "2.5×" and multiple(100.4) == "2×"


def test_container_output_is_handed_to_the_host_user(monkeypatch):
    monkeypatch.setattr(dockerctl.os, "getuid", lambda: 1000)
    monkeypatch.setattr(dockerctl.os, "getgid", lambda: 1001)
    tail = dockerctl.owner_suffix(["/out/draft_vocab.json"])
    assert tail == "; rc=$?; chown -R 1000:1001 /out/draft_vocab.json 2>/dev/null; exit $rc"
    assert dockerctl.owner_suffix([]) == ""


def test_a_leg_without_model_never_borrows_another_models_baseline():
    rows = [
        _leg(OFFICIAL, "NVFP4", 88.0, 60.0, {}),
        {**_leg(OFFICIAL, "NVFP4_DFLASH", 91.0, 77.0, {}), "model": None},
        {**_leg(MJ, "NVFP4_MJOLNIR_E2M3", 90.0, 80.0, {}), "model": None},
    ]
    assert [b["config"] for b in accuracy.baselines(rows)] == ["NVFP4", "NVFP4_DFLASH", "NVFP4_DFLASH"]
    labels = [r["label"] for r in accuracy.latest_legs([{"run_id": "r", "legs": rows}])]
    assert sum(label.startswith("unknown model · ") for label in labels) == 2  # never shown as the default model
