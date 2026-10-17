"""The autoresearch ledger + the fixed keep/discard rule (the part of the loop
the agents cannot influence)."""

from __future__ import annotations

import json
import subprocess

from mjolnir_cli import autoresearch as ar


def test_decide_is_noise_aware():
    assert ar.decide(None, True, 1.2, 0.02) == "crash"
    assert ar.decide(1.5, False, 1.2, 0.02) == "discard"  # wrong answers are never kept
    assert ar.decide(1.2, True, None, 0.02) == "baseline"
    assert ar.decide(1.23, True, 1.2, 0.02) == "keep"
    assert ar.decide(1.22, True, 1.2, 0.02) == "discard"  # +1.7%: inside the required gain


def test_ledger_roundtrip_and_best(tmp_path):
    rows = [
        {"id": 1, "ts": "2026-10-10T10:00:00+00:00", "commit": "a", "status": "baseline", "score": 1.20},
        {"id": 2, "ts": "2026-10-10T10:20:00+00:00", "commit": "b", "status": "discard", "score": 1.30},
        {"id": 3, "ts": "2026-10-10T10:40:00+00:00", "commit": "c", "status": "keep", "score": 1.25},
    ]
    for r in rows:
        ar.append_ledger(tmp_path, {**r, "hypothesis": "h\twith tab", "agent": "x"})
    back = ar.load_ledger(tmp_path)
    assert [r["id"] for r in back] == [1, 2, 3]
    assert ar.best_kept(back)["commit"] == "c"  # a discarded higher score never becomes "best"
    tsv = (tmp_path / "research" / "ledger.tsv").read_text().splitlines()
    assert tsv[0].split("\t") == ar.TSV_COLUMNS and len(tsv) == 4
    assert "\t".join(["h with tab"]) in tsv[1]


def test_export_writes_summary(tmp_path, monkeypatch):
    subprocess.run(["git", "init", "-q", "-b", "autoresearch/t1", str(tmp_path)], check=True)
    (tmp_path / "docker" / "vllm-thor").mkdir(parents=True)
    (tmp_path / "docker" / "vllm-thor" / "Dockerfile").write_text("#")
    subprocess.run(["git", "-C", str(tmp_path), "add", "-A"], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "-c", "user.email=a@b", "-c", "user.name=t", "commit", "-qm", "x"], check=True
    )
    monkeypatch.setenv("MJOLNIR_REPO", str(tmp_path))
    ar.append_ledger(
        tmp_path,
        {
            "id": 1,
            "ts": "t",
            "commit": "a",
            "status": "baseline",
            "score": 1.2,
            "hypothesis": "baseline",
            "agent": "human",
            "sources": [],
        },
    )
    ar.append_ledger(
        tmp_path,
        {
            "id": 2,
            "ts": "t",
            "commit": "b",
            "status": "keep",
            "score": 1.3,
            "hypothesis": "3-stage ring",
            "agent": "python-engineer",
            "sources": ["https://arxiv.org/abs/2407.08608"],
        },
    )
    ar.export(tag=None)
    out = tmp_path / "research" / "runs" / "t1"
    assert json.loads((out / "ledger.jsonl").read_text().splitlines()[1])["status"] == "keep"
    summary = (out / "SUMMARY.md").read_text()
    assert "best: **1.3×**" in summary and "3-stage ring" in summary


LS = ["128", "4096", "8192", "16384", "32768", "65536", "131072", "262144"]
BASE = {"status": "baseline", "score": 1.0, "per_L": dict.fromkeys(LS, 1.0), "fi_us": dict.fromkeys(LS, 100.0)}


def _row(mult: dict, fi: dict | None = None) -> dict:
    per_L = {L: mult.get(L, 1.0) for L in LS}
    return {
        "score": 1.0,
        "correctness": "pass",
        "per_L": per_L,
        "fi_us": {L: 100.0 * (fi or {}).get(L, 1.0) for L in LS},
    }


def test_judge_keeps_a_broad_win_and_tolerates_one_slow_L():
    st, why, rel = ar.judge(_row({L: 1.06 for L in LS} | {"16384": 0.95}), BASE)
    assert st == "keep", why
    assert rel["16384"] == 0.95


def test_judge_rejects_a_win_that_rests_on_one_L():
    st, why, _ = ar.judge(_row({"262144": 1.5}), BASE)  # +5% geomean, all of it at 256K
    assert st == "discard" and "rests on 256K" in why


def test_judge_rejects_short_L_regressions():
    st, why, _ = ar.judge(_row({"4096": 0.6, "8192": 0.7} | {L: 1.4 for L in LS[4:]}), BASE)
    assert st == "discard" and "slower at 4K" in why and "8K" in why


def test_judge_sets_aside_a_reference_outlier():
    # FlashInfer at 256K ran 1.7x slower in this run: Mjolnir "gains" 70% there
    st, why, _ = ar.judge(_row({"262144": 1.7}, fi={"262144": 1.7}), BASE)
    assert st == "discard" and "reference jumped: 256K" in why
    st, why, _ = ar.judge(_row({}, fi={L: 1.3 for L in LS[:4]}), BASE)
    assert st == "noisy"


def test_judge_crash_correctness_baseline():
    assert ar.judge({"score": None}, BASE)[0] == "crash"
    assert ar.judge({"score": 1.4, "correctness": "fail"}, BASE)[0] == "discard"
    assert ar.judge(_row({}), None)[0] == "baseline"


def test_progress_chart_renders(tmp_path):
    rows = [
        {"id": 1, **BASE, "hypothesis": "baseline", "branch": "autoresearch/t"},
        {"id": 2, "status": "crash", "score": None, "hypothesis": "x"},
        {"id": 3, **_row({"262144": 1.7}, fi={"262144": 1.7}), "status": "discard", "score": 1.07,
         "reason": "geomean +0.0% ≤ +2% (set aside, reference jumped: 256K)", "hypothesis": "y"},
        {"id": 4, **_row({L: 1.06 for L in LS}), "status": "keep", "score": 1.06, "hypothesis": "z"},
    ]  # fmt: skip
    p = ar.render_progress(tmp_path, rows, tmp_path / "progress.png")
    assert p and (tmp_path / "progress.png").exists() and (tmp_path / "progress-dark.png").exists()
    assert ar.render_progress(tmp_path, [], tmp_path / "empty.png")  # the placeholder
