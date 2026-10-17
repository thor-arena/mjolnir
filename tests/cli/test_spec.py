"""verify spec: config override, the streamed decode timing, the counters
around a block, best draft count, docs table / findings, chart."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from mjolnir_cli import gate
from mjolnir_cli.verify import spec

COUNTERS = {"drafts": 0}


class _Fake(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def do_GET(self):  # /metrics
        d = COUNTERS["drafts"]
        body = (
            f'vllm:spec_decode_num_drafts_total{{engine="0"}} {d}\n'
            f'vllm:spec_decode_num_draft_tokens_total{{engine="0"}} {3 * d}\n'
            f'vllm:spec_decode_num_accepted_tokens_total{{engine="0"}} {2 * d}\n'
            f'vllm:spec_decode_num_accepted_tokens_per_pos_total{{engine="0",position="0"}} {0.9 * d}\n'
            f'vllm:spec_decode_num_accepted_tokens_per_pos_total{{engine="0",position="1"}} {0.7 * d}\n'
            f'vllm:spec_decode_num_accepted_tokens_per_pos_total{{engine="0",position="2"}} {0.4 * d}\n'
        ).encode()
        self.send_response(200)
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):  # streamed chat completion: 3 chunks, 9 tokens
        self.rfile.read(int(self.headers["Content-Length"]))
        COUNTERS["drafts"] += 3
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for i in range(3):
            ev = {"choices": [{"delta": {"content": f"tok{i} "}}]}
            self.wfile.write(f"data: {json.dumps(ev)}\n\n".encode())
            self.wfile.flush()
            threading.Event().wait(0.02)
        self.wfile.write(f"data: {json.dumps({'choices': [], 'usage': {'completion_tokens': 9}})}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")


class _S:
    def __init__(self, port):
        self.base_url = f"http://127.0.0.1:{port}/v1"
        self.metrics_url = f"http://127.0.0.1:{port}/metrics"
        self.served_model = "m"


def test_block_against_a_fake_server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Fake)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        s = _S(srv.server_address[1])
        one = spec.stream_one(s.base_url, "m", "hi", 16)
        assert one["tokens"] == 9 and one["chunks"] == 3 and one["decode_tps"] > 0 and one["ttft_ms"] >= 0
        b = spec.run_block(s, ["a", "b", "c", "d"], conc=2, max_tokens=16)
        assert b["requests"] == 4 and b["tokens"] == 36 and not b["errors"]
        assert b["acceptance_length"] == 3.0 and b["per_pos"] == [0.9, 0.7, 0.4]
    finally:
        srv.shutdown()


def test_spec_override(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("model: x\nspeculative-config:\n  method: mtp\n  num_speculative_tokens: 3\n")
    assert json.loads(spec.spec_override(cfg, 2)) == {"method": "mtp", "num_speculative_tokens": 2}
    (tmp_path / "n.yaml").write_text("model: x\n")
    assert spec.spec_override(tmp_path / "n.yaml", 2) is None


def _rows():
    out = []
    for cfg, tps in (("NVFP4", (20, 23, 24, 23.5)), ("NVFP4_MJOLNIR_E2M3", (21, 24, 25.5, 25))):
        for k, v in zip((1, 2, 3, 4), tps):
            out.append(
                {
                    "config": cfg,
                    "k": k,
                    "concurrency": 1,
                    "decode_tps": v,
                    "agg_tps": v * 0.9,
                    "ttft_ms": 300,
                    "acceptance_length": 1 + 0.6 * k,
                    "per_pos": [0.9, 0.7, 0.5, 0.3][:k],
                    "image": "x",
                }
            )
    return out


def test_best_k_table_findings_chart():
    from mjolnir_cli import charts, theme
    from mjolnir_cli.readme.findings import findings
    from mjolnir_cli.readme.tables import TABLES

    rows = _rows()
    best = spec.best_k(rows)
    assert best[("NVFP4", 1)]["k"] == 3 and best[("NVFP4_MJOLNIR_E2M3", 1)]["k"] == 3
    doc = {"stage": "spec", "run_id": "20990101-000000", "rows": rows, "params": {}}
    table = TABLES["spec"](doc)
    assert "accept @4" in table and "**★**" in table
    found = findings("spec", doc)
    assert any("best with **3 drafts**" in f for f in found)
    assert any("Acceptance, 3 drafts" in f for f in found)
    assert charts.chart_spec({"spec": doc}, theme.DARK) is not None


def test_spec_delta_keeps_raw_position_counts():
    before = {"drafts": 0, "draft_tokens": 0, "accepted": 0, "per_pos": {0: 0, 1: 0}}
    after = {"drafts": 10, "draft_tokens": 20, "accepted": 12, "per_pos": {0: 8, 1: 4}}
    d = gate.spec_delta(before, after)
    assert d["per_pos_accepted"] == [8, 4] and d["per_pos"] == [0.8, 0.4]
