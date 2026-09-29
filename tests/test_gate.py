"""The clean-window gate against a local fake vLLM metrics endpoint — no GPU
involved. server_load parsing, wait_for_idle streak logic, and the preflight
failure path."""

from __future__ import annotations

import http.server
import threading

import pytest

from mjolnir.gate import CleanWindow, PreflightError, server_load, wait_for_idle


def _prom_body(running: float, waiting: float) -> bytes:
    return (
        "# HELP vllm:num_requests_running Requests currently running\n"
        "# TYPE vllm:num_requests_running gauge\n"
        f'vllm:num_requests_running{{model="x"}} {running}\n'
        "# HELP vllm:num_requests_waiting Requests waiting\n"
        f'vllm:num_requests_waiting{{model="x"}} {waiting}\n'
    ).encode()


class _MetricsHandler(http.server.BaseHTTPRequestHandler):
    body: bytes = _prom_body(0.0, 0.0)

    def do_GET(self):  # noqa: N802
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(self.body)

    def log_message(self, *args):  # silence
        pass


@pytest.fixture()
def metrics_server():
    """A fake vLLM /metrics on a random localhost port; set .body to steer."""
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _MetricsHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{server.server_address[1]}/metrics", server
    server.shutdown()


def test_server_load_parses_labeled_prometheus(metrics_server):
    url, _ = metrics_server
    _MetricsHandler.body = _prom_body(2.0, 0.0)
    assert server_load(url) == (2.0, 0.0)


def test_server_load_parses_bare_metric(metrics_server):
    url, _ = metrics_server
    _MetricsHandler.body = (
        b"vllm:num_requests_running 1.0\nvllm:num_requests_waiting 0.0\n"
    )
    assert server_load(url) == (1.0, 0.0)


def test_server_load_missing_metrics_is_none(metrics_server):
    url, _ = metrics_server
    _MetricsHandler.body = b"# only other metrics here\nfoo:bar 1.0\n"
    assert server_load(url) is None


def test_server_load_unreachable_is_none():
    assert server_load("http://127.0.0.1:1/metrics") is None  # port 1 = no listener


def test_wait_for_idle_opens_after_confirm_clean_samples(metrics_server):
    url, _ = metrics_server
    _MetricsHandler.body = _prom_body(0.0, 0.0)
    ts = wait_for_idle(url, poll_s=0.01, confirm=3, timeout_s=5)
    assert ts is not None and ts > 0


def test_wait_for_idle_dirty_sample_resets_streak(metrics_server):
    url, server = metrics_server
    # 1 0 0 1 0 0 0: the second 1 resets the streak, the window opens on the
    # trailing run of 3 clean samples (not on the earlier pair).
    sequence = iter([1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])

    class Dynamic(_MetricsHandler):
        def do_GET(self):  # noqa: N802
            _MetricsHandler.body = _prom_body(next(sequence, 0.0), 0.0)
            super().do_GET()

    server.RequestHandlerClass = Dynamic
    ts = wait_for_idle(url, poll_s=0.01, confirm=3, timeout_s=10)
    assert ts is not None


def test_wait_for_idle_times_out_when_busy(metrics_server):
    url, _ = metrics_server
    _MetricsHandler.body = _prom_body(1.0, 0.0)
    assert wait_for_idle(url, poll_s=0.01, confirm=3, timeout_s=0.1) is None


def test_clean_window_preflight_raises_when_endpoint_down():
    with pytest.raises(PreflightError, match="unreachable"):
        with CleanWindow("http://127.0.0.1:1/metrics", poll_s=0.01, timeout_s=1):
            pass


def test_clean_window_reports_clean_and_dirty(metrics_server):
    url, _ = metrics_server
    _MetricsHandler.body = _prom_body(0.0, 0.0)
    with CleanWindow(url, confirm=2, poll_s=0.01, timeout_s=10) as w:
        pass
    assert w.open_ts is not None
    assert w.clean is True
    assert w.dirty_samples == []
    summary = w.summary()
    assert summary["clean"] is True
    assert summary["n_in_window"] >= 1
