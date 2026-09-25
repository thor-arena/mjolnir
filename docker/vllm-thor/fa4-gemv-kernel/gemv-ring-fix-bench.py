#!/usr/bin/env python3
"""GEMV ring-depth fix — clean-window re-measure (L=8192, M=1, GQA 24/4, fp8 KV).

Two-lever study for the ``stages 2 -> 16`` fix in
``sm100_hd256_decode_gemv.py`` (see gemv-ring-fix-bench.md):

Lever 1 (ns, the parallelism axis):  st16 (fixed) at ns in {1, 4, 8, 20, 64}
                                     + ns=auto (no env knob -> _gemv_auto_num_splits).
Lever 2 (ring depth, the in-flight axis): stages=2 (pre-fix, monkey-patched)
                                     vs stages=16 (fixed default) at ns=1 and ns=20.
Headline:  st16+ns20 (combined fix) vs FlashInfer FA2-tc paged-16 (~73 us),
           measured in the SAME clean window.

All wall-clock legs run in one process, one gated window. Protocol:
  * gate: vllm:num_requests_running/waiting from http://127.0.0.1:6001/metrics,
    sampled every 2 s; the window OPENS after 6 consecutive (0,0) samples
    (CONFIRM=6, stricter than the original 3) and the run is DIRTY (exit 2)
    if ANY sample inside the window is not (0,0);
  * re-verify: server_load() must be (0,0) again IMMEDIATELY before each
    timed burst; a non-zero sample aborts the run (exit 4) without writing.
  * 20 warmup + 300 pooled CUDA-event timed iters per leg (median/p95/min/mean).

Leg ordering constraint: the interface's GEMV compile cache does NOT key on
``stages``, so the cache can hold only ONE stages variant at a time. Legs are
therefore ordered all-st16 -> all-st2 -> FI: each block is compiled (prep,
outside the window) and measured while its own compiled objects are the ones
in the cache. The ns env knob is set per timed call (global process state).

JIT compiles (both stages variants) happen during prep, outside the window.

Exit codes: 0 = clean + written; 2 = dirty window; 3 = idle-wait timeout;
            4 = non-zero load at a pre-burst re-check (abort).

Run (driver):
  docker run --rm --gpus all --network host --entrypoint python3 \
    -v $VFA_TREE:/usr/local/lib/python3.12/dist-packages/vllm/vllm_flash_attn \
    -v "$TASK":/p -e TMPDIR=/tmp/ringfix \
    mjolnir/vllm-thor:qwen38-sm110-v11 \
    /p/gemv-ring-fix-bench.py --out /p/gemv-ring-fix-bench.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import threading
import time
import traceback
import urllib.request

import torch

NUM_Q_HEADS = 24
NUM_KV_HEADS = 4
HEAD_DIM = 256
DEVICE = "cuda"
SOFTMAX_SCALE = HEAD_DIM ** -0.5
FP8 = torch.float8_e4m3fn
L = 8192

# All-st16 first (file default after the fix), then st2 (pre-fix, monkey-patched),
# then the FlashInfer baseline — see the module docstring for why.
LEGS: list[tuple[str, int | None, int | str]] = [
    ("st16_ns1", 16, 1),
    ("st16_ns4", 16, 4),
    ("st16_ns8", 16, 8),
    ("st16_ns20", 16, 20),
    ("st16_ns64", 16, 64),
    ("st16_auto", 16, "auto"),
    ("st2_ns1", 2, 1),
    ("st2_ns20", 2, 20),
    ("flashinfer", None, None),
]

WARMUP = 20
ITERS = 300
KV_BYTES = L * NUM_KV_HEADS * HEAD_DIM * 2  # K+V fp8 (nominal; ncu gives exact)
ROOFLINE_BW = 273e9  # B/s (Thor LPDDR5X)

METRICS_URL = "http://127.0.0.1:6001/metrics"
POLL_S = 2.0
CONFIRM = 6  # stricter than the original 3
WAIT_TIMEOUT_S = 3600


def server_load() -> tuple[float, float] | None:
    """(num_running, num_waiting) from the co-located vllm server, or None."""
    try:
        with urllib.request.urlopen(METRICS_URL, timeout=5) as r:
            text = r.read().decode()
    except Exception:  # noqa: BLE001
        return None
    out: dict[str, float] = {}
    for line in text.splitlines():
        for key in ("vllm:num_requests_running", "vllm:num_requests_waiting"):
            if line.startswith(key + "{") or line.startswith(key + " "):
                out[key.split(":")[1]] = float(line.rsplit(" ", 1)[1])
    if "num_requests_running" not in out or "num_requests_waiting" not in out:
        return None
    return (out["num_requests_running"], out["num_requests_waiting"])


class Monitor(threading.Thread):
    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.trace: list[tuple[float, tuple[float, float] | None]] = []
        self._stop = threading.Event()

    def run(self) -> None:
        while not self._stop.is_set():
            self.trace.append((time.time(), server_load()))
            self._stop.wait(POLL_S)

    def stop(self) -> None:
        self._stop.set()


def wait_for_idle(mon: Monitor) -> float | None:
    t_start = time.perf_counter()
    consumed = 0
    streak = 0
    while time.perf_counter() - t_start < WAIT_TIMEOUT_S:
        while consumed < len(mon.trace):
            _ts, load = mon.trace[consumed]
            consumed += 1
            if load == (0.0, 0.0):
                streak += 1
            else:
                streak = 0
        if streak >= CONFIRM:
            return mon.trace[consumed - 1][0]
        time.sleep(1.0)
    return None


# --- cases ----------------------------------------------------------------------
class Cases:
    def __init__(self) -> None:
        torch.manual_seed(20260925)
        self.q = torch.randn(NUM_Q_HEADS, HEAD_DIM, device=DEVICE,
                             dtype=torch.float32).to(torch.bfloat16)
        self.k = torch.randn(L, NUM_KV_HEADS, HEAD_DIM, device=DEVICE,
                             dtype=torch.float32).to(FP8)
        self.v = torch.randn(L, NUM_KV_HEADS, HEAD_DIM, device=DEVICE,
                             dtype=torch.float32).to(FP8)
        self.qd = torch.full((1, NUM_KV_HEADS), 1.06, dtype=torch.float32, device=DEVICE)
        self.kd = torch.full((1, NUM_KV_HEADS), 1.125, dtype=torch.float32, device=DEVICE)
        self.vd = torch.full((1, NUM_KV_HEADS), 1.25, dtype=torch.float32, device=DEVICE)
        self.f16 = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=DEVICE)


# The interface's GEMV compile cache does not key on stages, so it can hold
# only ONE stages variant at a time. Both variants are compiled during prep
# (outside the window) and the two cache snapshots are SWAPPED around the
# st16/st2 measurement blocks — no recompile inside the window.
# The __init__ monkey-patch (stages != 16) matters only at compile time.
_stages_patch = {"orig": None}


def gemv_cache(iface) -> dict:
    if not hasattr(iface._flash_attn_fwd, "gemv_compile_cache"):
        iface._flash_attn_fwd.gemv_compile_cache = {}
    return iface._flash_attn_fwd.gemv_compile_cache


def set_stages(stages: int, iface) -> None:
    from vllm.vllm_flash_attn.cute.sm100_hd256_decode_gemv import (
        BlackwellHd256DecodeGEMV as K,
    )
    if _stages_patch["orig"] is not None:
        K.__init__ = _stages_patch["orig"]
        _stages_patch["orig"] = None
    if stages != 16:
        orig = K.__init__

        def patched(self, *a, **kw):
            orig(self, *a, **kw)
            self.stages = stages

        K.__init__ = patched
        _stages_patch["orig"] = orig
    gemv_cache(iface).clear()


def make_gemv_leg(cases: Cases, ns: int | str):
    """One GEMV leg callable. The ns env knob is set PER CALL (global state)."""
    import vllm.vllm_flash_attn.cute.interface as iface

    q_in = cases.q.unsqueeze(0).unsqueeze(0)
    k_in = cases.k.unsqueeze(0).contiguous()
    v_in = cases.v.unsqueeze(0).contiguous()

    def fn() -> None:
        os.environ["VLLM_FA4_HD256_GEMV"] = "1"
        if ns == "auto":
            os.environ.pop("VLLM_FA4_HD256_GEMV_NUM_SPLITS", None)
        else:
            os.environ["VLLM_FA4_HD256_GEMV_NUM_SPLITS"] = str(ns)
        iface._flash_attn_fwd(
            q_in, k_in, v_in,
            max_seqlen_q=1, max_seqlen_k=L,
            softmax_scale=SOFTMAX_SCALE, causal=True,
            q_descale=cases.qd, k_descale=cases.kd, v_descale=cases.vd,
        )

    return fn


class FiLeg:
    """FlashInfer FA2-tc paged-16 wrapper — the ~73 us production baseline."""

    def __init__(self, cases: Cases) -> None:
        from flashinfer import BatchDecodeWithPagedKVCacheWrapper
        self.wrapper = BatchDecodeWithPagedKVCacheWrapper(
            cases.f16, kv_layout="NHD", use_cuda_graph=False,
            use_tensor_cores=True, backend="auto",
        )
        self.kv = (cases.k.view(L // 16, 16, NUM_KV_HEADS, HEAD_DIM).contiguous(),
                   cases.v.view(L // 16, 16, NUM_KV_HEADS, HEAD_DIM).contiguous())
        self.out = torch.empty(1, NUM_Q_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=DEVICE)
        self.q = cases.q.unsqueeze(0)  # (1, H, D) so q_len_per_req=1 is unambiguous
        nblk = L // 16
        self._plan_args = (
            torch.tensor([0, nblk], dtype=torch.int32, device=DEVICE),
            torch.arange(nblk, dtype=torch.int32, device=DEVICE),
            torch.full((1,), 16, dtype=torch.int32, device=DEVICE),
        )

    def __call__(self) -> None:
        self.wrapper.plan(
            *self._plan_args,
            num_qo_heads=NUM_Q_HEADS, num_kv_heads=NUM_KV_HEADS,
            head_dim=HEAD_DIM, page_size=16,
            q_data_type=torch.bfloat16, kv_data_type=FP8,
            o_data_type=torch.bfloat16, sm_scale=SOFTMAX_SCALE, q_len_per_req=1,
        )
        self.wrapper.run(self.q, self.kv, q_scale=1.0, k_scale=1.0, v_scale=1.0,
                         out=self.out)


# --- timing ----------------------------------------------------------------------
def time_iters(fn, warmup: int = WARMUP, iters: int = ITERS) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times: list[float] = []
    ev0 = torch.cuda.Event(enable_timing=True)
    ev1 = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        ev0.record()
        fn()
        ev1.record()
        torch.cuda.synchronize()
        times.append(ev0.elapsed_time(ev1) * 1000.0)
    return times


def summarize(times: list[float]) -> dict:
    st = statistics
    return {
        "median": round(st.median(times), 3),
        "p95": round(sorted(times)[int(0.95 * (len(times) - 1))], 3),
        "min": round(min(times), 3),
        "mean": round(st.mean(times), 3),
        "iters": len(times),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/p/gemv-ring-fix-bench.json")
    ap.add_argument("--no-gate", action="store_true")
    args = ap.parse_args()

    import vllm.vllm_flash_attn.cute.interface as iface

    props = torch.cuda.get_device_properties(0)
    env = {
        "device": str(props.name),
        "capability": list(torch.cuda.get_device_capability()),
        "sms": int(props.multi_processor_count),
        "torch": torch.__version__,
        "interface_sha256": hashlib.sha256(open(iface.__file__, "rb").read()).hexdigest(),
        "gemv_sha256": hashlib.sha256(
            open(iface.__file__.replace("interface.py", "sm100_hd256_decode_gemv.py"),
                 "rb").read()).hexdigest(),
        "L": L, "warmup": WARMUP, "iters": ITERS,
        "kv_bytes": KV_BYTES, "roofline_bw_bytes_per_s": ROOFLINE_BW,
        "gate": {"poll_s": POLL_S, "confirm_00_streak": CONFIRM},
    }

    # Init the CUDA context TINY before the heavy imports (OOM workaround).
    _tiny = torch.zeros(1, device=DEVICE)  # noqa: F841

    mon = Monitor()
    mon.start()

    print("[prep] building cases + compiling all legs (outside the window)...", flush=True)
    cases = Cases()
    legs: dict[str, object] = {}
    A_cache: dict = {}  # stages=16 compiled objects (fixed default)
    B_cache: dict = {}  # stages=2 compiled objects (pre-fix)
    last_stages: int | None = None
    for tag, stages, ns in LEGS:
        if stages is None:
            legs[tag] = FiLeg(cases)
        else:
            if stages != last_stages:
                if last_stages == 16:
                    A_cache = dict(gemv_cache(iface))
                set_stages(stages, iface)
                last_stages = stages
            legs[tag] = make_gemv_leg(cases, ns)
        legs[tag]()
        torch.cuda.synchronize()
        print(f"[prep]   {tag} ready", flush=True)
        if tag == "st2_ns20":
            B_cache = dict(gemv_cache(iface))
    # Restore the st16 state: unpatch + put the A snapshot back in the live cache.
    if _stages_patch["orig"] is not None:
        from vllm.vllm_flash_attn.cute.sm100_hd256_decode_gemv import (
            BlackwellHd256DecodeGEMV as K,
        )
        K.__init__ = _stages_patch["orig"]
        _stages_patch["orig"] = None
    gc = gemv_cache(iface)
    gc.clear()
    gc.update(A_cache)
    assert A_cache and B_cache, f"variant caches incomplete: A={len(A_cache)} B={len(B_cache)}"
    load0 = server_load()
    print(f"[prep] done; server load now: {load0}", flush=True)

    if args.no_gate:
        window_start = time.time()
        print("SMOKE MODE (--no-gate): window not gated")
    else:
        window_start = wait_for_idle(mon)
        if window_start is None:
            print(f"TIMEOUT waiting for {CONFIRM}x consecutive 0/0 after {WAIT_TIMEOUT_S}s")
            mon.stop()
            return 3
        print(f"CLEAN WINDOW OPEN ({CONFIRM} consecutive 0/0 samples)", flush=True)

    results: dict[str, dict] = {}
    swapped_to_b = False
    try:
        for tag in [t for t, _, _ in LEGS]:
            fn = legs[tag]
            if tag.startswith("st2") and not swapped_to_b:
                gc = gemv_cache(iface)
                gc.clear()
                gc.update(B_cache)
                swapped_to_b = True
                print("SWAP: stages=2 objects installed in the GEMV cache", flush=True)
            if not args.no_gate:
                load = server_load()
                if load != (0.0, 0.0):
                    print(f"ABORT: pre-burst re-check for {tag} saw load={load}", flush=True)
                    mon.stop()
                    return 4
            t = time_iters(fn)
            results[tag] = summarize(t)
            results[tag]["bw_gbs_nominal_kv"] = round(
                KV_BYTES / (results[tag]["median"] * 1e-6) / 1e9, 2)
            print(f"{tag:14s} med={results[tag]['median']:8.2f} us "
                  f"min={results[tag]['min']:8.2f} us "
                  f"p95={results[tag]['p95']:8.2f} us  "
                  f"(nominal KV BW {results[tag]['bw_gbs_nominal_kv']} GB/s)", flush=True)
    except Exception:  # noqa: BLE001
        print("EXCEPTION (verbatim):")
        print(traceback.format_exc())
        mon.stop()
        return 1

    window_end = time.time()
    load_end = server_load()
    mon.stop()
    in_window = [(ts, load) for ts, load in mon.trace if window_start <= ts <= window_end]
    dirty = [s for s in in_window if s[1] is None or s[1] != (0.0, 0.0)]
    clean = not dirty

    report = {
        "env": env,
        "window": {
            "start": window_start, "end": window_end,
            "clean": clean, "num_samples": len(in_window),
            "dirty_samples": [list(d) for d in dirty],
            "server_load_before_wait": load0, "server_load_at_end": load_end,
        },
        "results": results,
    }
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"wrote {args.out}")
    print(f"window clean: {clean} (samples={len(in_window)}, dirty={len(dirty)})")
    print(f"VERDICT: {'CLEAN' if clean else 'DIRTY'}")
    return 0 if clean else 2


if __name__ == "__main__":
    raise SystemExit(main())
