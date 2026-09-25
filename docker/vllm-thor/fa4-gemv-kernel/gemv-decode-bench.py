#!/usr/bin/env python3
"""GEMV decode Phase B1 — clean-window performance bench on Jetson Thor.

Measures the Phase-B1 GEMV decode kernel (SplitKV + paged + in-tree combine)
against the production baselines at M=1, L in {2048, 4096, 8192}, GQA 24/4,
hd=256, bf16 Q + fp8 e4m3 KV, non-unity descales [1.06, 1.125, 1.25]:

Modes (one process per mode; results merged externally):
  gemv_dense   : GEMV dense KV, ns in {1, 2, 4, 8} + auto (fill the SMs).
                 ns=1 is the Phase-A write-through path.
  gemv_paged   : GEMV paged-128 KV (identity page table), ns in {1, 4, 8} + auto.
  fa4_1cta     : FA4 hd256 1CTA carve-out (natural interface path, GEMV off) —
                 paged-128 varlen, the decode-1cta-clean baseline (~207us @L=8192).
                 Ctor calls are recorded and must all show use_2cta=False.
  flashinfer   : FlashInfer BatchDecodeWithPagedKVCacheWrapper (fa2
                 tensor-core, block-16, bf16 Q + fp8 KV, unity scales — the
                 production v9 path; ~73us @L=8192).

Roofline (in the report): bytes = L * HK * D * 2 * 1B (K+V fp8) at
~273 GB/s (Thor LPDDR5X) -> ~59 us at L=8192 (arxiv-gemv-decode.md §2.1.4).

Protocol: 20 warmup + 300 pooled CUDA-event timed iters per leg (median /
p95 / min / mean). Window gating: a monitor thread samples
vllm:num_requests_running/waiting from the co-located server
(http://127.0.0.1:6001/metrics, --network host) every 2 s; the window opens
after 3 consecutive (0,0) samples and the run is DIRTY (exit 2) if any sample
inside the window is not (0,0). Exit 3 = idle-wait timeout (driver retries).
JIT compiles happen during prep (outside the clean window).

Exit codes: 0 = clean + written; 2 = dirty window; 3 = idle-wait timeout.

Run (driver, one docker per mode):
  docker run --rm --gpus all --network host --entrypoint python3 \
    -v $VFA_TREE:/usr/local/lib/python3.12/dist-packages/vllm/vllm_flash_attn \
    -v "$TASK":/p -e TMPDIR=/tmp/gemv-b1-<mode> \
    mjolnir/vllm-thor:qwen38-sm110-v11 /p/gemv-decode-bench.py --mode <mode> --out /p/gemv-decode-bench-<mode>.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import threading
import time
import traceback
import urllib.request

import torch

NUM_Q_HEADS = 24
NUM_KV_HEADS = 4
HEAD_DIM = 256
BLOCK16 = 16
DEVICE = "cuda"
SOFTMAX_SCALE = HEAD_DIM ** -0.5
FP8 = torch.float8_e4m3fn

LS = (2048, 4096, 8192)
GEMV_NS_DENSE = (1, 2, 4, 8, "auto")
GEMV_NS_PAGED = (1, 4, 8, "auto")
WARMUP = 20
ITERS = 300
ROOFLINE_BW = 273e9  # B/s (Thor LPDDR5X, arxiv-gemv-decode.md §2.1.4)

METRICS_URL = "http://127.0.0.1:6001/metrics"
POLL_S = 2.0
CONFIRM = 3
WAIT_TIMEOUT_S = 1800

# FA4 1CTA ctor forcing: None = record only (natural path); True/False = force.
_ctor_log: list[bool] = []
_FORCE: bool | None = None


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


def auto_ns(L: int) -> int:
    """Host-side replica of the interface's auto plan (blocks_per_sm=4, 20 SMs,
    128-row chunk floor, combine cap 64) — for labeling the 'auto' legs."""
    num_SMs = int(torch.cuda.get_device_properties(0).multi_processor_count)
    ns = -((-(-((num_SMs * 4) // (NUM_KV_HEADS * 1))) // 1) if False else
           (num_SMs * 4 + NUM_KV_HEADS - 1) // NUM_KV_HEADS)
    return max(1, min(ns, 64, -(-L // 128)))


# --- case construction ---------------------------------------------------------
class Cases:
    """Shared KV dataset (identical values for every leg), per-L layouts."""

    def __init__(self) -> None:
        torch.manual_seed(20260925)
        self.q = torch.randn(NUM_Q_HEADS, HEAD_DIM, device=DEVICE, dtype=torch.float32).to(torch.bfloat16)
        self.k = torch.randn(LS[-1], NUM_KV_HEADS, HEAD_DIM, device=DEVICE, dtype=torch.float32).to(FP8)
        self.v = torch.randn(LS[-1], NUM_KV_HEADS, HEAD_DIM, device=DEVICE, dtype=torch.float32).to(FP8)
        self.qd = torch.full((1, NUM_KV_HEADS), 1.06, dtype=torch.float32, device=DEVICE)
        self.kd = torch.full((1, NUM_KV_HEADS), 1.125, dtype=torch.float32, device=DEVICE)
        self.vd = torch.full((1, NUM_KV_HEADS), 1.25, dtype=torch.float32, device=DEVICE)
        self.f16 = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=DEVICE)


def gemv_leg(cases: Cases, L: int, ns: int | str, page_size: int):
    """One GEMV (ns, layout) callable. Sets the env knobs (read per call)."""
    import os as _os
    from vllm.vllm_flash_attn.cute.interface import _flash_attn_fwd

    if ns == "auto":
        _os.environ.pop("VLLM_FA4_HD256_GEMV_NUM_SPLITS", None)
    else:
        _os.environ["VLLM_FA4_HD256_GEMV_NUM_SPLITS"] = str(ns)
    _os.environ["VLLM_FA4_HD256_GEMV"] = "1"
    k = cases.k[:L]
    v = cases.v[:L]
    if page_size == 0:
        k_in, v_in, pt = k.unsqueeze(0).contiguous(), v.unsqueeze(0).contiguous(), None
    else:
        npg = L // page_size
        k_in = k.view(npg, page_size, NUM_KV_HEADS, HEAD_DIM).contiguous()
        v_in = v.view(npg, page_size, NUM_KV_HEADS, HEAD_DIM).contiguous()
        pt = torch.arange(npg, dtype=torch.int32, device=DEVICE).unsqueeze(0)
    q_in = cases.q.unsqueeze(0).unsqueeze(0)  # (1, 1, H, D)

    def fn() -> None:
        _flash_attn_fwd(
            q_in, k_in, v_in, page_table=pt,
            max_seqlen_q=1, max_seqlen_k=L,
            softmax_scale=SOFTMAX_SCALE, causal=True,
            q_descale=cases.qd, k_descale=cases.kd, v_descale=cases.vd,
        )

    return fn


class Fa4Leg:
    def __init__(self, cases: Cases, L: int) -> None:
        from vllm.vllm_flash_attn import flash_attn_varlen_func
        self._call = flash_attn_varlen_func
        npg = L // 128
        self.args = dict(
            q=cases.q, k=cases.k[:L].view(npg, 128, NUM_KV_HEADS, HEAD_DIM).contiguous(),
            v=cases.v[:L].view(npg, 128, NUM_KV_HEADS, HEAD_DIM).contiguous(),
            max_seqlen_q=1,
            cu_seqlens_q=torch.tensor([0, 1], dtype=torch.int32, device=DEVICE),
            max_seqlen_k=L,
            seqused_k=torch.tensor([L], dtype=torch.int32, device=DEVICE),
            block_table=torch.arange(npg, dtype=torch.int32, device=DEVICE).unsqueeze(0),
            softmax_scale=SOFTMAX_SCALE, causal=True, fa_version=4,
            q_descale=cases.qd, k_descale=cases.kd, v_descale=cases.vd,
        )

    def __call__(self) -> None:
        self._call(**self.args)


class FiLeg:
    def __init__(self, cases: Cases, L: int) -> None:
        from flashinfer import BatchDecodeWithPagedKVCacheWrapper
        self._cls = BatchDecodeWithPagedKVCacheWrapper
        self.wrapper: BatchDecodeWithPagedKVCacheWrapper | None = None
        self.cases = cases
        self.L = L

    def setup(self) -> None:
        self.wrapper = self._cls(
            self.cases.f16, kv_layout="NHD", use_cuda_graph=False,
            use_tensor_cores=True, backend="auto",
        )
        nblk = self.L // BLOCK16
        k = self.cases.k[:self.L]
        v = self.cases.v[:self.L]
        self.kv = (k.view(nblk, BLOCK16, NUM_KV_HEADS, HEAD_DIM).contiguous(),
                   v.view(nblk, BLOCK16, NUM_KV_HEADS, HEAD_DIM).contiguous())
        self.out = torch.empty(1, NUM_Q_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=DEVICE)

    def __call__(self) -> None:
        assert self.wrapper is not None
        nblk = self.L // BLOCK16
        self.wrapper.plan(
            torch.tensor([0, nblk], dtype=torch.int32, device=DEVICE),
            torch.arange(nblk, dtype=torch.int32, device=DEVICE),
            torch.full((1,), BLOCK16, dtype=torch.int32, device=DEVICE),
            num_qo_heads=NUM_Q_HEADS, num_kv_heads=NUM_KV_HEADS,
            head_dim=HEAD_DIM, page_size=BLOCK16,
            q_data_type=torch.bfloat16, kv_data_type=FP8,
            o_data_type=torch.bfloat16, sm_scale=SOFTMAX_SCALE, q_len_per_req=1,
        )
        self.wrapper.run(self.cases.q, self.kv, q_scale=1.0, k_scale=1.0, v_scale=1.0, out=self.out)


# --- timing ---------------------------------------------------------------------
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
        "median": round(st.median(times), 2),
        "p95": round(sorted(times)[int(0.95 * (len(times) - 1))], 2),
        "min": round(min(times), 2),
        "mean": round(st.mean(times), 2),
        "iters": len(times),
    }


def install_ctor_hook() -> None:
    import vllm.vllm_flash_attn.cute.sm100_hd256_2cta_fmha_forward as fwd_mod

    bfa = fwd_mod.BlackwellFusedMultiHeadAttentionForward
    orig = bfa.__init__

    def hooked(self, *args, **kwargs):
        if _FORCE is not None:
            kwargs["use_2cta"] = _FORCE
        _ctor_log.append(bool(kwargs.get("use_2cta", True)))
        orig(self, *args, **kwargs)

    bfa.__init__ = hooked


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("gemv_dense", "gemv_paged", "fa4_1cta", "flashinfer"), required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-gate", action="store_true")
    args = ap.parse_args()
    if args.out is None:
        args.out = f"/tmp/gemv-bench-{args.mode}.json"

    import vllm.vllm_flash_attn.cute.interface as iface

    props = torch.cuda.get_device_properties(0)
    env = {
        "mode": args.mode,
        "device": str(props.name),
        "capability": list(torch.cuda.get_device_capability()),
        "sms": int(props.multi_processor_count),
        "torch": torch.__version__,
        "interface_sha256": hashlib.sha256(open(iface.__file__, "rb").read()).hexdigest(),
        "warmup": WARMUP,
        "iters": ITERS,
        "roofline_bw_bytes_per_s": ROOFLINE_BW,
        "descales": {"q": 1.06, "k": 1.125, "v": 1.25},
    }

    if args.mode == "fa4_1cta":
        install_ctor_hook()

    mon = Monitor()
    mon.start()

    # --- Phase A: prep (JIT compiles happen here, outside the clean window) ---
    print(f"[{args.mode}] prep: building cases + compiling/warming all legs...", flush=True)
    cases = Cases()
    legs: dict[str, object] = {}
    try:
        for L in LS:
            if args.mode == "gemv_dense":
                for ns in GEMV_NS_DENSE:
                    tag = f"ns{ns}" if ns != "auto" else f"ns{auto_ns(L)}(auto)"
                    legs[f"L{L}|{tag}"] = gemv_leg(cases, L, ns, page_size=0)
            elif args.mode == "gemv_paged":
                for ns in GEMV_NS_PAGED:
                    tag = f"ns{ns}" if ns != "auto" else f"ns{auto_ns(L)}(auto)"
                    legs[f"L{L}|{tag}"] = gemv_leg(cases, L, ns, page_size=128)
            elif args.mode == "fa4_1cta":
                legs[f"L{L}"] = Fa4Leg(cases, L)
            else:
                leg = FiLeg(cases, L)
                leg.setup()
                legs[f"L{L}"] = leg
            fn = legs[f"L{L}|ns1"] if args.mode in ("gemv_dense", "gemv_paged") else legs[f"L{L}"]
            fn()
            torch.cuda.synchronize()
            print(f"[{args.mode}]   prepped L={L} (first leg)", flush=True)
    except Exception:  # noqa: BLE001
        print("PREP EXCEPTION (verbatim):")
        print(traceback.format_exc())
        mon.stop()
        return 1

    load0 = server_load()
    print(f"[{args.mode}] prep done; server load now: {load0}", flush=True)

    # --- wait for a clean window ---------------------------------------------
    if args.no_gate:
        window_start = time.time()
        print(f"[{args.mode}] SMOKE MODE (--no-gate): window not gated")
    else:
        window_start = wait_for_idle(mon)
        if window_start is None:
            print(f"[{args.mode}] TIMEOUT waiting for 0/0 window after {WAIT_TIMEOUT_S}s")
            mon.stop()
            return 3
        print(f"[{args.mode}] CLEAN WINDOW OPEN (3 consecutive 0/0 samples)", flush=True)

    # --- Phase B: measurement window -----------------------------------------
    results: dict[str, dict] = {}
    try:
        for key in sorted(legs):
            fn = legs[key]
            t = time_iters(fn)
            results[key] = summarize(t)
            print(f"[{args.mode}] {key:24s} med={results[key]['median']:8.2f} us "
                  f"p95={results[key]['p95']:8.2f} us min={results[key]['min']:8.2f} us", flush=True)
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

    ctor_verdict: dict | None = None
    if args.mode == "fa4_1cta":
        bad = [v for v in _ctor_log if v]
        ctor_verdict = {
            "expected_use_2cta": False,
            "ctor_calls": len(_ctor_log),
            "all_as_expected": not bad,
            "unexpected": bad,
        }

    # Roofline per L (K+V fp8 bytes at ROOFLINE_BW).
    roofline = {
        f"L{L}": round(L * NUM_KV_HEADS * HEAD_DIM * 2 / ROOFLINE_BW * 1e6, 2)
        for L in LS
    }

    report = {
        "env": env,
        "window": {
            "start": window_start, "end": window_end,
            "clean": clean, "num_samples": len(in_window), "dirty_samples": dirty,
            "server_load_before_wait": load0, "server_load_at_end": load_end,
        },
        "ctor_verification": ctor_verdict,
        "roofline_us": roofline,
        "results": results,
    }
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[{args.mode}] wrote {args.out}")
    print(f"[{args.mode}] window clean: {clean} (samples={len(in_window)}, dirty={len(dirty)})")
    if ctor_verdict:
        print(f"[{args.mode}] ctor verification: {ctor_verdict}")
    print(f"[{args.mode}] VERDICT: {'CLEAN' if clean else 'DIRTY'}")
    return 0 if clean else 2


if __name__ == "__main__":
    raise SystemExit(main())
