#!/usr/bin/env python3
"""ncu driver for the GEMV ring-fix points (L=8192, dense fp8, ns/stages knobs).

Env:
  NCU_L (default 8192), NCU_NS (default 1), NCU_STAGES (default: file default=16;
  set 2 for the pre-fix ring), NCU_ITERS (profiled launches per kernel, default 4).

The host runs ncu with --launch-skip <2*warmup launches> and a launch-count
covering NCU_ITERS GEMV launches (for ns>1 each call also launches the
in-tree combine kernel; the per-kernel summary separates them).

  docker run --rm --gpus all --network host \
    --security-opt seccomp=unconfined \
    -v "$VFA":/usr/local/lib/python3.12/dist-packages/vllm/vllm_flash_attn \
    -v "$TASK":/p -v /opt/nvidia/nsight-compute/2025.3.0:/ncu:ro \
    -e TMPDIR=/tmp/ncu-ring -e NCU_NS=20 -e NCU_STAGES=16 -e NCU_ITERS=4 \
    mjolnir/vllm-thor:qwen38-sm110-v11 \
    /ncu/ncu --target-processes all --clock-control none --cache-control none \
      --launch-skip 2 --launch-count 8 \
      --metrics dram__bytes.sum,gpu__time_duration.sum,lts__t_sectors.sum,sm__cycles_active.sum \
      --print-summary per-kernel \
      python3 /p/fa4-hd256-fp8/ncu-gemv-ring.py
"""
from __future__ import annotations

import os

import torch

NUM_Q_HEADS = 24
NUM_KV_HEADS = 4
HEAD_DIM = 256
L = int(os.environ.get("NCU_L", "8192"))
NS = int(os.environ.get("NCU_NS", "1"))
STAGES = os.environ.get("NCU_STAGES", "")  # "" = file default (16)
ITERS = int(os.environ.get("NCU_ITERS", "4"))
DEVICE = "cuda"
SOFTMAX_SCALE = HEAD_DIM ** -0.5
FP8 = torch.float8_e4m3fn


def main() -> int:
    _tiny = torch.zeros(1, device=DEVICE)  # tiny ctx init before heavy import
    import vllm.vllm_flash_attn.cute.interface as iface
    from vllm.vllm_flash_attn.cute.sm100_hd256_decode_gemv import (
        BlackwellHd256DecodeGEMV,
    )

    if STAGES:
        orig = BlackwellHd256DecodeGEMV.__init__

        def patched(self, *a, **kw):
            orig(self, *a, **kw)
            self.stages = int(STAGES)

        BlackwellHd256DecodeGEMV.__init__ = patched
        if hasattr(iface._flash_attn_fwd, "gemv_compile_cache"):
            del iface._flash_attn_fwd.gemv_compile_cache

    torch.manual_seed(0)
    q = torch.randn(NUM_Q_HEADS, HEAD_DIM, device=DEVICE, dtype=torch.float32).to(torch.bfloat16)
    k = torch.randn(L, NUM_KV_HEADS, HEAD_DIM, device=DEVICE, dtype=torch.float32).to(FP8)
    v = torch.randn(L, NUM_KV_HEADS, HEAD_DIM, device=DEVICE, dtype=torch.float32).to(FP8)
    qd = torch.full((1, NUM_KV_HEADS), 1.06, dtype=torch.float32, device=DEVICE)
    kd = torch.full((1, NUM_KV_HEADS), 1.125, dtype=torch.float32, device=DEVICE)
    vd = torch.full((1, NUM_KV_HEADS), 1.25, dtype=torch.float32, device=DEVICE)

    os.environ["VLLM_FA4_HD256_GEMV"] = "1"
    os.environ["VLLM_FA4_HD256_GEMV_NUM_SPLITS"] = str(NS)

    # warm/compile outside the profiled region
    for _ in range(2):
        iface._flash_attn_fwd(
            q.unsqueeze(0).unsqueeze(0), k.unsqueeze(0).contiguous(),
            v.unsqueeze(0).contiguous(), max_seqlen_q=1, max_seqlen_k=L,
            softmax_scale=SOFTMAX_SCALE, causal=True,
            q_descale=qd, k_descale=kd, v_descale=vd,
        )
    torch.cuda.synchronize()
    print(f"[ncu-driver] L={L} ns={NS} stages={STAGES or 'default(16)'} "
          f"ready; profiling {ITERS} GEMV launches", flush=True)

    for _ in range(ITERS):
        iface._flash_attn_fwd(
            q.unsqueeze(0).unsqueeze(0), k.unsqueeze(0).contiguous(),
            v.unsqueeze(0).contiguous(), max_seqlen_q=1, max_seqlen_k=L,
            softmax_scale=SOFTMAX_SCALE, causal=True,
            q_descale=qd, k_descale=kd, v_descale=vd,
        )
        torch.cuda.synchronize()
    print("[ncu-driver] done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
