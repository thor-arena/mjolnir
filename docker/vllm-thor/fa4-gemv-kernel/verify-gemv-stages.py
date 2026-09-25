#!/usr/bin/env python3
"""GEMV ring-depth fix — correctness check (stages=16 vs stages=2).

The ring fix raises ``BlackwellHd256DecodeGEMV.stages`` 2 -> 16. The consumer
loop (rolling ``cp_async_wait_group(2*S-1)``) and the S-stage prologue preload
are S-agnostic, so the output must be unchanged. This script verifies, on
dense fp8 KV (bf16 Q, non-unity descales), M=1, GQA 24/4, hd=256:

1. stages=16 (new default) vs fp32 reference — L in {5, 128, 8192}, ns=1
   (same tolerances as verify-gemv-decode.py: 5e-2 abs / 5e-2 rel / 2e-3 LSE).
2. stages=16 vs stages=2 (pre-fix, monkey-patched) — same inputs: the two
   must be (near-)identical; stages only changes load timing, never the
   per-row FMA accumulation order.
3. auto-ns default: print ``_gemv_auto_num_splits(4, 1, device, arch)``.

The interface's GEMV compile cache does NOT key on stages, so each stages
variant is built in a fresh cache (separate process would also work).

Run (no clean-window gate — untimed):
  docker run --rm --gpus all --entrypoint python3 \
    -v $VFA_TREE:/usr/local/lib/python3.12/dist-packages/vllm/vllm_flash_attn \
    -v "$TASK":/p \
    mjolnir/vllm-thor:qwen38-sm110-v11 /p/verify-gemv-stages.py
"""
from __future__ import annotations

import os
import traceback

import torch

NUM_Q_HEADS = 24
NUM_KV_HEADS = 4
G = NUM_Q_HEADS // NUM_KV_HEADS  # 6
D = 256
DEVICE = "cuda"
SOFTMAX_SCALE = D ** -0.5
FP8 = torch.float8_e4m3fn
LS = (5, 128, 8192)
MAX_ABS_TOL = 5e-2
MAX_REL_TOL = 5e-2
LSE_TOL = 2e-3


def fp32_ref(q, k, v, qd, kd, vd):
    """M=1 GQA reference, fp32. q: (H,D) bf16; k/v: (L,Hkv,D) fp8."""
    H, Hkv = NUM_Q_HEADS, NUM_KV_HEADS
    out = torch.empty(H, D, dtype=torch.float32, device=DEVICE)
    lse = torch.empty(H, dtype=torch.float32, device=DEVICE)
    qs = q.float() * qd.view(-1, 1)  # (H, D)
    ks = k.float() * kd.view(1, -1, 1)  # (L, Hkv, D)
    vs = v.float() * vd.view(1, -1, 1)
    for h in range(H):
        hkv = h // G
        s = (qs[h] @ ks[:, hkv].T) * SOFTMAX_SCALE
        p = torch.softmax(s, dim=-1)
        out[h] = p @ vs[:, hkv]
        lse[h] = torch.logsumexp(s, dim=-1)
    return out, lse


def run_gemv(q, k, v, qd, kd, vd, stages: int) -> tuple[torch.Tensor, torch.Tensor]:
    """One GEMV call with a forced ring depth (fresh compile cache)."""
    import vllm.vllm_flash_attn.cute.interface as iface
    from vllm.vllm_flash_attn.cute.sm100_hd256_decode_gemv import (
        BlackwellHd256DecodeGEMV,
    )

    os.environ["VLLM_FA4_HD256_GEMV"] = "1"
    os.environ["VLLM_FA4_HD256_GEMV_NUM_SPLITS"] = "1"
    if hasattr(iface._flash_attn_fwd, "gemv_compile_cache"):
        del iface._flash_attn_fwd.gemv_compile_cache
    if stages is None:
        # file default (16 after the fix)
        kernel_cls = BlackwellHd256DecodeGEMV
    else:
        orig = BlackwellHd256DecodeGEMV.__init__

        def patched(self, *a, **kw):
            orig(self, *a, **kw)
            self.stages = stages

        kernel_cls = BlackwellHd256DecodeGEMV
        BlackwellHd256DecodeGEMV.__init__ = patched
    try:
        out, lse, _, _ = iface._flash_attn_fwd(
            q.unsqueeze(0).unsqueeze(0),
            k.unsqueeze(0).contiguous(),
            v.unsqueeze(0).contiguous(),
            max_seqlen_q=1,
            max_seqlen_k=k.shape[0],
            softmax_scale=SOFTMAX_SCALE,
            causal=True,
            return_lse=True,
            q_descale=qd,
            k_descale=kd,
            v_descale=vd,
        )
    finally:
        if stages is not None:
            BlackwellHd256DecodeGEMV.__init__ = orig
    return out.squeeze(0).squeeze(0).float(), lse.squeeze(-1)


def main() -> int:
    # Init the CUDA context with a TINY footprint BEFORE the heavy CuTe-DSL
    # import (importing first + lazy init OOMs — see probe-gemv-ncu.py).
    _tiny = torch.zeros(1, device=DEVICE)
    torch.manual_seed(20260925)
    import vllm.vllm_flash_attn.cute.interface as iface

    # (3) auto-ns default check
    device = torch.device(DEVICE)
    arch = int(torch.cuda.get_device_capability()[0]) * 10 + int(
        torch.cuda.get_device_capability()[1]
    )
    ns_auto = iface._gemv_auto_num_splits(NUM_KV_HEADS, 1, device, arch)
    ns_auto_cap = min(max(1, ns_auto), 64, (LS[-1] + 127) // 128)
    print(f"[auto] _gemv_auto_num_splits(4,1) = {ns_auto} -> capped {ns_auto_cap} "
          f"at L=8192 (grid {NUM_KV_HEADS}x{ns_auto_cap}x1 = "
          f"{NUM_KV_HEADS * ns_auto_cap} CTAs)")

    q = torch.randn(NUM_Q_HEADS, D, device=DEVICE, dtype=torch.float32).to(torch.bfloat16)
    k = torch.randn(LS[-1], NUM_KV_HEADS, D, device=DEVICE, dtype=torch.float32).to(FP8)
    v = torch.randn(LS[-1], NUM_KV_HEADS, D, device=DEVICE, dtype=torch.float32).to(FP8)
    qd = torch.full((1, NUM_KV_HEADS), 1.06, dtype=torch.float32, device=DEVICE)
    kd = torch.full((1, NUM_KV_HEADS), 1.125, dtype=torch.float32, device=DEVICE)
    vd = torch.full((1, NUM_KV_HEADS), 1.25, dtype=torch.float32, device=DEVICE)
    qd_h = torch.full((NUM_Q_HEADS,), 1.06, dtype=torch.float32, device=DEVICE)

    results: list[bool] = []
    for L in LS:
        kk, vv = k[:L].contiguous(), v[:L].contiguous()
        ref, ref_lse = fp32_ref(q, kk, vv, qd_h, kd[0], vd[0])

        out16, lse16 = run_gemv(q, kk, vv, qd, kd, vd, None)
        d = float((out16 - ref).abs().max())
        rel = d / float(ref.abs().max())
        lerr = float((lse16 - ref_lse).abs().max())
        ok = d <= MAX_ABS_TOL and rel <= MAX_REL_TOL and lerr <= LSE_TOL
        results.append(ok)
        print(f"[{'OK ' if ok else 'BAD'}] L={L:5d} st16 vs fp32ref: "
              f"max_abs={d:.4e} rel={rel:.4e} lse_err={lerr:.4e}", flush=True)

        out2, _ = run_gemv(q, kk, vv, qd, kd, vd, 2)
        x = float((out16 - out2).abs().max())
        okx = x <= 1e-6
        results.append(okx)
        print(f"[{'OK ' if okx else 'BAD'}] L={L:5d} st16 vs st2 (pre-fix): "
              f"max_abs={x:.4e} (expect ~0 — stages change load timing only)",
              flush=True)

    n_ok = sum(results)
    print(f"\ncases passing: {n_ok}/{len(results)}")
    print(f"STAGES-VERIFY VERDICT: {'GO' if all(results) else 'NO-GO'}")
    return 0 if all(results) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001
        print("\nEXCEPTION (verbatim traceback):")
        print(traceback.format_exc())
        print("STAGES-VERIFY VERDICT: NO-GO")
        raise SystemExit(1)
