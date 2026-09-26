#!/usr/bin/env python3
"""GEMV decode Phase B1 — correctness verify at L=8192 on Jetson Thor (sm_110).

Verifies the Phase-B1 GEMV decode kernel extensions
(``vllm.vllm_flash_attn.cute.sm100_hd256_decode_gemv`` + the
``VLLM_FA4_HD256_GEMV`` / ``VLLM_FA4_HD256_GEMV_NUM_SPLITS`` interface knobs):

1. SplitKV: dense fp8 KV, M=1, L=8192, num_splits in {1, 2, 4, 8} — each leg vs
   an fp32 reference (out + final LSE), and vs the serial leg (ns=1) (the
   per-split LSE normalization moves the fp32 grid slightly — expect a small
   nonzero delta, same effect A1b measured ~3.8e-3).
2. Paged: paged-128 fp8 KV, ns in {1, 2, 4, 8}; paged-16 fp8 KV, ns in {1, 4}
   — page-table gather correctness at both page sizes.
3. Cross-check: GEMV vs FA4 1CTA (natural interface path, GEMV off) with the
   SAME inputs (guards a shared descale-convention bug); `out` only — the
   1CTA descale leg returns no LSE (the hd256 descale path does not yet
   produce descale-correct LSE — by-design kernel assert).
4. FlashInfer: BatchDecodeWithPagedKVCacheWrapper (block-16, fa2
   tensor-core, bf16 Q + fp8 KV) with the same KV dequant scales — the
   production v9-style leg.

Dtypes: Q=bf16, KV=e4m3, non-unity per-(batch, kv_head) descales
[1.06, 1.125, 1.25] (mirrors probe2a-fp8.py), softmax_scale = 1/sqrt(256).
Tolerances: Phase-A acceptance (max_abs <= 5e-2, rel <= 5e-2 vs ref_max,
LSE max_abs <= 2e-3); GEMV-vs-1CTA cross-check max_abs <= 2e-2.

Run inside the v11 container with the live vfa tree mounted:
  docker run --rm --gpus all --network host --entrypoint python3 \
    -v $VFA_TREE:/usr/local/lib/python3.12/dist-packages/vllm/vllm_flash_attn \
    -v "$PWD":/p \
    mjolnir/vllm-thor:qwen38-sm110-v11 /p/verify-gemv-b1.py
"""
from __future__ import annotations

import os
import traceback

import torch

NUM_Q_HEADS = 24
NUM_KV_HEADS = 4
G = NUM_Q_HEADS // NUM_KV_HEADS  # 6
D = 256
L = 8192
BLOCK16 = 16
DEVICE = "cuda"
SOFTMAX_SCALE = D ** -0.5

MAX_ABS_TOL = 5e-2
MAX_REL_TOL = 5e-2
LSE_TOL = 2e-3
XCHECK_TOL = 2e-2

DESCALE_Q = 1.06
DESCALE_K = 1.125
DESCALE_V = 1.25

# One shared dataset for every leg (same Q / KV values, different layouts).
Q_BF16: torch.Tensor  # (1, 1, H, D) dense
Q_FLAT: torch.Tensor  # (H, D)
K_FLAT: torch.Tensor  # (L, HK, D) fp8
V_FLAT: torch.Tensor  # (L, HK, D) fp8
QD: torch.Tensor
KD: torch.Tensor
VD: torch.Tensor
K_DENSE: torch.Tensor  # (1, L, HK, D)
V_DENSE: torch.Tensor
K_POOL128: torch.Tensor
V_POOL128: torch.Tensor
PT128: torch.Tensor
K_POOL16: torch.Tensor
V_POOL16: torch.Tensor
PT16: torch.Tensor
Q_FLAT8: torch.Tensor  # (H, D) fp8 — for the all-fp8 1CTA cross-check
Q_DENSE8: torch.Tensor  # (1, 1, H, D) fp8


def _rand(dtype: torch.dtype, *shape) -> torch.Tensor:
    """Random normal data cast to ``dtype`` (the exact codes the kernel reads)."""
    return torch.randn(*shape, device=DEVICE, dtype=torch.float32).to(dtype)


def _build_data() -> None:
    global Q_BF16, Q_FLAT, K_FLAT, V_FLAT, QD, KD, VD
    global K_DENSE, V_DENSE, K_POOL128, V_POOL128, PT128, K_POOL16, V_POOL16, PT16
    global Q_FLAT8, Q_DENSE8
    torch.manual_seed(20260925)
    Q_BF16 = _rand(torch.bfloat16, 1, 1, NUM_Q_HEADS, D)
    Q_FLAT = Q_BF16[0, 0].contiguous()
    Q_FLAT8 = _rand(torch.float8_e4m3fn, NUM_Q_HEADS, D)  # all-fp8 cross-check Q
    Q_DENSE8 = Q_FLAT8.unsqueeze(0).unsqueeze(0).contiguous()
    K_FLAT = _rand(torch.float8_e4m3fn, L, NUM_KV_HEADS, D)
    V_FLAT = _rand(torch.float8_e4m3fn, L, NUM_KV_HEADS, D)
    QD = torch.full((1, NUM_KV_HEADS), DESCALE_Q, dtype=torch.float32, device=DEVICE)
    KD = torch.full((1, NUM_KV_HEADS), DESCALE_K, dtype=torch.float32, device=DEVICE)
    VD = torch.full((1, NUM_KV_HEADS), DESCALE_V, dtype=torch.float32, device=DEVICE)
    K_DENSE = K_FLAT.unsqueeze(0).contiguous()
    V_DENSE = V_FLAT.unsqueeze(0).contiguous()
    K_POOL128 = K_FLAT.view(L // 128, 128, NUM_KV_HEADS, D).contiguous()
    V_POOL128 = V_FLAT.view(L // 128, 128, NUM_KV_HEADS, D).contiguous()
    PT128 = torch.arange(L // 128, dtype=torch.int32, device=DEVICE).unsqueeze(0)
    K_POOL16 = K_FLAT.view(L // BLOCK16, BLOCK16, NUM_KV_HEADS, D).contiguous()
    V_POOL16 = V_FLAT.view(L // BLOCK16, BLOCK16, NUM_KV_HEADS, D).contiguous()
    PT16 = torch.arange(L // BLOCK16, dtype=torch.int32, device=DEVICE).unsqueeze(0)


def _fp32_ref_q(q_flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """M=1, batch=1 GQA fp32 reference for a given Q. Returns (out (1,1,H,D), lse (H,))."""
    qs = q_flat.float() * QD[0].repeat_interleave(G).view(NUM_Q_HEADS, 1)
    ks = K_FLAT.float() * KD[0].view(1, NUM_KV_HEADS, 1)
    vs = V_FLAT.float() * VD[0].view(1, NUM_KV_HEADS, 1)
    out = torch.empty(1, 1, NUM_Q_HEADS, D, dtype=torch.float32, device=DEVICE)
    lse = torch.empty(NUM_Q_HEADS, dtype=torch.float32, device=DEVICE)
    for h in range(NUM_Q_HEADS):
        hkv = h // G
        s = (qs[h] @ ks[:, hkv].T) * SOFTMAX_SCALE  # natural units
        p = torch.softmax(s, dim=-1)
        out[0, 0, h] = p @ vs[:, hkv]
        lse[h] = torch.logsumexp(s, dim=-1)
    return out, lse


def _gemv_call(ns: int | None, page_size: int,
               q: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """GEMV leg: dense (page_size=0) or paged; ns=None -> auto plan.
    q defaults to the bf16 Q (mixed-dtype GEMV path)."""
    os.environ["VLLM_FA4_HD256_GEMV"] = "1"
    if ns is None:
        os.environ.pop("VLLM_FA4_HD256_GEMV_NUM_SPLITS", None)
    else:
        os.environ["VLLM_FA4_HD256_GEMV_NUM_SPLITS"] = str(ns)
    from vllm.vllm_flash_attn.cute.interface import _flash_attn_fwd

    if q is None:
        q = Q_BF16
    if page_size == 0:
        k_in, v_in, pt = K_DENSE, V_DENSE, None
    else:
        (k_in, v_in, pt) = (
            (K_POOL128, V_POOL128, PT128) if page_size == 128 else (K_POOL16, V_POOL16, PT16)
        )
    out, lse_out, _, _ = _flash_attn_fwd(
        q, k_in, v_in,
        page_table=pt,
        max_seqlen_q=1,
        max_seqlen_k=L,
        softmax_scale=SOFTMAX_SCALE,
        causal=True,
        return_lse=True,
        q_descale=QD,
        k_descale=KD,
        v_descale=VD,
    )
    return out, lse_out


def _fa4_1cta_call() -> tuple[torch.Tensor, None]:
    """FA4 1CTA (natural interface path, GEMV off) — paged-128 varlen, the
    decode-1cta-clean shape. Returns (out (1,H,D), None): the descale path
    does not return LSE (it does not yet produce descale-correct LSE —
    by-design kernel assert), so the cross-check compares `out` only."""
    os.environ["VLLM_FA4_HD256_GEMV"] = "0"
    from vllm.vllm_flash_attn import flash_attn_varlen_func

    cu_q = torch.tensor([0, 1], dtype=torch.int32, device=DEVICE)
    sku = torch.tensor([L], dtype=torch.int32, device=DEVICE)
    raw = flash_attn_varlen_func(
        q=Q_FLAT8.unsqueeze(0), k=K_POOL128, v=V_POOL128,
        max_seqlen_q=1, cu_seqlens_q=cu_q,
        max_seqlen_k=L, seqused_k=sku, block_table=PT128,
        softmax_scale=SOFTMAX_SCALE, causal=True, fa_version=4,
        q_descale=QD, k_descale=KD, v_descale=VD,
    )
    out = raw[0] if isinstance(raw, tuple) else raw
    return out[0], None


def _fi_call() -> torch.Tensor:
    """FlashInfer block-16 paged decode (fa2 tensor-core, bf16 Q + fp8 KV),
    same KV dequant scales as the FA4 legs. Returns out (1,H,D) bf16."""
    from flashinfer import BatchDecodeWithPagedKVCacheWrapper

    work = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=DEVICE)
    wrapper = BatchDecodeWithPagedKVCacheWrapper(
        work, kv_layout="NHD", use_cuda_graph=False,
        use_tensor_cores=True, backend="auto",
    )
    nblk = L // BLOCK16
    indptr = torch.tensor([0, nblk], dtype=torch.int32, device=DEVICE)
    indices = torch.arange(nblk, dtype=torch.int32, device=DEVICE)
    last_page_len = torch.full((1,), BLOCK16, dtype=torch.int32, device=DEVICE)
    wrapper.plan(
        indptr, indices, last_page_len,
        num_qo_heads=NUM_Q_HEADS, num_kv_heads=NUM_KV_HEADS,
        head_dim=D, page_size=BLOCK16,
        q_data_type=torch.bfloat16, kv_data_type=torch.float8_e4m3fn,
        o_data_type=torch.bfloat16, sm_scale=SOFTMAX_SCALE,
        q_len_per_req=1,
    )
    out = torch.empty(1, NUM_Q_HEADS, D, dtype=torch.bfloat16, device=DEVICE)
    wrapper.run(
        Q_FLAT.unsqueeze(0), (K_POOL16, V_POOL16),  # (q_len=1, H, D)
        q_scale=DESCALE_Q, k_scale=DESCALE_K, v_scale=DESCALE_V, out=out,
    )
    return out


def _report_row(name: str, ok: bool, max_abs: float, rel: float,
                lse_err: float | None) -> None:
    tag = "OK " if ok else "BAD"
    lse_s = f" lse_err={lse_err:.4e}" if lse_err is not None else ""
    print(f"[{tag}] {name:38s} max_abs={max_abs:.4e} rel={rel:.4e}"
          f" (tol {MAX_ABS_TOL:.0e}/{MAX_REL_TOL:.0e}){lse_s}", flush=True)


def _check(name: str, out: torch.Tensor, lse: torch.Tensor | None,
           ref: torch.Tensor, ref_lse: torch.Tensor, tol: float = MAX_ABS_TOL,
           expect_lse: bool = True) -> bool:
    out32 = out.float().reshape(1, 1, NUM_Q_HEADS, D) if out.dim() == 3 else out.float()
    d_abs = float((out32 - ref).abs().max())
    ref_max = float(ref.abs().max())
    rel = d_abs / max(ref_max, 1e-6)
    lse_err = None
    if expect_lse and lse is not None:
        lse_err = float((lse.reshape(NUM_Q_HEADS) - ref_lse).abs().max())
    ok = d_abs <= tol and rel <= MAX_REL_TOL and (lse_err is None or lse_err <= LSE_TOL)
    _report_row(name, ok, d_abs, rel, lse_err)
    return ok


def main() -> int:
    if not torch.cuda.is_available():
        print("GEMV-B1-VERIFY VERDICT: NO-GO (no CUDA device)")
        return 1
    props = torch.cuda.get_device_properties(0)
    print(f"device: {props.name!r} SMs={props.multi_processor_count}")
    print(f"geometry: GQA {NUM_Q_HEADS}/{NUM_KV_HEADS} (g={G}), hd={D}, M=1, L={L}\n")

    _build_data()
    ref, ref_lse = _fp32_ref_q(Q_FLAT)      # bf16-Q ref (main GEMV legs)
    ref8, ref_lse8 = _fp32_ref_q(Q_FLAT8)   # fp8-Q ref (all-fp8 1CTA cross-check)

    results: list[bool] = []

    # ---------------- GEMV dense, ns sweep -----------------------------------
    for ns in (1, 2, 4, 8):
        out, lse = _gemv_call(ns, page_size=0)
        results.append(_check(f"gemv dense ns={ns}", out, lse, ref, ref_lse))

    # ---------------- GEMV dense, auto plan ----------------------------------
    out, lse = _gemv_call(None, page_size=0)
    results.append(_check("gemv dense ns=auto", out, lse, ref, ref_lse))

    # ---------------- GEMV paged-128, ns sweep --------------------------------
    for ns in (1, 2, 4, 8):
        out, lse = _gemv_call(ns, page_size=128)
        results.append(_check(f"gemv paged-128 ns={ns}", out, lse, ref, ref_lse))

    # ---------------- GEMV paged-16, ns sweep ---------------------------------
    for ns in (1, 4):
        out, lse = _gemv_call(ns, page_size=16)
        results.append(_check(f"gemv paged-16 ns={ns}", out, lse, ref, ref_lse))

    # ---------------- cross-check: GEMV ns=1 vs GEMV ns=4 ---------------------
    out1, _ = _gemv_call(1, page_size=0)
    out4, _ = _gemv_call(4, page_size=0)
    x = float((out1.float() - out4.float()).abs().max())
    ok = x <= XCHECK_TOL
    results.append(ok)
    _report_row("gemv ns=1 vs ns=4 (dense)", ok, x, x / max(float(ref.abs().max()), 1e-6), None)

    # ---------------- cross-check: GEMV vs FA4 1CTA (same all-fp8 inputs) ------
    # Both legs run on the SAME fp8 Q/KV (the natural 1CTA path rejects mixed
    # bf16-Q/fp8-KV). Guards a shared descale/LSE-convention bug.
    out1, _ = _gemv_call(1, page_size=0, q=Q_DENSE8)
    out_c, _ = _fa4_1cta_call()
    x = float((out1.float() - out_c.float()).abs().max())
    ref8_max = max(float(ref8.abs().max()), 1e-6)
    ok = x <= XCHECK_TOL
    results.append(ok)
    _report_row("gemv fp8q vs FA4-1CTA (same inputs)", ok, x, x / ref8_max, None)

    # ---------------- FlashInfer leg ------------------------------------------
    try:
        out_fi = _fi_call()
        x = float((out_fi.float() - ref).abs().max())
        ref_max = float(ref.abs().max())
        ok = x <= MAX_ABS_TOL and (x / ref_max) <= MAX_REL_TOL
        results.append(ok)
        _report_row("flashinfer (block-16 paged)", ok, x, x / ref_max, None)
    except Exception as e:  # noqa: BLE001 — report, isolate
        print(f"[BAD] flashinfer (block-16 paged)       CALL FAILED: {type(e).__name__}: {e}")
        results.append(False)

    n_ok = sum(results)
    print(f"\ncases passing: {n_ok}/{len(results)}")
    print(f"GEMV-B1-VERIFY VERDICT: {'GO' if all(results) else 'NO-GO'}")
    return 0 if all(results) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001 — verbatim traceback, then NO-GO
        print("\nEXCEPTION (verbatim traceback):")
        print(traceback.format_exc())
        print("GEMV-B1-VERIFY VERDICT: NO-GO")
        raise SystemExit(1)
