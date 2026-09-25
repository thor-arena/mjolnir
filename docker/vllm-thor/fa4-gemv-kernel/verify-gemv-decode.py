#!/usr/bin/env python3
"""GEMV decode Phase A — correctness verify on Jetson Thor (sm_110).

Verifies the new pure-FMA GEMV decode kernel
(``vllm.vllm_flash_attn.cute.sm100_hd256_decode_gemv``) against an fp32
reference, per docs/fa4-hd256-fp8/gemv-decode-design.md § 4.

Cases
-----
1. GEMV ON (VLLM_FA4_HD256_GEMV=1), bf16 Q + e4m3 KV + non-trivial descales:
   - dense, M=1, L in {1, 3, 5, 256, 512} x batch {1, 4}
     (L in {1,3,5} are not multiples of R=6 rows/stage -> tail-clamp path)
   - varlen, M=1, per-batch KV lengths {5, 1, 0} (the 0 exercises the
     kv_len==0 early exit)
2. GEMV ON, all-bf16 Q/KV (no descales) — exercises the 16-bit KV copy path.
3. GEMV OFF (env unset) — the untouched FA4 path must still be correct.
4. Cross-check: GEMV-ON output vs GEMV-OFF output (dense L=256).

Reference: dequant (q_descale/k_descale/v_descale per (batch, kv_head)),
scores in natural units, fp32 softmax, LSE = logsumexp in natural units.

Run inside the v11 container with the vfa tree mounted:
  docker run --rm --gpus all --network host --entrypoint python3 \
    -v $VFA_TREE:/usr/local/lib/python3.12/dist-packages/vllm/vllm_flash_attn \
    -v "$PWD":/p \
    mjolnir/vllm-thor:qwen38-sm110-v11 /p/verify-gemv-decode.py
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

# Phase A acceptance (gemv-decode-design.md § 4).
MAX_ABS_TOL = 5e-2
MAX_REL_TOL = 5e-2  # relative to ref_max
LSE_TOL = 2e-3


def _make_descales(batch: int):
    """Non-trivial per-(batch, kv_head) fp32 descales, in [1.0, 1.19] (like
    probe2a-fp8.py: qk_descale >= 1 keeps the softmax peaked)."""
    def pat(base: float, step: float, off: int) -> torch.Tensor:
        vals = [[base + step * (((b + g + off) % NUM_KV_HEADS) + 1)
                 for g in range(NUM_KV_HEADS)] for b in range(batch)]
        return torch.tensor(vals, dtype=torch.float32, device=DEVICE)
    return pat(1.0, 0.0625, 0), pat(1.0, 0.0625, 2), pat(1.0, 0.0625, 1)


def _rand(dtype: torch.dtype, *shape) -> torch.Tensor:
    """Random normal data cast to ``dtype`` (the exact codes the kernel reads)."""
    return torch.randn(*shape, device=DEVICE, dtype=torch.float32).to(dtype)


def _fp32_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kv_lens: list[int],
    q_descale: torch.Tensor | None,
    k_descale: torch.Tensor | None,
    v_descale: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """fp32 M=1 GQA reference. q: (B,1,H,D), k/v: (B,L_max,H_kv,D) with per-batch
    length kv_lens[b] (rows beyond the length are ignored). Returns (out, lse)
    with out: (B,1,H,D) fp32 and lse: (B,H) fp32 natural units."""
    batch = q.shape[0]
    out = torch.empty(batch, 1, NUM_Q_HEADS, D, dtype=torch.float32, device=DEVICE)
    lse = torch.empty(batch, NUM_Q_HEADS, dtype=torch.float32, device=DEVICE)
    for b in range(batch):
        L = kv_lens[b]
        qd = (
            q_descale[b].repeat_interleave(G).view(1, NUM_Q_HEADS, 1)
            if q_descale is not None
            else torch.ones(1, NUM_Q_HEADS, 1, device=DEVICE)
        )
        kd = (
            k_descale[b].view(1, NUM_KV_HEADS, 1)
            if k_descale is not None
            else torch.ones(1, NUM_KV_HEADS, 1, device=DEVICE)
        )
        vd = (
            v_descale[b].view(1, NUM_KV_HEADS, 1)
            if v_descale is not None
            else torch.ones(1, NUM_KV_HEADS, 1, device=DEVICE)
        )
        if L == 0:
            out[b] = 0.0
            lse[b] = float("-inf")
            continue
        qs = q[b, 0].float() * qd[0]  # (H, D)
        ks = k[b, :L].float() * kd   # (L, Hkv, D)
        vs = v[b, :L].float() * vd   # (L, Hkv, D)
        for h in range(NUM_Q_HEADS):
            hkv = h // G
            s = (qs[h] @ ks[:, hkv].T) * SOFTMAX_SCALE  # (L,) natural units
            p = torch.softmax(s, dim=-1)
            out[b, 0, h] = p @ vs[:, hkv]
            lse[b, h] = torch.logsumexp(s, dim=-1)
    return out, lse


def _run(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gemv: bool,
    cu_q: torch.Tensor | None,
    cu_k: torch.Tensor | None,
    max_seqlen_q: int,
    max_seqlen_k: int,
    descales: tuple[torch.Tensor | None, ...],
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Call the FA4 forward directly (same entry the FA4 backend uses). The
    top-level ``flash_attn_varlen_func`` wrapper is avoided because it
    requires seqused_k even for dense, which the GEMV gate rejects."""
    os.environ["VLLM_FA4_HD256_GEMV"] = "1" if gemv else "0"
    from vllm.vllm_flash_attn.cute.interface import _flash_attn_fwd

    out, lse, _, _ = _flash_attn_fwd(
        q,
        k,
        v,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        softmax_scale=SOFTMAX_SCALE,
        causal=True,
        return_lse=True,
        q_descale=descales[0],
        k_descale=descales[1],
        v_descale=descales[2],
    )
    return out, lse


def _report_row(name: str, ok: bool, max_abs: float, rel: float,
                lse_err: float | None) -> None:
    tag = "OK " if ok else "BAD"
    lse_s = f" lse_err={lse_err:.4e}" if lse_err is not None else ""
    print(f"[{tag}] {name:38s} max_abs={max_abs:.4e} rel={rel:.4e}"
          f" (tol {MAX_ABS_TOL:.0e}/{MAX_REL_TOL:.0e}){lse_s}", flush=True)


def main() -> int:
    if not torch.cuda.is_available():
        print("GEMV-VERIFY VERDICT: NO-GO (no CUDA device)")
        return 1
    props = torch.cuda.get_device_properties(0)
    print(f"device: {props.name!r} SMs={props.multi_processor_count}")
    print(f"geometry: GQA {NUM_Q_HEADS}/{NUM_KV_HEADS} (g={G}), hd={D}, M=1\n")

    results: list[bool] = []

    # ---------------- dense fp8-KV cases ------------------------------------
    descales = _make_descales(4)
    for L in (1, 3, 5, 256, 512):
        for batch in (1, 4):
            torch.manual_seed(777 + L * 13 + batch)
            q = _rand(torch.bfloat16, batch, 1, NUM_Q_HEADS, D)
            k = _rand(torch.float8_e4m3fn, batch, L, NUM_KV_HEADS, D)
            v = _rand(torch.float8_e4m3fn, batch, L, NUM_KV_HEADS, D)
            qd, kd, vd = (
                (descales[0][:batch], descales[1][:batch], descales[2][:batch])
                if v.dtype != torch.bfloat16
                else (None, None, None)
            )
            out, lse = _run(
                q, k, v, gemv=True, cu_q=None, cu_k=None,
                max_seqlen_q=1, max_seqlen_k=L, descales=(qd, kd, vd),
            )
            lse = lse.squeeze(-1)  # dense: (B, H, 1) -> (B, H)
            ref, ref_lse = _fp32_ref(q, k, v, [L] * batch, qd, kd, vd)
            d_abs = float((out.float() - ref).abs().max())
            ref_max = float(ref.abs().max())
            rel = d_abs / ref_max
            lse_err = float((lse - ref_lse).abs().max()) if L > 0 else 0.0
            ok = d_abs <= MAX_ABS_TOL and rel <= MAX_REL_TOL and lse_err <= LSE_TOL
            results.append(ok)
            _report_row(f"dense fp8KV L={L} B={batch}", ok, d_abs, rel, lse_err)

    # ---------------- varlen fp8-KV (incl. empty batch) -----------------------
    torch.manual_seed(4242)
    lens = [5, 1, 0]
    batch = len(lens)
    qd, kd, vd = _make_descales(batch)
    total_k = sum(lens)
    q = _rand(torch.bfloat16, batch, NUM_Q_HEADS, D)  # (total_q=3, H, D)
    k = _rand(torch.float8_e4m3fn, total_k, NUM_KV_HEADS, D)
    v = _rand(torch.float8_e4m3fn, total_k, NUM_KV_HEADS, D)
    cu_q = torch.tensor([0, 1, 2, 3], dtype=torch.int32, device=DEVICE)
    cu_k = torch.tensor([0] + list(torch.cumsum(torch.tensor(lens), 0).tolist()),
                        dtype=torch.int32, device=DEVICE)
    # Per-batch reference: de-page k/v by lens.
    k_b = [k[cu_k[b]:cu_k[b + 1]] for b in range(batch)]
    v_b = [v[cu_k[b]:cu_k[b + 1]] for b in range(batch)]
    k_ref = torch.stack(
        [x if x.shape[0] == max(lens) else torch.cat(
            [x, _rand(torch.float8_e4m3fn, max(lens) - x.shape[0], NUM_KV_HEADS, D)], 0)
         for x in k_b], 0)
    v_ref = torch.stack(
        [x if x.shape[0] == max(lens) else torch.cat(
            [x, _rand(torch.float8_e4m3fn, max(lens) - x.shape[0], NUM_KV_HEADS, D)], 0)
         for x in v_b], 0)
    out, lse = _run(
        q, k, v, gemv=True, cu_q=cu_q, cu_k=cu_k,
        max_seqlen_q=1, max_seqlen_k=max(lens), descales=(qd, kd, vd),
    )
    lse = lse.T  # varlen: (H, total_q) -> (B, H)
    ref, ref_lse = _fp32_ref(
        q.unsqueeze(1), k_ref, v_ref, lens, qd, kd, vd)
    d_abs = float((out.float() - ref.squeeze(1)).abs().max())
    ref_max = float(ref.abs().max())
    rel = d_abs / ref_max
    finite = torch.isfinite(ref_lse)
    lse_err = float((lse - ref_lse)[finite].abs().max()) if finite.any() else 0.0
    lse_inf_ok = bool(torch.isneginf(lse[~finite]).all()) if ~finite.all() else True
    ok = d_abs <= MAX_ABS_TOL and rel <= MAX_REL_TOL and lse_err <= LSE_TOL and lse_inf_ok
    results.append(ok)
    _report_row(f"varlen fp8KV lens={lens}", ok, d_abs, rel, lse_err)

    # ---------------- all-bf16 (no descale) ----------------------------------
    torch.manual_seed(99)
    for batch in (1, 2):
        q = _rand(torch.bfloat16, batch, 1, NUM_Q_HEADS, D)
        k = _rand(torch.bfloat16, batch, 64, NUM_KV_HEADS, D)
        v = _rand(torch.bfloat16, batch, 64, NUM_KV_HEADS, D)
        out, lse = _run(
            q, k, v, gemv=True, cu_q=None, cu_k=None,
            max_seqlen_q=1, max_seqlen_k=64, descales=(None, None, None),
        )
        lse = lse.squeeze(-1)
        ref, ref_lse = _fp32_ref(q, k, v, [64] * batch, None, None, None)
        d_abs = float((out.float() - ref).abs().max())
        ref_max = float(ref.abs().max())
        rel = d_abs / ref_max
        lse_err = float((lse - ref_lse).abs().max())
        ok = d_abs <= MAX_ABS_TOL and rel <= MAX_REL_TOL and lse_err <= LSE_TOL
        results.append(ok)
        _report_row(f"dense bf16 B={batch} L=64", ok, d_abs, rel, lse_err)

    # ---------------- GEMV OFF: untouched path still correct ------------------
    torch.manual_seed(31337)
    batch, L = 1, 256
    q = _rand(torch.float8_e4m3fn, batch, 1, NUM_Q_HEADS, D)
    k = _rand(torch.float8_e4m3fn, batch, L, NUM_KV_HEADS, D)
    v = _rand(torch.float8_e4m3fn, batch, L, NUM_KV_HEADS, D)
    qd, kd, vd = _make_descales(batch)
    out_off, lse_off = _run(
        q, k, v, gemv=False, cu_q=None, cu_k=None,
        max_seqlen_q=1, max_seqlen_k=L, descales=(qd, kd, vd),
    )
    lse_off = lse_off.squeeze(-1)
    ref, ref_lse = _fp32_ref(q, k, v, [L] * batch, qd, kd, vd)
    d_abs = float((out_off.float() - ref).abs().max())
    ref_max = float(ref.abs().max())
    rel = d_abs / ref_max
    lse_err = float((lse_off - ref_lse).abs().max())
    ok = d_abs <= MAX_ABS_TOL and rel <= MAX_REL_TOL and lse_err <= LSE_TOL
    results.append(ok)
    _report_row("GEMV-OFF (existing path) e4m3 L=256", ok, d_abs, rel, lse_err)

    # ---------------- cross-check GEMV-ON vs GEMV-OFF (same inputs) ----------
    out_on, _ = _run(
        q, k, v, gemv=True, cu_q=None, cu_k=None,
        max_seqlen_q=1, max_seqlen_k=L, descales=(qd, kd, vd),
    )
    x_abs = float((out_on.float() - out_off.float()).abs().max())
    ok = x_abs <= MAX_ABS_TOL
    results.append(ok)
    _report_row("GEMV-ON vs GEMV-OFF (e4m3 L=256)", ok, x_abs,
                x_abs / max(ref_max, 1e-6), None)

    n_ok = sum(results)
    print(f"\ncases passing: {n_ok}/{len(results)}")
    print(f"GEMV-VERIFY VERDICT: {'GO' if all(results) else 'NO-GO'}")
    return 0 if all(results) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001 — verbatim traceback, then NO-GO
        print("\nEXCEPTION (verbatim traceback):")
        print(traceback.format_exc())
        print("GEMV-VERIFY VERDICT: NO-GO")
        raise SystemExit(1)
