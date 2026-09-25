#!/usr/bin/env python3
"""GPU verification of the Thor DSpark/DSpark non-causal FULL cudagraph patch.

The 8th overlay patch (thor-dspark-draft-fi-noncausal-cudagraph) makes DSpark's
non-causal, multi-token parallel draft run under a *captured* FULL decode
CUDA graph on FlashInfer instead of eagerly. Two coupled changes:

  * P1 -- get_cudagraph_support() grants UNIFORM_BATCH for a non-causal
    multi-token drafter, so the dflash speculator captures FULL draft graphs.
  * P2 -- build() routes the non-causal uniform multi-token batch through a
    per-batch-size, cuda-graph-mode prefill wrapper whose KV plan lives in
    persistent GPU buffers, so the captured run() replays against stable
    addresses while the draft plans eagerly outside the graph each step.

This test (run inside the nightly image with the PATCHED tree on PYTHONPATH)
proves the real, patched code is *correct*, not just that the symbols landed:

  T1  gate: get_cudagraph_support grants UNIFORM_BATCH for non-causal + a
      drafter, and stays UNIFORM_SINGLE_TOKEN_DECODE when there is no drafter
      (the safety gate that keeps the target model causal path unchanged).
  T2  routing: a non-causal uniform multi-token batch is routed through the
      per-batch-size cuda-graph prefill wrapper (cuda-graph mode, per-batch
      identity), while a causal batch still uses the eager wrapper.
  T3  correctness: the routed wrapper's run() matches an independent non-causal
      SDPA reference for ragged, multi-page context lengths.
  T4  capture-safety: capture run() in a CUDA graph, re-plan with DIFFERENT
      ragged lengths (same batch size, the real per-step pattern), and replay --
      the output must still match the reference. This is the load-bearing
      guarantee that the FULL draft graph does not corrupt attention.
  T5  determinism: 8 repeats of the captured replay with a fixed re-plan.

It self-skips (exit 0) on no-GPU / transient OOM so a plain (non --gpus) build
passes; run with `--gpus all` to exercise it.
"""
from __future__ import annotations

import sys
import time

import torch

from vllm.v1.attention.backends.flashinfer import (
    FlashInferMetadataBuilder,
    AttentionCGSupport,
    BatchPrefillWithPagedKVCacheWrapper,
)
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import AttentionSpec
from vllm.v1.kv_cache_layout import KVCacheLayout

import types

# torch 2.13 raises `torch.AcceleratorError` (a RuntimeError subclass) on OOM;
# older/newer versions use `torch.cuda.OutOfMemoryError`.
_ALLOC_ERRS: tuple[type, ...] = tuple(
    filter(None, [getattr(torch, "AcceleratorError", None)]
    + [getattr(getattr(torch, "cuda", None), "OutOfMemoryError", None)])
) or (RuntimeError,)
_ALLOC_RETRIES = 8
_ALLOC_RETRY_SLEEP = 5.0


def _gpu_ok() -> bool:
    if not torch.cuda.is_available():
        return False
    for _ in range(_ALLOC_RETRIES):
        try:
            x = torch.zeros(512, 512, device="cuda", dtype=torch.bfloat16)
            del x
            return True
        except _ALLOC_ERRS:
            torch.cuda.empty_cache()
            time.sleep(_ALLOC_RETRY_SLEEP)
    return False


# Geometry mirroring the DSpark drafter draft on Qwen3.8 (GQA, fp8 KV).
PAGE = 16
HS = 5120
NH, NKV, HD = 8, 4, 64
Q = 7  # num_query_per_req (K=7 drafter)


def _spec() -> AttentionSpec:
    return AttentionSpec(block_size=PAGE, num_kv_heads=NKV, head_size=HD,
                         dtype=torch.bfloat16)


def _gate_cfg(non_causal: bool, has_spec: bool):
    return types.SimpleNamespace(
        attention_config=types.SimpleNamespace(use_non_causal=non_causal),
        parallel_config=types.SimpleNamespace(decode_context_parallel_size=1),
        model_config=types.SimpleNamespace(
            get_num_attention_heads=lambda pc: NH),
        speculative_config=(types.SimpleNamespace(num_speculative_tokens=Q)
                            if has_spec else None),
    )


def _gate() -> None:
    import vllm.v1.attention.backends.flashinfer as fi
    orig = fi.can_use_trtllm_attention
    fi.can_use_trtllm_attention = lambda **kw: False
    try:
        assert FlashInferMetadataBuilder.get_cudagraph_support(
            _gate_cfg(True, True), _spec()) == AttentionCGSupport.UNIFORM_BATCH
        assert FlashInferMetadataBuilder.get_cudagraph_support(
            _gate_cfg(True, False), _spec(
        )) == AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
    finally:
        fi.can_use_trtllm_attention = orig
    print("PASS T1 gate (non-causal+drafter->UNIFORM_BATCH; "
          "no-drafter->SINGLE_TOKEN)")


def _build_builder():
    b = FlashInferMetadataBuilder.__new__(FlashInferMetadataBuilder)
    dev = torch.device("cuda")
    b.device = dev
    b.kv_cache_spec = _spec()
    b.page_size = PAGE
    b.num_qo_heads = NH
    b.num_kv_heads = NKV
    b.head_dim = HD
    b.dcp_world_size = 1
    b.dcp_rank = 0
    b.use_dcp = False
    b.use_dedicated_xqa = False
    b.use_trtllm_decode_attention = False
    b.flashinfer_trtllm_api_decode_kernel = None
    b.use_trtllm_gen_varlen_decode = False
    b.enable_cuda_graph = True
    b._decode_cudagraph_max_bs = 1024
    b.enable_full_decode_cudagraph = True
    b.is_kvcache_nvfp4 = False
    b.cache_dtype = "fp8"
    b.kv_cache_dtype = torch.bfloat16
    b.q_data_type_prefill = torch.bfloat16
    b.q_data_type_decode = torch.bfloat16
    b.sm_scale = 1.0 / (HD ** 0.5)
    b.window_left = -1
    b.logits_soft_cap = 0.0
    b.has_sinks = False
    b.global_hyperparameters = types.SimpleNamespace(
        has_sinks=False, has_same_window_lefts=True, has_same_all_params=True)
    b.cache_config = types.SimpleNamespace(
        get_resolved_kv_cache_layout=lambda: KVCacheLayout.LBNHC)
    b.model_config = types.SimpleNamespace(dtype=torch.bfloat16)
    b.attention_config = types.SimpleNamespace(
        use_non_causal=True, use_trtllm_attention=None,
        disable_flashinfer_q_quantization=False)
    b.compilation_config = types.SimpleNamespace(max_cudagraph_capture_size=None)
    b.max_num_reqs = 16
    b._workspace_buffer = None
    b._prefill_wrapper = None
    b._noncausal_prefill_wrapper = None
    b._decode_wrapper = None
    b._decode_wrappers_cudagraph = {}
    b._noncausal_prefill_wrappers_cudagraph = {}
    b._decode_mask_cache = {}
    # Real CpuGpuBuffer keeps .cpu (torch) and .np (numpy) as views of the
    # same pinned memory; _compute_flashinfer_kv_metadata writes .np and
    # build() reads .cpu to hand to the wrapper, so they must stay in sync.
    import numpy as _np
    _ind_np = _np.zeros(64, dtype=_np.int32)
    _ind_cpu = torch.from_numpy(_ind_np)
    b.paged_kv_indptr = type("B", (), {
        "gpu": torch.zeros(64, dtype=torch.int32, device=dev),
        "cpu": _ind_cpu,
        "np": _ind_np,
    })()
    _lpl_np = _np.zeros(64, dtype=_np.int32)
    _lpl_cpu = torch.from_numpy(_lpl_np)
    b.paged_kv_last_page_len = type("B", (), {
        "gpu": torch.zeros(64, dtype=torch.int32, device=dev),
        "cpu": _lpl_cpu,
        "np": _lpl_np,
    })()
    b.paged_kv_indices = torch.zeros(1024, dtype=torch.int32, device=dev)
    b._nc_prefill_qo_indptr_buf = torch.zeros(
        32, dtype=torch.int32, device=dev)
    b._draft_seq_lens_gpu = None
    b._draft_block_table = None
    b.max_num_batched_tokens = 4096
    b.prefill_fixed_split_size = -1
    b.decode_fixed_split_size = -1
    b.disable_split_kv = False
    return b


def _common(num_reqs: int, ctx: list[int], q_len: int):
    dev = torch.device("cuda")
    max_blocks = max(c // PAGE + (1 if c % PAGE else 0) for c in ctx) + 1
    qsl = torch.arange(num_reqs + 1, dtype=torch.int32, device=dev) * q_len
    block_table_tensor = torch.zeros(
        num_reqs, max_blocks, dtype=torch.int32, device=dev)
    for r in range(num_reqs):
        base = r * 32
        n = ctx[r] // PAGE + (1 if ctx[r] % PAGE else 0)
        block_table_tensor[r, :n] = torch.arange(
            base, base + n, dtype=torch.int32, device=dev)
    return CommonAttentionMetadata(
        query_start_loc=qsl,
        query_start_loc_cpu=qsl.cpu().clone(),
        seq_lens=torch.tensor(ctx, dtype=torch.int32, device=dev),
        seq_lens_cpu_upper_bound=torch.tensor(
            ctx, dtype=torch.int32, device="cpu"),
        num_reqs=num_reqs,
        num_actual_tokens=num_reqs * q_len,
        max_query_len=q_len,
        max_seq_len=max(ctx),
        block_table_tensor=block_table_tensor,
        slot_mapping=torch.zeros(num_reqs * q_len, dtype=torch.int64,
                                 device=dev),
        causal=False,
    )


def _ref_noncausal(k, v, q, ctx, block_table_tensor) -> torch.Tensor:
    g = NH // NKV
    out = []
    for r in range(len(ctx)):
        L = ctx[r]
        # Page 0 is a *valid* page id (a zero in the padded block table is not),
        # so slice by the page count instead of filtering out zeros.
        n = L // PAGE + (1 if L % PAGE else 0)
        pgs = [int(x) for x in block_table_tensor[r, :n].tolist()]
        kr = k[pgs].reshape(-1, NKV, HD)[:L].permute(1, 0, 2).float()
        vr = v[pgs].reshape(-1, NKV, HD)[:L].permute(1, 0, 2).float()
        kr_rep = torch.stack([kr[i // g] for i in range(NH)])
        vr_rep = torch.stack([vr[i // g] for i in range(NH)])
        qr = q[r * Q:(r + 1) * Q].float()
        s = (qr.permute(1, 0, 2) @ kr_rep.transpose(1, 2)) / (HD ** 0.5)
        s = s.softmax(-1)
        o = s @ vr_rep
        out.append(o.to(torch.bfloat16).permute(1, 0, 2).reshape(Q, NH, HD))
    return torch.cat(out, 0)


def _kv_pages(num_pages: int):
    dev = torch.device("cuda")
    return (torch.randn(num_pages, PAGE, NKV, HD, device=dev,
                         dtype=torch.bfloat16),
            torch.randn(num_pages, PAGE, NKV, HD, device=dev,
                        dtype=torch.bfloat16))


def main() -> int:
    _gate()  # T1 needs no big GPU alloc; run it first

    if not _gpu_ok():
        print("kernel test dspark-noncausal: SKIPPED (no GPU / "
              "OOM-saturated). Re-run the build with --gpus all to exercise "
              "the non-causal cudagraph path.", flush=True)
        return 0

    import vllm.v1.attention.backends.flashinfer as fi
    orig = fi.can_use_trtllm_attention
    fi.can_use_trtllm_attention = lambda **kw: False
    try:
        b = _build_builder()
        dev = torch.device("cuda")
        # ---- T2 routing ----
        ctx_a = [40, 33]
        k, v = _kv_pages(32 * 2)
        common = _common(2, ctx_a, Q)
        meta = b.build(common_prefix_len=0, common_attn_metadata=common)
        assert meta.num_prefills == 2 and not meta.causal
        w = meta.prefill.wrapper
        assert isinstance(w, BatchPrefillWithPagedKVCacheWrapper)
        assert w.is_cuda_graph_enabled, "non-causal must use cuda-graph wrapper"
        assert w._fixed_batch_size == 2
        assert any(x is w for x in
                   b._noncausal_prefill_wrappers_cudagraph.values())
        # a causal batch must NOT use the non-causal cudagraph wrapper
        causal_common = _common(2, ctx_a, Q)
        causal_common.causal = True
        cmeta = b.build(common_prefix_len=0, common_attn_metadata=causal_common)
        cw = getattr(cmeta.prefill, "wrapper", None)
        assert cw is not w, "causal batch must not reuse the non-causal wrapper"
        assert not (cw is not None and any(x is cw for x in
                    b._noncausal_prefill_wrappers_cudagraph.values()))
        print("PASS T2 routing (non-causal->cudagraph wrapper B=2; "
              "causal->eager wrapper)")

        # ---- T3 correctness vs reference ----
        q = torch.randn(2 * Q, NH, HD, device=dev, dtype=torch.bfloat16)
        out = w.run(q=q, paged_kv_cache=(k, v)).reshape(2 * Q, NH, HD)
        ref = _ref_noncausal(k, v, q, ctx_a, common.block_table_tensor)
        d = (out.float() - ref.float()).abs().max().item()
        assert d < 0.1, f"T3 non-causal mismatch {d}"
        print(f"PASS T3 non-causal run vs ref (max|d|={d:.4f})")

        # ---- T4 capture-safety across ragged re-plans ----
        def plan_with(ctx):
            c = _common(2, ctx, Q)
            m = b.build(common_prefix_len=0, common_attn_metadata=c)
            return m, c

        # plan a fresh wrapper with ctx_a, capture run, then replay after
        # re-planning with DIFFERENT lengths (same batch size).
        _, ca = plan_with(ctx_a)
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            oc = w.run(q=q, paged_kv_cache=(k, v))
        ctx_b = [57, 24]  # different ragged lengths, same B
        plan_with(ctx_b)
        gr.replay()
        out_b = oc.reshape(2 * Q, NH, HD)
        # ctx_b pages were re-planned into the same persistent buffers; the
        # reference must use ctx_b's block table (same per-req page bases).
        ref_b = _ref_noncausal(k, v, q, ctx_b, plan_with(ctx_b)[1].block_table_tensor)
        db = (out_b.float() - ref_b.float()).abs().max().item()
        assert db < 0.1, f"T4 capture-safety mismatch {db}"
        print(f"PASS T4 capture-replay across ragged lengths (max|d|={db:.4f})")

        # ---- T5 determinism ----
        ctx_c = [20, 88]
        plan_with(ctx_c)
        gr.replay()
        first = oc.clone()
        ok = True
        for _ in range(8):
            plan_with(ctx_c)
            gr.replay()
            if (oc.reshape(2 * Q, NH, HD).float()
                    - first.float()).abs().max().item() > 1e-3:
                ok = False
                break
        assert ok
        print("PASS T5 determinism (8 repeats identical)")
    finally:
        fi.can_use_trtllm_attention = orig

    print("kernel test dspark-noncausal: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
