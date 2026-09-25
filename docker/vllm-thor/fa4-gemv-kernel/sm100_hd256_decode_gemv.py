"""Pure-FMA GEMV decode kernel for head_dim=256, M=1 (Phase A + B1).

CuTe-DSL port of the FlashInfer fast-decode GEMV kernel
(``flashinfer/attention/decode.cuh``, ``use_tensor_cores=False`` path) —
see ``docs/fa4-hd256-fp8/gemv-decode-design.md`` and
``docs/fa4-hd256-fp8/fi-decode-gemv-analysis.md``.

Architecture (M=1, dense/varlen/paged KV, SplitKV-capable since Phase B1):

- grid  : ``(num_kv_heads, num_splits, batch_size)`` — one CTA per
  (kv_head, kv-chunk, batch). ``num_splits = 1`` is the Phase-A serial
  form (write-through, no partials).
- block : ``16 * g`` threads (``g = qhead_per_kvhead``); ``tx = tidx % 16``
  is the 16-wide head-dim slice, ``ty = tidx // 16`` the q-head in the GQA
  group. Each 16-thread group is exactly one half-warp, so a 4-step
  butterfly shfl (offsets 1,2,4,8) reduces a 256-dot score within the group.
- state : per thread ``(m, l, O[16] fp32)`` in registers; scores are made
  group-uniform by the shfl reduce, so the epilogue needs no cross-thread
  merge — each thread writes its own 16-wide d-slice of the output row.
- QK    : 16-element FMA dot per thread + butterfly reduce, log2-domain
  online softmax (``exp2``), ``qk_descale`` folded into the score scale.
- PV    : FMA ``O += p * V``; ``v_descale`` applied once in the epilogue.
- KV    : cp.async 128-bit into an S-stage smem ring (S=16; S=2 is the
  original double buffer — see ``docs/gemv-ring-fix-bench.md``), K and V
  waited separately per iteration (FlashInfer consumer-loop order).

SplitKV (Phase B1, FlashInfer Alg.1 — scheduler.cuh): with
``num_splits > 1`` each CTA scans its chunk
``[s·ceil(kv_len/num_splits), min((s+1)·ceil(...), kv_len))`` (empty chunks
past the end take the ``kv_len == 0`` path) and emits a **chunk-normalized**
partial: ``O_partial[split] = o·v_descale / l`` (fp32) and
``LSE_partial[split] = (m + log2 l)·ln2`` (natural units). The host then
merges the partials with the in-tree ``_flash_attn_fwd_combine`` (the A1b
verified ⊕-merge): ``O = Σ_s exp(lse_s − L)·O_s``, ``L = max_s lse_s +
ln Σ_s exp(lse_s − max)``. With ``num_splits = 1`` the Phase-A epilogue
writes the final O directly (write-through, no partials/combine —
FlashInfer App. D.2).

Paged KV (Phase B1): ``page_size`` (constructor constant) is 0 for dense or
16/128 for the paged pool layout ``(num_pages, page_size, H_kv, D)``. Each
row is gathered via the int32 page table:
``page = row // page_size``, ``off = row − page·page_size``,
``phys = mPageTable[batch, page]·page_size + off`` (decode.cuh's page-table
walk; K and V of a row share the page). Tail rows are clamped to the last
valid row BEFORE the page lookup, so every cp.async stays in-bounds (the
Phase-A clamp semantics) and clamped rows are masked in compute.

Tail handling: gmem row indices are *clamped* to ``kv_len - 1`` so every
cp.async stays in-bounds and commit-group counts stay uniform (no
predicated loads); clamped rows are masked to ``-inf`` in compute (K data
unused; V data finite, weighted by ``p = 0``). A batch (chunk) with
``kv_len == 0`` writes ``O = 0`` / ``LSE = -inf``.

DSL notes: no nested functions and no early exits — the base-DSL
preprocessor rejects closure captures inside staged if/for bodies, so all
addressing is inlined. ``num_splits`` is a runtime value (the grid's y
dimension), so one split binary serves every ns; the partial-tensor
presence (``mO_partial is not None``) is the compile-time split flag.
"""

from typing import Optional

import math

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr
from cutlass.cute.nvgpu import cpasync

from vllm.vllm_flash_attn.cute.flash_fwd_sm100 import DescaleTensors


class BlackwellHd256DecodeGEMV:
    """GEMV decode (M=1) kernel for head_dim=256 on SM100/SM110.

    Q dtype in {bf16, fp16, e4m3}; KV dtype in {bf16, fp16, e4m3}.
    Output dtype is taken from the caller-provided ``mO`` tensor (final,
    non-split) or ``mO_partial`` (fp32, split mode).

    ``page_size`` (constructor constant): 0 = dense KV; 16 or 128 = paged
    pool ``K/V: (num_pages, page_size, H_kv, D)`` gathered through
    ``mPageTable``: ``(batch, max_pages)`` int32.
    """

    HEAD_DIM = 256
    VEC = 16  # elements per thread (128-bit for fp8 KV; 2x128-bit for 16-bit KV)

    def __init__(self, qhead_per_kvhead: int, tile_size_per_bdx: int = 1, page_size: int = 0):
        assert qhead_per_kvhead >= 1, "qhead_per_kvhead must be >= 1"
        assert self.HEAD_DIM % (16 * self.VEC) == 0
        assert page_size in (0, 16, 128), "page_size must be 0 (dense), 16, or 128"
        self.g = qhead_per_kvhead
        self.page_size = page_size
        self.bdx = self.HEAD_DIM // self.VEC  # 16 threads per KV row
        self.bdy = qhead_per_kvhead  # one q-head per (ty) thread
        self.tile = tile_size_per_bdx  # rows per (ty) thread (Phase A: 1)
        # KV ring depth (in-flight stages). S=16 -> S*R*D elems per K and V
        # (g=6: 16*6*256 = 24 KiB each, 48 KiB total) — deep enough to hide
        # LPDDR/L2 latency of the strided per-row access; the rolling
        # cp_async_wait_group(2*S-1) loop and the S-stage prologue preload
        # are S-agnostic (verified: only this value + smem shapes change).
        self.stages = 16
        self.rows_per_stage = self.bdy * self.tile  # R
        self.num_threads = self.bdx * self.bdy

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor] = None,
        mCuSeqlensQ: Optional[cute.Tensor] = None,
        mCuSeqlensK: Optional[cute.Tensor] = None,
        mDescale: Optional[DescaleTensors] = None,
        mO_partial: Optional[cute.Tensor] = None,
        mLSE_partial: Optional[cute.Tensor] = None,
        mPageTable: Optional[cute.Tensor] = None,
        seqlen_q: Int32 = 1,
        seqlen_k: Int32 = 0,
        softmax_scale_log2: Float32 = 0.0,
        num_splits: Int32 = 1,
        # Always keep stream as the last parameter (EnvStream: obtained
        # implicitly via TVM FFI).
        stream: cuda.CUstream = None,
    ):
        is_varlen_q = const_expr(mCuSeqlensQ is not None)
        if const_expr(is_varlen_q):
            batch_size = mCuSeqlensQ.shape[0] - 1
        else:
            batch_size = mQ.shape[0]
        num_kv_heads = mV.shape[-2]
        assert const_expr((mO_partial is None) == (mLSE_partial is None))
        self.kernel(
            mQ,
            mK,
            mV,
            mO,
            mLSE,
            mCuSeqlensQ,
            mCuSeqlensK,
            mDescale,
            mO_partial,
            mLSE_partial,
            mPageTable,
            seqlen_q,
            seqlen_k,
            softmax_scale_log2,
            num_splits,
        ).launch(
            grid=[num_kv_heads, num_splits, batch_size],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        mCuSeqlensQ: Optional[cute.Tensor],
        mCuSeqlensK: Optional[cute.Tensor],
        mDescale: Optional[DescaleTensors],
        mO_partial: Optional[cute.Tensor],
        mLSE_partial: Optional[cute.Tensor],
        mPageTable: Optional[cute.Tensor],
        seqlen_q: Int32,
        seqlen_k: Int32,
        softmax_scale_log2: Float32,
        num_splits: Int32,
    ):
        D = self.HEAD_DIM
        VEC = self.VEC
        BDX = self.bdx
        TILE = self.tile
        S = self.stages
        R = self.rows_per_stage
        PS = self.page_size

        IS_SPLIT = const_expr(mO_partial is not None)
        PAGED = const_expr(mPageTable is not None)

        kv_head = cute.arch.block_idx()[0]
        split_idx = cute.arch.block_idx()[1]
        batch_idx = cute.arch.block_idx()[2]
        tidx = cute.arch.thread_idx()[0]
        tx = tidx % BDX
        ty = tidx // BDX
        qo_head = kv_head * self.g + ty

        is_varlen_q = const_expr(mCuSeqlensQ is not None)
        is_varlen_k = const_expr(mCuSeqlensK is not None)

        # Per-batch bases / lengths.
        if const_expr(is_varlen_q):
            num_head = mQ.shape[1]
            q_row_base = Int32(mCuSeqlensQ[batch_idx])  # global row of the query
        else:
            num_head = mQ.shape[2]
            q_row_base = batch_idx * seqlen_q
        if const_expr(is_varlen_k):
            kv_row0 = Int32(mCuSeqlensK[batch_idx])  # global KV row of the batch
            kv_len = Int32(mCuSeqlensK[batch_idx + 1]) - kv_row0
        else:
            kv_row0 = batch_idx * seqlen_k
            kv_len = seqlen_k
        kv_row_stride = mV.shape[-2] * D  # elements per KV row (all heads)
        num_head_kv = mV.shape[-2]

        # ---- SplitKV: this CTA's KV chunk (identity at num_splits=1) --------
        kv_chunk = (kv_len + num_splits - 1) // num_splits
        kv_start = split_idx * kv_chunk
        # Cap by the chunk length: without the min(), every non-last split
        # over-scans to the end of the sequence (kv_len - kv_start > kv_chunk),
        # double-counting rows. ns=1 is unaffected (kv_chunk == kv_len).
        kv_len_c = cutlass.min(kv_chunk, cutlass.max(Int32(0), kv_len - kv_start))

        # Per-(batch, kv_head) descales (identity when absent). Index with a
        # 2D tuple coord, NOT a flat scalar: the descale memref is
        # (batch, H_kv) and a scalar coord into a multi-rank tensor does not
        # land at the flat offset (same class of bug as the LSE coord).
        qk_descale = Float32(1.0)
        v_descale = Float32(1.0)
        if const_expr(mDescale is not None):
            if const_expr(mDescale.q_descale is not None):
                qk_descale = qk_descale * Float32(
                    mDescale.q_descale[batch_idx, kv_head]
                )
            if const_expr(mDescale.k_descale is not None):
                qk_descale = qk_descale * Float32(
                    mDescale.k_descale[batch_idx, kv_head]
                )
            if const_expr(mDescale.v_descale is not None):
                v_descale = Float32(mDescale.v_descale[batch_idx, kv_head])
        scale_eff = softmax_scale_log2 * qk_descale

        # Q/O row offsets (shared by both branches below). In split mode the
        # O/LSE epilogue targets mO_partial/mLSE_partial at the same flat
        # row offset plus the split block (num_splits * rows * H * D elems).
        q_off = q_row_base * num_head * D + qo_head * D + tx * VEC
        o_off = q_row_base * num_head * D + qo_head * D + tx * VEC
        if const_expr(IS_SPLIT):
            # Stride of one split block = number of elements in mO_partial with
            # its leading (split) mode removed. NOT shape[0]*shape[1]*... (that
            # folds the split count into the stride and over-strides by
            # num_splits). Dense (ns,B,Sq,H,D) -> B*Sq*H*D;
            # varlen (ns,total_q,H,D) -> total_q*H*D.
            if const_expr(is_varlen_q):
                split_stride = mO_partial.shape[1] * num_head * D
            else:
                split_stride = mO_partial.shape[1] * mO_partial.shape[2] * num_head * D
            o_off = o_off + split_idx * split_stride
        # LSE coord (NOT a flat scalar — the DSL's scalar coord into a
        # multi-rank memref does not land where the flat offset says).
        # Dense: mLSE is (B, H, S):(H*S, S, 1) -> (batch, head, seqlen=0).
        # Varlen: mLSE is (H, total_q):(total_q, 1) -> (head, q_row_base).
        # Partial dense: (S, B, H, Sq) -> (split, batch, head, 0);
        # partial varlen: (S, H, total_q) -> (split, head, q_row_base).
        if const_expr(mLSE is not None or mLSE_partial is not None):
            if const_expr(IS_SPLIT):
                if const_expr(is_varlen_q):
                    lse_c = (split_idx, qo_head, q_row_base)
                else:
                    lse_c = (split_idx, batch_idx, qo_head, 0)
            else:
                if const_expr(is_varlen_q):
                    lse_c = (qo_head, q_row_base)
                else:
                    lse_c = (batch_idx, qo_head, 0)

        # ---- smem: 2-stage double buffer, K and V (flat, row-major) -------
        smem = cutlass.utils.SmemAllocator()
        kv_elems = S * R * D
        k_smem = smem.allocate_tensor(
            mK.element_type, cute.make_layout(kv_elems), byte_alignment=16
        )
        v_smem = smem.allocate_tensor(
            mV.element_type, cute.make_layout(kv_elems), byte_alignment=16
        )

        k_g2s_atom = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            mK.element_type,
            num_bits_per_copy=128,
        )
        v_g2s_atom = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            mV.element_type,
            num_bits_per_copy=128,
        )
        s2r_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), mK.element_type, num_bits_per_copy=128
        )
        v_s2r_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), mV.element_type, num_bits_per_copy=128
        )
        q_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), mQ.element_type, num_bits_per_copy=128
        )
        # O target dtype: the final O (mO) or the fp32 partial (mO_partial).
        # A 128-bit copy moves 128//width elements (4 for fp32 partials).
        o_elem = mO_partial.element_type if const_expr(IS_SPLIT) else mO.element_type
        o_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), o_elem, num_bits_per_copy=128
        )

        # Each copy atom moves 128 bits per instruction. For 16-bit dtypes
        # that is 8 elements, so a 16-element VEC tile needs N_SUB sub-copies
        # (8-bit: EPC=16 -> N_SUB=1, a single copy as before). The DSL does
        # NOT auto-tile an atom across a larger fragment (verified: a 128-bit
        # atom over a 16-elem bf16 tile copies only the first 8 elements).
        EPC_K = 16 if mK.element_type.width == 8 else 8
        N_SUB_K = VEC // EPC_K
        EPC_V = 16 if mV.element_type.width == 8 else 8
        N_SUB_V = VEC // EPC_V
        EPC_Q = 16 if mQ.element_type.width == 8 else 8
        N_SUB_Q = VEC // EPC_Q
        EPC_O = 128 // o_elem.width
        N_SUB_O = VEC // EPC_O

        # ---- gmem row offset for a clamped local row -----------------------
        # Dense: batch row (kv_row0 + kv_start + crow) in the (B, L) row space.
        # Paged: sequence row (kv_start + crow) -> page table -> pool row.
        # Both must include the split's chunk start (kv_start); ns=1 has
        # kv_start=0 so the omission was invisible in Phase A.
        # (Inlined at each of the 4 cp.async sites — no nested helpers.)
        # ---- empty sequence / empty chunk: O = 0, LSE = -inf ----------------
        # No early exit (the DSL forbids it in kernels) — branch instead.
        if kv_len_c == 0:
            o_frag = cute.make_rmem_tensor(VEC, o_elem)
            o_frag.fill(0.0)
            for sub in cutlass.range_constexpr(N_SUB_O):
                o_sub = cute.make_rmem_tensor(EPC_O, o_elem)
                for i in cutlass.range_constexpr(EPC_O):
                    o_sub[i] = o_frag[sub * EPC_O + i]
                o_dst = cute.make_tensor(
                    cute.make_ptr(
                        o_elem,
                        (
                            (mO_partial.iterator if const_expr(IS_SPLIT) else mO.iterator)
                            + o_off
                            + sub * EPC_O
                        ).toint(),
                        cute.AddressSpace.gmem,
                        assumed_align=16,
                    ),
                    cute.make_layout(EPC_O),
                )
                cute.copy(o_atom, o_sub, o_dst)
            if tx == 0:
                if const_expr(mLSE is not None or mLSE_partial is not None):
                    lse_tensor = (
                        mLSE_partial if const_expr(IS_SPLIT) else mLSE
                    )
                    lse_tensor[lse_c] = -Float32.inf
        else:
            # ---- Q: load this thread's d-slice of its q-head ---------------
            q_f32 = cute.make_rmem_tensor(VEC, Float32)
            for sub in cutlass.range_constexpr(N_SUB_Q):
                q_src = cute.make_tensor(
                    cute.make_ptr(
                        mQ.element_type,
                        (mQ.iterator + q_off + sub * EPC_Q).toint(),
                        cute.AddressSpace.gmem,
                        assumed_align=16,
                    ),
                    cute.make_layout(EPC_Q),
                )
                q_sub = cute.make_rmem_tensor(EPC_Q, mQ.element_type)
                cute.copy(q_atom, q_src, q_sub)
                for i in cutlass.range_constexpr(EPC_Q):
                    q_f32[sub * EPC_Q + i] = q_sub[i].to(Float32)

            # ---- register state ----------------------------------------------
            o_f32 = cute.make_rmem_tensor(VEC, Float32)
            o_f32.fill(0.0)
            s_vec = cute.make_rmem_tensor(R, Float32)
            m = -Float32.inf
            l = Float32(0.0)

            # ---- prologue: preload the two stages ----------------------------
            # Row indices are clamped to kv_len_c-1 (in-bounds tail); the
            # commits below keep the per-thread group count uniform.
            for st in cutlass.range_constexpr(S):
                # Each ty loads its own TILE row(s) into slots
                # (st*R + ty*TILE .. st*R + ty*TILE + TILE-1) — distinct slots,
                # no overwrites. (Reading later uses slot st*R + j, j in [0, R).)
                for j in cutlass.range_constexpr(TILE):
                    row = st * R + ty * TILE + j
                    crow = row if row < kv_len_c else kv_len_c - 1
                    if const_expr(PAGED):
                        # Paged gather: page = row//PS, off = row - page*PS,
                        # phys pool row = page_table[batch, page]*PS + off.
                        grow = kv_start + crow
                        page = grow // PS
                        pidx = mPageTable[batch_idx, page]
                        kv_base = (pidx * PS + (grow - page * PS)) * num_head_kv + kv_head
                        k_off = kv_base * D + tx * VEC
                        v_off = kv_base * D + tx * VEC
                    else:
                        k_off = (kv_row0 + kv_start + crow) * kv_row_stride + kv_head * D + tx * VEC
                        v_off = (kv_row0 + kv_start + crow) * kv_row_stride + kv_head * D + tx * VEC
                    ks_off = (st * R + ty * TILE + j) * D + tx * VEC
                    for sub in cutlass.range_constexpr(N_SUB_K):
                        k_src = cute.make_tensor(
                            cute.make_ptr(
                                mK.element_type,
                                (mK.iterator + k_off + sub * EPC_K).toint(),
                                cute.AddressSpace.gmem,
                                assumed_align=16,
                            ),
                            cute.make_layout(EPC_K),
                        )
                        ks = cute.make_tensor(
                            cute.make_ptr(
                                mK.element_type,
                                (k_smem.iterator + ks_off + sub * EPC_K).toint(),
                                cute.AddressSpace.smem,
                                assumed_align=16,
                            ),
                            cute.make_layout(EPC_K),
                        )
                        cute.copy(k_g2s_atom, k_src, ks)
                cute.arch.cp_async_commit_group()
                for j in cutlass.range_constexpr(TILE):
                    row = st * R + ty * TILE + j
                    crow = row if row < kv_len_c else kv_len_c - 1
                    if const_expr(PAGED):
                        grow = kv_start + crow
                        page = grow // PS
                        pidx = mPageTable[batch_idx, page]
                        kv_base = (pidx * PS + (grow - page * PS)) * num_head_kv + kv_head
                        v_off = kv_base * D + tx * VEC
                    else:
                        v_off = (kv_row0 + kv_start + crow) * kv_row_stride + kv_head * D + tx * VEC
                    vs_off = (st * R + ty * TILE + j) * D + tx * VEC
                    for sub in cutlass.range_constexpr(N_SUB_V):
                        v_src = cute.make_tensor(
                            cute.make_ptr(
                                mV.element_type,
                                (mV.iterator + v_off + sub * EPC_V).toint(),
                                cute.AddressSpace.gmem,
                                assumed_align=16,
                            ),
                            cute.make_layout(EPC_V),
                        )
                        vs = cute.make_tensor(
                            cute.make_ptr(
                                mV.element_type,
                                (v_smem.iterator + vs_off + sub * EPC_V).toint(),
                                cute.AddressSpace.smem,
                                assumed_align=16,
                            ),
                            cute.make_layout(EPC_V),
                        )
                        cute.copy(v_g2s_atom, v_src, vs)
                cute.arch.cp_async_commit_group()

            # ---- main pipeline -------------------------------------------------
            num_chunks = (kv_len_c + R - 1) // R
            for it in cutlass.range(num_chunks):
                st = it % S

                # K[st] ready -> QK for all R rows of the stage.
                cute.arch.cp_async_wait_group(2 * S - 1)
                cute.arch.sync_threads()
                m_prev = m
                for j in cutlass.range_constexpr(R):
                    ks_off = (st * R + j) * D + tx * VEC
                    s = Float32(0.0)
                    for sub in cutlass.range_constexpr(N_SUB_K):
                        k_slice = cute.make_tensor(
                            cute.make_ptr(
                                mK.element_type,
                                (k_smem.iterator + ks_off + sub * EPC_K).toint(),
                                cute.AddressSpace.smem,
                                assumed_align=16,
                            ),
                            cute.make_layout(EPC_K),
                        )
                        k_frag = cute.make_rmem_tensor(EPC_K, mK.element_type)
                        cute.copy(s2r_atom, k_slice, k_frag)
                        for i in cutlass.range_constexpr(EPC_K):
                            s = s + q_f32[sub * EPC_K + i] * k_frag[i].to(Float32)
                    # 16-lane butterfly reduce (offsets < 16 stay in ty group).
                    s = s + cute.arch.shuffle_sync_bfly(s, offset=1)
                    s = s + cute.arch.shuffle_sync_bfly(s, offset=2)
                    s = s + cute.arch.shuffle_sync_bfly(s, offset=4)
                    s = s + cute.arch.shuffle_sync_bfly(s, offset=8)
                    s = s * scale_eff
                    row = it * R + j
                    s = s if row < kv_len_c else -Float32.inf
                    s_vec[j] = s
                    m = s if s > m else m

                # Online rescale (log2 domain). m_prev - m is -inf only when
                # m_prev == -inf (first chunk) -> exp2(-inf) = 0.
                o_scale = cute.math.exp2(m_prev - m, fastmath=True)
                l = l * o_scale
                for i in cutlass.range_constexpr(VEC):
                    o_f32[i] = o_f32[i] * o_scale
                for j in cutlass.range_constexpr(R):
                    p = cute.math.exp2(s_vec[j] - m, fastmath=True)
                    s_vec[j] = p
                    l = l + p

                # Refill K for chunk it+S into the just-consumed stage.
                cute.arch.sync_threads()
                nxt = it + S
                # The unconditional commit keeps the in-flight group count
                # uniform, so wait_group(2*S-1) stays valid for every
                # iteration including the tail (clamped rows are in-bounds).
                # Same slot mapping as the prologue: row nxt*R + ty*TILE + j
                # goes into slot st*R + ty*TILE + j of the freed stage.
                for j in cutlass.range_constexpr(TILE):
                    row = nxt * R + ty * TILE + j
                    crow = row if row < kv_len_c else kv_len_c - 1
                    if const_expr(PAGED):
                        grow = kv_start + crow
                        page = grow // PS
                        pidx = mPageTable[batch_idx, page]
                        kv_base = (pidx * PS + (grow - page * PS)) * num_head_kv + kv_head
                        k_off = kv_base * D + tx * VEC
                    else:
                        k_off = (kv_row0 + kv_start + crow) * kv_row_stride + kv_head * D + tx * VEC
                    ks_off = (st * R + ty * TILE + j) * D + tx * VEC
                    for sub in cutlass.range_constexpr(N_SUB_K):
                        k_src = cute.make_tensor(
                            cute.make_ptr(
                                mK.element_type,
                                (mK.iterator + k_off + sub * EPC_K).toint(),
                                cute.AddressSpace.gmem,
                                assumed_align=16,
                            ),
                            cute.make_layout(EPC_K),
                        )
                        ks = cute.make_tensor(
                            cute.make_ptr(
                                mK.element_type,
                                (k_smem.iterator + ks_off + sub * EPC_K).toint(),
                                cute.AddressSpace.smem,
                                assumed_align=16,
                            ),
                            cute.make_layout(EPC_K),
                        )
                        cute.copy(k_g2s_atom, k_src, ks)
                cute.arch.cp_async_commit_group()

                # V[st] ready -> PV.
                cute.arch.cp_async_wait_group(2 * S - 1)
                cute.arch.sync_threads()
                for j in cutlass.range_constexpr(R):
                    vs_off = (st * R + j) * D + tx * VEC
                    p = s_vec[j]
                    for sub in cutlass.range_constexpr(N_SUB_V):
                        v_slice = cute.make_tensor(
                            cute.make_ptr(
                                mV.element_type,
                                (v_smem.iterator + vs_off + sub * EPC_V).toint(),
                                cute.AddressSpace.smem,
                                assumed_align=16,
                            ),
                            cute.make_layout(EPC_V),
                        )
                        v_frag = cute.make_rmem_tensor(EPC_V, mV.element_type)
                        cute.copy(v_s2r_atom, v_slice, v_frag)
                        for i in cutlass.range_constexpr(EPC_V):
                            d = sub * EPC_V + i
                            o_f32[d] = o_f32[d] + p * v_frag[i].to(Float32)

                # Refill V for chunk it+S into the just-consumed stage.
                cute.arch.sync_threads()
                for j in cutlass.range_constexpr(TILE):
                    row = nxt * R + ty * TILE + j
                    crow = row if row < kv_len_c else kv_len_c - 1
                    if const_expr(PAGED):
                        grow = kv_start + crow
                        page = grow // PS
                        pidx = mPageTable[batch_idx, page]
                        kv_base = (pidx * PS + (grow - page * PS)) * num_head_kv + kv_head
                        v_off = kv_base * D + tx * VEC
                    else:
                        v_off = (kv_row0 + kv_start + crow) * kv_row_stride + kv_head * D + tx * VEC
                    vs_off = (st * R + ty * TILE + j) * D + tx * VEC
                    for sub in cutlass.range_constexpr(N_SUB_V):
                        v_src = cute.make_tensor(
                            cute.make_ptr(
                                mV.element_type,
                                (mV.iterator + v_off + sub * EPC_V).toint(),
                                cute.AddressSpace.gmem,
                                assumed_align=16,
                            ),
                            cute.make_layout(EPC_V),
                        )
                        vs = cute.make_tensor(
                            cute.make_ptr(
                                mV.element_type,
                                (v_smem.iterator + vs_off + sub * EPC_V).toint(),
                                cute.AddressSpace.smem,
                                assumed_align=16,
                            ),
                            cute.make_layout(EPC_V),
                        )
                        cute.copy(v_g2s_atom, v_src, vs)
                cute.arch.cp_async_commit_group()

            cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()

            # ---- epilogue ------------------------------------------------------
            # Non-split (write-through): O = (o/l)·v_descale in the final
            # dtype, LSE = (m+log2 l)·ln2 into mLSE.
            # Split: the SAME chunk-normalized values go to the fp32 partial
            # O_partial[split] / LSE_partial[split]; the host combine
            # (flash_fwd_combine) applies the global rescale
            # O = Σ_s exp(lse_s − L)·O_s, L = max + ln Σ exp(lse_s − max).
            o_frag = cute.make_rmem_tensor(VEC, o_elem)
            scale_out = v_descale / l
            for i in cutlass.range_constexpr(VEC):
                o_frag[i] = (o_f32[i] * scale_out).to(o_elem)
            o_base = mO_partial.iterator if const_expr(IS_SPLIT) else mO.iterator
            for sub in cutlass.range_constexpr(N_SUB_O):
                o_sub = cute.make_rmem_tensor(EPC_O, o_elem)
                for i in cutlass.range_constexpr(EPC_O):
                    o_sub[i] = o_frag[sub * EPC_O + i]
                o_dst = cute.make_tensor(
                    cute.make_ptr(
                        o_elem,
                        (o_base + o_off + sub * EPC_O).toint(),
                        cute.AddressSpace.gmem,
                        assumed_align=16,
                    ),
                    cute.make_layout(EPC_O),
                )
                cute.copy(o_atom, o_sub, o_dst)
            if tx == 0:
                if const_expr(mLSE is not None or mLSE_partial is not None):
                    lse_tensor = mLSE_partial if const_expr(IS_SPLIT) else mLSE
                    lse_tensor[lse_c] = (
                        m + cute.math.log2(l, fastmath=True)
                    ) * Float32(math.log(2.0))
