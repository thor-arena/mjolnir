# FlashInfer "fast paged decode" (hd256, M=1, L=8192, ~73μs) — kernel identification & full algorithm

Source tree: `<scratch>`
Bench config that produced the 73μs: `decode-1cta-clean-bench.py` (B3) —
`BatchDecodeWithPagedKVCacheWrapper(use_tensor_cores=True, backend="auto")`,
Q=bf16, K/V=fp8 e4m3, O=bf16, **24 Q heads / 4 KV heads (GQA group 6)**, `page_size=16`,
kv_layout=NHD, hd=256, M=1, L=8192. (Verified in-script: `NUM_Q_HEADS=24`,
`NUM_KV_HEADS=4`, `q_data_type=bfloat16, kv_data_type=float8_e4m3fn`, `q_len_per_req=m`.)

Device facts (NVIDIA Thor, cc 11.0, queried live): **20 SM**, 1536 threads/SM,
**665600 B (650KB) smem/SM**, **232448 B (227KB) opt-in smem/block**.
Result: `decode-microbench-results.json` → `B3_flashinfer|M1|L8192` median **72.53μs**.

## 0. THE CORE CORRECTION

The 73μs kernel is **NOT** the CUDA-core GEMV kernel in `data/include/flashinfer/attention/decode.cuh`.
With `use_tensor_cores=True` and `backend="auto"` on Thor (sm_110, major=11):

- `determine_attention_backend` (decode.py) → `"fa2"` (Hopper sm90a TC path not available on sm110).
- plan+run go through the **prefill** module `get_batch_prefill_module("fa2", ...)`
  → C++ `BatchPrefillWithKVCachePlan` / `BatchPrefillWithPagedKVCacheRun`
  (`data/csrc/batch_prefill.cu:47,239`)
  → kernel `BatchPrefillWithPagedKVCacheKernel`
  (`data/include/flashinfer/attention/prefill.cuh:4123`), the **FA2 tensor-core kernel**.

So the 73μs number = FA2 TC prefill kernel (Ampere-style HMMA mma.sync + smem ldmatrix +
cp.async, 2-stage software pipeline) **plus a split-KV logsumexp merge kernel**, run in
"decode mode" (1 query row per request, GQA-packed into 16 rows).

The GEMV kernel in `decode.cuh` is the `use_tensor_cores=False` path (pure FMA dot + shfl
butterfly). It is documented here as well (section 8) since the task originally assumed it,
but it is **not** the 73μs kernel.

There is **no tmem, no TMA, no mma.sync...f8f8f32 (fp8 HMMA) usage** anywhere in this path.
FP8 K/V are **dequantized to bf16 in registers inside the mma loop** and multiplied with
bf16 Q/P using `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`.

## 1. Exact instantiation for the B3 benchmark (M=1, L=8192, hd256, GQA6, fp8 KV)

All numbers below are derived from the dispatch code paths; they are the *actual* kernel
geometry that produced the 73μs.

### 1.1 Plan (host side) — `PrefillPlanImpl` + `PrefillSplitQOKVIndptr`
`data/include/flashinfer/attention/scheduler.cuh:764` and `:550`

- `packed_qo_len = qo_len * group_size = 1 * 6 = 6` rows per request.
- `FA2DetermineCtaTileQ(6, head_dim=256, head_dim_qk=256, kv_bytes=1)`
  (`data/include/flashinfer/utils.cuh:408`):
  not ≥512, qk not ≥512, avg_packed_qo_len=6 ≤ 16 → **CTA_TILE_Q = 16**
  (smem probe: 16·256·2 + (256+256)·16·4·1 = 40960 B ≤ opt-in limit → CTA16, the 1×4 layout).
- Split-KV: `max_grid_size = 2 blocks/SM × 20 SM = 40`;
  `max_batch_size_if_split = 40 / 4 kv_heads = 10` (scheduler.cuh:788-791).
  `PrefillBinarySearchKVChunkSize` (scheduler.cuh:101) binary-searches page-unit chunk size
  `mid ∈ [8, 512]` (min = 128/page_size = 8 pages) so that
  `ceil(6/16)·ceil(512/mid) ≤ 10` → converges to **52 pages = 832 KV elements per chunk**.
  `split_kv = (52 < 512) = true`, `num_chunks_kv = ceil(512/52) = 10`,
  `new_batch_size = 1 × 10 = 10` CTAs (per q-tile).
- `kv_chunk_size` is stored in **page units** in the plan and multiplied by `page_size` at the
  end (scheduler.cuh:679); the kernel reads the element-unit value from
  `params.kv_chunk_size_ptr`.
- Workspace (only if split_kv): `tmp_v` (per-chunk O, bf16; allocated conservatively as
  `num_qo_heads · padded_batch · cta_tile_q · head_dim_vo · 4B`), `tmp_s` (per-chunk LSE,
  float), `merge_indptr` (scheduler.cuh:855-882).

### 1.2 Kernel geometry (dispatch: `BatchPrefillWithPagedKVCacheDispatched`, prefill.cuh:4335)

| parameter | value | source |
|---|---|---|
| grid | `(10, 1, 4)` = 40 CTAs | `nblks = (padded_batch_size, 1, num_kv_heads)`, prefill.cuh:4359 |
| block | `(32, 1, 4)` = 128 threads | `nthrs = (32, NUM_WARPS_Q, NUM_WARPS_KV)`, prefill.cuh:4360 |
| NUM_WARPS_Q | 1 | `get_num_warps_q(16)` = 1 (prefill.cuh:72) |
| NUM_WARPS_KV | 4 | `4 / NUM_WARPS_Q` (prefill.cuh:83) |
| NUM_MMA_Q | 1 | `get_num_mma_q(16)` = 1 (prefill.cuh:87) |
| NUM_MMA_D_QK | 16 | `HEAD_DIM_QK/16` (prefill.cuh:4362) |
| NUM_MMA_D_VO | 16 | `HEAD_DIM_VO/16` |
| NUM_MMA_KV | **6** | `min(smem budget, reg cap)`: per-TB budget = min(650KB/2, 227KB opt-in) = 232448 B; `(232448 − 8192)/32768 = 6`; reg cap `8/NUM_MMA_Q = 8` → 6 (prefill.cuh:4419-4437; verified live against Thor attributes) |
| CTA_TILE_KV | `4·6·16 = 384` KV rows | `NUM_WARPS_KV·NUM_MMA_KV·16` |
| USE_VO_SPLIT | false | needs `HD_VO/16 > 16` (prefill.cuh:130) |
| USE_KV_REPACK | false | needs `CTA_TILE_Q > 16` (prefill.cuh:115) → in-loop dequant path |
| USE_KV_SHARED_SMEM | false | fp8 KV + CTA_TILE_Q=16 → K and V in separate smem buffers (prefill.cuh:155-157) |
| DTypeQKAccum | float | `USE_FP16_QK_REDUCTION=false` (prefill.cuh:4364) |
| smem | **204800 B (200KB)**, union: `{q_smem 16×256 bf16 (8KB), k_smem 384×256 fp8 (96KB), v_smem 384×256 fp8 (96KB)}` vs `{cta_sync_o_smem 4·16·256 f32 (64KB), cta_sync_md_smem (2KB)}` → max = 200KB ≤ 227KB opt-in | `SharedStorageQKVO`, prefill.cuh:144-169 |

Kernel work per CTA: its KV chunk = 832 rows → `ceil(832/384) = 3` main-loop iterations
(last iteration handles the 64-row tail with mask).
GQA: the 6 Q heads of one KV head are packed as rows 0..5 of the 16-row Q tile
(`group_size.divmod` throughout, e.g. prefill.cuh:1430). Rows 6..15 of the tile are dummy;
their outputs are never stored (`o_idx < qo_upper_bound` guards, prefill.cuh:4011, 4095).

### 1.3 Per-iteration instruction budget (the "compute pattern")

Per main-loop iteration (384 KV rows), per CTA:
- Q·K: `NUM_MMA_D_QK(16) × NUM_MMA_KV(6) × 2` = **192 × `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`**
  (the "m16n16k16" wrapper is 2× m16n8k16 — `data/include/flashinfer/mma.cuh:317-360`).
- row-sum d: `NUM_MMA_KV(6) × 1` mma with **B-fragment = 0x3F800000 (bf16 1.0)** =
  P·1 trick, `m16k16_rowsum_f16f16f32` (mma.cuh:522-549).
- P·V: `NUM_MMA_D_VO(16) × NUM_MMA_KV(6) × 2` = **192 mma** (same m16n8k16 bf16→f32).
- Total = **390 HMMA/iter**, 3 iters/CTA, 40 CTAs → **≈ 46.8k mma** over 20 SMs.
- Plus per-mma-step fp8→bf16 dequant: 8-elem `cvt` (vec_cast) on K (16×6 steps) and V
  (16×6 transposed steps) fragments per iteration.

## 2. Kernel entry & dataflow (prefill.cuh:4123 `BatchPrefillWithPagedKVCacheKernel`,
device fn `BatchPrefillWithPagedKVCacheDevice` at :3510)

Per-CTA setup:
1. `variant = DefaultAttention<false,false,false,false>(params, request_idx, smem)`
   — `sm_scale_log2 = params.sm_scale · log2e` (variants.cuh:54). For the fp8 bench, Python
   has already folded `q_scale·k_scale` into `sm_scale` (see §6).
2. Load Q tile (≤16 rows, bf16, from gmem `params.q` packed as qo_len×heads×hd) into
   `q_smem` (swizzled, `cp_async::cp_async` 128-bit), commit group.
3. Paged K/V offset resolution: `paged_kv.protective_get_k_offset / protective_get_v_offset`
   (page.cuh) walk the **paged indices array** (page_size=16, per-KV-head page tables) to get
   per-thread gmem byte offsets for the first tile; page-walk cost is spread over threads
   via `KV_THR_LAYOUT` (each of the 128 threads handles a few 16B segments).
4. Prologue loads: `page_produce_kv` K(0) → `k_smem` (commit), V(0) → `v_smem` (commit) —
   128-bit `cp.async.cg`, `k_smem_offset_w`/`v_smem_offset_w` swizzled writes
   (SWIZZLE 128B for hd256 fp8: 256 B rows → 16 B/row chunks, 2 segments... see `smem_t`
   permuted offset, utils.cuh).
5. Init states: `m = -inf[2]`, `d = 0[2]` per thread (init_states, prefill.cuh:930),
   `o_frag` zeroed (register array `float o_frag[NUM_MMA_Q][NUM_MMA_D_VO][8]`).

### Main loop (prefill.cuh:3857-4051) — one pass per 384-row KV tile
```
for iter in ceil(chunk_size / CTA_TILE_KV):          # 3 iters for the 832-row bench chunk
    # (a) resolve next-tile paged offsets (K and V) — page.cuh walk, 8 offset slots/thread
    # (b) wait_group<1>; block.sync()   # current tile's K/V in smem
    # (c) compute_qk  (prefill.cuh:1179)      -> s_frag[1][6][8] f32
    # (d) logits_transform (prefill.cuh:1416) # identity for DefaultAttention (no softcap/alibi)
    # (e) logits_mask   (prefill.cuh:1484)     # set s = -inf where kv_idx >= chunk_end
        #     (causal/window logic present but inactive: mask_mode=NONE, no window)
    # (f) update_mdo_states (prefill.cuh:1595) # online softmax (detailed below)
    # (g) block.sync(); prefetch K(iter+1) -> k_smem (commit); wait_group<1>; block.sync()
    # (h) compute_sfm_v (prefill.cuh:1689)     # d += P·1 (mma), o += P·V (mma) (detailed below)
    # (i) block.sync(); prefetch V(iter+1) -> v_smem (commit)
wait_group<0>; block.sync()
```
The double-buffering is *temporal*: single smem buffer per K and V; the **next** tile's K is
copied during the current tile's softmax, and the next tile's V is copied during the current
tile's P·V (hence the interleave in (g)/(i)). No K/V overlap within the same smem buffer.

### compute_qk detail (prefill.cuh:1179-1280)
- A-frag (Q): `q_smem->ldmatrix_m8n8x4` → `a_frag[4]×uint32` per (mma_d, mma_q) —
  Q stays bf16 in smem, read via standard ldmatrix.
- B-frag (K), fp8 path (`sizeof(DTypeKV)==1`, not fp4, not repack):
  - `ldmatrix_m8n8x4_left_half / right_half` (odd/even mma_d) pulls **8 fp8 bytes** (32-bit
    ldmatrix lane-halves) from the 8-bit-packed swizzled `k_smem`;
  - `frag_layout_swizzle_16b_to_8b` reorders lanes for mma B layout;
  - `vec_cast<bf16, fp8>::cast<8>` — **8× `cvt` fp8 e4m3 → bf16** in registers;
  - then `mma_sync_m16n16k16_row_col_f16f16f32` (kInit on mma_d==0) accumulates into
    `s_frag[mma_q][mma_kv][8]` f32.
- So: `S = Q(bf16) × K(fp8→bf16)`, f32 accumulate, one m16n16k16 (=2×m16n8k16) per
  (mma_d, mma_kv). Loop: 16 mma_d × 6 mma_kv × 1 mma_q.

### logits/mask (prefill.cuh:1416, 1484)
- `logits_transform`: per s_frag element → `variant.LogitsTransform` (identity here).
- `logits_mask`: `kv_idx >= chunk_end` → `s_frag = MaskFillValue` (= -inf-ish,
  `KTraits::MaskFillValue`). For non-causal, no-window: only the tail partial tile is masked.

### update_mdo_states (online softmax) (prefill.cuh:1595-1686, float path)
Per (mma_q, j) — each thread owns 2 rows of S (mma m16: rows `lane/4` and `lane/4+8`):
1. `m_local` = max of the 4 registers of that row within each `s_frag[·][mma_kv]`
   (regs `j*2+0,1,4,5` — the two n-halves of the m16n16 output).
2. Row max across the 4 lanes sharing the row: `m = max(m, shfl_xor(m, 0x2)); m = max(m, shfl_xor(m, 0x1))` (prefill.cuh:1619-1620).
3. `o_scale = exp2(m_prev·sm_scale_log2 − m·sm_scale_log2)` (`ptx_exp2` = `ex2.approx.ftz.f32`).
4. `d *= o_scale`; rescale `o_frag[mma_q][mma_d][j*2|j*2+1|j*2+4|j*2+5] *= o_scale` for all 16 mma_d.
5. In-place: `s_frag = exp2(s_frag·sm_scale_log2 − m·sm_scale_log2)` for every mma_kv tile
   → this is the P matrix (bf16-convertible, still held in f32 registers).

Everything runs in **log2 domain** with `sm_scale_log2` folded in — no `·ln2` divisions.

### compute_sfm_v (P·V) (prefill.cuh:1688-1800)
1. `vec_cast<bf16, float>::cast<8>`: P (s_frag) f32 → **bf16** register fragments
   (`s_frag_f16[1][6][8]`) — required because the mma is bf16.
2. `d += rowsum`: `m16k16_rowsum_f16f16f32(d[mma_q], s_frag_f16[mma_q][mma_kv])`
   = `mma.sync.m16n8k16` with B = packed bf16 1.0 (1065369472) — P summed across 16 KV cols.
   (This is how FlashAttention's `D = rowsum(P)` is computed on tensor cores.)
3. V-frag, fp8 path: `v_smem->ldmatrix_m8n8x4_trans_left_half/right_half`
   (transposed ldmatrix — V is stored KV-row-major, mma needs it transposed),
   `frag_layout_swizzle_16b_to_8b_trans`, `vec_cast<bf16,fp8>::cast<8>`,
   `swap(b_frag[1], b_frag[2])` (n-half swap for trans-B layout).
4. `mma_sync_m16n16k16_row_col_f16f16f32(o_frag[mma_q][mma_d], s_frag_f16[mma_q][mma_kv], b_frag)`
   for mma_d in 0..15, mma_kv in 0..5 — **O += P(bf16)·V(fp8→bf16)**, f32 accum, 192 mma/iter.

Note P is NOT renormalized by 1/d inside the loop — `d` is carried separately (the
FlashAttention-2 trick: normalize once at the end, or in the merge kernel for split-KV).

## 3. Epilogue (prefill.cuh:4055-4110)

1. `finalize_m`: `m *= sm_scale_log2` (only if not -inf).
2. **Cross-KV-warp merge** (`threadblock_sync_mdo_states`, prefill.cuh:1860-1988,
   needed because NUM_WARPS_KV=4 — each KV warp w owns a 384/4=96-row KV slice with its own
   m/d/o partials):
   - stash `o_frag` (vec_t<float,8> × 16 mma_d) + `make_float2(m, d)` into
     `cta_sync_o_smem`/`cta_sync_md_smem` (the union with k/v smem), `__syncthreads()`;
   - log-sum-exp merge across the 4 warps: `m_new = max_i m_i`,
     `d_new = Σ_i d_i·exp2(m_i − m_new)`; `o = Σ_i o_i·exp2(m_i − m_new)` (per-reg FMA);
   - back to registers.
3. `transform_output` (prefill.cuh:1819) → `variant.OutputTransform` = base
   `return o / d` (variant_helper.cuh:66-73) → **final normalized O** (when not split-KV).
4. `write_o_reg_gmem` (prefill.cuh:1991): f32→bf16 cast, `stmatrix_m8n8x4` back into
   `q_smem` (reused as staging), then 128-bit gmem stores (8 bf16 per thread per step),
   guarded by `o_idx < qo_upper_bound`.
5. LSE (only if `variant.use_softmax`, true here): `lse = log2(d) + m`
   (m already in log2 domain) written by warp 0 only (prefill.cuh:4084-4109).

### Split-KV variant (active in the bench: 10 chunks)
- `params.o = tmp_v`, `params.lse = tmp_s` (csrc/batch_prefill.cu:343-347; launch :362-366;
  `params.partition_kv = true` at prefill.cuh:4494).
- Per chunk the kernel still runs the full epilogue (cross-warp merge + `transform_output`
  `o/d` + lse write); with `partition_kv` only the write strides change
  (`o_stride_n` × `num_kv_chunks`, lse row offset `o_indptr + qo_idx·num_kv_chunks + kv_tile_idx`,
  prefill.cuh:4067-4109). So tmp_v holds the **already chunk-normalized** O_i and tmp_s
  holds each chunk's LSE_i = `log2(d_i) + m_i` (log2 domain).
- `VariableLengthMergeStates` (`data/include/flashinfer/attention/cascade.cuh:687`,
  kernel at :368) merges the `num_index_sets` partials per Q row: `state_t::merge(o_i, lse_i,
  /*d=*/1)` (state.cuh:53) → weights `w_i = 2^(lse_i − M)`, `M = max lse_i`:
  `o_final = Σ w_i·O_i / Σ w_i`, `lse_final = M + log2(Σ w_i)`.
  Fast path: `num_index_sets == 1` → straight copy (cascade.cuh:407-415).
  Grid: persistent, `ceil(seq_len·num_heads / gridDim.x)` work items; PDL
  (`griddepcontrol` wait/launch) chained after the attention kernel.
- Python then applies `out *= v_scale` (decode.py:2287-2293).

## 4. FP8 descale handling (where the q/k/v scales actually apply)

- **In-kernel: none** (no scale-factor smem for this dtype — that's NVFP4-only,
  `k_sf_smem`/`v_sf_smem` paths in compute_qk/compute_sfm_v are `is_fp4_type_v`-gated).
- `q_scale · k_scale` is folded into `sm_scale` **in Python before the call**
  (decode.py:2049-2052): `sm_scale = sm_scale * q_scale * k_scale`, so the
  Q·K logits come out directly descaled (`sm_scale_log2` carries them).
- `v_scale` is applied **after the kernel in Python**: `out *= v_scale`
  (decode.py:2287-2293). (For the merge path this multiply applies to the merged O.)

## 5. How to replicate in FA4 / CuTe-DSL (blueprint)

To match this kernel's cost model for hd256 M=1 GQA6 fp8:
1. **Warp tiling**: 128 threads = 1 Q-warp × 4 KV-warps; each KV warp owns 96 KV rows
   (6 × 16-row mma steps). Grid = (q_tiles × kv_chunks, 1, num_kv_heads).
2. **Tiles**: CTA_Q=16 (GQA-packed rows), CTA_KV=384, K/V smem = 384×256 fp8 (96KB each),
   Q smem = 16×256 bf16 (8KB); 200KB total smem (227KB opt-in on Thor) → 2 CTAs/SM (650KB/SM).
3. **Pipeline**: 2-group cp.async (128-bit `cp.async.cg`), next-K loaded during softmax,
   next-V loaded during P·V; paged offsets resolved per-iteration from the indices array.
4. **GEMM**: Ampere HMMA only — `mma.sync.m16n8k16.f32.bf16.bf16.f32`, ldmatrix A/B,
   **in-loop fp8→bf16 cvt on the B-fragment** (8-elem `cvt`, 16b→8b swizzle fixups,
   B[1]↔B[2] swap for transposed V). No fp8-mma, no tmem, no TMA.
5. **Softmax**: log2-domain online (`ex2.approx.ftz.f32`), row max via 2 `shfl_xor`
   (0x2, 0x1) over the 4 lanes per m16 row; per-tile o-rescale before P·V; P cast to bf16
   in-place (registers) and fed straight into the P·V mma; **D=rowsum(P) via a ones-mma**
   (B-frag = bf16 1.0) rather than a separate add.
6. **Cross-warp combine**: smem LSE merge of the 4 KV warps at the end (64KB o-staging in
   the K/V smem union) — this smem is the reason hd256 is at the smem budget edge.
7. **Split-KV**: host binary search (scheduler.cuh:101) sizes chunks to fill
   `2·#SM/#KV_heads` CTAs; per-chunk unnormalized O + LSE to workspace, then a small
   `VariableLengthMergeStates` LSE-merge kernel. For the bench: 10 chunks × 832 rows.
8. **Scales**: fold q·k scale into sm_scale pre-kernel; v scale post-kernel. (An FA4/
   CuTe implementation can instead keep f32 descales in the epilogue — same cost class.)

Arithmetic summary (bench): ≈ 46.8k m16n8k16-bf16 mma (40 CTAs × 3 iters × 390 mma) +
≈ 40 × 3 × 192 fp8→bf16 cvt groups on K/V + 1 × 24-head × 1-row lse-merge (10 partials),
over 40 CTAs / 20 SMs.

## 6. The other kernel: GEMV decode (`decode.cuh`) — NOT the 73μs path

For completeness (the original task assumed this was the target).
`BatchDecodeWithPagedKVCacheKernel` (decode.cuh:613 kernel, :396 device fn, :741 dispatch):

- Config for fp8/hd256/GQA6: `vec_size = max(16/1, 256/32) = 16` elems (128-bit loads);
  `bdx = HEAD_DIM/vec_size = 16`, `bdy = GROUP_SIZE = 6`, `bdz = 1` → **96 threads**
  (`dim3(bdx,bdy,bdz)` launch, decode.cuh:694,719,774), 2-stage smem pipeline
  (`NUM_STAGES_SMEM=2`, utils.cuh), `tile_size_per_bdx=1` (GQA>1).
- GQA: one CTA per (request, kv_head); `ty`-dim (6) threads = the 6 Q heads, sharing the
  same K/V smem tile (K/V loaded once per CTA).
- Q·K: pure **FMA dot** over 16 fp8 elems (cast to f32, `vec_dtypes.cuh` cast_load) per thread
  + `shfl_xor_sync` butterfly over `bdx/2 = 8` offsets → 4 shfls... (offsets 8,4,2,1 →
  4 shfls) to get the per-row S in all threads.
- Online softmax: per-KV-row running max m, per-tile `o_scale = exp2(m_prev − m)`,
  rescale o, `d += exp2(s − m)`; single pass.
- P·V: per KV row, `o[i] += p · v[i]` FMA over 16 elems — one KV row at a time, o held in
  registers across the whole sequence.
- Split-KV: `BatchDecodeWithPagedKVCacheWorkEstimationDispatched` (scheduler.cuh:150)
  occupancy-based: if `batch·kv_heads < max_grid` (= blocks/SM × SMs) → binary-search page
  chunks (`PartitionPagedKVCacheBinarySearchMinNumPagePerBatch`), else single pass;
  partials combined by the same `VariableLengthMergeStates`.
- Descales: identical Python-side folding (q·k → sm_scale, v → post-kernel).

This kernel is register-resident-o, no smem for O, no mma; it's the CUDA-core fallback
(`use_tensor_cores=False`) and the one that matters for small batch on low-TC parts.
On Thor's 20 SMs at M=1 it under-utilizes (1 req × 4 kv_heads = 4 CTAs, no split by
  default) and is slower than the
TC+split-KV combination that produced the 73μs.

## 7. File index (all paths absolute, under `<scratch>`)

| what | file:line |
|---|---|
| FA2 TC kernel (the 73μs kernel) | `data/include/flashinfer/attention/prefill.cuh:4123` |
| FA2 device fn | prefill.cuh:3510 |
| FA2 dispatch/traits selection | prefill.cuh:4335-4519 |
| KernelTraits | prefill.cuh:236-366 |
| SharedStorageQKVO (smem union) | prefill.cuh:144-169 |
| compute_qk (ldmatrix + fp8 dequant + mma) | prefill.cuh:1179-1280 |
| update_mdo_states (online softmax) | prefill.cuh:1595-1686 |
| compute_sfm_v (rowsum-mma + P·V mma) | prefill.cuh:1688-1800 |
| threadblock_sync_mdo_states (cross-warp merge) | prefill.cuh:1860-1988 |
| transform_output / write_o_reg_gmem / lse | prefill.cuh:1819, 1991, 4084 |
| FA2 warp/mma helpers | prefill.cuh:72-96 |
| mma wrappers (m16n16k16 = 2×m16n8k16; rowsum ones-mma) | `data/include/flashinfer/mma.cuh:317-360, 522-549` |
| FA2DetermineCtaTileQ | `data/include/flashinfer/utils.cuh:408` |
| PrefillPlan / PrefillSplitQOKVIndptr / binary search | `data/include/flashinfer/attention/scheduler.cuh:764, 550, 101` |
| FA2 Plan/Run C++ entry | `data/csrc/batch_prefill.cu:47, 239` |
| FA2 kernel instantiation (CTA_TILE_Q set) | `data/csrc/batch_prefill_paged_kernel_inst.jinja` |
| DefaultAttention variant (sm_scale_log2, o/d) | `data/include/flashinfer/attention/variants.cuh:32-92` |
| Merge kernel | `data/include/flashinfer/attention/cascade.cuh:687` (VariableLengthMergeStates; state_t in state.cuh) |
| GEMV decode kernel | `data/include/flashinfer/attention/decode.cuh:613, 396, 741` |
| GEMV work-estimation/split | scheduler.cuh:150 |
| Python backend selection + fp8 scale folding | `decode.py` (`determine_attention_backend`; 2049-2052; 2287-2293) |
| JIT module gen (variant = DefaultAttention<false,false,false,false>) | `jit/attention/modules.py:484` |

## 8. One-line summary

The 73μs "FI fast paged decode" is the **FA2 tensor-core prefill kernel in decode guise**:
40 CTAs (10 KV-chunks × 4 KV-heads), 128 threads (1 Q-warp × 4 KV-warps), 16-row GQA-packed Q
tile (6 real rows) × 384-row KV tiles, cp.async 2-group pipeline, in-loop fp8→bf16 dequant
feeding `mma.sync.m16n8k16.f32.bf16.bf16.f32` for Q·K and P·V (390 mma/iter, 3 iters/chunk),
log2-domain online softmax with a ones-mma row-sum, smem cross-warp LSE merge, split-KV
partials + `VariableLengthMergeStates` — with q·k scale folded into sm_scale and v scale
applied post-kernel in Python. No tmem/TMA/fp8-HMMA; the GEMV kernel in `decode.cuh` is a separate,
slower CUDA-core path that this benchmark never runs.
