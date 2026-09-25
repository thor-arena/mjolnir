# FA4 hd256 decode-path tiling design (Option A)

Fix the hd256 small-M/long-KV **decode** attention kernel, which is 2.5–4.5× slower than
FlashInfer per step (v10 E2E decode regression root cause). In-kernel only.

## 1. Problem (measured)

From `decode-microbench.md` (clean run, server idle, paged KV, cudagraph):

| L | FA4 hd256 (M=1) | FlashInfer (M=1) | ratio |
|---|---|---|---|
| 256 | 84 μs | 31 μs | 2.7× |
| 2048 | 137 μs | 43 μs | 3.2× |
| 4096 | 197 μs | 56 μs | 3.5× |
| 8192 | 324 μs | 73 μs | 4.5× |

Per-KV-token slope: FA4 ≈30 μs/1024 tok vs FlashInfer ≈5.3 μs/1024 tok (~5.7×). Time is
**nearly M-independent** (M=1/2/4 ≈ same at a given L) and scales ~linearly with L.
This is why the v10 decode `tg t/s` is −10 to −18% at long context (16/64 layers are
full-attn; 27B decode is GEMM-dominated, so the raw 3–4.5× kernel gap shows up as a
modest E2E regression).

## 2. Current decode-path anatomy (v10 kernel scan)

Files: `vllm_flash_attn/cute/sm100_hd256_2cta_fmha_forward.py` (**FMHA**),
`tile_scheduler.py` (**SCHED**), `cute/interface.py` (**IFACE**).

- **Always 2CTA** (narrow carve-out only). M-tile is FIXED `m_block_size == 128`
  (assert FMHA:84), mma_tiler (128,128,256)/CTA, cluster spans 256 M rows.
  Decode M=1..4 still runs the 128-row (256-row-cluster) M-tile — one M-tile per query.
- **SplitKV forbidden** (pre-A1a): `assert not is_split_kv, "SM100 forward with head_dim=256 does
  not support SplitKV"` (FMHA:77); varlen scheduler hardcodes `num_splits = 1` (FMHA:423).
  Both lifted in A1a (§9).
- **Serial KV scan**: one 2CTA pair per M-tile walks ALL KV blocks
  (`while work_tile.is_valid_tile` FMHA:922; KV loop over `kv_coord` from
  `seqlen_kv_loop_start`..`seqlen_kv_loop_end` FMHA:1041-1123). Grid spans only
  (ceil(S/128), q_heads, batch) (SCHED:1886-1892) — **never the KV axis**.
- **tmem**: S@0 (2-stage double-buffered, `qk_acc_stage=2`), P in-place in S buffer,
  O@256, 512 cols.
- **S/P ping-pong ALREADY EXISTS** (1-block-deep: QK(i+1) ∥ softmax(i) ∥ PV(i-1),
  FMHA:1203-1258) → upstream PR #2817's technique is already present.

## 3. Dominant cause

**Serial per-CTA KV scan with no SplitKV.** Each decode query → one M-tile; a single
2CTA pair reads the entire `seqlen_k`. Work is M-independent and scales linearly with L
⇒ the ratio grows 2.7×→4.5× as L grows. FlashInfer splits long KV across many CTAs
(FlashDecoding/SplitKV) so its time grows slowly; FA4 hd256 cannot.

- S/P ping-pong: already present → not the cause, nothing to add.
- M-tile waste (127/128 empty rows for M=1): real but secondary — the M-independence
  proves it is not the dominant cost.

## 4. Recommended fix: SplitKV (primary)

Partition the L KV tokens across `num_splits` CTAs; each computes a partial
online-softmax state over its chunk; a combine pass merges them. This removes the
serial-L wall — the only change that fixes the L-scaling.

### 4.1 Grid change — plumbing already exists (A1a, done)
**No new grid axis had to be added.** The scheduler already emits the split axis:
`SCHED:299` `get_grid_shape` → `y = num_head * num_splits`, and `SCHED:307`
`get_current_work` already decodes `split_idx = divmod(head_idx, num_splits_divmod)`
when `is_split_kv`. And because the KV loop indexes blocks by **absolute** `kv_coord`
(FMHA:1041-1123), splitting is bounds-only: split `s` just clamps
`seqlen_kv_loop_start/steps` to its contiguous 128-block chunk
(A1a `_split_kv_clamp`: `chunk = ceil(ceil(seqlen_k/128)/num_splits)` blocks, applied at
all four trip sites — LOAD FMHA:941-957, MMA FMHA:1160-1173, SOFTMAX FMHA:1417-1428,
CORRECTION FMHA:1560-1570).
So "adding the split axis" reduced to threading `num_splits`/`is_split_kv` through FMHA
(was fixed 1 at FMHA:423) + the KV-bounds clamp: a ~5-line edit, no `tile_scheduler.py`
changes. Work units after split: q_heads(16) × batch × num_splits 2CTA pairs → hundreds
of CTAs over Thor's 20 SMs, each scanning only L/num_splits tokens.

### 4.2 Per-split partial state (scratch)
Each split writes, to a GMEM scratch indexed by (split, m-tile, head, batch):
`m_s` (row max of unnormalized scores), `l_s` (row sum of exp scores), `O_s` (partial
P·V over the chunk). Size (decode M=1): `num_splits × 16 heads × batch × [256 + 2]`
values. For batch=32, num_splits=10: ≈27 MB — trivial on Thor's 128 GB.

### 4.3 Combine (reduction) pass
A lightweight second kernel (or fused epilogue) per query/head reads the `num_splits`
partials and does the online-softmax combine:
`m = max_s m_s`; `w_s = exp(m_s − m)·l_s`; `L = Σ w_s`; `O = (1/L)·Σ (exp(m_s − m)·O_s)`.
`num_splits` is small (~4–20) → cheap per-query reduction. The dominant new cost is the
scratch write+read round-trip, not the arithmetic.

### 4.4 Threshold policy
- `num_splits = 1` for short L (≤ ~1024): the serial scan is cheap and the
  split/combine overhead would not pay off (L=256 ratio is only 2.7×).
- `num_splits` grows with L, e.g. `min(1, 1 + L//1024)` capped at a value that keeps
  total CTAs ≤ a few × SM count (cap ~10–16 on Thor). At L=8192 → ~8–10 splits.

## 5. Secondary cleanup (NOT the L-fix — note only)
The 128-row M-tile for M=1 wastes 127 rows; a smaller decode M-tile would save some MMA
work but does **not** fix the L-scaling (M-independence). Optional, defer.

## 6. Expected payoff (vs micro-bench)
- **L=8192**: the serial ~300 μs scan becomes ~num_splits-way parallel. With
  num_splits≈8–10 the scan portion drops several-fold; net target ≈90–150 μs
  (vs FlashInfer 73 μs), closing most of the 4.5× gap (a few-μs combine overhead remains).
- **L=256**: ~unchanged (num_splits=1).
- **E2E**: a ~2–3× faster hd256 decode attention should recover most of the 10–18% v10
  decode regression at long context.

## 7. LOC estimate + risk (revised post-A1a)
- **LOC**: A1a was **~5 lines** (assert removal + `num_splits` param + 4 clamp sites,
  zero SCHED edits) — the split grid axis, split_idx decode, and absolute-KV-coord loop
  all pre-existed (SCHED:299/307). Remaining: **A1b** ≈100–200 LOC (per-split partial
  `O_s` + LSE store in the correction/softmax epilogue paths) + **A1c** ≈50 LOC
  (IFACE `num_splits` lift, compile_key, varlen-b1 split axis). **No new combine
  kernel** — reuse the existing generic `_flash_attn_fwd_combine` (IFACE:1851-1863).
  Total remaining ≈150–250 LOC, down from the original ~300–500.
- **Risks**:
  1. **FP8 LSE blocker (top open item)**: `__call__` asserts `mLSE is None or
     descale_tensors is None` (FMHA:216-218) — the FP8 path currently produces no
     descale-correct LSE, but split partials + the generic combine need LSE. A1b is
     gated on producing a descale-correct LSE first (or a descale-aware combine).
  2. **Correctness of the online-softmax combine** (max/sum correction) — main bug risk
     in the reused combine; A/B-validate vs the num_splits=1 serial reference.
  3. Scratch round-trip overhead must stay below the scan savings (hence the L threshold).
  4. 2CTA cluster pairing with `num_splits>1`: verify the two CTAs of a cluster still land
     on the same (head, split) pair now that split_idx is decoded from the y-axis
     (SCHED:307) under the 2-CTA cluster rounding (FMHA:617).

## 8. Validation plan
- **Correctness**: reuse `probe2a-fp8.py` (full-FP8 A/B vs FA2 reference) + byte-identity;
  confirm the SplitKV output matches the `num_splits=1` serial path exactly for a fixed
  split count.
- **Performance**: re-run `decode-microbench.py` (B1) after the change; target
  L=8192 324 μs → ~90–150 μs, L=256 ~unchanged.

## 9. Phased implementation (A1)
- **A1a — DONE** (behavior-preserving plumbing; **byte-identity proven** — num_splits=1
  output sha256 unchanged vs pre-change):
  1. Removed `assert not is_split_kv` (FMHA:77; all other asserts intact).
  2. Added `num_splits: int = 1` as the last `__call__` param.
  3. Varlen scheduler branch now passes `num_splits=self.num_splits,
     is_split_kv=self.is_split_kv` (was hardcoded `num_splits=1`, FMHA:423).
  4. New `_split_kv_clamp` applied at all four trip sites (LOAD/MMA/SOFTMAX/CORRECTION):
     `chunk = ceil(ceil(seqlen_k/128)/num_splits)` 128-blocks; `seqlen_kv_loop_start/steps`
     clamped to the split's contiguous range.
  - `tile_scheduler.py` **not modified** — split axis + split_idx decode pre-existed
    (SCHED:299/307). All changes `cutlass.const_expr(is_split_kv)`-gated ⇒
    num_splits=1 is byte-identical.
- **A1b** (next, top open item = FP8 LSE): write per-split partial **unnormalized fp32
  `O_s` + LSE** (m_s, l_s) to the workspace, then invoke the existing generic
  `_flash_attn_fwd_combine` (IFACE:1851-1863, already wired). **FP8 LSE blocker**:
  `__call__` asserts `mLSE is None or descale_tensors is None` (FMHA:216-218) — the FP8
  path must first produce a descale-correct LSE (or the combine must be made
  descale-aware). Validate vs the num_splits=1 serial reference.
- **A1c**: lift `num_splits=1` for hd256 (IFACE:441-442), add `num_splits` to the hd256
  compile_key (IFACE:1398-1404), and give the **varlen-b1 (batch==1) static grid**
  (`FmhaStaticTileScheduler` path, FMHA:408-416) a split axis — that is the
  single-request decode case the micro-bench measures.
- **A2**: L-threshold `num_splits` policy + micro-bench sweep to pick the cap; full
  validation (A/B + micro-bench). "Done" when B1 at L=8192 drops ≥2× with no
  correctness regression.

## Upstream refs
- **#2817** (S/P ping-pong): already present here (FMHA:1203-1258) — no action.
- **#2905** (persistent scheduling + 3-deep TMA + full-D QKᵀ per KV block): prefill/dense-
  oriented; its persistent longest-first scheduling is orthogonal (launch-overhead),
  consider later. The decode fix is SplitKV.
- **e911031** (cluster-aware persistent scheduler): launch-overhead, unmerged; not the
  decode fix.
