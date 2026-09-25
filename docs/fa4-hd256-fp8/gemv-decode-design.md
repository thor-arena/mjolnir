# GEMV Decode Design — hd256 FP8 on Jetson Thor (FA4 CuTe-DSL)

Status: Phase A implementation (M=1 correctness). Phases B–D are the
planned perf/production track. Supersedes nothing; complements
`fi-decode-gemv-analysis.md` (algorithm source) and `real-splitkv-bench.md`
(harness conventions).

## 1. What we build and why

A **pure-FMA GEMV decode kernel** in the FA4 CuTe-DSL tree
(`<vfa-tree>/cute/`) for head_dim=256, M=1 decode:

- Algorithm: the FlashInfer `fast-decode` GEMV kernel
  (`flashinfer/attention/decode.cuh`, `use_tensor_cores=False` path) —
  see `fi-decode-gemv-analysis.md`. FlashInfer measures **73 µs** for the
  24 Q-head/4 KV-head, hd=256, M=1, L=8192, Q=bf16 / KV=e4m3 shape that the
  1CTA tcgen05 carve-out does in **207 µs** (`decode-1cta-clean.md`).
- Why GEMV wins at M=1: one query row per kv-head makes the score matmul a
  vector of 256-dots; doing it with FMA + shfl reduction beats a tensor-core
  pipeline whose launch/wgmma/tmem overhead dominates a single row.
- Phase A goal: **correctness only** — exact parity with an fp32 reference
  (bf16 Q, e4m3 KV, per-head descales, LSE). Perf tuning is Phase B.

## 2. Kernel architecture (Phase A)

### 2.1 Grid / block

- `grid = (num_kv_heads, 1, batch_size)` — one CTA per (kv_head, batch).
- `block = (16·g, 1, 1)` where `g = qhead_per_kvhead` (6 for Qwen3-32B 24/4):
  **96 threads = 3 warps**. `tx = tidx % 16` (d-slice), `ty = tidx // 16`
  (q-head in the GQA group). Each 16-thread group is exactly one half-warp
  starting at lane 0 or 16 — so a 4-step butterfly shfl with offsets {1,2,4,8}
  reduces within one q-head group (FlashInfer's `shfl_xor_sync` {8,4,2,1}).
- GQA fusion: thread `ty` owns q-head `h_kv·g + ty`. All 16 `tx` threads of the
  group share the identical per-head state (scores are broadcast by the shfl
  reduce), so **no inter-thread state merge is needed** at the epilogue —
  each thread writes its own 16-wide d-slice of the output row. (FlashInfer
  needs `sync_state` only when `bdz > 1`; we keep `bdz = 1`.)

### 2.2 Register state (per thread)

```
q_f32[16]   # Q d-slice, loaded once, bf16/fp16/e4m3 -> fp32
o_f32[16]   # running attention-weighted sum (fp32 accumulator)
m          # running row max in log2 domain (fp32), init -inf
l          # running row sum of exp2 scores (fp32), init 0
```
~54 live fp32 regs + temporaries ≈ 80 regs/thread — comfortable for 96
threads (3 warps) on sm_110.

### 2.3 Compute loop

```
load q_f32 (16 elements, 2x128-bit LDG)
prologue: 2 stages x { K rows, V rows } cp.async (128-bit), commit each
for it in range(ceil(kv_len / R)):          # R = rows/stage = g
    st = it % 2
    cp.async.wait_group(2·S-1); sync        # K[st] ready
    for j in range(R):                      # QK for all rows of the stage
        k_f32 = load 16 fp8 from k_smem[st·R+j] -> fp32
        s[j] = warp_reduce(Σ_i q[i]·k[i], add, width=16)   # 256-dot
        s[j] *= softmax_scale_log2 · qk_descale
        s[j] = s[j] if (it·R+j) < kv_len else -inf
        m = max(m, s[j])
    o_scale = exp2(m_prev - m)              # m_prev captured before the j-loop
    l *= o_scale
    for i: o[i] *= o_scale
    for j: p[j] = exp2(s[j] - m); l += p[j] # s[] reused as p[] (FlashInfer idiom)
    cp.async.wait_group(2·S-1); sync        # V[st] ready
    for j: for i: o[i] += p[j] · v_smem[..][i]
    sync                                    # before smem overwrite
    K/V refill for chunk it+2 into stage st (commit each)   # clamped address
tail: cp.async.wait_group(0); sync
epilogue: for i: o[i] ·= v_descale / l -> O dtype (bf16)
          write 16-elem d-slice; (tx==0) write LSE = (m + log2(l)) · ln2
```

Numerics: everything in the **log2 domain** (`exp2`, `s_log2 = s · scale ·
log2e`), mirroring `flash_fwd_mla_sm100.py` (line 3074) and FlashInfer
(`sm_scale_log2`). The descale is folded into the score scale
(`qk_descale = q_descale·k_descale`, per (batch, kv_head) — all g q-heads in
the group share it); `v_descale` is applied once at the epilogue. LSE is
emitted in **natural-log units** to match the in-tree convention:
`lse = (m + log2(l)) · ln2`.

### 2.4 KV pipeline

- 2-stage smem double buffer, `R = g` rows/stage; K and V buffers separate
  (each `2·R·256` fp8 = 3 KiB; total ~6 KiB — negligible vs the 227 KiB
  opt-in smem on sm_110).
- Per iteration the K group and V group of stage `st` are waited on
  separately (`wait_group(2·S-1)` between them), exactly like FlashInfer's
  consumer loop (`decode.cuh:310-352`): K arrives -> QK -> K refill ->
  V arrives -> PV -> V refill. One `commit_group` per (K-rows, V-rows) keeps
  the per-thread in-flight group count uniform, so the fixed
  `wait_group(2·S-1)` stays valid for every iteration **including the tail**.
- **Tail handling — clamped addressing.** FlashInfer predicates the cp.async
  (`@p cp.async` for K with `kNoFill`, `src_size=0` zero-fill for V). CuTe-DSL
  predicated copies exist (`cute.copy(..., pred=frag)`, MLA paged-KV line
  1822) but Phase A avoids them: the gmem row index is clamped
  (`row = min(row, kv_len-1)`) so every cp.async is in-bounds; OOB rows read
  the last valid row and are masked to `-inf` in compute (K data unused; V
  data finite, weighted by `p = exp2(-inf - m) = 0` — no NaN). Cost: at most
  `R-1` redundant row reads per CTA. `kv_len == 0` per batch (varlen) is an
  early-exit (O = 0, LSE = -inf). Phase B may switch to true predication.
- cp.async atom: `cpasync.CopyG2SOp()` + `make_copy_atom(..., num_bits_per_copy=128)`.
  16 fp8 elems = exactly one 128-bit instruction per thread/row.

### 2.5 Dtypes

Kernel is generic over Q dtype ∈ {bf16, fp16, e4m3} and KV dtype ∈ {bf16,
fp16, e4m3} (element type flows from the tensors; the copy atom is built from
`mK.element_type`). Primary config: **Q=bf16, KV=e4m3** (production
"bf16 weights + fp8 KV cache" shape). O dtype = Q dtype for non-fp8 KV,
bf16 for fp8 KV (matches `interface.py` `out_torch_dtype` logic, line 838).

### 2.6 Scope gates (Phase A)

M=1 (`max_seqlen_q == 1`), sm_100/sm_110, hd=hdv=256, dense **or** varlen
(`cu_seqlens`) KV, `page_table is None`, no `seqused_k`/`seqused_q`, no
softcap/sink/score_mod/mask_mod/aux/sparse/`qv`/`output_scale`,
`num_splits == 1`. Causal is allowed (M=1 ⇒ causal ≡ non-causal).

## 3. Dispatch integration (`interface.py`)

- Env knob `VLLM_FA4_HD256_GEMV=1` (default off → zero behavior change).
- Gate computed after the arch/shape/descale validation (post line ~830),
  before `fwd_cfg`. On hit: build `BlackwellHd256DecodeGEMV(g)`, convert
  tensors via `to_cute_tensor`, compile once per
  `(g, q_dtype, kv_dtype, varlen_q, varlen_k, has_lse, has_descale)` into a
  **separate** `_flash_attn_fwd.gemv_compile_cache`, launch, and return
  `(out, lse, None, None)` — bypassing the tcgen05 path entirely.
- The same-dtype assert (line ~766) is relaxed **only** when the knob is on
  and the (q,k,v) dtype set ⊆ {bf16, fp16, e4m3} — the existing
  `fp8_kv_dequant` path already establishes the "mixed Q/KV dtype" precedent
  (line 1827-1831).
- KV fp8 is passed as a uint8 view (same FFI workaround as the fp8 paths,
  line 1821-1826).
- The existing 1CTA carve-out, A1b SplitKV plumbing, and A1c test knob are
  untouched.

## 4. Verification (Phase A acceptance)

`verify-gemv-decode.py` (run in `mjolnir/vllm-thor:qwen38-sm110-v11` with
the vfa tree mounted, per `run-real-splitkv-bench.sh` pattern):

- Shapes: GQA 24/4 (g=6), hd=256, M=1, dense KV L ∈ {256, 512, 1, 3, 5}
  (small L incl. non-multiples of R=6 to exercise the tail clamp), batch ∈
  {1, 4}; plus one varlen case (cu_seqlens, uneven lengths, one empty seq).
- Dtypes: Q=bf16, KV=e4m3 with **non-trivial** per-head descales
  (q,k,v) ≠ 1.0, softmax_scale = 1/√256.
- Reference: fp32 in torch (dequant K/V by k_descale/v_descale, scores
  `q_descale·k_descale·(q·k)·sm_scale`, softmax in fp32, `l` for LSE check).
- Metrics: max-abs and max-rel error of O vs reference, LSE max-abs error.
  Acceptance: max-abs < 5e-2 and max-rel < 5e-2 (bf16-output quantization
  floor at scale ~1), LSE max-abs < 2e-3.
- Cross-check: same inputs through the FA4 1CTA kernel (`use_dedicated_
  hd256_kernel` path, no env knobs) — outputs must agree within the same
  tolerance (guards against a shared descale/scale convention bug).
- `VLLM_FA4_HD256_GEMV=0` run confirms the untouched path still works.

## 5. Phases B–D (planned, not started)

- **B — perf:** tune R/stages (bigger stage = fewer commits; smem headroom
  allows 4–8 stages of 6 rows), L2 prefetch hint variant, true predicated
  cp.async (drop the tail clamp), vector-width experiments for bf16 KV
  (VEC=8 with 32 threads/row), register pressure audit. Target: beat 73 µs
  at L=8192.
- **C — production shapes:** paged KV (page_table gather into the same smem
  staging; per-page cp.async), varlen-only fast path, optional `seqused_k`.
- **D — policy:** replace the env knob with a real dispatch policy
  (e.g., M ≤ 2 and L ≥ L* on sm_110), benchmark matrix vs 1CTA across
  L ∈ {256…16384} × M ∈ {1,2,4}, and a vLLM-side integration test.

## 6. Risks / open questions

- 96-thread block: legal (3 warps) but untested in this tree — first run
  will confirm; fallback is 128 threads with 32 idle (same shfl groups).
- `warp_reduce(..., width=16)` with a Python lambda op — traced by the DSL;
  if the lambda fails tracing, inline the 4-step butterfly (offsets 1,2,4,8).
- fp8→fp32 element conversion via `.to(Float32)` (idiom confirmed at
  `flash_bwd_preprocess.py:390`) — first compile will confirm for scalars.
- Mixed-dtype FFI: Q=bf16 passed natively, K/V as uint8 views declared
  e4m3 — same mechanism as the existing fp8 paths, low risk.
