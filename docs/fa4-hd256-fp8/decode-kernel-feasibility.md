# DK-0 — Lighter hd256 decode kernel (avoid prefill machinery): feasibility + design + go/no-go

**Decision gate before we invest in writing a lighter hd256 decode kernel.**
Date 2026-09-23 · Thor sm_110a (cap 11.0, 20 SMs) · kernel tree `<vfa-tree>/cute/`
Companion data: `decode-microbench.md` (FA4 vs FlashInfer), `splitkv-sweep.md` (SplitKV is the wrong lever on the 2CTA kernel).

---

## VERDICT (read this first)

**GO on the *1CTA* light decode kernel as a high-ROI intermediate win; NO-GO on the "shrink
the M-tile / minimal-tmem" variant; the *full* win to ~73 μs is best taken via Option B
(backend-split decode → existing FlashInfer), not a new GEMV kernel.**

The single most decision-relevant fact, and the one that **reframes the whole task**:

> **The cost is the 2CTA cluster, not the M-tile.** Measured on Thor, M=1, L∈{4096,8192}:
> **1CTA is 1.5–1.7× faster than 2CTA** (L=8192: 2CTA 320.6 μs vs 1CTA 190.7 μs median). The 2CTA
> cluster *doubles the per-KV-token scan slope* (0.032 vs 0.016 μs/tok) — so the penalty grows with
> L, exactly where decode hurts. This is measured, reproducible across two runs, and the 2CTA number
> matches the clean production microbench (324 μs) to within noise.

Consequences:
- The task's stated culprits "512-col tmem / 128-row M-tile (127 empty rows)" are **not** the
  dominant cost — the decode is **M-independent** (decode-microbench: M=1/2/4 ≈ equal), so shrinking
  the M-tile buys ~0, and the tmem is **pinned at 512 cols by head_dim_v=256 regardless of M** (§1).
- The real lever is **dropping the 2CTA cluster** → 1CTA. That alone closes **~53% of the gap**
  (324 → ~190 μs; 4.5× → ~2.6× vs FlashInfer) at low effort.
- Getting the *rest* (190 → 73 μs) needs to delete the per-block tmem/TMA-MMA/softmax machinery
  entirely — i.e. the **GEMV architecture**, which is exactly what FlashInfer already ships. So the
  full win is **Option B** (route decode → existing FlashInfer), not a new in-tree kernel.

One caveat that changes "how" not "whether": **1CTA is currently unreachable from the public API**
(§2) — the production decode path never sets `seqused_q`, and the 1CTA carve-out is dense-only.
Making 1CTA usable on the production paged+varlen shape is the main DK-1 work item, and the 1CTA+paged
path is unverified (top risk).

---

## Answer 1 — tcgen05 MMA min-M at D=256 (the crux)

**The tensor-core (tcgen05) MMA has a hard M-mode floor of 64 (1CTA) / 128 (2CTA). There is no
32- or 16-row MMA. But this floor is a non-binding constraint here, because the M-tile size does not
govern the decode cost (§2).**

The MMA M-mode is not set by the kernel; it is validated by the tcgen05 op against the `cta_group`:
- `tcgen05/mma.py:197-212` — **CtaGroup.ONE: M ∈ {64, 128}**; N ∈ 8..256 (16..256 for 8-bit MN-major).
- `tcgen05/mma.py:213-227` — **CtaGroup.TWO: M ∈ {128, 256}**; N ∈ 16..256 (32..256 for 8-bit MN-major).

How the kernel builds it (`sm100_hd256_2cta_fmha_forward.py`, FMHA):
- `:86` `assert m_block_size == 128 and n_block_size == 128` and `:97-98` `assert mma_tiler[0]==128 and mma_tiler[1]==128 and mma_tiler[2]==256` — the M-tile is hard-fixed at 128.
- `:92` `mma_tiler = (128, 128, 256)`; `:105` `cluster_size_m = 2 if use_2cta else 1`.
- `:106-110` `qk_mma_tiler = (cluster_size_m*128, 128, min(256,128)=128)`. So the **MMA M-mode = 256 (2CTA) or 128 (1CTA)** — both above the floor (256 for TWO is allowed; 128 for ONE is allowed).
- `:507` `cta_group = CtaGroup.TWO if self.use_2cta else CtaGroup.ONE`.
- `:117-118` `iterations_qk = 256//128 = 2`, `iterations_pv = 256//128 = 2` — the head_dim=256 K/N dimension is what is chunked, **not** M.

**Can the M-tile go to 64 / 32 / 16?**
- **64: yes, in 1CTA** (CtaGroup.ONE allows M=64). It would need relaxing the `:86`/`:97` asserts and
  re-deriving the tmem/softmax partitions for a 64-row tile. But it **buys ~0** (§2: M-independence).
- **32 / 16: no.** The tcgen05 MMA has no such M-mode for any cta_group. This is the "INHERENT" floor.
- **The removable M-tile waste is capped at 128→64 in 1CTA (i.e. 63 empty rows at M=1), not 128→1.**
  And because decode is memory-bound + M-independent, even that 128→64 reduction is a wash.

### The tmem is M-independent (why "minimal tmem" is impossible via M)
- `:144` `tmem_alloc_cols = get_max_tmem_alloc_cols("sm_100")` = **512** (`cute/arch/tmem.py:28-31` maps `sm_100→512`); `:959` `tmem.allocate(512)`.
- `:161-163` `tmem_s_offset=0`, `tmem_o_offset=256`, `tmem_p_offset=0` (P is written in-place in the S buffer).
- `:181` `qk_acc_stage = 2` → the S accumulator is double-buffered.
- So the 512 cols = **S: 2 stages × 128 = 256** (N=128) + **O: head_dim_v = 256** (N=256).
- **The O footprint is 256 cols because head_dim_v=256 — it is set by N, not M.** At M=64 the O is
  still 256 cols; S is still 2×128=256 cols. **Reducing M from 128→64 does not reduce tmem at all.**
  The only tmem savings available are dropping the S double-buffer (512→384, breaks the
  QK∥softmax ping-pong, `:1203-1258`), which is negligible.
- The GEMV path (FlashInfer) has **zero** tmem — O and state live in registers. This is the only way
  to eliminate the tmem, not to shrink it.

**Bottom line for #1:** the "M-tile waste" and "minimal tmem" goals in the task are red herrings for a
memory-bound M=1 decode. The MMA floor (64/128) confirms you can't get to M=1 with tensor cores, but
you don't need to — the cost is elsewhere.

---

## Answer 2 — Is there a functional 1CTA path, and does decode trigger it? (measured)

**Yes, the kernel functionally supports 1CTA (CtaGroup.ONE, cluster (1,1)) — but the production decode
does NOT trigger it, and it is currently unreachable from the public API.**

The carve-out that lowers `hd256_use_2cta` to False (`interface.py:1037-1046`):
```
hd256_use_2cta = not(
    seqused_q is not None and cu_seqlens_q is None and page_table is None and not local
    and ( (not causal and max_seqlen_q <= 1024) or (causal and max_seqlen_q<=2048 and max_seqlen_k<=2048) ) )
```
Three requirements that the **production decode cannot satisfy**:
1. `seqused_q is not None` — but the public FA4 dispatch never sets it: `flash_attn_interface.py:437-465`
   calls `_flash_attn_fwd(..., seqused_k=..., ...)` with **no `seqused_q`** (it is not even a param of
   the public `flash_attn_varlen_func`, `flash_attn_interface.py:176+`). Only the low-level
   `_flash_attn_fwd` (`cute/interface.py:606`, param at `:613`) has it. → **1CTA is unreachable via the public API.**
2. `page_table is None` — production decode is **paged** (page-128 TMA). The carve-out is **dense-only.**
3. `cu_seqlens_q is None` — production is varlen (`cu_seqlens_q` set).

Note also `interface.py:1015` `use_2cta_instrs = ... or use_dedicated_hd256_kernel` **forces 2CTA at the
interface level for all hd256**; the only escape is `hd256_use_2cta` passed to the kernel ctor at `:1610`.

### Measured 1CTA vs 2CTA (Thor, M=1, GQA 24:4, hd256, e4m3)
Probe `<scratch>`, run in the v10 container, dense non-paged, seqused_q
(non-causal → auto-1CTA; causal → auto-2CTA; for M=1 causal≡non-causal attention so the work is
identical, only the cluster differs). 50 iters, median+min, two independent runs:

| L | 1CTA (med / min) | 2CTA (med / min) | 2CTA/1CTA (med) | 2CTA/1CTA (min) |
|---:|---:|---:|---:|---:|
| 4096 | 123.8 / 119.6 μs | 188.8 / 186.4 μs | **1.53×** | 1.56× |
| 8192 | 190.7 / 145.5 μs | 320.6 / 308.6 μs | **1.68×** | 2.12× |

Per-token fit: 1CTA slope ≈ **0.016 μs/tok** (fixed ≈ 57 μs); 2CTA slope ≈ **0.032 μs/tok**
(fixed ≈ 57 μs). The fixed overhead is ~equal; **the 2CTA cluster roughly doubles the per-KV-token scan
cost**, so the gap widens with L (1.53× → 1.68×). The 2CTA L=8192 median (320.6) lands on the clean
production microbench value (324 μs, decode-microbench), confirming the 2CTA number is *not* inflated.

**Caveats (honest):** (a) the serving container was contending (steady 1 request running), so absolute
numbers are slightly hot, though the 2CTA value matches the clean run; (b) the 2CTA side was causal and
the 1CTA side non-causal — for M=1 the causal-mask work is negligible (1 valid row), so this is a minor
inflator (single-digit μs); (c) **the 1CTA path was measured on dense KV; the production paged path is
untested** — if paged 1CTA adds gather overhead, the win shrinks a bit. Even so, the cluster is clearly a
**large, avoidable** share of the hd256 decode cost.

**So: dropping 2CTA→1CTA is a real ~1.5–1.7× win.** This is the "cheaper intermediate win" (#5) — and it
is much larger than the modest 1.2× a static read of "M-tile waste is secondary" would have predicted.

---

## Answer 3 — FlashInfer's approach (the target architecture)

**FlashInfer's decode is a pure FMA GEMV kernel — no tensor cores, no tmem, SplitKV via the grid, state in
registers. `data/include/flashinfer/attention/decode.cuh`:**

- **Zero** `tcgen05/wgmma/tmem/mma.sync` references in the file (grep count = 0). It is FMA.
- `SingleDecodeWithKVCacheKernel` `:217`: `compute_qk` `:312` (Q·K → scores, a warp-level
  `__shfl_xor_sync` FMA reduction) and `update_local_state` `:332` (O += P·V, FMA). The per-thread state
  `st_local` (m, d, o) is in **registers** (`state_t<vec_size>` `:304`); K/V tiles are in smem
  (`:244-248`), multi-buffered by `num_stages_smem` with `cp_async` prefetch (`:280-346`). **No tmem.**
- **SplitKV across SMs:** `kv_chunk_idx = blockIdx.x` `:238` (KV split on the x axis), `kv_head_idx=
  blockIdx.y` `:236`. The dispatcher (`:691-713`) auto-sizes the split: `max_grid_size =
  num_blocks_per_sm * num_sm` (fill every SM), `kv_chunk_size = max(ceil_div(seq_len, max_num_kv_chunks),
  256)`, then a **combine** pass merges the per-chunk partial states (`:729+`).
- Block config (`:669-684`): `vec_size = max(16/sizeof(KV), head_dim/32)` (fp8 → 16), `bdx = head_dim/vec_size`,
  `bdy = GQA group size` (4), one block handles a **whole GQA group (4 q-heads) of one kv-head** and streams
  its KV through smem.

**Why FlashInfer wins at hd256 M=1:**
- **Memory-bandwidth roofline:** KV per step = `num_kv_heads(4) × L × head_dim(256) × 2(K+V) × 1B(fp8)`.
  At L=8192 that's **16 MiB**. FlashInfer's 73 μs = **~230 GB/s** ≈ ~84% of Thor's ~273 GB/s HBM3B →
  it is **bandwidth-bound and near-optimal**. FA4 2CTA at 324 μs is **~52 GB/s (19% of roofline)** →
  machinery-bound, not bandwidth-bound. The whole gap is machinery.
- **Cheap SplitKV works *because* the per-CTA cost is tiny** (a register state + small smem). Splitting
  the KV into `num_chunks` CTAs replicates almost nothing. This is the *opposite* of the FA4 2CTA kernel,
  where the per-CTA fixed cost (tmem-512 alloc, mbarrier init, cluster waves) is large, so the splitkv-sweep
  found SplitKV monotonically harmful (40/40 cells slower) — **that conclusion is specific to the 2CTA
  per-CTA cost and should be re-tested on a 1CTA kernel** (see DK-2 below).
- **Parallelism ceiling:** for GQA 24:4 the grid is `(num_chunks, 4 kv_heads)`. With 20 SMs and only 4 kv
  heads, effective SplitKV is ~20/4 ≈ 5 chunks (bounded by `num_blocks_per_sm` occupancy). It still fills the
  SMs via multi-CTA-per-SM; this is fine, not a blocker.
- **Avoids the GQA KV re-read:** one block handles a whole GQA group, so each kv-head's KV is streamed
  **once**. By contrast the FA4 hd256 kernel is **one CTA per q-head** (`pack_gqa` is asserted off,
  `FMHA:72,76`); each q-head CTA loads its group's KV via `head_idx // qhead_per_kvhead`
  (`FMHA:1067`), so for GQA 24:4 every kv-head's KV is loaded **4×** (L2-served if the 4 group-mates are
  co-scheduled, else DRAM). This is a residual factor both the 1CTA and 2CTA FA4 paths pay that
  FlashInfer does not — it is *not* the cluster (the 1CTA/2CTA probe is identical on this axis), but it is
  part of why 1CTA (~190 μs) does not reach FlashInfer (~73 μs). Unmeasured; `pack_gqa` is a separate,
  out-of-scope lever.

**Target to emulate:** a from-scratch kernel matching this design reaches ~73 μs — but that is *literally
what FlashInfer already is and already does in-tree (the microbench's B3 path).*

---

## Answer 4 — The light-kernel design (proposed)

Two distinct designs; I recommend doing **Design A (1CTA)** now and **Design B (GEMV) only via Option B**.

### Design A — "1CTA light MMA decode" (recommended, DK-1)
Keep the tcgen05 MMA kernel, drop the cluster, keep M-tile=128 (do **not** touch M-tile or tmem).

- **Grid:** `(ceil(S/128), q_heads, batch)` on **cluster (1,1)** — 24 CTAs for GQA 24:4, batch=1 (vs 12
  2-CTA clusters today). Fills the 20 SMs in ~1.2 waves either way; the win is per-CTA efficiency, not SM
  count.
- **MMA:** CtaGroup.ONE, M-mode=128 (floor allows 64; **stay at 128** — 64 adds a tmem/softmax partition
  refactor for ~0 gain). `qk_mma_tiler=(128,128,128)`, `pv_mma_tiler=(128,128,128)`, 2 K/N iterations.
- **tmem:** unchanged 512 cols (S 2×128 + O 256). *Not* shrinkable below ~384, and M=64 wouldn't help.
- **KV streaming:** keep the TMA paged-128 double-buffered load; the 1CTA form uses `k_stage=2, v_stage=3`
  (`:179-180`) — a shallower ring than the 2CTA `4/4`, which is part of why 1CTA streams ~2× faster/token.
- **Descale / cudagraph:** unchanged (same kernel, same FP8 descale fold, same cudagraph path).
- **Change surface:** extend `hd256_use_2cta` (`interface.py:1037-1046`) to fire for the production
  small-M decode shape (paged+varlen), and make the public path reach it (thread `seqused_q` through
  `flash_attn_interface.py` FA4 dispatch, or relax the `page_table is None` requirement). **Verify 1CTA+paged
  TMA works** (the carve-out has only ever run dense).

### Design B — "GEMV decode" (only via Option B; do NOT write in-tree)
A FlashInfer-style FMA kernel: register state, no tmem, cp_async KV streaming, SplitKV grid, combine pass,
GQA-group-per-block. This is the *only* design that reaches ~73 μs — and it is **a reimplementation of
FlashInfer**, which we already have. So it is scoped as **Option B: route the hd256 full-attn decode
layers to the existing FlashInfer paged-KV decode wrapper** (prefill stays on FA4), rather than writing a
second kernel. (See #5.)

---

## Answer 5 — Payoff, effort, risk, and the cheaper intermediate win

### Realistic payoff (L=8192, M=1) — honest numbers
| Path | L=8192 M=1 | vs FlashInfer (73) | gap closed (of 324→73) | effort |
|---|---:|---:|---:|---|
| Today (2CTA, production) | 324 μs | 4.5× | 0% | — |
| **A: 1CTA (dense measured)** | ~190 μs | **~2.6×** | **~53%** | ~1–3 d, ~50–100 LOC |
| A + M-tile→64 | ≈ ~190 μs | ~2.6× | ~53% | +~200–500 LOC, ~0 gain |
| **B / A-heavy: GEMV (≈ FlashInfer)** | **~73–90 μs** | **~1.0×** | **~100%** | Option B ~1–2 wk; new kernel ~500–1500 LOC / 1–3 wk |

- **A (1CTA) is a real, large intermediate win**: ~324 → ~190–210 μs on the production shape (210 accounts
  for paged-overhead risk), recovering ~53% of the decode gap at low effort. It is the **"cheaper
  intermediate win"** the task asks for — and it is *bigger* than the "secondary" label the old design doc
  gave M-tile work.
- **No in-tree MMA kernel reaches 73 μs.** The residual ~190→73 μs has two contributors: (a) the
  per-block tmem/TMA-MMA/softmax machinery (only a GEMV kernel removes it), and (b) the **GQA 4× KV
  re-read** (one CTA per q-head, `FMHA:1067`), which FlashInfer avoids via GQA-group-per-block. If
  self-containment (no second backend) is not mandatory, **Option B takes both residuals at far lower
  risk than writing a GEMV kernel.**

### Main risks
1. **1CTA+paged unverified (top risk for A).** The carve-out is dense-only; production is paged-128 TMA.
   The 1CTA paged path (cluster (1,1) + TMA paged gather) has never been exercised. Must be proven in DK-1
   before trusting the ~1.5–1.7×.
2. **1CTA unreachable from the public API today.** Requires threading `seqused_q` (or relaxing the
   `page_table is None` requirement) — a real plumbing change, though small.
3. **MMA min-M floor (64/128) is a non-issue for A** (we stay at M=128, which 1CTA supports) but **blocks
   any "M=1 rows via MMA"** — a hard reason the in-tree kernel can't be made M-optimal; only GEMV does.
4. **FP8 descale in a new path:** A needs none (same kernel); Option B needs none (FlashInfer handles it);
   a new GEMV kernel needs it (risk).
5. **cudagraph:** A reuses the proven FA4 capture; Option B must confirm FlashInfer decode is
   cudagraph-capturable — the `num_blocks_per_sm`/`cudaOccupancyMaxActiveBlocks` query at launch
   (`decode.cuh:707`) and the auto `num_chunks` likely need pre-computation at capture time.
6. **256-wide D:** the GEMV QK^T dot is a 256-FMA reduction per KV token — latency-sensitive; needs
   good memory-level parallelism (this is where FlashInfer's tuning lives and where a reimplementation
   could under-shoot 73 μs).

### Recommendation
1. **A (1CTA decode): GO** — best ROI. Ship 1CTA as the default for small-M hd256 decode.
2. **A+M64: NO-GO** — refactors the kernel for ~0 gain (M-independence, tmem M-independent).
3. **Full win to ~73 μs: GO via Option B** (decode → existing FlashInfer), **not** a new in-tree GEMV
   kernel (that only pays off if self-containment is a hard requirement).
4. Keep the A1a/A1b SplitKV plumbing; **re-test SplitKV on the 1CTA kernel** (the "wrong lever" verdict
   was for the 2CTA per-CTA cost; on 1CTA the per-CTA cost is ~1.6× lower, so SplitKV may finally help —
   a DK-2 experiment).

---

## Phased plan

**DK-1 — Prototype the 1CTA decode path (A)**
- Extend `hd256_use_2cta` to fire for the production small-M decode (paged+varlen, `max_seqlen_q` ≤ a small
  M, e.g. ≤ 8) and make the public FA4 dispatch reach it (thread `seqused_q` / relax the dense-only
  `page_table is None` term).
- **Verify 1CTA + paged-128 TMA** correctness and perf; A/B vs fp32 ref (tol `2e-2·ref_max + 1e-2`) and vs
  the 2CTA serial output.
- Measure on the *production paged shape* (not just dense) to confirm the ~1.5–1.7× survives.
- *Exit:* 1CTA paged decode is correct and ≥1.3× faster than 2CTA on the paged production shape.

**DK-2 — Validate + measure**
- Re-run `decode-microbench.py` (B1): target L=8192 M=1 324 → ~190–210 μs; L=256 ~unchanged (or slightly
  better).
- **Re-run the `splitkv-sweep.py` on the 1CTA kernel** — check whether SplitKV (num_splits>1) becomes
  effective now that the per-CTA cost is lower (the 2CTA verdict may not transfer). If it does, that is a
  bonus lever toward the residual gap.
- Full correctness sweep (causal/non-causal, GQA, paged, fp8 descale) + byte-identity for the
  unchanged prefill (2CTA) path.

**DK-3 — Integrate + decide the residual**
- Ship 1CTA as the default for small-M hd256 decode (2CTA retained for prefill/large-M); canary gate;
  E2E A/B at long context.
- In parallel, stand up **Option B** (route hd256 decode → existing FlashInfer) if the ~53% from DK-1/2 is
  judged insufficient — it is the lower-risk path to the full ~73 μs and the higher-value end state if
  self-containment is not a hard constraint.

---

## Provenance
- Kernel/tree: `<vfa-tree>/cute/` (live A1b tree; sha in `splitkv-sweep.md`).
- tcgen05 M-mode validation: `nvidia_cutlass_dsl/.../cute/nvgpu/tcgen05/mma.py:197-227`.
- tmem max: `.../cute/arch/tmem.py:28-31` (`sm_100→512`); FMHA `:144,:161-163,:181,:959`.
- 1CTA carve-out + API reachability: `interface.py:1014-1015,1037-1046,1610`; `flash_attn_interface.py:176,437-465`; `_flash_attn_fwd` `cute/interface.py:606,613`.
- FlashInfer: `data/include/flashinfer/attention/decode.cuh:217,236-238,304,312,332,669-684,691-713` (0 tensor-core refs).
- 1CTA-vs-2CTA measurement: `<scratch>` (dense, seqused_q, GQA 24:4, hd256 e4m3; 50 iters, med+min, 2 runs). Contended (co-located serving container busy, MemAvailable ~1–2.5 GB); 2CTA L=8192 median (320.6 μs) matches the clean production microbench (324 μs).
