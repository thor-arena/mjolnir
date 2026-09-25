# DK-1a — 1CTA + paged-128 TMA decode: trigger, verdict, measurement

**Status: COMPLETE — GO.** 1CTA+paged is correct and 1.60–1.62× faster than 2CTA on the
production paged decode shape (exit criterion ≥1.3× met). The production wiring edit works
end-to-end (interface selects and executes the 1CTA binary). No kernel changes are required.

Provenance: probes in this directory (`probe_dk1a_1cta_paged.py`, `probe_dk1a_interleave.py`,
`inspect_dk1a_key.py`, `inspect_dk1a_serving_seq.py`); live-tree sources
`<vfa-tree>/` (A1a+A1b kernel). All container-side experiments ran in the
`vllm` container against the installed tree (identical to the live tree for the touched
files). Container `interface.py` was temporarily patched for the wiring verification and
**fully restored** afterwards (verified: 0 matches for the probe block, syntax OK). The live
serving process was never affected (module already in its memory).

---

## 1. The 1CTA trigger and how to reach it from the vLLM/FA4 path

### 1.1 Where the 1CTA form is selected

`cute/interface.py` (live-tree line numbers):

- `:1017` `hd256_varlen_b1 = cu_seqlens_q is not None and batch_size == 1`
- `:1018-1027` `hd256_l2_swizzle` (only for `qhead_per_kvhead == 1`)
- `:1028-...` `hd256_mask_residual`
- `:1037-1046` the carve-out:

```python
hd256_use_2cta = not (
    seqused_q is not None
    and cu_seqlens_q is None
    and page_table is None
    and not local
    and (
        (not causal and max_seqlen_q <= 1024)
        or (causal and max_seqlen_q <= 2048 and max_seqlen_k <= 2048)
    )
)
```

- `:1400-1403` all four flags (incl. `hd256_use_2cta`) enter the compile key.
- `:1605-1610` the kernel ctor: `BlackwellFusedMultiHeadAttentionForward(..., use_2cta=hd256_use_2cta)`.

Inside the kernel (`sm100_hd256_2cta_fmha_forward.py`), `use_2cta` controls exactly:
`:105` `cluster_size_m = 2 if use_2cta else 1`; `:179-180` `k_stage = 4/2`, `v_stage = 4/3`
(ring depth); `:507` `CtaGroup.TWO/ONE` (TMA op `:585`); `:171` `ex2_emu_freq` (2CTA only);
`:657` grid `round_up` to cluster shape; `:1805/:1808` tmem dealloc barrier (cluster vs CTA).
Everything else (MMA tiler, paged TMA, softmax, epilogue) is cluster-independent.

### 1.2 Why production decode never reaches it

The FA4 dispatch (`flash_attn_interface.py:437-465`, `fa_version == 4` branch) calls
`_flash_attn_fwd` with `cu_seqlens_q` (varlen) and `page_table=block_table` (paged) — and
**never passes `seqused_q`** (it is hardcoded `None` in the public API slots,
`:160` / `:391`). The carve-out requires `seqused_q is not None AND cu_seqlens_q is None
AND page_table is None` — all three fail for production decode (and `:1037` uses
`max_seqlen_q` ≤ 1024/2048 bounds aimed at dense prefill). Hence every production decode
compiles/runs the 2CTA form.

### 1.3 Minimal change to reach 1CTA decode (verified, §3.3)

No new public-API parameter is needed — the interface already sees `max_seqlen_q` and
`page_size`. In `interface.py:1037`, prepend a decode condition to the carve-out:

```python
# decode-shaped hd256 calls (small M) take the 1CTA form
hd256_decode_1cta = (
    max_seqlen_q is not None
    and max_seqlen_q <= 8                       # decode M (1..8); prefill is 1024+
    and not local
    and (page_size in (None, tile_n))           # dense or paged-128 TMA (only legal hd256 paged mode)
)
hd256_use_2cta = not (
    hd256_decode_1cta
    or (
        seqused_q is not None
        and cu_seqlens_q is None
        and page_table is None
        and not local
        and (
            (not causal and max_seqlen_q <= 1024)
            or (causal and max_seqlen_q <= 2048 and max_seqlen_k <= 2048)
        )
    )
)
```

(`page_size`/`tile_n` are both in scope at that point; the kernel's own assert
`paged_kv_non_tma = page_size not in [None, tile_n]` makes dense or page-128 the only
legal hd256 KV layout, so the last term is a defensive tautology on the hd256 path.)
Threshold ≤ 8 is the canary choice — any M ≤ 128 is structurally safe (MMA floor M=64),
but keep the first cut conservative.

---

## 2. Verdict: does 1CTA + paged-128 TMA work? **Yes.**

The paged KV path in the 1CTA form is fully functional. Static evidence (live-tree lines,
`sm100_hd256_2cta_fmha_forward.py`):

- Paged K/V tensor build `:353-366` (`mPageTable` rank-2 `(b, num_pages)` view;
  `max_seqlen_k_paged = mPageTable.shape[1] * page_size`) — cluster-independent.
- Paged TMA load branch `:1032-1033` (dense) vs `:1066-1069` (paged): per-CTA KV-head
  select `head_kv_coord = curr_block_coord[2][0] // self.qhead_per_kvhead` — cluster-
  independent.
- Page-indexed TMA copies: K0 `:1109-1120` (`tKgK[None, 0, iter, k_page_idx]`), Ki
  `:1132-1145` (prefetched `k_page_idx = mPageTable[batch_coord, kv_coord]` at `:1134`,
  plain 32-bit GMEM read in the load warp), V `:1149-1156` / `:1173-1175`
  (`tVgV[None, iter, 0, v_page_idx_prev]` — V reuses K[i-1]'s page, no extra GMEM read).
  None of this references the cluster; the TMA atom is `CopyBulkTensorTileG2SOp(cta_group)`
  (`:585`) which degenerates to a plain G2S in the 1CTA group.
- `tKgK/tVgV` partitioning uses `kv_cta_layout`/`block_in_cluster_coord_vmnk[1]`, which
  degenerate to single-CTA form for `cluster_shape_m = 1`; `PipelineTmaUmma` tx-count and
  the `tmem` dealloc barrier (`:1808` `arrive_and_wait`) are CTA-local in 1CTA.
- Grid/scheduler: `:657` `grid = cute.round_up(grid, cluster_shape_mnk)` (no-op for
  cluster (1,1)); `:972` `mma_block_coord` `// cluster` divides consistently.

Empirical evidence (decisive): `probe_dk1a_1cta_paged.py` runs the **production call shape**
(paged-128, varlen b1, shuffled *physical* page table — catches page-index bugs, non-unity
per-batch/head descales, e4m3, GQA 24/4, causal, M=1) in both cluster forms and against an
independent fp32 dequant reference. Results (§3.1): 1CTA output matches the reference within
tolerance **identically** to 2CTA (L=8192: byte-identical to the 2CTA output).

GQA note (correction to the task statement): with 24 q-heads / 4 kv-heads,
`qhead_per_kvhead = 6` — each KV head's KV stream is loaded by **6** q-head CTAs (the
"4×" figure was the kv-head count, not the group size). This 6× re-read (L2-absorbed when
group-mates co-schedule) is identical in the 1CTA and 2CTA forms.

---

## 3. Measurement

Environment: `vllm` container, NVIDIA Thor, **continuous ~97–98% background serving load**
(live 4-request generation, spec-decoding — the server was never idle during this task).
Absolute medians therefore include queuing; the trustworthy quantity is the **per-iteration
interleaved pair ratio** (same instantaneous window for both modes) plus the min values.
The 1CTA leg was verified to be a true 1CTA binary (ctor `use_2cta=False` logged; binary
timing matches the standalone-verified 1CTA number).

### 3.1 Correctness (`probe_dk1a_1cta_paged.py`) — PASS

| L | 1CTA out vs fp32 ref (max err) | 2CTA out vs ref | tol (`2e-2·ref_max+1e-2`) | 1CTA vs 2CTA out (max abs) | LSE err (1CTA/2CTA) |
|---|---|---|---|---|---|
| 2048 | 5.49e-3 | 5.49e-3 (identical) | 1.52e-2 | 9.77e-4 (fp8 rounding) | 1.0e-6 / 4.8e-6 |
| 8192 | 3.22e-3 | 3.22e-3 (identical) | 1.35e-2 | **0.0 (byte-identical)** | 9.5e-7 / 9.5e-7 |

### 3.2 Performance, production paged shape (`probe_dk1a_interleave.py`, interleaved pairs)

| shape | L | 2CTA med (μs) | 1CTA med (μs) | 2CTA min | 1CTA min | pair-ratio med (p5–p95) |
|---|---|---|---|---|---|---|
| **paged (production)** | 2048 | 91.2 | **57.3** | 84.7 | 49.7 | **1.596** (1.368–1.776) |
| **paged (production)** | 8192 | 273.6 | **168.9** | 266.9 | 155.4 | **1.623** (1.409–1.743) |
| dense (DK-0 shape) | 8192 | 273.9 | **166.9** | 267.5 | 155.5 | **1.646** (1.464–1.759) |

- **The ~1.5–1.7× survives on the production paged shape** — exit criterion (≥1.3×) met.
- **Paged adds no penalty**: 1CTA paged 168.9 vs dense 166.9 μs (and 2CTA paged 273.6 vs
  dense 273.9) at L=8192 — the page-128 TMA path is cost-neutral vs dense.
- **The ~190 μs claim holds**: 1CTA L=8192 measured **168.9 μs median under ~98% background
  load** — faster than DK-0's clean-environment 190.7 μs dense number (1CTA's smaller CTA
  footprint queues less). Absolute numbers are environment-dependent (2CTA here is 273.6 vs
  DK-0's clean 324 μs); the pair ratio is the comparable quantity and matches DK-0's
  1.53–1.68× range.

### 3.3 Production wiring verification (the §1.3 edit, applied to the container copy)

- Clean process: patched interface, one production decode call → exactly **one** cache
  entry, compile-key last slot `(varlen_b1, l2_swizzle, mask_residual, use_2cta) =
  (True, False, True, **False**)`; measured binary **56.4 μs med** at L=2048 (2CTA: 91.2) —
  the 1CTA binary is what production caches and executes.
- Serving-like sequence (`inspect_dk1a_serving_seq.py`): compile a 2CTA **prefill** binary
  first (dense q=256, kv=8192 → `use_2cta=True` key), then the paged decode → decode cache
  entry key ends `use_2cta=False`; measured decode binary **171.2 μs med** at L=8192
  (2CTA decode would be ~274 μs). **Prefill (2CTA) and decode (1CTA) coexist correctly in
  one process** — the production deployment path is safe.

---

## 4. Caveats

1. **Contention**: all absolute numbers above are under sustained ~97–98% serving load
   (the live server never idled during this task). Ratios and min values are the reliable
   signal; absolute medians are upper bounds.
2. **Double-ctor artifact (unexplained, probe-only)**: when a process first compiles a
   2CTA binary for the *same decode shape* (ctor-forced by the probe) and then clears the
   cache and recompiles that same shape, the interface emits **two** kernel-ctor calls in
   one `_flash_attn_fwd` invocation (`use_2cta=False` then `True`) and the cached entry is
   the 2CTA binary. Mechanism not identified (single ctor call site, no retry/loop in the
   miss branch; `trace_dk1a_double_ctor.py`/`trace_dk1a_calls.py` reproduce/trace it).
   It does **not** occur in a clean process or in the serving-like prefill→decode sequence
   (§3.3), so production is unaffected. Flagged for DK-1b: if the team ever wants a runtime
   2CTA↔1CTA A/B toggle on identical shapes, this needs root-causing first.
3. **Restoration**: container `interface.py` reverted to stock after the wiring test
   (snapshot of the patched version: `<scratch>`); live server
   process never loaded the patch.

---

## 5. DK-1b plan (concrete)

1. **Apply the §1.3 edit to the live tree** (`vfa/cute/interface.py:1037`) — the exact
   verified diff; 6 added lines, 1 re-wrapped block. No kernel file changes (verified
   unnecessary).
2. **Gate for canary**: optionally wrap `hd256_decode_1cta` in an env flag
   (e.g. `VLLM_FLASH_ATTN_HD256_DECODE_1CTA=1`) for the first deployment; flip to default-on
   in DK-3.
3. **Regression A/B**: adopt `probe_dk1a_1cta_paged.py` as a canary test — paged + shuffled
   pages + non-unity descales + GQA + fp32 ref, asserting 1CTA-vs-2CTA ≤ `2e-2·ref_max+1e-2`
   and LSE ≤ 1e-4.
4. **Correctness sweep (with DK-2)**: causal/non-causal, M=1/2/4/8 (the threshold boundary),
   batch>1 varlen (SingleTileVarlenScheduler at cluster (1,1)), paged + dense, unit and
   non-unit descales; byte-identity for the unchanged prefill (2CTA) path.
5. **Measure (with DK-2)**: `decode-microbench.py` targets L=8192 M=1 → ~190–210 μs
   (predicted ~170–190 from this data); `splitkv-sweep.py` on the 1CTA kernel (SplitKV may
   become effective now that per-CTA cost dropped — DK-0's 2CTA verdict may not transfer).
6. **Optional tuning (not required)**: deepen the 1CTA K/V rings (`k_stage 2→3`,
   `v_stage 3→4`; SMEM headroom exists) — only if DK-2 shows the residual gap; paged is
   already cost-neutral.

**Bottom line:** DK-1a is a **GO** — 1CTA+paged-128 TMA decode is correct, reaches ≥1.3×
(measured 1.60–1.62× on the production paged shape), needs a 6-line interface edit, and the
edit is verified end-to-end in a serving-like compile sequence.
