# Why FlashInfer decode is ~3× faster than the FA4 GEMV kernel — final analysis

Shape: M=1, L=8192, GQA 24/4 (group 6), hd=256, Q bf16, K/V fp8 e4m3, dense + paged-128 KV.
16 MiB KV moved per call. All numbers below measured **today (2026-09-25 ~12:00) under
identical ncu conditions** (`--clock-control none --cache-control none`, L2 warm: KV fits the
32 MiB L2 → achieved "GB/s" is L2-fabric BW, up to ~300 GB/s; DRAM roofline 273 GB/s).

## 1. Side-by-side (same day, same ncu conditions)

| kernel | grid (CTAs) | threads | in-flight KV / CTA | t | BW |
|---|---|---|---|---|---|
| FA4 GEMV ns=1 (`sm100_hd256_decode_gemv`) | (4,1,1) = 4 | 96 | ~4.5 KB (2-stage × 6-row ring, K+V) | 1294 μs | **13.4 GB/s** |
| FA4 GEMV ns=20 (auto path) | (4,20,1) = 80 | 96 | ~4.5 KB | 184 μs (+11.5 μs combine) | **99.5 GB/s** |
| FA4 GEMV ns=64 | (4,64,1) = 256 | 96 | ~4.5 KB | 183 μs | **114.8 GB/s** |
| FI FA2 prefill, split-KV OFF (`tc_ns`) | (1,1,4) = 4 | 128 | **~192 KB** (384-row K + 384-row V tiles) | 243.6 μs | **69.8 GB/s** |
| FI FA2 prefill, split-KV ON (`tc`, the "73 μs / 219 GB/s" kernel) | (10,1,4) = 40 | 128 | **~192 KB** | **57.0 μs** | **300.8 GB/s** |
| FI GEMV (`BatchDecodeWithPagedKVCacheKernel`, tc=False) | (35,4,1) = 140 | 96 | ~2 KB (2-stage ring, paged) | 212 μs (+6 μs merge) | 83.0 GB/s |

Warps active/SM: GEMV ns20 ≈ 12; FI-gemv 14.1; FI-tc 7.4; FI-tc_ns 4.0 — i.e. the FI-tc
winner runs at *lower* occupancy than the GEMV variants. Occupancy is not the axis.

## 2. The two axes that explain the gap (disentangled by the 4-CTA pair)

**(a) CTA count (split-KV / ns).** Same GEMV kernel, shallow ring, 4 → 80 CTAs:
13.4 → 99.5 GB/s (**×7.4**). 80 → 256 CTAs plateaus (~100–115 GB/s): once every SM has
CTAs, adding more just serializes them on the SM.

**(b) Per-CTA in-flight bytes (pipeline depth).** Same CTA count (4), shallow 4.5 KB ring
vs deep 192 KB ring: 1294 μs vs 243.6 μs (**×4.8**). And at sufficient CTAs: GEMV 80 CTAs
(4.5 KB) = 99.5 GB/s vs FA2-tc 40 CTAs (192 KB) = 300.8 GB/s (**×3.2**). The strided
256 B/1024 B per-row access pattern (NHD paged layout, one 256 B row per KV row, 1024 B
stride) cannot hide per-row LPDDR/L2 access latency with only ~4.5 KB in flight per CTA;
it can with ~192 KB. (This matches the prior `splitkv-gemv-debug.md` finding that
concurrent strided streams don't add BW unless each stream is deep enough.)

**Null result — codegen/LDGSTS is NOT an axis.** Both the FA4 DSL `cp.async` (GEMV kernel:
`l1tex…op_ldgsts = 0`, all traffic in `op_ld`) and FlashInfer's C++ `cp.async.cg`
(FI-tc: `op_ld = 548 680` sectors, `op_ldgsts = 0`) lower to **plain LDG on sm_110a**.
Same instruction class, same 128-bit vector width, same access pattern. The gap is not
LDGSTS vs LDG.

## 3. Why the earlier "72 GB/s" number differs from today's 13.4 GB/s (co-tenancy)

The Sept-25 09:52 bench measured GEMV ns=1 at 228 μs (72 GB/s); the same kernel, same
code (interface sha identical), measures 1294 μs today (13.4 GB/s). The difference is
GPU co-tenancy: the desktop session (Xorg/gnome) currently shows **97% GPU util** — a
continuous GPU workload sharing the L2/fabric. Low-MLP, low-CTA kernels (4 CTAs, 12 warps
GPU-wide, issue-bound: `smsp…stalled_selected = 69%`, `wait = 19%`, `long_scoreboard ≈ 0`)
are hit hardest by the shared fabric; the 40-CTA FI-tc kernel is essentially unaffected
(300 GB/s now vs 219 GB/s on Sept 22). Practical consequence: **GEMV ns=1 performance is
fragile under co-tenancy**; the fix below is also the robustness fix.

Also: the 09:52 bench's flat ns legs (ns1/ns2/ns4/ns8 all ≈228 μs) are not reproducible
with the current code — ns scaling is verified working today (1294 → 184 μs, grids
(4,1,1) → (4,20,1) confirmed under ncu). Treat that JSON's ns>1 legs as suspect (likely
the env knob was not effective in that run's state).

## 4. Prescribed fix for the FA4 GEMV kernel (concrete)

**Fix 1 — dispatch: use the auto split (fill all 20 SMs) by default.**
`interface.py` already has `_gemv_auto_num_splits` (blocks_per_sm=4 → ns=20 at
kv_heads=4 → 80 CTAs) and the `gemv_split` plumbing + in-tree combine; the "72 GB/s"
corresponds to the forced ns=1 path. Make serving take the auto path (ns=20; the 128-row
chunk floor gives ns=20 at L=8192; combine adds 9–12 μs).
→ recovers 13.4 → ~100 GB/s (×7.4 under co-tenancy).

**Fix 2 — deepen the KV ring in `sm100_hd256_decode_gemv.py`.**
`self.stages = 2` (line 98) with `rows_per_stage = bdy×tile = 6` → ~4.5 KB in flight per
CTA. Raise to **`stages = 16`** (smem: 16 × 6 rows × 256 B × 2 = **48 KB** per CTA, well
under the 227 KB opt-in; 32 stages = 96 KB also fits if a bigger window is wanted).
The consumer loop already does the rolling pipeline (`cp_async_commit_group` per stage,
`cp_async_wait_group(2*S-1)` — lines ~424–465); only `stages`, the smem allocation
(`k_smem`/`v_smem` shapes at line ~271) and the prologue preload count change. No changes
to the FMA compute, epilogue, or split/combine logic.
→ expected to close the remaining ~×3 (99.5 → ~300 GB/s, i.e. FI-tc parity, at L2-resident
conditions), per the two controlled comparisons in §2(b).

**Verification plan (post-fix):** rerun `probe-gemv-ncu.py` ns=20 with
`gpu__time_duration.sum,lts__t_sectors.sum` (expect ≤ ~100 μs at L=8192), the
`gemv-decode-bench.py` dense+paged legs in a clean window, and correctness vs the
`decode-microbench.py` B3 reference (already in-tree via the existing combine tests).

## 5. One-paragraph answer

FlashInfer's "fast decode" is the FA2 tensor-core prefill kernel in split-KV guise:
**40 CTAs × ~192 KB of KV in flight per CTA**. The FA4 GEMV kernel at its forced ns=1
runs **4 CTAs × ~4.5 KB in flight per CTA** (2-stage × 6-row cp.async ring, 96 threads).
The 3× (up to 17× under GPU co-tenancy) is exactly the product of those two deficits:
×7.4 from CTA count (GEMV ns=1→ns=20: 13.4→99.5 GB/s) and ×3.2 from per-CTA in-flight depth
(4.5 KB vs 192 KB, shown at both 4 CTAs and at fill-the-SM CTA counts). Access pattern,
vector width, load instruction (both lower to LDG on sm_110a), and occupancy are NOT
differentiators — the ncu 4-CTA pair (1294 μs GEMV vs 243.6 μs FA2, same grid) isolates
pipeline depth alone at ×4.8. Fix: dispatch with auto ns (80 CTAs, +~10 μs combine) and
bump the GEMV ring from 2 to 16 stages (48 KB smem), which together should reach
FA2-tc parity (~57 μs / ~300 GB/s at L2-resident).

## 6. Postscript (2026-09-25, after the stages 2→16 fix landed) — prediction scorecard

Measured in a gated clean window (`gemv-ring-fix-bench.md`, same ncu conditions):

- **Confirmed:** the ns/CTA-count lever. ns=1→20 gave ×8.7 kernel BW
  (13.55 → 118.3 GB/s) and ×5.9 wall (1305 → 222.8 μs) — at least as large as
  the predicted ×7.4. ns=64 kernel BW reached 168.6 GB/s but wall regressed on
  combine cost.
- **Not confirmed:** the per-CTA in-flight-depth gain. The stages 2→16 ring
  (6 KB → 48 KB/CTA) is correctness-neutral (bit-identical output) and bought
  only **+12.9% kernel BW at ns=20** (118.3 vs 104.7 GB/s) — a wall wash
  (222.8 vs 221.4 μs); at ns=1 (4 CTAs) there is **zero** difference (the
  4-CTA regime is LSU-issue-bound, the shallow ring already covers the
  latency). The predicted "FA2-tc parity via ring depth" did not materialize:
  GEMV best (ns=20) is 222.8 μs vs FI 202.05 μs in the same window (~10%
  behind; both depressed by desktop co-tenancy).
- **Revised conclusion:** CTA count is the lever that was worth taking; ring
  depth matters only beyond L where the KV leaves the 32 MiB L2. Closing the
  remaining ~10% vs FI requires matching FI's *tile shape* (192 KB 384-row
  tiles, 128-thread blocks) — optional B3 in `AGENTS.md` — not a deeper ring.
