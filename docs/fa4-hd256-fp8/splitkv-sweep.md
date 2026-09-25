# A1c-1 — hd256 SplitKV sweep: num_splits × KV length (per-step performance proof)

Date: 2026-09-23 · Thor (sm_110a, capability (11,0), 20 SMs, unified memory) · torch 2.13.0+cu130
Raw data: `splitkv-sweep-results.json` (5 per-ns partials + merge) · harness: `splitkv-sweep.py`
(`run-splitkv-sweep.sh`, one docker run per num_splits, fresh `TMPDIR` JIT cache per run — the
interface compile key keys on `is_split_kv` only, so a shared cache would bake in a stale
num_splits) · attribution: `splitkv-attribution.py` / `splitkv-attribution.json`.

## Server state measured against

Co-located serving container (docker `vllm`, image tag `qwen38-sm110-v9`, port 6001 — the same
endpoint the decode-microbench called "the v10 server") was **0 running / 0 waiting at start AND
end of every one of the 7 measurement processes**, and per-case pre/post checks stayed clean
(`clean=True` on all 55 timed cases). `MemAvailable` 2.9–3.8 GiB throughout (A1b-2 unified-memory
guard). One in-session outlier (ns=1 re-run #2) was ~7–10% high at L≥2048 — a transient warm state;
two bracketing ns=1 runs (in-session first / final) agree within ≤3% on every row, so **no session
drift** and all ns-vs-ns comparisons are in-session pairings.

## Method

- Full per-step cost = whole `flash_attn_varlen_func` call: for ns>1 that is **kernel +
  `_flash_attn_fwd_combine`** (+ partial-buffer allocs) — exactly the production shape A1c will
  have. Timing verbatim from `decode-microbench.py`: 20 warmup + 100 timed iters, CUDA events,
  per-iter sync, median.
- num_splits>1 driven by the A1b TEST-ONLY knobs, exactly as `probe_a1b2_splitkv.py`: kernel env
  `A1B_TEST_NUM_SPLITS` (read at JIT trace) + in-process `_get_fwd_config` clamp lift. Kernel
  binary identical to A1b tree (sha `d173bca1a993…`).
- Shapes (GQA 24/4, hd256, page-128 TMA, e4m3 + non-unity descales 1.06/1.125/1.25, causal):
  - `M1b1` batch=1, q=1 — b1 static-grid path, the exact decode-microbench B1 anchor (ns=1 only;
    SplitKV is asserted off on this path, see §7).
  - `M1s` batch=2, q=1 — varlen non-b1 path, **SplitKV-capable**, M=1/seq (primary).
  - `M4s` batch=2, q=2 — varlen non-b1, 4 queries/step (MTP M=4 analogue; a single 4-query
    sequence is the b1 path → assert).
- Every case: fp32 dequant causal-GQA reference check first (tol 2e-2·ref_max+1e-2). **All 55
  cases ref_ok=True**, max_err 2.7e-3…2.0e-2 (fp8-P split paths at ns=2/4 slightly above ns=1,
  as A1b-2 predicted; all within e4m3 tolerance).

## 1. Sanity check — M1b1 anchor vs the recorded decode-microbench baseline

median μs per step, batch=1 q=1 (identical entry point / kernel binary / geometry):

| L | recorded (B1) | in-session run | final re-run | Δ vs recorded |
|---:|---:|---:|---:|---:|
| 256 | 83.79 | 82.42 | 85.02 | −1.6% / +1.6% |
| 1024 | — (new point) | 106.02 | 106.10 | consistent interpolation |
| 2048 | 136.56 | 135.76 | 138.46 | −0.6% / +1.4% |
| 4096 | 197.34 | 196.54 | 197.44 | −0.4% / +0.05% |
| 8192 | **324.08** (p95 354.11, min 270.02) | **340.53** | **336.46** | **+5.1% / +4.0%** |

**Verdict: reconciled.** L≤4096 reproduce within ≤1.6%; L=8192 lands +4–5% high in both
independent runs — inside the recorded run's own median→p95 band (324→354) and the box's
observed inter-session variance. Same harness, same binary, +5% at the longest L only. All
sweep conclusions below use in-session ns=1 pairings, which are drift-free (two ns=1 runs agree
to 0.1% on the M1s row: 499.58 vs 498.86).

## 2. Time tables — median μs per step (all clean, ref_ok)

### M1s (primary; batch=2, q=1 — the SplitKV-capable M=1 shape)

| L | ns=1 | ns=2 | ns=4 | ns=8 | ns=16 | best ns | ns=2/serial | ns=16/serial |
|---:|---:|---:|---:|---:|---:|:---:|---:|---:|
| 256 | 107.44 | 173.82 | 212.56 | 301.89 | 464.74 | **1** | 1.62× | 4.32× |
| 1024 | 144.62 | 207.15 | 290.14 | 470.72 | 639.84 | **1** | 1.43× | 4.42× |
| 2048 | 195.60 | 257.63 | 343.06 | 506.08 | 864.43 | **1** | 1.32× | 4.42× |
| 4096 | 294.90 | 359.66 | 441.14 | 601.82 | 927.60 | **1** | 1.22× | 3.15× |
| 8192 | **499.58** | **619.17** | 653.55 | 815.65 | 1128.59 | **1** | **1.24×** | **2.26×** |

### M4s (secondary; batch=2, q=2 — 4 queries/step)

| L | ns=1 | ns=2 | ns=4 | ns=8 | ns=16 | best ns |
|---:|---:|---:|---:|---:|---:|:---:|
| 256 | 103.94 | 233.26 | 210.22 | 297.66 | 465.18 | **1** |
| 1024 | 143.26 | 204.96 | 293.39 | 469.90 | 640.05 | **1** |
| 2048 | 194.21 | 300.69 | 345.60 | 506.32 | 862.54 | **1** |
| 4096 | 295.97 | 418.45 | 450.78 | 600.59 | 928.66 | **1** |
| 8192 | **497.57** | **577.36** | 649.97 | 823.70 | 1126.61 | **1** |

## 3. Headline — L=8192, M=1

- **Serial (ns=1): 499.58 μs. Best num_splits (ns=2): 619.17 μs → 1.24× SLOWER, not faster.**
  ns=4/8/16: 653.5 / 815.7 / 1128.6 μs — monotonically worse.
- No win toward FlashInfer's 72.53 μs (B3 M1 L=8192, recorded): serial is 6.9× of it; the best
  split config is 8.5×. The "≥2× toward FlashInfer" bar is not met by any ns.
- L=4096: 294.90 → 359.66 (ns=2), 1.22× slower. L=2048: 195.60 → 257.63 (ns=2), 1.32× slower.
- For the production c1 shape (M1b1, b1 path): L=8192 = 340.5 μs vs FlashInfer 72.5 = 4.7× — and
  SplitKV is architecturally unavailable there (§7).

## 4. Optimal num_splits per L — the "knee"

**Optimal ns = 1 at every L (256…8192), both M shapes.** There is no saturation knee in the
helpful direction: additional splits are *monotonically harmful* at every L. Cumulative penalty
above serial (M1s): L=256 → ns=2/4/8/16 = +66/+105/+195/+357 μs (≈ +20…39 μs per added split);
L=8192 → +120/+154/+316/+629 μs (ns=2 alone is +60 μs/split, then +40 and +39 μs per split over
the 4- and 8-split steps) — the per-split penalty at L=8192 is comparable to or larger than at
L=256, i.e. more splits do **not** amortize against longer KV. The hd256 2CTA kernel's per-CTA
fixed cost (warp-specialized setup, tmem-512 alloc, mbarrier init, cluster launch waves) is so
large that splitting a KV scan into ≤4-block chunks (ns=16 at L=8192: 4 blocks/CTA) multiplies
that fixed cost 16× while the useful work per CTA shrinks 16×.

## 5. Attribution (split vs serial, combine vs kernel) — `splitkv-attribution.json`

At M1s L=8192 (fresh process, in-process serial partner + isolated combine):

| measurement | μs |
|---|---:|
| serial ns=1 (full call) | 499.92 |
| split ns=2 (full call = kernel + combine) | 564.00 (+64.1) |
| `_flash_attn_fwd_combine` alone (interface-identical buffers) | 16.61 |
| → kernel-side share of the penalty | ≈ +47.5 μs |

So the combine pass is a real but **secondary** cost (~26% of the ns=2 penalty at L=8192);
~74% of the slowdown is inside the split kernel itself (grid/cluster waves + per-CTA overhead +
fp32 partial-O write traffic). Split confirmed active: out(ns=2) vs out(ns=1) max_abs =
3.78e-3 (≠0, within the A1b-2 5e-2 bound).

## 6. Structural finding — where SplitKV can even run

`hd256_varlen_b1 = cu_seqlen_q is not None and batch_size == 1` (interface.py:1017) routes
batch=1 varlen to the b1 static-grid scheduler, on which the kernel **asserts SplitKV off**
(`sm100_hd256_2cta_fmha_forward.py:438-443` — the b1 grid has no KV-split axis). Consequences:
- The primary v10 decode shape — concurrency-1 MTP verify (batch=1, q=M) — **cannot use
  SplitKV at all** with the current kernel; only batch≥2 (concurrency ≥2) reaches the
  SplitKV-capable `SingleTileVarlenScheduler` path.
- The sweep's split rows are therefore the batch=2 shapes (the closest SplitKV-capable
  analogues of M=1 / M=4 decode steps).

## 7. Recommendation — A1c-2 threshold policy

**The data does not support any policy that enables num_splits>1. Recommended policy:
`num_splits = 1` for all L (a degenerate threshold — do not ship SplitKV enablement on the
hd256 path).**

Rationale:
1. Every (L, ns>1) cell is slower than serial — 40/40 cells across L∈{256…8192} ×
   M∈{1s,4s} × ns∈{2,4,8,16}. The best case (ns=2, L=8192) is still +24%; there is no length at
   which splitting breaks even.
2. The cost structure is per-split-overhead-dominated at every L (no amortization knee): more
   splits → more cluster waves over 20 SMs with a per-CTA fixed cost that does not shrink as
   the KV chunk shrinks.
3. The decode gap vs FlashInfer (4.7–6.9× at L=8192) is entirely unaddressed by this lever;
   SplitKV makes it worse. The viable decode levers remain the ones in
   `decode-microbench.md`: backend split (decode→FlashInfer for the hd256 full-attn layers) or
   an in-kernel 1CTA decode-tiling fix — neither of which is SplitKV.
4. For A1c itself: keep the A1a/A1b plumbing (interface clamp lift + num_splits slot + correct
   combine — proven correct this sweep: all ref checks pass) so the door is open, but the
   production default stays ns=1, and the A1b TEST-ONLY knobs (`A1B_TEST_NUM_SPLITS`, the
   monkey-patch convention) should be removed as planned. If a future 1CTA/decode-tuned hd256
   kernel lands (cheaper per-CTA cost), re-run this sweep — the harness is in
   `splitkv-sweep.py` and takes ~10 min end-to-end.

## Provenance

- Kernel/interface source shas (identical across all 7 processes): kernel `d173bca1a993…`,
  interface `b9d41e7b2f06…` (live A1b tree at `<vfa-tree>`).
- Per-ns partials: `splitkv-sweep-ns{1,2,4,8,16}.json` (ns1 file = final re-run; in-session
  first-run numbers in the tables above, both agree ≤3%). Attribution: `splitkv-attribution.json`.
- Raw logs: `<log>`, `sweep-ns1-rerun.log`, `sweep-ns1-rerun2.log`,
  `sweep-attr2.log` (ephemeral — tables above are the canonical record).
- Run wall: ~6 min for the 5-ns sweep + ~1 min per ns=1 re-run + 13 s attribution; all runs
  single-digit-minute after JIT, server 0/0 throughout.
