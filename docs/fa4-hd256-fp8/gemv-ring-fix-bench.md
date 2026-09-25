# GEMV KV-ring depth fix (stages 2→16): clean-window re-measurement

**Date:** 2026-09-25 · **Device:** NVIDIA Thor (CC 11.0, 20 SMs, 32 MiB L2, LPDDR5X ~273 GB/s DRAM roofline)
**Fix:** one line in `vfa/cute/sm100_hd256_decode_gemv.py` — `self.stages = 2` → `self.stages = 16`
(48 KB dynamic smem ring instead of 6 KB; confirmed in compiled kernel:
`launch__shared_mem_per_block_dynamic = 49.15 KB` at ns∈{1,20,64}).

## TL;DR verdict

1. **The fix is correctness-neutral.** st16 output is bit-identical to st2 (max_abs = 0.0 at
   L=5/128/8192) and matches the fp32 reference within fp8 noise.
2. **The ns (split) lever is the real performance lever**: ns=1→20 gives ×8.7 achieved BW
   (13.5→118.3 GB/s L2-fabric) and ×5.9 wall (1305→223 μs). The auto path (ns=20 → 80 CTAs)
   captures it.
3. **The stages lever does NOT deliver its predicted gain.** At ns=20, st16 vs st2 kernel BW is
   +12.9% (118.3 vs 104.7 GB/s) but wall-clock difference is 0.6% (noise). At ns=1 there is no
   difference at all. The ×3–4.8 in-flight gain predicted in `gemv-fi-gap-final.md` did not
   materialize at L=8192.
4. **GEMV does not beat FlashInfer in wall-clock.** Best GEMV (ns=20) = 222.8 μs vs
   FI = 202.05 μs in the same clean window → **GEMV is ~10% slower.** The fix does not close
   the FI gap; FI (FA2-tc, 192 KB tile) remains the faster kernel at L=8192.

## Window / gating

- Wall bench: gated clean window — 6 consecutive 0/0 vLLM samples (2 s poll) opened the window;
  a direct 0/0 re-check ran before **each** of the 9 timed bursts; all passed.
  `VERDICT: CLEAN` (window clean, 0 dirty samples).
- ncu points: per-run vLLM load logged. st16_ns64, st2_ns1, st2_ns20 at 0/0.
  st16_ns20 re-run at 0/0 (first pass caught 1.0 running — my own agent inference).
  st16_ns1 ran with 1.0 running, but that point is insensitive to vLLM co-tenancy
  (1.27 ms in both the load=1 and the rerun; see `gemv-fi-gap-final.md` §4).
- **Not gated (caveat):** desktop Xorg/gnome GPU co-tenancy. It depresses absolute achieved BW
  for *every* kernel in the window (FI measures 202 μs today vs 73 μs on Sept 22 — same
  co-tenancy effect documented for GEMV). Same-window relative comparisons are valid;
  absolute GB/s are the lower end of the device's capability.

## Correctness (verify-gemv-stages.py, gated run)

| check | L=5 | L=128 | L=8192 |
|---|---|---|---|
| st16 vs fp32 ref (max_abs) | 7.7e-3 | 1.9e-3 | 2.4e-4 |
| st16 vs st2 (max_abs) | **0.0** | **0.0** | **0.0** |

Ring depth changes load timing only, never the per-token FMA order → bit-identical, as expected.
Original `verify-gemv-decode.py`: 15/15 pass with stages=16 as default.
Auto-ns at L=8192: `_gemv_auto_num_splits(4,1) = 20` → 80 CTAs (unchanged by the fix).

## Wall-clock (L=8192, dense fp8, gated clean window, med of 300)

| leg | wall med (μs) | nominal KV BW |
|---|---|---|
| st16 ns1 (4 CTAs) | 1305.4 | 12.85 |
| st16 ns4 | 365.8 | 45.87 |
| st16 ns8 | 316.2 | 53.07 |
| st16 ns20 (80 CTAs) | **222.8** | 75.30 |
| st16 ns64 (256 CTAs) | 233.5 | 71.84 |
| st16 auto | 224.2 | 74.83 |
| st2 ns1 (pre-fix) | 1277.7 | 13.13 |
| st2 ns20 (pre-fix) | 221.4 | 75.77 |
| **flashinfer** | **202.05** | 83.04 |

ns=64 wall regresses vs ns=20: combine kernel cost (64 partials) + launch overhead eat the
kernel gain. The auto path is the sensible operating point.

## ncu achieved BW (L2-fabric convention, med of 4 profiled launches)

`achieved BW = lts__t_sectors.sum × 32 B / gpu__time_duration.sum` — same convention as
`gemv-fi-gap-final.md` (on CC 11.0 `dram__bytes.sum` is n/a; sectors × 32 B is the L2-fabric
traffic; DRAM roofline 273 GB/s shown for reference only).

| config | kernel (μs) | L2-fabric BW (GB/s) | % of DRAM roofline | vs pre-fix (Δ BW) |
|---|---|---|---|---|
| st16 ns1 | 1270 | 13.55 | 5.0% | st2: 13.64 (−0.8%, noise) |
| st16 ns20 | 179.6 | **118.3** | 43.3% | st2: 104.7 (**+12.9%**) |
| st16 ns64 | 183.2 | 168.6 | 61.8% | — |
| st2 ns1 | 1250 | 13.64 | 5.0% | — |
| st2 ns20 | 174.7 | 104.7 | 38.4% | — |

## Lever analysis

**Lever 1 — ns (CTA count), stages fixed at 16.** ×8.7 BW and ×5.9 wall from ns=1→20.
ns=20→64 adds +42% kernel BW but *wall regresses* (combine + launch overhead). Plateau
behavior consistent with the prior report: once all 20 SMs are covered, BW scales sub-linearly
as the L2/DRAM pipe saturates.

**Lever 2 — ring depth (stages 2→16), ns fixed.**
- ns=1 (4 CTAs): zero difference (13.55 vs 13.64 GB/s; wall 1305 vs 1278 μs — st16 is if
  anything 2% *slower*, within noise). With 4 CTAs × 96 threads the kernel is LSU-issue-bound
  per CTA; the 6 KB st2 ring already covers the load latency at 4 CTAs — there is no
  in-flight deficit for st16 to fill.
- ns=20 (80 CTAs): +12.9% kernel BW (118.3 vs 104.7 GB/s) — the 48 KB ring keeps more of each
  CTA's stream in flight against the shared L2 fabric — but the wall-clock difference is
  0.6% (222.8 vs 221.4 μs): combine/launch overhead and the non-KV L2 traffic dominate the
  ~15 μs kernel-level difference.

**Why the predicted ×3–4.8 didn't show:** `gemv-fi-gap-final.md` extrapolated from the
4-CTA ncu context where per-CTA in-flight bytes looked limiting. The clean-window data shows
that regime is actually issue-rate-limited (zero stages gain at ns=1), and the 80-CTA regime
is already aggregate-saturated enough that a 8× per-CTA ring depth buys only +13% kernel BW.
The FI-tc kernel's edge is its *tile shape* (384-row K + 384-row V = 192 KB/CTA, 128-thread
blocks, 40 CTAs), not merely raw in-flight bytes.

## Combined fix vs FlashInfer (same clean window)

| | wall med (μs) | verdict |
|---|---|---|
| GEMV st16 auto (ns=20) | 224.2 | 11.1% slower than FI |
| GEMV st16 ns20 | 222.8 | **10.3% slower than FI** |
| FlashInfer (FA2-tc) | 202.05 | baseline |

**GEMV does not match or beat FI at L=8192.** The stages fix does not change the ranking;
it was never going to by itself — the ns lever already did its job in the pre-fix auto path.

## Honest assessment / recommendation

- **Keep or revert?** stages=16 is harmless (correctness-neutral, bit-identical), costs 42 KB
  extra smem/CTA (still fits 4 CTAs/SM), and is the right direction for **larger L** where KV
  stops fitting in 32 MiB L2 and per-CTA in-flight depth genuinely limits DRAM streaming.
  At L=8192 it is a wash in wall-clock. Suggestion: keep the fix, but do *not* attribute the
  ns-scaling numbers to it — the ns plumbing was already in the pre-fix tree.
- **If closing the FI gap is the goal**, the lever is the tile/CTA shape (FI's 192 KB
  384-row tiles, 128-thread blocks), not ring depth. That is a kernel redesign, not a
  one-line fix.
- **Measurement caveat to carry forward:** absolute BW in any window with desktop co-tenancy
  is ~2× pessimistic vs an exclusive window (FI: 202 μs today vs 73 μs Sept 22). Future
  comparisons should note the co-tenancy state.

## Files

- Fix: `<vfa-tree>/cute/sm100_hd256_decode_gemv.py` (stages=16, comment in place)
- Wall bench: `../../docker/vllm-thor/fa4-gemv-kernel/gemv-ring-fix-bench.py` (gated; `--no-gate` for smoke)
- ncu driver: `../../docker/vllm-thor/fa4-gemv-kernel/ncu-gemv-ring.py` (`NCU_NS`/`NCU_STAGES` env knobs)
- Verify: `../../docker/vllm-thor/fa4-gemv-kernel/verify-gemv-stages.py` (6/6 GO)
- Numbers: `../../benchmarks/raw/gemv-ring-fix-bench.json` (wall results; ncu table in this md —
  JSON is root-owned from the container run, ncu block to be merged on the host)
