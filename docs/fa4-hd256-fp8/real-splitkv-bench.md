# REAL-SPLITKV — 1CTA SplitKV on the hd256 path: enablement, correctness, verdict

Date: 2026-09-24 · Tree: live `<vfa-tree>/` (v11-equivalent) · GPU: NVIDIA Thor (sm_110a, 20 SMs)

## 1. Question

The hd256 kernel's SplitKV machinery (per-split LSE plane, `SmemGmemCopyAtomLSE`, combine
epilogue) was previously **unreachable**: the v11 interface forced `num_splits=1` on every
hd256 call, and the 1CTA decode path (batch=2 varlen) had no way to request splits. Does
enabling real SplitKV on the 1CTA path give a performance win for long-context decode,
or is the FlashInfer path the only real SplitKV win?

## 2. What was changed (live tree)

`cute/interface.py` (sha `12efaf80…`):
- `_get_fwd_config`: the hd256 `num_splits=1` clamp lifted — an explicit `num_splits>1`
  is honored only for decode-shaped calls (assert `max_seqlen_q<=8`, non-local).
- `_flash_attn_fwd`: test-only env knob `VLLM_FA4_HD256_DECODE_NUM_SPLITS=N` (labeled
  TEST KNOB, not policy), applied for decode-shaped hd256 calls with caller `num_splits<=1`.
- `_flash_attn_fwd`: `is_split_kv` on the hd256 kernel now asserts the 1CTA decode gate
  **and** the varlen non-b1 path (loud failure before the kernel's own asserts).
- hd256 compile key: `num_splits` appended (per-ns binaries coexist in one process).
- `num_splits` passed to the kernel **constructor** (trace-time constant, same pattern
  as `is_split_kv`/`use_2cta`).

`cute/sm100_hd256_2cta_fmha_forward.py` (sha `5beae692…`):
- `__init__` gained `num_splits: int = 1` (`self.num_splits`); the A1b test-only
  `A1B_TEST_NUM_SPLITS` env block removed.

**Empirical gotcha (kept for the record):** applying `num_splits` as a trailing FFI
argument of `__call__` breaks CuTe DSL IR verification
(`'cute.make_tile' op using value defined outside the region`) — the DSL treats the new
compile arg as a runtime value inside the device region. The constructor-constant
pattern is the only working shape. The new compile key also let one process hold
ns=1/2/4/8 binaries simultaneously (A1b needed one process per ns).

## 3. Correctness (GO)

Probe `probe_real_splitkv.py` → `real-splitkv-probe.json` (M1s: batch=2, q=1, paged-128,
e4m3, GQA 24/4, causal, non-unity descales; fp32 reference; ctor hook confirmed every
leg ran 1CTA):

| L | ns | out vs fp32 ref (max abs / tol) | out(ns) vs out(ns=1) max abs | final LSE vs fp32 (max) |
|---:|---:|---|---|---:|
| 256 | 1 | 1.305e-2 / 2.797e-2 | — | 9.5e-7 |
| 256 | 2 | 1.577e-2 | 1.367e-2 | 9.5e-7 |
| 256 | 4 | 1.577e-2 | 1.367e-2 | 9.5e-7 |
| 2048 | 1 | 5.794e-3 / 1.925e-2 | — | 9.5e-7 |
| 2048 | 2 | 5.452e-3 | 5.371e-3 | 9.5e-7 |
| 2048 | 4 | 5.629e-3 | 6.348e-3 | 9.5e-7 |
| 8192 | 1 | 3.150e-3 / 1.294e-2 | — | 1.9e-6 |
| 8192 | 2 | 2.788e-3 | 2.930e-3 | 1.9e-6 |
| 8192 | 4 | 3.050e-3 | 4.150e-3 | 9.5e-7 |
| 8192 | 8 | 2.692e-3 | 3.662e-3 | 9.5e-7 |

- All legs within fp8 tolerance. Per-split LSE plane + combine end-to-end correct:
  final LSE matches exact fp32 logsumexp to ~1e-6.
- out(ns=N) vs out(ns=1): small, **nonzero** → splits are actually active (per-split
  row_max moves the fp8-P quantization grid; same effect A1b observed, ~3.8e-3).
- Empty-split paths exercised (L=256, ns=4: two of four splits own zero KV blocks).

## 4. Performance

Two rounds, both gated on 3 consecutive (0,0) idle samples, all windows 100% clean,
all legs verified 1CTA by ctor hook. L∈{2048,4096,8192}. 20 warmup + 300 pooled
timed iters (3 round-robin rounds × 100), CUDA-event timed.

Legs: **M1b1** = batch=1,q=1 (b1 anchor, no SplitKV by design) · **M1s** = batch=2,q=1
(SplitKV-capable shape) · **FI** = FlashInfer BatchDecodeWithPagedKVCacheWrapper,
batch=1, M=1, page 16.

### Round 2 — in-session A/B (decision numbers; one process, interleaved per L)

| L | M1b1 ns1 | M1s ns1 (serial) | M1s ns2 | M1s ns4 | M1s ns8 | FI M1 |
|---:|---:|---:|---:|---:|---:|---:|
| 2048 | 53.07 | 86.05 | 102.02 (**+18.6%**) | 137.92 (**+60.3%**) | — | 73.82 |
| 4096 | 88.13 | 138.67 | 147.55 (**+6.4%**) | 185.73 (**+33.9%**) | — | 75.62 |
| 8192 | 165.49 | 265.81 | 260.10 (**−2.1%**) | 288.18 (**+8.4%**) | 352.08 (**+32.5%**) | 92.72 |

medians in μs; % is ns=N vs M1s ns1 (same shape, same process, drift-free).

Per-round stability (L=8192 M1s, ns2 vs serial): r0 259.86/266.30 (−2.4%),
r1 261.62/265.65 (−1.5%), r2 259.66/264.90 (−2.0%).

### Round 1 — per-process (one mode per process, independent windows)

| L | M1s ns1 | M1s ns2 | M1s ns4 | M1s ns8 | FI M1 |
|---:|---:|---:|---:|---:|---:|
| 2048 | 85.60 | 103.60 (+21.0%) | 139.41 (+62.6%) | 215.92 (+152%) | 38.82 |
| 4096 | 141.49 | 151.62 (+7.2%) | 188.67 (+33.4%) | 258.74 (+83%) | 61.39 |
| 8192 | 267.86 | 255.47 (−4.6%) | 286.78 (+7.1%) | 353.01 (+31.8%) | 96.30 |

Directions agree across both rounds. (FI absolute values drift between sessions —
documented FI-specific behavior, e.g. 38.82→73.82 at L=2048 — so FA4-leg ratios, not
FI absolutes, are the signal; the FA4 legs show no such drift between rounds.)

### FlashInfer comparison (same window as round 2)

| L | FI M1 | vs M1s serial | vs M1b1 (b1 anchor) |
|---:|---:|---:|---:|
| 2048 | 73.82 | −14.2% | +39.1% |
| 4096 | 75.62 | −45.5% | −14.5% |
| 8192 | 92.72 | **−65.1% (2.87×)** | **−44.1% (1.78×)** |

FI is batch=1/M=1, M1s is batch=2/M=2 — not the same shape, but both are the
decode-relevant small-M regime. At L=8192 the gap vs the 1CTA serial path is ~2.8×
on M1s and ~1.8× even vs the b1 anchor.

## 5. Verdict

**In-tree 1CTA SplitKV is NOT a performance win for hd256.**

- **L=2048 / L=4096: clear loss.** ns=2 is +18.6% / +6.4% slower than 1CTA serial;
  ns≥4 up to +60% slower. At short L the per-split fixed cost (extra CTA launches,
  fp32 partial LSE write, combine pass) exceeds the KV-scan reduction.
- **L=8192: marginal, noise-band "win" at ns=2 only.** ns=2 was faster in **all four**
  measurements (−1.5%/−2.0%/−2.4% per-round, −4.6% round 1) but the magnitude is inside
  the inter-run variance band, and ns=4/ns=8 are clear losses (+8.4%/+32.5%).
- **Cost of the capability:** a per-`num_splits` compile key (separate binaries per ns),
  a hard b1 exclusion (the common single-request decode path cannot split), and the
  extra partial-LSE/combine machinery — for at best ~2% at the single longest L tested.
- **The real long-context SplitKV win is the FlashInfer path:** 1.78–2.8× faster than
  1CTA FA4 serial at L=8192, and it is the existing default for batch=1 decode.

Recommendation: keep the enablement code (correctness-verified, useful as a test
harness and for future kernel work where per-CTA cost drops further), but do **not**
route production traffic to in-tree hd256 SplitKV. For long-context single-request
decode, FlashInfer remains the win; in-tree FA4-1CTA serial remains the best FA4
option.

## 6. Artifacts

- `real-splitkv-bench.json` — merged raw data (round-1 modes, round-2 A/B, probe).
- `real-splitkv-probe.json` — correctness probe.
- `probe_real_splitkv.py`, `real-splitkv-bench.py`, `real-splitkv-ab.py`,
  `run-real-splitkv-bench.sh` — reproducers (v11 image + live-tree mount).
- `raw/real-splitkv-bench-{serial,ns2,ns4,ns8,flashinfer}.json`, `raw/real-splitkv-ab.json`.
