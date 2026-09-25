# DK-1/DK-2 — CLEAN validation: 1CTA decode carve-out (FA4 hd256)

**Verdict: GO.** The 1CTA decode carve-out is applied to the live tree, the production
decode path now dispatches the 1CTA binary, the clean-window micro-bench measures
**1.55–1.56× (1CTA/2CTA) at L=8192** (exit criterion ≥1.3×, target ~1.6× — met), and
correctness is confirmed (probe2a-fp8 12/12; 1CTA==2CTA==fp32-ref within fp8 tolerance).

## 1. The applied edit (DK-1)

`vfa/cute/interface.py:1037` — decode-1CTA condition OR-ed into the hd256 2CTA carve-out
(exact DK-1a §1.3 edit; diff: `decode-1cta-interface.diff`, part of the DK-3 patch):

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

**Dispatch confirmation** (`probe_dk1_dispatch.py`, `dk1-dispatch-run.log`):
production decode (paged-128, varlen b1, M=1, e4m3, GQA 24/4, causal, L=2048) → exactly
one compile-cache entry, key[-4:] = `(varlen_b1, l2_swizzle, mask_residual, use_2cta) =
(True, False, True, False)` → **the 1CTA/cluster-(1,1) binary is what production caches and
executes**; output within fp32-ref tolerance (max err 3.42e-3 < tol 1.30e-2). The prefill
call (dense q=256 > 8, kv=8192) still selects **2CTA** (key[-4:] `(False, False, True, True)`)
— the general 2CTA path is untouched.

## 2. CLEAN micro-bench (DK-2) — server 0/0 gated

Env: NVIDIA Thor (sm_110, 20 SMs), vllm container co-located with the live v10 server
(unified memory). **Gating**: a 2 s monitor sampled `vllm:num_requests_running/waiting`
throughout; each mode's measurement window was opened only after 3 consecutive (0,0)
samples, and every sample inside the window was (0,0) (verified in the raw JSONs).
20 warmup + 100 CUDA-event timed iters per (M, L) (same protocol as `decode-microbench.py`).

Legs (one process each, binary verified via ctor-call log — exactly one kernel compile per
process, all decode shapes sharing the compile key by design, interface.py:1308):
- **B1-1CTA** = FA4 hd256 e4m3 paged-128, **natural** interface selection (the new
  carve-out; ctor log: 1 compile, `use_2cta=False`).
- **B1-2CTA** = same shapes, ctor-forced `use_2cta=True` (the exact flag the carve-out
  drives at interface.py:1610 — i.e. the carve-out disabled for this leg; ctor log:
  1 compile, `use_2cta=True`).
- **B3** = FlashInfer `BatchDecodeWithPagedKVCacheWrapper` (fa2 tensor-core, block-16,
  bf16 Q + fp8 KV) — the v9 reference path.

Geometry: hd=256, GQA 24q/4kv, paged-128, causal, non-unity descales [1.06, 1.125, 1.25];
M ∈ {1, 4}, L ∈ {256, 1024, 2048, 4096, 8192}; batch=1.

### Main table — medians (µs)

| L | M | FA4 1CTA med | FA4 2CTA med | FlashInfer med | 1CTA/2CTA | 1CTA/FlashInfer |
|---|---|---:|---:|---:|---:|---:|
| 256 | 1 | 70.2 | 84.1 | 28.0 | 1.20x | 2.50x |
| 256 | 4 | 70.1 | 82.7 | 21.6 | 1.18x | 3.24x |
| 1024 | 1 | 83.8 | 108.3 | 36.8 | 1.29x | 2.28x |
| 1024 | 4 | 83.5 | 106.2 | 28.6 | 1.27x | 2.92x |
| 2048 | 1 | 101.2 | 140.8 | 43.0 | 1.39x | 2.35x |
| 2048 | 4 | 102.7 | 136.9 | 42.6 | 1.33x | 2.41x |
| 4096 | 1 | 137.4 | 201.8 | 54.4 | 1.47x | 2.52x |
| 4096 | 4 | 137.1 | 197.7 | 60.5 | 1.44x | 2.27x |
| **8192** | **1** | **207.1** | **323.0** | **92.2** | **1.56x** | **2.25x** |
| **8192** | **4** | **207.5** | **321.2** | **96.5** | **1.55x** | **2.15x** |

Min-latency view (min per 100 iters — noisier, single-sample-sensitive; the medians are
the reliable signal):

| L | M | FA4 1CTA min | FA4 2CTA min | FlashInfer min | 1CTA/2CTA (min) |
|---|---|---:|---:|---:|---:|
| 256 | 1 | 19.3 | 81.2 | 16.6 | 4.22x |
| 256 | 4 | 67.4 | 78.9 | 17.7 | 1.17x |
| 1024 | 1 | 56.4 | 58.9 | 22.6 | 1.04x |
| 1024 | 4 | 77.3 | 103.8 | 22.2 | 1.34x |
| 2048 | 1 | 95.5 | 86.6 | 28.1 | 0.91x |
| 2048 | 4 | 50.2 | 134.0 | 34.9 | 2.67x |
| 4096 | 1 | 85.7 | 194.1 | 39.7 | 2.27x |
| 4096 | 4 | 134.5 | 145.6 | 54.5 | 1.08x |
| 8192 | 1 | 203.7 | 268.5 | 58.9 | 1.32x |
| 8192 | 4 | 203.8 | 267.5 | 92.3 | 1.31x |

### Key results

- **1CTA/2CTA = 1.56× (M=1) / 1.55× (M=4) at L=8192** — exit criterion ≥1.3× met; target
  ~1.6× essentially met. The ratio grows with L (1.20×→1.56× for M=1) — the 1CTA form's
  smaller CTA footprint pays off the more KV there is. Matches DK-1a's interleaved pair
  ratios (1.60–1.62×).
- **1CTA absolute at L=8192/M=1: 207.1 µs median (203.7 min)** — within the DK-1b
  prediction of ~190–210 µs clean, and below the prior 2CTA clean number (324.1 µs) by
  the expected ~1.6×. 2CTA clean (323.0 µs) reproduces the prior clean 2CTA number
  (324.1 µs) to 0.3% — the GPU/environment state is consistent.
- **FlashInfer absolute drift (reported, not hidden)**: FI here is 92.2 µs at L=8192/M=1
  (reproduced in two consecutive clean windows: 92.16 / 92.85), ~27% above the prior
  isolated-container clean run (72.5 µs, `decode-microbench.md`). The FA4 legs are
  unaffected (2CTA matches the prior clean run to 0.3%), so the drift is FI-specific —
  consistent with residual co-location effects of the live server process inside the same
  container even at 0/0 requests (unified memory / allocator state). Against the prior
  isolated clean FI reference, **1CTA/FI at L=8192 = 207.1/72.5 = 2.86×** — inside the
  expected ~2.6–2.8× range. In-environment (same window) 1CTA/FI = 2.25× (M=1) / 2.15× (M=4).
- **M=4 ≈ M=1 for both FA4 modes** (~0.4 µs delta) — the M≤8 carve-out holds for the MTP
  verify batch shape.

## 3. Correctness (DK-2)

**probe2a-fp8 (full-FP8 A/B vs fp32 dequant ref): 12/12, GO** (`dk12-probe2a-run.log`).
All 6 prefill shapes (q_len 256/1024/4096 > 8 → 2CTA, unchanged path) pass. All 6 decode
shapes (q_len=1 → **now run the 1CTA binary** via the carve-out; fresh 1CTA compiles logged
in the wall times) pass the fp8 tolerance — the carve-out does not regress the general
2CTA prefill path and 1CTA decode meets the fp8 tolerance.

**1CTA == 2CTA == fp32-ref, production decode geometry** (`probe_dk12_identity.py`,
two-process A/B — see shared-key note below; `raw/dk12-identity-merged.json`):
paged-128, e4m3, GQA 24/4, non-unity per-(batch,kv_head) descales, **shuffled physical
page table**, causal, M=1.

| L | out(1CTA) vs fp32 ref | out(2CTA) vs fp32 ref | out(1CTA) vs out(2CTA) | tol | LSE err 1CTA/2CTA |
|---|---|---|---|---|---|
| 2048 | 5.49e-3 | 5.49e-3 (identical) | **9.77e-4** (fp8-level) | 1.52e-2 | 9.5e-7 / 4.8e-6 |
| 8192 | 3.22e-3 | 3.22e-3 (identical) | **4.88e-4** (fp8-level) | 1.35e-2 | 9.5e-7 / 3.8e-6 |

Both legs are verified **true binaries** (each process: exactly one kernel ctor call,
flag as intended — the natural carve-out path for 1CTA). 1CTA-vs-2CTA outputs agree at the
fp8-rounding level (≈1e-3, far below the 2e-2·ref_max+1e-2 tolerance) and both match the
fp32 reference at the same error (the kernels are equally correct; residual diff is
fp8/softmax numerics, not a cluster-form difference). The L=2048 rows match DK-1a §3.1
exactly (9.77e-4). DK-1a's "byte-identical 0.0" at L=8192 is superseded: that single-process
measurement was contaminated by the shared compile key (below); the clean two-process
measurement shows a small non-zero fp8-level diff at both L.

### Shared-key / recompile-artifact notes (do not confuse with a regression)

1. **The compile key excludes seqlen** (interface.py:1308): all paged-decode shapes share
   ONE cache key per mode, so one compiled binary serves all (M, L) — the bench's one
   compile per mode-process is correct, and it means a single process can hold at most one
   FA4 decode binary; the 1CTA-vs-2CTA A/B therefore required separate processes.
2. **Probe-only 2CTA→1CTA recompile artifact (DK-1a caveat 2)**: if a process compiles a
   2CTA binary under the shared decode key, clears the cache, and recompiles that key with
   the 1CTA decision, the interface can emit two ctor calls and cache the 2CTA binary —
   the "1CTA" leg would silently measure 2CTA (this is what contaminated the first version
   of this task's identity probe and DK-1a's L=8192 line). **It is absent in production**:
   a production process compiles each key exactly once (no clears, no forcing), and our
   1CTA process verifies exactly one ctor call (`use_2cta=False`) for the whole process.

## 4. Provenance

- Live-tree edit: `<vfa-tree>/cute/interface.py` (interface sha256
  `86de92f3…`, kernel `sm100_hd256_2cta_fmha_forward.py` sha256 `d173bca1…` — kernel
  untouched, byte-identical across tree and container).
- Scripts: `probe_dk1_dispatch.py`, `probe_dk12_identity.py`, `probe_dk12_merge.py`,
  `decode-1cta-clean-bench.py`, `merge-dk12-clean.py`, `run-dk12-bench.sh`.
- Raw JSONs (window traces + ctor audits + full summaries): `raw/clean-bench-1cta.json`,
  `raw/clean-bench-2cta.json`, `raw/clean-bench-flashinfer.json` (+`-rerun`),
  `raw/dk12-identity-merged.json`.
- Logs: `dk1-dispatch-run.log`, `dk12-probe2a-run.log`, `dk12-identity-run.log`,
  `/tmp/dk12-bench-driver.log`.
- The container's installed `interface.py` was synced with the live tree (diff = this edit
  only); the live serving process is unaffected (module already resident in its memory).
