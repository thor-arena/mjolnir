# Spec-Decode End-to-End Forensics — v10 (FA4 hd256 FP8) vs v9 (FlashInfer)

> **Status:** Complete — read-only log forensics · **Date:** 2026-09-22 · **Scope:** end-to-end comparison of the v10 serving image (FA4 hd256 FP8, config `NVFP4_FA4hd256`, Qwen3.8-27B NVFP4, MTP spec decode, `num_speculative_tokens=3`) against the v9 reference image (FlashInfer backend), extracted from the v10 serving log

Read-only forensics task on the v10 serving log (no files modified; log read-only). Raw benchmark data lives under `benchmarks/`.

---

## TL;DR

- v10 actually ran the FA4 hd256 kernel end-to-end: FA4 selected with no fallback, and the `BlackwellFusedMultiHeadAttentionForward` CuTe-DSL JIT fired at first inference.
- Mean acceptance length is **comparable-to-slightly-higher** than v9 (2.97 vs 2.92; weighted 2.95 vs 2.92) — not lower, despite the attention-backend change.
- All three per-position acceptance rates are at or above v9; the largest delta is **pos1** (+0.025 absolute, ~3% relative). pos2/pos3 deltas are within plausible run-to-run variance.
- No decode-path anomalies in the log: no fallback, no "not supported", no recompute, no cudagraph demotion. Only startup/one-time warnings.
- Steady 4-request decode clusters ran ~48–68 gen t/s; 1–2-request clusters ~12–30 t/s.

---

## 1. Source Verified

- Log header line 9: `MODEL_QUANT: NVFP4_FA4hd256`
- Line 94 (non-default args): `'attention_config': AttentionConfig(backend=<AttentionBackendEnum.FLASH_ATTN...>, flash_attn_version=4, ...)` — FA4 selected, no fallback line anywhere.
- Line 225: `CuTeDSL JIT compilation during inference: BlackwellFusedMultiHeadAttentionForward` — the FA4 hd256 cute-DSL kernel actually ran (JIT spike at 16:19:21, first inference window).

## 2. Window Counts

- `SpecDecoding metrics` (metrics.py:120): **55 windows**, 16:19:33 → 16:28:53, one per 10 s.
- `Engine 000:` (loggers.py:320): **58 windows**, 16:19:33 → 16:29:03.

## 3. Extracted Statistics (v10)

Unweighted mean over the 55 windows (min / mean / max):

| metric | min | mean | max |
|---|---|---|---|
| Mean acceptance length | 2.49 | **2.97** | 3.35 |
| Per-position pos1 | 0.714 | **0.832** | 1.000 |
| Per-position pos2 | 0.400 | **0.633** | 0.775 |
| Per-position pos3 | 0.319 | **0.506** | 0.675 |
| Avg draft acceptance rate (%) | 49.6 | **65.7** | 78.3 |

Steps-weighted (weights = drafted_tokens/3, i.e. spec-decode steps; totals ≈ 5481 steps, 10696 accepted / 16443 drafted tokens):

| metric | weighted |
|---|---|
| Mean acceptance length | **2.95** |
| pos1 | 0.822 |
| pos2 | 0.632 |
| pos3 | 0.498 |
| overall draft acceptance | 65.0% |

Internal consistency check: per-window `mean acceptance length == 1 + Accepted/(Drafted/3)` (e.g. line 494: 1 + 70/47 = 2.49) and `draft acceptance rate == Accepted/Drafted` — holds for all 55 windows.

## 4. Comparison with v9

| metric | v10 min | v10 mean | v10 max | v9 reference | v10 − v9 (mean) |
|---|---|---|---|---|---|
| Mean acceptance length | 2.49 | 2.97 (weighted 2.95) | 3.35 | 2.92 (range 2.43–3.30) | **+0.05** (weighted +0.03) |
| pos1 | 0.714 | 0.832 | 1.000 | 0.807 | **+0.025** |
| pos2 | 0.400 | 0.633 | 0.775 | 0.625 | **+0.008** |
| pos3 | 0.319 | 0.506 | 0.675 | 0.494 | **+0.012** |
| Avg draft acceptance | 49.6% | 65.7% | 78.3% | 64.2% (47.8–76.6) | **+1.5 pp** |

Answer to the key question: v10's mean acceptance length is **comparable-to-slightly-higher**, not lower (2.97 vs 2.92; weighted 2.95 vs 2.92). All three per-position rates are at or above v9; the largest delta is at **pos1** (+0.025 absolute, ~3% relative). pos2 (+0.008) and pos3 (+0.012) are within a few percent of v9; v9's per-position values were single means (no ranges provided), so these deltas are within plausible run-to-run variance.

## 5. Throughput Windows (loggers.py:320, 10 s cadence)

Decode-heavy (prompt = 0.0) generation t/s, with Running/WAITING:

| time | gen t/s | Running |
|---|---|---|
| 16:19:33 | 13.4 | 0 (startup) |
| 16:19:53 | 21.1 | 2 |
| 16:20:23 | 69.9 | 4 |
| 16:20:33 | 62.6 | 4 |
| 16:20:43 | 24.4 | 1 |
| 16:20:53 | 20.0 | 1 |
| 16:21:03 | 25.5 | 2 |
| 16:21:13 | 30.4 | 1 |
| 16:21:23 | 26.1 | 0 (draining) |
| 16:21:53 | 47.7 | 4 |
| 16:22:03 | 51.2 | 0 (draining) |
| 16:22:13 | 12.8 | 0 |
| 16:22:23 | 13.1 | 1 |
| 16:22:33 | 12.5 | 2 |
| 16:22:43 | 25.6 | 0 |
| 16:22:53 | 25.6 | 0 |
| 16:23:03 | 24.6 | 1 |
| 16:23:13 | 1.0 | 3 (1 waiting) |
| 16:23:23 | 50.4 | 1 |
| 16:23:33 | 0.8 | 3 (1 waiting) |
| 16:23:43 | 51.2 | 0 (draining) |
| 16:23:53 | 0.1 | 4 |
| 16:24:03 | 51.1 | 0 (draining) |
| 16:24:13 | 0.0 | 0 (idle) |
| 16:24:23 | 18.0 | 1 |
| 16:24:33 | 21.5 | 2 |
| 16:24:43 | 43.2 | 2 |
| 16:24:53 | 37.1 | 4 |
| 16:25:03 | 61.1 | 4 |
| 16:25:13 | 67.5 | 4 |
| 16:25:23 | 33.2 | 0 |
| 16:25:33 | 17.0 | 1 |
| 16:25:43 | 21.4 | 2 |
| 16:25:53 | 34.2 | 2 |
| 16:26:03 | 29.8 | 0 |
| 16:26:13 | 25.1 | 4 |
| 16:26:23 | 26.1 | 4 |
| 16:26:33 | 51.2 | 2 |
| 16:26:43 | 50.7 | 1 |
| 16:26:53 | 13.3 | 0 |
| 16:27:03 | 12.8 | 0 |
| 16:27:13 | 12.8 | 2 |
| 16:27:23 | 25.6 | 0 |
| 16:27:33 | 11.9 | 2 |
| 16:27:43 | 18.7 | 2 |
| 16:27:53 | 20.6 | 3 (1 waiting) |
| 16:28:13 | 24.4 | 3 (1 waiting) |
| 16:28:33 | 15.4 | 3 (1 waiting) |
| 16:28:53 | 5.9 | 0 (draining) |
| 16:29:03 | 0.0 | 0 (idle) |

(Prefill-mixed windows show higher gen t/s because 10-s averages include prefill output; the steady 4-req decode clusters are ~48–68 t/s, the 1–2-req decode clusters ~12–30 t/s.)

## 6. Representative Raw SpecDecoding Lines (verbatim)

Low (line 494, 16:27:33; 70/141 tok — healthy sample):

```
(APIServer pid=118) INFO 09-22 16:27:33 [metrics.py:120] SpecDecoding metrics: Mean acceptance length: 2.49, Accepted throughput: 7.00 tokens/s, Drafted throughput: 14.10 tokens/s, Accepted: 70 tokens, Drafted: 141 tokens, Per-position acceptance rate: 0.723, 0.447, 0.319, Avg Draft acceptance rate: 49.6%
```

Mid (line 272, 16:20:23; 463/705 tok — largest sample in log):

```
(APIServer pid=118) INFO 09-22 16:20:23 [metrics.py:120] SpecDecoding metrics: Mean acceptance length: 2.97, Accepted throughput: 46.30 tokens/s, Drafted throughput: 70.50 tokens/s, Accepted: 463 tokens, Drafted: 705 tokens, Per-position acceptance rate: 0.813, 0.664, 0.494, Avg Draft acceptance rate: 65.7%
```

High (line 238, 16:19:33; 94/120 tok — first window):

```
(APIServer pid=118) INFO 09-22 16:19:33 [metrics.py:120] SpecDecoding metrics: Mean acceptance length: 3.35, Accepted throughput: 4.48 tokens/s, Drafted throughput: 5.72 tokens/s, Accepted: 94 tokens, Drafted: 120 tokens, Per-position acceptance rate: 0.925, 0.775, 0.650, Avg Draft acceptance rate: 78.3%
```

Small-sample outliers to note (tiny windows, low denominator):

```
(APIServer pid=118) INFO 09-22 16:23:33 [metrics.py:120] SpecDecoding metrics: Mean acceptance length: 3.33, ... Accepted: 7 tokens, Drafted: 9 tokens, Per-position acceptance rate: 1.000, 0.667, 0.667, Avg Draft acceptance rate: 77.8%
(APIServer pid=118) INFO 09-22 16:23:13 [metrics.py:120] SpecDecoding metrics: Mean acceptance length: 2.60, ... Accepted: 8 tokens, Drafted: 15 tokens, Per-position acceptance rate: 0.800, 0.400, 0.400, Avg Draft acceptance rate: 53.3%
```

## 7. Decode-Path Anomaly Scan

Patterns searched (case-insensitive): `fallback`, `defaulting`, `not supported`, `recompute`, `demot*`, `cudagraph`, `WARNING`, `ERROR`, `Failed`.

**No decode-path anomaly lines**: no fallback, no "not supported", no recompute, no cudagraph demotion. Only startup/one-time items:

- L103: `WARNING [speculative.py:1360] Enabling num_speculative_tokens > 1 will run multiple times of forward on same MTP layer, which may result in lower acceptance rate` — generic config warning, expected for MTP×3.
- L107: `WARNING [config.py:819] Qwen3.5 model specifies mamba_ssm_dtype='float32' ... but --mamba-ssm-cache-dtype='bfloat16' was passed. Using the user-specified value.`
- L122, L125, L226: cutlass `UserWarning: Argument aux_data ... not supported` for `...FlashAttentionForwardSm100...` / `...sm100_hd256_2cta_fmha_forward...` — JIT-arg typing warnings, one recurs at 16:19:21 during first inference; not a functional fallback.
- L225: `WARNING [jit_monitor.py:140] CuTeDSL JIT compilation during inference: BlackwellFusedMultiHeadAttentionForward` (16:19:21).
- L229–L231 (16:19:24): Triton JIT compilation during inference: `_compute_local_logits_stats_kernel`, `_rejection_kernel`, `_resample_kernel` — the spec-decode sampling kernels JIT-compiled inside the first metrics window (16:19:33).
- L246–L248 (16:19:50), L263 (16:20:11): further one-time Triton JIT (top-p/top-k sampling kernels).
- L99: `WARNING Unknown vLLM environment variable detected: VLLM_MAIN_SERVER_PORT`.
- L151/L161: `Add 3 padding layers, may waste at most 6.25% KV cache memory`.
- L196: sampling params overridden by `generation_config.json` (intended by the server config).

All post-16:20:11 (i.e. after first ~5 windows) the log is clean of warnings/errors.
