# Decode micro-bench — FA4 hd256 vs FlashInfer (per-step attention latency)

## Env (from results.json)
- NVIDIA Thor, capability **(11,0)**, 20 SMs, torch 2.13.0+cu130, flashinfer 0.6.18.
- **Clean run**: `num_requests_running` = 0 at start AND end (no GPU contention).
- batch=1, q_len=M (MTP verify batch), KV=L, causal, hd=256, GQA 24:4, fp8 KV.
- 20 warmup + 100 timed iters, CUDA events, median reported.
- B1 = FA4 hd256 FP8 (e4m3 Q/K/V, block-128) — v10's path. B2 = FA4 hd256 BF16 (same kernel, isolates FP8).
  B3 = FlashInfer BatchDecodeWithPagedKVCacheWrapper (bf16 Q + fp8 KV, block-16) — v9's path.

## Core table — median μs per decode step
| L | B1 FA4-fp8 (M1/M2/M4) | B2 FA4-bf16 (M1/M4) | B3 FlashInfer (M1/M2/M4) | B1/B3 (M1) |
|---|---|---|---|---:|
| 256 | 83.8 / 85.2 / 82.2 | 76.1 / 73.6 | 30.8 / 31.0 / 32.6 | 2.72× |
| 2048 | 136.6 / 137.3 / 134.9 | 130.7 / 130.4 | 43.0 / 44.1 / 49.8 | 3.17× |
| 4096 | 197.3 / 195.9 / 196.8 | 194.5 / 195.6 | 55.7 / 55.7 / 67.4 | 3.54× |
| 8192 | **324.1 / 319.4 / 318.5** | 323.6 / 324.7 | **72.5 / 73.3 / 102.1** | **4.47×** (M1) / 3.12× (M4) |

## Findings
1. **FA4 hd256 decode is 2.5–4.5× slower than FlashInfer per step, and the ratio grows with L.**
   Per-KV-token attention slope: FA4 ≈ **30 μs/1024 tok** vs FlashInfer ≈ **5.3 μs/1024 tok** (~5.7×).
2. **FP8 is free**: B1 ≈ B2 at every point (Δ ≤ 2 μs). e4m3 Q/K/V costs nothing vs bf16 in this kernel.
3. **Descale is free**: Δ_descale = **−0.34 μs** (unity vs non-unity descales). Ruled out as a cause.
4. **cudagraph helps but doesn't close the gap**: eager 318.4 μs → captured 275.3 μs (Δ = **43 μs**, ~13%) at L=8192/M=1.

## Attribution
The entire decode gap is the **raw 1CTA decode-kernel efficiency** of the FA4 hd256 kernel vs FlashInfer's
decode-tuned path. FA4 hd256 is prefill-tuned (2CTA, large-M tiles); its 1CTA small-M decode path (M≤64)
scans the long KV ~5.7× less efficiently. Not FP8, not descale, not cudagraph.

## Reconciliation with the A/B bench (3–4.5× kernel gap vs only 10–18% E2E)
The A/B bench showed v10 decode regressing only ~10–18% at d8192 (vs the 3–4.5× raw kernel gap). The
attenuation: full-attn layers are 16 of 64 (25%) of Qwen3.8; a 27B per-token decode is
GEMM/weight-loading-dominated, so the KV-attention scan (where FA4 hd256 runs) is a small fraction of
per-token cost. A 3–4.5× slowdown on that small fraction → ~10–18% E2E. Consistent.

## Recommendation
- **Root cause = FA4 hd256 1CTA decode-kernel efficiency.** v10 is still a net E2E win (prefill
  dominates: +17–53%), so it is shippable as-is.
- To close the decode gap, two paths:
  - **(a) In-kernel**: optimize the FA4 hd256 1CTA small-M decode path (wide-head, long-KV, M≤64 tiling).
    Keeps FA4 self-contained; harder, payoff uncertain.
  - **(b) Backend split**: route the hd256 full-attn layers prefill→FA4 / decode→FlashInfer. Recovers the
    full decode win while keeping the prefill win; needs vLLM surgery (a layer's forward would branch on
    is_prefill) + feasibility check (cudagraph capture across two kernels).
- Suggested next: a focused feasibility probe of (b) before committing — it is the higher-value fix if vLLM
  can split the phase per backend.
