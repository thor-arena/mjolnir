# Docs

Research, methodology, and workstream reports for the mjolnir vLLM Thor stack.
Every claim in this tree is traceable to raw artifacts: raw bench JSON in
[`benchmarks/raw/`](../benchmarks/raw/), raw kernel microbench JSON, NCU tables,
or upstream PR/issue anchors cited inline.

## How to read this tree

| Path | What it is |
|---|---|
| [`methodology/`](methodology/) | **The measurement discipline.** How every number in this repo was (and must be) produced: the clean-window gate, wall-clock vs NCU achieved-BW rules, bench protocols. Read this first. |
| [`fa4-hd256-fp8/`](fa4-hd256-fp8/) | **The kernel workstream** — FA4 head_dim=256 decode on Thor: feasibility, 1CTA patch, real SplitKV, the GEMV decode kernel (design, analysis, ring fix, benches). The reports behind `docker/vllm-thor/fa4-gemv-kernel/`. |
| [`thor-stack/`](thor-stack/) | **The serving stack** — why the 13 patches exist: the FA4 hd256 downgrade route trace, the FP8-on-sm_110 kernel proof, GDN prefill enablement, the nightly-bump process, end-to-end forensics. |
| [`draft-cudagraph/`](draft-cudagraph/) | **The draft-cudagraph c1 regression** — study + resulting gate patch design. |
| [`research/`](research/) | **Formal research notes** — upstream status, prior-art surveys, methodology literature review, field notes. |

## Suggested reading order

1. [`methodology/benchmarking.md`](methodology/benchmarking.md) — the rules every
   number in this repo follows.
2. [`thor-stack/fa4-hd256-route-trace.md`](thor-stack/fa4-hd256-route-trace.md) —
   why FA4 + FP8-KV + hd256 is gated on sm_110 (the reason the patch stack exists).
3. [`fa4-hd256-fp8/gemv-decode-design.md`](fa4-hd256-fp8/gemv-decode-design.md) —
   the GEMV decode kernel design (the headline result).
4. [`fa4-hd256-fp8/gemv-ring-fix-bench.md`](fa4-hd256-fp8/gemv-ring-fix-bench.md) —
   the final numbers and the ring-queue fix.
5. [`research/fa4-hd256-upstream-status.md`](research/fa4-hd256-upstream-status.md) —
   where upstream stands (what we can wait for vs. what is ours).

## Index

### methodology/
| File | TL;DR |
|---|---|
| [benchmarking.md](methodology/benchmarking.md) | Clean-window gate, NCU achieved-BW vs wall-clock, bench protocols, provenance rules. |

### fa4-hd256-fp8/ (kernel workstream)
| File | TL;DR |
|---|---|
| [decode-kernel-feasibility.md](fa4-hd256-fp8/decode-kernel-feasibility.md) | Go/no-go on a custom hd256 decode kernel: Thor is bandwidth-bound at M=1; a GEMV path is viable. |
| [dk1a-1cta-paged.md](fa4-hd256-fp8/dk1a-1cta-paged.md) | DK-1a: 1CTA + paged-128 TMA trigger and wiring — GO, 1.60–1.62× vs 2CTA, no kernel changes needed. |
| [decode-microbench.md](fa4-hd256-fp8/decode-microbench.md) | First clean microbench: 2CTA vs 1CTA, FI baseline, achieved-BW vs roofline. |
| [v11-1cta-patch.md](fa4-hd256-fp8/v11-1cta-patch.md) | The 1CTA patch: design, live-tree inventory, ×1.56 result. |
| [splitkv-sweep.md](fa4-hd256-fp8/splitkv-sweep.md) | SplitKV ns sweep on the 2CTA kernel; where the ns knee sits. |
| [real-splitkv-bench.md](fa4-hd256-fp8/real-splitkv-bench.md) | Real (paged) SplitKV on the 1CTA kernel; the stages regression. |
| [gemv-decode-design.md](fa4-hd256-fp8/gemv-decode-design.md) | The GEMV decode kernel: pure-FMA SplitKV design, correctness phases A/B1. |
| [fi-decode-gemv-analysis.md](fa4-hd256-fp8/fi-decode-gemv-analysis.md) | Reverse-engineering FlashInfer's GEMV decode; the gap analysis that set the target. |
| [decode-tiling-design.md](fa4-hd256-fp8/decode-tiling-design.md) | Tiling design notes for the decode kernels. |
| [decode-1cta-clean.md](fa4-hd256-fp8/decode-1cta-clean.md) | Clean-window 1CTA bench: the ×1.56 reference numbers. |
| [gemv-fi-gap-final.md](fa4-hd256-fp8/gemv-fi-gap-final.md) | Final GEMV-vs-FI gap analysis and the remaining ~10%. |
| [gemv-ring-fix-bench.md](fa4-hd256-fp8/gemv-ring-fix-bench.md) | The ring-queue fix (stages 2→16, ×2) + the final GEMV bench + NCU tables. |

### thor-stack/ (serving stack)
| File | TL;DR |
|---|---|
| [base-bump-2026-09-26.md](thor-stack/base-bump-2026-09-26.md) | Base bump to the vLLM 0.30.0 release image: per-patch outcomes, the two docstring-context re-adaptations, canary results. |
| [fa4-hd256-route-trace.md](thor-stack/fa4-hd256-route-trace.md) | Exact downgrade path for FA4+FP8-KV+hd256 on sm_110; what enabling takes (weeks, not a new tile). |
| [fa4-fp8kv-sm110.md](thor-stack/fa4-fp8kv-sm110.md) | Empirical proof FA4 hd128 FP8 2CTA runs on sm_110a (probes, PTX evidence). |
| [gdn-prefill-fi-enable.md](thor-stack/gdn-prefill-fi-enable.md) | GDN prefill enablement: the 1-line FlashInfer hunk, gates, 3.9–5.2× live canary. |
| [nightly-bump-process.md](thor-stack/nightly-bump-process.md) | Process doc: bumping the vLLM nightly base and re-verifying the patch stack. |
| [spec-decode-e2e-forensics.md](thor-stack/spec-decode-e2e-forensics.md) | End-to-end forensics: v10 (FA4 hd256) vs v9 (FlashInfer) serving logs. |

### draft-cudagraph/
| File | TL;DR |
|---|---|
| [c1-regression-study.md](draft-cudagraph/c1-regression-study.md) | c1 regression: the forensics mechanism doesn't hold (FA2 `batch_prefill` is replay-safe by construction); the resulting gate patch. |

### research/ (formal research notes)
| File | TL;DR |
|---|---|
| [fa4-hd256-upstream-status.md](research/fa4-hd256-upstream-status.md) | Upstream status (2026-09-21): RFC #2456 table, the four policy gates, GO verdict. |
| [gemv-decode-methodology.md](research/gemv-decode-methodology.md) | Literature review: FlashAttention online softmax, FlashDecoding/SplitKV, FlashInfer decode internals, Triton backends. |
| [draft-decode-metadata-prior-art.md](research/draft-decode-metadata-prior-art.md) | Prior-art survey: `update_draft_decode_metadata` (PR #46849), what is upstream vs. net-new. |
| [vllm-0.29-concurrent-stall.md](research/vllm-0.29-concurrent-stall.md) | vLLM 0.29 release-delta notes for the concurrent-stall investigation. |
| [thor-build-env-vars.md](research/thor-build-env-vars.md) | Build/runtime env-var audit for vLLM on Thor aarch64. |
| [nvfp4-sm110-field-notes.md](research/nvfp4-sm110-field-notes.md) | Field notes: NVFP4/FP8 on sm_110/sm_12x (garbled FP8, CUTLASS, Marlin, MoE, Qwen3-VL). |
| [tool-eval-bench-perf-runs.md](research/tool-eval-bench-perf-runs.md) | What tool-eval-bench's perf mode does (and why mjolnir wraps llama-benchy directly). |
