# Decode Attention Methodology — FlashDecoding, SplitKV, and GEMV (Literature Review)

> **Status:** Final · **Date:** 2026-09-25

Literature review of the decode-attention state of the art — online softmax,
FlashDecoding/split-KV parallelization, the FlashInfer decode stack, and the
Triton paged-attention kernels used by vLLM — consolidated from primary
sources (papers, vendor blog posts, API documentation) to motivate the
pure-FMA split-KV GEMV decode kernel in [`docker/vllm-thor/fa4-gemv-kernel/`](../../docker/vllm-thor/fa4-gemv-kernel/).

## TL;DR

- Standard attention materializes the `N×N` score matrix; FlashAttention's online softmax
  replaces it with a block-wise rescale-and-accumulate that keeps only per-row statistics
  (max / logsumexp) in registers — the primitive that makes split-KV decode exact.
- FlashDecoding adds one parallelization dimension — the KV sequence length — splitting KV
  into chunks, computing partial attention plus a per-chunk logsumexp in parallel, then
  merging partials with LSE-weighted averaging. It delivers up to 8× end-to-end decode
  speedups on very long sequences (attention kernel up to 50× faster) by saturating all SMs
  at batch size 1.
- FlashInfer's default decode path is a CUDA-core **GEMV** kernel (no tensor cores, no MMA):
  one CTA per (KV chunk, KV head), 128-bit vector FMA dots reduced by warp shuffles,
  register-resident base-2 online softmax, 2-stage `cp.async` double-buffered KV tiles, a
  shared-memory page-offset table for paged KV, and GQA head-group fusion. The split plan is
  occupancy-driven: `max_grid = active_blocks_per_SM × num_SM`, then a binary search over
  minimum pages-per-chunk.
- The vLLM Triton backend independently converged on the same decode design — a "3D"
  parallel-tiled-softmax kernel (split-KV grid + LSE reduction) plus static/persistent
  launch grids for CUDA-graph compatibility — reaching 100.7% of FlashAttention-3 on
  long-decode workloads on H100 from ~800 lines of source.
- For M=1 decode at `head_dim=256` on a 20-SM part, decode is bandwidth-bound by roofline;
  tensor-core machinery is per-KV-token overhead. This review is the design basis for
  mjolnir's FA4-native GEMV kernel — see [`docs/fa4-hd256-fp8/fi-decode-gemv-analysis.md`](../fa4-hd256-fp8/fi-decode-gemv-analysis.md)
  and the kernel README.

## 1. Attention Math and Online Softmax

**Source:** [FlashAttention-2: Faster Attention with Better Parallelism and Work
Partitioning (arXiv:2307.08691)](https://arxiv.org/abs/2307.08691)

Standard attention materializes the score matrix `S = QKᵀ` and the probability matrix
`P = softmax(S)` in HBM, then computes `O = PV`. Since `N ≫ d` is typical
(sequence length `N` on the order of 1k–8k versus head dimension `d` around 64–128),
this requires `O(N²)` memory, is bounded by memory bandwidth for most of its
operations, and must additionally retain `P ∈ ℝ^{N×N}` for the backward pass.
The three standard steps — GEMM for `S`, a HBM round-trip for softmax, GEMM for `O` —
translate directly into slow wall-clock time.

**Online softmax** (from Milakov & Gimelshein) is the enabler for tiling. For a row
block `[S⁽¹⁾ S⁽²⁾]` with `S⁽¹⁾, S⁽²⁾ ∈ ℝ^{Br×Bc}`, and value blocks
`[V⁽¹⁾; V⁽²⁾]` with `V⁽ᵏ⁾ ∈ ℝ^{Bc×d}`, instead of computing the full softmax over the
concatenated row, one computes a *local* (unnormalized) attention per block and keeps
running per-row statistics, rescaling on each new block. With row max `m` and row sum
of exponentials `ℓ`, the update for a new block `j` is:

```
m_new      = max(m_old, m_j)
ℓ_new      = 2^(m_old − m_new) · ℓ_old + 2^(m_j − m_new) · ℓ_j
O_new      = 2^(m_old − m_new) · O_old + 2^(m_j − m_new) · P_j · V_j
```

(the base-2 form with `log2e` folded into the softmax scale is the kernel convention
used by FlashAttention-2, FlashInfer, and mjolnir's GEMV kernel). The per-row attention
output is `o = O_new / ℓ_new`, and the row's logsumexp is `lse = m + log2(ℓ)`. Because
the rescale is exact, the final output is bit-identical to un-tilable attention while
never writing `S` or `P` to HBM.

FlashAttention-2's forward-pass refinement replaces the separate row-max and
row-sum-of-exponentials bookkeeping with the row-wise logsumexp `ℓ` alone, reducing
non-matmul FLOPs. Its parallelization assigns each worker (thread block) a block of
*rows* of the attention matrix in the forward pass (a block of *columns* in the
backward pass), which must keep five matrix multiplies worth of values in SRAM versus
two for the forward pass.

Reported numbers from the paper:

- FlashAttention forward pass reaches 30–50% of theoretical peak FLOPs/s of the device,
  and the backward pass only 25–35% on A100 — versus 80–90% for optimized GEMM — due to
  suboptimal work partitioning between thread blocks and warps.
- FlashAttention-2 is 1.7–3.0× faster than FlashAttention, 1.3–2.5× faster than
  FlashAttention in Triton, and 3–10× faster than a standard implementation, reaching up
  to 230 TFLOPs/s — 73% of the theoretical maximum on A100.
- Training GPT-style models on 8× A100 reaches up to 225 TFLOPs/s (72% model FLOPs
  utilization).

**Why it matters here:** online softmax is not merely a memory optimization — it makes
attention *composable*. Keeping each partial result's `(output, logsumexp)` pair means
partials from disjoint KV ranges can be merged exactly, which is what FlashDecoding's
split-KV reduction (Section 2) and FlashInfer's cross-CTA "attention composition"
(Section 3) both exploit.

## 2. FlashDecoding: Split-KV Parallelization for Decode

**Source:** [CRFM blog: "Flash-Decoding for long-context inference"
(Dao, Haziza, Massa, Sizov)](https://crfm.stanford.edu/2023-10-12/flashdecoding.html)

**Motivation.** LLM decoding is iterative: generating a sentence of `N` tokens requires
`N` forward passes. KV caching makes each generation step independent of context length
*except* for attention, which scales with context length. Context windows grew from ~2k
(GPT-3, 2022) to 32k (Llama-2-32k) and 100k (CodeLlama), and even at moderate contexts
the attention memory traffic scales with the *batch* dimension, making decode a
bottleneck in both regimes.

**The gap FlashDecoding fills.** FlashAttention optimizes the *training* case, where the
bottleneck is the bandwidth of reading/writing `QKᵀ`; it parallelizes over batch size
and query length. During inference the query length is 1, so if the batch size is
smaller than the GPU's SM count (108 on A100), the kernel uses a small fraction of the
device — at batch size 1, less than 1% of the GPU. The alternative, plain
matrix-multiply primitives, occupies the whole GPU but launches many kernels that
write and read intermediate results, which is not optimal either.

**The algorithm.** FlashDecoding is FlashAttention plus a third parallelization
dimension: the keys/values sequence length. It works in three steps:

1. Split the keys/values into smaller chunks (these are *views* of the full KV tensors
   — no GPU operation).
2. Compute the query's attention with each split in parallel using FlashAttention,
   writing one extra scalar per row and per split: the log-sum-exp of the attention
   values.
3. Reduce over all splits to produce the final output, using each split's log-sum-exp
   to scale its contribution.

Steps 2 and 3 are two separate kernels. Step 3 is exactly the online-softmax merge from
Section 1, applied *between* CTAs rather than within one.

**Measured results (from the blog).** Benchmarks decode CodeLlama-34B (Llama-2
architecture, so results generalize) at sequence lengths 512 to 64k, comparing
pure-PyTorch attention, FlashAttention v2 (pre-2.2), FasterTransformer, FlashDecoding,
and an upper bound computed as the time to read the entire model plus KV cache from
memory:

- FlashDecoding unlocks up to **8×** decoding speedup for very large sequences and scales
  much better than the alternatives; at batch size 1, increasing sequence length has
  little impact on generation speed.
- A100 micro-benchmarks of scaled multi-head attention (f16, batch 1, 16 query heads of
  dimension 128 over 2 KV heads — the grouped-query configuration of CodeLlama-34B on
  4 GPUs) show FlashDecoding's attention runtime staying nearly constant as sequence
  length scales to 64k: the attention kernel is up to **50×** faster than
  FlashAttention, and roughly constant up to 32k because the GPU is fully utilized.
  The 8× end-to-end speedup is made possible by this ~50× kernel speedup.

**Availability.** FlashDecoding shipped in the FlashAttention package from version 2.2
(https://github.com/Dao-AILab/flash-attention) and in xFormers from 0.0.22 via
`xformers.ops.memory_efficient_attention` (https://github.com/facebookresearch/xformers),
whose dispatcher selects between FlashDecoding and FlashAttention by problem size and
falls back to a Triton kernel implementing the same algorithm where needed.

## 3. FlashInfer Decode Architecture: Planning, Page Tables, and Paged KV

**Sources:** [FlashInfer: Efficient and Customizable Attention Engine for LLM Inference
Serving (arXiv:2501.01005)](https://arxiv.org/abs/2501.01005);
[FlashInfer introduction blog](https://flashinfer.ai/2024-02-02/introduce-flashinfer.html);
[FlashInfer attention API documentation](https://docs.flashinfer.ai/api/attention.html)

### 3.1 Serving stages and roofline positioning

FlashInfer structures serving around three attention stages — *prefill*, *decode*, and
*append* (decode attention fills one row of the attention map at a time; prefill fills
the whole causal map; append fills the trapezoid region and is the form used by
[speculative decoding](https://arxiv.org/abs/2211.17192)). Its roofline analysis places
each stage: **decode is always under the peak-bandwidth ceiling — IO-bound**;
prefill has high operational intensity and is under the peak-compute ceiling; append is
IO-bound at small query length and compute-bound at large query length. This is the
core design constraint for mjolnir's target shape (M=1 decode).

### 3.2 Paged KV, page tables, and the BSR abstraction

FlashInfer implements single-request and batch kernels for all three stages over
versatile KV-cache formats (ragged tensors, page tables) and was the first library to
ship **prefill/append kernels for paged KV cache**. Page-table management follows
[PagedAttention (vLLM, arXiv:2309.06180)](https://arxiv.org/abs/2309.06180): the API
takes `kv_page_indptr` (CSR offsets, shape `[batch+1]`), `kv_page_indices` (physical
page IDs), and `kv_last_page_len` (entries in each request's final page,
`1 ≤ len ≤ page_size`); the cache is stored as `NHD` (token-major) or `HND` (head-major)
page pools.

The FlashInfer paper shows these structures unify under a **Block Compressed Sparse
Row (BSR)** format: a paged KV cache is a block-sparse matrix in which non-zero blocks
are the KV pages accessed by queries. BSR improves register reuse and compatibility
with hardware matrix units over CSR, and empty blocks can be skipped. Composable
formats extend this: requests sharing a prefix form a dense submatrix stored with a
larger block size, requiring no data movement — only recomputed index arrays — which
is what enables FlashInfer's cascade/shared-prefix kernels (up to **31×** speedup over
the baseline vLLM PageAttention at 32,768-token prompts and batch 256;
[cascade post](https://flashinfer.ai/2024-02-02/cascade-inference)).

A key paged-KV result from the introduction blog: FlashInfer's decode kernels
**prefetch page indices into shared memory**, so kernel performance is unaffected by
page size (ablation over four page sizes on A100 shows near-identical bandwidth
utilization, close to the single-request non-paged curve). This directly targets the
page-size-1 PageAttention variants used by LightLLM and SGLang for complex serving
scenarios.

### 3.3 Planning and load-balanced scheduling (split-KV)

The paper's runtime design separates *planning* from *execution*, following the
Inspector-Executor model: `plan()` runs on the host per generation step (as sequence
lengths change), computes (1) the work queue of each CTA and (2) the index mapping
between partial and final outputs, and asynchronously copies the plan into a fixed
region of a user-provided workspace; `run()` then executes persistent
attention/contraction kernels that read that plan. The split is CUDAGraph-friendly by
construction: the persistent kernels launch with a fixed grid size every step and with
fixed workspace pointers, and the plan can be captured/reused while `plan()` itself
stays outside the graph.

Why the scheduler exists at all: attention kernels do not produce final outputs
directly — long KV sequences are split into chunks and each CTA produces a *partial
output plus scale*; the final output is the contraction of all partials using the
**attention composition operator** (from Block-Parallel Transformer: outputs for the
same query over different KV ranges compose when both the output and its scale are
kept, the scale being the log-sum-exp over the range). FlashInfer's composition
operator handles variable-length aggregation. The plan amortizes over all layers of a
generation step.

The batch-decode API (`BatchDecodeWithPagedKVCacheWrapper`) exposes this split to the
serving stack: `plan()` is called before any `run()` and creates/caches the auxiliary
data structures (page-table metadata, partial-output workspace) reused across all
Transformer layers; auxiliary buffers can be pinned for CUDA-graph capture (in which
case the batch size cannot change over the wrapper's lifetime).

### 3.4 Tensor-core vs CUDA-core (GEMV) decode paths

The introduction blog benchmarks single-request decode on Llama2-7B settings
(`num_kv_heads = num_qo_heads = 32`, `head_dim = 128`, sequence 32–65,536) against
FlashAttention 2.4.2 (which itself includes FlashDecoding) and vLLM 0.2.6
PageAttention, using `nvbench` with "cold" GPU time (L2 flushed before each launch),
built from source with NVCC in CUDA 12.3.1. Headline facts:

- **Decode is bandwidth-bound, and the right split depends on the SM/CUDA-core mix.**
  Split-KV (splitting the KV cache along the sequence dimension, the same trick as
  GEMM split-K reductions and FlashDecoding) does *not* help on RTX Ada 6000 or RTX 4090:
  they have relatively small memory bandwidth and strong CUDA cores, and non-GQA decode
  attention has low operational intensity and runs on CUDA cores. Global memory traffic
  is shared across SMs, so using only 32 of 108 SMs (Llama2-7B's head count) can still
  saturate bandwidth when the operator is not compute-bound. On A100, whose CUDA cores
  are weak (20 TFLOPs/s), 32 of 108 SMs gives only 5.9 TFLOPs/s — far below what the
  `exp`-heavy attention arithmetic needs — so the kernel becomes compute-bound and
  split-KV is required. On A100/H100, FlashInfer's decode reaches close to 100% of GPU
  bandwidth utilization at long sequences.
- **GQA flips the roofline.** Grouped-Query Attention (arXiv:2305.13245) raises
  operational intensity from `O(1)` to `O(Hqo/Hkv)`; with A100/H100's weak non-tensor
  cores, traditional CUDA-core GQA decode becomes compute-bound. FlashInfer's answer is
  a *tensor-core* decode (using prefill-style kernels) for GQA — up to 2–3× faster than
  vLLM at `num_kv_heads=4, num_qo_heads=32` (Llama2-70B, tp=2) — while its CUDA-core GQA
  kernel achieves only 40%+ bandwidth utilization. Its **head-group fusion** (paper,
  Appendix A1) maps different KV heads to different threadblocks and fuses query heads
  into the row dimension, so a single shared-memory load of the KV tile serves the whole
  query-head group.
- **The default path is the GEMV kernel.** With `use_tensor_cores=False` (the default),
  FlashInfer's decode kernel is a CUDA-core vector kernel with no MMA: this is the
  pattern mjolnir replicates (Section 5 and
  [`docs/fa4-hd256-fp8/fi-decode-gemv-analysis.md`](../fa4-hd256-fp8/fi-decode-gemv-analysis.md)).
- **Quantization and fusion.** FP8 KV decode kernels run up to 2× faster than their fp16
  counterparts (Atom adds int4 on top); Fused-RoPE kernels apply RoPE on the fly (needed
  because pruned caches — H2O, Streaming-LLM — make stored post-RoPE keys meaningless);
  append kernels benefit strongly from split-KV once append length (128, 256) pushes
  operational intensity past the ridge point (e.g., RTX 4090's tensor-core fp32-accumulator
  ridge point is 163 = 165 TFLOPs/s ÷ 1008 GB/s).
- **Batch decode** PageAttention with prefetched page indices shows consistent speedups
  over vLLM 0.2.6 across batch sizes 1/16/64 and sequence lengths.

The API surface additionally exposes `fixed_split_size` — a fixed, in-pages split size
for the tensor-core (FA2) split-KV decode that makes the `merge_states` reduction
deterministic and batch-size-invariant (relevant to
[reproducible inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)
and noted as not CUDA-graph-compatible because the CTA count varies with KV length) —
and `q_len_per_req > 1` support for speculative decoding. The backend selector
(`auto`/`fa2`/`fa3`/`trtllm-gen`/`cute-dsl`/`prims-ts`) routes to tensor-core kernels
or task-scheduled persistent kernels per architecture; the `cute-dsl` backend is a
separate SM100+ tensor-core GQA decode kernel, **not** the GEMV pattern.

### 3.5 End-to-end results (FlashInfer paper)

Evaluated on A100/H100 (CUDA 12.4, PyTorch 2.4.0, f16) with SGLang v0.3.4 against the
Triton backend, using ShareGPT and a synthetic variable-length workload at request rates
holding P99 TTFT under 200 ms:

- **29–69%** inter-token-latency reduction on the serving benchmark (the decode win is
  attributed to the load-balanced scheduler plus versatile tile-size selection —
  FlashAttention uses a suboptimal tile size for decode),
- **28–30%** latency reduction for long-context inference (Streaming-LLM: a fused
  RoPE+attention kernel generated with ~20 extra lines of functor code achieves
  1.6–3.7× higher bandwidth utilization than the unfused pair),
- **13–17%** speedup for parallel generation (4 ≤ n ≤ 32; peak at n=4: ITL −13.73%
  (8B) / −17.42% (70B), TTFT −16.41% / −22.86%).

Related-work positioning from the paper: FlashDecoding applies Split-K to decode
kernels; LeanAttention (arXiv:2405.10480) uses StreamK to cut wave quantization at fixed
lengths; FlashInfer generalizes both by decoupling computation from tile scheduling so
the runtime scheduler can adopt FlashDecoding-, StreamK-, or load-balanced policies
over variable-length sequences.

## 4. Triton Attention Internals and vLLM's Triton Decode Backend

**Sources:** [The Anatomy of a Triton Attention Kernel (arXiv:2511.11581)](https://arxiv.org/abs/2511.11581);
[vLLM blog: "vLLM Triton Attention Backend Deep Dive"
(2026-03-04)](https://vllm.ai/blog/2026-03-04-vllm-triton-backend-deep-dive)

### 4.1 Why Triton

The Anatomy paper builds a feature-complete, cross-platform, state-of-the-art paged
attention kernel in OpenAI Triton — a tiling DSL whose JIT compiler automates memory
coalescing, shared-memory allocation, and synchronization via hierarchical tiles, with
kernel configuration parameters (e.g., `BLOCK_SIZE`) controlling work partitioning.
Autotuning benchmarks candidate configurations empirically and explores an order of
magnitude more variants than hand tuning; prior work showed a Triton FlashAttention-2
kernel reaching vendor-library-level performance on both A100 and MI250 from the same
source. The resulting kernels were integrated into vLLM and adopted as its **default
attention backend for AMD GPUs**; the kernel and micro-benchmark suite are open-sourced
(https://ibm.biz/vllm-ibm-triton-lib).

### 4.2 Kernel structure

- **Paged KV baseline.** Following PagedAttention (Kwon et al.), the kernel assumes
  `Q`, `K`, `V` are already computed and stored in the paged KV cache, accessed through
  a block table (analogous to a page table) with `BLOCK_SIZE` the maximum tokens per KV
  block. Per-row numerically-stable softmax (row max `m`) produces the attention
  distribution over keys. The baseline processes one (query token, query head) pair per
  program instance; the launch grid is `num_seqs × tot_query_length` — the sum of prompt
  lengths for prefill, the sequence count for decode (query length 1).
- **Q-block (prefill/GQA) optimization.** `BLOCK_M` combines `BLOCK_Q` successive query
  tokens with `BLOCK_M / BLOCK_Q` query heads, with the ratio set to
  `num_query_heads / num_kv_heads` so each Q block covers exactly the query heads that
  map to one KV head. One KV load then serves all of them — raising arithmetic density
  and cutting memory bandwidth. This is the same idea as FlashInfer's head-group
  fusion (Section 3.4).
- **The decode problem.** For decode, the first grid dimension is the batch's sequence
  count, so small batches launch few program instances and underutilize the GPU
  (prefill is unaffected — prompts contain many tokens). The fix, shared with
  FlashDecoding and FlashInfer, is **parallel tiled softmax**: a three-dimensional launch
  grid whose third dimension splits each (Q block, KV head) combination into segments;
  each segment processes a tile subset of the KV sequence, writes its intermediate
  (partial output, logsumexp) results to memory, and a follow-up **reduction kernel**
  combines segment-level partials into the final output. It is selected heuristically,
  only for decode attention on small batches with long sequences.
- **Adjustable tile sizes.** Decoupling the softmax tile size from the KV-cache
  `BLOCK_SIZE` allows independent prefill/decode tuning and supports hybrid
  Transformer/SSM (e.g., Mamba) models whose page alignment needs large non-power-of-two
  block sizes.

### 4.3 Autotuning and vLLM integration

Autotuning Triton kernels is slow (tuning FlashAttention-2 took nearly 24 hours per GPU
type) and its result cache only helps for exactly repeated scenarios (tuned at 32
tokens, a 33-token request retriggers tuning). The authors' two-step approach:
micro-benchmarking *outside* the serving runtime (same kernel code, simulated request
patterns and architectures — Llama3-8B shape, 20 warmup + 100 measurement iterations),
then folding the results into heuristics. vLLM integration required new attention
metadata: a per-sequence count of decodes (to pick the parallel-tiled kernel) and an
accumulated Q-block tensor that each program instance binary-searches to map its Q-block
index back to a sequence.

### 4.4 CUDA graphs and static launch grids

CUDA/HIP graphs record a fixed execution graph; attention kernels with variable launch
grids (grid scales with batch size and sequence length) interact poorly with them.
Recording graphs at all power-of-two batch sizes up to 128 with dummy requests at
maximum model length freezes the grid to the worst case, so excess kernel instances
early-exit — correct, but the extra scheduled "waves" cost more than the saved launch
overhead in nearly all cases. The fix: **static launch grids** (close to but smaller
than the core count) with work derived from GPU-resident metadata; in the current vLLM
backend this evolved into **persistent kernels** — a fixed number of instances equal
to available compute resources, each reading metadata from GPU memory to determine its
work, keeping the launch grid constant so graphs replay efficiently.

### 4.5 Measured results

Micro-benchmarks (H100 and MI300, Llama3-8B shape: head size 128, 32 query heads, 8 KV
heads, variable-length batches):

- The naive Triton paged kernel is nearly an order of magnitude slower than
  FlashAttention-3 across all sequence lengths.
- The Q-block (GQA) optimization helps at small sequence length and batch size —
  sometimes beating FlashAttention — but not beyond ~500–1000 tokens.
- When results are regrouped by *share of decode requests in the batch* rather than
  sequence length, the picture inverts: the Q-block kernel is strongest on
  prefill-heavy (compute-bound) batches, while the **parallel-tiled-softmax kernel
  nearly matches FlashAttention on decode-heavy batches** and beats the Q-block kernel
  on very long decodes — the memory-bound decode phase is exactly where split-KV
  parallelism pays.
- Flexible (decoupled) tile sizes beat their fixed-tile predecessors in all
  configurations.

The paper reports the investigated optimization steps (Q-block, parallel tiled softmax,
flexible tiles, static launch grid) yield a total speedup of up to **589%** over the
naive baseline, with micro-benchmarks warmed up 20 iterations and averaged over 100, and
end-to-end vLLM benchmark runs (prefix caching disabled, default 10 warmup / 30
measurement iterations) on H100 and MI300.

vLLM blog end-to-end results: on H100 the Triton attention backend achieves
**100.7%** of FlashAttention-3 performance for long decode requests; on MI300 an
~**5.8×** speedup over earlier implementations — from the same ~800-line kernel source
(versus ~70,000 lines for FlashAttention-3). A Helion (tiled-PyTorch DSL) paged
attention prototype shows promising early results (PyTorch blog; draft vLLM PR
vllm-project/vllm#27293). The vLLM ecosystem context also includes decode context
parallelism (KV cache sharded across GPUs by sequence dimension, 3× throughput on
long-context agentic workloads vs tensor parallelism).

### 4.6 Implementation lessons

Two findings are directly transferable to hand-written CUDA:

- **"Triton kernels need to be specific."** Fusing prefill and decode into one branching
  kernel lost at least **2×** performance — far outweighing the ~150 µs of saved launch
  overhead — because software pipelining did not produce useful pipelines in the fused
  form. Kernels should be written around one specific problem with strong internal data
  dependency, and multiple launches "paid for" when the alternative degrades code
  quality.
- Graph capture pins exact binary kernel variants (Triton specializes on constants and
  on access patterns such as stride divisibility), so in scenarios where a single
  kernel run time is comparable to the average Triton launch overhead (~200 µs),
  HIP/CUDA graphs do not reduce overall latency unless the kernel is designed for it
  (static grids, in-kernel decision trees, NOP-masked loops — Triton loops have no
  `break`/`return`).

## 5. Relevance to the mjolnir GEMV Kernel

This review converges on one canonical decode design, and every independent
implementation — the CRFM FlashDecoding recipe, FlashInfer's default decode path, and
vLLM's Triton backend — landed on the same structure: partition the KV sequence into
chunks, compute per-chunk partial attention plus logsumexp in parallel across SMs, and
merge partials with the base-2 LSE-weighted reduction of Section 1. FlashInfer's
default (`use_tensor_cores=False`) is a pure CUDA-core **GEMV** kernel: one CTA per
(KV chunk, KV head); 128-bit vector FMA dots (`q·k`, then `p·v`) reduced by
warp-shuffle butterfly; register-resident base-2 online softmax; 2-stage `cp.async`
double-buffered KV tiles; a shared-memory page-offset table so page size does not affect
bandwidth; and GQA head-group fusion so one KV load serves the whole query-head group.
Its split plan is occupancy-driven — `max_grid =
active_blocks_per_SM × num_SM` from the CUDA occupancy API, then a binary search for
the minimum pages-per-chunk that fits the grid budget — which is the practical form of
the FlashDecoding 3-step algorithm on a real machine.

That pattern is precisely the right one for mjolnir's target shape: M=1 decode at
`head_dim=256` on the NVIDIA Jetson AGX Thor (sm_110a, 20 SMs). By roofline (Section
3.1) this shape is bandwidth-bound: per KV token the arithmetic intensity is
`O(1)`–`O(g)` FLOP/byte, far below any modern GPU's ridge point, so the FLOPs cannot
be reduced (they are already trivial) — the only wins are maximizing achieved KV
bandwidth and minimizing instruction overhead per byte moved. Tensor-core GEMM machinery
(128-row M-tiles, tmem staging, multi-CTA clusters) is per-KV-token overhead at M=1: it
pays for compute that is free. FlashInfer's own analysis of *when* split-KV helps
(Section 3.4) predicts mjolnir's situation directly: a part with few SMs and a
bandwidth ceiling that a small CTA count cannot saturate must split the KV to fill all
SMs — at batch 1 with 4 KV heads, only 4 of 20 SMs would otherwise be active, and the
split plan lifts the scan to ~`4 × #SM` CTAs.

The motivation is confirmed by measurement: FlashInfer's GEMV path measures **72.5 µs**
for the full decode step (plan cached, GEMV + merge) at L=8192 on Thor, versus
**324.1 µs** for the tensor-core FA4 hd256 fp8 kernel — a 4.47× gap, with per-KV-token
scan slopes of 5.3 vs ~30 µs/1024 tokens (~5.7×). The gap is not FP8, not descaling,
and not CUDA-graph effects; it is raw small-M decode-scan efficiency. The full source
analysis of the FlashInfer GEMV kernel is in
[`docs/fa4-hd256-fp8/fi-decode-gemv-analysis.md`](../fa4-hd256-fp8/fi-decode-gemv-analysis.md);
the resulting FA4-native kernel — `sm100_hd256_decode_gemv.py`, an FA4/CuTe-DSL
pure-FMA GEMV with 16-wide FMA dots, log2-domain online softmax, a `cp.async` KV ring,
split-KV with in-tree LSE-merge, and the FlashInfer-Algorithm-1-style auto split plan
(`4 × #SM` CTAs), env-gated and default-off — is documented in
[`docker/vllm-thor/fa4-gemv-kernel/README.md`](../../docker/vllm-thor/fa4-gemv-kernel/README.md).

## Sources

Primary sources (archived and read in full during the research session):

| # | Source | Used in |
|---|--------|---------|
| 1 | FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning — https://arxiv.org/abs/2307.08691 (read from https://arxiv.org/html/2307.08691v1) | §1 |
| 2 | "Flash-Decoding for long-context inference" (CRFM / HuggingFace blog, Dao, Haziza, Massa, Sizov) — https://crfm.stanford.edu/2023-10-12/flashdecoding.html | §2, §5 |
| 3 | "FlashInfer: An efficient and customizable attention engine for LLM inference serving" (introduction blog) — https://flashinfer.ai/2024-02-02/introduce-flashinfer.html | §3 |
| 4 | FlashInfer attention API documentation — https://docs.flashinfer.ai/api/attention.html | §3 |
| 5 | FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving (paper) — https://arxiv.org/abs/2501.01005 (read from https://arxiv.org/html/2501.01005v2) | §3 |
| 6 | The Anatomy of a Triton Attention Kernel (paper) — https://arxiv.org/abs/2511.11581 (read from https://arxiv.org/html/2511.11581v1) | §4 |
| 7 | "vLLM Triton Attention Backend Deep Dive" (vLLM blog, 2026-03-04) — https://vllm.ai/blog/2026-03-04-vllm-triton-backend-deep-dive | §4 |

Cited/referenced within the sources:

- FlashAttention repo: https://github.com/Dao-AILab/flash-attention
- xFormers: https://github.com/facebookresearch/xformers
- FlashInfer cascade/shared-prefix post: https://flashinfer.ai/2024-02-02/cascade-inference
- FlashInfer DeepSeek-MLA post: https://flashinfer.ai/2025-02-10/flashinfer-deepseek-mla.html
- FlashInfer GitHub (incl. Sept 1, 2023 split-KV checkpoint `2977506`): https://github.com/flashinfer-ai/flashinfer
- TVM Unity Open Development Meeting talk (Sept 5, 2023): https://youtu.be/GcbuODb51Sc
- CUTLASS Split-K/parallelized reductions: https://github.com/NVIDIA/cutlass
- nvbench (kernel profiling): https://github.com/NVIDIA/nvbench
- vLLM / PagedAttention paper: https://arxiv.org/abs/2309.06180
- Grouped-Query Attention paper: https://arxiv.org/abs/2305.13245
- Speculative decoding (Medusa et al.): https://arxiv.org/abs/2211.17192
- H2O KV pruning: https://arxiv.org/abs/2306.14048 · Streaming-LLM: https://github.com/mit-han-lab/streaming-llm
- LightLLM: https://github.com/ModelTC/lightllm · SGLang: https://github.com/sgl-project/sglang
- Atom (int4 KV on FlashInfer): https://github.com/efeslab/Atom/
- LeanAttention: https://arxiv.org/abs/2405.10480
- Skip-softmax sparsity: https://arxiv.org/abs/2512.12087
- Reproducible inference / deterministic split-KV: https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/
- Triton open-source kernel + micro-benchmark suite: https://ibm.biz/vllm-ibm-triton-lib
- PyTorch blog, enabling vLLM v1 on AMD GPUs with Triton: https://pytorch.org/blog/enabling-vllm-v1-on-amd-gpus-with-triton/
- PyTorch blog, portable paged attention in Helion: https://pytorch.org/blog/portable-paged-attention-in-helion/
- vLLM repo (attention backends): https://github.com/vllm-project/vllm · Helion draft PR: https://github.com/vllm-project/vllm/pull/27293 · Helion: https://github.com/pytorch/helion
- Triton: https://github.com/triton-lang/triton · PTX ISA: https://docs.nvidia.com/cuda/parallel-thread-execution/ · DLPack: https://github.com/dmlc/dlpack
- AMD ROCm 6.1.1 documentation, "Optimizing Triton Kernels" (autotunable kernel
  configuration; the archived source's URL string is corrupted by page-scrape artifacts,
  so it is cited by name only)
