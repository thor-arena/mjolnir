# fa4-gemv-kernel — head_dim=256 GEMV decode kernel for vLLM FA4 on Thor

An FA4-native (CuTe-DSL) **pure-FMA GEMV decode kernel** for `head_dim=256` on
SM100/SM110 (NVIDIA Thor, CC 11.0), added to the vLLM `vllm_flash_attn` cute
interface. It replaces the tcgen05/tensor-core path for the M=1 (decode) shape
with a FlashInfer-style GEMV: one CTA per `(kv_head, kv_chunk, batch)`,
16-wide FMA dots + butterfly shuffle-reduce, log2-domain online softmax,
`cp.async` KV ring — no tensor cores.

**Supported scope** (gates live in the dispatch block, see `interface-gemv-dispatch.diff`):
- M=1 (`max_seqlen_q == 1`), `head_dim == head_dim_v == 256`, non-local
- GQA any `qhead_per_kvhead` (measured at 24/4 = g=6)
- dense **or** paged KV (`page_size` 16/128), bf16/fp16/fp8-e4m3 Q and KV
  (mixed Q/KV dtypes allowed only under the GEMV knob — fp8 + descales supported)
- SplitKV (`num_splits > 1`) with in-tree LSE-merge combine; write-through
  (no partials) when `num_splits == 1`

## Install

> **This kernel is now a real default-on build patch.** In the `vllm-thor`
> build: the **kernel `.py`** (`sm100_hd256_decode_gemv.py`, in this package)
> is installed by a **separate `COPY`** in the `Dockerfile` (a NEW file
> upstream has none of), and the **4 `interface.py` dispatch hunks** (the
> `interface-gemv-dispatch.diff` reference, rebased onto the current interface)
> land via `patches/thor-fa4-hd256-gemv-decode-sm110.patch`, applied by
> `apply_patches.py` after the 1CTA-decode patch. The dispatch is **default-on**
> (opt out with `VLLM_FA4_HD256_GEMV=0`). The working bench trees in
> `docs/fa4-hd256-fp8/` remain the measurement record.

1. Kernel file → in-image path (imported as
   `from vllm.vllm_flash_attn.cute.sm100_hd256_decode_gemv import BlackwellHd256DecodeGEMV`):

   ```
   vllm/vllm_flash_attn/cute/sm100_hd256_decode_gemv.py
   ```

    In the `vllm-thor` image, this file is `COPY`ed from this package directly
    into `/usr/local/lib/python3.12/dist-packages/vllm/vllm_flash_attn/cute/`
    (a separate `Dockerfile` step, not a patch hunk). The standalone bench
    harness still mounts the `fa4-gemv-kernel/` package separately.

2. Interface dispatch → the 4 hunks are in
   `patches/thor-fa4-hd256-gemv-decode-sm110.patch` (applied by
   `apply_patches.py`). `interface-gemv-dispatch.diff` in this package is the
   same 4 sections (import, `_gemv_auto_num_splits` FlashInfer-Alg.1 split
   plan, the `_gemv_dtype_ok` mixed-dtype allowance, and the `use_gemv_hd256`
   dispatch block) cut against the stock v10 base — kept as the reference;
   the build patch is the authoritative, rebased copy.

## Dispatch knobs

| env var | default | meaning |
|---|---|---|
| `VLLM_FA4_HD256_GEMV` | `1` (**on**) | `=1` (the default) routes in-scope M=1 hd=256 calls to the GEMV kernel; `=0` opts back to the tcgen05/1CTA path |
| `VLLM_FA4_HD256_GEMV_NUM_SPLITS` | unset | SplitKV plan precedence: this env (>0) > caller `num_splits` > `_gemv_auto_num_splits` auto (fills `4 × #SM` CTAs). Capped by the 128-row kv-chunk floor and 64 (combine max). At L=8192, GQA 4-head KV: auto → ns=20 → 80 CTAs |
| `VLLM_FA4_HD256_GEMV_DEBUG` | `0` | `=1` prints per-call ns/split/grid diagnostics to stderr |

The kernel is compile-cached in-process (`_flash_attn_fwd.gemv_compile_cache`,
keyed by shape/dtype/split/paged — **not** by `stages`; see the bench's leg
ordering constraint).

## Results (L=8192, M=1, GQA 24/4, bf16 Q + fp8 KV, gated clean window)

Wall-clock, median of 300, in one gated clean window (see "Methodology" in
`docs/gemv-ring-fix-bench.md`):

| config | wall (μs) | note |
|---|---|---|
| GEMV ns=1 (4 CTAs, write-through) | 1305 | |
| GEMV ns=20 (80 CTAs) | **222.8** | best GEMV; auto picks this |
| GEMV ns=64 (256 CTAs) | 233.5 | combine cost eats the kernel gain |
| FlashInfer FA2-tc (same window) | **202.05** | production baseline |

- **GEMV (auto/ns=20) is within ~10% of FlashInfer** (222.8 vs 202.05 μs).
  Both numbers are depressed by ungatable desktop Xorg GPU co-tenancy; the
  same FI kernel measured **73 μs** in a fully-clean window on Sept 22.
- The two levers, decomposed (kernel ncu achieved BW, L2-fabric convention):
  - **SplitKV / CTA count is the real lever**: ns=1→20 = ×8.7 kernel BW
    (13.5 → 118.3 GB/s; roofline 273 GB/s) and ×5.9 wall.
  - **Ring depth (stages 2→16) is correctness-neutral** (st16 output is
    bit-identical to st2) and only adds +12.9% kernel BW at ns=20 — a wall
    wash at L=8192. It is the right direction for larger L where the KV
    leaves the 32 MiB L2 and per-CTA in-flight depth limits DRAM streaming.
- Null results (ruled out, do not re-litigate): not LDGSTS-vs-LDG (both
  lower to plain LDG on sm_110a), not occupancy (96-thread CTAs, 4 CTA/SM
  smem-bound at st16, not the limit).
- Closing the remaining ~10% vs FI would require matching FI's tile shape
  (192 KB 384-row K+V tiles, 128-thread blocks — an optional B3 mma.sync
  redesign), not a ring-depth tweak.

## Bench / verify scripts

All scripts assume the v11 bench container (`mjolnir/vllm-thor:qwen38-sm110-v11`)
with the GEMV'd `vllm_flash_attn` tree mounted at
`/usr/local/lib/python3.12/dist-packages/vllm/vllm_flash_attn` and the working
dir at `/p`. Example (paths adjusted for this package):

```
docker run --rm --gpus all --network host --entrypoint python3 \
  -v $VFA_TREE:/usr/local/lib/python3.12/dist-packages/vllm/vllm_flash_attn \
  -v $WORKDIR:/p -e TMPDIR=/tmp/ringfix \
  mjolnir/vllm-thor:qwen38-sm110-v11 /p/gemv-ring-fix-bench.py --out /p/gemv-ring-fix-bench.json
```

| script | purpose |
|---|---|
| `verify-gemv-decode.py` | Phase-A correctness vs fp32 ref (dense/varlen, mixed dtypes, GEMV on/off) — 15 cases |
| `verify-gemv-b1.py` | Phase-B1 correctness (SplitKV partials, paged, combine) |
| `verify-gemv-stages.py` | stages 2 vs 16 vs fp32 ref — confirms the ring fix is bit-neutral |
| `gemv-decode-bench.py` | B1 four-mode bench (gemv_dense/gemv_paged/fa4_1cta/flashinfer), gated |
| `gemv-ring-fix-bench.py` | the ring-fix re-measure: ns sweep + stages contrast + FI, one gated window (stricter CONFIRM=6 + per-burst re-check) |
| `ncu-gemv-ring.py` | ncu driver (`NCU_NS`/`NCU_STAGES` env knobs; run under `ncu --privileged` with counter perms) |
| `run-gemv-bench-all.sh` | driver: all four bench modes, one docker run each (edit the `VFA`/`TASK` paths at the top) |

## Methodology (how the numbers above were measured)

- **Clean-window gate** (wall-clock): the Thor GPU is shared with the live
  vLLM server that serves these agents. Every bench polls
  `vllm:num_requests_running/waiting` at `http://127.0.0.1:6001/metrics`
  every 2 s; the timed window **opens only after several consecutive 0/0
  samples** (CONFIRM=6 for the ring-fix bench; 3 for the B1 bench) and the
  run is aborted (DIRTY) if any in-window sample is non-zero, with an
  additional 0/0 re-check immediately before each timed burst. The vLLM
  server is **never** stopped or restarted.
- **ncu achieved-BW over wall-clock under desktop co-tenancy**: the desktop
  Xorg session's GPU load cannot be gated (no request-level metric). Kernel
  counters (`lts__t_sectors.sum × 32 B / gpu__time_duration.sum` — L2-fabric
  BW; `dram__bytes.sum` is n/a on CC 11.0) are far more robust, so absolute
  BW conclusions rest on ncu and wall-clock is used for relative ratios.
- **Relative ratios in the same gated window**: ns-vs-ns and st16-vs-st2 are
  compared inside one window so co-tenancy cancels; the stages contrast is
  done by swapping compiled variants in the GEMV compile cache between
  measurement blocks (the cache does not key on `stages`).
- **Two-lever decomposition**: CTA-count (ns) × per-CTA in-flight depth
  (stages), varied independently at L=8192.
- Full write-up: `docs/gemv-ring-fix-bench.md` (results + analysis),
  `docs/gemv-fi-gap-final.md` (earlier ncu gap analysis vs FI),
  `docs/gemv-decode-design.md` (kernel design, Phase A/B1).
- Raw numbers: `docs/gemv-ring-fix-bench.json` (wall + ncu + verification).
