# FA4 + FP8-KV on Jetson Thor (sm_110) — Empirical Verification

> **Status:** Verified — live probes + PTX inspection on sm_110a · **Date:** 2026-09 (workstream window 2026-09-18 → 2026-09-21; verified against the base nightly `vllm 0.29.1rc1.dev347`, 2026-09-18) · **Scope:** whether FA4 (flash-attn v4, CuTe-DSL) can run with an FP8 KV cache on Jetson Thor in vLLM 0.29.1, and exactly what blocks it in stock vLLM

**Verdict: YES.** FA4 can run with an FP8 KV cache on Jetson Thor (sm_110) in this exact vLLM build. The `arch // 10 == 10` restriction that stock vLLM enforces is a **policy gate, not a hard hardware constraint**. The real FA4 kernel — pure-Python CuTe-DSL, JIT-compiled at runtime — compiles cleanly **and** executes correctly for FP8 on `sm_110a`, matching an fp32 reference to within FP8 quantization error.

What is true: **stock vLLM refuses the combination at startup** via two Python policy checks. Once those two checks are widened to admit fam(110), FA4+FP8-KV runs end-to-end (dense *and* paged/varlen/GQA/causal), with no JIT-compile error, no cubin-load error, and no illegal instruction.

Everything below was verified against the byte-identical code in both the pristine vLLM 0.29.1 package root and the v6 overlay image (`mjolnir/vllm-thor:qwen38-sm110-v6`; md5 of the three gating files is identical in both — see §7).

---

## TL;DR (the six questions)

| # | Question | Answer |
|---|----------|--------|
| 1 | Does FA4 even get *selected* on a GDN hybrid model on sm_110? | Yes for the 16 full-attn layers (`FLASH_ATTN` is the first candidate). The 48 GDN layers never touch FA — they route through a separate `MambaAttentionBackendEnum.GDN_ATTN` path. No hybrid-layout blocker. |
| 2 | Which FA version does vLLM pick, and how to force FA4? | Default on sm_110 is **FA2** (major==11 falls into the FA2 fallback). Force FA4 with `--attention-config.flash_attn_version=4` (config field, **no env var exists**). |
| 3 | Where is the FP8-KV gate? | `fa_utils.py:306-308` — `(fa==3∧fam90)∨(fa==4∧fam100)`. This is the *only* policy gate that rejects FA4+FP8 on sm_110. |
| 4 | What does the kernel itself require? | FA4 = CuTe-DSL JIT. On sm_110 the FP8 kernel **compiles and runs correctly**. The kernel-internal assert `interface.py:910` (`arch//10==10`) is the *only* hard stop, and it is a policy assert, not an arch capability check. |
| 5 | If forced, where does it fail? | Ranked: (a) startup `ValueError` from the backend-eligibility gate; (b) if only that is bypassed, `AssertionError` at `interface.py:910` on the first forward; (c) with **both** widened — it **works**. |
| 6 | Cudagraph interaction? | FA4 → `UNIFORM_BATCH` (`flash_attn.py:562-566`), the same cudagraph level the existing Thor FlashInfer patches provide. No new cudagraph regression. |

---

## 1. Backend Eligibility on a GDN Hybrid

`platforms/cuda.py:167-179` puts `FLASH_ATTN` **first** in the candidate list on sm_110. For a Qwen3.8-27B GDN hybrid (16 full-attn + 48 GDN layers):

- **Full-attn layers (16):** each goes through `get_attn_backend` (per-layer selector). `FLASH_ATTN` is a valid candidate → it is selected (subject to the FP8 gate, §3).
- **GDN layers (48):** route through a *different* mechanism that never consults `get_attn_backend`:
  - `model_executor/layers/mamba/gdn/base.py:50-51` → `MambaAttentionBackendEnum.GDN_ATTN`
  - `model_executor/layers/mamba/abstract.py:86-88` → `get_mamba_attn_backend(mamba_type)`
  - `model_executor/layers/mamba/mamba_utils.py:21,373` consumes it.

So the GDN layers use the mamba backend independently and impose **no** constraint on the FA backend choice. There is no "hybrid layout" blocker — the per-group selection flow is exactly the one that already selects FlashInfer for these layers. Switching the full-attn group to `FLASH_ATTN` is a clean per-group override (`AttentionConfig.backend_per_kind`, `config/attention.py:46-55`).

---

## 2. Version Selection & How to Force FA4

`get_flash_attn_version` (`v1/attention/backends/fa_utils.py:74-221`):

```
74  def get_flash_attn_version(...):
...
99      if device_capability.major == 9 and is_fa_version_supported(3):
101         fa_version = 3                       # Hopper -> FA3
102     elif device_capability.major == 10 and is_fa_version_supported(4):
104         fa_version = 4                       # SM100 -> FA4
105     else:
107         fa_version = 2                       # <- sm_110 (major==11) lands HERE
...
109     # 2. override if passed by environment or config
113     if (vllm_config is not None
115         and vllm_config.attention_config.flash_attn_version is not None):
117         fa_version = vllm_config.attention_config.flash_attn_version
...
120     if device_capability.major >= 10 and fa_version == 3:
125         fa_version = 4 if is_fa_version_supported(4) else 2
```

Consequences on sm_110:

- **Default = FA2** (major==11 is neither 9 nor 10 → fallback at `:107`).
- **To force FA4:** set `AttentionConfig.flash_attn_version = 4` (`config/attention.py:57-59`). CLI: `--attention-config '{"flash_attn_version": 4}'`. There is **no** `VLLM_FLASH_ATTN_VERSION` env var anywhere in the tree — the override is config-only.
- `is_fa_version_supported(4)` returns **True** on fam(110) (`vllm_flash_attn/flash_attn_interface.py:77-85`), so FA4 is an *allowed* version on Thor; the only thing standing between "allowed" and "selected with FP8" is the gate in §3.

---

## 3. The FP8-KV Policy Gate (the one that must be opened)

`flash_attn_supports_kv_cache_dtype` (`v1/attention/backends/fa_utils.py:282-308`):

```
306  return (fa_version == 3 and current_platform.is_device_capability_family(90)) or (
307      fa_version == 4 and current_platform.is_device_capability_family(100)
308  )
```

For `fa_version==4` this returns True **only** on fam(100). On fam(110) it returns False. This is consumed by `FLASH_ATTN.supports_combination` (`v1/attention/backends/flash_attn.py:425-434`):

```
434      return "FP8 KV cache requires FA3 on SM90 or FA4 on SM100"
```

That string is the exact error a user sees when forcing FA4+FP8-KV on Thor.
`flash_attn_supports_quant_query_input` (`fa_utils.py:311-312`) is unrelated (returns True on all non-XPU) and is not a blocker.

> **This line 306-308 is the single policy gate.** Widening `family(100)` to also accept `family(110)` for FA4 removes the startup rejection. (§6 shows the concrete patch.)

---

## 4. Kernel-Level Truth (the decisive evidence)

### 4.1 What FA4 actually is on this build

FA4 is **not** a prebuilt cubin. It is pure-Python CuTe-DSL (`nvidia-cutlass-dsl` 4.7.1) JIT-compiled at runtime:

- `vllm_flash_attn/cute/interface.py` → `_flash_attn_fwd`
- `vllm_flash_attn/cute/flash_fwd_sm100.py` → the `FlashAttentionForwardSm100` kernel (used for **both** arch 10 and arch 11)
- JIT path: `cutlass.cute` → MLIR → PTX → **ptxas** → cubin, at first use.

By contrast the older backends *are* prebuilt .so files, and their arch strings prove they were never built for sm_110:

| kernel | artifact | arch (from .so / metadata) |
|--------|----------|----------------------------|
| FA2 | `_vllm_fa2_C.abi3.so` (307 MB) | **sm_80 only** |
| FA3 | `_vllm_fa3_C.abi3.so` (145 MB) | **sm_90a only** |
| FA4 | (no .so — pure Python `cute/`) | JIT → whatever the device is |

So FA2 and FA3 *cannot* run FP8 on sm_110 at all (wrong-arch prebuilt binaries). FA4 is the **only** FA variant with any path to sm_110, because it recompiles per-arch.

### 4.2 The kernel-internal arch assert

`vllm_flash_attn/cute/interface.py:909-910`:

```
909   if is_fp8:
910       assert arch // 10 == 10, "FP8 is only supported on SM100 (compute capability 10.x) for FA4 CuTe."
```

This is a **Python `assert`, not a PTX/cubin capability check**. It runs *before* JIT compilation, as a fast-fail. The surrounding code has no other arch-specific branch that would break on fam(110):

- `fp8_kv_dequant` (bf16 Q + fp8 KV, in-kernel dequant) is SM90-only (`:903`) and is a **separate** feature — vLLM's Attention layer quantizes Q to FP8 when KV is FP8 (`model_executor/layers/attention/attention.py:516-517`), so the *full-FP8* (`is_fp8`) path is what vLLM uses. That path has no SM90 restriction.
- The FP8 tuning/registry config in `flash_fwd_sm100.py:112-120` and `_FP8_SMALL_HDIM_REGS` in `interface.py` explicitly carry entries keyed by `paged_kv_non_tma` True/False — i.e. the paged (cp.async) KV path is a **supported** FP8 configuration, not just the TMA path.
- `interface.py:669` documents `output_scale` as "SM100/SM110 only" — the codebase already *anticipates* sm_110 in this kernel family.

### 4.3 Live probes on the Thor (sm_110a)

All probes ran in the pristine nightly image with only the single assert `interface.py:910` widened from `== 10` to `in (10, 11)` (no other change).

**A. Dense FP8 fwd (GQA, hd128, non-causal):**

```
fp8 fwd on sm_110 OK: (1, 64, 2, 128) torch.bfloat16
fp8 max abs err vs fp32 ref: 0.0156   (relerr ~2.25% — consistent with e4m3 quantization)
```

**B. BF16 fwd (proves the tcgen05+TMEM datapath works on sm_110):**

```
bf16 dense maxerr: 0.0028 (MHA) / 0.0027 (GQA nc) / 0.0076 (GQA causal)
```

**C. Paged + varlen + GQA + causal — the *exact* vLLM production shape** (4 requests, kv 96/128/64/112, block_size=16, block_table, `seqused_k`, GQA 8q/4kv, hd128, `causal=True`):

```
bf16 paged: kernel_nan=False ref_nan=False maxerr=0.0022
fp8  paged: kernel_nan=False ref_nan=False maxerr=0.0149
```

**D. Paged-vs-dense cross-check** (isolates the de-paging logic):

```
paged-vs-dense kernel diff: 0.00000   (identical to bf16 precision)
```

> The kernel's de-paging is exactly correct (diff 0.0 vs the dense kernel on the same de-paged data), and the FP8 paged result matches an independent fp32 reference to e4m3 quantization error. This is the configuration vLLM actually issues.

### 4.4 PTX evidence (the hardware verdict)

The JIT PTX was dumped (`CUTE_DSL_KEEP_PTX=1 CUTE_DSL_DUMP_DIR=...`) and inspected:

```
file:  ...FlashAttentionForwardSm100_...sm_110a.ptx        <- compiled FOR sm_110a
.target sm_110a
.address_size 64

instruction inventory (from the FP8 fwd kernel):
  16  tcgen05.mma  ...  kind::f8f6f4      <- 5th-gen tensor-core MMA, FP8/FP6/FP4 operands
  24  tcgen05.ld                                   <- TMEM loads
  16  tcgen05.st                                   <- TMEM stores
   8  tcgen05.commit / 5 tcgen05.wait::st
   1  tcgen05.alloc / 1 tcgen05.dealloc / 1 tcgen05.relinquish_alloc_permit
  28  tmem* references total
 144  e4m3/e5m2/f8f6f4 dtype markers
```

**This is conclusive.** The FA4 FP8 kernel, when JIT-compiled for `sm_110a`, emits `tcgen05.mma kind::f8f6f4` (the 5th-generation tensor-core FP8 MMA) and TMEM alloc/ld/st, and **ptxas (CUDA 13.0) accepts all of it for `sm_110a`**. If sm_110's ISA did not support these instructions, ptxas would have failed the build — it did not. Thor's tensor cores are the same `tcgen05`/TMEM generation as B200; the FP8 path is genuinely available on the silicon.

> Note on TMEM sizing: `flash_fwd_sm100.py:310` hardcodes `get_max_tmem_alloc_cols("sm_100")` (the `nvidia_cutlass_dsl` TMEM map only has sm_100/sm_103/sm_120, no sm_110 entry). Because the bf16 *and* fp8 runs succeeded, the sm_100 TMEM column budget is a **safe upper bound** on sm_110 (Thor's TMEM is the same size as B200's). This is a sizing constant, not a functional gap.

---

## 5. If Forced, Where Does It Fail? (Ranked)

Forcing = `--attention-config '{"backend":"FLASH_ATTN","flash_attn_version":4}'` plus `--kv-cache-dtype fp8_e4m3`, on stock vLLM, sm_110:

1. **Startup `ValueError`** — backend-eligibility gate rejects before any kernel work: `platforms/cuda.py` forced-backend path (`:459-474`) → `FLASH_ATTN.supports_combination` (`flash_attn.py:434`) → `"FP8 KV cache requires FA3 on SM90 or FA4 on SM100"`. *(This is the gate at `fa_utils.py:306-308`.)*
2. **`AssertionError` at `interface.py:910`** — if only step-1's gate is widened but the kernel assert is left alone, the first forward that reaches `_flash_attn_fwd` with `is_fp8=True` trips `assert arch // 10 == 10`. Fails **before** JIT compilation.
3. **No further failures.** With both (1) and (2) widened, there is **no** JIT-compile error, no cubin-load error, and no illegal-instruction abort — the kernel compiles to `sm_110a` PTX/cubin and produces correct output (proven in §4.3-4.4).

So the entire "cost" of enabling FA4+FP8-KV on Thor is **two policy assertions** — there is no missing kernel, no missing arch cubin, and no ISA gap.

---

## 6. The Probe-Gated Enablement Patch

Minimal, arch-gated, non-invasive — two one-line widenings, each guarded so it only affects fam(110). This is the [thor-fa4-fp8kv-sm110](../../docker/vllm-thor/patches/thor-fa4-fp8kv-sm110.patch) patch in the stack (see [docker/vllm-thor/PATCHES.md](../../docker/vllm-thor/PATCHES.md)).

**Patch 1 — open the backend-eligibility gate**
`v1/attention/backends/fa_utils.py:306-308`:

```python
    return (fa_version == 3 and current_platform.is_device_capability_family(90)) or (
        fa_version == 4
        and (
            current_platform.is_device_capability_family(100)
            or current_platform.is_device_capability_family(110)   # + Thor (sm_110)
        )
    )
```

**Patch 2 — open the kernel arch assert**
`vllm_flash_attn/cute/interface.py:910`:

```python
    if is_fp8:
        assert arch // 10 in (10, 11), (
            "FP8 is only supported on SM100/SM110 (compute capability 10.x/11.x) for FA4 CuTe."
        )
```

Then launch:

```
vllm serve ... \
  --kv-cache-dtype fp8_e4m3 \
  --attention-config '{"backend":"FLASH_ATTN","flash_attn_version":4}'
```

(For the GDN hybrid, prefer scoping the override to the full-attn group so the GDN layers keep their mamba backend: use `backend_per_kind` / the per-group selector rather than a global `backend`, or rely on the fact that GDN layers never consult `FLASH_ATTN`.)

---

## 7. Build Provenance

The three gating files are **byte-identical** between the pristine package root and the running Thor image, so every `file:line` above applies to the deployed build (`vllm 0.29.1rc1.dev347+gdee37d891`, docker `vllm/vllm-openai:nightly-aarch64` and the local `mjolnir/vllm-thor:qwen38-sm110-v6`):

```
74ae06affeed112fe5677de050a09d06  vllm_flash_attn/cute/interface.py
e2c98ad19b6809d89003027603a1aefc  v1/attention/backends/fa_utils.py
b47e0aed5fc7bbe9239561d441868a39  v1/attention/backends/flash_attn.py
```

(The running image's local patches are all FlashInfer/cudagraph-related and do not touch any of these three files; see [docker/vllm-thor/PATCHES.md](../../docker/vllm-thor/PATCHES.md).)

---

## 8. Caveats & What to Measure Next

1. **Numerics.** FP8 (e4m3) attention on Thor shows ~0.015 max abs error vs an fp32 reference in the probes — expected for e4m3 KV. Confirm end-to-end task accuracy / perplexity is acceptable for the NVFP4 model; FA4's FP8 tuning tables were *developed and tuned on B200/sm_100*, so the register/tile choices on sm_110 are inherited, not Thor-tuned.
2. **Performance.** The probes prove *correctness*, not speed. FA4's `tcgen05` FP8 path should be fast on Thor (same tensor-core generation as B200), but the sm_100-derived TMEM/tile config and the `page_size != tile_n` → cp.async (non-TMA) KV path (vLLM default `block_size=16`) mean throughput must be benchmarked. If KV `block_size` can be set to 128 (= `tile_n`), the faster TMA KV path is used.
3. **First-run JIT cost.** FA4 compiles on first use (a few seconds per distinct config). With cudagraph capture the compiled kernel is cached; ensure the capture warm-up covers the FP8 config so the JIT happens before/under graph capture as intended.
4. **Cudagraph.** FA4 → `UNIFORM_BATCH` (`flash_attn.py:562-566`), the same level the existing Thor FlashInfer patches provide — no new cudagraph regression introduced by switching the full-attn group to FA4.

---

## 9. Bottom Line

- **Can FA4 run with an FP8 KV cache on Jetson Thor sm_110 in vLLM 0.29.1?** **Yes.**
- The hardware (sm_110a) fully supports the required `tcgen05.mma kind::f8f6f4` + TMEM instructions; the FA4 CuTe-DSL kernel compiles to `sm_110a` and computes correct FP8 attention (dense *and* paged/varlen/GQA/causal) to e4m3 precision.
- Stock vLLM blocks it purely via **two Python policy checks** (`fa_utils.py:306-308` and `interface.py:910`), both of which are one-line, arch-gated widenings. FA2/FA3 are *not* viable for FP8 on sm_110 (wrong-arch prebuilt binaries), so FA4 is the only FA path on Thor.
- Next step: apply the two-line patch, benchmark throughput + accuracy of the GDN-hybrid NVFP4 model with `--kv-cache-dtype fp8_e4m3` and FA4, and compare against the FlashInfer baseline.
