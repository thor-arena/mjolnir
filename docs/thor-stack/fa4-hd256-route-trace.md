# FA4 hd256 Downgrade Route Trace — Why FA4 + FP8-KV + hd256 Is Gated on sm_110

> **Status:** Complete — code-level trace against the pristine vLLM 0.29.1 package root · **Date:** 2026-09 (workstream window 2026-09-18 → 2026-09-21; anchors to the `vllm 0.29.1rc1.dev347` nightly of 2026-09-18) · **Scope:** the FA4→FA2 downgrade path for `head_dim=256` attention with quantized KV caches on sm_100/sm_110, and the kernel work it would take to enable FA4 + FP8-KV + hd256 on Thor

Companion documents: the hd128 kernel proof ([fa4-fp8kv-sm110.md](fa4-fp8kv-sm110.md)); the serving-config verdict and the per-patch inventory are covered in [docker/vllm-thor/PATCHES.md](../../docker/vllm-thor/PATCHES.md).

All anchors are in the pristine vLLM 0.29.1 package root (**flat layout**: `v1/...`, `vllm_flash_attn/...`, `model_executor/...`, `platforms/...`, `utils/...` are direct subdirectories of the package root).

---

## TL;DR

- The downgrade is condition **(b), not (a)/(c)/(d)**: on `major ∈ (10, 11)`, `head_size == 256` (and `head_size_v ∈ (None, 256)`) with a *resolved* FA version 4, vLLM downgrades FA4→FA2 when **any** of six fallback reasons fires. The one that kills the target deployment is **any quantized KV cache dtype** — `is_quantized_kv_cache(cache_config.cache_dtype)` (`fa_utils.py:252-253`; `utils/torch_utils.py:78-83` covers `fp8*`, `nvfp4*`, `*per_token_head`). It is dtype-based, **paged/dense agnostic**, reads the **engine-global** cache dtype (not per-layer), and fires identically on SM100 and SM110. Paged-ness only matters through a *separate, fourth* reason: `kv_cache_block_size % 128 != 0` (`fa_utils.py:254-256`), which fires even with **bf16** KV at the default block size 16.
- **hd256 + bf16 KV does allow FA4 on both major(10) and major(11)** — no escape hatch needed beyond `kv_cache_block_size` being a multiple of 128 (which the FA backend self-enforces: `flash_attn.py:295-318` makes the backend advertise exactly block size 128 whenever the FA4 hd256 path resolves). No env var, no config flag, no dense-only mode can bypass the quantized-KV reason.
- **The downgrade matches kernel reality on every Blackwell** — it is not a conservative sm_110 policy:
  - the main FA4 kernel (`FlashAttentionForwardSm100`) **cannot express hd256 at all**: `tmem_total = 256 + 2·hdv_padded` (cols) and `assert tmem_total <= 512` (`flash_fwd_sm100.py:347-353`) → hd128 fills exactly 512/512, hd256 needs 768;
  - the dedicated hd256 kernel (`BlackwellFusedMultiHeadAttentionForward`, `sm100_hd256_2cta_fmha_forward.py`) exists for **bf16/fp16 only**: `assert descale_tensors is None` (`:211-213`), and vLLM **always** passes q/k/v descales for FP8 KV (`flash_attn.py:1272-1280` → `flash_attn_interface.py:430-433, 461-463`). On SM100 an fp8+hd256 call therefore dies at that kernel assert; on sm_110 it dies earlier at the `arch // 10 == 10` assert (`interface.py:909-910`). The vLLM policy downgrade is the host-side mirror of these asserts, applied preemptively.
- **What enabling takes**: fp8 is the *only* missing dtype on an existing hd256 tile. Concretely: descale plumbing + the fp8 P-scaling machinery + FP8 tuning entries in `sm100_hd256_2cta_fmha_forward.py` (all three already exist, in identical form, in the sibling kernel `flash_fwd_sm100.py`), plus two vLLM gate edits. **Effort class: weeks** (focused 1–3 weeks), not days and not a new tile.

---

## 1. Exact Downgrade Conditions (`v1/attention/backends/fa_utils.py`)

### 1.1 The decision code

`get_flash_attn_version` (`fa_utils.py:74-221`), in order:

```python
 99      if device_capability.major == 9 and is_fa_version_supported(3):
101          fa_version = 3                        # Hopper default
102      elif device_capability.major == 10 and is_fa_version_supported(4):
104          fa_version = 4                        # SM100 default
105      else:
107          fa_version = 2                        # <- sm_110 (major==11) lands here
...
113      if (vllm_config is not None
115          and vllm_config.attention_config.flash_attn_version is not None):
117          fa_version = vllm_config.attention_config.flash_attn_version  # config-only override; NO env var
...
120      if device_capability.major >= 10 and fa_version == 3:
125          fa_version = 4 if is_fa_version_supported(4) else 2   # FA3 banned on Blackwell
...
171      if envs.VLLM_BATCH_INVARIANT and fa_version == 4:      # pushes the other way (4 -> 2)
176          fa_version = 2
...
178      if fa_version == 4 and uses_fa4_hd256_kernel(head_size, head_size_v):
179          if not supports_fa4_hd256:
180              fa_version = 2                       # silent (no log)
181          elif (reason := _fa4_hd256_fallback_reason(
183                  has_sinks, requires_softcap, kv_cache_block_size, vllm_config)) is not None:
186              logger.warning_once(
187                  "FA4's Blackwell head_size=256 kernel does not support %s, "
188                  "defaulting to FA version 2.", reason)
191              fa_version = 2
...
193      # FA4 head dimensions on Blackwell are limited by TMEM capacity.
194      if (fa_version == 4 and device_capability.major >= 10
197          and head_size is not None and head_size > 128
199          and not ((head_size == 256 and head_size_v in (None, 256))
201              or (head_size == 192 and head_size_v == 128))):
209          fa_version = 2
```

`uses_fa4_hd256_kernel` (`fa_utils.py:224-233`):

```python
228      if head_size != 256:
229          return False
230      if head_size_v is not None and head_size_v != 256:
231          return False
232      capability = current_platform.get_device_capability()
233      return capability is not None and capability.major in (10, 11)
```

`_fa4_hd256_fallback_reason` (`fa_utils.py:236-268`), first match wins:

```python
244      if has_sinks:
245          return "attention sinks"
246      if requires_softcap or (model_config ... "attn_logit_softcapping" ...):
251          return "logits soft capping"
252      if cache_config is not None and is_quantized_kv_cache(cache_config.cache_dtype):
253          return f"quantized KV cache dtype {cache_config.cache_dtype}"     # <<< the target killer
254      if kv_cache_block_size is not None and kv_cache_block_size % FA4_HD256_PAGE_SIZE:  # FA4_HD256_PAGE_SIZE = 128 (fa_utils.py:14)
256          return f"a KV cache block size of {kv_cache_block_size}"
257      if model_config is not None:
258          if model_config.is_mm_prefix_lm:  return "mm_prefix bidirectional attention"
260          if model_config.rswa_window is not None:  return "R-SWA"
262      if vllm_config.parallel_config.decode_context_parallel_size > 1:
267          return "decode context parallelism"
```

### 1.2 Enumeration — what exactly triggers FA4→FA2

All of these must hold:

1. **resolved `fa_version == 4`** — i.e. SM100 by default, or explicit `attention-config.flash_attn_version=4` (required on sm_110, whose default is 2 at `:107`);
2. **`head_size == 256`** and **`head_size_v is None or 256`**;
3. **device major ∈ (10, 11)** (SM100 *and* sm_110 — no sm distinction anywhere in this branch);
4. **one of**: `supports_fa4_hd256=False` (silent, `:179-180`; only non-FA callers like `turboquant_attn.py:352` and `mm_encoder_attention.py:364` use the default False — every `flash_attn.py` caller passes True: `:295-311`, `:408-450`, `:651-659`, `:1097-1131`), or one of the six fallback reasons in `:236-268`.

Precise answers to the sub-questions:

- **(a) hd256 + fp8-KV always?** Yes — for this deployment the `:252-253` line is the one that fires, and it fires for any fp8 KV (fp8/fp8_e4m3/fp8_e5m2).
- **(b) hd256 + any quantized KV?** **Yes — this is the exact condition**: `is_quantized_kv_cache` (`utils/torch_utils.py:78-83`) is `dtype.startswith("fp8") or endswith("per_token_head") or startswith("nvfp4")` — so nvfp4 and per-token-head KV caches trigger the identical downgrade.
- **(c) hd256 + fp8 + paged?** Paging is irrelevant to the quantized-KV reason — it reads `vllm_config.cache_config.cache_dtype` (engine-global, set at model load), never the paged/dense layout of a call. Dense or paged, fp8 or nvfp4: same downgrade.
- **(d) page_size/tile dependent?** Only via the *separate* fourth reason (`:254-256`): `kv_cache_block_size % 128 != 0` downgrades **any KV dtype**, including bf16. (128 is `FA4_HD256_PAGE_SIZE`, the TMA page the dedicated kernel requires — §2.2.)
- **hd256 + bf16 KV → FA4 allowed on major(10,11)?** **Yes.** No reason fires when cache_dtype is bf16 *and* block_size % 128 == 0 *and* no sinks/softcap/mm_prefix/R-SWA/DCP. On the model path the block size is forced to 128 anyway: `FlashAttentionBackend._get_fa4_hd256_block_size` (`flash_attn.py:295-311`) returns `FA4_HD256_PAGE_SIZE` whenever `get_flash_attn_version(..., supports_fa4_hd256=True) == 4`, and `get_supported_kernel_block_sizes` / `get_preferred_block_size` (`flash_attn.py:313-318`, `:322-328`) then pin the KV layout to 128-token pages. (Chicken-and-egg note: that probe call passes no `kv_cache_block_size`, so the `:254-256` reason cannot fire there; the fp8 reason *can* — and does — which is why an fp8-KV engine never even advertises 128 and falls back to the default 16.)
- **Escape hatches: none.**
  - Env vars: the only env var in this function is `VLLM_BATCH_INVARIANT` (`:171-176`), which forces FA4→FA2. No env var touches the hd256 branch.
  - Config flags: `supports_fa4_hd256` is hardwired `True` in every FA backend caller; the quantized check reads the global `cache_config` — no per-layer or per-group override.
  - Page-size choice: `kv-cache-block-size=128` silences only the `:254-256` reason, not `:252-253`.
  - Dense-only mode: vLLM decoder attention is always paged (the hd256 impl path even re-pads to page-aligned lengths, `flash_attn.py:1380-1386`); the encoder/dense path would still hit the same global cache-dtype downgrade.

### 1.3 Where the downgraded version is consumed

`flash_attn_supports_kv_cache_dtype` (`fa_utils.py:282-308`) re-runs `get_flash_attn_version` with the layer's head sizes (`:297-305`) and then applies the family gate:

```python
306      return (fa_version == 3 and current_platform.is_device_capability_family(90)) or (
307          fa_version == 4 and current_platform.is_device_capability_family(100)
308      )
```

For hd256+fp8 the version is *already* 2 at this point on **both** SM100 and SM110, so the gate is false even where the family would match. Consumed at:

- forced-backend validation: `platforms/cuda.py:457-477` → `FLASH_ATTN.supports_combination` (`flash_attn.py:408-450`, fp8 check at `:422-434`) → `ValueError(... "FP8 KV cache requires FA3 on SM90 or FA4 on SM100")` (`flash_attn.py:434`);
- per-layer construction: `flash_attn.py:1122-1136` → `NotImplementedError("FlashAttention does not support {kv_cache_dtype} kv-cache on this device.")`.

**Refinement of the configuration-side finding** (companion finding, covered in [docker/vllm-thor/PATCHES.md](../../docker/vllm-thor/PATCHES.md)): the claim "SM100 behaves identically" is confirmed, and the reason is now kernel-level, not policy-level (§2.2): the dedicated hd256 kernel has *no FP8 mode on any Blackwell arch*.

---

## 2. Kernel-Side Truth for hd256 (`vllm_flash_attn/cute/`)

### 2.1 What head dims the FA4 (CuTe) kernels support

Head-dim validation, `interface.py:112-129` (`_validate_head_dims`):

```python
117      is_standard_range = 8 <= head_dim <= 128 and 8 <= head_dim_v <= 128
119      is_sm90_range = 8 <= head_dim <= 512 and 8 <= head_dim_v <= 512
120      if compute_capability == 9:
121          assert is_sm90_range and ...                        # SM90: 8..512
125      elif compute_capability in [10, 11]:
126          assert (is_standard_range or is_deepseek_shape            # (192, 128)
127                  or is_deepseek_mla_absorbed_shape                # (64/128, 512) absorbed MLA
128                  or is_dedicate_kernel_shape) ...                 # (256, 256)
```

Dispatch, `interface.py:1011-1013, 1566-1636`:

```python
1012     use_dedicated_hd256_kernel = arch // 10 in [10, 11] and head_dim == 256 and head_dim_v == 256
1013     use_2cta_instrs = use_2cta_instrs or use_dedicated_hd256_kernel
...
1588                 if use_dedicated_hd256_kernel:
1589                     fa_fwd = BlackwellFusedMultiHeadAttentionForward(...)  # sm100_hd256_2cta_fmha_forward.py
1610                 else:
1611                     fa_fwd = FlashAttentionForwardSm100(...)                # flash_fwd_sm100.py
```

- **Main kernel (`FlashAttentionForwardSm100`, `flash_fwd_sm100.py`)**: arch family gate is **sm_100 *or* sm_110** —

  ```python
  193          self.arch = BaseDSL._get_dsl().get_arch_enum()
  194          assert self.arch.is_family_of(Arch.sm_100f) or self.arch.is_family_of(Arch.sm_110f), \
  195              "Only SM 10.x and 11.x are supported"
  ```

  — and its TMEM budget is the hard head-dim limit:

  ```python
  310          self.tmem_alloc_cols = cute.arch.get_max_tmem_alloc_cols("sm_100")   # 512 cols
  347          self.tmem_s_offset = [0, self.n_block_size]                         # S stages: cols 0, 128
  348          self.tmem_o_offset = [
  349              self.tmem_s_offset[-1] + self.n_block_size + i * self.head_dim_v_padded
  351          ]                                                                    # O stages: 256, 256+hdv
  352          self.tmem_total = self.tmem_o_offset[-1] + self.head_dim_v_padded   # = 256 + 2*hdv
  353          assert self.tmem_total <= self.tmem_alloc_cols
  ```

  i.e. **hdv ≤ 128** (hd128 → exactly 512/512; DeepSeek (192,128) → 512/512; hd256 → 768 → **impossible**). This is the TMEM fact behind the policy comment `fa_utils.py:193-209`. Note TMEM depends only on the fp32 accumulators — **fp8 vs bf16 changes nothing there**; it changes SMEM (Q/K/V storage, width-aware at `flash_fwd_sm100.py:395-404`) and registers.

- **Dedicated hd256 kernel (`BlackwellFusedMultiHeadAttentionForward`, `sm100_hd256_2cta_fmha_forward.py`)**: exists for **both** major 10 and 11 (dispatched at `interface.py:1012`), fixed 128×128 tile, 2CTA by default, TMEM exactly full:

  ```python
   65          assert head_dim == 256 and head_dim_v == 256
   68          assert score_mod is None
   69          assert not paged_kv_non_tma        # "TMA paged KV only (page_size must equal tile_n=128)"
   72          assert not pack_gqa
   73          assert not is_split_kv
   80          assert m_block_size == 128 and n_block_size == 128
  133          self.tmem_alloc_cols = cute.arch.get_max_tmem_alloc_cols("sm_100")  # 512
  150          self.tmem_s_offset = 0           # S/P: two 128-col fp32 stages -> cols 0..255
  151          self.tmem_o_offset = 256         # O: 256 cols fp32              -> cols 256..511  (512/512)
  ```

  Register budget (hd256 exception, `flash_fwd_sm100.py:83`): `num_regs_softmax=256`, `num_regs_correction=160`, `num_regs_other=56/32` (`sm100_hd256_2cta_fmha_forward.py:155-159`, selected via the `_TUNING_CONFIG` key `(True, is_causal, 256, False)` = `flash_fwd_sm100.py:109-110`). Pipeline stages: `q_stage=2`, `k_stage=4`/`v_stage=4` in 2CTA (`:166-171`).

### 2.2 Is there an hd256 code path for fp8? What exactly is missing?

**Yes for bf16/fp16; no for fp8 — on either Blackwell arch.** The dtype is read from the input tensors (`sm100_hd256_2cta_fmha_forward.py:397-401`, type-consistency `:456-459`) and the MMA atoms are built generically from `q_dtype/k_dtype/v_dtype` (`:466-484`), so nothing in the tile *math* is 16-bit-only — but three fp8 prerequisites are absent:

1. **Descale plumbing** — the single hard gate:

   ```python
   211          assert descale_tensors is None, (
   212              "SM100 forward with head_dim=256 does not support descale_tensors"
   213          )
   ```

   vLLM **always** supplies descales for FP8 KV: the attention layer quantizes Q (`model_executor/layers/attention/attention.py:463-482` setup, `:505-517` forward — `supports_quant_query_input` is True for FA, `fa_utils.py:311-312`), the impl builds `(batch, num_kv_heads)` fp32 descale tensors from `layer._q_scale/_k_scale/_v_scale` (`flash_attn.py:1272-1280`), and `flash_attn_interface.py:430-433` keeps them whenever `v.dtype` is fp8 → passed at `:461-463`. So an fp8+hd256 call dies at this assert on **SM100 too** (after passing the `interface.py:909-910` arch assert, which SM100 satisfies).
2. **FP8 P-precision machinery** — the main kernel scales P by 2^max_offset before the e4m3 cast and corrects in the PV epilogue (`flash_fwd_sm100.py:2190` `max_offset = 8 if q_dtype.width == 8 else 0`, `:2198-2201` rescale_threshold bounded by `_LOG2_DTYPE_MAX` `:93-98`, `:2632-2634` `max_offset_scale`, `:2751`, `:2855`). The hd256 kernel has none of this: its P is stored in `q_dtype` (`:1825, :1855`) with only ex2 emulation tuning (`:1822-1850`) — an fp8 P cast there would saturate the top probabilities.
3. **FP8 register/ex2 tuning** — `_FP8_TUNING_CONFIG` has **hd128 keys only** (`flash_fwd_sm100.py:112-119`; small-hdim regs `:120-123`), and the hd256 kernel selects its tuning with the 16-bit key (`sm100_hd256_2cta_fmha_forward.py:155-156`).

Also absent (not needed for vLLM serving): fused FP8 *output* (`output_scale`) — explicitly gated off for the hd256 kernel (`interface.py:1544-1548`, TODO at `:1718-1722`); vLLM rejects `output_scale` for the FA backend anyway (`flash_attn.py:1204-1207`).

Conclusion: **the hd256 tile itself exists and is tuned for bf16 on sm_100 and sm_110; FP8 is the only gap** (descales + P-scaling + tuning). No new tile variant, no TMEM re-budget (accumulators stay fp32; 512/512 cols in both dtypes; SMEM staging is width-aware and *frees* space at 8-bit KV).

### 2.3 The `fp8_kv_dequant` path (`interface.py:902-903`) — why SM90-only; does an sm_110 extension help?

Gate and feature wiring:

```python
 902      if fp8_kv_dequant:
 903          assert arch // 10 == 9, "fp8_kv_dequant is an SM90-only forward (compute capability 9.x)"
 904          # Compute is fp16 ... the fp8->fp16 cvt is the only single-instruction widening on SM90 ...
...
1541                 fp8_kv_dequant=fp8_kv_dequant,        # passed to FlashAttentionForwardSm90 ONLY
```

- It is a **Hopper-only feature**: only the SM90 kernel accepts the flag (`interface.py:1516-1542`); the SM80/SM100/SM110/SM120 constructors don't take it. On Hopper the QMMA path can't do fp8×fp8→fp32 the way Blackwell's `tcgen05.mma kind::f8f6f4` does, so fp8 K/V must be dequantized to fp16 in-kernel (the single-instruction cvt, `:904-907`), with fp16 compute and forced RS-mode PV (`:959-962`).
- Its shape limits are degenerate for hd256: TMA-only producer ⇒ `page_size == tile_n` (`:966-969`) and on SM90 hd256 gets `tile_n = 80` (`_tile_size_fwd_sm90`, `interface.py:172-174`) → 80-token pages — a layout vLLM never builds (FA advertises `MultipleOf(16)` or, for the live hd256 path, exactly 128, `flash_attn.py:313-318`); plus `head_dim == head_dim_v` (`:971-974`).
- **Extending it to sm_110 is the wrong extension**: on sm_110 the dispatched kernel is the SM100-family tcgen05 kernel (`interface.py:1543-1636`), not `FlashAttentionForwardSm90` (dispatched only for `arch // 10 == 9`). Porting a Hopper WGMMA dequant kernel to Blackwell buys nothing — the SM100/110 kernel **already has native fp8 MMA** (proven on-device in [fa4-fp8kv-sm110.md](fa4-fp8kv-sm110.md) §5.4: `tcgen05.mma kind::f8f6f4` PTX for sm_110a). The useful move is native fp8 (fp8 Q + fp8 K/V + descales) in the hd256 kernel — or, equivalently complex, a dequant-to-bf16 mode of that kernel. Either way it is the same work item as §6.

### 2.4 Which kernel feature the vLLM FP8 path needs for the target's hd256 layers

vLLM's fp8 KV path is **full FP8 with per-(batch, kv_head) descales**: Q quantized in the layer (`attention.py:516-517`), KV stored e4m3, O bf16 (`interface.py:827`), descales `(batch, num_kv_heads)` fp32 (`flash_attn.py:1272-1280`), applied inside the kernel as `qk_descale` folded into the softmax scale (`flash_fwd_sm100.py:2187-2196`) and `v_descale` folded into the PV correction (`:2626-2634, :2751, :2855`) — "FA3 descale semantics" (`:2005`). So the exact kernel feature required for hd256+FP8 is: **`DescaleTensors` support in `BlackwellFusedMultiHeadAttentionForward`** (load + both folds) + the fp8 P-scaling constants + fp8 tuning. `flash_attn_supports_quant_query_input` (`fa_utils.py:311-312`) needs no change (it already returns True, and the Q-quant already fires).

---

## 3. What FlashInfer Does with hd256 FP8 on sm_110 Today (Baseline)

The 16 hd256 full-attn layers in every current champion run on **FlashInfer with fp8 KV as a first-class dtype** (`v1/attention/backends/flashinfer.py`):

- `supported_kv_cache_dtypes` includes `fp8 / fp8_e4m3 / fp8_e5m2` (`:414-420`); `get_dtype_for_flashinfer` maps them to storage dtype (`:484-492`); `supports_kv_cache_dtype` admits them (`:495-502`, nvfp4 separately gated at `:496`).
- Kernels: the FI-native paged wrappers — `BatchPrefillWithPagedKVCacheWrapper` (prefill/extend; wrapper build `:295-334`, field `:558`) and `BatchDecodeWithPagedKVCacheWrapper` (decode, `:565`), planned per-step with `paged_kv_indptr/indices/last_page_len` at block 16, GQA 24:4 — the exact mapping already established in the champion configs (NVFP4_12). A Triton trtllm prefill variant with in-kernel kvfp8 dequant exists as an alternate prefill path (`_trtllm_prefill_attn_kvfp8_dequant`, `:159-263`).

This is the performance/behavior baseline any FA4+FP8+hd256 must beat on Thor.

---

## 4. FA3 Reference (hd256 + fp8 in the FA family)

- **FA3 is a prebuilt `sm_90a`-only `.so`** in this tree (`_vllm_fa3_C.abi3.so`; availability gate `_is_fa3_supported`, `vllm_flash_attn/flash_attn_interface.py:62-69` — fam90 only). It cannot run on sm_100/sm_110, and policy bans it there regardless (`fa_utils.py:120-125`: major ≥ 10 + version 3 → forced to 4-or-2).
- **Policy-wise, hd256 + fp8 is a known-supported FA shape on Hopper**: the fp8-KV gate admits FA3 on fam90 (`fa_utils.py:306`), the hd256 fallback branch applies only to FA4 (`:178`), and `supports_head_size` allows ≤256 for all versions (`flash_attn.py:374-381`). A Qwen3.8-27B serve on an H100 with fp8 KV would therefore land its 16 hd256 layers on FA3+fp8. (Upstream FA3 is the first FA version with fp8 at all — FA2 has no fp8 mode in this tree: the fp8 gate is false for fa2, and the FA2 call explicitly rejects descales, `flash_attn_interface.py:328-338`.)
- So: **hd256+fp8 first shipped in FA3 (Hopper, 16-bit compute, fp8 storage); FA4 brought hd256 to Blackwell in bf16 only** — its fp8 path (`interface.py:909-910`) is deliberately restricted to the hd≤128 main kernel, while the dedicated hd256 kernel carries no fp8/descale mode at all.

---

## 5. Verdict Matrix (hd256 × KV dtype × FA version × arch)

Legend: **OK** = serves; **DG** = vLLM silently downgrades FA4→FA2 (logged); **RJ** = startup rejection (`platforms/cuda.py:457-477` / `flash_attn.py:1122-1136`); **NR** = not runnable in this build (wrong-arch prebuilt binary or policy-banned). FA2 prebuilt = sm_80 cubin only; FA3 prebuilt = sm_90a only ([fa4-fp8kv-sm110.md](fa4-fp8kv-sm110.md) §5.1).

| arch   | KV    | FA2                              | FA3                                   | FA4 |
|--------|-------|----------------------------------|---------------------------------------|-----|
| sm_100 | bf16  | NR (sm_80 prebuilt)              | NR (sm_90a prebuilt + ban `fa_utils.py:120-125`) | **OK** (default, `fa_utils.py:102-104`; block forced 128 via `flash_attn.py:295-318`; kernel `sm100_hd256_2cta_fmha_forward.py`) |
| sm_100 | fp8   | RJ (gate `fa_utils.py:306-308` false for fa2) | NR (ban `:120-125`) | **DG→RJ**: `fa_utils.py:178-191` via `:252-253` → gate `:306-308` → `flash_attn.py:434`; kernel *would also die*: `sm100_hd256_2cta_fmha_forward.py:211-213` |
| sm_110 | bf16  | NR (sm_80 prebuilt; also the *default* — `fa_utils.py:105-107`) | NR (sm_90a prebuilt + ban `:120-125`) | **OK\*** with `flash_attn_version=4`; no fallback reason (block self-forced 128); JIT: `interface.py:1012, 1588-1609`, no arch assert on the bf16 path; *unverified on-device* |
| sm_110 | fp8   | RJ (same gate) | NR | **DG→RJ**: same `:252-253`; gate `:306-308` false for fam110 *even without* the downgrade; kernel-level: `interface.py:909-910` then `sm100_hd256_2cta_fmha_forward.py:211-213` |

\* sm_110 + bf16 + hd256 + FA4 is the one *newly identified viable* combo of this survey: policy-legal today, kernel-plausible (JIT per-arch, same 512/512 TMEM budget, no assert), but never probed on Thor (the fa4-fp8kv probes all used hd128). It is not useful for this deployment (bf16 KV defeats the purpose), but it proves the hd256 tile family runs on sm_110.

Cross-checks embedded in the matrix: FA2/FA3 are dead on **both** Blackwell arches (prebuilt arch), so FA4 is the only FA candidate on Thor *and* B200; and the fp8+hd256 rejection is **identical on sm_100** — the downgrade is not a Thor quirk, it is the kernel capability mirror (§2.2).

---

## 6. What It Would Take to Enable FA4 + FP8-KV + hd256 on sm_110

### 6.1 Kernel changes (CuTe source, `vllm_flash_attn/cute/`)

1. **`sm100_hd256_2cta_fmha_forward.py` — descale plumbing** (the gate): remove `:211-213` assert; accept `DescaleTensors`; per-tile per-(batch, kv_head) load and the two folds, in the exact pattern of the sibling kernel: `_load_effective_descales` (`flash_fwd_sm100.py:2011-2027`), softmax-warp fold into `softmax_scale(_log2)` (`:2187-2196`), correction-warp `v_descale`/`max_offset_scale` fold (`:2626-2634, :2751, :2855`).
2. **`sm100_hd256_2cta_fmha_forward.py` — fp8 P-precision machinery**: the `max_offset=8` / `rescale_threshold` / `max_offset_scale` + `_LOG2_DTYPE_MAX` bound (`flash_fwd_sm100.py:93-98, 2190-2201, 2632-2634, 2751, 2855`) around the P→e4m3 cast (hd256 kernel `:1825, :1855`) — without it the top probabilities saturate in e4m3.
3. **`flash_fwd_sm100.py` — FP8 tuning**: add hd256 keys to `_FP8_TUNING_CONFIG` (`:112-119`, currently hd128-only) and re-tune ex2/regs for the hd256 kernel's warp layout (its `ex2_emu_freq/res/start_frg` + reg split, `sm100_hd256_2cta_fmha_forward.py:155-162`) on B200 *and* Thor.
4. **`interface.py:909-910`** — arch assert `arch // 10 in (10, 11)` (already part of the [thor-fa4-fp8kv-sm110](../../docker/vllm-thor/patches/thor-fa4-fp8kv-sm110.patch) patch).
5. **Not required**: no new tile (128×128 2CTA tile already exists and is tuned for 16-bit on both families), no TMEM re-budget (fp32 accumulators; 512/512 identical for fp8 and bf16 — `sm100_hd256_2cta_fmha_forward.py:133, 150-152`; `flash_fwd_sm100.py:347-353`), no SMEM changes (stage math is width-aware, `flash_fwd_sm100.py:395-404` — fp8 K/V *frees* SMEM), no SplitKV/pack_gqa work (the hd256 kernel is single-split by design, `:73`, `flash_attn.py:1722-1723`), and no `output_scale` work (vLLM never uses it on FA, `flash_attn.py:1204-1207`).

### 6.2 vLLM gate changes afterwards

1. **`fa_utils.py:252-253`** — stop the quantized-KV fallback for the dedicated hd256 kernel once it accepts descales (e.g. gate the reason behind a kernel-capability flag instead of an unconditional `is_quantized_kv_cache` check). Without this the kernel work is unreachable — this is the *first* gate hit, before `:306-308`.
2. **`fa_utils.py:306-308`** — widen fam(100) → fam(110) (already in the companion patch).
3. Keep `:254-256` (128-token page) — the kernel genuinely requires `page_size == tile_n == 128` (`sm100_hd256_2cta_fmha_forward.py:69-71`), and `_get_fa4_hd256_block_size` (`flash_attn.py:295-311`) already pins the layout once the version resolves to 4.
4. `interface.py:1544-1548` fused-FP8-output guard — irrelevant for vLLM serving; leave.

### 6.3 Is the missing piece already present for hd256 in some dtype?

**Yes.** The hd256 tile is not green-field: it exists, is warp-specialized 2CTA, tuned (regs/ex2), and dispatchable on *both* sm_100 and sm_110 for bf16/fp16. The only absent things are the **fp8 dtype mode** (descale folds + P-saturation scaling) and its **tuning entries** — both have exact, small, in-repo reference implementations in the sibling kernel.

### 6.4 Effort class

**Weeks** (focused 1–3 weeks for someone fluent in this kernel family): the code delta is localized and mostly transcription from `flash_fwd_sm100.py`, but it is new JIT/PTX verification of a 2CTA warp-specialized kernel with fp8 operands on sm_110, B200/Thor tuning, fp8-numerics validation (e4m3 P-saturation behavior at hd256), and an e2e A/B against the FlashInfer baseline (§3). Not *days* (tuning + bring-up + verification dominate), and not *a new kernel variant* (no new tile; a new dtype mode + plumbing on an existing one).

---

## 7. Source Index (all anchors, pristine vLLM 0.29.1 package root)

| File | Key lines |
|------|-----------|
| `v1/attention/backends/fa_utils.py` | :14 `FA4_HD256_PAGE_SIZE=128`; :74-221 `get_flash_attn_version` (defaults :99-107, config override :109-117, FA3 ban on Blackwell :120-125, batch-invariant :171-176, **hd256 branch :178-191**, TMEM-limit check :193-209); :224-233 `uses_fa4_hd256_kernel`; :236-268 `_fa4_hd256_fallback_reason` (sinks :244-245, softcap :246-251, **quantized KV :252-253**, block size :254-256, mm_prefix/R-SWA :257-261, DCP :262-267); :282-308 `flash_attn_supports_kv_cache_dtype` (family gate :306-308); :311-312 `flash_attn_supports_quant_query_input` |
| `utils/torch_utils.py` | :78-83 `is_quantized_kv_cache` (`fp8*` / `*per_token_head` / `nvfp4*`) |
| `v1/attention/backends/flash_attn.py` | :159-201 warmup (hd256 page :171-178); :295-311 `_get_fa4_hd256_block_size`; :313-318 `get_supported_kernel_block_sizes`; :322-328 `get_preferred_block_size`; :374-381 `supports_head_size`; :408-450 `supports_combination` (fp8 gate :422-434, error string :434); :562-566 `_cudagraph_support`; :1097-1114 impl version + `fa4_hd256`; :1122-1136 construction-time `NotImplementedError`; :1204-1207 `output_scale` rejection; :1259-1262 fp8 KV view; :1272-1280 descale construction; :1380-1386 hd256 page-alignment; :1722-1723 `num_splits=1` for hd256 |
| `vllm_flash_attn/flash_attn_interface.py` | :62-69 `_is_fa3_supported` (fam90); :72-86 `_is_fa4_supported` (fam 90/100/110); :89-97 `is_fa_version_supported`; :328-338 FA2 descale rejection; :418-465 FA4 dispatch (:430-433 descale clearing, :461-463 descale passing) |
| `vllm_flash_attn/cute/interface.py` | :61-63 hd256 kernel import; :112-129 `_validate_head_dims` (sm100/110 range :125-129); :436-445 fwd config (hd256→no SplitKV :438-442); :736-747 `fp8_kv_dequant` requirements; :811 `is_fp8`; :902-908 `fp8_kv_dequant` SM90 assert (:903); :909-910 `is_fp8` SM100 assert; :959-974 `fp8_kv_dequant` page_size/hd limits; :995-1009 2CTA heuristic; :1011-1013 `use_dedicated_hd256_kernel`; :1014-1044 hd256 flags; :1174-1176 hd256 no dynamic scheduler; :1544-1548 output_scale guard (incl. hd256); :1566-1609 hd256 dispatch (feature asserts :1568-1572, paged asserts :1573-1584); :1610-1636 main kernel dispatch; :1708-1709 descale compile args (arch 10/11); :1718-1722 hd256 `output_scale` TODO |
| `vllm_flash_attn/cute/flash_fwd_sm100.py` (main kernel) | :93-98 `_LOG2_DTYPE_MAX`; :100-111 `_TUNING_CONFIG` (hd256 entries :109-110); :112-119 `_FP8_TUNING_CONFIG` (hd128 only); :120-123 `_FP8_SMALL_HDIM_REGS`; :127-133 `DescaleTensors`; :193-195 arch assert (sm_100f/sm_110f); :310 `tmem_alloc_cols`; :347-353 TMEM layout + assert; :362-381 register budget; :395-404 SMEM stages (width-aware); :446 `descale_tensors` param; :507-510 dtype consistency; :511-528 fp8 register override; :2011-2027 `_load_effective_descales`; :2187-2196 softmax fold (:2190 `max_offset`, :2198-2201 threshold); :2626-2634 correction fold; :2751, :2855 `max_offset` corrections |
| `vllm_flash_attn/cute/sm100_hd256_2cta_fmha_forward.py` (dedicated hd256) | :64-82 shape asserts ((256,256) :65-67, TMA-paged-only :69-71, no pack_gqa :72, no SplitKV :73, 128×128 :80-82); :86-92 tiler; :133 TMEM max; :150-152 TMEM offsets (512/512); :155-162 tuning (16-bit hd256 key); :166-171 pipeline stages; :198-213 feature asserts (**descale :211-213**); :397-401 dtype from tensor; :456-459 q/k/v dtype equality; :466-484 MMA atoms; :1825/:1855 P in `q_dtype`; :1822-1850 ex2 emulation |
| `model_executor/layers/attention/attention.py` | :463-482 `query_quant` setup (fp8/nvfp4 KV); :505-517 Q quantization in forward (:516-517) |
| `platforms/cuda.py` | :457-477 forced-backend validation (ValueError :466-474); :479-503 auto-select + no-valid-backend ValueError |
| `v1/attention/backends/flashinfer.py` | :14-15 wrappers; :159-263 trtllm kvfp8-dequant Triton; :295-334 prefill wrapper; :414-420 `supported_kv_cache_dtypes`; :484-492 `get_dtype_for_flashinfer`; :495-502 `supports_kv_cache_dtype`; :558/:565 prefill/decode wrapper fields |
