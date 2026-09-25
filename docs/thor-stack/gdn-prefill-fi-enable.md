# GDN Prefill: FlashInfer Enablement on sm_110

> **Status:** KEEP — in the 13-patch stack (`thor-gdn-prefill-fi-sm110`) · **Date:** 2026-09-21 · **Scope:** the one-line FlashInfer-side allowlist widening that makes the GDN prefill CuTe-DSL kernel reachable on sm_110, plus the dual-root (vLLM + FlashInfer) patch machinery it introduced

**Stack:** vllm-thor 12-patch v7 base + 1 new flashinfer-side hunk (13 patches).
**Image tested:** `mjolnir/vllm-thor:qwen38-sm110-v8-gdnfi` (built from the pinned base `vllm/vllm-openai@sha256:c27ab1587…`, vllm 0.29.1rc1.dev347+gdee37d891, flashinfer 0.6.18.post1, torch 2.13.0+cu130).
**Host:** Jetson Thor sm_110a, 20 SMs.

Question: can FlashInfer itself be patched to enable the GDN prefill DSL kernel on sm_110?
**Answer: yes — the wall at `gdn_prefill.py:576` is an FA4-type Python allowlist, and the kernel JIT-compiles for sm_110a and passes the probe A/B. Live T6: PASS; the FlashInfer GDN prefill kernel is 3.9–5.2× faster than the Triton/FLA reference. KEEP in the stack.**

---

## TL;DR

- The dispatch wall is an if/elif chain on arch major — major 11 (Thor) simply falls into the `else: raise NotImplementedError` branch. It is a Python allowlist, not an arch-targeted kernel wall.
- The kernel behind the SM100 branch is arch-generic in the FA4 sense (device-JIT via CuTe-DSL, no explicit `GPUArch`); its only sm_100-specific construct is a TMEM sizing-constant lookup.
- The fix is one line: `if _arch_major == 10:` → `if _arch_major in (10, 11):  # THOR_SM110_GDN_ALLOWLIST`. Inert on every other device.
- The stack's patch machinery gained a second patch root (the flashinfer site-packages): `FLASHINFER_PATCH_ORDER` in `apply_patches.py`, `FLASHINFER_FIXES` in `verify_patches.py`; the Dockerfile is unchanged.
- Live canary: T6 PASS (FI vs Triton/FLA A/B within tolerances), T7/T8 remain expected inert walls, T9/T10 unchanged. Chunk-prefill speedup 3.94–5.24× vs Triton/FLA.

---

## 1. Wall Characterization — FA4-type (fixable), not B12x-type

### 1.1 The check that raises (flashinfer 0.6.18.post1, in-image at `/usr/local/lib/python3.12/dist-packages/flashinfer/`)

`gdn_prefill.py` — `chunk_gated_delta_rule`'s non-CP dispatch is an if/elif chain on the device's arch major, **not** a single allowlist expression:

| Line | Code |
|---|---|
| `gdn_prefill.py:460` | `if _arch_major == 10:` → SM100 CuTe-DSL branch (`chunk_gated_delta_rule_sm100`) |
| `gdn_prefill.py:519` | `elif _arch_major == 12:` → SM120 delta-rule DSL branch |
| `gdn_prefill.py:546` | `elif _arch_major == 9:` → SM90 delta-rule DSL branch |
| `gdn_prefill.py:575-576` | `else:` → **`raise NotImplementedError("GDN prefill DSL kernel is unavailable")`** ← the v7 probe wall |

major 11 (Thor) falls into the `else`. This is the *only* consult site of the non-CP GDN-prefill allowlist in the package (package-wide grep; the trace template `trace/templates/gdn.py` is inert). Two sibling sites were examined and deliberately **not** touched:

- `gdn_prefill.py:359` — CP heuristic `cp_heuristic_matches = _arch_major in (9, 10, 12)`. The CP kernels are arch-pinned, not device-JIT: `gdn_kernels/blackwell/gdn_cp_prefill.py:38-42` hard-rejects `major != 10` (`"SM100 CP delta rule requires a compute 10.x device"`), and the SM90/SM120 delta-rule DSL kernels pin `cute.GPUArch("sm_90a")` / `sm_12xa` (`delta_rule_dsl/delta_rule_sm90.py:2465`, `delta_rule_cp_sm90.py:49`, `delta_rule_dsl/custom_compile_cache.py:29`). Widening 11 into the CP heuristic would make Thor *crash* in the CP pre-compile; left as-is, sm_110 simply never routes to CP (`use_cp="auto"` is the vLLM path) and falls through to the non-CP branch.
- `gdn_prefill.py:371` — `state_indices` is SM100-only by API contract; vLLM's production wrapper and the probe never pass it. Unchanged.

### 1.2 Is the kernel behind the SM100 branch arch-generic or sm_120/sm_100-targeted?

**Arch-generic in the FA4 sense (device-JIT); one sm_100 sizing constant.**

The SM100 branch calls `chunk_gated_delta_rule_sm100` (`gdn_kernels/blackwell/gdn_prefill.py`), which wraps `GatedDeltaNetChunkedKernel` (`gdn_kernels/blackwell/gated_delta_net_chunked.py`, 4755 lines) and compiles it at `gdn_prefill.py:293`:

```python
compiled = cute.compile(gdn, …, options="--enable-tvm-ffi --opt-level 3")   # no GPUArch option
```

CuTe-DSL with no explicit `cute.GPUArch` targets the **current device** (overridable via `CUTE_DSL_ARCH`, `base_dsl/cache_helpers.py:71`) — exactly the FA4 pattern that JIT-compiled for sm_110a and passed the live probe in [fa4-fp8kv-sm110.md](fa4-fp8kv-sm110.md) (maxerr 0.01965).

The only sm_100-specific construct in the kernel is:

| File:line | Construct | Role |
|---|---|---|
| `gated_delta_net_chunked.py:193` | class attr `arch = "sm_100"` | — |
| `gated_delta_net_chunked.py:280` | `self.tmem_alloc_cols = cute.arch.get_max_tmem_alloc_cols(self.arch)` | **hardware-constant lookup only** (returns 512 TMEM columns) |

The kernel's heavy hardware dependence is `tcgen05` (5th-gen tensor-core MMA) + TMEM (L98 `from cutlass.cute.nvgpu import cpasync, tcgen05`, L562 `tcgen05.MmaF16BF16Op`, TMEM staging throughout). That is SM100+ *datacenter-class* Blackwell silicon — which Thor's sm_110a **has** (it is not consumer SM12x, which lacks tcgen05/TMEM). Independent in-image evidence that sm_110a is a first-class tcgen05/DSL target in this very flashinfer release:

- v7 serve log L39: `DSL_FMHA_ARCHS: ('sm_100a', 'sm_103a', 'sm_107a', 'sm_110a')` — flashinfer's own DSL-FMHA allowlist includes sm_110a (those FMHA kernels are tcgen05-based).
- `fused_moe/api.py:292`: `_CUTLASS_BF16_ARCHS = (89, 90, 100, 103, 107, 110, 120, 121)` — 110 already allowlisted there.
- FA4 precedent: the tcgen05-based FA4 CuTe-DSL kernel JIT-compiled and passed live on the Thor host after its Python assert was widened ([fa4-fp8kv-sm110.md](fa4-fp8kv-sm110.md); [thor-fa4-fp8kv-sm110.patch](../../docker/vllm-thor/patches/thor-fa4-fp8kv-sm110.patch)).

Contrast with the B12x wall (unfixable archetype): the B12x kernel *source* targets `sm_120a/sm_120f/sm_121a/sm_121f` exclusively — its JIT raises `DSLUserCodeError: expects arch to be one of [sm_120a, sm_120f, sm_121a, sm_121f], but got sm_110a` at `gemm/kernels/dense_blockscaled_gemm_sm120_b12x.py:320` (v7 log L117-126). No allowlist widening reaches it. The GDN SM100 kernel has **no such target list** — its JIT targets the device, so a one-line allowlist widening is sufficient (and the probe remains the runtime gate, as designed).

---

## 2. Patch Contents + Mechanism

### 2.1 FlashInfer side — `patches/thor-gdn-prefill-fi-sm110.patch` (new; 1 line)

```diff
--- a/flashinfer/gdn_prefill.py
+++ b/flashinfer/gdn_prefill.py
@@ -457,7 +457,7 @@
         if output_final_state:
             return output, output_state
         return output
-    if _arch_major == 10:
+    if _arch_major in (10, 11):  # THOR_SM110_GDN_ALLOWLIST
        if _cuda_major < 13:
            raise NotImplementedError(
                "Blackwell GDN prefill is only supported on CUDA 13+"
```

The widened SM100 branch keeps its own guards (`CUDA ≥ 13`, `head_size == 128` assert at `gdn_prefill.py:469`, kernel-present check). On any non-sm_110 device this hunk is a no-op (the branch condition only *gains* major 11).

### 2.2 Mechanism — second patch root for the flashinfer site-packages

`apply_patches.py` previously patched only the installed **vllm** package; the GDN allowlist lives in the separate **flashinfer** site-packages. Changes (conventions mirrored: sentinel + `--forward` + authoritative verify):

- `apply_patches.py`: new `FLASHINFER_PATCH_ORDER = ["thor-gdn-prefill-fi-sm110.patch"]`; new `flashinfer_package_root(explicit)` (importlib-resolved, like `package_root`); `main()` applies the flashinfer patches with the existing generic `apply_one` (`patch -p1 --forward`, patch paths are `a/flashinfer/…`), then verifies **both** roots. New optional arg 2 = explicit flashinfer root (arg 1 = explicit vllm root), matching the existing explicit-root semantics.
- `verify_patches.py`: new `FLASHINFER_FIXES` table — fix `"thor sm110 GDN prefill: FI non-CP dispatch allowlist widened to major 11"` checked by sentinel `if _arch_major in (10, 11):  # THOR_SM110_GDN_ALLOWLIST` in `gdn_prefill.py` (the sentinel rides on the changed line — for a one-line hunk, present ⇔ hunk applied). `check_root` factored into `_check_root(root, fixes)`; new `verify_flashinfer(root)` / `check_flashinfer_root`; `__main__` verifies both roots (optional arg 2).
- `Dockerfile`: **unchanged** — it already runs `apply_patches.py && verify_patches.py` against the installed packages, and both now cover flashinfer.
- vLLM side: **patch A unchanged** (`thor-gdn-prefill-sm110` keeps the probe-gated `family(110) and head_k_dim==128 and cudaRT>=13 and _thor_gdn_prefill_probe()` grant; probe tolerances untouched). The widened allowlist makes the probe *executable* on Thor; the probe remains the runtime gate.
- `PATCHES.md`: 12→13 patches; GDN row "May activate" → "Enablement (probe-gated, live-verified)"; new section "Thor sm_110 GDN prefill enablement (flashinfer side)".

---

## 3. No-GPU Gate (fresh trees from the pinned base digest)

Fresh gate tree = pristine `vllm` (0.29.1rc1.dev347+gdee37d891) + pristine `flashinfer` (0.6.18.post1) extracted from `vllm/vllm-openai:nightly-aarch64` (RepoDigests[0] == the Dockerfile-pinned `sha256:c27ab1587…`), all runs in a no-GPU container of that same base image:

| Gate | Result |
|---|---|
| `apply_patches.py <vllm root> <flashinfer root>` | **rc 0** — all 12 vllm patches applied (55519's known 20-line offset aside, identical to prior runs) + `flashinfer/gdn_prefill.py` hunk applied; `all 12 fixes verified in <vllm root>` + `all 1 flashinfer fixes verified in <flashinfer root>` |
| `verify_patches.py <vllm root> <flashinfer root>` (standalone) | **rc 0** — `all 12 fixes present` + `all 1 flashinfer fixes present` |
| Idempotency: re-apply on the already-patched tree | **rc 0** — 17 hunks "previously applied" skipped, both verifications still pass (the `--forward` bump-safety semantics now hold for the flashinfer hunk too) |
| `py_compile` patched `flashinfer/gdn_prefill.py` + `qwen_gdn_linear_attn.py` | both OK |
| Import smoke (no GPU): `import flashinfer` + sentinel in source of `flashinfer.gdn_prefill` | OK (flashinfer 0.6.18.post1, vllm 0.29.1rc1.dev347) |
| Canary, strict no-GPU (`CUDA_VISIBLE_DEVICES=""`, PYTHONPATH → patched trees) | T6–T9 **SKIP**, T10 **PASS (4/4)**, `DONE (exit 0)` |
| `docker build` (no `--gpus`) → `mjolnir/vllm-thor:qwen38-sm110-v8-gdnfi` | All Dockerfile gates green (apply+verify, `functional_check_55390_55519.py`, fi-update/dspark kernel tests self-skipped, py_compile+import smoke). Post-build in-image check: `verify()` → 0 missing, `verify_flashinfer()` → 0 missing, hunk present in the installed `flashinfer/gdn_prefill.py` |

---

## 4. Live Test on the Thor (`--gpus all`, image v8-gdnfi)

Full canary `kernel_test_sm110_gates.py` (fresh container; `CUTE_DSL_CACHE_DIR` on a host-mounted volume, initially empty — i.e. the GDN kernel's JIT was cold):

```
INFO 09-21 11:25:18 [qwen_gdn_linear_attn.py:190] THOR GDN prefill probe: FI kernel enabled on sm_110
T6 A/B re-check: FI vs FLA output+state still match
PASS T6 GDN prefill: FI kernel verified on this device; resolved ('auto', 'flashinfer')
INERT-OK T7 B12x NVFP4: probe disabled (expected on Thor today — CuTe-DSL arch-targets sm_120a–sm121f); selected FlashInferCutlassNvFp4LinearKernel, CUTLASS gate True
INERT-OK T8 XQA/TRTLLM decode: probe disabled (expected on Thor today — flashinfer 0.6.18 whitelist is major [9, 10, 12]); decode gate False, prefill gate False
T9 passive-by-default: default config leaves the probe untriggered, FP8-KV gate closed, auto-selection -> FlashInferBackend
PASS T9 FA4 FP8-KV: probe enabled on sm_110; gate open for the explicit FA4 config; default config stays passive (FlashInferBackend)
PASS T10 draft-CG gate: pure decision logic (4/4 cases)
kernel test sm110 gates: DONE (exit 0)
```

**Verdict: PASS.** The FI GDN prefill kernel JIT-compiled for sm_110a (cold, seconds-scale — T6 completed within ~30 s of container start incl. imports) and matched the Triton/FLA reference on the deterministic state-carrying 2-chunk input (output **and** final state within the untouched probe tolerances rtol 2e-2 / atol 1e-3·1e-2). The backend resolution now grants `('auto', 'flashinfer')` — the 48/64 GDN layers' prefill moves to the FI kernel on the v8 build with zero further changes (patch A's design point). T7/T8 remain the expected inert walls; T9/T10 unchanged. No revert needed.

### Perf (FI vs Triton/FLA, exact production wrappers, probe geometry B=1 H=2 K=V=128 bf16, CUDA-event timing, 25 iters after warmup, warm JIT cache)

| Shape | FI (CuTe-DSL, SM100-branch) | Triton/FLA | Speedup |
|---|---|---|---|
| T=256, B=1 | 0.0476 ms/iter | 0.2496 ms/iter | **5.24×** |
| T=1024, B=1 | 0.0738 ms/iter | 0.2907 ms/iter | **3.94×** |

First-call (warm cache) overhead: 6.3 s once per process (TMA-descriptor/workspace setup). Cold-JIT cost per process is seconds-scale (the canary's cold-cache T6 above).

---

## 5. Final Recommendation — KEEP in the Stack (13 patches)

- **Real improvement** (not "may activate"): the #1 kernel gap in the e2e analysis (GDN prefill on 48/64 layers) now runs the FlashInfer CuTe-DSL kernel on Thor at ~4–5× the Triton/FLA rate on chunk prefill shapes. Not subject to the no-improvement removal rule.
- **Risk profile unchanged by design:** the vLLM-side grant is still probe-gated (patch A, tolerances untouched) and the flashinfer hunk alone is behaviorally inert on every device; any probe failure on any future flashinfer bump degrades to the exact v7 (Triton) behavior. The probe now actually *tests the FI kernel* instead of tripping the arch wall, so the gate is now an honest runtime gate.
- **Watch items:** (a) one-time-per-process JIT at startup — the probe compiles its H=2 config, production compiles the model's head config (a few seconds each, once; same class of startup cost FI GDN prefill already has on SM90/SM100 — vLLM even logs a JIT warning for it); (b) the kernel is a tcgen05/TMEM Blackwell design sized for 512 TMEM columns — verified on sm_110a today, but a future flashinfer that retargets the SM100 branch would need re-probing (the probe covers this automatically).
- Next serve build: use `mjolnir/vllm-thor:qwen38-sm110-v8-gdnfi` (or re-tag as the next vN). Recommended: one end-to-end serve check of a long-prompt request on the v8 image to watch the first real GDN prefill through the FI path (belt-and-suspenders over the canary's synthetic shapes).

---

## Files Changed

- `docker/vllm-thor/patches/thor-gdn-prefill-fi-sm110.patch` — new (1-line flashinfer hunk)
- `docker/vllm-thor/apply_patches.py` — `FLASHINFER_PATCH_ORDER` + `flashinfer_package_root` + dual-root verify
- `docker/vllm-thor/verify_patches.py` — `FLASHINFER_FIXES` + `verify_flashinfer` + optional flashinfer root arg
- `docker/vllm-thor/PATCHES.md` — counts, GDN row, new enablement section
- Dockerfile: unchanged · canary: unchanged · patch A: unchanged

Workstream records: the live canary log and the perf script are kept with the workstream artifacts.
