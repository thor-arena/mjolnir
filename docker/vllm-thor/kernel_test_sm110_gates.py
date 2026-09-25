#!/usr/bin/env python3
"""Thor sm_110 gate-probe canary: T6 (GDN prefill), T9 (FA4 FP8-KV),
T10 (draft-CG gate decision, pure python — runs without a GPU),
T11 (FA4 hd256 FP8-KV — the dedicated head_dim=256 kernel gate).

A single manual GPU test for the Thor sm_110 gate probes (patches
`thor-gdn-prefill-sm110`, `thor-fa4-fp8kv-sm110`,
`thor-fa4-hd256-fp8-sm110`). It imports the installed (patched) vllm and
triggers each probe's cached path with a small synthetic input — it does
NOT spin up a server. T9 additionally asserts the passive-by-default
invariant: with no explicit FA4 attention-config, the FP8-KV gate stays
closed and auto backend selection still lands on FlashInfer on sm_110.

Each section reports one of:

  PASS      the probe found working kernel support, and the consuming
            gate / kernel selection agrees with the verdict;
  INERT-OK  the probe correctly resolved to "disabled" and the legacy
            path is intact. This is the *expected* state on Thor today
            for the FA4 probe (passive by default — inert until the
            config selects FA4; the GDN probe now PASSes there) — a
            disabled probe is not a failure;
  SKIP      no CUDA device visible (self-skip, exit 0);
  FAIL      an internal inconsistency (verdict disagrees with its gate or
            selection) or an unexpected exception — the only case that
            makes the script exit non-zero.

Run it on a Thor (or any CUDA box) against the patched image:

  docker run --rm --gpus all --entrypoint python3 \
    -v <repo>/docker/vllm-thor/kernel_test_sm110_gates.py:\
/tmp/kernel_test_sm110_gates.py:ro \
    mjolnir/vllm-thor:qwen38-sm110-v6 /tmp/kernel_test_sm110_gates.py
"""
from __future__ import annotations

import logging
import sys
import types
import traceback

import torch


def _t6_gdn_prefill() -> None:
    """T6 — GDN prefill probe verdict + correctness A/B + resolved backend."""
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        _resolve_gdn_prefill_backend,
        _thor_gdn_prefill_probe,
        _thor_gdn_prefill_probe_run,
    )
    from vllm.platforms import current_platform

    # The Qwen3.8-27B GDN geometry: head_k_dim 128 (the arch clause value).
    cfg = types.SimpleNamespace(
        additional_config={},
        model_config=types.SimpleNamespace(
            hf_text_config=types.SimpleNamespace(linear_key_head_dim=128)),
    )

    # First call performs the one-shot FI-vs-FLA A/B; later calls are a
    # cached bool read.
    verdict = _thor_gdn_prefill_probe()

    # Correctness A/B: when the probe passed, re-run the raw A/B comparison
    # (deterministic, seeded) — a second run must agree.
    if verdict:
        assert _thor_gdn_prefill_probe_run(), (
            "GDN prefill A/B re-run disagrees with the probe verdict"
        )
        print("T6 A/B re-check: FI vs FLA output+state still match")

    resolved = _resolve_gdn_prefill_backend(cfg)
    arch_eligible = (
        current_platform.is_device_capability_family(110)
        and current_platform.get_cuda_runtime_major() >= 13
    )
    if arch_eligible:
        # On sm_110 the grant is exactly "probe passed" (head_k_dim 128 is
        # set in cfg above).
        expected = "flashinfer" if verdict else "triton"
        assert resolved == ("auto", expected), (
            f"GDN prefill resolution {resolved} != expected "
            f"('auto', {expected!r}) for probe verdict {verdict}"
        )
    if verdict:
        print(f"PASS T6 GDN prefill: FI kernel verified on this device; "
              f"resolved {resolved}")
    else:
        print(f"INERT-OK T6 GDN prefill: probe disabled (expected on "
              f"Thor today); resolved {resolved} (Triton path intact)")


def _t9_fa4_fp8kv() -> None:
    """T9 — FA4 FP8-KV probe verdict + gate consistency + passive-by-default guard."""
    import vllm.v1.attention.backends.fa_utils as FA
    from vllm.config import AttentionConfig, VllmConfig, set_current_vllm_config
    from vllm.platforms import current_platform
    from vllm.v1.attention.backends.registry import AttentionBackendEnum
    from vllm.v1.attention.selector import get_attn_backend

    family110 = current_platform.is_device_capability_family(110)

    # PASSIVE-BY-DEFAULT GUARD (runs first, before anything may have
    # triggered the probe): under the DEFAULT config there is no
    # flash_attn_version override, so on sm_110 the FA version resolves to
    # 2 and the probe-gated OR-term of the gate short-circuits on
    # `fa_version == 4`. The probe must never run, the FP8-KV gate must
    # stay closed, and auto backend selection must still land on
    # FlashInfer — the patch alone changes zero behavior.
    with set_current_vllm_config(VllmConfig()):
        assert FA._THOR_FA4_FP8KV_PROBE_RESULT is None, (
            "the FA4 FP8-KV probe ran although no explicit FA4 "
            "attention-config is set — passive-by-default is broken"
        )
        assert not FA.flash_attn_supports_kv_cache_dtype("fp8_e4m3"), (
            "FP8-KV gate open under the DEFAULT config — passive-by-"
            "default is broken"
        )
        default_backend = get_attn_backend(
            head_size=128,
            dtype=torch.bfloat16,
            kv_cache_dtype="fp8_e4m3",
            num_heads=8,
        )
        assert default_backend.__name__ == "FlashInferBackend", (
            f"auto backend selection with the default config chose "
            f"{default_backend.__name__}, expected FlashInferBackend — "
            f"passive-by-default regression"
        )
    print(f"T9 passive-by-default: default config leaves the probe "
          f"untriggered, FP8-KV gate closed, auto-selection -> "
          f"{default_backend.__name__}")

    # Probe verdict + gate consistency under the explicit FA4 config
    # (the required opt-in: backend=FLASH_ATTN + flash_attn_version=4).
    verdict = FA._thor_fa4_fp8kv_probe()
    with set_current_vllm_config(
        VllmConfig(
            attention_config=AttentionConfig(
                backend=AttentionBackendEnum.FLASH_ATTN,
                flash_attn_version=4,
            )
        )
    ):
        gate = FA.flash_attn_supports_kv_cache_dtype("fp8_e4m3")
    assert gate == verdict, (
        f"flash_attn_supports_kv_cache_dtype (explicit FA4 config) = "
        f"{gate} disagrees with the probe verdict {verdict}"
    )

    if family110:
        if verdict:
            print(f"PASS T9 FA4 FP8-KV: probe enabled on sm_110; gate "
                  f"open for the explicit FA4 config; default config "
                  f"stays passive ({default_backend.__name__})")
        else:
            print(f"INERT-OK T9 FA4 FP8-KV: probe disabled on sm_110 "
                  f"(reason in the 'THOR FA4 FP8-KV probe: disabled' log "
                  f"line); gate closed for FA4; default config stays "
                  f"passive ({default_backend.__name__})")
    else:
        assert verdict is False, (
            f"probe must be a no-op off sm_110, got verdict={verdict}"
        )
        assert not gate
        print(f"INERT-OK T9 FA4 FP8-KV: not an sm_110 device; probe is a "
              f"no-op (False); gate {gate}, default selection "
              f"{default_backend.__name__} (unchanged by the patch)")


def _t11_fa4_hd256_fp8kv() -> None:
    """T11 — FA4 hd256 FP8-KV probe verdict + gate consistency + passive guard."""
    import vllm.v1.attention.backends.fa_utils as FA
    from vllm.config import AttentionConfig, CacheConfig, VllmConfig, set_current_vllm_config
    from vllm.platforms import current_platform
    from vllm.v1.attention.backends.registry import AttentionBackendEnum
    from vllm.v1.attention.selector import get_attn_backend

    family110 = current_platform.is_device_capability_family(110)

    # PASSIVE-BY-DEFAULT GUARD (runs first, before anything may have
    # triggered the hd256 probe): under the DEFAULT config there is no
    # flash_attn_version override, so on sm_110 the FA version resolves
    # to 2, the hd256 fallback-reject grant is never reached, and the
    # gate's OR-term short-circuits on `fa_version == 4`. The hd256
    # probe must never run, the FP8-KV gate must stay closed for a
    # head_size=256 + fp8-e4m3 + block-128 request, and auto backend
    # selection must still land on FlashInfer — the patch alone changes
    # zero default behavior.
    with set_current_vllm_config(VllmConfig()):
        assert FA._THOR_FA4_HD256_FP8KV_PROBE_RESULT is None, (
            "the FA4 hd256 FP8-KV probe ran although no explicit FA4 "
            "attention-config is set — passive-by-default is broken"
        )
        assert not FA.flash_attn_supports_kv_cache_dtype(
            "fp8_e4m3",
            head_size=256,
            head_size_v=256,
            kv_cache_block_size=128,
            supports_fa4_hd256=True,
        ), (
            "FP8-KV gate open for the hd256 model under the DEFAULT "
            "config — passive-by-default is broken"
        )
        default_backend = get_attn_backend(
            head_size=256,
            dtype=torch.bfloat16,
            kv_cache_dtype="fp8_e4m3",
            num_heads=24,
        )
        assert default_backend.__name__ == "FlashInferBackend", (
            f"auto backend selection with the default config chose "
            f"{default_backend.__name__}, expected FlashInferBackend — "
            f"passive-by-default regression"
        )
    print(f"T11 passive-by-default: default config leaves the hd256 "
          f"probe untriggered, FP8-KV gate closed for the hd256 model, "
          f"auto-selection -> {default_backend.__name__}")

    # Probe verdict + gate consistency under the explicit FA4 config
    # (the required opt-in: backend=FLASH_ATTN +
    # flash_attn_version=4), an FP8 KV cache, and the hd256 kernel's
    # required block size (128).
    verdict = FA._thor_fa4_hd256_fp8kv_probe()
    fa4_config = VllmConfig(
        attention_config=AttentionConfig(
            backend=AttentionBackendEnum.FLASH_ATTN,
            flash_attn_version=4,
        ),
        cache_config=CacheConfig(cache_dtype="fp8_e4m3"),
    )
    with set_current_vllm_config(fa4_config):
        gate = FA.flash_attn_supports_kv_cache_dtype(
            "fp8_e4m3",
            head_size=256,
            head_size_v=256,
            kv_cache_block_size=128,
            supports_fa4_hd256=True,
        )
        # block_size=128 clears the `% 128` reject, so the only reject
        # that can still fire for this config is the quantized-KV one —
        # which the Thor grant lifts exactly when the probe passed.
        fallback = FA._fa4_hd256_fallback_reason(False, False, 128, fa4_config)
    assert gate == verdict, (
        f"flash_attn_supports_kv_cache_dtype (explicit FA4, hd256) = "
        f"{gate} disagrees with the hd256 probe verdict {verdict}"
    )
    expected_reason = (
        None if verdict else "quantized KV cache dtype fp8_e4m3"
    )
    assert fallback == expected_reason, (
        f"_fa4_hd256_fallback_reason (block_size=128, FP8 KV) = "
        f"{fallback!r} disagrees with the hd256 probe verdict "
        f"{verdict} (expected {expected_reason!r}) — the hd256 FP8 "
        f"path would {'be downgraded to FA2' if fallback else 'be admitted'}"
    )

    if family110:
        if verdict:
            print(f"PASS T11 FA4 hd256 FP8-KV: probe enabled on sm_110; "
                  f"gate open + no FA2 downgrade for the explicit FA4 "
                  f"config; default config stays passive "
                  f"({default_backend.__name__})")
        else:
            print(f"INERT-OK T11 FA4 hd256 FP8-KV: probe disabled on "
                  f"sm_110 (reason in the 'THOR FA4 hd256 FP8-KV probe: "
                  f"disabled' log line); gate closed; the quantized-KV "
                  f"reject is intact ({fallback!r}); default config "
                  f"stays passive ({default_backend.__name__})")
    else:
        assert verdict is False, (
            f"probe must be a no-op off sm_110, got verdict={verdict}"
        )
        assert not gate
        assert fallback == "quantized KV cache dtype fp8_e4m3", (
            f"the quantized-KV reject must stay stock off sm_110, got "
            f"{fallback!r}"
        )
        print(f"INERT-OK T11 FA4 hd256 FP8-KV: not an sm_110 device; "
              f"probe is a no-op (False); gate {gate}; default "
              f"selection {default_backend.__name__} (unchanged by the "
              f"patch)")


def _t10_draft_cg_gate() -> None:
    """T10 — the draft-CG gate's decision logic (pure python, no GPU).

    `_thor_draft_cg_use_cuda_graph` takes the capability as an argument,
    so it is importable and testable without CUDA — the canary asserts
    the four spec cases: sm_110 below/above the threshold, the off-arch
    no-op, and the min_batch=1 knob (= today's behavior).
    """
    from vllm.v1.worker.gpu.spec_decode.dflash.speculator import (
        _thor_draft_cg_use_cuda_graph,
    )

    cases = [
        # (capability, num_reqs, min_batch, expected use_cg)
        ((11, 0), 1, 2, False),  # sm_110, N=1 < min_batch=2 -> eager
        ((11, 0), 2, 2, True),  # sm_110, N=2 >= min_batch=2 -> CG
        ((9, 0), 1, 2, True),  # off-sm_110 arch no-op -> CG
        ((11, 0), 1, 1, True),  # knob: min_batch=1 -> CG for all N
    ]
    for capability, num_reqs, min_batch, expected in cases:
        got = _thor_draft_cg_use_cuda_graph(num_reqs, capability, min_batch)
        assert got is expected, (
            f"_thor_draft_cg_use_cuda_graph(num_reqs={num_reqs}, "
            f"capability={capability}, min_batch={min_batch}) = {got}, "
            f"expected {expected}"
        )
    print("PASS T10 draft-CG gate: pure decision logic (4/4 cases)")


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("vllm").setLevel(logging.INFO)

    failures = 0
    if not torch.cuda.is_available():
        for name in (
            "T6 GDN prefill",
            "T9 FA4 FP8-KV",
            "T11 FA4 hd256 FP8-KV",
        ):
            print(f"SKIP {name}: no CUDA device visible (re-run with "
                  f"--gpus all on a Thor)")
    else:
        print(f"device capability: {torch.cuda.get_device_capability()}")
        for name, section in (
            ("T6 GDN prefill", _t6_gdn_prefill),
            ("T9 FA4 FP8-KV", _t9_fa4_fp8kv),
            ("T11 FA4 hd256 FP8-KV", _t11_fa4_hd256_fp8kv),
        ):
            try:
                section()
            except Exception:
                failures += 1
                print(f"FAIL {name}:", file=sys.stderr)
                traceback.print_exc()

    # T10 is pure python (the helper takes the capability as an argument)
    # — it runs even without a GPU.
    try:
        _t10_draft_cg_gate()
    except Exception:
        failures += 1
        print("FAIL T10 draft-CG gate:", file=sys.stderr)
        traceback.print_exc()

    if failures:
        print(f"kernel test sm110 gates: {failures} section(s) FAILED")
        return 1
    print("kernel test sm110 gates: DONE (exit 0)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
