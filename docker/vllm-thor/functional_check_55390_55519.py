"""Functional check for PR #55390 + #55519 (MTP draft-group annotation + warning gate).

Mirrors the PRs' unit tests (which live in tests/v1/core/test_kv_cache_utils.py
and are not shipped in the wheel) by constructing Qwen3.5-shaped hybrid specs
directly. Run inside the image: python3 /functional_check_55390_55519.py
"""
from types import SimpleNamespace

import torch

import vllm.v1.core.kv_cache_utils as kcu
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec


def _mamba() -> MambaSpec:
    return MambaSpec(
        block_size=256,
        shapes=((4, 128), (128, 128)),
        dtypes=(torch.bfloat16, torch.bfloat16),
        mamba_cache_mode="align",
    )


def _full() -> FullAttentionSpec:
    return FullAttentionSpec(
        block_size=256,
        num_kv_heads=4,
        head_size=128,
        dtype=torch.float8_e4m3fn,
    )


def _config(method: str, block_drop: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
        cache_config=SimpleNamespace(
            mamba_cache_mode="align",
            get_resolved_kv_cache_layout=lambda: SimpleNamespace(
                is_block_outermost=False
            ),
        ),
        speculative_config=SimpleNamespace(
            method=method,
            use_eagle=lambda: True,
            use_eagle_block_drop=lambda: block_drop,
        ),
        model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="qwen3_5")),
    )


def _qwen3_5_hybrid_specs(with_mtp_layer: bool) -> dict:
    """Qwen3.5-shaped hybrid: repeating [GDN x3, full-attn x1] blocks, with the
    MTP drafter's full-attn layer (spec-identical to the target's) last."""
    specs = {}
    idx = 0
    for _ in range(2):
        for _ in range(3):
            specs[f"model.layers.{idx}.linear_attn"] = _mamba()
            idx += 1
        specs[f"model.layers.{idx}.self_attn.attn"] = _full()
        idx += 1
    if with_mtp_layer:
        specs["mtp.layers.0.self_attn.attn"] = _full()
    return specs


def _flagged(groups) -> list:
    return [g for g in groups if g.is_eagle_group]


def _has_mamba(group) -> bool:
    return any(
        isinstance(s, MambaSpec) for s in kcu.iter_layer_specs(group.kv_cache_spec)
    )


def main() -> int:
    failures = []

    # T1 (#55390): hybrid MTP — exactly one group flagged, holding the MTP
    # layer; Mamba groups unflagged.
    groups = kcu.get_kv_cache_groups(
        _config("mtp"), _qwen3_5_hybrid_specs(with_mtp_layer=True)
    )
    flagged = _flagged(groups)
    if len(flagged) != 1:
        failures.append(f"T1: expected 1 flagged group, got {len(flagged)}")
    elif "mtp.layers.0.self_attn.attn" not in flagged[0].layer_names:
        failures.append(
            f"T1: flagged group lacks MTP layer: {flagged[0].layer_names}"
        )
    for g in groups:
        if _has_mamba(g) and g.is_eagle_group:
            failures.append(f"T1: mamba group flagged: {g.layer_names}")

    # T2 (#55390): non-MTP eagle family — stays conservative (no annotation).
    groups = kcu.get_kv_cache_groups(
        _config("eagle3"), _qwen3_5_hybrid_specs(with_mtp_layer=True)
    )
    if _flagged(groups):
        failures.append("T2: eagle3 should leave everything unflagged")

    # T3 (#55390): non-MTP model_type gate removed — 'mtp' method is enough
    # even when model_type is not a known MTP family.
    cfg = _config("mtp")
    cfg.model_config = SimpleNamespace(hf_config=SimpleNamespace(model_type="other"))
    groups = kcu.get_kv_cache_groups(cfg, _qwen3_5_hybrid_specs(with_mtp_layer=True))
    if len(_flagged(groups)) != 1:
        failures.append("T3: method=mtp must annotate regardless of model_type")

    # T4 (#55390): broken partition (group dropped a layer) — rule must not fire.
    specs = _qwen3_5_hybrid_specs(with_mtp_layer=True)
    groups = kcu.get_kv_cache_groups(_config("mtp"), specs)
    for g in groups:
        g.is_eagle_group = False
    trimmed = [
        kcu.KVCacheGroupSpec(
            [n for n in g.layer_names if n != "model.layers.0.linear_attn"],
            g.kv_cache_spec,
        )
        for g in groups
    ]
    kcu._annotate_eagle_groups(cfg, specs, trimmed, use_trailing_layer_fallback=True)
    if _flagged(trimmed):
        failures.append("T4: trailing-layer rule fired on a non-partitioning group set")

    # T5 (#55519): warning gate is use_eagle_block_drop, not use_eagle.
    import inspect

    src = inspect.getsource(kcu._warn_if_unannotated_eagle_mamba)
    if "use_eagle_block_drop()" not in src:
        failures.append("T5: _warn_if_unannotated_eagle_mamba not gated on use_eagle_block_drop")
    scheduler_path = (
        __import__("pathlib").Path(kcu.__file__).parent / "sched" / "scheduler.py"
    )
    if "stays eligible for cross-request" not in scheduler_path.read_text():
        failures.append("T5: scheduler 'drop disabled' warning not reworded")

    if failures:
        for f in failures:
            print("FAIL:", f)
        return 1
    print("functional check 55390/55519: ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
