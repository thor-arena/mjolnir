"""Human labels for serving configs, shared by the README tables and the charts:
``kvcache_label`` for a kv-cache-size row, ``spec_label`` for a spec config."""

from __future__ import annotations


def kvcache_label(r: dict) -> str:
    """'official · FlashInfer fp8', 'Mjolnir · fp6 e2m3', ..."""
    kv = str(r.get("kv_dtype") or "auto").replace("fp6_e2m3", "fp6 e2m3")
    if r.get("side") != "official":
        extra = "".join(
            f" + {name}"
            for tag, name in (("LMH8", "fp8 lm_head"), ("VT64K", "64K draft vocab"), ("VT32K", "32K draft vocab"))
            if tag in str(r.get("config", "")).split("_")
        )
        return f"Mjolnir · {kv} KV{extra}"
    be = {"flashinfer": "FlashInfer", "flash_attn": "FlashAttention"}.get(str(r.get("backend") or "").lower())
    return f"official vLLM · {be or r.get('backend') or '?'} · {kv} KV"


def spec_label(cfg: str) -> str:
    return {
        "NVFP4": "official vLLM · FlashInfer fp8",
        "NVFP4_MJOLNIR": "Mjolnir · fp8 KV",
        "NVFP4_MJOLNIR_E2M3": "Mjolnir · fp6 KV",
        "NVFP4_MJOLNIR_E2M3_LMH8_VT64K": "Mjolnir · fp6 KV + fp8 lm_head + 64K draft vocab",
    }.get(cfg, cfg)
