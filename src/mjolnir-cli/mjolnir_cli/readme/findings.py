"""The data-driven one-line findings under each stage's table — sentences
computed from a verify run's raw document, never hand-written numbers."""

from __future__ import annotations

from mjolnir_cli.labels import kvcache_label, spec_label
from mjolnir_cli.readme.fmt import L_label
from mjolnir_cli.verify import results
from mjolnir_cli.verify.spec import best_k
from mjolnir_cli.verify.stages import NOISE_PCT


def findings(stage: str, doc: dict) -> list[str]:
    """Data-driven one-liners for the docs (no hand-written numbers)."""
    out: list[str] = []
    if stage == "spec":
        rows = [r for r in doc.get("rows", []) if r.get("decode_tps")]
        for (cfg, conc), b in sorted(best_k(rows).items()):
            ref = next((r for r in rows if r["config"] == cfg and r["concurrency"] == conc and r["k"] == 3), None)
            vs = (
                f", {(b['decode_tps'] / ref['decode_tps'] - 1) * 100:+.1f}% vs 3 drafts" if ref and b is not ref else ""
            )
            out.append(
                f"{spec_label(cfg)}, {conc} request{'s' if conc > 1 else ''}: best with **{b['k']} draft"
                f"{'s' if b['k'] > 1 else ''}** — {b['decode_tps']:.1f} t/s per request, "
                f"{b['acceptance_length']:.2f} tokens per step{vs}."
            )
        by_cfg: dict = {}
        for r in rows:
            if r["k"] == 3 and r["concurrency"] == 1 and r.get("acceptance_length"):
                by_cfg[r["config"]] = r["acceptance_length"]
        if "NVFP4" in by_cfg:
            for cfg, a in by_cfg.items():
                if cfg != "NVFP4":
                    out.append(
                        f"Acceptance, 3 drafts, 1 request: {spec_label(cfg)} {a:.2f} vs official "
                        f"{by_cfg['NVFP4']:.2f} tokens/step ({(a / by_cfg['NVFP4'] - 1) * 100:+.1f}%) — "
                        "same prompts, temperature 0."
                    )
        return out
    if stage == "kv-cache-size":
        rows = {r["id"]: r for r in doc.get("rows", []) if r.get("kv_tokens")}
        e2m3 = next((r for r in rows.values() if str(r.get("kv_dtype", "")).startswith("fp6")), None)
        for other in rows.values():
            if e2m3 and other is not e2m3:
                gain = e2m3["kv_tokens"] / other["kv_tokens"] - 1
                out.append(
                    f"**{kvcache_label(e2m3)}** holds **{e2m3['kv_tokens']:,} tokens**, "
                    f"{gain * 100:+.0f}% vs {kvcache_label(other)} ({other['kv_tokens']:,})."
                )
        for r in rows.values():
            if r.get("max_concurrency"):
                out.append(
                    f"{kvcache_label(r)}: {r['max_concurrency']:.1f} requests at the full "
                    f"{L_label(r.get('max_model_len') or 0)} context (vLLM's own estimate)."
                )
        return out
    by = results.series_by_id(doc)
    noise = NOISE_PCT / 100
    if stage in ("sweep", "hndsweep"):
        base = by.get("fi")
        for sid in ("mjolnir", "mjolnir-fp8", "fi-hnd", "fa2"):
            if base and sid in by:
                sp = results.speedup(base, by[sid])
                if not sp:
                    continue
                g = results.geomean(sp.values())
                lo, hi = min(sp, key=sp.get), max(sp, key=sp.get)
                beats = sum(1 for v in sp.values() if v >= 1 + noise)
                out.append(
                    f"**{by[sid]['label']}** vs {base['label']}: geomean **{g:.2f}×**, "
                    f"from {sp[lo]:.2f}× (L={L_label(lo)}) to {sp[hi]:.2f}× (L={L_label(hi)}); "
                    f"faster beyond the ±{NOISE_PCT:.0f}% noise band at {beats} of {len(sp)} context lengths."
                )
    if stage in ("ring", "prefetch", "ab", "kvfmt") and "fi" in by:
        variants = [s for s in doc.get("series", []) if s.get("kind") == "t5"]
        if variants:
            Ls = sorted(results.point_map(by["fi"]))
            winners = {}
            for L in Ls:
                best = min(variants, key=lambda s: results.point_map(s).get(L, float("inf")))
                winners[L] = best["label"]
            counts: dict[str, int] = {}
            for w in winners.values():
                counts[w] = counts.get(w, 0) + 1
            top = max(counts, key=counts.get)
            out.append(f"Fastest variant at the most context lengths: **`{top}`** ({counts[top]} of {len(Ls)}).")
            geo = {s["label"]: results.geomean(results.speedup(by["fi"], s).values()) for s in variants}
            geo = {k: v for k, v in geo.items() if v}
            if geo:
                g_top = max(geo, key=geo.get)
                g_low = min(geo, key=geo.get)
                out.append(
                    f"Best geomean vs FlashInfer: **`{g_top}`** ({geo[g_top]:.2f}×); "
                    f"worst: `{g_low}` ({geo[g_low]:.2f}×)."
                )
                spread = geo[g_top] / geo[g_low] - 1
                if spread < noise:
                    out.append(f"All variants sit within {spread * 100:.0f}% of each other — inside the noise band.")
    if stage == "quant":
        by_f: dict[str, list[float]] = {}
        for r in doc.get("rows", []):
            if r.get("rel_l2") is not None:
                by_f.setdefault(r["format"], []).append(r["rel_l2"])
        if "fp8" in by_f:
            fp8 = sum(by_f["fp8"]) / len(by_f["fp8"])
            for f, v in sorted(by_f.items(), key=lambda kv: sum(kv[1]) / len(kv[1])):
                m = sum(v) / len(v)
                out.append(f"`{f}`: mean relative L2 error **{m:.3f}** ({m / fp8:.1f}× fp8's {fp8:.3f}).")
    if stage == "pdl":
        ds = [r["delta_pct"] for r in doc.get("rows", []) if r.get("delta_pct") is not None]
        if ds:
            inside = sum(1 for d in ds if abs(d) < NOISE_PCT)
            out.append(
                f"{inside} of {len(ds)} measurements move by less than the ±{NOISE_PCT:.0f}% noise band "
                f"(range {min(ds):+.1f}% … {max(ds):+.1f}%)."
            )
    if stage == "fp6prof":
        profs = [p for p in doc.get("profiles", []) if p.get("ok")]
        by_lab: dict[str, list[tuple[float, float, float]]] = {}
        by_L: dict[int, dict[str, dict]] = {}
        for p in profs:
            by_L.setdefault(p["L"], {})[p["label"]] = p
        for L, ps in by_L.items():
            labs = list(ps)
            b = ps[labs[0]]
            for lab in labs[1:]:
                p = ps[lab]
                t = p["metrics"]["gpu__time_duration.sum"] / b["metrics"]["gpu__time_duration.sum"]
                ideal = p["kv_bits"] / b["kv_bits"]
                by_lab.setdefault(lab, []).append((t, ideal, L))
        for lab, xs in by_lab.items():
            t = sum(x[0] for x in xs) / len(xs)
            ideal = xs[0][1]
            realized = (1 - t) / (1 - ideal) if ideal < 1 else 0
            out.append(
                f"`{lab}`: kernel time **{t:.2f}×** the baseline for {ideal:.2f}× the bytes — "
                f"{realized * 100:.0f}% of the byte saving becomes speed (mean over L)."
            )
    if stage == "correctness":
        cs = doc.get("cases", [])
        ok = sum(1 for c in cs if c.get("passed"))
        out.append(
            f"**{ok} / {len(cs)}** checks pass (kernel output and LSE vs the bit-faithful emulator, "
            "poisoned unused pages must stay NaN-free)."
        )
    return out
