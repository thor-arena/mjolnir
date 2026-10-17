"""The collapsed "How the % and × here are calculated" panels, one under every
table / figure that shows a % or × (``how(ctx, kind)`` in the templates)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # type-only: how reads a context, it never builds one
    from mjolnir_cli.readme.context import ReadmeContext

_HOW: dict[str, str] = {
    "kernel": """\
- **Time per decode step** = Nsight Compute kernel time (`gpu__time_duration`) of the decode kernel **plus** its split-KV merge kernel, at base clocks with a cold L2, median over the iterations. The reference (FlashInfer, FlashAttention 2) is measured the same way, at the same L, in the same run.
- **Speedup ×** = reference time ÷ Mjolnir time. `1.24×` means Mjolnir needs 1 ÷ 1.24 = 81% of the reference's time per step (19% less time); below 1 the reference is faster.
- **Geomean over L** = exp(mean of ln(speedup at each L)): every context length weighs the same, and one large ratio cannot dominate the way it would in an arithmetic mean.
- **Colors / "beyond the noise band"**: green = speedup ≥ 1.03, grey = within ±3%, red ≤ 0.97. ±3% (`NOISE_PCT`) is how far a same-run ratio moves between repeated runs on a dedicated Thor.""",
    "variant": """\
- Each cell = **reference time ÷ variant time** (Nsight Compute kernel time, decode + merge, base clocks, cold L2, same run, same L). `1.10×` = the variant needs 1 ÷ 1.10 = 91% of the reference's time; below 1 it is slower.
- **geomean** = exp(mean of ln(ratio)) over the context lengths; **bold** = the fastest variant at that L.
- Green ≥ 1.03, grey within ±3% (`NOISE_PCT`, the run-to-run spread of a same-run ratio), red ≤ 0.97.""",
    "wall": """\
- Wall-clock median per launch over the stage's iterations (CUDA events), every variant in the same run.
- **%** = (variant − first column) ÷ first column × 100 — the change in **time**: negative = faster. `−6.0%` = 6% less time than the first column.
- Green = at least 3% less time, red = at least 3% more, grey = within ±3% (`NOISE_PCT`).""",
    "pdl": """\
- **%** = (time with PDL − time without PDL) ÷ time without PDL × 100, harness median per L, both measured back to back in the same run. Negative = PDL saves time.
- Green ≤ −3%, red ≥ +3%, grey within ±3% (`NOISE_PCT`).""",
    "fp6prof": """\
- **time ratio** = this row's Nsight Compute kernel time ÷ the first row's at the same L; **L2 bytes ratio** = L2 read sectors ÷ the first row's; **ideal** = this row's KV bits ÷ the first row's — the ratio a purely bandwidth-bound kernel would show.
- **bytes / L2 request** = 32 B × L2 read sectors ÷ L2 read requests.""",
    "kvcache": """\
- **KV tokens** and KV memory come from vLLM's own startup log ("GPU KV cache size", "Available KV cache memory") for the same model and memory budget.
- **vs first** = this row's KV tokens ÷ the first row's (`1.33×` = 33% more tokens).
- **fits @ L** = ⌊KV tokens ÷ (L + 2,048 prompt + 128 generated)⌋ — whole requests of the e2e bench's shape that fit at once (`max-num-seqs` ignored).""",
    "spec": """\
- **tokens/step** = 1 + accepted draft tokens ÷ verify steps, from vLLM's speculative-decoding counters read before and after the block; **accept @i** = the share of verify steps whose i-th draft was accepted.
- **decode t/s per request** = (generated tokens − 1) ÷ (time of the last token − time of the first token); **aggregate t/s** = all generated tokens ÷ the block's wall time.
- ★ = the draft count with the highest decode t/s for that config and concurrency.""",
    "accuracy": """\
- **Score** = points ÷ maximum points × 100, each scenario 2 (pass), 1 (partial) or 0 (fail); GSM8K / MMLU = accuracy in %; IFEval = prompt-level accuracy in %.
- **(±)** = this row's score − the baseline row's score, in percentage **points**, not a relative change: 86.5 vs 88.0 is −1.5 points (a relative change would be −1.7%).
- **n** = runs averaged (`--trials`); **completion** = the share of scenarios the endpoint answered — compare rows only at equal completion.""",
}


def _signed(v: float) -> str:
    return f"{'−' if v < 0 else '+'}{abs(v):.1f}%"


def _how_e2e(ctx: "ReadmeContext") -> str:
    ex = ctx.e2e_example
    lines = [
        "- **Throughput** (decode, prefill — tokens/s, higher is better): change = (Mjolnir − official) ÷ official × 100. "
        "`+57%` = 1.57× the tokens per second. On the infographic a change of +100% or more is written as the multiple "
        "Mjolnir ÷ official (`+200%` = `3×`).",
        "- **Latency** (TTFT, ITL — milliseconds, lower is better): change = (Mjolnir − official) ÷ official × 100, so a "
        "negative value is a **latency reduction**: `−27%` = 27% fewer milliseconds. It is *not* the reciprocal speedup "
        "official ÷ Mjolnir − 1, which reads larger for the same pair.",
    ]
    if ex:
        lines.append(
            f"- **Worked example** ({ex['ctx']} context, 1 request): TTFT {ex['a']:,.0f} → {ex['b']:,.0f} ms: "
            f"({ex['b']:,.0f} − {ex['a']:,.0f}) ÷ {ex['a']:,.0f} × 100 = **{_signed(ex['change'])}** (a latency reduction "
            f"of {-ex['change']:.1f}%). The reciprocal speedup would be {ex['a']:,.0f} ÷ {ex['b']:,.0f} = {ex['ratio']:.2f}× "
            f"({_signed((ex['ratio'] - 1) * 100)}) — the same measurement, a different convention."
        )
    lines += [
        "- **Values**: the mean over the measured runs per cell (6 measured after 2 warm-ups, llama-benchy): decode = "
        "generated tokens ÷ generation time per request (summed over requests at higher concurrency), prefill = prompt "
        "tokens ÷ prompt-processing time, TTFT = time to the first content token, ITL = 1000 ÷ the per-request decode rate.",
        f"- **Colors**: green = better by at least {ctx.e2e_noise_pct}%, grey = within ±{ctx.e2e_noise_pct}% "
        "(`E2E_NOISE_PCT`), red = worse by at least that. With speculative decoding a single decode cell moves more "
        "between runs (acceptance depends on the prompt), so repeat (`--repeat`) before reading a few-% difference.",
    ]
    return "\n".join(lines)


def how(ctx: "ReadmeContext", kind: str) -> str:
    """A collapsed "How this is calculated" panel for the tables / figures of
    ``kind`` (GitHub renders the markdown inside ``<details>``)."""
    body = _how_e2e(ctx) if kind == "e2e" else _HOW[kind]
    return f"<details>\n<summary>How the % and × here are calculated</summary>\n\n{body}\n\n</details><br />"
