"""``mjolnir bench``: run the end-to-end benchmarks (perf, A/B, accuracy,
e2e) leg by leg, and compare runs from the history.

Printing results and re-rendering the charts / README is
``commands.bench_report``'s job."""

from __future__ import annotations

import json
import shlex
from typing import Optional

import typer

from mjolnir_cli import benchy, charts, dockerctl, readme
from mjolnir_cli import legs as legs_
from mjolnir_cli.commands._common import HELP_CTX, settings
from mjolnir_cli.commands.bench_report import print_accuracy, print_compare, rerender
from mjolnir_cli.config import (
    DEFAULT_IMAGE,
    DEFAULT_QUANT,
    E2E_DEPTHS,
    STOCK_IMAGE,
    VLLM_VERSION,
    ConfigEntry,
    Settings,
    load_layout,
    scan_all_configs,
)
from mjolnir_cli.e2e import MAX_CTX as _E2E_MAX_CTX

bench_app = typer.Typer(
    help="Benchmarks — gated, raw data, chartable.", no_args_is_help=True, context_settings=HELP_CTX
)


# ── bench ───────────────────────────────────────────────────────────────────


@bench_app.command("perf")
def bench_perf(
    runs: int = typer.Option(6, "--runs", help="measured runs per cell (llama-benchy --runs)"),
    warmup_runs: int = typer.Option(2, "--warmup-runs"),
    repeat: int = typer.Option(1, "--repeat", help="independent gated sweeps (default: one sweep, 6 runs per cell)"),
    depths: str = typer.Option(",".join(map(str, E2E_DEPTHS)), "--depths", help="context depths"),
    concurrency: str = typer.Option("1,2,4", "--concurrency"),
    pp: int = typer.Option(2048, "--pp"),
    tg: int = typer.Option(128, "--tg"),
    exact_tg: bool = typer.Option(
        True, "--exact-tg/--no-exact-tg", help="pin output length to --tg (kills EOS variance)"
    ),
    gate: bool = typer.Option(True, "--gate/--no-gate"),
    confirm: int = typer.Option(3, "--confirm", help="clean-window confirm samples"),
    gate_timeout: float = typer.Option(3600.0, "--gate-timeout"),
    label: Optional[str] = typer.Option(None, "--label", help="backend label for the history log"),
    api_key: Optional[str] = typer.Option(None, "--api-key"),
    model: Optional[str] = typer.Option(None, "--model"),
    config: Optional[str] = typer.Option(None, "--config"),
    image: Optional[str] = typer.Option(None, "--image"),
    port: Optional[int] = typer.Option(None, "--port"),
    plot: bool = typer.Option(True, "--plot/--no-plot"),
):
    """End-to-end perf bench (gated clean window, raw JSON, history log)."""
    s = settings(model, config, image, port)
    layout = load_layout()

    def split_ints(v):
        return [int(x) for x in str(v).split(",") if x.strip()]

    try:
        benchy.run_perf(
            s,
            layout,
            runs=runs,
            warmup_runs=warmup_runs,
            exact_tg=exact_tg,
            repeat=repeat,
            depths=split_ints(depths),
            concurrency=split_ints(concurrency),
            pp=pp,
            tg=tg,
            gate=gate,
            confirm=confirm,
            gate_timeout_s=gate_timeout,
            api_key=api_key,
            label=label,
        )
    except benchy.BenchError as e:
        typer.secho(str(e), fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    if plot:
        try:
            charts.render_all(layout)
        except RuntimeError as e:
            typer.secho(f"(plots skipped: {e})", fg=typer.colors.YELLOW, err=True)
        try:
            readme.render_readme(layout)
        except (RuntimeError, FileNotFoundError) as e:
            typer.secho(f"(readme skipped: {e})", fg=typer.colors.YELLOW, err=True)


def _ab_legs(
    specs: str, default_model: str, default_image: str, entries: list[ConfigEntry]
) -> list[tuple[str, str, str]]:
    """Parse ``--backends`` into ``(model, image, quant)`` legs.

    Each element is one of:
      * ``<config>``            — the default model's config, served from
                                   ``default_image``;
      * ``<image>:<config>``    — the default model's config pinned to an
                                   image (the last ``:`` splits, so the image
                                   may carry its own registry port / tag);
      * ``<model>/<config>``    — a different model (``vendor/Model``) with
                                   its config, served from ``default_image``;
      * ``<image>:<model>/<config>`` — the same, with the image pinned too
                                   (the last ``:`` splits the image off).
    """
    quants = {e.quant for e in entries if e.model == default_model}
    model_quants = {(e.model, e.quant) for e in entries}
    legs: list[tuple[str, str, str]] = []
    for spec in (b.strip() for b in specs.split(",")):
        if not spec:
            continue
        if ":" in spec:
            image, _, spec = spec.rpartition(":")
            if not image.strip():
                raise ValueError(f"bad leg '{spec}' — expected '<config>', '<model>/<config>' or '<image>:<config>'")
        else:
            image = None
        if "/" in spec:
            model, _, quant = spec.rpartition("/")
            if not model or not quant or model.count("/") != 1:
                raise ValueError(f"bad leg '{spec}' — expected '<config>', '<model>/<config>' or '<image>:<config>'")
            if (model, quant) not in model_quants:
                raise ValueError(f"unknown model config '{model}/{quant}' — available configs: mjolnir model list")
            legs.append((model, image or default_image, quant))
        else:
            if spec not in quants:
                raise ValueError(
                    f"unknown config '{spec}' for model {default_model} — available configs: mjolnir model list"
                )
            legs.append((default_model, image or default_image, spec))
    if not legs:
        raise ValueError("--backends is empty")
    return legs


@bench_app.command("ab")
def bench_ab(
    backends: str = typer.Option(
        legs_.bench_legs("bench.ab"),
        "--backends",
        help="comma-separated legs: each is "
        "'<config>' (the --model/state "
        "model, served from --image), "
        "'<image>:<config>', "
        "'<model>/<config>' or "
        "'<image>:<model>/<config>' — "
        "available: mjolnir model list",
    ),
    depths: str = typer.Option(
        ",".join(map(str, E2E_DEPTHS)), "--depths", help="context depths (tokens already in the KV)"
    ),
    runs: int = typer.Option(6, "--runs"),
    repeat: int = typer.Option(1, "--repeat"),
    gate: bool = typer.Option(True, "--gate/--no-gate"),
    model: Optional[str] = typer.Option(None, "--model"),
    image: Optional[str] = typer.Option(
        None, "--image", help="image for bare-config legs (default: $MJOLNIR_IMAGE / built-in default)"
    ),
    port: Optional[int] = typer.Option(None, "--port"),
):
    """Headline A/B: restart the server per leg, gated perf bench for each,
    one history row-set per leg.

    Each leg is a model config (quant), optionally model-qualified and/or
    image-pinned: '<config>', '<image>:<config>', '<model>/<config>' or
    '<image>:<model>/<config>'. The image carries the backend (kernel stack
    is baked in); the config (configs/<model>/<quant>/config.yaml, repo or local)
    carries the serving parameters. Bare configs use --model/--image.
    Compare images on one model and/or models on one image. Charts
    re-render at the end."""
    s = settings(model, None, image, port)
    layout = load_layout()
    entries = scan_all_configs(layout)
    try:
        legs = _ab_legs(backends, s.model, s.image, entries)
    except ValueError as e:
        typer.secho(str(e), fg=typer.colors.RED, err=True)
        raise typer.Exit(4)
    from mjolnir_cli import e2e

    for m_, img, q in legs:
        if q != DEFAULT_QUANT and e2e.is_mjolnir({"image": img}):
            typer.secho(
                f"note: {q} on {img} is not native vLLM (the Mjolnir image patches FlashInfer too) — it is kept in "
                f"history but never charted; measure native backends on {STOCK_IMAGE}",
                fg=typer.colors.YELLOW,
            )
    run_legs(
        s,
        layout,
        [(m, i, q, None) for m, i, q in legs],
        runs=runs,
        repeat=repeat,
        gate=gate,
        depths=[int(x) for x in depths.replace(" ", ",").split(",") if x.strip()],
    )
    rerender(layout)


@bench_app.command(
    "accuracy",
    context_settings={**HELP_CTX, "allow_extra_args": True, "ignore_unknown_options": True},
)
def bench_accuracy(
    ctx: typer.Context,
    backends: str = typer.Option(
        legs_.bench_legs("bench.accuracy"),
        "--backends",
        help="comma-separated legs, as in 'bench ab': '<config>', '<image>:<config>', '<model>/<config>' or "
        "'<image>:<model>/<config>' (default: mjolnir_cli.legs)",
    ),
    restart: bool = typer.Option(
        True,
        "--restart/--no-restart",
        help="serve each leg (restarts the server; the GPU must be yours) — --no-restart measures the running "
        "server as one leg (pass its config with --backends)",
    ),
    model: Optional[str] = typer.Option(None, "--model"),
    image: Optional[str] = typer.Option(None, "--image", help="image for bare-config legs"),
    port: Optional[int] = typer.Option(None, "--port"),
    dry_run: bool = typer.Option(False, "--dry-run", help="print the legs and the tool-eval-bench command"),
    plot: bool = typer.Option(True, "--plot/--no-plot"),
):
    """Model quality per serving config with tool-eval-bench: does fp6 KV /
    the fp8 lm_head keep the model's answers?

    Default per leg: the 69 standard tool-calling scenarios + the 28 Hard Mode
    ones (`tool-eval-bench run --hardmode --seed 42 --timeout 600`). Anything
    after `--` replaces that and is forwarded verbatim (the leg's --base-url is
    added), e.g. `-- run --hardmode --no-think --trials 3` or
    `-- --gsm8k --mmlu --ifeval --skip-tool-eval --gsm8k-limit 200`.
    Results: benchmarks/raw/accuracy/<run>/accuracy.json (committed; the
    tool's SQLite, reports and the server log stay next to it, untracked);
    docs/kernel/accuracy.md is rendered from the newest result per config and
    suite."""
    from mjolnir_cli import accuracy

    s = settings(model, None, image, port)
    layout = load_layout()
    entries = scan_all_configs(layout)
    try:
        legs = _ab_legs(backends, s.model, s.image, entries)
    except ValueError as e:
        typer.secho(str(e), fg=typer.colors.RED, err=True)
        raise typer.Exit(4)
    passthrough = list(ctx.args)
    if not restart and len(legs) != 1:
        typer.secho("--no-restart measures the running server: give exactly one leg", fg=typer.colors.RED, err=True)
        raise typer.Exit(2)
    if dry_run:
        try:
            tool = accuracy.find_tool()
        except accuracy.AccuracyError as e:
            tool = [f"<{e}>"]
        for m_, img, q in legs:
            typer.echo(f"{m_}/{q} on {img}:")
            typer.echo("  " + shlex.join(tool + accuracy.tool_args(passthrough, s.base_url)))
        return
    if restart:
        typer.secho(
            "bench accuracy restarts the vLLM server for each leg — the GPU must be yours for the whole run.",
            fg=typer.colors.YELLOW,
        )
    elif not dockerctl.container_running():
        typer.secho("--no-restart: no vLLM server is running (mjolnir serve up)", fg=typer.colors.RED, err=True)
        raise typer.Exit(3)
    try:
        out = accuracy.run_accuracy(s, layout, legs, passthrough, restart=restart)
    except accuracy.AccuracyError as e:
        typer.secho(str(e), fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    print_accuracy(json.loads(out.read_text()))
    typer.secho(f"wrote {out}", fg=typer.colors.GREEN)
    if plot:
        rerender(layout)


def run_legs(s: Settings, layout, legs: list[tuple], **perf) -> None:
    """Serve each (model, image, config, label) leg in turn — the server is
    restarted per leg on purpose — run one gated perf bench on it, and stop
    the leg's server when it is done (or fails), so no vLLM stays up holding
    the GPU after the bench."""
    for leg_model, leg_image, quant, label in legs:
        s.model, s.image, s.quant = leg_model, leg_image, quant
        s.label = label
        typer.secho(f"\n{'=' * 62}\n  leg: {leg_model}/{quant}  (image {leg_image})\n{'=' * 62}", fg=typer.colors.CYAN)
        dockerctl.serve_down(s)
        try:
            dockerctl.serve_up(s, wait=True)
            benchy.run_perf(s, layout, warmup_runs=2, exact_tg=True, **perf)
        except benchy.BenchError as e:
            typer.secho(str(e), fg=typer.colors.RED, err=True)
            raise typer.Exit(1)
        finally:
            typer.secho(f"  leg done: stopping the vLLM server ({leg_image})", fg=typer.colors.CYAN)
            dockerctl.serve_down(s)


@bench_app.command("e2e")
def bench_e2e(
    official_image: str = typer.Option(STOCK_IMAGE, "--official-image", help="the upstream vLLM image (baseline)"),
    official_config: str = typer.Option(
        legs_.side_configs("bench.e2e", legs_.OFFICIAL)[0],
        "--official-config",
        help="config served on the official image (stock backends only)",
    ),
    image: str = typer.Option(DEFAULT_IMAGE, "--image", help="the Mjolnir image"),
    config: str = typer.Option(
        legs_.side_configs("bench.e2e", legs_.MJOLNIR)[0], "--config", help="config served on the Mjolnir image"
    ),
    model: Optional[str] = typer.Option(None, "--model"),
    runs: int = typer.Option(6, "--runs", help="measured runs per cell"),
    repeat: int = typer.Option(1, "--repeat", help="independent gated sweeps per leg"),
    depths: str = typer.Option(
        ",".join(str(d) for d in E2E_DEPTHS if d <= _E2E_MAX_CTX),
        "--depths",
        help="context depths (tokens already in the KV); the comparison shows up to 32K",
    ),
    concurrency: str = typer.Option("1,2,4", "--concurrency"),
    pp: int = typer.Option(2048, "--pp", help="prompt tokens per request (prefill)"),
    tg: int = typer.Option(128, "--tg", help="generated tokens per request (decode)"),
    gate: bool = typer.Option(True, "--gate/--no-gate"),
    port: Optional[int] = typer.Option(None, "--port"),
    only: Optional[str] = typer.Option(
        None,
        "--only",
        help="run one leg: 'mjolnir' or 'official'; the comparison pairs it with the other side's latest record",
    ),
    also: str = typer.Option(
        ",".join(legs_.side_configs("bench.e2e", legs_.MJOLNIR)[1:]),
        "--also",
        help="more legs to measure after the pair, comma-separated: a config (Mjolnir image) or "
        "official:<config> / mjolnir:<config>; '' = none. They land in the history (bench compare "
        "--config <cfg>); the README table, charts and infographic stay on --config",
    ),
    optional: bool = typer.Option(
        False,
        "--optional",
        help="also measure the opt-in legs (legs.OPTIONAL: DSpark and DFlash2 on the official and the Mjolnir "
        "image) — hours per leg",
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="print the serve commands, run nothing"),
    allow_stale_image: bool = typer.Option(
        False, "--allow-stale-image", help="benchmark a Mjolnir image older than the checkout's kernel"
    ),
):
    """Official vLLM image vs the Mjolnir image, end to end: decode and prefill
    throughput, time to first token and inter-token latency per context x
    concurrency. Serves each image in turn (the server is restarted per leg),
    benchmarks it with llama-benchy, then prints the comparison and re-renders
    the README (infographic + table)."""
    s = settings(model, None, None, port)
    layout = load_layout()

    def ints(v: str) -> list[int]:
        return [int(x) for x in v.replace(" ", ",").split(",") if x.strip()]

    legs = [
        (s.model, official_image, official_config, f"vLLM {VLLM_VERSION} (official image)"),
        (s.model, image, config, None),
    ]
    if only:
        side = only.strip().lower()
        if side not in ("mjolnir", "official"):
            typer.secho(f"--only must be 'mjolnir' or 'official', got {only!r}", fg=typer.colors.RED, err=True)
            raise typer.Exit(2)
        legs = [legs[1] if side == "mjolnir" else legs[0]]
    extra = [legs_.parse_leg(q) for q in also.split(",") if q.strip()]
    if optional:
        extra += list(legs_.OPTIONAL["bench.e2e"])
    side_only = only.strip().lower() if only else None
    for side, q in dict.fromkeys(extra):
        if side_only and side != side_only:
            continue
        if side == legs_.OFFICIAL and q != official_config:
            legs.append((s.model, official_image, q, f"vLLM {VLLM_VERSION} (official image)"))
        elif side == legs_.MJOLNIR and q != config:
            legs.append((s.model, image, q, None))
    stale = dockerctl.stale_image(image) if any(leg[1] == image for leg in legs) else None
    if stale and not dry_run:
        if not allow_stale_image:
            typer.secho(f"{stale} (or --allow-stale-image)", fg=typer.colors.RED, err=True)
            raise typer.Exit(2)
        typer.secho(f"warning: {stale}", fg=typer.colors.YELLOW, err=True)
    if dry_run:
        for m_, img, q, _ in legs:
            s.model, s.image, s.quant = m_, img, q
            dockerctl.serve_up(s, wait=False, dry_run=True)
        return
    typer.secho(
        "bench e2e restarts the vLLM server for each image — the GPU must be yours for the whole run.",
        fg=typer.colors.YELLOW,
    )
    run_legs(
        s,
        layout,
        legs,
        runs=runs,
        repeat=repeat,
        gate=gate,
        depths=ints(depths),
        concurrency=ints(concurrency),
        pp=pp,
        tg=tg,
    )
    print_compare(layout, config, official_config=official_config, model=s.model)
    measured = [(img, q) for _, img, q, _ in legs if q not in (config, official_config)]
    if measured:
        typer.echo("optional legs — compare them with:")
        offs = [q for img, q in measured if img == official_image]
        for img, q in measured:
            if img != image:
                continue
            twin = next((o for o in offs if o.split("_")[-1] == q.split("_")[-1]), None)
            typer.echo(
                f"  mjolnir bench compare --model {s.model} --config {q}"
                + (f"   |   --official-config {twin} --config {q}" if twin else "")
            )
    rerender(layout)


@bench_app.command("compare")
def bench_compare(
    config: Optional[str] = typer.Option(None, "--config", help="Mjolnir-side config (default: the Mjolnir backend)"),
    official_config: Optional[str] = typer.Option(
        None, "--official-config", help="official-side config (default: the stock baseline, NVFP4)"
    ),
    as_json: bool = typer.Option(False, "--json", "-j"),
    model: Optional[str] = typer.Option(None, "--model", help="whose runs to pair (default: the active model)"),
):
    """Official vLLM image vs Mjolnir image from the history — no new run.
    `--official-config NVFP4_DSPARK --config NVFP4_MJOLNIR_E2M3_DSPARK` compares
    the same drafter on both images; both sides are runs of one model."""
    model = settings(model, None, None, None).model
    raise typer.Exit(print_compare(load_layout(), config, as_json, official_config, model))
