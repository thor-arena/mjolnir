"""The ``mjolnir`` CLI — one command surface for the whole repo.

    mjolnir serve up|down|status|logs    drive the vLLM container
    mjolnir model use|show              remember the active model/config
    mjolnir image build|gates           build the patched image / run canaries
    mjolnir vfa prepare                 build the GEMV'd vllm_flash_attn tree
    mjolnir gate                        the clean-window gate
    mjolnir bench perf|ab|kernel        end-to-end perf / A/B / kernel benches
    mjolnir verify                      kernel correctness suites
    mjolnir plot|history                charts + the raw-data log
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Optional

import typer

from mjolnir import __version__
from mjolnir.config import (BACKEND_LABELS, BASELINE_QUANT, Settings,
                           load_layout, resolve)
from mjolnir import benchy, dockerctl, plots, tasks, vfa

serve_app = typer.Typer(help="Drive the vLLM container (the patched image).")
bench_app = typer.Typer(help="Benchmarks — gated, raw data, chartable.")
model_app = typer.Typer(help="Pick the active model / config.")
image_app = typer.Typer(help="The patched vLLM image.")

app = typer.Typer(
    no_args_is_help=True,
    help="Mjolnir — vLLM on Jetson Thor (sm_110): patched image, FA4 GEMV "
         "decode kernel, one-command serve/bench/verify CLI.")
app.add_typer(serve_app, name="serve")
app.add_typer(bench_app, name="bench")
app.add_typer(model_app, name="model")
app.add_typer(image_app, name="image")


def _state_file() -> Path:
    return Path("~/.mjolnir-state.json").expanduser()


def _load_state() -> dict:
    p = _state_file()
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:  # noqa: BLE001
            pass
    return {}


def _settings(model: Optional[str], quant: Optional[str],
              image: Optional[str], port: Optional[int],
              gemv: Optional[bool] = None) -> Settings:
    st = _load_state()
    s = resolve(model or st.get("model"), quant or st.get("quant"),
                image, port, gemv)
    return s


# ── serve ───────────────────────────────────────────────────────────────────

@serve_app.command("up")
def serve_up(model: Optional[str] = typer.Option(None, "--model"),
              config: Optional[str] = typer.Option(None, "--config",
                                                  help="config quant name, "
                                                       "e.g. NVFP4_14_FA4hd256"),
              image: Optional[str] = typer.Option(None, "--image"),
              port: Optional[int] = typer.Option(None, "--port"),
              gemv: bool = typer.Option(True, "--gemv/--no-gemv",
                                        help="route hd256 decode to the GEMV "
                                             "kernel (VLLM_FA4_HD256_GEMV)"),
              wait: bool = typer.Option(True, "--wait/--no-wait",
                                        help="wait for /health before "
                                             "returning"),
              wait_timeout: float = typer.Option(1800.0, "--wait-timeout"),
              dry_run: bool = typer.Option(False, "--dry-run")):
    """Start the vLLM server (custom image, config from --model/--config)."""
    s = _settings(model, config, image, port, gemv)
    layout = load_layout()
    if not s.config_path.exists():
        typer.secho(f"config not found: {s.config_path}", fg=typer.colors.RED,
                    err=True)
        raise typer.Exit(4)
    try:
        dockerctl.serve_up(s, layout.repo_root, wait=wait,
                           wait_timeout_s=wait_timeout, dry_run=dry_run)
    except (dockerctl.DockerError, TimeoutError) as e:
        typer.secho(str(e), fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    typer.secho(f"\n  serving {s.model} ({s.quant}, "
                f"{s.backend_label}) at {s.base_url}", fg=typer.colors.GREEN)
    typer.echo(f"  gate:  mjolnir gate --once")
    typer.echo(f"  bench: mjolnir bench perf")


@serve_app.command("down")
def serve_down(model: Optional[str] = typer.Option(None, "--model"),
               config: Optional[str] = typer.Option(None, "--config"),
               image: Optional[str] = typer.Option(None, "--image"),
               port: Optional[int] = typer.Option(None, "--port"),
               dry_run: bool = typer.Option(False, "--dry-run")):
    """Stop the vLLM container."""
    s = _settings(model, config, image, port)
    dockerctl.serve_down(s, load_layout().repo_root, dry_run=dry_run)


@serve_app.command("status")
def serve_status(port: Optional[int] = typer.Option(None, "--port")):
    """Container state + /health + current queue load."""
    s = _settings(None, None, None, port)
    st = dockerctl.status(s)
    typer.echo(json.dumps(st, indent=2))
    from mjolnir.gate import server_load
    load = server_load(s.metrics_url)
    if load is not None:
        typer.echo(f"  queue: running={load[0]:g} waiting={load[1]:g}")


@serve_app.command("logs")
def serve_logs(port: Optional[int] = typer.Option(None, "--port"),
               follow: bool = typer.Option(False, "--follow", "-f"),
               tail: int = typer.Option(100, "--tail")):
    """Tail the vLLM container logs."""
    s = _settings(None, None, None, port)
    raise typer.Exit(dockerctl.logs(s, follow=follow, tail=tail))


# ── model ───────────────────────────────────────────────────────────────────

@model_app.command("use")
def model_use(model: str = typer.Argument(..., help="vendor/Model, e.g. "
                                                    "Qwen/Qwen3.8-27B"),
               quant: str = typer.Argument(..., help="config name, e.g. "
                                                     "NVFP4_14_FA4hd256")):
    """Remember the active model/config for serve/bench (no .env needed)."""
    layout = load_layout()
    cfg = layout.repo_root / "configs" / model / f"{quant}.yaml"
    if not cfg.exists():
        known = sorted(str(p.relative_to(layout.repo_root / "configs"))
                       for p in (layout.repo_root / "configs").rglob("*.yaml"))
        typer.secho(f"config not found: {cfg}\nknown configs:\n  "
                    + "\n  ".join(known), fg=typer.colors.RED, err=True)
        raise typer.Exit(4)
    _state_file().write_text(json.dumps({"model": model, "quant": quant},
                                        indent=2))
    typer.secho(f"active: {model} / {quant}  "
                f"(→ {BACKEND_LABELS.get(quant, quant)})",
                fg=typer.colors.GREEN)


@model_app.command("show")
def model_show():
    """Show the active model/config."""
    st = _load_state() or {"model": None, "quant": None}
    typer.echo(json.dumps(st, indent=2))


# ── image ───────────────────────────────────────────────────────────────────

@image_app.command("build")
def image_build(tag: Optional[str] = typer.Option(None, "--tag"),
                image: Optional[str] = typer.Option(None, "--image"),
                dry_run: bool = typer.Option(False, "--dry-run")):
    """Build the patched vLLM image (applies + verifies the patch stack)."""
    s = _settings(None, None, image, None)
    tag = tag or s.image
    layout = load_layout()
    ctx = layout.repo_root / "docker" / "vllm-thor"
    cmd = ["docker", "build", "-t", tag, str(ctx)]
    if dry_run:
        print(" ".join(cmd))
        return
    import subprocess
    rc = subprocess.call(cmd)
    if rc == 0:
        typer.secho(f"built {tag} — canary:  mjolnir image gates",
                    fg=typer.colors.GREEN)
    raise typer.Exit(rc)


@image_app.command("gates")
def image_gates(image: Optional[str] = typer.Option(None, "--image"),
                dry_run: bool = typer.Option(False, "--dry-run")):
    """Run the sm_110 gate-probe canaries against the image (fresh container,
    no server needed)."""
    s = _settings(None, None, image, None)
    raise typer.Exit(tasks.run_task(s, "gates", [], dry_run=dry_run))


# ── vfa ─────────────────────────────────────────────────────────────────────

@app.command()
def vfa(image: Optional[str] = typer.Option(None, "--image"),
        out: Optional[Path] = typer.Option(None, "--out"),
        ) -> None:
    """Prepare the GEMV'd vllm_flash_attn tree (image tree + kernel + diff)."""
    s = _settings(None, None, image, None)
    try:
        vfa.prepare_vfa_tree(s, out)
    except RuntimeError as e:
        typer.secho(str(e), fg=typer.colors.RED, err=True)
        raise typer.Exit(1)


# ── gate ────────────────────────────────────────────────────────────────────

@app.command()
def gate(port: Optional[int] = typer.Option(None, "--port"),
         once: bool = typer.Option(False, "--once",
                                   help="check now and exit"),
         confirm: int = typer.Option(6, "--confirm"),
         timeout: float = typer.Option(3600.0, "--timeout")):
    """The clean-window gate (vLLM request queue 0/0 × N consecutive samples)."""
    s = _settings(None, None, None, port)
    from mjolnir.tasks import _run_gate
    args = ["--once"] if once else [f"--confirm", str(confirm),
                                    "--timeout", str(timeout)]
    raise typer.Exit(_run_gate(s, args, False))


# ── bench ───────────────────────────────────────────────────────────────────

@bench_app.command("perf")
def bench_perf(runs: int = typer.Option(5, "--runs",
                                        help="measured runs per cell "
                                             "(llama-benchy --runs)"),
                warmup_runs: int = typer.Option(2, "--warmup-runs"),
                repeat: int = typer.Option(3, "--repeat",
                                           help="independent gated sweeps"),
                depths: str = typer.Option("0,4096,8192", "--depths"),
                concurrency: str = typer.Option("1,2,4", "--concurrency"),
                pp: int = typer.Option(2048, "--pp"),
                tg: int = typer.Option(128, "--tg"),
                exact_tg: bool = typer.Option(True, "--exact-tg/--no-exact-tg",
                                              help="pin output length to "
                                                   "--tg (kills EOS variance)"),
                gate: bool = typer.Option(True, "--gate/--no-gate"),
                confirm: int = typer.Option(3, "--confirm",
                                            help="clean-window confirm "
                                                 "samples"),
                gate_timeout: float = typer.Option(3600.0, "--gate-timeout"),
                label: Optional[str] = typer.Option(None, "--label",
                                                    help="backend label for "
                                                         "the history log"),
                api_key: Optional[str] = typer.Option(None, "--api-key"),
                model: Optional[str] = typer.Option(None, "--model"),
                config: Optional[str] = typer.Option(None, "--config"),
                image: Optional[str] = typer.Option(None, "--image"),
                port: Optional[int] = typer.Option(None, "--port"),
                plot: bool = typer.Option(True, "--plot/--no-plot")):
    """End-to-end perf bench (gated clean window, raw JSON, history log)."""
    s = _settings(model, config, image, port)
    layout = load_layout()
    split_ints = lambda v: [int(x) for x in str(v).split(",") if x.strip()]
    try:
        benchy.run_perf(s, layout, runs=runs, warmup_runs=warmup_runs,
                        exact_tg=exact_tg, repeat=repeat,
                        depths=split_ints(depths),
                        concurrency=split_ints(concurrency), pp=pp, tg=tg,
                        gate=gate, confirm=confirm, gate_timeout_s=gate_timeout,
                        api_key=api_key, label=label)
    except benchy.BenchError as e:
        typer.secho(str(e), fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    if plot:
        try:
            plots.render_all(layout)
        except RuntimeError as e:
            typer.secho(f"(plots skipped: {e})", fg=typer.colors.YELLOW,
                        err=True)


@bench_app.command("ab")
def bench_ab(backends: str = typer.Option("fa4-gemv,flashinfer", "--backends",
                                          help="comma list: fa4-gemv,"
                                               "fa4-1cta,flashinfer"),
             runs: int = typer.Option(5, "--runs"),
             repeat: int = typer.Option(2, "--repeat"),
             gate: bool = typer.Option(True, "--gate/--no-gate"),
             model: Optional[str] = typer.Option(None, "--model"),
             image: Optional[str] = typer.Option(None, "--image"),
             port: Optional[int] = typer.Option(None, "--port")):
    """Headline A/B: restart the server per backend (FA4 GEMV vs FlashInfer),
    gated perf bench for each, one history row-set per leg."""
    s = _settings(model, None, image, port)
    layout = load_layout()
    plan = {
        "fa4-gemv": ("NVFP4_14_FA4hd256", True),
        "fa4-1cta": ("NVFP4_14_FA4hd256", False),
        "flashinfer": (BASELINE_QUANT, False),
    }
    legs = [b.strip() for b in backends.split(",") if b.strip()]
    for b in legs:
        if b not in plan:
            typer.secho(f"unknown backend '{b}' — "
                        f"known: {', '.join(plan)}", fg=typer.colors.RED,
                        err=True)
            raise typer.Exit(4)
    for b in legs:
        quant, gemv = plan[b]
        s.quant = quant
        s.gemv = gemv
        s.backend_label = BACKEND_LABELS.get(quant, quant)
        typer.secho(f"\n{'=' * 62}\n  A/B leg: {b}  "
                    f"(config {quant}, GEMV={'on' if gemv else 'off'})\n"
                    f"{'=' * 62}", fg=typer.colors.CYAN)
        if not s.config_path.exists():
            typer.secho(f"config not found: {s.config_path}",
                        fg=typer.colors.RED, err=True)
            raise typer.Exit(4)
        dockerctl.serve_down(s, layout.repo_root)
        dockerctl.serve_up(s, layout.repo_root, wait=True)
        try:
            benchy.run_perf(s, layout, runs=runs, repeat=repeat,
                            warmup_runs=2, exact_tg=True, gate=gate)
        except benchy.BenchError as e:
            typer.secho(str(e), fg=typer.colors.RED, err=True)
            raise typer.Exit(1)
    try:
        plots.render_all(layout)
    except RuntimeError as e:
        typer.secho(f"(plots skipped: {e})", fg=typer.colors.YELLOW, err=True)


@bench_app.command("kernel")
def bench_kernel(task: str = typer.Argument(..., help="task name — see "
                                                     "mjolnir bench kernel "
                                                     "--help-list"),
                 args: List[str] = typer.Argument(None, help="script args "
                                                            "(passed "
                                                             "through)"),
                 image: Optional[str] = typer.Option(None, "--image"),
                 vfa_tree: Optional[Path] = typer.Option(None, "--vfa-tree"),
                 dry_run: bool = typer.Option(False, "--dry-run"),
                 skip_preflight: bool = typer.Option(False, "--skip-preflight"),
                 help_list: bool = typer.Option(False, "--help-list",
                                               help="list the tasks")):
    """Run a kernel verification / benchmark (gated docker task)."""
    if help_list:
        for t in tasks.TASKS:
            typer.echo(f"  {t.name:<15} [{t.kind:<5}]  {t.desc}")
        raise typer.Exit(0)
    s = _settings(None, None, image, None)
    raise typer.Exit(tasks.run_task(s, task, args or [], image=image,
                                    vfa_tree=vfa_tree, dry_run=dry_run,
                                    skip_preflight=skip_preflight))


# ── verify ──────────────────────────────────────────────────────────────────

@app.command()
def verify(image: Optional[str] = typer.Option(None, "--image"),
           vfa_tree: Optional[Path] = typer.Option(None, "--vfa-tree"),
           dry_run: bool = typer.Option(False, "--dry-run")):
    """Kernel correctness: all GEMV verification suites (gated, vs fp32 ref)."""
    s = _settings(None, None, image, None)
    rc = 0
    for name in ("gemv-verify", "gemv-verify-b1", "gemv-stages"):
        typer.secho(f"\n── verify: {name} " + "─" * 40, fg=typer.colors.CYAN)
        r = tasks.run_task(s, name, [], image=image, vfa_tree=vfa_tree,
                           dry_run=dry_run)
        if r != 0:
            typer.secho(f"  {name} FAILED (exit {r})", fg=typer.colors.RED)
            rc = r
            break
    if rc == 0:
        typer.secho("\n  all GEMV verification suites PASS",
                    fg=typer.colors.GREEN)
    raise typer.Exit(rc)


# ── plot / history ──────────────────────────────────────────────────────────

@app.command()
def plot(concurrency: int = typer.Option(1, "--concurrency"),
         context: int = typer.Option(8192, "--context")):
    """Render the README charts (kernel.png, fa4-vs-fi.png, history.png)."""
    layout = load_layout()
    try:
        plots.render_all(layout, concurrency, context)
    except RuntimeError as e:
        typer.secho(str(e), fg=typer.colors.RED, err=True)
        raise typer.Exit(1)


@app.command()
def history(limit: int = typer.Option(20, "--limit")):
    """Show the latest benchmark history rows."""
    layout = load_layout()
    from mjolnir.history import load_records
    recs = load_records(layout.history_file)[-limit:]
    if not recs:
        typer.secho("no history yet — run: mjolnir bench perf",
                    fg=typer.colors.YELLOW)
        return
    for r in recs:
        n = len(r.get("cells", []))
        gate = (r.get("gate") or {}).get("clean")
        typer.echo(f"  {r['ts']:<25} {r['backend']:<12} "
                   f"{r.get('image', ''):<45} cells={n} "
                   f"runs={r.get('runs')} gate={gate}")


@app.command()
def version():
    """Print the version."""
    typer.echo(f"mjolnir {__version__}")


if __name__ == "__main__":
    app()
