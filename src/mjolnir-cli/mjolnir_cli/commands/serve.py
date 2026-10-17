"""``mjolnir serve``: drive the vLLM container (start, stop, status, logs)."""

from __future__ import annotations

import json
from typing import Optional

import typer

from mjolnir_cli import dockerctl, perses
from mjolnir_cli.commands._common import HELP_CTX, settings

serve_app = typer.Typer(
    help="Drive the vLLM container (the patched image).", no_args_is_help=True, context_settings=HELP_CTX
)


# ── serve ───────────────────────────────────────────────────────────────────


@serve_app.command("up")
def serve_up(
    model: Optional[str] = typer.Option(None, "--model"),
    config: Optional[str] = typer.Option(None, "--config", help="config quant name, e.g. NVFP4_FA4hd256"),
    image: Optional[str] = typer.Option(None, "--image"),
    port: Optional[int] = typer.Option(None, "--port"),
    wait: bool = typer.Option(True, "--wait/--no-wait", help="wait for /health before returning"),
    wait_timeout: float = typer.Option(1800.0, "--wait-timeout"),
    dry_run: bool = typer.Option(False, "--dry-run"),
):
    """Start the vLLM server (custom image, config from --model/--config)."""
    s = settings(model, config, image, port)
    if not s.config_path.exists():
        typer.secho(f"config not found: {s.config_path}", fg=typer.colors.RED, err=True)
        raise typer.Exit(4)
    stale = dockerctl.stale_image(s.image)
    if stale:
        typer.secho(f"warning: {stale}", fg=typer.colors.YELLOW, err=True)
    try:
        dockerctl.serve_up(s, wait=wait, wait_timeout_s=wait_timeout, dry_run=dry_run)
    except (dockerctl.DockerError, TimeoutError) as e:
        typer.secho(str(e), fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    typer.secho(f"\n  serving {s.model} ({s.quant}, {s.backend_label}) at {s.base_url}", fg=typer.colors.GREEN)
    typer.echo("  gate:  mjolnir gate --once")
    typer.echo("  bench: mjolnir bench perf   ·   kernel: mjolnir verify sweep")


@serve_app.command("down")
def serve_down(
    model: Optional[str] = typer.Option(None, "--model"),
    config: Optional[str] = typer.Option(None, "--config"),
    image: Optional[str] = typer.Option(None, "--image"),
    port: Optional[int] = typer.Option(None, "--port"),
    dry_run: bool = typer.Option(False, "--dry-run"),
):
    """Stop the vLLM container."""
    s = settings(model, config, image, port)
    dockerctl.serve_down(s, dry_run=dry_run)


@serve_app.command("status")
def serve_status(port: Optional[int] = typer.Option(None, "--port")):
    """Container state + /health + current queue load."""
    s = settings(None, None, None, port)
    st = dockerctl.status(s)
    typer.echo(json.dumps(st, indent=2))
    from mjolnir_cli.gate import server_load

    load = server_load(s.metrics_url)
    if load is not None:
        typer.echo(f"  queue: running={load[0]:g} waiting={load[1]:g}")
    pport = perses.running_port()
    if pport:
        typer.echo(f"  dashboard: http://127.0.0.1:{pport} (mjolnir perses status)")


@serve_app.command("logs")
def serve_logs(
    port: Optional[int] = typer.Option(None, "--port"),
    follow: bool = typer.Option(False, "--follow", "-f"),
    tail: int = typer.Option(100, "--tail"),
):
    """Tail the vLLM container logs."""
    s = settings(None, None, None, port)
    raise typer.Exit(dockerctl.logs(s, follow=follow, tail=tail))
