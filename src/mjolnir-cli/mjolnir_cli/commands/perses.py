"""``mjolnir perses``: drive the Perses + Prometheus stack (``mjolnir_cli.perses``)."""

from __future__ import annotations

import json
from typing import Optional

import typer

from mjolnir_cli import perses
from mjolnir_cli.commands._common import HELP_CTX, settings

perses_app = typer.Typer(
    help="The Perses + Prometheus observability stack for the vLLM server — "
    "independent of the vLLM server: 'perses down' leaves the server "
    "running, and 'serve down' leaves the stack up. Prometheus scrapes "
    "the server's /metrics (host.docker.internal + the resolved port); "
    "Perses serves the two vLLM dashboards (vendored from "
    "vllm-project/vllm, docker/perses/). Data: "
    "~/.local/share/mjolnir/perses/.",
    no_args_is_help=True,
    context_settings=HELP_CTX,
)


# ── perses ──────────────────────────────────────────────────────────────────


@perses_app.command("up")
def perses_up(
    port: Optional[int] = typer.Option(
        None, "--port", help=f"host port of the Perses UI (default {perses.DEFAULT_PORT})"
    ),
    wait: bool = typer.Option(True, "--wait/--no-wait", help="wait for the Perses API before returning"),
    wait_timeout: float = typer.Option(300.0, "--wait-timeout"),
    dry_run: bool = typer.Option(False, "--dry-run"),
):
    """Start the observability stack (prometheus + perses) with the two vLLM
    dashboards provisioned. The vLLM scrape target = the resolved server port
    (state/$MJOLNIR_PORT — the same one `mjolnir serve` uses; override with
    PERSER_VLLM_TARGET in the perses env file). The server may be up or down."""
    s = settings(None, None, None, None)
    try:
        perses.up(s, port=port, wait=wait, wait_timeout_s=wait_timeout, dry_run=dry_run)
    except (perses.PersesError, FileNotFoundError) as e:
        typer.secho(str(e), fg=typer.colors.RED, err=True)
        raise typer.Exit(4 if "not found" in str(e) else 1)
    typer.secho(
        f"\n  dashboards:  mjolnir perses status  (UI: http://127.0.0.1:{port or perses.DEFAULT_PORT})",
        fg=typer.colors.GREEN,
    )


@perses_app.command("down")
def perses_down(
    purge: bool = typer.Option(False, "--purge", help="also wipe the stack data (prometheus TSDB + perses db)"),
    dry_run: bool = typer.Option(False, "--dry-run"),
):
    """Stop the stack (the vLLM server keeps running; data kept by default,
    --purge wipes the prometheus TSDB + the perses db)."""
    raise typer.Exit(perses.down(purge=purge, dry_run=dry_run))


@perses_app.command("status")
def perses_status(port: Optional[int] = typer.Option(None, "--port")):
    """Container state + API health + the vLLM scrape target + the data dir."""
    s = settings(None, None, None, None)
    typer.echo(json.dumps(perses.status(s, port), indent=2))


@perses_app.command("logs")
def perses_logs(
    follow: bool = typer.Option(False, "--follow", "-f"),
    tail: int = typer.Option(100, "--tail"),
    service: Optional[str] = typer.Option(None, "--service", help="perses (default) | prometheus"),
):
    """Tail a stack service's logs (default: perses)."""
    raise typer.Exit(perses.logs(follow=follow, tail=tail, service=service))
