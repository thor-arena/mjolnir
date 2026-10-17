"""Root commands that keep a measurement window clean: the request-queue
gate (``mjolnir gate``) and the host page cache (``mjolnir grant-permissions``,
``mjolnir drop-caches``)."""

from __future__ import annotations

from typing import Optional

import typer

from mjolnir_cli import cachectl
from mjolnir_cli.commands._common import settings

# ── gate ────────────────────────────────────────────────────────────────────


def gate(
    port: Optional[int] = typer.Option(None, "--port"),
    once: bool = typer.Option(False, "--once", help="check now and exit"),
    confirm: int = typer.Option(6, "--confirm"),
    timeout: float = typer.Option(3600.0, "--timeout"),
):
    """The clean-window gate (vLLM request queue 0/0 × N consecutive samples)."""
    s = settings(None, None, None, port)
    from mjolnir_cli.tasks import run_gate

    args = ["--once"] if once else ["--confirm", str(confirm), "--timeout", str(timeout)]
    raise typer.Exit(run_gate(s, args, False))


# ── permissions / cache ─────────────────────────────────────────────────────


def grant_permissions(
    yes: bool = typer.Option(False, "--yes", "-y", help="don't prompt to confirm the install"),
):
    """Check for — and, if missing, install — the NOPASSWD sudo grant the
    kernel benches use to clear the page cache.

    Idempotent: a present grant is left untouched. The install adds only the
    cache-clear command (``sysctl -w vm.drop_caches=3``) — never full sudo —
    and asks for the sudo password exactly once."""
    raise typer.Exit(cachectl.grant_drop_caches(assume_yes=yes))


def drop_caches():
    """Clear the host page cache (``sysctl -w vm.drop_caches=3``) for a clean
    kernel-bench baseline. Uses the NOPASSWD grant (run
    ``mjolnir grant-permissions`` if it's missing); never prompts."""
    if not cachectl.drop_caches_granted():
        typer.secho(
            "drop-caches grant missing — run: mjolnir grant-permissions",
            fg=typer.colors.YELLOW,
            err=True,
        )
        raise typer.Exit(4)
    rc = cachectl.drop_caches()
    if rc == 0:
        typer.secho("page cache cleared (vm.drop_caches=3)", fg=typer.colors.GREEN)
    else:
        typer.secho(f"drop_caches failed (exit {rc})", fg=typer.colors.RED, err=True)
    raise typer.Exit(rc)
