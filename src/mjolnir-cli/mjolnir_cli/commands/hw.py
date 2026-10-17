"""``mjolnir hw``: provision the Thor host and show its fan profiles."""

from __future__ import annotations

import os
import re
from pathlib import Path

import typer

from mjolnir_cli import thor
from mjolnir_cli.commands._common import HELP_CTX

hw_app = typer.Typer(
    help="Thor host pre-configuration — one-time, idempotent provisioning "
    "(sudo where needed): headless boot target, apt upgrade, Docker + "
    "NVIDIA default runtime, pip, jtop, swap, fan profiles "
    "(recommended + max installed, default selected), service "
    "cleanup, locked clocks, MAXN power mode. Run 'hw setup' before "
    "serving or benchmarking.",
    no_args_is_help=True,
    context_settings=HELP_CTX,
)


# ── hw ─────────────────────────────────────────────────────────────────────


@hw_app.command("setup")
def hw_setup(
    yes: bool = typer.Option(False, "--yes", "-y", help="don't prompt to continue"),
    upgrade: bool = typer.Option(
        True,
        "--upgrade/--no-upgrade",
        help="apt update + full-upgrade (default on — the original script's NVIDIA-carrier path)",
    ),
    keep_gui: bool = typer.Option(
        False, "--keep-gui", help="keep the graphical boot target (default: headless multi-user)"
    ),
    fan_profile: str = typer.Option(
        "recommended",
        "--fan-profile",
        help="recommended (default) | max | cool | quiet — recommended/max are installed by this step if missing",
    ),
    swap_size: int = typer.Option(32, "--swap-size", help="swap file size in GB (default 32 — the 128 GB Thor SoM)"),
    skip: str = typer.Option(
        "", "--skip", help="comma-separated steps to skip: gui,upgrade,pip,docker,jtop,memory,fan,host,clocks,power"
    ),
    reboot: bool = typer.Option(False, "--reboot", help="reboot at the end if a step required one (non-interactive)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="print the plan, change nothing"),
):
    """Pre-configure the Thor host (port of the ~/thor HW setup script).

    Runs ``scripts/hw/setup-thor.sh``: headless boot target, apt upgrade,
    Docker + NVIDIA default runtime, pip, jtop, 32 GB swap (zRAM off),
    the fan profiles (``recommended`` + ``max`` installed per-profile
    gated, with the .bck rollback backup; default selected —
    ``recommended`` by default), service cleanup, locked max clocks,
    MAXN power mode. Idempotent — safe to re-run; the script asks for
    sudo."""
    try:
        rc = thor.run_setup(
            yes=yes,
            upgrade=upgrade,
            keep_gui=keep_gui,
            fan_profile=fan_profile,
            swap_size=swap_size,
            skip=skip,
            reboot=reboot,
            dry_run=dry_run,
        )
    except (ValueError, FileNotFoundError) as e:
        typer.secho(str(e), fg=typer.colors.RED, err=True)
        raise typer.Exit(4)
    raise typer.Exit(rc)


@hw_app.command("status")
def hw_status():
    """Installed fan profiles and the selected default (read-only, no sudo).

    The daemon's live state (which profile it is actually running) needs
    sudo — check it with `sudo nvfancontrol -q`."""
    conf = Path(os.environ.get("MJOLNIR_NVFANCONF", "/etc/nvfancontrol.conf"))
    if not conf.exists():
        typer.secho(f"no conf at {conf}", fg=typer.colors.RED, err=True)
        raise typer.Exit(4)
    text = conf.read_text()
    profiles = re.findall(r"^[ \t]*FAN_PROFILE\s+(\w+)", text, re.M)
    m = re.search(r"^[ \t]*FAN_DEFAULT_PROFILE\s+(\S+)", text, re.M)
    bck = conf.with_name(conf.name + ".bck")
    print(f"conf:     {conf}")
    print(f"profiles: {', '.join(profiles)}")
    print(f"default:  {m.group(1) if m else '<unset>'}")
    print(f"backup:   {bck}" + (" (exists)" if bck.exists() else " (missing)"))
