"""``mjolnir hw setup`` — Jetson AGX Thor host pre-configuration.

The ported Thor HW setup (``scripts/hw/setup-thor.sh``, from the ~/thor
scripts) brings a box to the serving baseline the rest of the CLI
assumes: headless boot target, apt upgrade, Docker with the NVIDIA
runtime as default, pip, jtop, a 32 GB swap file (zRAM off), the fan
profiles (``recommended`` + ``max`` installed, default selected — the
default profile is ``recommended``), trimmed host services, locked max
clocks, and the MAXN power mode. Every step is idempotent.

This module only locates and runs the script: flags map 1:1 to its
options, and ``--dry-run`` previews the plan without touching the box.
The script needs sudo — it asks on the first privileged call.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from mjolnir.config import find_repo_root

# The --skip names, in run order (mirror of the script's steps).
STEPS: tuple[str, ...] = (
    "gui",
    "upgrade",
    "pip",
    "docker",
    "jtop",
    "memory",
    "fan",
    "host",
    "clocks",
    "power",
)

FAN_PROFILES: tuple[str, ...] = ("recommended", "max", "cool", "quiet")
DEFAULT_FAN_PROFILE = "recommended"


def setup_script() -> Path:
    """The ported setup script (the repo's ``scripts/hw/setup-thor.sh``)."""
    p = find_repo_root() / "scripts" / "hw" / "setup-thor.sh"
    if not p.is_file():
        raise FileNotFoundError(f"setup script not found: {p}")
    return p


def build_cmd(
    yes: bool = False,
    upgrade: bool = True,
    keep_gui: bool = False,
    fan_profile: str = DEFAULT_FAN_PROFILE,
    swap_size: int = 32,
    skip: str = "",
    reboot: bool = False,
    dry_run: bool = False,
) -> list[str]:
    """The ``bash scripts/hw/setup-thor.sh ...`` argv for the given flags."""
    if fan_profile not in FAN_PROFILES:
        raise ValueError(f"bad --fan-profile '{fan_profile}' (want one of: {', '.join(FAN_PROFILES)})")
    names = [s.strip() for s in skip.split(",") if s.strip()]
    bad = sorted(set(names) - set(STEPS))
    if bad:
        raise ValueError(f"unknown --skip step(s): {', '.join(bad)} (valid: {', '.join(STEPS)})")

    cmd = ["bash", str(setup_script())]
    if yes:
        cmd.append("--yes")
    if not upgrade:
        cmd.append("--no-upgrade")
    if keep_gui:
        cmd.append("--keep-gui")
    cmd += ["--fan-profile", fan_profile]
    cmd += ["--swap-size", str(swap_size)]
    if names:
        cmd += ["--skip", ",".join(names)]
    if reboot:
        cmd.append("--reboot")
    if dry_run:
        cmd.append("--dry-run")
    return cmd


def run_setup(
    *,
    yes: bool = False,
    upgrade: bool = True,
    keep_gui: bool = False,
    fan_profile: str = DEFAULT_FAN_PROFILE,
    swap_size: int = 32,
    skip: str = "",
    reboot: bool = False,
    dry_run: bool = False,
) -> int:
    """Run the Thor setup script; return its exit code (0 = done)."""
    return subprocess.call(
        build_cmd(
            yes=yes,
            upgrade=upgrade,
            keep_gui=keep_gui,
            fan_profile=fan_profile,
            swap_size=swap_size,
            skip=skip,
            reboot=reboot,
            dry_run=dry_run,
        )
    )
