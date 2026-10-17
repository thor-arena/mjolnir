"""What every ``mjolnir`` command module shares: the console, the help
context, the bare-group picker, the state file (the active model / config /
image) and settings resolution.

The bottom layer of ``commands/``: it imports no other ``commands`` module."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from mjolnir_cli.config import Settings, resolve

console = Console()

HELP_CTX = {"help_option_names": ["-h", "--help"]}


def picker_group(bare_fn) -> type[typer.core.TyperGroup]:
    """A group that, when invoked bare (no subcommand), runs ``bare_fn``
    (the arrow-key picker) instead of erroring out."""

    class PickerGroup(typer.core.TyperGroup):
        def invoke(self, ctx):
            if not ctx._protected_args:
                bare_fn()
                return
            return super().invoke(ctx)

    return PickerGroup


def state_file() -> Path:
    return Path(os.environ.get("MJOLNIR_STATE", str(Path.home() / ".mjolnir-state.json"))).expanduser()


def load_state() -> dict:
    p = state_file()
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:  # noqa: BLE001
            pass
    return {}


def save_state(patch: dict) -> None:
    """Merge ``patch`` into the state file (the active model/config/image)."""
    st = load_state()
    st.update(patch)
    state_file().parent.mkdir(parents=True, exist_ok=True)
    state_file().write_text(json.dumps(st, indent=2))


def settings(model: Optional[str], quant: Optional[str], image: Optional[str], port: Optional[int]) -> Settings:
    st = load_state()
    return resolve(model or st.get("model"), quant or st.get("quant"), image or st.get("image"), port)
