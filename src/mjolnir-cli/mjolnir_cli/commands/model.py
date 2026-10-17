"""``mjolnir model``: pick and list the active model / config."""

from __future__ import annotations

import json
import sys
from typing import Optional

import typer
from rich import box
from rich.style import Style
from rich.table import Table

from mjolnir_cli import picker
from mjolnir_cli.commands._common import HELP_CTX, console, load_state, picker_group, save_state, state_file
from mjolnir_cli.config import (
    BACKEND_LABELS,
    ConfigEntry,
    configs_root,
    load_layout,
    scan_all_configs,
    user_configs_dir,
)

model_app = typer.Typer(
    help="Pick the active model / config.", context_settings=HELP_CTX, cls=picker_group(lambda: model_use(None, None))
)


# ── model ───────────────────────────────────────────────────────────────────


def _config_table(entries: list[ConfigEntry], active_model, active_quant) -> Table:
    t = Table(box=box.SIMPLE_HEAD, pad_edge=False)
    t.add_column("", justify="center")
    t.add_column("model", style="bold cyan")
    t.add_column("quant", style="green")
    t.add_column("backend")
    t.add_column("where")
    for e in entries:
        active = e.model == active_model and e.quant == active_quant
        t.add_row(
            "[green]●[/green]" if active else "",
            e.model,
            e.quant,
            e.backend,
            e.source,
            style=Style(bold=True) if active else None,
        )
    return t


def _config_labels(entries: list[ConfigEntry]) -> list[str]:
    return [f"{e.model} / {e.quant}" for e in entries]


def _pick_config(
    entries: list[ConfigEntry],
    st: dict,
    candidates: list[ConfigEntry] | None = None,
    title: str = "Pick the active model / config:",
) -> ConfigEntry:
    """Arrow-key pick over the candidate configs (default: the active one)."""
    pool = candidates or entries
    labels = _config_labels(pool)
    cur = st.get("model") and st.get("quant")
    default = f"{cur[0]} / {cur[1]}" if cur and f"{cur[0]} / {cur[1]}" in labels else None
    if not sys.stdin.isatty():
        console.print(_config_table(entries, st.get("model"), st.get("quant")))
        typer.secho("non-interactive session — pass model + quant explicitly", fg=typer.colors.RED, err=True)
        raise typer.Exit(4)
    try:
        pick = picker.select(title, labels, default=default)
    except picker.Cancelled:
        typer.secho("cancelled — active config unchanged", fg=typer.colors.YELLOW, err=True)
        raise typer.Exit(1)
    return pool[labels.index(pick)]


@model_app.command("use")
def model_use(
    model: Optional[str] = typer.Argument(None, help="vendor/Model, e.g. Qwen3.8-27B (omit to pick interactively)"),
    quant: Optional[str] = typer.Argument(None, help="config name, e.g. NVFP4_FA4hd256 (omit to pick interactively)"),
):
    """Remember the active model/config (bare ``mjolnir model`` or no args
    here = arrow-key picker over configs: repo configs/ + user-local
    ~/.local/share/mjolnir/configs).

    Persisted to a small state file ($MJOLNIR_STATE, default
    ~/.mjolnir-state.json) — not an env file."""
    layout = load_layout()
    entries = scan_all_configs(layout)
    if not entries:
        typer.secho(
            f"no configs found under {configs_root()} or {user_configs_dir()}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(4)
    st = load_state()

    if model and quant:
        if not any(e.model == model and e.quant == quant for e in entries):
            typer.secho(
                f"config not found: {model}/{quant} (available: mjolnir model list)", fg=typer.colors.RED, err=True
            )
            raise typer.Exit(4)
        chosen = next(e for e in entries if e.model == model and e.quant == quant)
    elif model or quant:
        candidates = [e for e in entries if (not model or e.model == model) and (not quant or e.quant == quant)]
        if not candidates:
            typer.secho(
                "no config matches the given model/quant (available: mjolnir model list)", fg=typer.colors.RED, err=True
            )
            raise typer.Exit(4)
        if len(candidates) == 1:
            chosen = candidates[0]
        else:
            chosen = _pick_config(entries, st, candidates)
    else:
        chosen = _pick_config(entries, st)

    save_state({"model": chosen.model, "quant": chosen.quant})
    typer.secho(
        f"active: {chosen.model} / {chosen.quant}  (→ {BACKEND_LABELS.get(chosen.quant, chosen.quant)})",
        fg=typer.colors.GREEN,
    )
    typer.echo(f"  state: {state_file()}")


@model_app.command("list")
def model_list(as_json: bool = typer.Option(False, "--json", "-j", help="emit JSON (for scripts)")):
    """List every config (repo configs/ + the user-local dir; ● = the active
    one; the state file is printed below the table)."""
    layout = load_layout()
    entries = scan_all_configs(layout)
    st = load_state()
    if as_json:
        typer.echo(
            json.dumps(
                {
                    "active": {"model": st.get("model"), "quant": st.get("quant")},
                    "state_file": str(state_file()),
                    "local_configs_dir": str(user_configs_dir()),
                    "configs": [
                        {
                            "model": e.model,
                            "quant": e.quant,
                            "backend": e.backend,
                            "source": e.source,
                            "path": str(e.path),
                        }
                        for e in entries
                    ],
                },
                indent=2,
            )
        )
        return
    if not entries:
        typer.secho("no configs found", fg=typer.colors.YELLOW)
        return
    console.print(_config_table(entries, st.get("model"), st.get("quant")))
    if any(e.source == "local" for e in entries):
        typer.echo(f"    local = {user_configs_dir()}")
    if st.get("model"):
        typer.secho(f"  ● active: {st['model']} / {st['quant']}", fg=typer.colors.GREEN)
        typer.echo(f"    state: {state_file()}  (change: mjolnir model)")
    else:
        typer.echo("  no active config yet — run 'mjolnir model' to pick one")
