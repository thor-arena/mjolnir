"""``mjolnir image``: pick, list, build and gate-check the vLLM image."""

from __future__ import annotations

import subprocess
import sys
from typing import Optional

import typer

from mjolnir_cli import picker, tasks
from mjolnir_cli.commands._common import HELP_CTX, load_state, picker_group, save_state, settings, state_file
from mjolnir_cli.config import DEFAULT_IMAGE, load_layout

image_app = typer.Typer(
    help="Pick the active vLLM image / build the patched one.",
    context_settings=HELP_CTX,
    cls=picker_group(lambda: image_use(None)),
)


# ── image ───────────────────────────────────────────────────────────────────


def _local_images() -> list[str]:
    """Every local docker image (``repo:tag``), deduped, order preserved."""
    try:
        p = subprocess.run(
            ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"], capture_output=True, text=True, check=True
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []
    out: list[str] = []
    for line in p.stdout.splitlines():
        line = line.strip()
        if line and line != "<none>:<none>" and line not in out:
            out.append(line)
    return out


def _active_image(st: dict) -> str:
    return st.get("image") or settings(None, None, None, None).image


@image_app.command("use")
def image_use(image: Optional[str] = typer.Argument(None, help="docker image reference (omit to pick interactively)")):
    """Remember the active vLLM image (bare ``mjolnir image`` or no args
    here = arrow-key picker over local docker images).

    Persisted to the state file ($MJOLNIR_STATE, default
    ~/.mjolnir-state.json) — serve/bench pick it up from there."""
    st = load_state()
    imgs = _local_images()
    if not imgs:
        typer.secho("no local docker images found — build or pull one first", fg=typer.colors.RED, err=True)
        raise typer.Exit(4)
    if image:
        if image not in imgs:
            typer.secho(
                f"image '{image}' not found locally ({len(imgs)} image(s) available: mjolnir image list)",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(4)
        chosen = image
    else:
        if not sys.stdin.isatty():
            for i in imgs:
                typer.echo(f"  {'●' if i == _active_image(st) else ' '}{i}")
            typer.secho("non-interactive session — pass the image explicitly", fg=typer.colors.RED, err=True)
            raise typer.Exit(4)
        try:
            chosen = picker.select("Pick the vLLM image:", imgs, default=_active_image(st))
        except picker.Cancelled:
            typer.secho("cancelled — active image unchanged", fg=typer.colors.YELLOW, err=True)
            raise typer.Exit(1)
    save_state({"image": chosen})
    typer.secho(f"active image: {chosen}", fg=typer.colors.GREEN)
    typer.echo(f"  state: {state_file()}")


@image_app.command("list")
def image_list():
    """List local docker images (● = the active one)."""
    st = load_state()
    active = _active_image(st)
    for i in _local_images():
        if i == active:
            typer.secho(f"  ● {i}  (active)", fg=typer.colors.GREEN)
        else:
            typer.echo(f"    {i}")


@image_app.command("build")
def image_build(
    tag: Optional[str] = typer.Option(
        None,
        "--tag",
        help=f"image tag (default: {DEFAULT_IMAGE} — the repo + the vLLM version the Dockerfile builds on)",
    ),
    dry_run: bool = typer.Option(False, "--dry-run"),
):
    """Build the Mjolnir vLLM image: the patch stack + the mjolnir-kernel
    package from this checkout (repo-root build context).

    The default tag is always the canonical one (IMAGE_REPO:v<vLLM version>),
    never the remembered active image — that may be an older build."""
    tag = tag or DEFAULT_IMAGE
    try:
        root = load_layout().require_repo("image build")
    except FileNotFoundError as e:
        typer.secho(f"{e} — or pull the published image: docker pull {DEFAULT_IMAGE}", fg=typer.colors.RED, err=True)
        raise typer.Exit(4)
    rev = subprocess.run(["git", "-C", str(root), "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    cmd = ["docker", "build", "-f", str(root / "docker" / "vllm-thor" / "Dockerfile"), "-t", tag]
    if rev.returncode == 0 and rev.stdout.strip():
        cmd += ["--build-arg", f"MJOLNIR_REVISION={rev.stdout.strip()}"]
    cmd.append(str(root))
    if dry_run:
        print(" ".join(cmd))
        return
    rc = subprocess.call(cmd)
    if rc == 0:
        typer.secho(f"built {tag} — canary:  mjolnir image gates --image {tag}", fg=typer.colors.GREEN)
        active = settings(None, None, None, None).image
        if active != tag:
            typer.secho(
                f"note: the active image is {active} — make the new build active:  mjolnir image use {tag}",
                fg=typer.colors.YELLOW,
            )
    raise typer.Exit(rc)


@image_app.command("gates")
def image_gates(image: Optional[str] = typer.Option(None, "--image"), dry_run: bool = typer.Option(False, "--dry-run")):
    """Run the sm_110 gate-probe canaries against the image (fresh container,
    no server needed)."""
    s = settings(None, None, image, None)
    raise typer.Exit(tasks.run_task(s, "gates", [], dry_run=dry_run))
