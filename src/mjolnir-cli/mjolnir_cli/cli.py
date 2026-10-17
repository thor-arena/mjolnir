"""The ``mjolnir`` CLI — one command surface for the whole repo.

mjolnir hw setup|status             pre-configure the Thor host (HW
                                    provisioning, idempotent)
mjolnir serve up|down|status|logs    drive the vLLM container
mjolnir model [use|list]            active model/config (bare = arrow-key
                                    picker over configs/)
mjolnir image [use|list|build|gates] active vLLM image (bare = arrow-key
                                    picker over local docker images)
mjolnir verify <stage>              kernel correctness + ncu benchmarks (Mjolnir vs
                                    FlashInfer vs FA2) -> benchmarks/raw/verify
mjolnir autoresearch ...            the autonomous kernel-research loop (OpenCode
                                    agents, every experiment on the ledger)
mjolnir gate                        the clean-window gate
mjolnir bench perf|ab               end-to-end perf / A/B (vLLM server)
mjolnir plot|history                charts (+ README.md with --readme) + the raw-data log
mjolnir litellm up|down|status|logs  the LiteLLM proxy stack (independent
                                    of the vLLM server)
mjolnir perses up|down|status|logs   the Perses + Prometheus observability
                                    stack (independent of the vLLM server)
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Optional

import typer
from rich import box
from rich.console import Console
from rich.style import Style
from rich.table import Table

from mjolnir_cli import __version__
from mjolnir_cli import legs as legs_
from mjolnir_cli.e2e import MAX_CTX as _E2E_MAX_CTX
from mjolnir_cli.config import (
    BACKEND_LABELS,
    DEFAULT_IMAGE,
    DEFAULT_QUANT,
    E2E_DEPTHS,
    STOCK_IMAGE,
    VLLM_VERSION,
    Settings,
    ConfigEntry,
    load_layout,
    resolve,
    configs_root,
    scan_all_configs,
    user_configs_dir,
)
from mjolnir_cli import benchy, cachectl, charts, clidoc, dockerctl, litellm, perses, picker, readme, tasks, thor
from mjolnir_cli.verify.cli import app as verify_app
from mjolnir_cli.autoresearch import app as autoresearch_app

_console = Console()

_HELP_CTX = {"help_option_names": ["-h", "--help"]}


def _picker_group(bare_fn) -> type[typer.core.TyperGroup]:
    """A group that, when invoked bare (no subcommand), runs ``bare_fn``
    (the arrow-key picker) instead of erroring out."""

    class PickerGroup(typer.core.TyperGroup):
        def invoke(self, ctx):
            if not ctx._protected_args:
                bare_fn()
                return
            return super().invoke(ctx)

    return PickerGroup


_NEXT_STEPS = """
Next steps:
  mjolnir serve up         serve the model (baked-in defaults: Qwen3.8-27B NVFP4,
                           --attention-backend MJOLNIR, port 6001)
  mjolnir serve status     container + health + request queue
  mjolnir verify sweep     Mjolnir vs FlashInfer vs FA2 at every L (ncu) -> raw data
  mjolnir verify all       correctness + every benchmark stage -> charts + README
  mjolnir bench ab         end-to-end A/B: Mjolnir vs FlashInfer vs FA2 configs
  mjolnir model / image    pick another model config / serving image
"""


def _root_group() -> type[typer.core.TyperGroup]:
    """The root group: bare ``mjolnir`` prints the help plus a next-steps
    guide instead of just the help."""

    class RootGroup(typer.core.TyperGroup):
        def invoke(self, ctx):
            if not ctx._protected_args:
                ctx.command.get_help(ctx)
                print(_NEXT_STEPS, flush=True)
                return
            return super().invoke(ctx)

    return RootGroup


serve_app = typer.Typer(
    help="Drive the vLLM container (the patched image).", no_args_is_help=True, context_settings=_HELP_CTX
)
bench_app = typer.Typer(
    help="Benchmarks — gated, raw data, chartable.", no_args_is_help=True, context_settings=_HELP_CTX
)
model_app = typer.Typer(
    help="Pick the active model / config.", context_settings=_HELP_CTX, cls=_picker_group(lambda: model_use(None, None))
)
image_app = typer.Typer(
    help="Pick the active vLLM image / build the patched one.",
    context_settings=_HELP_CTX,
    cls=_picker_group(lambda: image_use(None)),
)
litellm_app = typer.Typer(
    help="The LiteLLM proxy stack (proxy + postgres + redis) — independent "
    "of the vLLM server: 'litellm down' leaves vLLM running, and "
    "'serve down' leaves the proxy up. Reads your template from the "
    "litellm data dir, ~/.local/share/mjolnir/litellm/ (see "
    "docker/litellm/).",
    no_args_is_help=True,
    context_settings=_HELP_CTX,
)
perses_app = typer.Typer(
    help="The Perses + Prometheus observability stack for the vLLM server — "
    "independent of the vLLM server: 'perses down' leaves the server "
    "running, and 'serve down' leaves the stack up. Prometheus scrapes "
    "the server's /metrics (host.docker.internal + the resolved port); "
    "Perses serves the two vLLM dashboards (vendored from "
    "vllm-project/vllm, docker/perses/). Data: "
    "~/.local/share/mjolnir/perses/.",
    no_args_is_help=True,
    context_settings=_HELP_CTX,
)
hw_app = typer.Typer(
    help="Thor host pre-configuration — one-time, idempotent provisioning "
    "(sudo where needed): headless boot target, apt upgrade, Docker + "
    "NVIDIA default runtime, pip, jtop, swap, fan profiles "
    "(recommended + max installed, default selected), service "
    "cleanup, locked clocks, MAXN power mode. Run 'hw setup' before "
    "serving or benchmarking.",
    no_args_is_help=True,
    context_settings=_HELP_CTX,
)

app = typer.Typer(
    context_settings=_HELP_CTX,
    cls=_root_group(),
    help="Mjolnir — a decode-attention kernel for Jetson AGX Thor (sm_110) and the "
    "vLLM image that serves it: serve / verify / bench / plot.",
)
app.add_typer(serve_app, name="serve")
app.add_typer(bench_app, name="bench")
app.add_typer(model_app, name="model")
app.add_typer(image_app, name="image")
app.add_typer(verify_app, name="verify")
app.add_typer(autoresearch_app, name="autoresearch")
app.add_typer(litellm_app, name="litellm")
app.add_typer(perses_app, name="perses")
app.add_typer(hw_app, name="hw")


def _state_file() -> Path:
    return Path(os.environ.get("MJOLNIR_STATE", str(Path.home() / ".mjolnir-state.json"))).expanduser()


def _load_state() -> dict:
    p = _state_file()
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:  # noqa: BLE001
            pass
    return {}


def _save_state(patch: dict) -> None:
    """Merge ``patch`` into the state file (the active model/config/image)."""
    st = _load_state()
    st.update(patch)
    _state_file().parent.mkdir(parents=True, exist_ok=True)
    _state_file().write_text(json.dumps(st, indent=2))


def _settings(model: Optional[str], quant: Optional[str], image: Optional[str], port: Optional[int]) -> Settings:
    st = _load_state()
    return resolve(model or st.get("model"), quant or st.get("quant"), image or st.get("image"), port)


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
    s = _settings(model, config, image, port)
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
    s = _settings(model, config, image, port)
    dockerctl.serve_down(s, dry_run=dry_run)


@serve_app.command("status")
def serve_status(port: Optional[int] = typer.Option(None, "--port")):
    """Container state + /health + current queue load."""
    s = _settings(None, None, None, port)
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
    s = _settings(None, None, None, port)
    raise typer.Exit(dockerctl.logs(s, follow=follow, tail=tail))


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
        _console.print(_config_table(entries, st.get("model"), st.get("quant")))
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
    st = _load_state()

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

    _save_state({"model": chosen.model, "quant": chosen.quant})
    typer.secho(
        f"active: {chosen.model} / {chosen.quant}  (→ {BACKEND_LABELS.get(chosen.quant, chosen.quant)})",
        fg=typer.colors.GREEN,
    )
    typer.echo(f"  state: {_state_file()}")


@model_app.command("list")
def model_list(as_json: bool = typer.Option(False, "--json", "-j", help="emit JSON (for scripts)")):
    """List every config (repo configs/ + the user-local dir; ● = the active
    one; the state file is printed below the table)."""
    layout = load_layout()
    entries = scan_all_configs(layout)
    st = _load_state()
    if as_json:
        typer.echo(
            json.dumps(
                {
                    "active": {"model": st.get("model"), "quant": st.get("quant")},
                    "state_file": str(_state_file()),
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
    _console.print(_config_table(entries, st.get("model"), st.get("quant")))
    if any(e.source == "local" for e in entries):
        typer.echo(f"    local = {user_configs_dir()}")
    if st.get("model"):
        typer.secho(f"  ● active: {st['model']} / {st['quant']}", fg=typer.colors.GREEN)
        typer.echo(f"    state: {_state_file()}  (change: mjolnir model)")
    else:
        typer.echo("  no active config yet — run 'mjolnir model' to pick one")


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
    return st.get("image") or _settings(None, None, None, None).image


@image_app.command("use")
def image_use(image: Optional[str] = typer.Argument(None, help="docker image reference (omit to pick interactively)")):
    """Remember the active vLLM image (bare ``mjolnir image`` or no args
    here = arrow-key picker over local docker images).

    Persisted to the state file ($MJOLNIR_STATE, default
    ~/.mjolnir-state.json) — serve/bench pick it up from there."""
    st = _load_state()
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
    _save_state({"image": chosen})
    typer.secho(f"active image: {chosen}", fg=typer.colors.GREEN)
    typer.echo(f"  state: {_state_file()}")


@image_app.command("list")
def image_list():
    """List local docker images (● = the active one)."""
    st = _load_state()
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
        active = _settings(None, None, None, None).image
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
    s = _settings(None, None, image, None)
    raise typer.Exit(tasks.run_task(s, "gates", [], dry_run=dry_run))


# ── litellm ─────────────────────────────────────────────────────────────────


@litellm_app.command("up")
def litellm_up(
    model: Optional[str] = typer.Option(None, "--model"),
    config: Optional[str] = typer.Option(None, "--config", help="config quant name, e.g. NVFP4_FA4hd256"),
    port: Optional[int] = typer.Option(
        None, "--port", help=f"host port of the proxy (default {litellm.DEFAULT_PROXY_PORT})"
    ),
    wait: bool = typer.Option(True, "--wait/--no-wait", help="wait for the proxy's liveliness before returning"),
    wait_timeout: float = typer.Option(600.0, "--wait-timeout"),
    dry_run: bool = typer.Option(False, "--dry-run"),
):
    """Render the config from your template (in the litellm data dir,
    ~/.local/share/mjolnir/litellm/) with the currently selected
    model/config, and start the stack (proxy + postgres + redis). The vLLM
    server is reached via the host — it may be up or down."""
    s = _settings(model, config, None, None)
    try:
        litellm.up(s, proxy_port=port, wait=wait, wait_timeout_s=wait_timeout, dry_run=dry_run)
    except (litellm.LitellmError, FileNotFoundError) as e:
        typer.secho(str(e), fg=typer.colors.RED, err=True)
        raise typer.Exit(4 if "not found" in str(e) else 1)
    typer.secho("\n  proxy:  mjolnir litellm status", fg=typer.colors.GREEN)


@litellm_app.command("down")
def litellm_down(
    purge: bool = typer.Option(False, "--purge", help="also wipe the stack data (postgres + redis)"),
    dry_run: bool = typer.Option(False, "--dry-run"),
):
    """Stop the stack (the vLLM server keeps running; data kept by
    default, --purge wipes postgres + redis state)."""
    raise typer.Exit(litellm.down(purge=purge, dry_run=dry_run))


@litellm_app.command("status")
def litellm_status(port: Optional[int] = typer.Option(None, "--port")):
    """Container state + proxy liveliness + the rendered config path."""
    s = _settings(None, None, None, None)
    try:
        st = litellm.status(s, port)
    except FileNotFoundError as e:
        typer.secho(str(e), fg=typer.colors.RED, err=True)
        raise typer.Exit(4)
    typer.echo(json.dumps(st, indent=2))


@litellm_app.command("logs")
def litellm_logs(follow: bool = typer.Option(False, "--follow", "-f"), tail: int = typer.Option(100, "--tail")):
    """Tail the proxy container logs."""
    raise typer.Exit(litellm.logs(follow=follow, tail=tail))


@litellm_app.command("config")
def litellm_config(
    model: Optional[str] = typer.Option(None, "--model"),
    config: Optional[str] = typer.Option(None, "--config"),
    show: bool = typer.Option(False, "--show", help="print the rendered config instead of just validating it"),
):
    """Render the template with the current model/config (no containers) —
    validates the variables and the YAML."""
    s = _settings(model, config, None, None)
    try:
        p = litellm.render_config(s)
    except (litellm.LitellmError, FileNotFoundError) as e:
        typer.secho(str(e), fg=typer.colors.RED, err=True)
        raise typer.Exit(4 if "not found" in str(e) else 1)
    typer.secho(f"rendered ok: {p}  (model: {s.served_model})", fg=typer.colors.GREEN)
    if show:
        typer.echo(p.read_text())


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
    s = _settings(None, None, None, None)
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
    s = _settings(None, None, None, None)
    typer.echo(json.dumps(perses.status(s, port), indent=2))


@perses_app.command("logs")
def perses_logs(
    follow: bool = typer.Option(False, "--follow", "-f"),
    tail: int = typer.Option(100, "--tail"),
    service: Optional[str] = typer.Option(None, "--service", help="perses (default) | prometheus"),
):
    """Tail a stack service's logs (default: perses)."""
    raise typer.Exit(perses.logs(follow=follow, tail=tail, service=service))


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


# ── gate ────────────────────────────────────────────────────────────────────


@app.command()
def gate(
    port: Optional[int] = typer.Option(None, "--port"),
    once: bool = typer.Option(False, "--once", help="check now and exit"),
    confirm: int = typer.Option(6, "--confirm"),
    timeout: float = typer.Option(3600.0, "--timeout"),
):
    """The clean-window gate (vLLM request queue 0/0 × N consecutive samples)."""
    s = _settings(None, None, None, port)
    from mjolnir_cli.tasks import _run_gate

    args = ["--once"] if once else ["--confirm", str(confirm), "--timeout", str(timeout)]
    raise typer.Exit(_run_gate(s, args, False))


# ── calibrate (fp6 KV scales) ───────────────────────────────────────────────


@app.command()
def calibrate(
    model: Optional[str] = typer.Option(None, "--model", help="vendor/Model (default: the active one)"),
    config: Optional[str] = typer.Option(None, "--config", help="the config to calibrate (default: the active one)"),
    image: Optional[str] = typer.Option(None, "--image", help="the Mjolnir image"),
    out: Optional[Path] = typer.Option(
        None,
        "--out",
        help="default: the model's kv_scales.json next to the config dirs — kv_scales.json inside the config "
        "dir for a config whose own drafter (DSpark) runs on Mjolnir",
    ),
    prompts: int = typer.Option(48, "--prompts", help="short prompts"),
    long: int = typer.Option(6, "--long", help="long-context prompts"),
    max_model_len: int = typer.Option(32768, "--max-model-len"),
    draft_vocab: bool = typer.Option(
        False,
        "--draft-vocab",
        help="instead: count the target's generated tokens -> draft_vocab.json (VocabTrim, MJOLNIR_DRAFT_VOCAB)",
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="print the docker command, run nothing"),
):
    """Calibrate per-(layer, KV head, channel) K/V scales for the fp6 (e2m3) KV cache
    (or, with --draft-vocab, the drafter's vocabulary for VocabTrim).

    Runs the config's model on fp8 KV in the Mjolnir image, records the
    per-channel amax of every K/V written to the cache over a varied prompt set,
    and writes ``kv_scales.json`` next to the config — what an
    ``kv-cache-dtype: fp6_e2m3`` config is served with. Needs the GPU to
    itself: refuses while the vLLM server is up (stop it with
    ``mjolnir serve down`` in a window you own)."""
    s = _settings(model, config, image, None)
    if not dry_run and dockerctl.container_running():
        typer.secho(
            "the vLLM server is running — calibration loads the model itself. Stop the server "
            "first (mjolnir serve down), in a window you own.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(3)
    target = out or (
        dockerctl.kv_scales_host_path(s).with_name("draft_vocab.json")
        if draft_vocab
        else dockerctl.kv_scales_host_path(s)
    )
    if not os.access(target.parent if target.parent.exists() else Path.home(), os.W_OK):
        from mjolnir_cli.config import user_configs_dir

        target = user_configs_dir() / s.model / target.name
    target.parent.mkdir(parents=True, exist_ok=True)
    mounts = [(str(target.parent.resolve()), "/out")]
    env = []
    try:  # a checkout's kernel package over the image's (calibrate.py may be newer)
        from mjolnir_cli.config import find_repo_root

        src = find_repo_root() / "src" / "mjolnir"
        if (src / "mjolnir" / ("draftvocab.py" if draft_vocab else "calibrate.py")).exists():
            mounts.append((str(src), "/opt/mjolnir-src:ro"))
            env.append("PYTHONPATH=/opt/mjolnir-src")
    except FileNotFoundError:
        pass
    cmd = (
        f"python3 -m mjolnir.{'draftvocab' if draft_vocab else 'calibrate'} "
        f"--config {dockerctl.config_model_dir(s)}/{s.quant}/config.yaml --out /out/{target.name} "
        f"--download-dir /data/models/huggingface --max-model-len {max_model_len} "
        f"--prompts {prompts} --long {long}"
    )
    argv = dockerctl.build_run_args(
        s,
        name="mjolnir-calibrate",
        detach=False,
        command=cmd,
        extra_env=tuple(env),
        extra_mounts=tuple(mounts),
        chown=(f"/out/{target.name}",),
    )
    if dry_run:
        typer.echo(" ".join(argv))
        return
    typer.secho(f"calibrating {s.model}/{s.quant} on {s.image} -> {target}", fg=typer.colors.CYAN)
    rc = dockerctl.run_once(argv, "mjolnir-calibrate")
    if rc == 0:
        typer.secho(f"wrote {target}", fg=typer.colors.GREEN)
    raise typer.Exit(rc)


# ── permissions / cache ─────────────────────────────────────────────────────


@app.command("grant-permissions")
def grant_permissions(
    yes: bool = typer.Option(False, "--yes", "-y", help="don't prompt to confirm the install"),
):
    """Check for — and, if missing, install — the NOPASSWD sudo grant the
    kernel benches use to clear the page cache.

    Idempotent: a present grant is left untouched. The install adds only the
    cache-clear command (``sysctl -w vm.drop_caches=3``) — never full sudo —
    and asks for the sudo password exactly once."""
    raise typer.Exit(cachectl.grant_drop_caches(assume_yes=yes))


@app.command("drop-caches")
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
    s = _settings(model, config, image, port)
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
    s = _settings(model, None, image, port)
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
    _run_legs(
        s,
        layout,
        [(m, i, q, None) for m, i, q in legs],
        runs=runs,
        repeat=repeat,
        gate=gate,
        depths=[int(x) for x in depths.replace(" ", ",").split(",") if x.strip()],
    )
    _rerender(layout)


@bench_app.command(
    "accuracy",
    context_settings={**_HELP_CTX, "allow_extra_args": True, "ignore_unknown_options": True},
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

    s = _settings(model, None, image, port)
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
    _print_accuracy(json.loads(out.read_text()))
    typer.secho(f"wrote {out}", fg=typer.colors.GREEN)
    if plot:
        _rerender(layout)


def _print_accuracy(doc: dict) -> None:
    from mjolnir_cli import accuracy

    rows = doc.get("legs", [])
    suites = accuracy.suites_in([{"suites": r.get("suites", {})} for r in rows])
    t = Table(box=box.SIMPLE_HEAD, title=f"accuracy · run {doc['run_id']} · tool-eval-bench {' '.join(doc['args'])}")
    t.add_column("config")
    for n in suites:
        t.add_column(f"{accuracy.suite_label(n)} %", justify="right")
    t.add_column("completion", justify="right")
    bases = accuracy.baselines(rows)  # a run may hold several models: each vs its own
    for r, base in zip(rows, bases):
        cells = []
        for n in suites:
            v = (r.get("suites", {}).get(n) or {}).get("score")
            cells.append("—" if v is None else f"{v:.1f}{accuracy.fmt_delta(accuracy.delta(r, base, n))}")
        comp = [x.get("completion_rate") for x in r.get("suites", {}).values() if x.get("completion_rate") is not None]
        t.add_row(r.get("label", r.get("id", "")), *cells, f"{min(comp):.0f}%" if comp else "—")
    if accuracy.any_excluded(rows, bases, suites):
        t.caption = accuracy.EXCLUDED_NOTE
    Console().print(t)


def _rerender(layout) -> None:
    try:
        charts.render_all(layout)
    except RuntimeError as e:
        typer.secho(f"(plots skipped: {e})", fg=typer.colors.YELLOW, err=True)
    try:
        readme.render_readme(layout)
    except (RuntimeError, FileNotFoundError) as e:
        typer.secho(f"(readme skipped: {e})", fg=typer.colors.YELLOW, err=True)


def _run_legs(s: Settings, layout, legs: list[tuple], **perf) -> None:
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


def _print_compare(
    layout,
    mjolnir_config: Optional[str] = None,
    as_json: bool = False,
    official_config: Optional[str] = None,
    model: Optional[str] = None,
) -> int:
    from mjolnir_cli import e2e

    off, mj = e2e.latest_pair(layout.history_file, mjolnir_config, official_config, model)
    if not off or not mj:
        missing = "the official vLLM image" if not off else "a Mjolnir image"
        typer.secho(f"no run on {missing} in {layout.history_file} — run: mjolnir bench e2e", fg=typer.colors.YELLOW)
        return 4
    rows = e2e.compare(off, mj)
    if as_json:
        typer.echo(
            json.dumps({"official": e2e.side_label(off), "mjolnir": e2e.side_label(mj), "cells": rows}, indent=2)
        )
        return 0
    con = Console()
    t = Table(
        box=box.SIMPLE_HEAD,
        title="official vLLM image → Mjolnir image (gain: + = Mjolnir better)",
        caption=f"official: {e2e.side_label(off)}\nmjolnir:  {e2e.side_label(mj)}\nboth: {e2e.spec_label(off, mj)}"
        + e2e.acceptance_line(off, mj),
    )
    t.add_column("ctx", justify="right")
    t.add_column("c", justify="right")
    for m in e2e.METRICS:
        t.add_column(
            f"{m.label} {m.unit} {'↑' if m.higher_better else '↓'} ({'throughput' if m.higher_better else 'latency'} change)",
            justify="right",
        )
    for r in rows:
        cells = []
        for m in e2e.METRICS:
            a, b, g = r[m.key]
            if a is None or b is None:
                cells.append("—")
                continue
            color = "green" if g >= e2e.E2E_NOISE_PCT else ("red" if g <= -e2e.E2E_NOISE_PCT else "grey50")
            cells.append(
                f"{m.fmt.format(a)} → {m.fmt.format(b)} [{color}]{e2e.pct(g if m.higher_better else -g)}[/{color}]"
            )
        t.add_row(e2e.ctx_label(r["ctx"]), str(r["conc"]), *cells)
    con.print(t)
    h = e2e.hero(off, mj)
    if h:
        con.print(
            f"[bold]1 request @ {e2e.ctx_label(h['ctx'])}:[/bold] decode {e2e.pct(h['tg_tps'][2])} · "
            f"prefill {e2e.pct(h['pp_tps'][2])} · TTFT {e2e.pct(-h['ttft_ms'][2])} · ITL {e2e.pct(-h['itl_ms'][2])} "
            "(throughput: % more tokens/s; latency: % change in ms)"
        )
    return 0


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
    s = _settings(model, None, None, port)
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
    _run_legs(
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
    _print_compare(layout, config, official_config=official_config, model=s.model)
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
    _rerender(layout)


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
    model = _settings(model, None, None, None).model
    raise typer.Exit(_print_compare(load_layout(), config, as_json, official_config, model))


# ── plot / history ──────────────────────────────────────────────────────────


@app.command()
def plot(
    render_readme: bool = typer.Option(
        False,
        "--readme",
        help="also render README.md from the README.md.j2 template "
        "(every table + number computed from the committed raw data)",
    ),
    readme_template: Optional[Path] = typer.Option(
        None, "--readme-template", help="template override (default: <repo>/README.md.j2)"
    ),
    readme_out: Optional[Path] = typer.Option(None, "--readme-out", help="output override (default: <repo>/README.md)"),
    site: Optional[Path] = typer.Option(
        None, "--site", help="also build the interactive benchmark site (index.html) into this directory"
    ),
):
    """Render every chart (light + dark PNGs in assets/benchmarks/, from the
    latest default `mjolnir verify` run of each stage + the e2e history).
    --readme also renders README.md and docs/**/*.md from their .md.j2
    templates; --site builds the interactive GitHub Pages site. Everything is
    computed from the committed raw data."""
    layout = load_layout()
    try:
        charts.render_all(layout)
    except RuntimeError as e:
        typer.secho(str(e), fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    if layout.repo_root:  # research/progress.png: the live ledger, else the newest exported run
        from mjolnir_cli import autoresearch

        autoresearch._render_progress(layout.repo_root, autoresearch.latest_rows(layout.repo_root))
    if render_readme:
        try:
            readme.render_readme(layout, template=readme_template, out=readme_out)
            print(f"[mjolnir] wrote {clidoc.write_reference(layout.repo_root, app)}")
        except (RuntimeError, FileNotFoundError, OSError) as e:
            typer.secho(str(e), fg=typer.colors.RED, err=True)
            raise typer.Exit(1)
    if site is not None:
        from mjolnir_cli import interactive

        interactive.build(layout, site)


def _ctx_label(ctx: int | None) -> str:
    if not ctx:
        return "0"
    return f"{ctx // 1024}K" if ctx % 1024 == 0 else str(ctx)


def _history_record_table(r: dict) -> Table:
    """One history record → a small tg-t/s (decode throughput) table."""
    by: dict[int | None, dict] = {}
    for c in r.get("cells", []):
        tg = (c.get("tg_tps") or {}).get("mean")
        if tg is None:
            continue
        by.setdefault(c.get("context"), {})[c.get("concurrency")] = tg
    concs = sorted({cc for v in by.values() for cc in v}, key=lambda x: (x is None, x or 0))
    t = Table(box=box.SIMPLE_HEAD, pad_edge=False)
    t.add_column("ctx")
    for cc in concs:
        t.add_column(f"c={cc} · t/s")
    for ctx in sorted(by, key=lambda x: (x is None, x or 0)):
        t.add_row(_ctx_label(ctx), *[f"{by[ctx][cc]:.1f}" if cc in by[ctx] else "—" for cc in concs])
    return t


@app.command()
def history(
    limit: int = typer.Option(5, "--limit", help="how many recent runs to show"),
    as_json: bool = typer.Option(False, "--json", "-j", help="emit the raw JSON records (for scripts)"),
):
    """Bench results at a glance: decode throughput (t/s) per recent run.

    The raw substrate (every sweep, every cell) is committed in
    benchmarks/history.jsonl — ``--json`` dumps the records."""
    layout = load_layout()
    from mjolnir_cli.history import load_records

    recs = load_records(layout.history_file)[-limit:]
    if not recs:
        typer.secho("no history yet — run: mjolnir bench perf", fg=typer.colors.YELLOW)
        return
    if as_json:
        typer.echo(json.dumps(recs, indent=2))
        return
    for r in recs:
        ts = str(r.get("ts", ""))[:16].replace("T", " ")
        _console.print(f"  {ts}   {r.get('backend', '')} — {r.get('image', '')}")
        _console.print(_history_record_table(r))
        _console.print()


@app.command()
def version():
    """Print the version."""
    typer.echo(f"mjolnir {__version__}")


if __name__ == "__main__":
    app()
