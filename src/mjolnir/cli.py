"""The ``mjolnir`` CLI — one command surface for the whole repo.

    mjolnir setup                       pre-configure the Thor host hardware
    mjolnir serve up|down|status|logs    drive the vLLM container
    mjolnir model [use|list]            active model/config (bare = arrow-key
                                        picker over configs/)
    mjolnir image [use|list|build|gates] active vLLM image (bare = arrow-key
                                        picker over local docker images)
    mjolnir vfa prepare                 build the GEMV'd vllm_flash_attn tree
    mjolnir gate                        the clean-window gate
    mjolnir bench perf|ab|kernel        end-to-end perf / A/B / kernel benches
    mjolnir verify                      kernel correctness suites
    mjolnir plot|history                charts + the raw-data log
    mjolnir litellm up|down|status|logs  the LiteLLM proxy stack (independent
                                        of the vLLM server)
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import typer
from rich import box
from rich.console import Console
from rich.style import Style
from rich.table import Table

from mjolnir import __version__
from mjolnir.config import (BACKEND_LABELS, BASELINE_QUANT, DEFAULT_QUANT,
                            Settings, ConfigEntry, load_layout, resolve,
                            scan_all_configs, user_configs_dir)
from mjolnir import benchy, dockerctl, litellm, plots, picker, tasks, thor, vfa

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
  mjolnir serve up        serve the model (baked-in defaults: Qwen3.8-27B
                          NVFP4 + FA4 GEMV, port 6001)
  mjolnir serve status    container + health + request queue
  mjolnir bench perf      gated end-to-end throughput sweep -> history + charts
  mjolnir verify          GEMV kernel correctness suites (clean-window gated)
  mjolnir model / image   pick another model config / serving image
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


serve_app = typer.Typer(help="Drive the vLLM container (the patched image).",
                        no_args_is_help=True, context_settings=_HELP_CTX)
bench_app = typer.Typer(help="Benchmarks — gated, raw data, chartable.",
                        no_args_is_help=True, context_settings=_HELP_CTX)
model_app = typer.Typer(help="Pick the active model / config.",
                        context_settings=_HELP_CTX,
                        cls=_picker_group(lambda: model_use(None, None)))
image_app = typer.Typer(help="Pick the active vLLM image / build the patched one.",
                        context_settings=_HELP_CTX,
                        cls=_picker_group(lambda: image_use(None)))
vfa_app = typer.Typer(help="The GEMV'd vllm_flash_attn tree (kernel iteration).",
                      no_args_is_help=True, context_settings=_HELP_CTX)
litellm_app = typer.Typer(
    help="The LiteLLM proxy stack (proxy + postgres + redis) — independent "
         "of the vLLM server: 'litellm down' leaves vLLM running, and "
         "'serve down' leaves the proxy up. Reads your template from the "
         "litellm data dir, ~/.local/share/mjolnir/litellm/ (see "
         "docker/litellm/).",
     no_args_is_help=True, context_settings=_HELP_CTX)
hw_app = typer.Typer(
    help="Thor host hardware setup — one-time, idempotent provisioning steps "
         "that run on the host (sudo where needed), e.g. the fan-profile "
         "install. 'hw setup' must run before any fan-mode selection or "
         "benchmarking.",
    no_args_is_help=True, context_settings=_HELP_CTX)

app = typer.Typer(
    context_settings=_HELP_CTX,
    cls=_root_group(),
    help="Mjolnir — vLLM on Jetson Thor (sm_110): patched image, FA4 GEMV "
         "decode kernel, one-command serve/bench/verify CLI.")
app.add_typer(serve_app, name="serve")
app.add_typer(bench_app, name="bench")
app.add_typer(model_app, name="model")
app.add_typer(image_app, name="image")
app.add_typer(vfa_app, name="vfa")
app.add_typer(litellm_app, name="litellm")
app.add_typer(hw_app, name="hw")


def _state_file() -> Path:
    return Path(os.environ.get(
        "MJOLNIR_STATE", str(Path.home() / ".mjolnir-state.json"))).expanduser()


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


def _settings(model: Optional[str], quant: Optional[str],
              image: Optional[str], port: Optional[int],
              gemv: Optional[bool] = None) -> Settings:
    st = _load_state()
    s = resolve(model or st.get("model"), quant or st.get("quant"),
                image or st.get("image"), port, gemv)
    return s


# ── setup ───────────────────────────────────────────────────────────────────

@app.command("setup")
def setup_cmd(yes: bool = typer.Option(False, "--yes", "-y",
                                      help="don't prompt to continue"),
              upgrade: bool = typer.Option(True, "--upgrade/--no-upgrade",
                                           help="apt update + full-upgrade "
                                                "(default on — the original "
                                                "script's NVIDIA-carrier path)"),
              keep_gui: bool = typer.Option(False, "--keep-gui",
                                            help="keep the graphical boot "
                                                 "target (default: headless "
                                                 "multi-user)"),
              fan_profile: str = typer.Option("cool", "--fan-profile",
                                              help="cool|quiet (default "
                                                   "cool; applied only if "
                                                   "nvfancontrol is "
                                                   "configured)"),
              swap_size: int = typer.Option(32, "--swap-size",
                                            help="swap file size in GB "
                                                 "(default 32 — the 128 GB "
                                                 "Thor SoM)"),
              skip: str = typer.Option("", "--skip",
                                       help="comma-separated steps to skip: "
                                            "gui,upgrade,pip,docker,jtop,"
                                            "memory,fan,host,clocks,power"),
              reboot: bool = typer.Option(False, "--reboot",
                                         help="reboot at the end if a step "
                                              "required one (non-interactive)"),
              dry_run: bool = typer.Option(False, "--dry-run",
                                          help="print the plan, change "
                                              "nothing")):
    """Pre-configure the Thor host (port of the ~/thor HW setup script).

    Runs ``scripts/setup-thor.sh``: headless boot target, apt upgrade,
    Docker + NVIDIA default runtime, pip, jtop, 32 GB swap (zRAM off),
    fan profile, service cleanup, locked max clocks, MAXN power mode.
    Idempotent — safe to re-run; the script asks for sudo."""
    try:
        rc = thor.run_setup(yes=yes, upgrade=upgrade, keep_gui=keep_gui,
                            fan_profile=fan_profile, swap_size=swap_size,
                            skip=skip, reboot=reboot, dry_run=dry_run)
    except (ValueError, FileNotFoundError) as e:
        typer.secho(str(e), fg=typer.colors.RED, err=True)
        raise typer.Exit(4)
    raise typer.Exit(rc)


# ── serve ───────────────────────────────────────────────────────────────────

@serve_app.command("up")
def serve_up(model: Optional[str] = typer.Option(None, "--model"),
              config: Optional[str] = typer.Option(None, "--config",
                                                  help="config quant name, "
                                                       "e.g. NVFP4_FA4hd256"),
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

def _config_table(entries: list[ConfigEntry],
                  active_model, active_quant) -> Table:
    t = Table(box=box.SIMPLE_HEAD, pad_edge=False)
    t.add_column("", justify="center")
    t.add_column("model", style="bold cyan")
    t.add_column("quant", style="green")
    t.add_column("backend")
    t.add_column("where")
    for e in entries:
        active = e.model == active_model and e.quant == active_quant
        t.add_row("[green]●[/green]" if active else "",
                  e.model, e.quant, e.backend, e.source,
                  style=Style(bold=True) if active else None)
    return t


def _config_labels(entries: list[ConfigEntry]) -> list[str]:
    return [f"{e.model} / {e.quant}" for e in entries]


def _pick_config(entries: list[ConfigEntry], st: dict,
                 candidates: list[ConfigEntry] | None = None,
                 title: str = "Pick the active model / config:") -> ConfigEntry:
    """Arrow-key pick over the candidate configs (default: the active one)."""
    pool = candidates or entries
    labels = _config_labels(pool)
    cur = st.get("model") and st.get("quant")
    default = f"{cur[0]} / {cur[1]}" if cur and f"{cur[0]} / {cur[1]}" in labels \
        else None
    if not sys.stdin.isatty():
        _console.print(_config_table(entries, st.get("model"), st.get("quant")))
        typer.secho("non-interactive session — pass model + quant "
                    "explicitly", fg=typer.colors.RED, err=True)
        raise typer.Exit(4)
    try:
        pick = picker.select(title, labels, default=default)
    except picker.Cancelled:
        typer.secho("cancelled — active config unchanged",
                    fg=typer.colors.YELLOW, err=True)
        raise typer.Exit(1)
    return pool[labels.index(pick)]


@model_app.command("use")
def model_use(model: Optional[str] = typer.Argument(None, help="vendor/Model, "
             "e.g. Qwen3.8-27B (omit to pick interactively)"),
              quant: Optional[str] = typer.Argument(None, help="config name, "
             "e.g. NVFP4_FA4hd256 (omit to pick interactively)")):
    """Remember the active model/config (bare ``mjolnir model`` or no args
    here = arrow-key picker over configs: repo configs/ + user-local
    ~/.local/share/mjolnir/configs).

    Persisted to a small state file ($MJOLNIR_STATE, default
    ~/.mjolnir-state.json) — not an env file."""
    layout = load_layout()
    entries = scan_all_configs(layout)
    if not entries:
        typer.secho(f"no configs found under {layout.repo_root / 'configs'} "
                    f"or {user_configs_dir()}",
                    fg=typer.colors.RED, err=True)
        raise typer.Exit(4)
    st = _load_state()

    if model and quant:
        if not any(e.model == model and e.quant == quant for e in entries):
            typer.secho(f"config not found: {model}/{quant}.yaml "
                        f"(available: mjolnir model list)",
                        fg=typer.colors.RED, err=True)
            raise typer.Exit(4)
        chosen = next(e for e in entries
                      if e.model == model and e.quant == quant)
    elif model or quant:
        candidates = [e for e in entries
                      if (not model or e.model == model)
                      and (not quant or e.quant == quant)]
        if not candidates:
            typer.secho("no config matches the given model/quant "
                        "(available: mjolnir model list)",
                        fg=typer.colors.RED, err=True)
            raise typer.Exit(4)
        if len(candidates) == 1:
            chosen = candidates[0]
        else:
            chosen = _pick_config(entries, st, candidates)
    else:
        chosen = _pick_config(entries, st)

    _save_state({"model": chosen.model, "quant": chosen.quant})
    typer.secho(f"active: {chosen.model} / {chosen.quant}  "
                f"(→ {BACKEND_LABELS.get(chosen.quant, chosen.quant)})",
                fg=typer.colors.GREEN)
    typer.echo(f"  state: {_state_file()}")


@model_app.command("list")
def model_list(as_json: bool = typer.Option(False, "--json", "-j",
                                            help="emit JSON (for scripts)")):
    """List every config (repo configs/ + the user-local dir; ● = the active
    one; the state file is printed below the table)."""
    layout = load_layout()
    entries = scan_all_configs(layout)
    st = _load_state()
    if as_json:
        typer.echo(json.dumps(
            {"active": {"model": st.get("model"), "quant": st.get("quant")},
             "state_file": str(_state_file()),
             "local_configs_dir": str(user_configs_dir()),
             "configs": [{"model": e.model, "quant": e.quant,
                           "backend": e.backend,
                           "source": e.source,
                           "path": str(e.path)} for e in entries]},
            indent=2))
        return
    if not entries:
        typer.secho("no configs found", fg=typer.colors.YELLOW)
        return
    _console.print(_config_table(entries, st.get("model"), st.get("quant")))
    if any(e.source == "local" for e in entries):
        typer.echo(f"    local = {user_configs_dir()}")
    if st.get("model"):
        typer.secho(f"  ● active: {st['model']} / {st['quant']}",
                    fg=typer.colors.GREEN)
        typer.echo(f"    state: {_state_file()}  (change: mjolnir model)")
    else:
        typer.echo("  no active config yet — run 'mjolnir model' to pick one")


# ── image ───────────────────────────────────────────────────────────────────

def _local_images() -> list[str]:
    """Every local docker image (``repo:tag``), deduped, order preserved."""
    try:
        p = subprocess.run(["docker", "images", "--format",
                            "{{.Repository}}:{{.Tag}}"],
                           capture_output=True, text=True, check=True)
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
def image_use(image: Optional[str] = typer.Argument(None,
              help="docker image reference (omit to pick interactively)")):
    """Remember the active vLLM image (bare ``mjolnir image`` or no args
    here = arrow-key picker over local docker images).

    Persisted to the state file ($MJOLNIR_STATE, default
    ~/.mjolnir-state.json) — serve/bench pick it up from there."""
    st = _load_state()
    imgs = _local_images()
    if not imgs:
        typer.secho("no local docker images found — build or pull one first",
                    fg=typer.colors.RED, err=True)
        raise typer.Exit(4)
    if image:
        if image not in imgs:
            typer.secho(f"image '{image}' not found locally "
                        f"({len(imgs)} image(s) available: mjolnir image list)",
                        fg=typer.colors.RED, err=True)
            raise typer.Exit(4)
        chosen = image
    else:
        if not sys.stdin.isatty():
            for i in imgs:
                typer.echo(f"  {'●' if i == _active_image(st) else ' '}{i}")
            typer.secho("non-interactive session — pass the image "
                        "explicitly", fg=typer.colors.RED, err=True)
            raise typer.Exit(4)
        try:
            chosen = picker.select("Pick the vLLM image:", imgs,
                                   default=_active_image(st))
        except picker.Cancelled:
            typer.secho("cancelled — active image unchanged",
                        fg=typer.colors.YELLOW, err=True)
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

@vfa_app.command("prepare")
def vfa_prepare(image: Optional[str] = typer.Option(None, "--image"),
                out: Optional[Path] = typer.Option(None, "--out")):
    """Prepare the GEMV'd vllm_flash_attn tree (image tree + kernel + diff)."""
    s = _settings(None, None, image, None)
    try:
        vfa.prepare_vfa_tree(s, out)
    except RuntimeError as e:
        typer.secho(str(e), fg=typer.colors.RED, err=True)
        raise typer.Exit(1)


# ── litellm ─────────────────────────────────────────────────────────────────

@litellm_app.command("up")
def litellm_up(model: Optional[str] = typer.Option(None, "--model"),
               config: Optional[str] = typer.Option(None, "--config",
                                                   help="config quant name, "
                                                        "e.g. NVFP4_FA4hd256"),
               port: Optional[int] = typer.Option(None, "--port",
                                                  help=f"host port of the "
                                                       f"proxy (default "
                                                       f"{litellm.DEFAULT_PROXY_PORT})"),
               wait: bool = typer.Option(True, "--wait/--no-wait",
                                         help="wait for the proxy's "
                                              "liveliness before returning"),
               wait_timeout: float = typer.Option(600.0, "--wait-timeout"),
               dry_run: bool = typer.Option(False, "--dry-run")):
    """Render the config from your template (in the litellm data dir,
    ~/.local/share/mjolnir/litellm/) with the currently selected
    model/config, and start the stack (proxy + postgres + redis). The vLLM
    server is reached via the host — it may be up or down."""
    s = _settings(model, config, None, None)
    try:
        litellm.up(s, proxy_port=port, wait=wait,
                   wait_timeout_s=wait_timeout, dry_run=dry_run)
    except (litellm.LitellmError, FileNotFoundError) as e:
        typer.secho(str(e), fg=typer.colors.RED, err=True)
        raise typer.Exit(4 if "not found" in str(e) else 1)
    typer.secho("\n  proxy:  mjolnir litellm status", fg=typer.colors.GREEN)


@litellm_app.command("down")
def litellm_down(purge: bool = typer.Option(False, "--purge",
                                            help="also wipe the stack data "
                                                 "(postgres + redis)"),
                 dry_run: bool = typer.Option(False, "--dry-run")):
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
def litellm_logs(follow: bool = typer.Option(False, "--follow", "-f"),
                 tail: int = typer.Option(100, "--tail")):
    """Tail the proxy container logs."""
    raise typer.Exit(litellm.logs(follow=follow, tail=tail))


@litellm_app.command("config")
def litellm_config(model: Optional[str] = typer.Option(None, "--model"),
                   config: Optional[str] = typer.Option(None, "--config"),
                   show: bool = typer.Option(False, "--show",
                                             help="print the rendered "
                                                  "config instead of just "
                                                  "validating it")):
    """Render the template with the current model/config (no containers) —
    validates the variables and the YAML."""
    s = _settings(model, config, None, None)
    try:
        p = litellm.render_config(s)
    except (litellm.LitellmError, FileNotFoundError) as e:
        typer.secho(str(e), fg=typer.colors.RED, err=True)
        raise typer.Exit(4 if "not found" in str(e) else 1)
    typer.secho(f"rendered ok: {p}  (model: {s.served_model})",
                fg=typer.colors.GREEN)
    if show:
        typer.echo(p.read_text())


# ── hw ─────────────────────────────────────────────────────────────────────

@hw_app.command("setup")
def hw_setup(dry_run: bool = typer.Option(
        False, "--dry-run",
        help="transform on a scratch copy and show the diff — no sudo, no "
             "writes to /etc")):
    """One-time, idempotent fan-profile install.

    Injects the `recommended` and `max` fan profiles into
    /etc/nvfancontrol.conf (each gated — never duplicated if already
    present), dumps the untouched conf to /etc/nvfancontrol.conf.bck on
    first run (the .bck always holds the original — the rollback point),
    then selects the fan mode `recommended` (instead of the stock `cool`)
    and restarts nvfancontrol. Run this before any fan-mode selection or
    benchmarking."""
    script = load_layout().repo_root / "scripts" / "hw" / "fan-profiles.sh"
    if not script.exists():
        typer.secho(f"setup script not found: {script}", fg=typer.colors.RED,
                    err=True)
        raise typer.Exit(4)
    args = ["bash", str(script)] + (["--dry-run"] if dry_run else [])
    raise typer.Exit(subprocess.call(args))


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


def _ab_legs(specs: str, default_model: str, default_image: str,
             entries: list[ConfigEntry]) -> list[tuple[str, str, str]]:
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
                raise ValueError(
                    f"bad leg '{spec}' — expected '<config>', "
                    f"'<model>/<config>' or '<image>:<config>'")
        else:
            image = None
        if "/" in spec:
            model, _, quant = spec.rpartition("/")
            if not model or not quant or model.count("/") != 1:
                raise ValueError(
                    f"bad leg '{spec}' — expected '<config>', "
                    f"'<model>/<config>' or '<image>:<config>'")
            if (model, quant) not in model_quants:
                raise ValueError(
                    f"unknown model config '{model}/{quant}' — "
                    f"available configs: mjolnir model list")
            legs.append((model, image or default_image, quant))
        else:
            if spec not in quants:
                raise ValueError(
                    f"unknown config '{spec}' for model {default_model} — "
                    f"available configs: mjolnir model list")
            legs.append((default_model, image or default_image, spec))
    if not legs:
        raise ValueError("--backends is empty")
    return legs


@bench_app.command("ab")
def bench_ab(backends: str = typer.Option(f"{DEFAULT_QUANT},{BASELINE_QUANT}",
                                          "--backends",
                                          help="comma-separated legs: each is "
                                               "'<config>' (the --model/state "
                                               "model, served from --image), "
                                               "'<image>:<config>', "
                                               "'<model>/<config>' or "
                                               "'<image>:<model>/<config>' — "
                                               "available: mjolnir model list"),
             runs: int = typer.Option(5, "--runs"),
             repeat: int = typer.Option(2, "--repeat"),
             gate: bool = typer.Option(True, "--gate/--no-gate"),
             model: Optional[str] = typer.Option(None, "--model"),
             image: Optional[str] = typer.Option(None, "--image",
                                                 help="image for bare-config "
                                                      "legs (default: "
                                                      "$MJOLNIR_IMAGE / "
                                                      "built-in default)"),
             port: Optional[int] = typer.Option(None, "--port")):
    """Headline A/B: restart the server per leg, gated perf bench for each,
    one history row-set per leg.

    Each leg is a model config (quant), optionally model-qualified and/or
    image-pinned: '<config>', '<image>:<config>', '<model>/<config>' or
    '<image>:<model>/<config>'. The image carries the backend (kernel stack
    is baked in); the config (configs/<model>/<quant>.yaml, repo or local)
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
    for leg_model, leg_image, quant in legs:
        s.model = leg_model
        s.image = leg_image
        s.quant = quant
        typer.secho(f"\n{'=' * 62}\n  A/B leg: {leg_model}/{quant}  "
                    f"(image {leg_image})\n"
                    f"{'=' * 62}", fg=typer.colors.CYAN)
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


@bench_app.command("kernel",
                   # Pass-through: the task's own flags (--out, --mode,
                   # --no-gate, …) are unknown to the launcher, so ignore
                   # them here and collect them into `args` to forward
                   # verbatim to the script.
                   context_settings={**_HELP_CTX,
                                     "ignore_unknown_options": True,
                                     "allow_extra_args": True})
def bench_kernel(task: str = typer.Argument(..., help="task name — see "
                                                     "mjolnir bench kernel "
                                                     "--help-list"),
                 args: List[str] = typer.Argument(None, help="script args "
                                                            "(passed "
                                                             "through)"),
                  image: Optional[str] = typer.Option(None, "--image"),
                  vfa_tree: Optional[Path] = typer.Option(None, "--vfa-tree"),
                  port: Optional[int] = typer.Option(None, "--port",
                                                     help="live vLLM server "
                                                          "port — the "
                                                          "clean-window gate "
                                                          "polls "
                                                          "127.0.0.1:<port>/metrics"),
                  dry_run: bool = typer.Option(False, "--dry-run"),
                  skip_preflight: bool = typer.Option(False, "--skip-preflight"),
                  help_list: bool = typer.Option(False, "--help-list",
                                                help="list the tasks")):
    """Run a kernel verification / benchmark (gated docker task)."""
    if help_list:
        for t in tasks.TASKS:
            typer.echo(f"  {t.name:<15} [{t.kind:<5}]  {t.desc}")
        raise typer.Exit(0)
    s = _settings(None, None, image, port)
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
    concs = sorted({cc for v in by.values() for cc in v},
                   key=lambda x: (x is None, x or 0))
    t = Table(box=box.SIMPLE_HEAD, pad_edge=False)
    t.add_column("ctx")
    for cc in concs:
        t.add_column(f"c={cc} · t/s")
    for ctx in sorted(by, key=lambda x: (x is None, x or 0)):
        t.add_row(_ctx_label(ctx),
                  *[f"{by[ctx][cc]:.1f}" if cc in by[ctx] else "—"
                    for cc in concs])
    return t


@app.command()
def history(limit: int = typer.Option(5, "--limit",
                                     help="how many recent runs to show"),
           as_json: bool = typer.Option(False, "--json", "-j",
                                        help="emit the raw JSON records "
                                             "(for scripts)")):
    """Bench results at a glance: decode throughput (t/s) per recent run.

    The raw substrate (every sweep, every cell) is committed in
    benchmarks/history.jsonl — ``--json`` dumps the records."""
    layout = load_layout()
    from mjolnir.history import load_records
    recs = load_records(layout.history_file)[-limit:]
    if not recs:
        typer.secho("no history yet — run: mjolnir bench perf",
                    fg=typer.colors.YELLOW)
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
