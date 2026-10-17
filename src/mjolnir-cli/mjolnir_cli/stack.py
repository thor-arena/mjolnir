"""Generic docker-compose stack driver — the shared machinery for every
side-stack the CLI manages (litellm, perses, ...).

A *stack* is a docker-compose project (``docker/<name>/docker-compose.yaml``)
driven by the CLI under the ``mjolnir-<name>`` compose project. Stacks are
*independent* of the vLLM server: ``<name> down`` never touches vLLM, and
``serve down`` never touches a stack. They reach the host (e.g. the vLLM
port) through ``host.docker.internal`` (host-gateway), never through a
shared network.

What lives here (the part every stack would otherwise re-implement):

  * the path conventions — data dir (``$MJOLNIR_DATA/<name>``, overridable
    per stack), env file (``<name>.env`` in the data dir), compose file
    (repo-relative)
  * env-file parsing + the merge order (env file < injected < defaults)
  * the compose plumbing — v2 check, argv building, ``up -d`` (with the
    best-effort rollback on failure), ``down`` (the "nothing to stop" check
    + purge), ``ps`` parsing (the NDJSON quirk), ``logs``
  * envsubst-style template rendering
  * the up-wait (health-URL poll)

Adding a new stack = ``docker/<name>/docker-compose.yaml`` (+ templates) + a
thin ``src/mjolnir/<name>.py`` (a ``StackSpec`` + glue: what to render, the
health URL, the banner) + a typer group in ``commands/<name>.py``, mounted in
``cli.py``.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from mjolnir_cli.config import resource_path

# envsubst-style variables: ${NAME} or bare $NAME
VAR_RE = re.compile(r"\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?")


class StackError(RuntimeError):
    """The compose driver failed (up failed, health poll timed out, ...)."""


@dataclass(frozen=True)
class StackSpec:
    """Declarative description of one compose stack.

    name            the CLI command group + the ``mjolnir-<name>`` project
    compose_rel     the compose file, relative to the repo root
    env_file_var    env var overriding the env-file location
    data_dir_var    env var overriding the data-dir location
    data_dir_env    the env var the compose file interpolates for the data
                    dir (``${...}``); set so the value always reaches compose
                    (defaults to ``data_dir_var``)
    data_subdirs    dirs under the data dir created on up / wiped on
                    ``down --purge``
    port_env        env var holding the host port of the stack's main service
    default_port    its default
    wait_timeout_s  default timeout for the up health-wait
    """

    name: str
    compose_rel: Path
    env_file_var: str = ""
    data_dir_var: str = ""
    data_dir_env: str = ""
    data_subdirs: tuple[str, ...] = ()
    port_env: str = ""
    default_port: int = 0
    wait_timeout_s: float = 600.0

    @property
    def project(self) -> str:
        return f"mjolnir-{self.name}"

    def data_dir(self) -> Path:
        """Stack data root: ``$<data_dir_var>`` or
        ``$MJOLNIR_DATA/<name>`` (default ``~/.local/share/mjolnir/<name>``)."""
        env = self.data_dir_var and os.environ.get(self.data_dir_var)
        if not env:
            mjolnir_data = os.environ.get("MJOLNIR_DATA", str(Path.home() / ".local" / "share" / "mjolnir"))
            env = str(Path(mjolnir_data).expanduser() / self.name)
        return Path(env).expanduser()

    def env_file_path(self) -> Path:
        """The user's env file: ``$<env_file_var>`` or
        ``<data_dir>/<name>.env``."""
        env = self.env_file_var and os.environ.get(self.env_file_var)
        if env:
            return Path(env).expanduser()
        return self.data_dir() / f"{self.name}.env"

    def compose_file(self) -> Path:
        return resource_path(self.compose_rel)


# ── env file + rendering ────────────────────────────────────────────────────


def load_env_file(path: Path) -> dict[str, str]:
    """Parse a ``KEY=VALUE`` env file (comments/blank lines ignored, optional
    single/double quotes around the value)."""
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"":
            v = v[1:-1]
        if k:
            out[k] = v
    return out


def resolve_env(spec: StackSpec, extra: dict[str, str] | None = None) -> dict[str, str]:
    """Merged env for rendering + compose interpolation:
    user env file < extra (injected by the stack's glue) < stack defaults
    (data dir, main port)."""
    env = load_env_file(spec.env_file_path())
    if extra:
        env.update(extra)
    if spec.data_dir_var:
        # the value the compose file interpolates ($LITELLM_DATA_DIR, ...)
        env.setdefault(spec.data_dir_env or spec.data_dir_var, str(spec.data_dir()))
    if spec.port_env:
        env.setdefault(spec.port_env, str(spec.default_port))
    return env


def render(template_text: str, env: dict[str, str]) -> tuple[str, list[str]]:
    """envsubst-style substitution; returns (rendered, missing-vars).
    Comment lines keep their placeholders (they may document ${...})."""

    def sub(m: re.Match) -> str:
        return env.get(m.group(1), m.group(0))

    rendered = VAR_RE.sub(sub, template_text)
    missing: list[str] = []
    for line in rendered.splitlines():
        if line.lstrip().startswith("#"):
            continue
        for m in VAR_RE.finditer(line):
            if m.group(1) not in missing:
                missing.append(m.group(1))
    return rendered, missing


# ── compose plumbing ────────────────────────────────────────────────────────


def _docker() -> str:
    if shutil.which("docker") is None:
        raise StackError("docker not found on PATH")
    return "docker"


def compose_ok() -> None:
    p = subprocess.run(["docker", "compose", "version", "--short"], capture_output=True, text=True)
    if p.returncode != 0:
        raise StackError("docker compose v2 plugin not found (needed: sudo apt install docker-compose-plugin)")


def compose_argv(spec: StackSpec, extra: list[str], env: dict[str] | None = None) -> list[str]:
    """The full ``docker compose ...`` argv for one subcommand."""
    return [_docker(), "compose", "-f", str(spec.compose_file()), "-p", spec.project, *extra]


def _run_compose(spec: StackSpec, extra: list[str], env: dict[str], **kw) -> subprocess.CompletedProcess:
    compose_ok()
    return subprocess.run(compose_argv(spec, extra, env), env={**os.environ, **env}, **kw)


def project_services(spec: StackSpec) -> dict[str, dict]:
    """``{service: {name, state, health, status}}`` from ``compose ps -a``
    (empty dict when the project has no containers)."""
    p = _run_compose(spec, ["ps", "-a", "--format", "json"], resolve_env(spec), capture_output=True, text=True)
    if p.returncode != 0 or not p.stdout.strip():
        return {}
    # compose v5 emits NDJSON (one object per line), older v2 an array
    rows: list[dict] = []
    for line in p.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        rows.extend(obj if isinstance(obj, list) else [obj])
    out: dict[str, dict] = {}
    for r in rows:
        name = r.get("Name") or ""
        svc = r.get("Service")
        if not svc:
            base = name[len(spec.project) + 1 :] if name.startswith(spec.project + "-") else name
            svc = base.rsplit("-", 1)[0] if base.endswith("-1") else base
        out[svc] = {"name": name, "state": r.get("State"), "health": r.get("Health"), "status": r.get("Status")}
    return out


def probe(url: str, timeout_s: float = 5.0) -> str:
    """GET ``url`` once → ``"ok"`` (200) | ``"HTTP <code>"`` | ``"unreachable"``."""
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as r:
            return "ok" if r.status == 200 else f"HTTP {r.status}"
    except Exception:  # noqa: BLE001
        return "unreachable"


# ── up / down / status / logs ────────────────────────────────────────────────


def up(
    spec: StackSpec,
    s: object,
    prepare: object = None,
    *,
    wait: bool = True,
    wait_timeout_s: float | None = None,
    dry_run: bool = False,
    health: object = None,
    banner: object = None,
    compose_wait: bool = False,
) -> None:
    """Start the stack.

    ``prepare(s) -> extra env vars`` — the stack's glue renders its files
    here (and may print notes). ``health(env) -> URL`` must answer 200 before
    up returns (with ``wait``). ``banner(env) -> [lines]`` is the success
    output. ``compose_wait`` adds ``--wait`` to ``up -d`` (only useful when
    the services define healthchecks; the URL poll always does the real
    waiting). A failed up rolls the stack back down (no partial state)."""
    wait_timeout_s = spec.wait_timeout_s if wait_timeout_s is None else wait_timeout_s
    extra = prepare(s) if prepare is not None else None
    env = resolve_env(spec, extra)
    up_args = ["up", "-d"] + (["--wait"] if (wait and compose_wait) else [])
    if dry_run:
        print(" ".join(compose_argv(spec, up_args, env)))
        return
    for sub in spec.data_subdirs:
        (spec.data_dir() / sub).mkdir(parents=True, exist_ok=True)
    p = _run_compose(spec, up_args, env, text=True, stderr=subprocess.PIPE)
    if p.returncode != 0:
        err = (p.stderr or "").strip()
        # best-effort rollback: a failed up never leaves partial state
        _run_compose(spec, ["down"], env, capture_output=True, text=True)
        raise StackError(f"docker compose up failed ({p.returncode}); rolled the stack back down:\n{err}")
    if wait and health is not None:
        url = health(env)
        t0 = time.time()
        while time.time() - t0 < wait_timeout_s:
            if probe(url) == "ok":
                break
            time.sleep(3)
        else:
            raise StackError(
                f"{spec.name} not responsive at {url} after {wait_timeout_s:.0f}s — check: mjolnir {spec.name} logs"
            )
    if banner is not None:
        for line in banner(env):
            print(line)


def down(spec: StackSpec, purge: bool = False, dry_run: bool = False, keep_note: str = "stack state survives") -> int:
    """Stop the stack. Data kept under the data dir by default; ``purge``
    wipes the ``data_subdirs``."""
    env = resolve_env(spec)
    if not dry_run and not project_services(spec):
        print(f"nothing to stop (no {spec.name} stack running)")
        return 0
    if dry_run:
        print(" ".join(compose_argv(spec, ["down"], env)))
        if purge:
            print(
                f"(purge: rm -rf {' '.join(str(spec.data_dir() / s) for s in spec.data_subdirs)}"
                " — via a throwaway root container)"
            )
        return 0
    _run_compose(spec, ["down"], env, text=True)
    if purge:
        _purge_data(spec)
        print(f"stopped + purged the stack data ({', '.join(str(spec.data_dir() / s) for s in spec.data_subdirs)})")
    else:
        print(f"stopped (data kept under {spec.data_dir()} — {keep_note})")
    return 0


def _purge_data(spec: StackSpec) -> None:
    """Wipe the stack's ``data_subdirs``.

    The services run as their images' baked-in nonroot uid (perses 65532,
    prometheus 65534) and create ``0750`` subdirs under the bind-mounted
    data dirs that the host user can't traverse or delete — a host-side
    ``shutil.rmtree`` would silently fail to empty them. Instead, wipe them
    in a throwaway **root** container (root bypasses the permissions).
    Falls back to a best-effort host rmtree if docker can't run."""
    data_dir = spec.data_dir()
    rm = " ".join(f"/d/{sub}" for sub in spec.data_subdirs)
    try:
        subprocess.run(
            ["docker", "run", "--rm", "-v", f"{data_dir}:/d", "busybox", "sh", "-c", f"rm -rf {rm}"],
            check=True,
            capture_output=True,
        )
        return
    except Exception:  # noqa: BLE001 — fall back to host rmtree
        pass
    for sub in spec.data_subdirs:
        shutil.rmtree(data_dir / sub, ignore_errors=True)


def status(spec: StackSpec, extra: dict | None = None) -> dict:
    """``{project, services}`` + the stack's glue-specific fields."""
    st: dict = {"project": spec.project, "services": project_services(spec)}
    if extra:
        st.update(extra)
    return st


def logs(spec: StackSpec, follow: bool = False, tail: int = 100, service: str | None = None) -> int:
    """Tail one service's logs (default: the service named ``spec.name``)."""
    env = resolve_env(spec)
    args = compose_argv(spec, ["logs", "--tail", str(tail)], env)
    if follow:
        args.append("-f")
    args.append(service or spec.name)
    try:
        return subprocess.call(args, env={**os.environ, **env})
    except KeyboardInterrupt:
        return 0
