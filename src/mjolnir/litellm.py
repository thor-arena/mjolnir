"""LiteLLM proxy stack: ``litellm`` + ``postgres`` + ``redis``.

Independent of the vLLM server — ``mjolnir litellm down`` leaves the vLLM
running, and ``mjolnir serve down`` leaves the proxy up. The proxy reaches
the vLLM through ``host.docker.internal`` (the host-published vLLM port), so
the two never share a docker network and neither knows the other's
container name.

The config is rendered on every ``mjolnir litellm up`` from the user's
template (``$MJOLNIR_LITELLM_TEMPLATE`` or, by default, inside the litellm
data dir: ``~/.local/share/mjolnir/litellm/litellm_config.template.yaml``)
with the *currently selected* model/config (``mjolnir model``):
``${MAIN_LLM_MODEL}`` = the served name, ``${MAIN_LLM_MODEL_NAME}`` = the raw
``model:`` field, ``${RAW_MODEL_SUFFIXED}`` = ``<model>-<quant>``.
Everything else resolves from the user's env file (``$MJOLNIR_LITELLM_ENV``,
default ``~/.local/share/mjolnir/litellm/litellm.env``).

ALL user-persisted litellm files live in one dir — ``$MJOLNIR_DATA/litellm``
(default ``~/.local/share/mjolnir/litellm``): the template, the env file,
the rendered config, and the stack data (postgres + redis).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

import yaml

from mjolnir.config import Settings, find_repo_root

PROJECT = "mjolnir-litellm"
COMPOSE_REL = Path("docker/litellm/docker-compose.yaml")
DEFAULT_PROXY_PORT = 8000
VAR_RE = re.compile(r"\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?")


class LitellmError(RuntimeError):
    pass


def _docker() -> str:
    if subprocess.run(["docker", "version"], capture_output=True).returncode != 0:
        raise LitellmError("docker not found on PATH")
    return "docker"


def _compose_version_ok() -> None:
    p = subprocess.run(["docker", "compose", "version", "--short"], capture_output=True, text=True)
    if p.returncode != 0:
        raise LitellmError("docker compose v2 plugin not found (needed: sudo apt install docker-compose-plugin)")


# ── paths ──────────────────────────────────────────────────────────────────


def template_path() -> Path:
    """The user's template: ``$MJOLNIR_LITELLM_TEMPLATE`` or, by default,
    in the litellm data dir:
    ``~/.local/share/mjolnir/litellm/litellm_config.template.yaml``."""
    env = os.environ.get("MJOLNIR_LITELLM_TEMPLATE")
    if env:
        return Path(env).expanduser()
    return data_dir() / "litellm_config.template.yaml"


def env_file_path() -> Path:
    """The user's env file (optional): ``$MJOLNIR_LITELLM_ENV`` or, by
    default, in the litellm data dir:
    ``~/.local/share/mjolnir/litellm/litellm.env``."""
    env = os.environ.get("MJOLNIR_LITELLM_ENV")
    if env:
        return Path(env).expanduser()
    return data_dir() / "litellm.env"


def data_dir() -> Path:
    """Stack data root: ``$MJOLNIR_LITELLM_DATA`` or
    ``$MJOLNIR_DATA/litellm`` (default ``~/.local/share/mjolnir/litellm``);
    holds the rendered config + the postgres/redis state dirs."""
    env = os.environ.get("MJOLNIR_LITELLM_DATA")
    if not env:
        mjolnir_data = os.environ.get("MJOLNIR_DATA", str(Path.home() / ".local" / "share" / "mjolnir"))
        env = str(Path(mjolnir_data).expanduser() / "litellm")
    return Path(env).expanduser()


def compose_file() -> Path:
    return find_repo_root() / COMPOSE_REL


# ── env loading + rendering ────────────────────────────────────────────────


def load_env_file(path: Path) -> dict[str, str]:
    """Parse a ``KEY=VALUE`` env file (comments/blank lines ignored,
    optional single/double quotes around the value)."""
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


def _model_field(s: Settings) -> str:
    """The raw ``model:`` field of the config yaml (falls back to the model
    path) — the ``MAIN_LLM_MODEL_NAME`` the template's alias table wants."""
    from mjolnir.config import _config_field

    return _config_field(s.config_path, "model") or s.model


def injected_env(s: Settings, override_base_url: str | None = None) -> dict[str, str]:
    """The variables mjolnir injects from the active model/config."""
    name = _model_field(s)
    return {
        "MAIN_LLM_MODEL": s.served_model,
        "MAIN_LLM_MODEL_NAME": name,
        "RAW_MODEL_SUFFIXED": f"{name}-{s.quant}",
        "LITELLM_MAIN_MODEL_BASE_URL": override_base_url or f"http://host.docker.internal:{s.port}/v1",
    }


def resolved_env(s: Settings, proxy_port: int | None = None) -> dict[str, str]:
    """Merged env for rendering + compose interpolation:
    user env file < injected model variables (the CLI selection wins) <
    stack defaults (data dir, proxy port)."""
    env = load_env_file(env_file_path())
    env.update(injected_env(s, override_base_url=env.get("LITELLM_MAIN_MODEL_BASE_URL")))
    env.setdefault("LITELLM_DATA_DIR", str(data_dir()))
    # CLI flag > env file > default
    if proxy_port:
        env["LITELLM_PORT"] = str(proxy_port)
    env.setdefault("LITELLM_PORT", str(DEFAULT_PROXY_PORT))
    return env


def render(template_text: str, env: dict[str, str]) -> tuple[str, list[str]]:
    """envsubst-style substitution; returns (rendered, missing-vars)."""

    def sub(m: re.Match) -> str:
        return env.get(m.group(1), m.group(0))

    rendered = VAR_RE.sub(sub, template_text)
    missing: list[str] = []
    for line in rendered.splitlines():
        if line.lstrip().startswith("#"):
            continue  # commented-out sections may keep ${...} placeholders
        for m in VAR_RE.finditer(line):
            if m.group(1) not in missing:
                missing.append(m.group(1))
    return rendered, missing


def validate(rendered: str) -> list[str]:
    """Problems with a rendered config (empty list = ok)."""
    problems: list[str] = []
    try:
        doc = yaml.safe_load(rendered)
    except yaml.YAMLError as e:
        return [f"rendered config is not valid YAML: {e}".replace("\n", " ")]
    ml = doc.get("model_list") if isinstance(doc, dict) else None
    if not isinstance(ml, list):
        problems.append("no model_list in the rendered config")
    else:
        for i, e in enumerate(ml):
            name = (e or {}).get("model_name")
            if not name or not str(name).strip():
                problems.append(f"model_list[{i}] has an empty model_name — check the template variables")
    return problems


def render_config(s: Settings, proxy_port: int | None = None) -> Path:
    """Render the template → ``data_dir()/litellm_config.yaml``; raises
    ``LitellmError`` on a missing template or unresolved variables."""
    tpl = template_path()
    if not tpl.exists():
        example = find_repo_root() / COMPOSE_REL.parent / "litellm_config.template.example.yaml"
        msg = (
            f"template not found: {tpl}\n"
            "  start from the example (or override the location with "
            "$MJOLNIR_LITELLM_TEMPLATE):\n"
            f"  cp {example} {tpl}"
        )
        legacy = Path.home() / "litellm_config.template.yaml"
        if legacy.exists() and legacy != tpl:
            msg += f"\n  found a legacy copy — move it:  mv {legacy} {tpl}"
        raise LitellmError(msg)
    env = resolved_env(s, proxy_port)
    rendered, missing = render(tpl.read_text(), env)
    if missing:
        src = env_file_path()
        raise LitellmError(
            "unresolved template variable(s): "
            + ", ".join("${" + m + "}" for m in missing)
            + f"\n  define them in {src} (see the examples in "
            f"{find_repo_root() / COMPOSE_REL.parent})"
        )
    problems = validate(rendered)
    if problems:
        raise LitellmError("rendered config invalid: " + "; ".join(problems))
    out = data_dir() / "litellm_config.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(rendered)
    return out


# ── compose driver ─────────────────────────────────────────────────────────


def _compose(extra: list[str], env: dict[str, str] | None = None) -> list[str]:
    return [_docker(), "compose", "-f", str(compose_file()), "-p", PROJECT, *extra]


def _run_compose(extra: list[str], env: dict[str], **kw) -> subprocess.CompletedProcess:
    _compose_version_ok()
    full_env = {**os.environ, **env}
    return subprocess.run(_compose(extra, env), env=full_env, **kw)


def _port_of(env: dict[str]) -> int:
    try:
        return int(env.get("LITELLM_PORT", DEFAULT_PROXY_PORT))
    except ValueError:
        return DEFAULT_PROXY_PORT


def _liveliness(port: int, timeout_s: float = 5) -> str:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health/liveliness", timeout=timeout_s) as r:
            return "ok" if r.status == 200 else f"HTTP {r.status}"
    except Exception:  # noqa: BLE001
        return "unreachable"


def up(
    s: Settings, proxy_port: int | None = None, wait: bool = True, wait_timeout_s: float = 600.0, dry_run: bool = False
) -> int:
    """Render the config and start the stack (proxy + postgres + redis)."""
    cfg = render_config(s, proxy_port)
    env = resolved_env(s, proxy_port)
    port = _port_of(env)
    for sub in ("db", "redis"):
        (data_dir() / sub).mkdir(parents=True, exist_ok=True)
    up_args = ["up", "-d"] + (["--wait"] if wait else [])
    if dry_run:
        print(f"(rendered config: {cfg})")
        print(" ".join(_compose(up_args, env)))
        return 0
    p = _run_compose(up_args, env, text=True, stderr=subprocess.PIPE)
    if p.returncode != 0:
        err = (p.stderr or "").strip()
        # best-effort rollback: a failed up never leaves partial state
        _run_compose(["down"], env, capture_output=True, text=True)
        hint = (
            f"\n  is host port {port} already taken? try: mjolnir litellm up --port <other>"
            if "port is already allocated" in err
            else ""
        )
        raise LitellmError(f"docker compose up failed ({p.returncode}); rolled the stack back down:\n{err}{hint}")
    if wait:
        t0 = time.time()
        while time.time() - t0 < wait_timeout_s:
            if _liveliness(port) == "ok":
                break
            time.sleep(3)
        else:
            raise LitellmError(
                f"proxy not responsive at http://127.0.0.1:{port} after "
                f"{wait_timeout_s:.0f}s — check: mjolnir litellm logs"
            )
    print(f"litellm proxy: http://127.0.0.1:{port} (master key: {env.get('LITELLM_MASTER_KEY', 'sk-1234')})")
    print(
        f"  UI:  http://127.0.0.1:{port}/ui"
        f"  ({env.get('LITELLM_UI_USERNAME', 'admin')}/"
        f"{env.get('LITELLM_UI_PASSWORD', 'admin')})"
    )
    print(f"  vllm main model: {env['MAIN_LLM_MODEL']} @ {env['LITELLM_MAIN_MODEL_BASE_URL']}")
    return 0


def _project_services(env: dict[str]) -> dict[str, dict]:
    p = _run_compose(["ps", "-a", "--format", "json"], env, capture_output=True, text=True)
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
            base = name[len(PROJECT) + 1 :] if name.startswith(PROJECT + "-") else name
            svc = base.rsplit("-", 1)[0] if base.endswith("-1") else base
        out[svc] = {"name": name, "state": r.get("State"), "health": r.get("Health"), "status": r.get("Status")}
    return out


def down(purge: bool = False, dry_run: bool = False) -> int:
    """Stop the stack (data kept; ``purge`` wipes postgres + redis data)."""
    env = load_env_file(env_file_path())
    env.setdefault("LITELLM_DATA_DIR", str(data_dir()))
    if not dry_run and not _project_services(env):
        print("nothing to stop (no litellm stack running)")
        return 0
    if dry_run:
        print(" ".join(_compose(["down"], env)))
        if purge:
            print(f"(purge: rm -rf {data_dir() / 'db'} {data_dir() / 'redis'})")
        return 0
    _run_compose(["down"], env, text=True)
    if purge:
        for sub in ("db", "redis"):
            shutil.rmtree(data_dir() / sub, ignore_errors=True)
        print(f"stopped + purged the stack data ({data_dir() / 'db'}, {data_dir() / 'redis'})")
    else:
        print(f"stopped (data kept under {data_dir()} — postgres/redis state survives)")
    return 0


def status(s: Settings, proxy_port: int | None = None) -> dict:
    env = resolved_env(s, proxy_port)
    port = _port_of(env)
    return {
        "project": PROJECT,
        "services": _project_services(env),
        "proxy": {
            "url": f"http://127.0.0.1:{port}",
            "liveliness": _liveliness(port),
            "config": str(data_dir() / "litellm_config.yaml"),
            "template": str(template_path()),
        },
    }


def logs(follow: bool = False, tail: int = 100) -> int:
    env = load_env_file(env_file_path())
    env.setdefault("LITELLM_DATA_DIR", str(data_dir()))
    _compose_version_ok()
    args = [_docker(), "compose", "-f", str(compose_file()), "-p", PROJECT, "logs", "--tail", str(tail)]
    if follow:
        args += ["-f"]
    args += ["litellm"]
    try:
        return subprocess.call(args, env={**os.environ, **env})
    except KeyboardInterrupt:
        return 0
