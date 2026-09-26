"""Mjolnir defaults + config resolution.

Everything the CLI needs has a baked-in default (the Qwen3.8-27B NVFP4 + FA4
GEMV setup that this repo benchmarks); flags on each command override them,
then the remembered state (``mjolnir model`` / ``mjolnir image`` →
``$MJOLNIR_STATE``, default ``~/.mjolnir-state.json``), then ``$MJOLNIR_*``
env vars. No ``.env`` file required.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# ── Model / config ───────────────────────────────────────────────────────────
DEFAULT_MODEL = "Qwen/Qwen3.8-27B"
DEFAULT_QUANT = "NVFP4_FA4hd256"      # FA4 + the GEMV decode kernel (this repo's default)
BASELINE_QUANT = "NVFP4"               # FlashInfer baseline config
SERVED_MODEL_NAME = "Qwen/Qwen3.8-27B"

# ── Image / serving ──────────────────────────────────────────────────────────
DEFAULT_IMAGE = "mjolnir/vllm-thor:qwen38-sm110-v11"
DEFAULT_PORT = 6001                      # host port; vLLM metrics live on the same port
CONTAINER_NAME = "mjolnir-vllm"
VFA_DIST_PATH = "/usr/local/lib/python3.12/dist-packages/vllm/vllm_flash_attn"

# Friendly backend labels for the history log / charts, keyed by quant config.
BACKEND_LABELS = {
    "NVFP4_FA4hd256": "FA4-GEMV",     # FA4 hd256 + our pure-FMA GEMV decode kernel
    "NVFP4_13_FA4": "FA4-1CTA",
    "NVFP4": "FlashInfer",
}


@dataclass(frozen=True)
class ConfigEntry:
    """One model config discovered under ``configs/<vendor>/<model>/<quant>.yaml``."""

    model: str          # "vendor/Model", e.g. "Qwen/Qwen3.8-27B"
    quant: str          # config name, e.g. "NVFP4_FA4hd256"
    path: Path

    @property
    def backend(self) -> str:
        return BACKEND_LABELS.get(self.quant, self.quant)


def scan_configs(configs_dir: Path) -> list[ConfigEntry]:
    """Scan ``configs/<vendor>/<model>/<quant>.yaml`` into pickable entries."""
    out: list[ConfigEntry] = []
    if not configs_dir.is_dir():
        return out
    for p in sorted(configs_dir.glob("*/*/*.yaml")):
        out.append(ConfigEntry(
            model=str(p.parent.relative_to(configs_dir)),
            quant=p.stem, path=p))
    return out


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


@dataclass
class RepoLayout:
    """Paths inside the repo (or the user's home for state)."""

    repo_root: Path
    data_dir: Path
    state_file: Path
    benchmarks_dir: Path
    history_file: Path

    @property
    def raw_dir(self) -> Path:
        return self.benchmarks_dir / "raw"

    @property
    def charts_dir(self) -> Path:
        return self.repo_root / "assets" / "benchmarks"


def find_repo_root() -> Path:
    """Locate the mjolnir repo: $MJOLNIR_REPO wins, else walk up from CWD
    looking for ``docker/vllm-thor/Dockerfile`` (works when running from a
    checkout; installed-CLI users set $MJOLNIR_REPO)."""
    env = os.environ.get("MJOLNIR_REPO")
    if env:
        return Path(env)
    cur = Path.cwd().resolve()
    for cand in [cur, *cur.parents]:
        if (cand / "docker" / "vllm-thor" / "Dockerfile").exists():
            return cand
    raise FileNotFoundError(
        "mjolnir repo not found (set $MJOLNIR_REPO to the repo root, or run "
        "from inside the repo)")


def load_layout() -> RepoLayout:
    root = find_repo_root()
    data = Path(_env("MJOLNIR_DATA", "~/.local/share/mjolnir")).expanduser()
    return RepoLayout(
        repo_root=root,
        data_dir=data,
        state_file=Path(_env("MJOLNIR_STATE", str(Path.home() / ".mjolnir-state.json"))),
        benchmarks_dir=Path(_env("MJOLNIR_BENCHMARKS", str(root / "benchmarks"))),
        history_file=Path(_env("MJOLNIR_HISTORY", str(root / "benchmarks" / "history.jsonl"))),
    )


@dataclass
class Settings:
    """Resolved runtime settings for a command."""

    model: str = field(default_factory=lambda: _env("MJOLNIR_MODEL", DEFAULT_MODEL))
    quant: str = field(default_factory=lambda: _env("MJOLNIR_QUANT", DEFAULT_QUANT))
    image: str = field(default_factory=lambda: _env("MJOLNIR_IMAGE", DEFAULT_IMAGE))
    port: int = field(default_factory=lambda: _env_int("MJOLNIR_PORT", DEFAULT_PORT))
    gemv: bool = True                      # VLLM_FA4_HD256_GEMV default on: the kernel is the point
    data_dir: Path = field(default_factory=lambda: Path("~/.local/share/mjolnir").expanduser())

    @property
    def config_path(self) -> Path:
        root = find_repo_root()
        return root / "configs" / self.model / f"{self.quant}.yaml"

    @property
    def served_model(self) -> str:
        return SERVED_MODEL_NAME

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    @property
    def metrics_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/metrics"

    @property
    def backend_label(self) -> str:
        return BACKEND_LABELS.get(self.quant, self.quant)


def resolve(cli_model: str | None, cli_quant: str | None,
            cli_image: str | None, cli_port: int | None,
            gemv: bool | None = None) -> Settings:
    """CLI flags > env vars > defaults."""
    s = Settings()
    if cli_model:
        s.model = cli_model
    if cli_quant:
        s.quant = cli_quant
    if cli_image:
        s.image = cli_image
    if cli_port:
        s.port = cli_port
    if gemv is not None:
        s.gemv = gemv
    return s
