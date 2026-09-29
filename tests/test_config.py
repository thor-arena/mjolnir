"""Settings resolution: the baked-in defaults, the env layer, and CLI-flag
precedence (flag > env > default) — the contract the whole CLI relies on."""

from __future__ import annotations

from pathlib import Path

import pytest

from mjolnir import config
from mjolnir.config import (
    BACKEND_LABELS,
    DEFAULT_IMAGE,
    DEFAULT_MODEL,
    DEFAULT_PORT,
    DEFAULT_QUANT,
    Settings,
    scan_all_configs,
    scan_configs,
)


@pytest.fixture(autouse=True)
def _clean_mjolnir_env(monkeypatch):
    for var in ("MJOLNIR_MODEL", "MJOLNIR_QUANT", "MJOLNIR_IMAGE", "MJOLNIR_PORT", "MJOLNIR_REPO", "MJOLNIR_CONFIGS_DIR"):
        monkeypatch.delenv(var, raising=False)


def test_defaults_with_no_env_or_cli():
    s = config.resolve(None, None, None, None)
    assert (s.model, s.quant, s.image, s.port) == (DEFAULT_MODEL, DEFAULT_QUANT, DEFAULT_IMAGE, DEFAULT_PORT)
    assert s.gemv is True  # the kernel is the point
    assert s.backend_label == "FA4-GEMV"  # via BACKEND_LABELS[DEFAULT_QUANT]


def test_env_overrides_defaults(monkeypatch):
    monkeypatch.setenv("MJOLNIR_MODEL", "vendor/Other")
    monkeypatch.setenv("MJOLNIR_PORT", "7001")
    s = config.resolve(None, None, None, None)
    assert s.model == "vendor/Other"
    assert s.port == 7001
    assert s.quant == DEFAULT_QUANT  # untouched key keeps its default


def test_cli_flag_beats_env(monkeypatch):
    monkeypatch.setenv("MJOLNIR_MODEL", "env-model")
    monkeypatch.setenv("MJOLNIR_IMAGE", "env-image")
    s = config.resolve("cli-model", None, None, None)
    assert s.model == "cli-model"
    assert s.image == "env-image"  # no flag for image -> env wins


def test_bad_port_env_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("MJOLNIR_PORT", "not-a-port")
    assert config.resolve(None, None, None, None).port == DEFAULT_PORT


def test_gmv_flag_precedence():
    assert config.resolve(None, None, None, None, gemv=False).gemv is False
    assert config.resolve(None, None, None, None, gemv=None).gemv is True  # None = don't override


def _repo_with_configs(root: Path, entries: dict[str, str]) -> None:
    for rel, body in entries.items():
        p = root / "configs" / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body)


def test_scan_configs_finds_the_three_level_tree(tmp_path):
    d = tmp_path / "configs" / "Qwen" / "Qwen3.8-27B"
    d.mkdir(parents=True)
    (d / "NVFP4.yaml").write_text("model: x\n")
    (d / "not-yaml.txt").write_text("nope")
    (tmp_path / "configs" / "stray.yaml").write_text("model: y\n")  # wrong depth: ignored
    entries = scan_configs(tmp_path / "configs")
    assert [(e.model, e.quant) for e in entries] == [("Qwen/Qwen3.8-27B", "NVFP4")]
    assert entries[0].source == "repo"


def test_scan_configs_missing_dir_is_empty(tmp_path):
    assert scan_configs(tmp_path) == []


def test_scan_all_configs_repo_wins_on_clash(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    (repo / "docker" / "vllm-thor").mkdir(parents=True)
    (repo / "docker" / "vllm-thor" / "Dockerfile").write_text("# base")
    _repo_with_configs(repo, {"Qwen/Qwen3.8-27B/NVFP4.yaml": "model: repo-copy\n"})
    local = tmp_path / "local"
    (local / "Qwen" / "Qwen3.8-27B").mkdir(parents=True)
    (local / "Qwen" / "Qwen3.8-27B" / "NVFP4.yaml").write_text("model: local-copy\n")
    (local / "Qwen" / "Qwen3.8-27B" / "Extra.yaml").write_text("model: local-only\n")
    monkeypatch.setenv("MJOLNIR_REPO", str(repo))
    monkeypatch.setenv("MJOLNIR_CONFIGS_DIR", str(local))

    entries = {e.model + "/" + e.quant: e for e in scan_all_configs()}
    assert len(entries) == 2
    winner = entries["Qwen/Qwen3.8-27B/NVFP4"]
    assert winner.source == "repo"  # the repo copy shadows the local one
    assert "repo-copy" in winner.path.read_text()
    assert entries["Qwen/Qwen3.8-27B/Extra"].source == "local"
    # sorted by (model, quant)
    assert [e.model + "/" + e.quant for e in scan_all_configs()] == sorted(entries)


def test_find_repo_root_env_wins(tmp_path, monkeypatch):
    monkeypatch.setenv("MJOLNIR_REPO", str(tmp_path))
    assert config.find_repo_root() == tmp_path


def test_find_repo_root_fails_outside_repo(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match="MJOLNIR_REPO"):
        config.find_repo_root()


def _settings_with_repo(tmp_path: Path, monkeypatch) -> Settings:
    repo = tmp_path / "repo"
    (repo / "docker" / "vllm-thor").mkdir(parents=True)
    (repo / "docker" / "vllm-thor" / "Dockerfile").write_text("# base")
    monkeypatch.setenv("MJOLNIR_REPO", str(repo))
    return Settings()


def test_config_path_repo_wins_over_local(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    (repo / "docker" / "vllm-thor").mkdir(parents=True)
    (repo / "docker" / "vllm-thor" / "Dockerfile").write_text("# base")
    monkeypatch.setenv("MJOLNIR_REPO", str(repo))
    model, quant = "Qwen/Qwen3.8-27B", "NVFP4"
    repo_cfg = repo / "configs" / model / f"{quant}.yaml"
    repo_cfg.parent.mkdir(parents=True)
    repo_cfg.write_text("model: m\n")
    local_cfg = tmp_path / "local" / model / f"{quant}.yaml"
    local_cfg.parent.mkdir(parents=True)
    local_cfg.write_text("model: m\n")
    monkeypatch.setenv("MJOLNIR_CONFIGS_DIR", str(tmp_path / "local"))

    assert Settings(model=model, quant=quant).config_path == repo_cfg  # repo wins
    repo_cfg.unlink()
    assert Settings(model=model, quant=quant).config_path == local_cfg  # local fallback
    local_cfg.unlink()  # neither -> the repo path (for the error message)
    assert Settings(model=model, quant=quant).config_path == repo_cfg


def test_served_model_precedence(tmp_path, monkeypatch):
    s = _settings_with_repo(tmp_path, monkeypatch)
    model, quant = "Qwen/Qwen3.8-27B", "NVFP4"
    repo_cfg = tmp_path / "repo" / "configs" / model / f"{quant}.yaml"
    repo_cfg.parent.mkdir(parents=True)

    s.model, s.quant = model, quant
    repo_cfg.write_text("")  # empty config -> the baked-in constant
    assert s.served_model == config.SERVED_MODEL_NAME

    repo_cfg.write_text("model: Qwen/Qwen3.8-27B\n")
    assert s.served_model == "Qwen/Qwen3.8-27B"  # the 'model' field

    repo_cfg.write_text("model: Qwen/Qwen3.8-27B\nserved-model-name: mjolnir-qwen\n")
    assert s.served_model == "mjolnir-qwen"  # explicit name wins


def test_backend_label_explicit_wins():
    assert Settings(label="Custom").backend_label == "Custom"
    assert Settings(quant="NVFP4").backend_label == BACKEND_LABELS["NVFP4"]
    assert Settings(quant="SomeQuant").backend_label == "SomeQuant"  # unknown -> the quant string
