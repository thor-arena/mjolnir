"""Charts + README + docs render from raw verify data — and degrade to
"pending" notes when a stage has no run (the state of a fresh checkout)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from mjolnir_cli import charts, interactive, readme
from mjolnir_cli.config import load_layout

from verify_fixtures import write_all

REPO = Path(__file__).resolve().parents[2]

pytest.importorskip("matplotlib")


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A minimal checkout: the templates + Dockerfile + the real history."""
    root = tmp_path / "repo"
    (root / "docker" / "vllm-thor").mkdir(parents=True)
    shutil.copy(REPO / "docker" / "vllm-thor" / "Dockerfile", root / "docker" / "vllm-thor" / "Dockerfile")
    shutil.copy(REPO / "docker" / "vllm-thor" / "PATCHES.md", root / "docker" / "vllm-thor" / "PATCHES.md")
    shutil.copy(REPO / "docker" / "vllm-thor" / "README.md.j2", root / "docker" / "vllm-thor" / "README.md.j2")
    shutil.copy(REPO / "README.md.j2", root / "README.md.j2")
    shutil.copytree(REPO / "docs" / "kernel", root / "docs" / "kernel", ignore=shutil.ignore_patterns("*.md"))
    for pkg, init in (("mjolnir/mjolnir", "__init__.py"), ("mjolnir-cli/mjolnir_cli", "__init__.py")):
        (root / "src" / pkg).mkdir(parents=True)
        shutil.copy(REPO / "src" / pkg / init, root / "src" / pkg / init)
    (root / "benchmarks").mkdir()
    shutil.copy(REPO / "benchmarks" / "history.jsonl", root / "benchmarks" / "history.jsonl")
    monkeypatch.setenv("MJOLNIR_REPO", str(root))
    for v in ("MJOLNIR_BENCHMARKS", "MJOLNIR_HISTORY"):
        monkeypatch.delenv(v, raising=False)
    return root


def test_full_render_from_fixture_data(repo):
    layout = load_layout()
    write_all(layout.verify_dir)
    manifest = charts.render_all(layout)
    for name in (
        "kernel-speedup",
        "kernel-latency",
        "hnd-layout",
        "kvfmt-heatmap",
        "ring-heatmap",
        "prefetch-heatmap",
        "ab-heatmap",
        "pdl-delta",
        "quant-accuracy",
        "fp6prof-bytes-vs-time",
    ):
        assert name in manifest, name
        assert (repo / manifest[name]["light"]).exists() and (repo / manifest[name]["dark"]).exists()
    readme.render_readme(layout)
    text = (repo / "README.md").read_text()
    assert "{{" not in text and "{%" not in text
    # the Docker Hub description: rendered too, absolute URLs only (Docker Hub
    # has no repo file hosting to resolve relatives against)
    dh = (repo / "docker" / "vllm-thor" / "README.md").read_text()
    assert "{{" not in dh and "<img" not in dh
    assert "https://raw.githubusercontent.com/thor-arena/mjolnir/main/assets/hero.png" in dh
    # the patch inventory re-emitted verbatim from docker/vllm-thor/PATCHES.md
    assert "| File | PR | What it fixes |" in dh and "thor-fp6-e2m3-kv-cache.patch" in dh
    # the geomean vs FlashInfer from the fixture sweep (the case study's headline line)
    assert "**Where it stands:** 1.26× FlashInfer" in (repo / "docs" / "kernel" / "README.md").read_text()
    assert 'srcset="assets/benchmarks/kernel-latency-dark.png"' in text
    page = (repo / "docs" / "kernel" / "ring.md").read_text()
    assert "../../assets/benchmarks/ring-heatmap.png" in page  # paths are relative to the page
    assert "Fastest variant at the most context lengths" in page
    site = interactive.build(layout, repo / "site")
    data = json.loads((repo / "site" / "data.json").read_text())
    assert "sweep" in data["stages"] and site.read_text().count("cdn.jsdelivr.net/npm/plotly.js-dist-min@") == 1


def test_render_without_verify_data_is_pending(repo):
    layout = load_layout()
    charts.render_all(layout)  # e2e charts only (history), no kernel charts
    readme.render_readme(layout)
    text = (repo / "README.md").read_text()
    assert "Pending: `mjolnir verify sweep`" in text
    assert "Pending — run `mjolnir verify ring`" in (repo / "docs" / "kernel" / "ring.md").read_text()


def test_render_readme_renders_templates_only_and_leaves_docs_cli_md_alone(repo):
    layout = load_layout()
    written = readme.render_readme(layout)
    assert repo / "docs" / "cli.md" not in written
    assert not (repo / "docs" / "cli.md").exists()


def test_render_is_idempotent(repo):
    layout = load_layout()
    write_all(layout.verify_dir)
    charts.render_all(layout)
    readme.render_readme(layout)
    first = (repo / "README.md").read_text()
    readme.render_readme(layout)
    assert (repo / "README.md").read_text() == first


def test_hero_overlays_the_background_under_the_logotype(tmp_path):
    from PIL import Image

    from mjolnir_cli.config import RepoLayout, find_repo_root

    root = find_repo_root()
    (tmp_path / "assets").mkdir()
    shutil.copy(root / "assets" / charts.HERO_BACKGROUND, tmp_path / "assets" / charts.HERO_BACKGROUND)
    layout = RepoLayout(
        repo_root=tmp_path,
        data_dir=tmp_path,
        state_file=tmp_path / "s.json",
        benchmarks_dir=tmp_path / "benchmarks",
        history_file=tmp_path / "h.jsonl",
    )
    out = charts.render_hero(layout)
    bg = Image.open(tmp_path / "assets" / charts.HERO_BACKGROUND)
    hero = Image.open(out)
    assert out.name == "hero.png" and hero.size == bg.size
    left, top, right, bottom = charts._logotype_box(bg)
    assert left > bg.size[0] * 0.6 and bottom < bg.size[1] * 0.8  # the MJOLNIR logotype, right side
    # the logotype is untouched; the area under it now carries the slogan and pills
    assert (
        hero.crop((left, top, right, bottom)).tobytes() == bg.convert("RGB").crop((left, top, right, bottom)).tobytes()
    )
    under = (left, bottom + 20, right, min(bg.size[1], bottom + 200))
    assert hero.crop(under).tobytes() != bg.convert("RGB").crop(under).tobytes()


def test_readme_uses_the_hero():
    from mjolnir_cli.config import find_repo_root

    tpl = (find_repo_root() / "README.md.j2").read_text()
    assert 'src="assets/hero.png"' in tpl and "header.png" not in tpl
