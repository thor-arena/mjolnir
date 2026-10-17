"""Pins the public surface of the ``mjolnir_cli.charts`` package and its
host-side import hygiene, so a later change cannot silently drop a name other
modules rely on, re-grow the facade into a re-export dump, or pull matplotlib /
the GPU stack in at import.

* the facade (``charts/__init__.py``) exports exactly the API that callers
  outside ``charts`` use, and nothing kept only for tests;
* every other name lives in, and is imported from, the module that owns it,
  under a public name (no leading underscore once it crosses a module).
"""

from __future__ import annotations

import importlib
import subprocess
import sys

import pytest

# what callers outside charts use: cli / verify.cli (render_all), interactive
# (e2e_rows), autoresearch (pyplot, chart_autoresearch, save_png), readme's
# tests (load_verify, the verify layer's loader)
FACADE = {"render_all", "load_verify", "e2e_rows", "chart_autoresearch", "pyplot", "save_png"}

# (owning module, name): what tests and the facade import from each module
OWNED = (
    ("mjolnir_cli.charts._common", "pyplot"),
    ("mjolnir_cli.charts._common", "save_png"),
    ("mjolnir_cli.charts._common", "display_family"),
    ("mjolnir_cli.charts._common", "L_label"),
    ("mjolnir_cli.charts.kernel", "KERNEL_CHARTS"),
    ("mjolnir_cli.charts.kernel", "chart_spec"),
    ("mjolnir_cli.charts.kernel", "chart_kvcache"),
    ("mjolnir_cli.charts.e2e", "E2E_CHARTS"),
    ("mjolnir_cli.charts.e2e", "e2e_rows"),
    ("mjolnir_cli.charts.hero", "HERO_BACKGROUND"),
    ("mjolnir_cli.charts.hero", "render_hero"),
    ("mjolnir_cli.charts.hero", "logotype_box"),
    ("mjolnir_cli.charts.infographic", "INFOGRAPHIC"),
    ("mjolnir_cli.charts.infographic", "render_infographic"),
    ("mjolnir_cli.charts.infographic", "multiple"),
    ("mjolnir_cli.charts.accuracy", "ACCURACY_CHARTS"),
    ("mjolnir_cli.charts.autoresearch", "chart_autoresearch"),
)

# the old private spellings of names that now cross a module boundary
RETIRED = ("_plt", "_save", "_logotype_box", "_display_family")

HEAVY = ("matplotlib", "torch", "vllm", "cutlass", "flashinfer", "triton")


def test_charts_facade_exports_exactly_the_external_api():
    charts = importlib.import_module("mjolnir_cli.charts")
    assert set(charts.__all__) == FACADE
    missing = [n for n in charts.__all__ if not hasattr(charts, n)]
    assert missing == []


@pytest.mark.parametrize("module,name", OWNED)
def test_charts_name_is_importable_from_its_owning_module(module, name):
    assert hasattr(importlib.import_module(module), name), f"{module} lost {name}"


@pytest.mark.parametrize("module,name", [(m, n) for m, n in OWNED if n not in FACADE])
def test_charts_facade_does_not_re_export_family_only_names(module, name):
    charts = importlib.import_module("mjolnir_cli.charts")
    assert not hasattr(charts, name), f"mjolnir_cli.charts re-exports {name}: import it from {module}"


@pytest.mark.parametrize("name", RETIRED)
def test_charts_retired_private_names_are_gone(name):
    mods = {m for m, _ in OWNED} | {"mjolnir_cli.charts"}
    still = [m for m in sorted(mods) if hasattr(importlib.import_module(m), name)]
    assert still == []


def test_charts_import_loads_no_matplotlib_or_gpu_stack():
    code = (
        "import sys, mjolnir_cli.charts;"
        f"bad = [m for m in sys.modules if m.split('.')[0] in {HEAVY!r}];"
        "print(bad); sys.exit(1 if bad else 0)"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
