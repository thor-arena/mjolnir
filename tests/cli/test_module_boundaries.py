"""Pins mjolnir_cli's import graph, so a later change that re-tangles it fails CI.

Read straight from the source with ``ast`` (every import counts, function-local
ones included; only ``if TYPE_CHECKING:`` blocks are skipped, since a type-only
import never runs):

* the ``readme`` and ``charts`` packages are layered as ``ALLOWED`` declares,
  and hold exactly the modules it declares: readme is fmt <- tables / findings
  <- context / how <- ``__init__``; charts is ``_common`` <- one module per
  chart family (kernel, e2e, hero, infographic, accuracy, autoresearch) <-
  ``__init__``, so a family imports only ``_common`` from ``charts`` and only
  ``__init__`` crosses families;
* modules outside a ``FACADES`` package reach it only through its facade;
* ``FORBIDDEN`` lists the packages that must not import each other: nothing in
  ``readme`` reaches up to ``charts``, ``cli`` or ``clidoc``, and ``charts``
  never imports ``readme`` (the labels both need live in ``mjolnir_cli.labels``,
  the verify data loader in ``verify.latest``), ``cli`` or ``clidoc``;
* a name with a leading underscore stays inside its module: governed modules
  import none from elsewhere, and nobody imports one from a governed module.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src" / "mjolnir-cli"
PKG = "mjolnir_cli"
README = "mjolnir_cli.readme"
CHARTS = "mjolnir_cli.charts"

# Modules whose private names must not cross a module boundary, in either
# direction. The file-length split (#33) adds each package here as it is
# split: readme and charts so far; the cli commands come next.
GOVERNED = {"mjolnir_cli.readme", "mjolnir_cli.labels", "mjolnir_cli.verify.latest", CHARTS}


@dataclass(frozen=True)
class Use:
    """One reference from a module to another mjolnir_cli module."""

    target: str  # the module imported (or imported from)
    name: str | None  # the name taken from it (`from x import name`, `x.name`), None for the module itself
    line: int


def _module_name(path: Path) -> str:
    parts = path.relative_to(SRC).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def _modules() -> dict[str, Path]:
    return {_module_name(p): p for p in sorted((SRC / PKG).rglob("*.py"))}


def _is_type_checking(node: ast.If) -> bool:
    t = node.test
    return (isinstance(t, ast.Name) and t.id == "TYPE_CHECKING") or (
        isinstance(t, ast.Attribute) and t.attr == "TYPE_CHECKING"
    )


def _private(name: str | None) -> bool:
    return bool(name) and name.startswith("_") and not (name.startswith("__") and name.endswith("__"))


def uses(source: str, module: str, known: set[str], is_pkg: bool = False) -> list[Use]:
    """Every mjolnir_cli module ``source`` (the module ``module``, a package's
    ``__init__`` when ``is_pkg``) refers to."""
    package = module if is_pkg else module.rpartition(".")[0]
    out: list[Use] = []
    aliases: dict[str, str] = {}  # local name -> the mjolnir_cli module it is bound to

    def resolve(node: ast.ImportFrom) -> str:
        if not node.level:
            return node.module or ""
        base = package.split(".")
        base = base[: len(base) - node.level + 1]
        return ".".join(base + ([node.module] if node.module else []))

    def visit(node: ast.AST) -> None:
        if isinstance(node, ast.If) and _is_type_checking(node):
            for child in node.orelse:
                visit(child)
            return
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name == PKG or a.name.startswith(PKG + "."):
                    out.append(Use(a.name, None, node.lineno))
                    aliases[a.asname or a.name] = a.name
        elif isinstance(node, ast.ImportFrom):
            src = resolve(node)
            if src == PKG or src.startswith(PKG + "."):
                for a in node.names:
                    sub = f"{src}.{a.name}"
                    if sub in known:  # `from mjolnir_cli import readme` imports a module
                        out.append(Use(sub, None, node.lineno))
                        aliases[a.asname or a.name] = sub
                    else:
                        out.append(Use(src, a.name, node.lineno))
        elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in aliases:
            out.append(Use(aliases[node.value.id], node.attr, node.lineno))
        for child in ast.iter_child_nodes(node):
            visit(child)

    visit(ast.parse(source))
    return out


@pytest.fixture(scope="module")
def graph() -> dict[str, list[Use]]:
    mods = _modules()
    known = set(mods)
    return {m: uses(p.read_text(), m, known, p.name == "__init__.py") for m, p in mods.items()}


def _in(module: str, root: str) -> bool:
    return module == root or module.startswith(root + ".")


FMT, TABLES, FINDINGS = f"{README}.fmt", f"{README}.tables", f"{README}.findings"
CHARTS_COMMON = f"{CHARTS}._common"
CHART_FAMILIES = ("kernel", "e2e", "hero", "infographic", "accuracy", "autoresearch")

# The layered packages: each one's ``__init__`` may import any of its modules.
LAYERED = (README, CHARTS)

# module of a layered package -> the other modules of that package it may import
ALLOWED = {
    FMT: set(),
    TABLES: {FMT},
    FINDINGS: {FMT},
    f"{README}.context": {FMT, TABLES, FINDINGS},
    f"{README}.how": {FMT, TABLES, FINDINGS},
    CHARTS_COMMON: set(),
    **{f"{CHARTS}.{family}": {CHARTS_COMMON} for family in CHART_FAMILIES},
}

# packages that modules outside them may import only through their facade
FACADES = (CHARTS,)

# importer package -> the mjolnir_cli packages nothing in it may import
FORBIDDEN = [
    (README, CHARTS),
    (README, "mjolnir_cli.cli"),
    (README, "mjolnir_cli.clidoc"),
    (CHARTS, README),
    (CHARTS, "mjolnir_cli.cli"),
    (CHARTS, "mjolnir_cli.clidoc"),
]


def test_walker_sees_function_local_and_attribute_uses_but_not_type_checking_blocks():
    src = (
        "from typing import TYPE_CHECKING\n"
        "if TYPE_CHECKING:\n    from mjolnir_cli import cli\n"
        "def f():\n    from mjolnir_cli import charts\n    return charts._x\n"
        "from .fmt import _y\n"
    )
    known = {"mjolnir_cli", "mjolnir_cli.charts", "mjolnir_cli.cli", "mjolnir_cli.readme", "mjolnir_cli.readme.fmt"}
    found = {(u.target, u.name) for u in uses(src, "mjolnir_cli.readme.tables", known)}
    assert found == {("mjolnir_cli.charts", None), ("mjolnir_cli.charts", "_x"), ("mjolnir_cli.readme.fmt", "_y")}


def _package(module: str) -> str:
    return next(p for p in LAYERED if _in(module, p))


@pytest.mark.parametrize("package", LAYERED)
def test_layered_package_holds_exactly_its_declared_modules(graph, package):
    assert {m for m in graph if _in(m, package)} == {package, *(m for m in ALLOWED if _package(m) == package)}


@pytest.mark.parametrize("module", sorted(ALLOWED))
def test_layered_module_imports_only_its_allowed_package_modules(graph, module):
    package = _package(module)
    assert module in graph, f"{module} is missing — the {package} package layout changed"
    deps = {u.target for u in graph[module] if _in(u.target, package) and u.target != module}
    assert deps <= ALLOWED[module]


@pytest.mark.parametrize("package", FACADES)
def test_modules_outside_a_facade_package_reach_it_only_through_its_facade(graph, package):
    bad = [
        (m, u.target, u.line)
        for m in graph
        if not _in(m, package)
        for u in graph[m]
        if _in(u.target, package) and u.target != package
    ]
    assert bad == []


@pytest.mark.parametrize(("importer", "target"), FORBIDDEN)
def test_package_never_imports_forbidden_package(graph, importer, target):
    mods = [m for m in graph if _in(m, importer)]
    assert mods and any(_in(m, target) for m in graph), (importer, target)
    bad = [(m, u.target, u.line) for m in mods for u in graph[m] if _in(u.target, target)]
    assert bad == []


def _governed(module: str) -> bool:
    return any(_in(module, g) for g in GOVERNED)


def test_governed_modules_import_no_underscore_name_from_another_module(graph):
    assert all(any(_in(m, g) for m in graph) for g in GOVERNED), "a governed module is missing"
    bad = [
        (m, u.target, u.name, u.line)
        for m in graph
        if _governed(m)
        for u in graph[m]
        if u.target != m and _private(u.name)
    ]
    assert bad == []


def test_no_module_imports_an_underscore_name_from_a_governed_module(graph):
    bad = [
        (m, u.target, u.name, u.line)
        for m in graph
        for u in graph[m]
        if u.target != m and _governed(u.target) and _private(u.name)
    ]
    assert bad == []
