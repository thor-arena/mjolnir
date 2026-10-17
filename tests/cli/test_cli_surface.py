"""Pin the public surface of ``mjolnir_cli.cli``.

These tests describe behavior that must survive splitting cli.py into
modules: ``mjolnir_cli.cli`` is a composition root whose only export is
``app`` (the console script target), every command body lives in a
``mjolnir_cli.commands`` module that owns its helpers, and the --help of every
sub-app keeps the order its commands are registered in. The import direction
(nothing but ``__main__`` imports ``mjolnir_cli.cli``) and the ``commands``
layering are rows of the tables in test_module_boundaries.py.
"""

from __future__ import annotations

import __future__
import inspect
import re
from importlib.metadata import entry_points

import pytest
from typer.testing import CliRunner

runner = CliRunner()

# Registration order, read by hand from the @<sub>_app.command decorators in commands/<sub>.py.
SUB_APP_COMMANDS = {
    "serve": ["up", "down", "status", "logs"],
    "model": ["use", "list"],
    "image": ["use", "list", "build", "gates"],
    "litellm": ["up", "down", "status", "logs", "config"],
    "perses": ["up", "down", "status", "logs"],
    "hw": ["setup", "status"],
    "bench": ["perf", "ab", "accuracy", "e2e", "compare"],
}

# Root commands (not sub-apps): their --help shows these options.
ROOT_COMMAND_OPTIONS = {
    "history": ["--limit", "--json"],
    "plot": ["--readme", "--readme-template", "--readme-out", "--site"],
}


def _listed_commands(help_text: str) -> list[str]:
    """Command names from the rich 'Commands' panel, in the order shown."""
    in_panel = False
    names = []
    for line in help_text.splitlines():
        if line.startswith("╭─ Commands"):
            in_panel = True
        elif in_panel and line.startswith("╰"):
            break
        elif in_panel:
            m = re.match(r"│ ([a-z][\w-]*)\s{2,}", line)
            if m:
                names.append(m.group(1))
    return names


_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _invoke_help(*args: str) -> str:
    """Help text as plain text: rich colors it when it detects CI (GITHUB_ACTIONS)."""
    from mjolnir_cli.cli import app

    result = runner.invoke(app, [*args, "--help"])
    assert result.exit_code == 0, result.output
    return _ANSI.sub("", result.output)


def test_cli_helpers_import_from_the_modules_that_own_them():
    from mjolnir_cli.cli import app
    from mjolnir_cli.commands._common import settings
    from mjolnir_cli.commands.bench import run_legs
    from mjolnir_cli.commands.bench_report import print_accuracy, print_compare

    assert callable(app)
    for fn in (settings, run_legs, print_compare, print_accuracy):
        assert callable(fn)


def test_cli_facade_exports_only_app():
    from mjolnir_cli import cli

    assert cli.__all__ == ["app"]
    public = {
        n
        for n, v in vars(cli).items()
        if not n.startswith("_") and not inspect.ismodule(v) and not isinstance(v, __future__._Feature)
    }
    assert public == {"app"}


def test_cli_is_a_composition_root_that_defines_no_command():
    from mjolnir_cli import cli

    defined = {n for n, v in vars(cli).items() if inspect.isfunction(v) and v.__module__ == cli.__name__}
    assert defined == {"_root_group"}


def _callbacks(cmd, path=()):
    """(path, callback) of every leaf command under ``cmd``."""
    import typer.main

    try:  # typer >= 0.20 vendors its own click
        from typer._click import Context
    except ImportError:  # pragma: no cover
        from click import Context

    if not path:
        cmd = typer.main.get_command(cmd)
    ctx = Context(cmd, info_name=path[-1] if path else "mjolnir")
    if not hasattr(cmd, "list_commands"):
        return [(path, cmd.callback)]
    out = []
    for name in cmd.list_commands(ctx):
        out += _callbacks(cmd.get_command(ctx, name), (*path, name))
    return out


def test_every_root_and_sub_app_command_body_lives_in_a_commands_module():
    from mjolnir_cli.cli import app

    owned = [(p, cb) for p, cb in _callbacks(app) if p[0] not in ("verify", "autoresearch")]
    assert len(owned) == 7 + sum(len(v) for v in SUB_APP_COMMANDS.values())
    bad = [(p, cb.__module__) for p, cb in owned if not cb.__module__.startswith("mjolnir_cli.commands.")]
    assert bad == []


def test_mjolnir_console_script_resolves_to_cli_app():
    scripts = [ep for ep in entry_points(group="console_scripts") if ep.name == "mjolnir"]
    assert [ep.value for ep in scripts] == ["mjolnir_cli.cli:app"]
    from mjolnir_cli.cli import app

    assert scripts[0].load() is app


@pytest.mark.parametrize("sub", sorted(SUB_APP_COMMANDS))
def test_sub_app_help_exits_zero_and_lists_commands_in_registration_order(sub):
    assert _listed_commands(_invoke_help(sub)) == SUB_APP_COMMANDS[sub]


@pytest.mark.parametrize("sub", sorted(SUB_APP_COMMANDS))
def test_every_sub_app_command_help_exits_zero(sub):
    for cmd in SUB_APP_COMMANDS[sub]:
        _invoke_help(sub, cmd)


@pytest.mark.parametrize("name", sorted(ROOT_COMMAND_OPTIONS))
def test_root_command_help_exits_zero_and_shows_its_options(name):
    out = _invoke_help(name)
    assert re.search(rf"Usage: \S+ {name} \[OPTIONS\]", out)
    for opt in ROOT_COMMAND_OPTIONS[name]:
        assert opt in out


def test_root_help_exits_zero_and_lists_top_level_commands_in_current_order():
    assert _listed_commands(_invoke_help()) == [
        "gate",
        "calibrate",
        "grant-permissions",
        "drop-caches",
        "plot",
        "history",
        "version",
        "serve",
        "bench",
        "model",
        "image",
        "verify",
        "autoresearch",
        "litellm",
        "perses",
        "hw",
    ]
