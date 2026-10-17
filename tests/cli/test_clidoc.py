"""clidoc: docs/cli.md and the README's command table come from the typer commands."""

from __future__ import annotations

from mjolnir_cli import clidoc
from mjolnir_cli.cli import app


def test_reference_covers_every_command():
    cmds = clidoc.walk(app)
    ref = clidoc.reference(app)
    assert len(cmds) > 5
    for c in cmds[1:]:
        assert f"## `{c.path}`" in ref
    assert "<run>" not in ref.replace("`<run>`", "")  # placeholders escaped outside code


def test_readme_overview_names_real_commands():
    from pathlib import Path

    t = (Path(__file__).resolve().parents[2] / "README.md.j2").read_text()
    block = t[t.index("## The CLI") : t.index("## Contributing")]
    named = clidoc.block_commands(block)
    have = {c.path for c in clidoc.walk(app)}
    assert len(named) > 10
    assert [n for n in named if n not in have] == []  # a renamed command breaks the README


def test_block_parser():
    block = """
mjolnir                               # help
mjolnir serve [up|down]               # x
mjolnir bench perf | ab               # y
mjolnir calibrate [--draft-vocab]     # z
mjolnir verify <stage>                # w
"""
    assert clidoc.block_commands(block) == [
        "mjolnir",
        "mjolnir serve up",
        "mjolnir serve down",
        "mjolnir bench perf",
        "mjolnir bench ab",
        "mjolnir calibrate",
        "mjolnir verify",
    ]


def test_rst_conversion():
    assert clidoc._rst("``a <b>`` and <c>") == "`a <b>` and &lt;c&gt;"


def test_write_reference_writes_exactly_the_reference_to_docs_cli_md(tmp_path):
    path = clidoc.write_reference(tmp_path, app)
    assert path == tmp_path / "docs" / "cli.md"
    assert path.read_text() == clidoc.reference(app)
