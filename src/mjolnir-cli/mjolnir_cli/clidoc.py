"""The CLI reference, generated from the typer commands themselves.

``mjolnir plot --readme`` writes ``docs/cli.md`` (every command and sub-command
with its description, arguments and options, defaults included), so the docs
can never drift from the code: change a command's docstring or an option's
help and the next render carries it — CI fails if the committed copy is stale.
The README keeps a short hand-written overview; :func:`block_commands` lets a
test check that every command it names exists.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:  # typer >= 0.20 vendors its own click
    from typer._click import Context
except ImportError:  # pragma: no cover
    from click import Context


@dataclass
class Cmd:
    path: str  # "mjolnir serve up"
    short: str  # first line of the help
    help: str  # the whole help text
    params: list[Any]
    group: bool


def _clean(text: str | None) -> str:
    import inspect

    return inspect.cleandoc(text or "").replace("\b\n", "").strip()


def _rst(text: str) -> str:
    """``x`` -> `x`, and <placeholders> outside code spans escaped (GitHub would
    swallow them as HTML tags)."""
    text = re.sub(r"``([^`]+)``", r"`\1`", text)
    parts = re.split(r"(`[^`]*`)", text)
    return "".join(p if p.startswith("`") else p.replace("<", "&lt;").replace(">", "&gt;") for p in parts)


# options many commands share, documented once here (their typer help is empty)
COMMON_HELP = {
    "--model": "the model (`vendor/Model`; default: the active one, `mjolnir model`)",
    "--config": "the model config (default: the active one)",
    "--image": "the vLLM image (default: the active one, `mjolnir image`)",
    "--port": "the server port (default: the active one)",
    "--dry-run": "print what would run, run nothing",
    "--gate": "wait for a clean window (vLLM queue 0/0) before measuring",
    "--plot": "re-render the charts / README afterwards",
    "--timeout": "seconds before a step is given up",
}


def _short(cmd: Any) -> str:
    """The first paragraph of the help, on one line."""
    text = _clean(cmd.short_help or cmd.help)
    return _rst(text.split("\n\n", 1)[0].replace("\n", " ").strip())


def walk(app) -> list[Cmd]:
    """Every command, depth first, in the order the CLI lists them."""
    import typer.main

    root = typer.main.get_command(app)
    out: list[Cmd] = []

    def visit(cmd: Any, path: str, ctx: Any) -> None:
        if getattr(cmd, "hidden", False):
            return
        group = hasattr(cmd, "list_commands")
        out.append(Cmd(path, _short(cmd), _rst(_clean(cmd.help)), list(cmd.params), group))
        if group:
            for name in cmd.list_commands(ctx):
                sub = cmd.get_command(ctx, name)
                if sub is not None:
                    visit(sub, f"{path} {name}", Context(sub, info_name=name, parent=ctx))

    visit(root, "mjolnir", Context(root, info_name="mjolnir"))
    return out


def _is_option(p: Any) -> bool:
    return getattr(p, "param_type_name", "") == "option"


def _default(p: Any) -> str:
    d = p.default
    if callable(d) or d is None or d == () or d == []:
        return ""
    if _is_option(p) and getattr(p, "is_flag", False) and p.secondary_opts:
        return p.opts[0] if d else p.secondary_opts[0]
    if isinstance(d, bool):
        return "on" if d else "off"
    return f"`{d}`"


def _param_row(p: Any) -> list[str] | None:
    if _is_option(p):
        if p.hidden or p.name in ("help",):
            return None
        names = " / ".join(f"`{o}`" for o in [*p.opts, *p.secondary_opts])
        help_ = _rst(p.help or COMMON_HELP.get(p.opts[0], "")).replace("\n", " ").replace("|", "\\|")
    else:
        names = f"`{(p.human_readable_name or p.name or '').upper()}`"
        help_ = "argument"
    return [names, help_, _default(p)]


def reference(app) -> str:
    """docs/cli.md."""
    cmds = walk(app)
    lines = [
        "# The `mjolnir` CLI reference",
        "",
        "> Generated from the commands themselves by `mjolnir plot --readme` "
        "(`mjolnir_cli/clidoc.py`) — do not edit by hand. `mjolnir <command> --help` prints the same.",
        "",
        "| command | what it does |",
        "|---|---|",
    ]
    for c in cmds[1:]:
        anchor = c.path.replace(" ", "-")
        lines.append(f"| [`{c.path}`](#{anchor}) | {c.short.replace('|', chr(92) + '|')} |")
    for c in cmds[1:]:
        lines += ["", f'<a id="{c.path.replace(" ", "-")}"></a>', "", f"## `{c.path}`", ""]
        if c.help:
            lines += [c.help, ""]
        rows = [r for r in (_param_row(p) for p in c.params) if r]
        if rows:
            lines += ["| option | description | default |", "|---|---|---|"]
            lines += ["| " + " | ".join(r) + " |" for r in rows]
            lines.append("")
        if c.group:
            lines.append("Sub-commands: " + ", ".join(f"[`{s.path}`](#{s.path.replace(' ', '-')})"
                                                     for s in cmds if s.path.rsplit(" ", 1)[0] == c.path) + ".")  # fmt: skip
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_reference(root: Path, app) -> Path:
    """Write :func:`reference` to ``<root>/docs/cli.md``; returns the path."""
    path = root / "docs" / "cli.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(reference(app))
    return path


def block_commands(block: str) -> list[str]:
    """Command paths named in a README-style overview block, one per line:
    ``mjolnir serve [up|down]  # …`` -> ``mjolnir serve up``, ``mjolnir serve
    down``; ``mjolnir bench perf | ab`` -> ``mjolnir bench perf``, ``mjolnir
    bench ab``. Options (``--x``, ``[--x]``), placeholders (``<stage>``) and
    comments are ignored."""
    out: list[str] = []
    for line in block.splitlines():
        line = line.split("#", 1)[0].strip()
        if line != "mjolnir" and not line.startswith("mjolnir "):
            continue
        words = []
        for w in line.split()[1:]:
            if w.startswith(("-", "<", "[-")):
                break
            words.append(w)
        text = " ".join(words)
        sub = re.search(r"\[([^\]]+)\]", text)
        if sub:  # mjolnir serve [up|down]: sub-commands of the prefix
            prefix = text[: sub.start()].split()
            out += [" ".join(["mjolnir", *prefix, a]) for a in sub.group(1).split("|")]
            continue
        alts = [a.split() for a in text.split("|")] if text else [[]]
        head = alts[0][:-1]  # mjolnir verify bench | all: siblings of the last word
        out += [" ".join(["mjolnir", *head, a[-1]] if a else ["mjolnir"]) for a in alts]
    return list(dict.fromkeys(out))
