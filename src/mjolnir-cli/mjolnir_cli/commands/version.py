"""``mjolnir version``: print the CLI version."""

from __future__ import annotations

import typer

from mjolnir_cli import __version__


def version():
    """Print the version."""
    typer.echo(f"mjolnir {__version__}")
