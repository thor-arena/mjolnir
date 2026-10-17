"""The latest default (headline) run of every verify stage — the data the
charts, the README and the interactive site are rendered from."""

from __future__ import annotations

from mjolnir_cli.config import RepoLayout
from mjolnir_cli.verify import results
from mjolnir_cli.verify.stages import STAGES


def load_verify(layout: RepoLayout) -> dict[str, dict]:
    """The latest default run of every stage that has one."""
    out = {}
    for name in STAGES:
        doc = results.latest(layout.verify_dir, name)
        if doc:
            out[name] = doc
    return out
