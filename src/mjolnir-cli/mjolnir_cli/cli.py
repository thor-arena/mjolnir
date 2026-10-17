"""The ``mjolnir`` CLI — one command surface for the whole repo.

mjolnir hw setup|status             pre-configure the Thor host (HW
                                    provisioning, idempotent)
mjolnir serve up|down|status|logs    drive the vLLM container
mjolnir model [use|list]            active model/config (bare = arrow-key
                                    picker over configs/)
mjolnir image [use|list|build|gates] active vLLM image (bare = arrow-key
                                    picker over local docker images)
mjolnir verify <stage>              kernel correctness + ncu benchmarks (Mjolnir vs
                                    FlashInfer vs FA2) -> benchmarks/raw/verify
mjolnir autoresearch ...            the autonomous kernel-research loop (OpenCode
                                    agents, every experiment on the ledger)
mjolnir gate                        the clean-window gate
mjolnir bench perf|ab               end-to-end perf / A/B (vLLM server)
mjolnir plot|history                charts (+ README.md with --readme) + the raw-data log
mjolnir litellm up|down|status|logs  the LiteLLM proxy stack (independent
                                    of the vLLM server)
mjolnir perses up|down|status|logs   the Perses + Prometheus observability
                                    stack (independent of the vLLM server)

This module is the composition root: it builds ``app``, mounts the sub-apps
and registers the root commands in the order ``--help`` lists them. Every
command body lives in ``mjolnir_cli.commands``.
"""

from __future__ import annotations

import typer

from mjolnir_cli import autoresearch
from mjolnir_cli.commands import (
    _common,
    bench,
    calibrate,
    hw,
    image,
    litellm,
    model,
    perses,
    results,
    serve,
    version,
    window,
)
from mjolnir_cli.verify import cli as verify_cli

__all__ = ["app"]

_NEXT_STEPS = """
Next steps:
  mjolnir serve up         serve the model (baked-in defaults: Qwen3.8-27B NVFP4,
                           --attention-backend MJOLNIR, port 6001)
  mjolnir serve status     container + health + request queue
  mjolnir verify sweep     Mjolnir vs FlashInfer vs FA2 at every L (ncu) -> raw data
  mjolnir verify all       correctness + every benchmark stage -> charts + README
  mjolnir bench ab         end-to-end A/B: Mjolnir vs FlashInfer vs FA2 configs
  mjolnir model / image    pick another model config / serving image
"""


def _root_group() -> type[typer.core.TyperGroup]:
    """The root group: bare ``mjolnir`` prints the help plus a next-steps
    guide instead of just the help."""

    class RootGroup(typer.core.TyperGroup):
        def invoke(self, ctx):
            if not ctx._protected_args:
                ctx.command.get_help(ctx)
                print(_NEXT_STEPS, flush=True)
                return
            return super().invoke(ctx)

    return RootGroup


app = typer.Typer(
    context_settings=_common.HELP_CTX,
    cls=_root_group(),
    help="Mjolnir — a decode-attention kernel for Jetson AGX Thor (sm_110) and the "
    "vLLM image that serves it: serve / verify / bench / plot.",
)
app.add_typer(serve.serve_app, name="serve")
app.add_typer(bench.bench_app, name="bench")
app.add_typer(model.model_app, name="model")
app.add_typer(image.image_app, name="image")
app.add_typer(verify_cli.app, name="verify")
app.add_typer(autoresearch.app, name="autoresearch")
app.add_typer(litellm.litellm_app, name="litellm")
app.add_typer(perses.perses_app, name="perses")
app.add_typer(hw.hw_app, name="hw")

# Root commands, registered in the order --help lists them.
app.command()(window.gate)
app.command()(calibrate.calibrate)
app.command("grant-permissions")(window.grant_permissions)
app.command("drop-caches")(window.drop_caches)
app.command()(results.make_plot(app))
app.command()(results.history)
app.command()(version.version)


if __name__ == "__main__":
    app()
