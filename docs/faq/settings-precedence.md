# Settings: what wins, and where the state lives

**Story:** "I passed a flag, set an env var, and changed the state — which
one wins? Where is each thing persisted? How do I script this non-interactively?"

## The precedence chain (high → low)

```
CLI flag  >  state file  >  $MJOLNIR_* env var  >  baked-in default (src/mjolnir/config.py)
```

| Setting | Flag | Env var | Baked-in default |
|---|---|---|---|
| model | `--model` | `MJOLNIR_MODEL` | `Qwen/Qwen3.8-27B` (`DEFAULT_MODEL`) |
| quant/config | `--config` | `MJOLNIR_QUANT` | `NVFP4_FA4hd256` (`DEFAULT_QUANT`) |
| image | `--image` | `MJOLNIR_IMAGE` | `mjolnir/vllm-thor:qwen38-sm110-v13` (`DEFAULT_IMAGE`) |
| port | `--port` | `MJOLNIR_PORT` | `6001` |
| GEMV kernel on/off | `--gemv/--no-gemv` | `VLLM_FA4_HD256_GEMV` (in-container) | on (it's the point) |

## The state file

`mjolnir model use` and `mjolnir image use` persist the active selection to
a small JSON state file — **not** an env file:

```bash
cat ~/.mjolnir-state.json
# {"model": "Qwen/Qwen3.8-27B", "quant": "NVFP4_FA4hd256",
#  "image": "mjolnir/vllm-thor:qwen38-sm110-v13"}
```

- Location: `$MJOLNIR_STATE`, default `~/.mjolnir-state.json`. Point it at
  another path (or a throwaway file) for experiments.
- It stores **selections only** — model, quant, image. Config *contents*
  live in the config roots (repo `configs/` + user-local
  `~/.local/share/mjolnir/configs/`); see [new-model-config.md](new-model-config.md).

## Project data paths (the `~/` footprint)

| Path | Env override | Content |
|---|---|---|
| `~/.mjolnir-state.json` | `MJOLNIR_STATE` | active model/quant/image |
| `~/.local/share/mjolnir/` | `MJOLNIR_DATA` | models, FlashInfer/Triton/inductor caches, tiktoken |
| `~/.local/share/mjolnir/configs/` | `MJOLNIR_CONFIGS_DIR` | user-local model configs |
| `<repo>/benchmarks/` | `MJOLNIR_BENCHMARKS` | raw JSONs + `history.jsonl` |

(`MJOLNIR_REPO` pins the repo root for installed-CLI users; the repo is
otherwise located by walking up from the CWD.)

## Scripting (no TTY, no picker)

The arrow-key pickers (`mjolnir model`, `mjolnir image` bare) **refuse to
run non-interactively** — they print the table and exit 4 telling you to
pass arguments. So for scripts:

```bash
mjolnir model use Qwen/Qwen3.8-27B NVFP4_FA4hd256   # explicit, idempotent
mjolnir model list --json | jq -r '.configs[].model'
mjolnir serve up --dry-run                            # the exact docker run, no side effects
mjolnir bench kernel gemv-bench --dry-run
mjolnir history --json
```

Everything that emits data has a `--json` mode; everything that touches
docker or the GPU has a `--dry-run`. Exit codes: 0 ok, 1 failed, 2
unreachable/preflight, 3 gate timeout, 4 not-found/bad-usage.
