# Adding a new model config

**Story:** "I want to serve/bench a model that isn't in the repo's
`configs/` — without forking the repo. The config should live on *my*
machine, under the project's home path."

## Where configs live

Two roots, both scanned by every `mjolnir` command that lists or resolves
configs:

| Root | Path | Use |
|---|---|---|
| **repo** (canonical) | `<repo>/configs/<vendor>/<model>/<quant>.yaml` | committed, reviewed, what the project ships |
| **user-local** (persisted at `~/`) | `~/.local/share/mjolnir/configs/<vendor>/<model>/<quant>.yaml` | your experiments; override with `$MJOLNIR_CONFIGS_DIR` |

Rules:

- Same layout in both: `<vendor>/<model>/<quant>.yaml`
  (e.g. `Meta/Llama-3.3-70B/NVFP4.yaml`).
- On a `(model, quant)` clash the **repo copy wins** — local never shadows a
  committed config (the repo one is canonical).
- Local configs are your personal, uncommitted state; the state file
  (`~/.mjolnir-state.json`) remembers which config is *active*, not the
  config contents.

## Create one

```bash
mkdir -p ~/.local/share/mjolnir/configs/Meta/Llama-3.3-70B
$EDITOR ~/.local/share/mjolnir/configs/Meta/Llama-3.3-70B/NVFP4.yaml
```

Minimal working config (copy a repo one — e.g.
`configs/Qwen/Qwen3.8-27B/NVFP4.yaml` — and change the model block; the
attention-backend / KV-cache / GDN sections are Thor-specific and usually
stay):

```yaml
model: Meta-Llama-3.3-70B-Instruct      # HF checkpoint to load
served-model-name: meta-llama/Llama-3.3-70B   # what the API answers to

dtype: bfloat16
attention-backend: flashinfer
kv-cache-dtype: fp8
max-model-len: 32K
gpu-memory-utilization: 0.88
# …same serving params as the closest existing config…
```

The two fields that matter to the CLI:

- `model` — the HF checkpoint vLLM loads (downloads into
  `~/.local/share/mjolnir/models/` on first launch).
- `served-model-name` — the name `mjolnir bench perf` targets in the
  OpenAI API. If you omit it, the `model:` field is used (that's vLLM's
  default served name).

## Use it — the whole flow

```bash
mjolnir model list          # it appears now, under the 'local' column
mjolnir model use Meta/Llama-3.3-70B NVFP4     # make it active (state file)
mjolnir serve up            # serves from it: the file is mounted into the
                            # container at /configs/Meta/Llama-3.3-70B/NVFP4.yaml
mjolnir bench perf          # benchmarks it (targets the served-model-name)
```

`serve up --dry-run` is the no-risk way to check the wiring: you should see
`-v …/configs/Meta/Llama-3.3-70B/NVFP4.yaml:/configs/Meta/Llama-3.3-70B/NVFP4.yaml:ro`
in the printed command.

Promote a good local config to the repo when it's worth sharing: copy the
file to `<repo>/configs/<vendor>/<model>/<quant>.yaml` and commit it — the
local copy stops shadowing anything and the repo one becomes canonical.
