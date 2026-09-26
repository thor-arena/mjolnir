# Changing the default model / quant config

**Story:** "I want the CLI to serve and bench *this* model with *this*
quantization config from now on — I don't want to pass `--model`/`--config`
every time."

## Pick / remember the active model + config

```bash
mjolnir model                          # arrow-key picker over every config
mjolnir model list                     # table, ● = active; also --json for scripts
mjolnir model use Qwen/Qwen3.8-27B NVFP4_FA4hd256
mjolnir model use Qwen/Qwen3.8-27B     # one config per model → auto-selected
```

Both values are persisted to the state file (`~/.mjolnir-state.json` by
default). After that, `mjolnir serve up`, `mjolnir bench perf`, `mjolnir
verify`, … all target the active model/config.

What a "config" is: a per-model serving yaml —
`configs/<vendor>/<model>/<quant>.yaml` (repo) or the user-local dir
([new-model-config.md](new-model-config.md)). The same model usually has
several configs (e.g. `NVFP4` = FlashInfer baseline, `NVFP4_FA4hd256` =
FA4 + GEMV decode kernel); picking a quant *is* picking the backend.

## One-off (state untouched)

```bash
mjolnir serve up --model Qwen/Qwen3.8-27B --config NVFP4
MJOLNIR_MODEL=… MJOLNIR_QUANT=… mjolnir bench perf
```

Precedence: **CLI flag > state file > `$MJOLNIR_MODEL` / `$MJOLNIR_QUANT` >
baked-in defaults** (`DEFAULT_MODEL` / `DEFAULT_QUANT` in
`src/mjolnir/config.py`). The backend *label* for history rows and charts is
looked up from the quant name in `BACKEND_LABELS` (`NVFP4_FA4hd256` →
"FA4-GEMV", `NVFP4` → "FlashInfer").
