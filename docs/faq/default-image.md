# Changing the default vLLM image

**Story:** "I want `mjolnir serve up` / `mjolnir bench …` to use *this*
image from now on (not the baked-in default), and to see what's available
locally."

## Pick / remember the active image

```bash
mjolnir image             # arrow-key picker over every local docker image
mjolnir image list        # list with ● on the active one
mjolnir image use mjolnir/vllm-thor:qwen38-sm110-v13   # non-interactive
```

The choice is written to the state file (`~/.mjolnir-state.json` by default;
see [settings-precedence.md](settings-precedence.md)). Every later
`serve` / `bench` / `verify` / `bench kernel` picks it up automatically —
no flags needed.

## One-off (state untouched)

```bash
mjolnir serve up --image narandill/vllm-thor:qwen38-sm110-v12-gemv
MJOLNIR_IMAGE=narandill/vllm-thor:qwen38-sm110-v9 mjolnir bench perf
```

Precedence is always: **CLI flag > state file > `$MJOLNIR_IMAGE` > baked-in
default** (`DEFAULT_IMAGE` in `src/mjolnir/config.py`).

## Build a new one

```bash
mjolnir image build --tag mjolnir/vllm-thor:qwen38-sm110-v14
mjolnir image gates                    # sm_110 gate-probe canaries, fresh container
```

- `build` = `docker build -t <tag> docker/vllm-thor/`; the build applies and
  verifies the 14-patch stack and **fails** if a patch stops applying
  (that's your "the base moved" signal —
  [base-image-bump.md](base-image-bump.md)).
- `gates` runs the sm_110 gate-probe canaries (GDN prefill, FA4 FP8-KV,
  draft-CG, hd256) against the image in a **fresh container — no server
  needed**, no impact on the live server.
- After a green build + canary: `mjolnir image use <new-tag>` makes it
  active. Keep the old tag around for A/B reference.
