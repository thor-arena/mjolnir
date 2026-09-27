# Running the LiteLLM proxy in front of vLLM

**Story:** "I want the LiteLLM proxy (with its bundled postgres + redis)
running next to my vLLM server — driven by `mjolnir` like everything else,
and **independent**: stopping vLLM leaves the proxy up, and stopping the
proxy never touches vLLM (or the GPU)."

## Why it's a separate stack

The proxy is plain third-party software (no Thor patches, no GPU) — three
containers from `docker/litellm/docker-compose.yaml`:

| service        | role                                                        |
|----------------|-------------------------------------------------------------|
| `litellm`      | the proxy (OpenAI-compatible API + web UI)                  |
| `litellm_db`   | postgres 16 — spend logs + UI state (persistent)           |
| `litellm_redis`| redis — response cache + router (persistent)               |

The proxy reaches the vLLM server through `host.docker.internal` (the
host-published vLLM port, default 6001) — the two stacks never share a
docker network, so neither knows the other's container name and neither
dies when the other goes down.

## Setup (once)

All user-persisted litellm files live in one dir — the litellm data dir,
`~/.local/share/mjolnir/litellm/` (`$MJOLNIR_DATA/litellm` to move it):

| file | role |
|------|------|
| `litellm_config.template.yaml` | **required** — your config template |
| `litellm.env` | optional — keys, ports, `${...}` template variables |
| `litellm_config.yaml` | the rendered config (written on each `up`) |
| `db/`, `redis/` | postgres + redis state (persistent) |

1. **Template** — start from the example and edit it for your topology.
   `mjolnir litellm` only works while the template exists:

   ```bash
   cp docker/litellm/litellm_config.template.example.yaml \
      ~/.local/share/mjolnir/litellm/litellm_config.template.yaml
   ```

   (override the location with `$MJOLNIR_LITELLM_TEMPLATE`).

2. **Env file (optional)** — `~/.local/share/mjolnir/litellm/litellm.env`
   (`$MJOLNIR_LITELLM_ENV` to override): master key, UI credentials, proxy
   port, redis/postgres settings, and the `${...}` variables for any
   task-model sections you uncommented. Every value has a baked-in default —
   see `docker/litellm/litellm.variables.example`.

## The commands

```bash
mjolnir litellm up         # render config + start the stack (proxy+db+redis)
mjolnir litellm status     # container states + proxy liveliness + config path
mjolnir litellm logs -f    # tail the proxy logs
mjolnir litellm down       # stop the stack (data kept — postgres/redis survive)
mjolnir litellm down --purge   # stop + wipe the stack data
mjolnir litellm config     # render-only: validate the template, no containers
mjolnir litellm up --dry-run   # show the compose command, do nothing
```

`up` re-renders the config on every run, so **switching the active model
(`mjolnir model`) + `mjolnir litellm up` is the whole workflow** — the
proxy's `${MAIN_LLM_MODEL}` always tracks the currently selected
model/config.

## What gets injected (vs. what you write)

Injected from the active model/config on every `up` (CLI selection wins):

| variable                        | value                                             |
|---------------------------------|---------------------------------------------------|
| `${MAIN_LLM_MODEL}`             | the served name (`served-model-name` of the config) |
| `${MAIN_LLM_MODEL_NAME}`        | the raw `model:` field of the config yaml         |
| `${RAW_MODEL_SUFFIXED}`         | `<model>-<quant>`                                 |
| `${LITELLM_MAIN_MODEL_BASE_URL}`| `http://host.docker.internal:<vLLM port>/v1`      |

Everything else (`${TASKS_LLM_MODEL}`, `${EMBEDDINGS_MODEL}`, …) resolves
from your env file. Any **unresolved** variable on a non-comment line makes
`up` fail with the list of missing names — commented-out template sections
may keep `${...}` placeholders.

## Ports

The proxy listens on host port **8000** (`--port` / `LITELLM_PORT` to
change); the vLLM server keeps its own port (default 6001). They never
clash. Postgres and redis are internal to the stack (exposed, not
published).

```bash
curl http://127.0.0.1:8000/v1/models -H "Authorization: Bearer <master key>"
```

## Independence guarantees

- `mjolnir litellm down` → only the three proxy-stack containers stop;
  vLLM keeps serving.
- `mjolnir serve down` → vLLM stops; the proxy stays up (its main model
  simply reports unhealthy until vLLM is back; the other model entries in
  your template still work).
- The stack never restarts, rebuilds, or depends on the vLLM container —
  same rule as the gate: the server is never killed to clean the GPU.

## Data

The data dir (see the table above) holds everything mjolnir persists for
the stack; `down` keeps it (restarts are fast, spend logs survive),
`down --purge` wipes the `db/` + `redis/` state.
