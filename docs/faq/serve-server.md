# Starting, stopping, and inspecting the server

**Story:** "I want to serve the active model with the active image, check
that it's healthy, and look at what's in the queue — without hand-rolling
`docker run`."

## The commands

```bash
mjolnir serve up          # start (active model/config/image; baked-in defaults if no state)
mjolnir serve status       # container state + /health + queue load (running/waiting)
mjolnir serve logs -f      # tail the vLLM container logs
mjolnir serve down         # stop the container
```

`serve up` waits for `/health` before returning (first launch takes minutes:
model download + FlashInfer/Triton JIT on Thor). It never touches a running
server's state — see the
[clean-window gate](clean-window-gate.md) for why killing the server to
"clean the GPU" is forbidden.

## One-off overrides (no state change)

```bash
mjolnir serve up --model Qwen/Qwen3.8-27B --config NVFP4 \
                 --image mjolnir/vllm-thor:qwen38-sm110-v13 \
                 --port 6001 --no-gemv
mjolnir serve up --dry-run     # print the exact docker run, do nothing
```

- `--gemv/--no-gemv` — route M=1 hd256 decode to the GEMV kernel
  (`VLLM_FA4_HD256_GEMV=1`/`0`); default on.
- `--wait/--no-wait`, `--wait-timeout` — health-wait behavior.

## What a first launch does

- Downloads the HF checkpoint into `~/.local/share/mjolnir/models/`
  (`--download-dir /data/models/huggingface` in the container).
- Keeps the FlashInfer JIT cache, Triton cache, and kernel cache on the
  persistent `~/.local/share/mjolnir/` volumes, so **restarts are fast** and
  the one-time JIT cost is paid once.
- Mounts `configs/` read-only at `/configs` inside the container; the serve
  command is `vllm serve --config /configs/<model>/<quant>.yaml …`
  (add your local config: [new-model-config.md](new-model-config.md)).

## Sanity check after up

```bash
mjolnir serve status     # expect: running + health ok + queue 0/0
mjolnir gate --once      # expect: CLEAN
```
