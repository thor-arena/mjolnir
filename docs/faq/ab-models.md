# A/B: different models against each other

**Story:** "I want a fair end-to-end throughput comparison of two models on
the same Thor — e.g. Qwen3.8-27B vs Llama-3.3-70B — with the same
methodology as everything else in this repo."

## Step 1 — one config per model

Models aren't compared directly; **configs** are (the config carries the
serving parameters, the backend, the KV dtype). Make sure each model has a
config — repo or user-local
([new-model-config.md](new-model-config.md)):

```bash
mjolnir model list        # see what exists (● active, 'repo'/'local' column)
```

`mjolnir bench perf --model <vendor/Model>` targets a model by its config
directory, so both models' configs must exist.

## Step 2 — the A/B

Legs can be model-qualified: `<model>/<config>` (and
`<image>:<model>/<config>` to pin the image too). One command, server
restarted per leg, gated perf bench per leg:

```bash
mjolnir bench ab --backends \
  'Qwen/Qwen3.8-27B/NVFP4_FA4hd256,Meta/Llama-3.3-70B/NVFP4'
```

Both legs serve from the active image (override per leg with the
`<image>:<model>/<config>` form — useful when you're comparing *and* the
images differ). Or run the legs by hand if you want to inspect the server
between them:

```bash
mjolnir bench perf --model Qwen/Qwen3.8-27B --config NVFP4_FA4hd256
mjolnir serve down && mjolnir serve up --model Meta/Llama-3.3-70B --config NVFP4
mjolnir bench perf --model Meta/Llama-3.3-70B --config NVFP4
```

## What makes the comparison fair

- **Same protocol:** `bench ab` guarantees it — same `--runs`/`--repeat`,
  same contexts/concurrency, `--exact-tg` on, every sweep in its own
  clean window. Don't hand-roll a different `--pp/--tg` per model.
- **Same image** (unless the image *is* the variable — that's
  [ab-image.md](ab-image.md)).
- **`served-model-name`:** `bench perf` targets the name from each config's
  `served-model-name` (or its `model:` field). If two configs serve the
  *same* name, the history rows are indistinguishable — give each model a
  distinct served name.
- **Memory budget differs per model:** each config's
  `gpu-memory-utilization` / `max-model-len` / batch caps are its own; that's
  part of what you're comparing (how the model's shape + serving params map
  onto one 128 GB Thor).

## Read it

```bash
mjolnir history            # each row-set is labeled with model + config + backend
mjolnir plot               # re-renders assets/benchmarks/*.png from history.jsonl
```

History rows carry `model` + `config` + `image` + `backend`, so a model A/B
and an image A/B accumulate in the *same* log — compare `tg_tps.mean` per
cell (context × concurrency) across the row-sets; raw per-run values live
in `benchmarks/raw/perf-<ts>/`.
