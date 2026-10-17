# AGENTS.md — working on the Mjolnir repo

This repo is **the Mjolnir decode-attention kernel for the NVIDIA Jetson AGX
Thor** (`sm_110`, 20 SMs) and everything around it: the vLLM image it ships in
(`narandill/mjolnir-vllm`), the `MJOLNIR` vLLM attention backend, and the
`mjolnir` CLI that serves, verifies and benchmarks it. Every result in the
README and `docs/kernel/` is rendered from committed raw data. This file is the
operational doc: critical rules, layout, how to work. Read it before doing any
work here.

## Critical rules (read first)

### The GPU is shared — never kill the server, wait for idle
The Thor GPU is shared with a **live vLLM server that may be serving the LLM
the agents run on**. When doing anything on the GPU:
- **NEVER** stop / restart / kill / rebuild the vLLM server (or any container
  you did not start) to "clean" the GPU.
- **Wait for a clean window** — vLLM running + waiting requests = 0 for
  several consecutive samples — before measuring. The gate is
  `src/mjolnir-cli/mjolnir_cli/gate.py` (`CleanWindow`, `wait_for_idle`,
  `server_load`); **use it, don't re-implement it.** Every `mjolnir verify`
  stage and `mjolnir bench` gates itself (`--gate`, default on);
  `mjolnir gate --once` shows the state.
- **ncu** (base clocks, cold L2, kernel replay) is the instrument for kernel
  claims; wall clock is only for **ratios measured in the same run**.
- Noise on a dedicated Thor: a same-run ratio (reference ÷ variant) moves
  **±3%** between runs (`NOISE_PCT`, cv 0.6–2.6%, median 1.25%), but absolute
  µs drift ~10% between runs at long L — compare legs measured in the same run
  and require a win across L. **L=16K is bimodal** (0.98× or 1.14× vs
  FlashInfer depending on the run): never decide on 16K alone. End-to-end
  serving numbers use the same ±3% band (`E2E_NOISE_PCT`), but with MTP the
  per-run decode rate spreads 10–15% (acceptance varies by prompt): decode
  cells need `--repeat` before a few-% difference means anything.
- **Native vLLM baselines (FlashInfer, FA2) are measured only on the official
  `vllm/vllm-openai` image**, never on ours — the Mjolnir image patches
  FlashInfer paths too. `mjolnir verify` runs reference legs there
  (`MJOLNIR_REF_IMAGE` overrides); e2e charts ignore native configs served
  from a Mjolnir image.

### Every number ships with its raw artifact
- Kernel results: `mjolnir verify <stage>` → one `mjolnir.verify/1` JSON per run
  in `benchmarks/raw/verify/<stage>/<run_id>.json` (`"default": true` only when
  every parameter is the stage default — only those feed headline charts).
- End-to-end results: `mjolnir bench perf|ab` → `benchmarks/raw/perf-<ts>/` +
  one row per run in `benchmarks/history.jsonl`.
- `README.md`, `docs/kernel/*.md` and `assets/benchmarks/` are **generated**
  (`uv run mjolnir plot --readme`) from `README.md.j2` / `docs/kernel/*.md.j2` and
  the raw data. Never hand-edit them; CI fails if they don't reproduce.

### The page-cache clear
`mjolnir_cli/cachectl.py` clears the host page cache (`sysctl -w
vm.drop_caches=3`) before `serve up` and after `serve down` — and around every
one-shot container (`dockerctl.run_once`: calibrate, image gates; after each
verify stage), since on Thor the cache a container filled stays resident after
it is removed — gated on a
*targeted* NOPASSWD grant (`mjolnir grant-permissions` installs only that
rule; `mjolnir drop-caches` clears on demand). Without the grant the clear is
skipped with a hint — nothing blocks on a password.

### Keep hand-written files under the length limit

**Keep every hand-written file within the `limit` in `[tool.mjolnir.file-length]`
(root `pyproject.toml`); every line counts.**

- When a file nears the limit, split it along its natural seam in a
  **separate commit** first.
- A grandfathered file keeps its ceiling equal to its line count, so the
  ceiling moves down as the file shrinks. Once the file is within the limit,
  set its ceiling to the limit; the pre-commit hook retires the entry.
- Before committing, run `uv run scripts/check_file_length.py --base origin/main`.

## Current state

- **The kernel** (`src/mjolnir/`, PyPI `mjolnir-kernel`): CuTe-DSL paged decode
  on `tcgen05` tensor cores — occupancy-pinned split-KV plan (one wave of
  2 CTAs × 20 SMs), dynamic fp8 Q, TMA KV ring with L2 evict-first, TMEM
  accumulators, softmax warpgroups, LSE merge kernel; q_len ≤ 1 + spec tokens;
  full CUDA graphs. Harness default: **e2m3 (fp6) KV on head-major (HND) pages,
  2-stage ring for fp6 at 2 CTAs/SM**; vLLM serves **e2m3 (fp6) KV** by default
  (`DEFAULT_QUANT` = `NVFP4_MJOLNIR_E2M3_LMH8_VT64K`: fp6 KV with calibrated
  per-channel scales, `mjolnir calibrate`, `docs/faq/fp6-kv.md`; plus an fp8
  lm_head, a 64K draft vocab and MTP=4); fp8 KV with `NVFP4_MJOLNIR`. Results: README +
  `docs/kernel/`.
- **The backend**: `vllm serve <model> --attention-backend MJOLNIR` on the image
  (enum patch `thor-mjolnir-attention-backend.patch`); `--attention-backend
  CUSTOM` on stock vLLM 0.31.0 (the `vllm.general_plugins` entry point
  `mjolnir = "mjolnir:register"` fills both slots). Prefill / mixed steps run on
  the fallback (`MJOLNIR_ATTN_FALLBACK`: FA4 on the image, FlashInfer on stock
  vLLM; the FA4 default lives in the enum patch's `fa_utils` hunk). Knobs: `MJOLNIR_ATTN_*`
  (legacy `THOR_T5_*` honored) — `src/mjolnir/README.md`.
- **The image**: `narandill/mjolnir-vllm:v0.31.0` on
  `vllm/vllm-openai:v0.31.0-ubuntu2404`; 13 vLLM patches + 1 FlashInfer patch,
  applied and verified at build time; the kernel package installed from
  `src/mjolnir`. Tags follow the **upstream vLLM version**
  (`v<vllm>`, `v<vllm>-<sha>`, `latest`), built by `docker-image.yml`.
  Inventory: `docker/vllm-thor/PATCHES.md`.
- **Configs** (`configs/Qwen/Qwen3.8-27B/`): `NVFP4_MJOLNIR_E2M3_LMH8_VT64K`
  (default: Mjolnir, fp6 KV, fp8 lm_head, 64K draft vocab, MTP=4),
  `NVFP4_MJOLNIR_E2M3` (Mjolnir, fp6 KV), `NVFP4_MJOLNIR` (Mjolnir, fp8 KV),
  `NVFP4` (FlashInfer), `NVFP4_FA2` (FlashAttention 2, bf16 KV — stock vLLM's
  choice on Thor without FlashInfer), `NVFP4_FA4hd256` (patched FA4).
  Opt-in drafter legs (`legs.OPTIONAL`, `bench e2e --optional`;
  `docs/faq/spec-drafters.md`): `NVFP4_DSPARK` / `NVFP4_DFLASH2` (official
  image) and `NVFP4_MJOLNIR_E2M3_DSPARK` / `…_DFLASH2` (Mjolnir) — DSpark or
  DFlash2, K=7, instead of MTP. DSpark on Mjolnir runs its drafter on Mjolnir
  + fp6 too: its own calibration (`<config>.kv_scales.json`, `mjolnir
  calibrate --config …`) and the kernel's non-causal variant for its block
  steps (`MJOLNIR_ATTN_NONCAUSAL`). DFlash2's sliding-window drafter stays on
  bf16 KV + FlashInfer. Not yet run on a Thor.
- **Configs** (`configs/nvidia/Qwen3.6-35B-A3B/`, MoE 256x8, hybrid
  Gated-DeltaNet + 10 full-attention layers): `NVFP4_MJOLNIR_E2M3`
  (Mjolnir, fp6 KV, MTP=5; the lm_head is already ModelOpt-quantized, so no LMH8) and `NVFP4_DFLASH` (native baseline:
  a production server's flags, fp8 KV, DFlash K=12, official image only).
  Production shape: 262K, one sequence, 0.45. Keep the Qwen3.8 extras
  (perf-mode, opt-level 3, 16K batching, TORCH_SDPA ViT, mamba align) out, and
  no draft vocab: `MJOLNIR_DRAFT_VOCAB` on this model adds ~40 GB at startup
  (#25). Open: a silent Thor hard hang with the E2M3 worker idle (#27).
- **Autoresearch**: `mjolnir autoresearch init|run|evaluate|ledger|status|export`
  + the OpenCode orchestrator `.opencode/agent/autoresearch.md` (it may delegate
  to user-defined agents `arxiv-researcher`, `web-researcher`, `python-engineer`,
  `debugger` from the user's global OpenCode config; none ship with the repo). The loop
  is `research/program.md`; every experiment is a ledger row.
- **Retired**: the FA4 GEMV decode kernel and its tooling were removed; the
  last snapshot is in git history at
  [`gemv-archive:deprecated/gemv/`](https://github.com/thor-arena/mjolnir/tree/gemv-archive/deprecated/gemv).
- **Side stacks**: `mjolnir litellm …` (LiteLLM proxy, `docker/litellm/`) and
  `mjolnir perses …` (Perses + Prometheus dashboards, `docker/perses/`), both
  independent of the vLLM server; user files under `~/.local/share/mjolnir/`.
  `mjolnir hw setup` provisions a fresh Thor (`scripts/hw/setup-thor.sh`).

## Repo layout

- `src/mjolnir/` — **the kernel package** (`mjolnir-kernel`, import `mjolnir`):
  - `mjolnir/t5_decode.py` — the CuTe-DSL kernel, split-KV merge, standalone
    harness (`python -m mjolnir.t5_decode --help`) and `emulate_decode`, the
    bit-faithful correctness reference. The single source of truth for the
    kernel; `mjolnir verify` mounts it over the image's copy.
  - `mjolnir/runner.py` — plan buffers, compile cache, launch, KV views (no vLLM).
  - `mjolnir/backend.py` — the vLLM backend / builder / impl / metadata.
  - `mjolnir/__init__.py` — `register()` (the plugin entry point), `BACKEND_NAME`.
- `src/mjolnir-cli/` — **the CLI** (`mjolnir-cli`, import `mjolnir_cli`, command
  `mjolnir`; typer):
  - `cli.py` (commands), `config.py` (defaults: `DEFAULT_QUANT`, `IMAGE_REPO`,
    `VLLM_VERSION`, `BACKEND_LABELS`; `RepoLayout`; bundled resources),
    `legs.py` (the default legs of every bench and verify stage — the one
    place to change what is measured),
    `dockerctl.py` (serve), `gate.py`, `cachectl.py`, `benchy.py` +
    `history.py` (e2e), `tasks.py` (canary tasks for `image gates`),
    `stack.py` / `litellm.py` / `perses.py`, `thor.py` (hw), `picker.py`.
  - `verify/` — `mjolnir verify`: `stages.py` (every stage, `ALL_LS`, the
    correctness matrix), `legs.py` (Mjolnir / FlashInfer / FA2 legs and ncu
    filters), `docker.py` (one container per leg), `ncu.py` (summary parsing),
    `results.py` (the `mjolnir.verify/1` schema), `latest.py` (`load_verify`:
    the latest headline run per stage), `report.py`, `cli.py`.
  - `drivers/` — the FlashInfer (NHD/HND) and FA2 benchmark drivers run in the
    image.
  - `charts.py` + `theme.py` (every chart; the design system), `labels.py`
    (config labels both charts and docs use), `readme/` (README/docs
    rendering: `__init__.py` renders and `chart()`, `fmt.py` formatting,
    `tables.py` tables, `findings.py` findings, `context.py` the template
    context, `how.py` the formula panels; layering pinned by
    `tests/cli/test_module_boundaries.py`), `interactive.py` (the Plotly
    site), `autoresearch.py`.
  - `hatch_build.py` bundles `configs/`, `docker/litellm`, `docker/perses`,
    `scripts/hw` into the wheel so the CLI works outside a checkout.
- `tests/cli/` (pytest, runs in CI) and `tests/kernel/` (needs CUDA + CUTLASS
  DSL; runs on a Thor via `mjolnir verify vllm`).
- `docker/vllm-thor/` — the image: `Dockerfile` (repo-root build context),
  `patches/`, `apply_patches.py` + `verify_patches.py` (the build gate),
  `PATCHES.md`, canary scripts.
- `configs/<vendor>/<model>/<quant>/config.yaml` — serving configs, each in its
  own dir (with a `.env` sidecar inside when it has env switches); the model's
  calibration files (`kv_scales.json`, `draft_vocab.json`) sit next to the
  config dirs. User-local configs in `~/.local/share/mjolnir/configs/`
  (`$MJOLNIR_CONFIGS_DIR`), same layout.
- `benchmarks/` — `raw/verify/` (committed kernel raws), `raw/perf-*` (e2e raws),
  `history.jsonl`, `logs/` (untracked).
- `assets/benchmarks/` — generated charts (`*.png`, `*-dark.png`, `manifest.json`).
- `README.md.j2`, `docs/kernel/*.md.j2` — the templates.
- `docs/` — case study (`kernel/`), methodology, FAQ, stack research; index
  `docs/README.md`.
- `research/` — autoresearch program, ledger (untracked while live), exported runs.
- `.opencode/` — agents (`agent/`), the `/autoresearch` command, skills
  (`kernel-iteration`, `update-base-image`).
- `.github/workflows/` — `ci.yml` (pull requests), `release.yml` (manual: tags
  + GitHub Release + dispatch), `release-kernel.yml`, `release-cli.yml`,
  `docker-image.yml`, `pages.yml` (all tag-driven); `scripts/release_plan.py`.

## How to work

### Kernel: verify and benchmark — use `mjolnir verify`, don't hand-roll `docker run`
```
mjolnir verify list                       # stages + saved runs
mjolnir verify correctness [--quick]      # vs the emulator
mjolnir verify sweep                      # Mjolnir vs FlashInfer vs FA2 (ncu)
mjolnir verify hndsweep|kvfmt|quant|ring|prefetch|pdl|ab|fp6prof|timing
mjolnir verify vllm                       # the backend inside vLLM
mjolnir verify kv-cache-size              # KV tokens per config (restarts the server per leg; refuses while one runs)
mjolnir verify spec                       # MTP acceptance per position + decode t/s for 1-4 drafts per config (same)
mjolnir verify profile                    # torch-profiler trace per config: graph replays, eager kernels, GPU idle (same)
mjolnir verify bench | all                # every stage once (+ correctness)
mjolnir verify <stage> --dry-run          # the docker commands, nothing run
mjolnir verify show <stage>               # the latest saved run
```
Default L for every stage and chart: **128 4096 8192 16384 32768 65536 131072
262144**. `--ls` narrows a run (saved as non-default). Logs:
`benchmarks/logs/verify-<run_id>/`. Workflow:
`.opencode/skills/kernel-iteration/SKILL.md`.

### Charts, README, site
```
uv run mjolnir plot --readme              # charts + README.md + docs/kernel/*.md
uv run mjolnir plot --site _site          # the interactive page (pages.yml publishes it)
```
New chart → a function in `charts.py` using `theme.py` (series colors: Mjolnir
orange, FlashInfer blue, FA2 aqua — never ad-hoc colors; light + dark variants),
registered in `render_all`; new table → `readme/tables.py` `TABLES`; reference both from
the template. Sentences that state a result come from `readme/findings.py`,
not hand-typed numbers.

### End to end
`mjolnir bench e2e` (official vLLM image vs the Mjolnir image: decode, prefill,
TTFT, ITL — feeds the README infographic; `bench compare` re-prints it);
`mjolnir bench perf` (gated llama-benchy; `perf` and `ab` default to
context depths 0–128K (`config.E2E_DEPTHS`), `e2e` to 0–32K (`e2e.MAX_CTX`: the
README table, e2e charts and infographic stop there; the infographic takes the
1-request cell with the highest decode gain), raw JSON + server log + a history row,
incl. `spec_stats`: MTP tokens per verify step and per-position acceptance from
the server's counters — decode t/s = tokens/step ÷ step time, so compare it
before blaming the kernel);
`mjolnir bench accuracy` (model quality per serving config with
tool-eval-bench, pinned commit via `uvx`: the standard 69 tool-call scenarios +
28 Hard Mode by default; anything after `--` is forwarded, e.g. `-- --gsm8k
--mmlu --ifeval --skip-tool-eval`; legs as in `bench ab`, restarts per leg;
`benchmarks/raw/accuracy/<run>/accuracy.json` → `docs/kernel/accuracy.md`);
`mjolnir bench ab` (default legs in `legs.py`; legs
are `<config>` or `<image>:<config>`) — **it restarts the server per leg**: only on
purpose, in a window you own.

### Building the image
```
mjolnir image build --tag <tag>   # docker build -f docker/vllm-thor/Dockerfile -t <tag> .
mjolnir image gates               # sm_110 gate-probe canaries (fresh container)
```
The build fails if a patch stops applying — the signal the base moved. Bumping:
`.opencode/skills/update-base-image/SKILL.md` (bump `BASE_IMAGE` **and**
`VLLM_VERSION` in `config.py` together).

### Autoresearch
```
mjolnir autoresearch init <tag>   # branch autoresearch/<tag> + baseline row
mjolnir autoresearch run          # OpenCode orchestrator; never stops by itself
mjolnir autoresearch ledger | status | chart | export
```
Agents may change only `src/mjolnir/mjolnir/t5_decode.py` and work on the fp6
(e2m3, HND) path only; the keep/discard rule is in the harness
(`autoresearch.judge`, per L against the best kept row: > 2% geomean gain that
survives dropping its best L, at most one L slower than noise, FlashInfer
outliers set aside; correctness required). `research/progress.png` is rendered
from the ledger by `mjolnir plot` / every `evaluate`.

### Releases
One manual workflow, **Actions → Release** (`release.yml`): it plans the next
versions from the tag history (`scripts/release_plan.py`), runs the CI gates,
tags `main` (`mjolnir-kernel-vX.Y.Z`, `mjolnir-cli-vX.Y.Z`, umbrella
`release-YYYY.MM.DD`), writes the GitHub Release (`scripts/release_notes.py`: one section per
shipped artifact, the PRs that touched its paths) and dispatches the tag-driven
publishers: `release-kernel.yml` / `release-cli.yml` (PyPI, trusted
publishing; `__version__` is stamped from the tag — never bump it by hand
unless you want a jump), `docker-image.yml` (tag = the pinned vLLM version) and
`pages.yml`. CI runs on pull requests only. Setup + details:
`docs/faq/releasing.md`.

## Open work
- [ ] **Headline kernel data**: `mjolnir verify bench` on the current kernel,
      commit `benchmarks/raw/verify/**`, re-render (the README shows "pending"
      for stages without a default run).
- [ ] **End-to-end**: `mjolnir bench e2e` (native FlashInfer on the official
      vLLM image vs Mjolnir) — restarts the server per leg; schedule deliberately.
- [x] **fp6 KV in vLLM — validated, now the default** (via `DEFAULT_QUANT`): the path is in
      (`thor-fp6-e2m3-kv-cache.patch`, `mjolnir/kvfp6.py`, `NVFP4_MJOLNIR_E2M3`).
      Per-head scales failed smoke e2m3 (Qwen3-0.6B: K outlier channels, amax
      486 in layer 0, rounded the other channels to 0 — wrong first token);
      scales are now per (layer, KV head, channel), `kv_scales/2` (a `/1` file
      still loads but is per head: re-calibrate after model changes:
      `mjolnir calibrate`, commit `configs/<model>/kv_scales.json`).
- [ ] **e2e at the shapes that matter**: decode gains need long contexts
      (64K+) and concurrency; `mjolnir verify sweep --batch 4 --q-len 4`
      measures the MTP=3 verify shape in ncu.
- [ ] **Decode beyond attention**: the ranked experiment list with A/B
      commands is `docs/research/decode-experiments.md` (draft-step overhead,
      fp8 lm_head, VocabTrim draft head, fp8 MTP layer, draft length, long L).
      What every bench measures by default — serving legs of `verify
      spec|kv-cache-size|profile` and `bench ab|e2e|accuracy`, kernel legs of
      the ncu stages, the `verify vllm` smoke modes — is set in ONE place:
      `src/mjolnir-cli/mjolnir_cli/legs.py`.
      Opt-in experiment configs carry their switches in the `.env` inside
      their config dir (`NVFP4_MJOLNIR_E2M3_LMH8`, `…_VT32K`, `…_LMH8_VT32K`, `…_LMH8_VT64K`;
      `mjolnir.lmhead`, `mjolnir calibrate --draft-vocab`).
- [ ] **fp6 byte-vs-time gap**: not the 64-byte L2 requests (SW128 tried: ~5% at
      ≥64K only) — the kernel is latency-bound and e2m3 has a 16K–32K hole
      (`docs/kernel/fp6prof.md`, ranked ideas in `research/program.md`).
- [x] Fuse the split-KV merge into the last CTA per sequence — tried
      (autoresearch run 1): 0.56× at 4K; retired in favour of a cheaper merge.
- [ ] fp4 with finer scales (per-channel K folded into Q, per-token V into P).

### FA4 + hd256 + FP8 KV (prefill) — blocked upstream, plan to port later
FA4's dedicated hd256 kernel is bf16-only upstream (asserts
`descale_tensors is None`), and vLLM downgrades FA4→FA2 for hd256 + any
quantized KV — so **FlashInfer remains the only FP8-KV path for the hd256
layers** (FA4+FP8 for hd≤128 *is* proven on Thor —
`docs/thor-stack/fa4-fp8kv-sm110.md`). Depth:
`docs/thor-stack/fa4-hd256-route-trace.md` (kernel/policy trace),
`docs/research/fa4-hd256-upstream-status.md` (upstream status).

Plan (as probe-gated patches, same pattern as the shipped ones):
1. Wait for (or request early access to) the upstream FP8-descale branch for
   the hd256 kernel + the paged-kv (TMA, page-128) branch; varlen is the long
   pole — check RFC #2456 status before investing.
2. Port the FP8-descale plumbing into the vendored
   `sm100_hd256_2cta_fmha_forward.py` (transcribable from the generic kernel
   `flash_fwd_sm100.py` `DescaleTensors` path, ~200–400 LOC) + relax the
   `descale_tensors is None` assert + widen the arch assert.
3. Widen vLLM's policy gates (`fa_utils.py` "quantized KV" reason, the
   `family(110)` term in `flash_attn_supports_kv_cache_dtype`).
4. sm_110 bring-up: TMEM table entry, register/tile tuning on real Thor
   silicon (all upstream tuning is B200-derived), fp8 numerics validation.
5. Config side: `block_size=128` (TMA path) for the hd256 kernel; verify the
   kernel's other forbids (no pack_gqa, no SplitKV>1, no score_mod) against
   GQA 24:4 + paged.
6. Probe-gated patch + end-to-end A/B vs the FlashInfer baseline before any
   workload switch.

## Upstream tracking

- Dao-AILab/flash-attention RFC [#2456](https://github.com/Dao-AILab/flash-attention/issues/2456)
  (owners @Johnsonms, @tzadouri):
  - FA4 hd256 fwd+bwd merged ([#2412](https://github.com/Dao-AILab/flash-attention/pull/2412), Apr 2026).
  - **FP8 input (descaled KV) for hd256: "🔨 code complete, perf to be improved"** —
    written, unmerged, not public on any branch/PR (verified 2026-09-21).
    Est. merge: Q4 2026.
   - Merged since the RFC: exp2-emu (#2488), paged-kv TMA (#2489),
     seqused_k/q (#2810), sm_110 arch gating (#2590 — the 2CTA kernels *do* run
     on sm_110; only a software postprocess bug, #2491).
   - **hd256 SplitKV fwd merged** ([#2916](https://github.com/Dao-AILab/flash-attention/pull/2916)
     + refactor [#2917](https://github.com/Dao-AILab/flash-attention/pull/2917),
     2026-09-25, squash `e9cf2c1`): static splits on the dedicated 2CTA fwd
     kernel (LSE partials, empty-split safe, varlen scheduler, paged+split
     tests; 2.0–3.9× decode on GB300). The new CTA-count heuristic is gated
     `arch // 10 in [10, 11]` (sm_110 in scope) but auto-disables on the
     20-SM Thor for our decode shapes (48 clusters > 20 SMs → 0 splits) —
     corroborates our in-tree NO-WIN. NOT the FP8-descale work (still private
     per #2456), and NOT in vLLM 0.30.0's or 0.31.0's vendored FA4 (pre-#2916), so the fa4
     patch group is unaffected. Depth:
     `docs/research/fa4-hd256-upstream-status.md` (2026-09-26 update).
   - Open: persistent-cluster scheduler (the 20-SM small-batch lever),
     sliding-window (#2749), hd256 backward + seqused (#2891), hd512 (#2877).
- FlashInfer: no hd256+FP8 work; XQA sm_110 gap open, unassigned
  ([#2522](https://github.com/flashinfer-ai/flashinfer/issues/2522)).
- vLLM: [#55366](https://github.com/vllm-project/vllm/pull/55366) (include SM110
  in FA4 auto-selection — we currently force `flash_attn_version=4`),
  [#54705](https://github.com/vllm-project/vllm/pull/54705) (fp8 e4m3 KV cache
  with per-tensor scales on SM100), [#55196](https://github.com/vllm-project/vllm/issues/55196)
  (open RFC questioning the FP8-KV memory win on GDN hybrids — our exact model
  class).
- Draft-decode cudagraphs:
  - vLLM [PR #53383](https://github.com/vllm-project/vllm/pull/53383) (open):
    triton verify builders bake `seq_lens=1` at cudagraph capture — same bug
    class as our c1 regression, different kernel. Watch for merge.
  - vLLM [#55581](https://github.com/vllm-project/vllm/issues/55581) +
    [PR #56289](https://github.com/vllm-project/vllm/pull/56289) (in progress):
    `get_cudagraph_support()` head-count mismatch can kill/limit draft
    cudagraphs — directly relevant to the DSpark draft CG grant.
  - vLLM [#49547](https://github.com/vllm-project/vllm/issues/49547): FlashInfer
    native decode caps at UNIFORM_SINGLE_TOKEN_DECODE → PIECEWISE downgrade
    under spec-decode.
  - FlashInfer [PR #3871](https://github.com/flashinfer-ai/flashinfer/pull/3871)
    (merged Jul 2026): graph-safe uniform multi-token decode; design statement
    confirms frozen plan_info depends only on (batch, total_rows, uniform_q_len)
    — the property our c1 regression study relies on
    (`docs/draft-cudagraph/c1-regression-study.md`).

## Conventions
- **New Thor patches** go in `docker/vllm-thor/patches/` (hand-adapted, **not**
  a raw `git apply`), registered in `apply_patches.py` `PATCH_ORDER`, with a
  `verify_patches.py` sentinel and a `PATCHES.md` entry. The **probe-gated
  pattern** (active only on sm_110, passive until a config opts in) is the norm.
- **Kernel changes** go behind a harness flag first; a default flips only with a
  same-run A/B win across L. Never change `emulate_decode` to make a test pass.
- **Bench discipline:** gate via `gate.py`, raw JSON first, charts second; no
  prose-only results.
- **Reporting changes:** % is the relative change of the measured value —
  throughput +x% (more tokens/s), latency −x% (fewer ms: a latency
  reduction, never the reciprocal speedup); × is a ratio of times (kernel:
  reference ÷ Mjolnir) or of throughputs (≥ +100% on the infographic). Every
  table / figure with % or × gets a `{{ how("<kind>") }}` panel
  (`readme/how.py` `_HOW`) that states the formula.
- **Lint:** `uvx ruff@0.14.14 check src/ tests/` + `format` (pre-commit runs it).
- **Container output:** containers run as root; a file one writes into a host
  mount (the checkout) is handed back with `build_run_args(..., chown=...)` /
  `dockerctl.owner_suffix` — root-owned files break the pre-commit hooks.
- **Licensing:** Apache-2.0 (root `LICENSE`, no per-file headers in our files).
  `src/mjolnir/mjolnir/t5_decode.py` is NVIDIA-CUTLASS-derived and stays
  BSD-3-Clause — never remove NVIDIA's header. New third-party code → `NOTICE`.
  `LICENSE`/`NOTICE` are copied into both packages (`tests/cli/test_license.py`
  checks they match). The image marks every file a patch changes
  (`apply_patches.py`) and ships `LICENSE`, `NOTICE` and the patches in
  `/usr/share/doc/mjolnir/`.
- **Commit style:** short, imperative, grouped by concern (see `git log`).

## Pointers
- The kernel case study: `docs/kernel/README.md`.
- Kernel package + env knobs: `src/mjolnir/README.md`.
- Patch inventory: `docker/vllm-thor/PATCHES.md`.
- Measurement discipline: `docs/methodology/benchmarking.md`.
- Docs index: `docs/README.md`; FAQ: `docs/faq/index.md`.
- The retired GEMV kernel (git history): [`gemv-archive:deprecated/gemv/`](https://github.com/thor-arena/mjolnir/tree/gemv-archive/deprecated/gemv).
