# Kernel benches & correctness suites (without hand-rolling docker)

**Story:** "I want to run a kernel-level verification or benchmark. What's
available, which ones need the server, and how do I get raw numbers out?"

## List the tasks

```bash
mjolnir bench kernel dummy --help-list
```

| kind | means |
|---|---|
| `test` | fresh container against the image, **no server** (GPU ones self-skip on a box without CUDA) |
| `bench` | **gates on the vLLM server** (server offline → the run proceeds ungated); the GEMV'd vfa tree is mounted, the workdir at `/p`, `benchmarks/raw` at `/raw`, host networking |
| `host` | the gate itself |

Tasks (as of the v13 stack):

```
gates           [test ]  sm_110 gate-probe canary (T6 GDN, T9 FA4, T10 draft-CG, T11 hd256)
fi-update       [test ]  fused draft-decode FI kernel (advance plan vs numpy ref)
dspark-nc       [test ]  DSpark non-causal draft cudagraph
functional      [test ]  55390/55519 draft-group annotation + warning check (no GPU)
gate            [host ]  clean-window gate
gemv-verify     [bench]  GEMV Phase-A correctness vs fp32 ref (15 cases)
gemv-verify-b1  [bench]  GEMV Phase-B1 (SplitKV partials, paged, combine)
gemv-stages     [bench]  GEMV stages 2 vs 16 vs fp32 ref (ring fix bit-neutral)
gemv-bench      [bench]  GEMV 4-mode bench (gemv_dense/gemv_paged/fa4_1cta/flashinfer)
gemv-ringfix    [bench]  ns sweep + stages contrast + FI in one gated window
gemv-ncu        [bench]  ncu driver (NCU_L/NCU_NS/NCU_STAGES/NCU_ITERS env)
```

## Running them

```bash
# no server, no GPU (CI-friendly):
mjolnir bench kernel functional

# sm_110 gate-probe canaries in a fresh container (no server needed):
mjolnir image gates            # == mjolnir bench kernel gates

# gated benches — server must be up (mjolnir serve up), and the vfa tree:
mjolnir vfa prepare
mjolnir verify                          # all GEMV correctness suites
mjolnir bench kernel gemv-bench --mode all
```

## Getting raw numbers (the grep-able contract)

A **bare run** (no `--out`) auto-writes the GEMV benches into
`benchmarks/raw/` — `gemv-bench` → `gemv-decode-bench-<mode>.json`,
`gemv-ringfix` → `gemv-ring-fix-bench.json` — so the committed raws the
charts read stay current. `--out` overrides (the task's own flags pass
through verbatim; launcher flags like `--image/--dry-run` can appear
anywhere):

```bash
mjolnir bench kernel gemv-ringfix
grep -E '"(median|p95|bw_gbs_nominal_kv|clean)"' benchmarks/raw/gemv-ring-fix-bench.json
```

- `--out` is a **container** path: the workdir `docker/vllm-thor/fa4-gemv-kernel/`
  is mounted at `/p` and `benchmarks/raw/` at `/raw`, so the file lands back
  in the matching host directory.
- Preview any launch without touching the GPU: `--dry-run` (prints the
  `docker run` argv).
- Point a bench at your iteration tree: `--vfa-tree vfa-tree/vllm_flash_attn`.
- Preflight is automatic for `bench` tasks: if the metrics endpoint is down
  the launcher notes it and the bench runs **ungated** (server offline →
  nothing co-located to gate against, marked `"gated": false` in the JSON) —
  the launcher never fails for an offline server and never restarts it.

Methodology for what a kernel number means (gate, ncu vs wall, same-window
ratios): [gemv-kernel-improvement.md](gemv-kernel-improvement.md) and
[clean-window-gate.md](clean-window-gate.md).
