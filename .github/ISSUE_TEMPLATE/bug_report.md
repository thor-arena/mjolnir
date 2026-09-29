---
name: "Bug report"
about: Something is failing: a CLI command, image build, patch, kernel verify, or bench
title: ""
labels: ["bug"]
assignees: []
---

## What happened

A clear, concise description of the failure. What command did you run and what error did you get (paste the traceback / stderr verbatim).

## What you expected to happen

## How to reproduce

Exact commands, in order, on a Thor host:

```
$ # e.g.
$ mjolnir serve up --image mjolnir/vllm-thor:qwen38-sm110-v13
$ mjolnir bench kernel gemv-ringfix
...
```

## Environment

| Field | Value |
|---|---|
| Image tag | `mjolnir image` (e.g. `mjolnir/vllm-thor:qwen38-sm110-v13`) |
| Base vLLM nightly | (e.g. `vllm/vllm-openai:v0.30.0-ubuntu2404`) |
| Model + quant config | (e.g. `configs/Qwen/Qwen3.8-27B/NVFP4_FA4hd256.yaml`) |
| `mjolnir` version | (e.g. `mjolnir --version` / git commit) |
| Thor state | output of `mjolnir hw status` (or "not run yet" if pre-setup) |

## Server / GPU state at the time

Was the live vLLM server up, and did you run inside a clean window?

```
$ mjolnir gate --once
...
```

If this is a **number** that looks wrong (not a crash): state whether the run was gated or ungated (the raw JSON carries `"gated": true/false`), and include the raw JSON — wall-clock claims only count as same-window ratios, absolute kernel claims need NCU achieved bandwidth.

## Raw artifacts

Attach the relevant raws / logs:

- [ ] bench raw JSON (`--out <path>`, or the file from `benchmarks/raw/`)
- [ ] server log / `docker logs` excerpt
- [ ] `mjolnir image build` output (for patch/build failures)
- [ ] NCU profile (for kernel perf claims)

## Anything else

Workarounds found, relevant prior issues/PRs, upstream tracker links.
