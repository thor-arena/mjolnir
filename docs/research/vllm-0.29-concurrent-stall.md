# vLLM 0.29 Release-Delta Notes — Concurrent-Request Stall Investigation

> **Status:** Release-delta notes (working research; **no conclusions reached** — the investigation was a first-pass mapping of the 0.29.0 delta against the symptom, and the open questions below remain open) · **Date:** 2026-09-19 · **Scope:** vLLM v0.29.0 release delta, scoped to the five areas most likely to affect a concurrent-request stall on the Thor deployment (API server/request lifecycle, async scheduling, multiprocess main server, MTP spec-decode, reasoning/tool-call parsers).

## TL;DR

- v0.29.0 was published **2026-09-09** (594 commits / 277 contributors); v0.28.0 on 2026-08-26; v0.27.1 on 2026-08-11.
- The **Rust frontend is new in 0.29.0**, and it is the most likely home for streaming/keep-alive behavior changes: optimized SSE hot path (#51321), **SSE keep-alive comments for idle streams (#51034)** — directly relevant to idle streaming — plus gRPC inputs, `--generation-config vllm`, and pure-Rust protox replacing protoc (#52892).
- **Model Runner V2 (MRV2) became the default for all models** (#53183). The release notes state that certain features — including **certain speculative decoding methods** — are not yet supported in MRV2, and vLLM falls back to MRV1 when they are configured. Whether the Qwen3.8-27B + MTP configuration runs on MRV2 or falls back to MRV1 was an open question.
- MTP/spec-decode changes in 0.29.0 include padded FULL cudagraph dispatch for uniform decode under spec decode (#53407), DP-sync skipping before EAGLE/MTP draft prefill (#53694), and spec decode no longer padded up to `max_model_len` (#53962).
- Async scheduling and the multiprocess "main server" (`VLLM_MAIN_SERVER_PORT`, `VLLM_HTTP_TIMEOUT_KEEP_ALIVE`) are **not named in the 0.29.0 release notes**; their status in 0.29.0 was left as open questions.

## Release timeline

- v0.29.0 — published 2026-09-09 (594 commits / 277 contributors)
- v0.28.0 — published 2026-08-26
- v0.27.1 — 2026-08-11; v0.27.0 — 2026-08-10
- v0.26.0 — 2026-07-27

## 0.29.0 delta, scoped to the stall-relevant areas

### (a) API server / request lifecycle

- **Rust frontend is NEW in 0.29.0**: HY3 unified parser, gRPC inputs, gRPC LoRA lifecycle, `--generation-config vllm`, `truncate_prompt_tokens`, OpenAI edge-case alignment (#53218), **optimized SSE hot path (#51321)**, pure-Rust protox replacing protoc (#52892).
- **SSE keep-alive comments for idle streams (#51034)** — new in 0.29.0 (API & Frontend section). Directly relevant to idle streaming.
- `--max-num-queued-reqs` / `--max-num-queued-tokens` admission control (#49445)
- DP supervisor inheriting uvicorn config (#52473)
- `/v1/messages/render` (#45803), `/cohere/v2/chat/render` (#53219)
- `python -m vllm.entrypoints.openai.api_server` deprecated in favor of `vllm serve` (#52131)

### (b) Async scheduling

- **Not explicitly mentioned in the 0.29.0 release notes by name.** Open question at the time: when it became default, and whether its semantics changed in 0.29.

### (c) Multiprocess "main server" / `VLLM_MAIN_SERVER_PORT`

- Not named in the release notes. The Rust frontend + gRPC + DP supervisor machinery is the likely home. Open question at the time: what `VLLM_MAIN_SERVER_PORT` and `VLLM_HTTP_TIMEOUT_KEEP_ALIVE` are (server-side request timeout?) — to be resolved by code search.

### (d) MTP spec decode

- Padded FULL cudagraph dispatch for uniform decode under spec decode (#53407)
- DP-sync skipping before EAGLE/MTP draft prefill (#53694)
- Fused GDN MTP for all Qwen head ratios (#52539)
- Widest uniform decode batch captured by default (#50488) + memory-safe graph sizes (#54418)
- Decoupled draft/target gumbel noise streams (#54282)
- Spec decode no longer padded up to `max_model_len` (#53962)
- Per-request spec decode metrics (#48915)

### (e) Reasoning / tool-call parsers

- **Reasoning-end detection scoped to the current turn (#54089)**
- **Spurious FSM errors after speculative reasoning end (#53046)** — highly relevant (reasoning parser + spec decode)
- Unused request-local reasoners skipped (#52573)
- Shared parser engine adapters (#52830)
- XGrammar termination in batches (#52805)
- Terminal grammars stop under `min_tokens` (#54218)

## Big change — Model Runner V2 default

- **MRV2 is now the default for all models (#53183).**
- Release notes: "Some features are not yet supported in MRV2 … These include sequence parallelism, dual-batch overlap, elastic expert parallelism, custom logits processors and **certain speculative decoding methods**. For now, vLLM will still fall back to use MRV1 if any of these features are configured."
- MRV1 deprecation targeted for v0.32.
- Note: deployment configs saying "V1 engine only" refer to the **V1 engine** (vs the old V0), **not** Model Runner V1. Determining whether Qwen3.8-27B + MTP runs on MRV2 or falls back to MRV1 was left open.

## Other notable 0.29.0 items

- New defaults: FlashInfer all-reduce default (#52998); NONE_HASH deterministic (#51875)
- `prefix_cache_retention_interval` default dense→0 for SWA/SSM (#52216); dense retention restored for EAGLE/MTP hybrid (#55760, #55861)
- Mamba prefix caching TTFT improvement (#52789)

## Open questions (as of the last research pass)

- What is `VLLM_MAIN_SERVER_PORT`? (main-server multiprocess)
- What is `VLLM_HTTP_TIMEOUT_KEEP_ALIVE`? (server-side request timeout?)
- Async scheduling: default in 0.29? semantic change in 0.29?
- MRV2 + MTP spec-decode concurrency hang bugs
- Rust frontend streaming / keep-alive / client-disconnect propagation bugs
- Candidate issue searches around 0.29.0: "concurrent streaming hang", "stuck streaming", "1 of N completes", "API server stuck", "async scheduling bug"
