# Security

## Report a Vulnerability

Please do **not** open public GitHub issues for security bugs. Use GitHub's
private vulnerability disclosure form:
[thor-arena/mjolnir — Security Advisories](https://github.com/thor-arena/mjolnir/security/advisories/new).

## Scope

Mjolnir is a thin overlay on upstream vLLM / FlashInfer: a 14-patch image,
the GEMV decode kernel, and the `mjolnir` CLI. Most serving code is upstream
— report vulnerabilities in vLLM or FlashInfer to their respective projects.

Two properties of the stack worth knowing:

- **The served API is unauthenticated.** The image runs
  `vllm serve --host 0.0.0.0` inside the container and maps the host port
  (default `6001`). The host-side port binding is the only network
  boundary — if you expose the server beyond your trusted LAN, put an
  authenticating reverse proxy in front of it.
- **The patch stack adds no new attack surface of its own**: no network
  listeners, no telemetry, no outbound connections beyond upstream
  behavior. It changes kernel selection, CUDA-graph capture, and policy
  gates inside vLLM/FlashInfer.
