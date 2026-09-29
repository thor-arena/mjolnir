---
name: "Feature request"
about: A new CLI command, config, patch, kernel feature, or doc
title: ""
labels: ["enhancement"]
assignees: []
---

## Problem you want to solve

The concrete pain. Who hits it and when (e.g. "benchmarks restart the live server per leg, so I can't run an A/B while the agents are working").

## Proposed solution

The shape you want — the command, the flag, the config key, the patch. Prefer the smallest surface that covers the case.

## Alternatives considered

What you tried or ruled out (including upstream routes — if the feature exists upstream but is gated, say which gate blocks it).

## Why Mjolnir (and not upstream)

This repo is a probe-gated overlay: features that upstream owns get tracked, not re-implemented. Briefly state why this belongs here:

- [ ] Upstream is blocked/unmerged — link the upstream issue/PR and where you left it
- [ ] sm_110-specific grant (patch would be passive on other arches, opt-in via config)
- [ ] Tooling around the stack (CLI, gates, bench, docs) that upstream doesn't own

## Acceptance criteria

- [ ] (e.g. `mjolnir bench ab` does not restart the server when both legs share an image)
- [ ] (e.g. new patch follows the probe-gated pattern: granted only on `capability == (11, 0)`, registered in `apply_patches.py` / `verify_patches.py` / `PATCHES.md`)
- [ ] (e.g. any measured claim ships with raw JSON + clean-window-gated run)

## Prior art / related work

Links to upstream RFCs/PRs, docs in this repo (`docs/…`), or external prior art.
