#!/usr/bin/env python3
"""Verify the vLLM PR fixes are present in the installed vllm package (and
the FlashInfer-side sm_110 hunk in the installed flashinfer package).

Run standalone (resolves the installed vllm and flashinfer themselves) or
with explicit package roots (``python3 verify_patches.py
/path/to/site-packages/vllm /path/to/site-packages/flashinfer``), or
imported as ``verify(root)`` / ``verify_flashinfer(root)`` from
``apply_patches.py``. Exits non-zero if any fix is missing, so the Dockerfile
build fails fast when a base nightly no longer matches the assumptions the
hand-adapted patches were written against.
An explicit root argument is always honored: the verdict is about that tree,
never silently about the installed package.

Each fix is detected by a sentinel that is present only after the fix is in
place. The sentinels are intentionally chosen to be stable across vllm
releases (a function/attribute name that the fix introduces) so the check
keeps working as the surrounding code churns.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

# fix id -> list of (relative path from the vllm package root, sentinel text).
# A leading "!" on a sentinel asserts the text is ABSENT (rename guards),
# e.g. 55390 removes _is_deepseek_v4_eagle entirely.
FIXES: dict[str, list[tuple[str, str]]] = {
    "50885 flashinfer native FULL decode cudagraphs for spec-decode": [
        (
            "v1/attention/backends/flashinfer.py",
            "def flashinfer_supports_uniform_multi_token_decode",
        ),
    ],
    "49652 draft-decode capture under dynamic SD (base manager guard)": [
        # Rebased onto the Dynamic-SD candidate expansion that upstream added
        # to the BASE CudaGraphManager: draft-decode managers run at
        # decode_query_len == 1, and the expansion must be skipped for them
        # (it would produce non-positive decode_query_lens). The guard line is
        # absent in the pristine tree and present only after this patch.
        (
            "v1/worker/gpu/cudagraph_utils.py",
            "and self.decode_query_len > 1",
        ),
    ],
    "54165 hybrid mamba align cache hits under spec decode (KV connector)": [
        (
            "config/speculative.py",
            "def use_eagle_preserves_target_kv_cache",
        ),
    ],
    "55390 MTP draft KV group positional annotation (hybrid path)": [
        (
            "v1/core/kv_cache_utils.py",
            "def _uses_trailing_mtp_layers",
        ),
        (
            "v1/core/kv_cache_utils.py",
            "def _groups_partition_layers_exactly",
        ),
        (
            # renamed away by the fix; its presence means the fix did not apply
            "v1/core/kv_cache_utils.py",
            "!def _is_deepseek_v4_eagle",
        ),
    ],
    "55519 no draft-group warning when eagle block drop is off": [
        (
            "v1/core/kv_cache_utils.py",
            'if spec_config is None or not spec_config.use_eagle_block_drop():\n        return\n    if any(group.is_eagle_group for group in kv_cache_groups):',
        ),
        (
            "v1/core/sched/scheduler.py",
            "so the trailing block stays eligible for cross-request",
        ),
    ],
    "thor fused draft-decode: in-place FlashInfer native decode plan advance": [
        (
            "v1/attention/backends/flashinfer.py",
            "def _advance_fi_decode_plan_kernel",
        ),
        (
            "v1/attention/backends/flashinfer.py",
            "def update_draft_decode_metadata",
        ),
    ],
    "thor dspark non-causal draft: FULL cudagraph via the FI prefill wrapper": [
        # Sentinels assert the non-causal draft actually *routes* to the
        # cudagraph-mode prefill wrapper, not just that the symbols exist (the
        # 52244 lesson — a partial apply must fail the build, so a defined-but-
        # unused wrapper must not pass).
        #
        # The wrapper factory (definition)
        (
            "v1/attention/backends/flashinfer.py",
            "def _get_noncausal_prefill_wrapper_cudagraph",
        ),
        # The per-batch-size wrapper dict + qo_indptr buffer allocation in
        # __init__ (where the wrapper state lives)
        (
            "v1/attention/backends/flashinfer.py",
            "self._noncausal_prefill_wrappers_cudagraph: dict[",
        ),
        # The new non-causal UNIFORM_BATCH grant inside get_cudagraph_support.
        # Gated on a multi-token drafter being configured; this line is the
        # patch-introduced gate (the *first* gate condition is the bare
        # vllm_config.attention_config.use_non_causal, which pre-dates the
        # patch).
        (
            "v1/attention/backends/flashinfer.py",
            "and vllm_config.speculative_config is not None",
        ),
        # The routing CALL SITE in build() — the load-bearing "used, not just
        # defined" sentinel. The prefill pathway must actually dispatch the
        # non-causal batch to the cudagraph wrapper.
        (
            "v1/attention/backends/flashinfer.py",
            "self._get_noncausal_prefill_wrapper_cudagraph(num_prefills)",
        ),
        # The routing condition variable that selects that call site
        (
            "v1/attention/backends/flashinfer.py",
            "use_nc_cudagraph =",
        ),
    ],
    "thor sm110 GDN prefill: FI prefill grant gated by a one-shot runtime probe": [
        # The probe-gated grant clause in _resolve_gdn_prefill_backend — the
        # only file the patch touches.
        (
            "model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py",
            "# THOR_SM110_GDN_PREFILL_PROBE",
        ),
    ],
    "thor sm110 FA4 FP8-KV: probe-gated kv-cache-dtype grant": [
        # The probe definition + its probe-gated OR-term in the FP8-KV
        # eligibility gate (the only file the patch's fa_utils.py hunks
        # touch). The marker occurs exactly once in that file: a partial
        # apply that lands the probe without the gate (or vice versa)
        # must fail the build, since either half alone is behaviorally
        # inert or unsafe.
        (
            "v1/attention/backends/fa_utils.py",
            "# THOR_SM110_FA4_PROBE",
        ),
    ],
    "thor sm110 FA4 hd256 FP8-KV: probe-gated hd256 kernel policy grant": [
        # Kernel-file sentinel (N-1): the descale-load helper is absent from
        # the pristine kernel and present post-patch, so a partial apply that
        # lands the fa_utils policy hunks but misses the kernel fold hunks
        # must fail the build (the probe sentinels alone cannot catch that).
        # Path is relative to the vllm package root (like every other
        # sentinel here), so no leading ``vllm/``.
        (
            "vllm_flash_attn/cute/sm100_hd256_2cta_fmha_forward.py",
            "_load_effective_descales",
        ),
        # The hd256 probe definition — present only after this patch.
        # The marker occurs exactly once in fa_utils.py (the comment
        # block above the probe definition).
        (
            "v1/attention/backends/fa_utils.py",
            "# THOR_SM110_FA4_HD256_FP8_PROBE",
        ),
        (
            "v1/attention/backends/fa_utils.py",
            "def _thor_fa4_hd256_fp8kv_probe",
        ),
        # The quantized-KV reject relaxation in
        # _fa4_hd256_fallback_reason — the "used, not just defined"
        # sentinel: the grant must be wired into the reject site, not
        # merely defined in the probe (a partial apply that lands the
        # probe without the relaxation would leave an explicit-FA4
        # block-128 FP8-KV hd256 config downgraded to FA2, so it must
        # fail the build).
        (
            "v1/attention/backends/fa_utils.py",
            "# THOR_SM110_FA4_HD256_FP8:",
        ),
        # The gate term's consumer site — a head_dim=256 request must
        # be backed by the hd256 probe (a partial apply that leaves the
        # family(110) term backed by the hd128 probe only would admit
        # an hd256 model without validating the hd256 kernel, so it
        # must fail the build).
        (
            "v1/attention/backends/fa_utils.py",
            "_thor_fa4_hd256_fp8kv_probe()\n            if uses_fa4_hd256_kernel",
        ),
    ],
    "thor sm110 FA4 hd256 GEMV decode: default-on M=1 hd256 GEMV kernel dispatch": [
        # New kernel file (N-1): the GEMV kernel class is absent from the
        # pristine tree and present only post-patch. A partial apply that
        # lands the interface.py dispatch hunks but misses the kernel file
        # must fail the build (the interface sentinels alone cannot catch a
        # missing kernel). Path is relative to the vllm package root.
        (
            "vllm_flash_attn/cute/sm100_hd256_decode_gemv.py",
            "class BlackwellHd256DecodeGEMV",
        ),
        # The import — present only after the patch's import hunk.
        (
            "vllm_flash_attn/cute/interface.py",
            "from vllm.vllm_flash_attn.cute.sm100_hd256_decode_gemv import "
            "BlackwellHd256DecodeGEMV",
        ),
        # The FlashInfer Alg.1 split-KV heuristic — defined only post-patch.
        (
            "vllm_flash_attn/cute/interface.py",
            "def _gemv_auto_num_splits(",
        ),
        # The dispatch gate — the "used, not just defined" sentinel: the
        # GEMV bypass must actually be wired into the forward dispatch (a
        # partial apply that defines the kernel + helper but never routes
        # M=1 hd256 calls to it would be behaviorally inert, so it must fail
        # the build).
        (
            "vllm_flash_attn/cute/interface.py",
            "use_gemv_hd256 = (",
        ),
        # The DEFAULT-ON gate: the knob defaults to "1" (opt out with
        # VLLM_FA4_HD256_GEMV=0). If a regression flips it back to "0" the
        # kernel would be inert-by-default, so the default is pinned here.
        (
            "vllm_flash_attn/cute/interface.py",
            'os.environ.get("VLLM_FA4_HD256_GEMV", "1")',
        ),
    ],
    "thor sm110 draft-CG gate: shape-gated draft cudagraph replay (c1 fix)": [
        # The pure decision helper (THOR_DRAFT_CG_GATE sentinel lives in
        # its docstring) — present only after this patch.
        (
            "v1/worker/gpu/spec_decode/dflash/speculator.py",
            "def _thor_draft_cg_use_cuda_graph",
        ),
        # The dispatch-site call — the "used, not just defined" sentinel
        # (a partial apply that lands the helper but never calls it would
        # leave the c1 regression in place, so it must fail the build).
        (
            "v1/worker/gpu/spec_decode/dflash/speculator.py",
            "need_eager = is_profile or _thor_draft_cg_gate_wants_eager(num_reqs)",
        ),
    ],
}


# fix id -> list of (relative path from the flashinfer package root, sentinel
# text). Same sentinel conventions as FIXES; a leading "!" asserts absence.
# The flashinfer-side hunk is verified by verify_flashinfer() (separate root:
# the installed flashinfer site-packages, not the vllm package).
FLASHINFER_FIXES: dict[str, list[tuple[str, str]]] = {
    "thor sm110 GDN prefill: FI non-CP dispatch allowlist widened to major 11": [
        # The single widened condition in chunk_gated_delta_rule's non-CP
        # dispatch — the comment sentinel rides on the changed line, so it
        # is present only when the hunk applied (a partial apply is
        # impossible for a one-line hunk).
        (
            "gdn_prefill.py",
            "if _arch_major in (10, 11):  # THOR_SM110_GDN_ALLOWLIST",
        ),
    ],
}


def package_root() -> Path:
    spec = importlib.util.find_spec("vllm")
    if spec is None or spec.origin is None:
        raise SystemExit("could not locate the installed vllm package")
    return Path(spec.origin).parent


def flashinfer_package_root() -> Path:
    spec = importlib.util.find_spec("flashinfer")
    if spec is None or spec.origin is None:
        raise SystemExit("could not locate the installed flashinfer package")
    return Path(spec.origin).parent


def _check_root(
    root: Path, fixes: dict[str, list[tuple[str, str]]]
) -> list[str]:
    """Return the list of missing fixes under ``root`` (empty == all present)."""
    missing: list[str] = []
    for fix, sentinels in fixes.items():
        present = True
        for rel, sentinel in sentinels:
            path = root / rel
            text = path.read_text() if path.exists() else ""
            if sentinel.startswith("!"):
                # absence sentinel: the renamed-away name must be gone
                ok = sentinel[1:] not in text
            else:
                ok = sentinel in text
            if not ok:
                present = False
                break
        if not present:
            missing.append(fix)
    return missing


def check_root(root: Path) -> list[str]:
    return _check_root(root, FIXES)


def check_flashinfer_root(root: Path) -> list[str]:
    return _check_root(root, FLASHINFER_FIXES)


def verify(root: str | Path | None = None) -> list[str]:
    # Normalize at the contract boundary: a str root used to crash inside
    # check_root (``root / rel`` -> TypeError) instead of verifying.
    root = Path(root) if root is not None else package_root()
    return check_root(root)


def verify_flashinfer(root: str | Path | None = None) -> list[str]:
    root = Path(root) if root is not None else flashinfer_package_root()
    return check_flashinfer_root(root)


if __name__ == "__main__":
    # An explicit root argument checks that tree. (Previously the argument
    # was silently ignored and the *installed* vllm was always checked, so
    # ``verify_patches.py <pristine-tree>`` could exit 0 with "all fixes
    # present" while reporting on a different, already-patched tree.)
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else package_root()
    fi_root = (
        Path(sys.argv[2]) if len(sys.argv) > 2 else flashinfer_package_root()
    )
    problems = verify(root)
    fi_problems = verify_flashinfer(fi_root)
    if problems or fi_problems:
        print("PATCH VERIFICATION FAILED — missing fixes:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        for p in fi_problems:
            print(f"  - [flashinfer] {p}", file=sys.stderr)
        sys.exit(1)
    print(f"all {len(FIXES)} fixes present in {root}")
    print(f"all {len(FLASHINFER_FIXES)} flashinfer fixes present in {fi_root}")
