"""Kernel verification / benchmark task registry + docker launcher.

Every kernel test / bench in the repo is registered here so one command
(``mjolnir bench kernel <task>``) runs it instead of re-implementing the
docker launch (image, mounts, entrypoint) each time.

Task kinds:
  * test  — fresh container against the image, no server needed; GPU test
            tasks self-skip (exit 0) on a box with no CUDA device.
  * bench — needs the vLLM server running (the clean-window gate). The vfa
            tree is mounted, the workdir is mounted at /p, --network host is
            added (the gate polls 127.0.0.1:<port> from inside).
  * host  — runs on the host directly, no container (the gate itself).
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from mjolnir.config import Settings, VFA_DIST_PATH, find_repo_root
from mjolnir.dockerctl import _docker, _run_quiet
from mjolnir.gate import server_load


@dataclass(frozen=True)
class Task:
    name: str
    kind: str                       # "test" | "bench" | "host"
    script: str                      # repo-relative path
    desc: str
    gpu: bool = True
    tmpdir: str = ""
    host_script: bool = False        # host tasks run a host script


def task_script(task: Task, s: Settings) -> Path:
    """The script to run (repo-relative, or in-gate for host tasks)."""
    root = find_repo_root()
    if task.host_script:
        # Host tasks import the gate module; run the package directly.
        return Path()
    return root / task.script


TASKS: list[Task] = [
    # ---- TEST: fresh container, no server (GPU ones self-skip) -------------
    Task("gates", "test",
         "docker/vllm-thor/kernel_test_sm110_gates.py",
         "sm_110 gate-probe canary (T6 GDN, T9 FA4, T10 draft-CG, T11 hd256)"),
    Task("fi-update", "test",
         "docker/vllm-thor/kernel_test_fi_update.py",
         "fused draft-decode FI kernel (advance plan vs numpy reference)"),
    Task("dspark-nc", "test",
         "docker/vllm-thor/kernel_test_dspark_draft_nonncausal.py",
         "DSpark non-causal draft cudagraph (gate/route/capture-replay/determinism)"),
    Task("functional", "test",
         "docker/vllm-thor/functional_check_55390_55519.py",
         "55390/55519 draft-group annotation + warning check (no GPU)", gpu=False),
    # ---- HOST: the clean-window gate itself ---------------------------------
    Task("gate", "host", "",
         "clean-window gate: --once = check now; default = block until a "
         "window opens (--confirm N, --timeout S)",
         gpu=False, host_script=True),
    # ---- BENCH: gated, vfa tree mounted, workdir at /p, --network host ------
    Task("gemv-verify", "bench",
         "docker/vllm-thor/fa4-gemv-kernel/verify-gemv-decode.py",
         "GEMV Phase-A correctness vs fp32 ref (dense/varlen, mixed dtypes, "
         "15 cases)", tmpdir="/tmp/kt-gemv-verify"),
    Task("gemv-verify-b1", "bench",
         "docker/vllm-thor/fa4-gemv-kernel/verify-gemv-b1.py",
         "GEMV Phase-B1 correctness (SplitKV partials, paged, combine)",
         tmpdir="/tmp/kt-gemv-verify-b1"),
    Task("gemv-stages", "bench",
         "docker/vllm-thor/fa4-gemv-kernel/verify-gemv-stages.py",
         "GEMV stages 2 vs 16 vs fp32 ref (confirms the ring fix is bit-neutral)",
         tmpdir="/tmp/kt-gemv-stages"),
    Task("gemv-bench", "bench",
         "docker/vllm-thor/fa4-gemv-kernel/gemv-decode-bench.py",
         "GEMV 4-mode bench (gemv_dense/gemv_paged/fa4_1cta/flashinfer), "
         "gated (--mode, --out)", tmpdir="/tmp/kt-gemv-bench"),
    Task("gemv-ringfix", "bench",
         "docker/vllm-thor/fa4-gemv-kernel/gemv-ring-fix-bench.py",
         "GEMV ns sweep + stages contrast + FI in one gated window "
         "(stricter CONFIRM=6; --out)", tmpdir="/tmp/kt-gemv-ringfix"),
    Task("gemv-ncu", "bench",
         "docker/vllm-thor/fa4-gemv-kernel/ncu-gemv-ring.py",
         "GEMV ncu driver (NCU_L/NCU_NS/NCU_STAGES/NCU_ITERS env) — for ncu",
         tmpdir="/tmp/kt-gemv-ncu"),
]

BY_NAME = {t.name: t for t in TASKS}

GEMV_DIR_RELPATH = Path("docker/vllm-thor/fa4-gemv-kernel")


def preflight_bench(task: Task, s: Settings) -> int:
    """For BENCH tasks, probe the vLLM metrics endpoint (informational only —
    an offline server is fine: the bench then just runs ungated)."""
    load = server_load(s.metrics_url)
    if load is None:
        print(f"[mjolnir] vLLM metrics {s.metrics_url} unreachable — server "
              f"offline; {task.name} will run UNGATED (nothing co-located to "
              f"gate against).", file=sys.stderr)
        return 0
    print(f"[mjolnir] vLLM up (running={load[0]:g} waiting={load[1]:g}); "
          f"the bench will wait for a clean window.", file=sys.stderr)
    return 0


def build_docker(task: Task, s: Settings, vfa_tree: Path, workdir: Path,
                 extra: list[str]) -> list[str]:
    """Assemble the ``docker run`` argv for a task."""
    cmd = [_docker(), "run", "--rm"]
    if task.gpu:
        cmd += ["--gpus", "all"]
    if task.kind == "bench":
        # Host networking: the clean-window gate polls 127.0.0.1:<port>
        # from INSIDE the container. The gate URL (which port) is passed via
        # env so ``mjolnir bench kernel --port <p>`` targets the live server.
        cmd += ["--network", "host", "--entrypoint", "python3"]
        cmd += ["-v", f"{vfa_tree}:{VFA_DIST_PATH}"]
        cmd += ["-v", f"{workdir}:/p"]
        # The committed raw JSONs (the charts' input) are mounted at /raw so
        # a bare run (no --out) updates benchmarks/raw in place, and
        # ``--out /raw/<name>`` can target any raw filename.
        raw_dir = find_repo_root() / "benchmarks" / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        cmd += ["-v", f"{raw_dir}:/raw"]
        cmd += ["-e", f"MJOLNIR_METRICS_URL={s.metrics_url}"]
        if task.tmpdir:
            cmd += ["-e", f"TMPDIR={task.tmpdir}"]
        cmd += [s.image, f"/p/{task_script(task, s).name}"] + list(extra)
        return cmd
    # test: mount the script itself at /tmp/<basename>
    script = task_script(task, s)
    cmd += ["--entrypoint", "python3",
            "-v", f"{script}:/tmp/{script.name}:ro"]
    cmd += [s.image, f"/tmp/{script.name}"] + list(extra)
    return cmd


def run_task(s: Settings, name: str, script_args: list[str],
             image: str | None = None, vfa_tree: Path | None = None,
             workdir: Path | None = None, dry_run: bool = False,
             skip_preflight: bool = False) -> int:
    """Run a registered task. Returns the exit code."""
    task = BY_NAME.get(name)
    if task is None:
        known = ", ".join(BY_NAME)
        print(f"[mjolnir] unknown task '{name}'. Known: {known}",
              file=sys.stderr)
        return 4

    root = find_repo_root()
    img = image or s.image

    # ---- host tasks (the gate) -------------------------------------------
    if task.kind == "host":
        return _run_gate(s, script_args, dry_run)

    # ---- container tasks ---------------------------------------------------
    script = task_script(task, s)
    if not script.exists():
        print(f"[mjolnir] ERROR: script not found: {script}", file=sys.stderr)
        return 4

    if task.kind == "bench":
        workdir = workdir or root / GEMV_DIR_RELPATH
        if vfa_tree is None:
            if dry_run:
                # Preview only — no container, no rebuild: reuse the cached
                # tree (the mount path is identical either way).
                default_vfa = root / "vfa-tree" / "vllm_flash_attn"
                vfa_tree = default_vfa if default_vfa.exists() else None
                if vfa_tree is None:
                    print("[mjolnir] no GEMV'd vfa tree — prepare it first:\n"
                          "  mjolnir vfa prepare\n"
                          "(or pass --vfa-tree <path>)", file=sys.stderr)
                    return 4
            else:
                # Real run: ALWAYS (re)build the default GEMV'd tree, so a run
                # measures the current kernel source of truth (fa4-gemv-kernel/)
                # + current image, never a stale cached copy. Cheap — an
                # in-image cp + kernel drop + dispatch hunk, no image build.
                # Pass --vfa-tree <path> to opt out (use that tree as-is).
                from mjolnir import vfa
                default_out = root / "vfa-tree"
                print("[mjolnir] (re)preparing GEMV'd vfa tree (always "
                      "fresh) …", file=sys.stderr)
                try:
                    vfa_tree = vfa.prepare_vfa_tree(s, default_out, img)
                except RuntimeError as e:
                    print(f"[mjolnir] {e}", file=sys.stderr)
                    return 4
        else:
            vfa_tree = Path(vfa_tree)
        if not vfa_tree.exists():
            print(f"[mjolnir] vfa tree not found: {vfa_tree}", file=sys.stderr)
            return 4
    else:
        workdir = workdir or script.parent

    s.image = img
    if task.kind == "bench" and not skip_preflight:
        rc = preflight_bench(task, s)
        if rc != 0:
            return rc

    cmd = build_docker(task, s, vfa_tree, workdir, script_args)
    if dry_run:
        print(" ".join(shlex.quote(c) if any(x in c for x in " \t\"'$")
                       else c for c in cmd))
        return 0
    print(f"[mjolnir] running: {' '.join(cmd)}", file=sys.stderr)
    return subprocess.call(cmd)


ALL_KERNEL_BENCHES: list[tuple[str, list[str]]] = [
    ("gemv-ringfix", []),
    ("gemv-bench", ["--mode", "gemv_dense"]),
    ("gemv-bench", ["--mode", "gemv_paged"]),
    ("gemv-bench", ["--mode", "fa4_1cta"]),
    ("gemv-bench", ["--mode", "flashinfer"]),
]


def run_all_bench(s: Settings, image: str | None = None,
                  vfa_tree: Path | None = None, dry_run: bool = False,
                  skip_preflight: bool = False) -> int:
    """The full GEMV bench set (``bench kernel all``): gemv-ringfix + the
    four gemv-bench modes, one gated window per leg, stopping at the first
    failure. Each bare leg writes its raw JSON into benchmarks/raw (the
    /raw mount). Returns the exit code."""
    root = find_repo_root()
    default_tree = root / "vfa-tree" / "vllm_flash_attn"
    rc = 0
    for i, (name, extra) in enumerate(ALL_KERNEL_BENCHES):
        label = f"{name} {' '.join(extra)}".strip()
        print(f"\n── bench kernel {label} " + "─" * 40, file=sys.stderr)
        # The first real leg rebuilds the default GEMV'd tree; the rest reuse
        # it (opting out of the per-leg rebuild — same tree, same window set).
        reuse = vfa_tree if vfa_tree is not None else (
            default_tree if i > 0 and not dry_run else None)
        r = run_task(s, name, extra, image=image, vfa_tree=reuse,
                     dry_run=dry_run, skip_preflight=skip_preflight)
        if r != 0:
            print(f"[mjolnir] {label} FAILED (exit {r}) — stopping the set",
                  file=sys.stderr)
            rc = r
            break
    if rc == 0 and not dry_run:
        print("\n  all GEMV bench legs done "
              "(raws in benchmarks/raw/)", file=sys.stderr)
    return rc


def _run_gate(s: Settings, script_args: list[str], dry_run: bool) -> int:
    """Run the clean-window gate on the host (no container)."""
    from mjolnir.gate import wait_for_idle, server_load
    import time as _time

    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-url", default=s.metrics_url)
    ap.add_argument("--confirm", type=int, default=6)
    ap.add_argument("--timeout", type=float, default=3600)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args(script_args)

    if args.once:
        load = server_load(args.metrics_url)
        if load is None:
            print("UNREACHABLE (vLLM metrics endpoint not up)")
            return 2
        clean = load == (0.0, 0.0)
        print(f"now: running={load[0]:g} waiting={load[1]:g} -> "
              f"{'CLEAN' if clean else 'BUSY'}")
        return 0 if clean else 1
    if dry_run:
        print(f"(host) wait_for_idle({args.metrics_url}, confirm={args.confirm})")
        return 0
    t0 = _time.time()
    ts = wait_for_idle(args.metrics_url, confirm=args.confirm,
                       timeout_s=args.timeout)
    if ts is None:
        print(f"TIMEOUT: no {args.confirm}x consecutive 0/0 after "
              f"{args.timeout:.0f}s", file=sys.stderr)
        return 3
    print(f"CLEAN WINDOW OPEN ({args.confirm} consecutive 0/0 samples) at "
          f"t={_time.time() - ts:.2f}s ago")
    return 0
