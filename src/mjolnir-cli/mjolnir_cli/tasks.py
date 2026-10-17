"""Image canary tasks + the clean-window gate.

The canaries are the sm_110 gate probes and the Thor-authored kernel tests
that ship with the image build (``docker/vllm-thor/``); ``mjolnir image
gates`` runs them in a fresh container against an image (no server needed;
GPU tests self-skip on a box without CUDA). Kernel correctness and
benchmarks live in ``mjolnir verify``.
"""

from __future__ import annotations

import argparse
import os
import shlex
import sys
from dataclasses import dataclass

from mjolnir_cli.config import Settings, find_repo_root
from mjolnir_cli.dockerctl import _docker


@dataclass(frozen=True)
class Task:
    name: str
    script: str  # repo-relative path
    desc: str
    gpu: bool = True


TASKS: list[Task] = [
    Task(
        "gates",
        "docker/vllm-thor/kernel_test_sm110_gates.py",
        "sm_110 gate-probe canary (T6 GDN, T9 FA4, T10 draft-CG, T11 hd256)",
    ),
    Task(
        "fi-update",
        "docker/vllm-thor/kernel_test_fi_update.py",
        "fused draft-decode FI kernel (advance plan vs numpy reference)",
    ),
    Task(
        "dspark-nc",
        "docker/vllm-thor/kernel_test_dspark_draft_nonncausal.py",
        "DSpark non-causal draft cudagraph (gate/route/capture-replay/determinism)",
    ),
    Task(
        "functional",
        "docker/vllm-thor/functional_check_55519.py",
        "55519 draft-group warning gate check (no GPU)",
        gpu=False,
    ),
]

BY_NAME = {t.name: t for t in TASKS}


def task_container(task: Task) -> str:
    return f"mjolnir-task-{task.name}-{os.getpid()}"


def build_docker(task: Task, s: Settings) -> list[str]:
    script = find_repo_root() / task.script
    cmd = [_docker(), "run", "--rm", "--name", task_container(task)]
    if task.gpu:
        cmd += ["--gpus", "all"]
    cmd += ["--entrypoint", "python3", "-v", f"{script}:/tmp/{script.name}:ro"]
    return cmd + [s.image, f"/tmp/{script.name}"]


def run_task(s: Settings, name: str, script_args: list[str], dry_run: bool = False) -> int:
    """Run one canary task (or ``gates`` = the gate probe) in a fresh container."""
    task = BY_NAME.get(name)
    if task is None:
        print(f"[mjolnir] unknown task '{name}'. Known: {', '.join(BY_NAME)}", file=sys.stderr)
        return 4
    try:
        script = find_repo_root() / task.script
    except FileNotFoundError as e:
        print(f"[mjolnir] {e}", file=sys.stderr)
        return 4
    if not script.exists():
        print(f"[mjolnir] ERROR: script not found: {script}", file=sys.stderr)
        return 4
    cmd = build_docker(task, s) + list(script_args)
    if dry_run:
        print(" ".join(shlex.quote(c) for c in cmd))
        return 0
    print(f"[mjolnir] running: {' '.join(cmd)}", file=sys.stderr)
    from mjolnir_cli.dockerctl import run_once

    return run_once(cmd, task_container(task))


def run_gate(s: Settings, script_args: list[str], dry_run: bool) -> int:
    """Run the clean-window gate on the host (no container)."""
    from mjolnir_cli.gate import wait_for_idle, server_load
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
        print(f"now: running={load[0]:g} waiting={load[1]:g} -> {'CLEAN' if clean else 'BUSY'}")
        return 0 if clean else 1
    if dry_run:
        print(f"(host) wait_for_idle({args.metrics_url}, confirm={args.confirm})")
        return 0
    ts = wait_for_idle(args.metrics_url, confirm=args.confirm, timeout_s=args.timeout)
    if ts is None:
        print(f"TIMEOUT: no {args.confirm}x consecutive 0/0 after {args.timeout:.0f}s", file=sys.stderr)
        return 3
    print(f"CLEAN WINDOW OPEN ({args.confirm} consecutive 0/0 samples) at t={_time.time() - ts:.2f}s ago")
    return 0
