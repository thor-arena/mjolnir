"""Serve orchestration: drive the vLLM docker container from Python.

The vLLM server runs in a container because the whole point of this repo is
a *custom* image (the sm_110 patch stack + the FA4 GEMV decode kernel) —
``mjolnir serve`` replaces the bash/docker-compose glue, not the container.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from mjolnir.config import (CONTAINER_NAME, Settings, VFA_DIST_PATH)

# Container-side env (the Thor-specific knobs from the old vllm.yaml service).
_CONTAINER_ENV: list[str] = [
    "DO_NOT_TRACK=1", "VLLM_NO_USAGE_STATS=1",
    "MAX_JOBS=4", "NVCC_THREADS=2", "FLASHINFER_NVCC_THREADS=2",
    "VLLM_HTTP_TIMEOUT_KEEP_ALIVE=600", "FLASHINFER_DISABLE_VERSION_CHECK=1",
    "VLLM_CACHE_ROOT=/data/cache", "VLLM_ASSETS_CACHE=/data/assets",
    "VLLM_TUNED_CONFIG_FOLDER=/data/kernels-cache",
    "FLASHINFER_JIT_DIR=/data/flashinfer-cache",
    "TRITON_CACHE_DIR=/data/triton-cache",
    "TORCHINDUCTOR_CACHE_DIR=/data/cache/inductor",
    "TORCH_COMPILE_CACHE_DIR=/data/cache/torch_compile_cache",
    "XDG_CACHE_HOME=/data/cache/xdg",
    "TIKTOKEN_ENCODINGS_BASE=/data/tiktoken", "TIKTOKEN_RS_CACHE_DIR=/data/tiktoken",
    "VLLM_LOGGING_LEVEL=INFO", "NVIDIA_VISIBLE_DEVICES=all",
    "NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics",
    "CUDA_VISIBLE_DEVICES=0", "VLLM_TRITON_USE_TD=1", "PYTHONUNBUFFERED=1",
    "SAFETENSORS_FAST_GPU=1", "NVIDIA_DISABLE_REQUIRE=1",
    "NVIDIA_FORWARD_COMPAT=1", "NCCL_P2P_DISABLE=1", "ENABLE_TRIATTENTION=0",
    "CUDA_MANAGED_FORCE_DEVICE_ALLOC=1",
    "TORCH_CUDA_ARCH_LIST=11.0a", "FLASHINFER_CUDA_ARCH_LIST=11.0a",
    "CUDA_TOOLKIT_PATH=/usr/local/cuda-13", "CUTE_DSL_ARCH=sm_110a",
    "TORCH_MATMUL_PRECISION=high",
    "VLLM_DISABLED_KERNELS=CutlassFP8ScaledMMLinearKernel",
    "VLLM_USAGE_SOURCE=production",
    "TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas",
    "VLLM_USE_BREAKABLE_CUDAGRAPH=0",
    "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
    # Thor-only host overrides from the old compose file:
    "VLLM_ALLOW_LONG_MAX_MODEL_LEN=1", "VLLM_MARLIN_USE_ATOMIC_ADD=1",
]

_HOST_MOUNTS = [
    "/etc/nv_tegra_release",   # L4T id (Jetson)
    "/tmp/nv_jetson_model",
    "/etc/localtime",
    "/etc/machine-id",
]


class DockerError(RuntimeError):
    pass


def _docker() -> str:
    if shutil.which("docker") is None:
        raise DockerError("docker not found on PATH")
    return "docker"


def _run(args: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(args, check=True, **kw)


def _run_quiet(args: list[str], ok_codes: tuple[int, ...] = (0, 1)
              ) -> tuple[int, str]:
    p = subprocess.run(args, capture_output=True, text=True)
    if p.returncode not in ok_codes:
        raise DockerError(
            f"{' '.join(args[:3])}… failed ({p.returncode}): "
            f"{(p.stderr or p.stdout).strip()[:500]}")
    return p.returncode, p.stdout.strip()


def container_running(name: str = CONTAINER_NAME) -> bool:
    _, out = _run_quiet([_docker(), "inspect", "-f",
                         "{{.State.Running}}", name])
    return out == "true"


def build_run_args(s: Settings, workdir: Path, image: str | None = None) -> list[str]:
    """Assemble the full ``docker run`` argv for the vLLM container."""
    image = image or s.image
    cmd = [_docker(), "run", "-d",
           "--name", CONTAINER_NAME,
           "--gpus", "all", "--ipc", "host",
           "--privileged", "--user", "0:0",
           "--memory-swappiness", "0",
           "--ulimit", "memlock=-1",
           "--ulimit", "stack=67108864",
           "--ulimit", "nofile=65536:65536",
           "--restart", "no"]

    for k in _CONTAINER_ENV:
        cmd += ["-e", k]
    if s.gemv:
        cmd += ["-e", "VLLM_FA4_HD256_GEMV=1"]
    else:
        cmd += ["-e", "VLLM_FA4_HD256_GEMV=0"]
    if os.environ.get("HF_TOKEN"):
        cmd += ["-e", f"HF_TOKEN={os.environ['HF_TOKEN']}"]

    d = s.data_dir
    mounts = [
        (str(d / "models"), "/data/models"),
        (str(d / "cache"), "/data/cache"),
        (str(d / "kernels-cache"), "/data/kernels-cache"),
        (str(d / "flashinfer-cache"), "/data/flashinfer-cache"),
        (str(d / "triton-cache"), "/data/triton-cache"),
        (str(d / "tiktoken"), "/data/tiktoken"),
        (str(workdir / "configs"), "/configs:ro"),
        ("/dev/shm", "/dev/shm"),
        ("/etc/localtime", "/etc/localtime:ro"),
        ("/etc/machine-id", "/etc/machine-id:ro"),
    ]
    # A user-local model config (outside the repo's configs/) is mounted as a
    # single file on top of the /configs mount (the more-specific mount wins).
    local_cfg = s.config_path
    repo_cfg = workdir / "configs" / s.model / f"{s.quant}.yaml"
    if local_cfg != repo_cfg and local_cfg.exists():
        mounts.append((str(local_cfg),
                       f"/configs/{s.model}/{s.quant}.yaml:ro"))
    for host, container in mounts:
        if host.startswith("/data") is False and not Path(host).exists() \
                and not host.startswith(("/dev/shm", "/etc/")):
            continue  # optional host files (e.g. /etc/nv_tegra_release)
        cmd += ["-v", f"{host}:{container}"]
    cmd += ["-p", f"{s.port}:8000"]

    serve = (
        "vllm serve "
        f"--config /configs/{s.model}/{s.quant}.yaml "
        "--host 0.0.0.0 "
        "--cpu-offload-gb 0 "
        "--pipeline-parallel-size 1 "
        "--disable-fastapi-docs "
        "--enable-force-include-usage "
        "--enable-prompt-tokens-details "
        "--download-dir /data/models/huggingface")
    cmd += [image, "bash", "-lc", serve]
    return cmd


def serve_up(s: Settings, workdir: Path, wait: bool = True,
             wait_timeout_s: float = 1800.0, dry_run: bool = False) -> None:
    cmd = build_run_args(s, workdir)
    if dry_run:
        print(" ".join(cmd))
        return
    for sub in [s.data_dir / p for p in
                ("models", "cache", "kernels-cache", "flashinfer-cache",
                 "triton-cache", "tiktoken")]:
        sub.mkdir(parents=True, exist_ok=True)
    if container_running():
        print("container already running — stopping the old one first")
        serve_down(s, workdir)
    _run(cmd)
    if wait:
        wait_healthy(s, wait_timeout_s)


def serve_down(s: Settings, workdir: Path, dry_run: bool = False) -> None:
    if not container_running():
        print("nothing to stop (container not running)")
        return
    if dry_run:
        print(f"{_docker()} rm -f {CONTAINER_NAME}")
        return
    _run_quiet([_docker(), "rm", "-f", CONTAINER_NAME])
    print("stopped")


def wait_healthy(s: Settings, timeout_s: float = 1800.0) -> None:
    """Poll the OpenAI /health endpoint until the server answers (model load
    + JIT on Thor takes minutes)."""
    url = f"http://127.0.0.1:{s.port}/health"
    t0 = time.time()
    print(f"waiting for the server at {url} (up to {timeout_s:.0f}s)…")
    while time.time() - t0 < timeout_s:
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    print(f"healthy after {time.time() - t0:.0f}s")
                    return
        except Exception:  # noqa: BLE001
            time.sleep(5)
    raise TimeoutError(
        f"server not healthy after {timeout_s:.0f}s — check: "
        f"mjolnir serve logs")


def status(s: Settings) -> dict:
    running = container_running()
    state: dict = {"container": CONTAINER_NAME, "running": running}
    if running:
        _, out = _run_quiet([_docker(), "inspect", "-f",
                             "{{.State.Status}} started {{.State.StartedAt}}",
                             CONTAINER_NAME])
        state["state"] = out
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{s.port}/health",
                                    timeout=5) as r:
            state["health"] = "ok" if r.status == 200 else f"HTTP {r.status}"
    except Exception:  # noqa: BLE001
        state["health"] = "unreachable"
    return state


def logs(s: Settings, follow: bool = False, tail: int = 100) -> int:
    if not container_running():
        print("container not running", file=sys.stderr)
        return 1
    args = [_docker(), "logs", "--tail", str(tail)]
    if follow:
        args += ["-f"]
    args += [CONTAINER_NAME]
    try:
        return subprocess.call(args)
    except KeyboardInterrupt:
        return 0


# ANSI escape sequences: CSI (SGR colors, cursor moves) + OSC (e.g. hyperlinks).
_ANSI_RE = re.compile(
    r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


def dump_server_log(since_epoch: float | None, out_path: Path) -> bool:
    """Capture the vLLM container's logs to ``out_path``, with ANSI colors
    stripped. ``since_epoch`` (unix seconds) bounds the capture from that
    instant; ``None`` captures the whole container log since it started
    (startup kernel-dispatch lines + everything after — what ``bench perf``
    saves for kernel debugging). Best-effort: returns ``False`` (no raise)
    when the container is absent/not running or docker fails — a log
    capture failure never fails the bench."""
    if not container_running():
        return False
    args = [_docker(), "logs"]
    if since_epoch is not None:
        ts = time.strftime("%Y-%m-%dT%H:%M:%S.000000000Z",
                           time.gmtime(since_epoch))
        args += ["--since", ts]
    args += [CONTAINER_NAME]
    try:
        with out_path.open("w") as f:
            subprocess.run(args, stdout=f, stderr=f, check=False)
    except (OSError, DockerError):
        return False
    try:
        out_path.write_text(_strip_ansi(out_path.read_text()))
    except OSError:
        return False
    return True
