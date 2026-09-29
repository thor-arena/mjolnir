"""Prepare a GEMV'd ``vllm_flash_attn`` tree for the bench containers.

The agent workhorse for kernel iteration: bench containers mount this tree
over the image's in-tree ``vllm_flash_attn``, so the GEMV kernel + dispatch
can be edited without rebuilding the image.

``mjolnir vfa prepare`` builds the tree:

1. copy the image's ``vllm/vllm_flash_attn`` tree out of a throwaway
   container,
2. drop the repo's GEMV kernel file into ``cute/`` (the iteration source of
   truth),
3. apply the GEMV dispatch hunks from the build patch
   ``thor-fa4-hd256-gemv-decode-sm110.patch`` (source of truth) — unless the
   image already ships it (recent images, v12-gemv/v13: the default-on
   build patch bakes the dispatch in; ``git apply -p2 --reverse --check``
   proves it and the step is skipped).
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path

from mjolnir.config import Settings, VFA_DIST_PATH, find_repo_root

GEMV_PKG_REL = Path("docker/vllm-thor/fa4-gemv-kernel")


def prepare_vfa_tree(s: Settings, out: Path | None = None, image: str | None = None) -> Path:
    root = find_repo_root()
    out = out or root / "vfa-tree"
    img = image or s.image
    pkg = root / GEMV_PKG_REL

    print(f"[mjolnir] preparing GEMV'd vfa tree at {out} (image: {img})")
    out.mkdir(parents=True, exist_ok=True)

    # 1. extract the in-image vllm_flash_attn tree. Runs as root (NOT
    #    --user): a previously-benched tree is left with root-owned
    #    __pycache__/ (bench containers run as root and import the mounted
    #    tree), which the host user can't delete — so the ``rm -rf`` must be
    #    root. ``chown -R`` then hands the fresh copy to the host user so it
    #    stays editable for kernel iteration.
    extract = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{out}:/out",
        "--entrypoint",
        "bash",
        img,
        "-lc",
        f"rm -rf /out/vllm_flash_attn && cp -r {VFA_DIST_PATH} /out/vllm_flash_attn"
        f" && chown -R {os.getuid()}:{os.getgid()} /out/vllm_flash_attn",
    ]
    print("[mjolnir]   extracting the in-image vllm_flash_attn tree …", file=sys.stderr)
    if subprocess.call(extract, stdout=subprocess.DEVNULL) != 0:
        raise RuntimeError(f"failed to extract {VFA_DIST_PATH} from image {img} (does the image contain vllm?)")
    tree = out / "vllm_flash_attn"

    # 2. drop the GEMV kernel file into cute/
    kernel = pkg / "sm100_hd256_decode_gemv.py"
    dest = tree / "cute" / "sm100_hd256_decode_gemv.py"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(kernel.read_bytes())
    print("[mjolnir]   kernel file → cute/sm100_hd256_decode_gemv.py", file=sys.stderr)

    # 3. apply the GEMV dispatch hunks — unless the image already ships
    #    them (GEMV-baked: a clean reverse-apply proves they're in-tree).
    #    The BUILD patch is the source of truth (the in-image dispatch may
    #    have evolved past the fa4-gemv-kernel/ reference diff). Paths
    #    carry a vllm/ prefix (a/vllm/vllm_flash_attn/...), so -p2 from the
    #    tree root `out`.
    diff = root / "docker" / "vllm-thor" / "patches" / ("thor-fa4-hd256-gemv-decode-sm110.patch")
    rev = subprocess.run(
        ["git", "apply", "-p2", "--reverse", "--check", str(diff)], cwd=out, capture_output=True, text=True
    )
    if rev.returncode == 0:
        print("[mjolnir]   dispatch diff already in-tree (GEMV-baked image) — skipped", file=sys.stderr)
    else:
        p = subprocess.run(["git", "apply", "-p2", str(diff)], cwd=out, capture_output=True, text=True)
        if p.returncode != 0:
            p = subprocess.run(["patch", "-p2", "-i", str(diff)], cwd=out, capture_output=True, text=True)
            if p.returncode != 0:
                raise RuntimeError(
                    "could not apply the GEMV dispatch patch "
                    f"({diff.name}) — the image's interface.py has moved "
                    f"past its assumptions:\n{p.stdout}{p.stderr}"
                )
        print(
            "[mjolnir]   dispatch patch applied (import + split plan + dtype gate + use_gemv_hd256 block)",
            file=sys.stderr,
        )
    print(f"\n[mjolnir] done — use with:\n  mjolnir bench kernel <task> --vfa-tree {shlex.quote(str(tree))}")
    return tree
