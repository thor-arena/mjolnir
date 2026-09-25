"""Prepare a GEMV'd ``vllm_flash_attn`` tree for the bench containers.

The GEMV kernel is a *test-harness* kernel: it is not baked into the image.
``mjolnir vfa prepare`` builds the tree the bench containers mount over the
image's in-tree ``vllm_flash_attn``:

1. copy the image's ``vllm/vllm_flash_attn`` tree out of a throwaway
   container,
2. drop the GEMV kernel file into ``cute/``,
3. apply ``interface-gemv-dispatch.diff`` (GEMV-only, 4 hunks; env-gated,
   default off in-tree — the bench containers opt in via
   ``VLLM_FA4_HD256_GEMV=1``).
"""
from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path

from mjolnir.config import Settings, VFA_DIST_PATH, find_repo_root

GEMV_PKG_REL = Path("docker/vllm-thor/fa4-gemv-kernel")


def prepare_vfa_tree(s: Settings, out: Path | None = None,
                     image: str | None = None) -> Path:
    root = find_repo_root()
    out = out or root / "vfa-tree"
    img = image or s.image
    pkg = root / GEMV_PKG_REL

    print(f"[mjolnir] preparing GEMV'd vfa tree at {out} (image: {img})")
    out.mkdir(parents=True, exist_ok=True)

    # 1. extract the in-image vllm_flash_attn tree
    extract = ["docker", "run", "--rm",
               "-v", f"{out}:/out", img, "bash", "-lc",
               f"rm -rf /out/vllm_flash_attn && cp -r {VFA_DIST_PATH} /out/vllm_flash_attn"]
    print("[mjolnir]   extracting the in-image vllm_flash_attn tree …",
          file=sys.stderr)
    if subprocess.call(extract, stdout=subprocess.DEVNULL) != 0:
        raise RuntimeError(
            f"failed to extract {VFA_DIST_PATH} from image {img} "
            f"(does the image contain vllm?)")
    tree = out / "vllm_flash_attn"

    # 2. drop the GEMV kernel file into cute/
    kernel = pkg / "sm100_hd256_decode_gemv.py"
    dest = tree / "cute" / "sm100_hd256_decode_gemv.py"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(kernel.read_bytes())
    print("[mjolnir]   kernel file → cute/sm100_hd256_decode_gemv.py",
          file=sys.stderr)

    # 3. apply the dispatch diff (GEMV-only 4 hunks, byte-exact round-trip)
    diff = pkg / "interface-gemv-dispatch.diff"
    p = subprocess.run(["git", "apply", str(diff)],
                       cwd=tree, capture_output=True, text=True)
    if p.returncode != 0:
        p = subprocess.run(["patch", "-p1", "-i", str(diff)],
                           cwd=tree, capture_output=True, text=True)
        if p.returncode != 0:
            raise RuntimeError(
                "could not apply interface-gemv-dispatch.diff — the image's "
                "interface.py has moved past the diff's assumptions:\n"
                f"{p.stderr}")
    print("[mjolnir]   dispatch diff applied (import + split plan + "
          "dtype gate + use_gemv_hd256 block)", file=sys.stderr)
    print(f"\n[mjolnir] done — use with:\n"
          f"  mjolnir bench kernel <task> --vfa-tree {shlex.quote(str(tree))}")
    return tree
