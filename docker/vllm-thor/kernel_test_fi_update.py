#!/usr/bin/env python3
"""GPU verification of the Thor fused-draft-decode Triton kernel.

`verify_patches.py` only proves the symbols landed in the patched vllm
package; this test proves the actual kernel is *correct* by running
`_advance_fi_decode_plan_kernel` (imported from the installed vllm, not a
copy) against an independent numpy reference over:

  * random seq lens (typical decode batch),
  * exact page boundaries (last page full -> last_page_len == page_size),
  * empty / padded requests (seq_len == 0),
  * a multi-page request (exercises the page-index copy loop),
  * a single request,
  * a large random batch (exercises the O(num_reqs) prefix-sum path),
  * a 20x repeat of one batch (catches any nondeterminism / data race).

It reuses fixed GPU buffers (no per-case allocation) so it runs on a GPU that
already has other memory committed. Run inside the nightly image on a GPU:

    docker run --rm --gpus all --entrypoint python3 <image> /path/to/kernel_test_fi_update.py

Exits non-zero on any mismatch.
"""
from __future__ import annotations

import sys
import time

import numpy as np
import torch

from vllm.v1.attention.backends.flashinfer import _advance_fi_decode_plan_kernel

# The page size the Qwen3.8-27B hybrid resolves to on Thor
# (attention block size set so its page byte-size matches the mamba page).
PAGE_SIZE = 832
N_MAX = 16
MAX_BLOCKS = 40  # covers seq_len up to 40 * 832 = 33280

# The build may run on a Thor that is also serving (the live engine can hold
# most of the 128GB and fragment the free pool), so a small allocation can
# transiently OOM. Retry before concluding the GPU is unavailable, and if it
# truly cannot be allocated, skip (exit 0) rather than flake the build.
_ALLOC_RETRIES = 8
_ALLOC_RETRY_SLEEP = 5.0
# torch 2.13 raises `torch.AcceleratorError` (a RuntimeError subclass) on
# OOM; older/newer versions use `torch.cuda.OutOfMemoryError`. Catch whatever
# is available.
_ALLOC_ERRS: tuple[type, ...] = tuple(
    filter(None, [getattr(torch, "AcceleratorError", None)]
    + [getattr(getattr(torch, "cuda", None), "OutOfMemoryError", None)])
) or (RuntimeError,)


def reference(seq_lens: np.ndarray, block_table: np.ndarray):
    """Independent reference: the exact formula build() uses."""
    n = len(seq_lens)
    blocks = np.zeros(n, dtype=np.int32)
    lpl = np.zeros(n, dtype=np.int32)
    for r in range(n):
        sl = int(seq_lens[r])
        if sl == 0:
            blocks[r] = 0
            lpl[r] = 0
        else:
            nf = sl // PAGE_SIZE
            last = sl - nf * PAGE_SIZE
            blocks[r] = nf + (last > 0)
            lpl[r] = PAGE_SIZE if last == 0 else last
    indptr = np.zeros(n + 1, dtype=np.int32)
    np.cumsum(blocks, out=indptr[1:])
    total = int(indptr[-1])
    indices = np.zeros(total, dtype=np.int32)
    for r in range(n):
        s = int(indptr[r])
        e = int(indptr[r + 1])
        indices[s:e] = block_table[r, : e - s]
    return indptr, lpl, indices


def _alloc(*sizes):
    """Allocate the reusable GPU buffers, retrying through transient OOM."""
    tensors = None
    for attempt in range(_ALLOC_RETRIES):
        try:
            tensors = [
                torch.zeros(n, dtype=torch.int32, device="cuda") for n in sizes
            ]
            break
        except _ALLOC_ERRS:
            if attempt + 1 == _ALLOC_RETRIES:
                return None
            torch.cuda.empty_cache()
            time.sleep(_ALLOC_RETRY_SLEEP)
    return tensors


def main() -> int:
    if not torch.cuda.is_available():
        print(
            "kernel test fi-update: SKIPPED (no CUDA device visible to this "
            "build container). Re-run the build with --gpus all to exercise "
            "the kernel.",
            flush=True,
        )
        return 0
    buffers = _alloc(
        N_MAX, MAX_BLOCKS * N_MAX, N_MAX + 1, N_MAX, MAX_BLOCKS * N_MAX
    )
    if buffers is None:
        print(
            "kernel test fi-update: SKIPPED (could not allocate a few KB on "
            "the GPU after retries; it is saturated by another process). "
            "Run it with a free GPU to exercise the kernel.",
            flush=True,
        )
        return 0
    seq_t, bt_t, indptr, lpl, page_indices = buffers
    # bt_t must be a 2-D view (N_MAX, MAX_BLOCKS).
    bt_t = bt_t.view(N_MAX, MAX_BLOCKS)

    failures: list[str] = []

    def check(seq_lens: np.ndarray, block_table: np.ndarray, label: str) -> None:
        n = len(seq_lens)
        seq_t[:n].copy_(torch.from_numpy(seq_lens.astype(np.int32)).cpu())
        bt_t[:n, : block_table.shape[1]].zero_()
        bt_t[:n, : block_table.shape[1]].copy_(
            torch.from_numpy(block_table.astype(np.int32)).cpu()
        )
        indptr.zero_()
        lpl.zero_()
        page_indices.zero_()
        _advance_fi_decode_plan_kernel[(n,)](
            seq_t,
            bt_t,
            bt_t.stride(0),
            indptr,
            lpl,
            page_indices,
            PAGE_SIZE,
            BLOCK_SIZE=1024,
            num_warps=4,
        )
        torch.cuda.synchronize()

        exp_i, exp_l, exp_p = reference(seq_lens, block_table)
        ok = True
        if not np.array_equal(indptr[: n + 1].cpu().numpy(), exp_i):
            print(f"FAIL {label} indptr\n got={indptr[: n + 1].cpu().numpy()}\n exp={exp_i}")
            ok = False
        if not np.array_equal(lpl[:n].cpu().numpy(), exp_l):
            print(f"FAIL {label} lpl\n got={lpl[:n].cpu().numpy()}\n exp={exp_l}")
            ok = False
        if not np.array_equal(page_indices[: len(exp_p)].cpu().numpy(), exp_p):
            print(
                f"FAIL {label} indices\n"
                f" got={page_indices[: len(exp_p)].cpu().numpy()}\n exp={exp_p}"
            )
            ok = False
        print(("PASS " if ok else "FAIL ") + label)
        if not ok:
            failures.append(label)

    rng = np.random.default_rng(0)

    n = 10
    sl = rng.integers(1, 3000, size=n)
    mb = int(sl.max() // PAGE_SIZE + 1)
    check(sl, np.arange(n * (mb + 2)).reshape(n, mb + 2).astype(np.int32), "T1 random")

    sl = np.array([0, 1, PAGE_SIZE, PAGE_SIZE + 1, 2 * PAGE_SIZE,
                   2 * PAGE_SIZE + 5, PAGE_SIZE * 3 - 1])
    mb = int(sl.max() // PAGE_SIZE + 1)
    check(sl, np.arange(len(sl) * (mb + 2)).reshape(len(sl), mb + 2).astype(np.int32),
          "T2 boundary")

    sl = np.array([0, 0, 500, 0, 1200, 0, 0, 1])
    mb = int(sl.max() // PAGE_SIZE + 1)
    check(sl, np.arange(len(sl) * (mb + 2)).reshape(len(sl), mb + 2).astype(np.int32),
          "T3 padded")

    sl = np.array([PAGE_SIZE * 3, 17, PAGE_SIZE + 3])
    mb = int(sl.max() // PAGE_SIZE + 1)
    check(sl, np.arange(len(sl) * (mb + 2)).reshape(len(sl), mb + 2).astype(np.int32),
          "T4 multipage")

    sl = np.array([12345])
    mb = int(sl.max() // PAGE_SIZE + 1)
    check(sl, np.arange(mb + 2).reshape(1, mb + 2).astype(np.int32), "T5 single")

    n = 16
    sl = rng.integers(0, 20000, size=n)
    mb = int(sl.max() // PAGE_SIZE + 1)
    check(sl, np.arange(n * (mb + 2)).reshape(n, mb + 2).astype(np.int32), "T6 large")

    # Determinism / race check: 20 repeats of one random batch.
    n = 12
    sl = rng.integers(1, 3000, size=n)
    mb = int(sl.max() // PAGE_SIZE + 1)
    bt = np.arange(n * (mb + 2)).reshape(n, mb + 2).astype(np.int32)
    for it in range(20):
        seq_t[:n].copy_(torch.from_numpy(sl.astype(np.int32)).cpu())
        bt_t[:n, : mb + 2].zero_()
        bt_t[:n, : mb + 2].copy_(torch.from_numpy(bt).cpu())
        indptr.zero_()
        lpl.zero_()
        page_indices.zero_()
        _advance_fi_decode_plan_kernel[(n,)](
            seq_t, bt_t, bt_t.stride(0), indptr, lpl, page_indices,
            PAGE_SIZE, BLOCK_SIZE=1024, num_warps=4,
        )
        torch.cuda.synchronize()
        exp_i, exp_l, exp_p = reference(sl, bt)
        if not (
            np.array_equal(indptr[: n + 1].cpu().numpy(), exp_i)
            and np.array_equal(lpl[:n].cpu().numpy(), exp_l)
            and np.array_equal(page_indices[: len(exp_p)].cpu().numpy(), exp_p)
        ):
            print(f"FAIL T7 repeat {it}")
            failures.append("T7")
            break
    else:
        print("PASS T7 repeat x20")

    if failures:
        print(f"KERNEL TEST FAILURES: {failures}")
        return 1
    print("kernel test fi-update: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
