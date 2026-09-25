#!/usr/bin/env bash
# Driver: run all 4 GEMV-B1 bench modes sequentially (each waits for its own
# clean 0/0 window internally). One docker per mode (fresh JIT + TMPDIR).
set -u
TASK=<private-repo>/task/fa4-hd256-fp8
IMG=mjolnir/vllm-thor:qwen38-sm110-v11
VFA=<vfa-tree>
LOG=<scratch>
mkdir -p <scratch>
: > "$LOG"

for MODE in gemv_dense gemv_paged fa4_1cta flashinfer; do
  echo "==================== MODE: $MODE ====================" >> "$LOG"
  docker run --rm --gpus all --network host --entrypoint python3 \
    -v "$VFA":/usr/local/lib/python3.12/dist-packages/vllm/vllm_flash_attn \
    -v "$TASK":/p \
    -e TMPDIR=<scratch>"$MODE" \
    "$IMG" /p/gemv-decode-bench.py --mode "$MODE" \
      --out "/p/gemv-decode-bench-$MODE.json" >> "$LOG" 2>&1
  rc=$?
  echo "---- MODE $MODE exit=$rc ----" >> "$LOG"
done

echo "==================== ALL MODES DONE ====================" >> "$LOG"
