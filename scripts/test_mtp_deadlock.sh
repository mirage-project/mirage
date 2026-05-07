#!/bin/bash
# Smallest-possible workload to test whether enabling
# `use_mtp_prefill_attention=True` in builder.py:2144 still triggers the
# Q_LEN>=9 deadlock that the comment from commit 54de0a31 (2026-04-30)
# warned about.
#
# Workload: TP=1 (no MPI), layers=0-1, mbt=16 (prefill), mtp=1, prompt=8,
# decode=1. About as small as the demo can run while still being a
# Q_LEN>=9 chunked-prefill case.
#
# If it hangs > 5 min: deadlock confirmed; revert builder.py.
# If it completes: run the full regression suite.
#
# Usage:
#   bash scripts/test_mtp_deadlock.sh

set -euo pipefail

MIRAGE_REPO="${MIRAGE_REPO:-/home/muhengl/mirage}"
MODEL_PATH="${MODEL_PATH:-/raid/catalyst/models/DeepSeek-V3}"
MPI_HOME="${MPI_HOME:-/usr/mpi/gcc/openmpi-4.1.9a1}"
NVSHMEM_HOME="${NVSHMEM_HOME:-/home/muhengl/local/nvshmem-3.6.5-dev/usr}"
NVSHMEM_PRELOAD="${NVSHMEM_PRELOAD:-/home/muhengl/local/nvshmem-3.6.5-extract/usr/lib/x86_64-linux-gnu/nvshmem/13/libnvshmem_host.so.3.6.5}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.2}"
VENV="${MIRAGE_VENV:-$MIRAGE_REPO/.venv}"

GPU="${GPU:-3}"
TIMEOUT="${TIMEOUT:-300}"  # 5-minute hard timeout — deadlock detection

OUT="${OUT:-$MIRAGE_REPO/outputs/mtp_deadlock_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT"

export PATH="$CUDA_HOME/bin:$MPI_HOME/bin:$PATH"
export CUDA_HOME
export CUDA_VISIBLE_DEVICES="$GPU"
export LD_LIBRARY_PATH="$NVSHMEM_HOME/lib/x86_64-linux-gnu/nvshmem/13:$MPI_HOME/lib:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="$NVSHMEM_PRELOAD"
export MPI_INC_PATH="$MPI_HOME/include"
export MPI_LIB_PATH="$MPI_HOME/lib"
export NVSHMEM_INC_PATH="$NVSHMEM_HOME/include/nvshmem_13"
export NVSHMEM_LIB_PATH="$NVSHMEM_HOME/lib/x86_64-linux-gnu/nvshmem/13"
export NVSHMEM_SYMMETRIC_SIZE=4294967296

LOG="$OUT/run.log"

echo "================================================================"
echo "MTP deadlock test"
echo "  GPU=$GPU TIMEOUT=${TIMEOUT}s  cuda=$CUDA_HOME  venv=$VENV"
echo "  log=$LOG"
echo "================================================================"
START=$(date +%s)
set +e
timeout "$TIMEOUT" "$VENV/bin/python" "$MIRAGE_REPO/demo/deepseek_v3/demo.py" \
    --model-path "$MODEL_PATH" --use-mirage \
    --layers 0-1 \
    --max-num-batched-tokens 16 \
    --max-num-batched-requests 1 \
    --prompt-length 8 \
    --max-new-tokens 1 \
    --max-seq-length 32 \
    --max-num-pages 4 \
    --page-size 128 \
    --mtp 1 \
    --ep-size 1 \
    --ignore-eos \
    > "$LOG" 2>&1
RC=$?
set -e
EL=$(( $(date +%s) - START ))

if (( RC == 124 )); then
    echo "DEADLOCK or HANG: timed out after ${EL}s"
    echo "RESULT: HANG"
    exit 2
elif (( RC != 0 )); then
    echo "FAIL: exit $RC after ${EL}s. Last 30 lines:"
    tail -30 "$LOG"
    echo "RESULT: FAIL"
    exit 1
else
    echo "PASS: completed in ${EL}s"
    grep -E "per-token latency|Generation" "$LOG" | tail -3
    echo "RESULT: PASS"
    exit 0
fi
