#!/usr/bin/env bash
# Qwen3 TP=4 smoke verifying nothing broke after the fuse-Q + ROPE kernel changes.
set -euo pipefail

ts=${1:-$(date +%H%M%S)}
OUT=/home/muhengl/mirage/outputs/qwen3_tp4_smoke_${ts}
mkdir -p "${OUT}"

export MPI_HOME=${MPI_HOME:-/usr/mpi/gcc/openmpi-4.1.9a1}
export PATH=$MPI_HOME/bin:$PATH
export MPI_INC_PATH=$MPI_HOME/include
export MPI_LIB_PATH=$MPI_HOME/lib

export NVSHMEM_HOME=${NVSHMEM_HOME:-/home/muhengl/local/nvshmem-3.6.5-dev/usr}
export NVSHMEM_INC_PATH=$NVSHMEM_HOME/include/nvshmem_13
export NVSHMEM_LIB_PATH=$NVSHMEM_HOME/lib/x86_64-linux-gnu/nvshmem/13
export LD_LIBRARY_PATH=$NVSHMEM_LIB_PATH:$MPI_HOME/lib:${LD_LIBRARY_PATH:-}
export LD_PRELOAD=$NVSHMEM_LIB_PATH/libnvshmem_host.so.3.6.5
export NVSHMEM_MAX_TEAMS=128

cd /home/muhengl/mirage
CUDA_VISIBLE_DEVICES=0,1,2,3 mpirun --allow-run-as-root -np 4 \
    -x CUDA_VISIBLE_DEVICES -x LD_LIBRARY_PATH -x LD_PRELOAD -x PATH \
    -x MPI_INC_PATH -x MPI_LIB_PATH -x NVSHMEM_INC_PATH -x NVSHMEM_LIB_PATH \
    -x NVSHMEM_MAX_TEAMS \
    /home/muhengl/mirage/.venv/bin/python demo/qwen3/demo.py \
    --use-mirage --max-new-tokens 4 --ignore-eos \
    --max-num-batched-tokens 1 --max-num-batched-requests 16 \
    --max-seq-length 4096 --max-num-pages 16 --page-size 4096 \
    --output-dir "${OUT}" \
    > "${OUT}/run.log" 2>&1

rc=$?
echo "OUT=${OUT}"
echo "rc=${rc}"
grep -E "per-token latency|Generated|ERROR|Traceback" "${OUT}/run.log" | tail -10 || true
