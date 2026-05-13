#!/usr/bin/env bash
# QKV-a fusion smoke test: DSv3 TP=4 EP=2 layers 0-3, prompt=128 + 1 new tok.
set -euo pipefail

ts=${1:-$(date +%H%M%S)}
mode=${2:-fused}  # fused | baseline
OUT=/home/muhengl/mirage/outputs/dpskv3_qkva_${mode}_${ts}
mkdir -p "${OUT}/build" "${OUT}/dump"

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

if [ "${mode}" = "fused" ]; then
    export MPK_DSV3_QKV_A_FUSED=1
    export MPK_DEEPSEEK_WEIGHT_CACHE_DIR=/tmp/dpskv3_v8_weight_cache_qkva_fused
else
    export MPK_DSV3_QKV_A_FUSED=0
    export MPK_DEEPSEEK_WEIGHT_CACHE_DIR=/tmp/dpskv3_v8_weight_cache_qkva_baseline
fi
mkdir -p "${MPK_DEEPSEEK_WEIGHT_CACHE_DIR}"

cd /home/muhengl/mirage
CUDA_VISIBLE_DEVICES=0,1,2,3 mpirun --allow-run-as-root -np 4 \
    -x CUDA_VISIBLE_DEVICES -x LD_LIBRARY_PATH -x LD_PRELOAD -x PATH \
    -x MPI_INC_PATH -x MPI_LIB_PATH -x NVSHMEM_INC_PATH -x NVSHMEM_LIB_PATH \
    -x NVSHMEM_MAX_TEAMS \
    -x MPK_DSV3_QKV_A_FUSED \
    -x MPK_DEEPSEEK_WEIGHT_CACHE_DIR \
    /home/muhengl/mirage/.venv/bin/python demo/deepseek_v3/demo.py \
    --model-path /raid/catalyst/models/DeepSeek-V3 \
    --use-mirage \
    --max-num-batched-tokens 128 \
    --max-num-batched-requests 1 \
    --page-size 128 --max-num-pages 2 \
    --max-seq-length 256 \
    --prompt-length 128 --ignore-eos \
    --max-new-tokens 1 \
    --layers 0-3 \
    --mtp 0 --ep-size 2 \
    --output-dir "${OUT}/build" \
    --dump-hidden-dir "${OUT}/dump" \
    > "${OUT}/run.log" 2>&1

rc=$?
echo "OUT=${OUT}"
echo "rc=${rc}"
grep -E "per-token latency|ERROR|Traceback|Assertion" "${OUT}/run.log" | tail -10 || true
