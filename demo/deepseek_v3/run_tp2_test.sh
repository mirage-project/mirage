#!/bin/bash
# TP=2 MoE correctness test for DeepSeek V3
# Usage: CUDA_VISIBLE_DEVICES=4,6 bash demo/deepseek_v3/run_tp2_test.sh

set -e

source /raid/user_data/muhengl/.venv/bin/activate
export PATH=/usr/mpi/gcc/openmpi-4.1.9a1/bin:$PATH
export LD_LIBRARY_PATH=/home/muhengl/local/nvshmem-3.6.5-dev/usr/lib/x86_64-linux-gnu/nvshmem/13:/usr/mpi/gcc/openmpi-4.1.9a1/lib:$LD_LIBRARY_PATH
# Preload NVSHMEM 3.6.5 to override system nvshmem in ldconfig cache
# (RUNPATH has lower priority than ldcache; LD_PRELOAD overrides everything)
export LD_PRELOAD=/home/muhengl/local/nvshmem-3.6.5-extract/usr/lib/x86_64-linux-gnu/nvshmem/13/libnvshmem_host.so.3.6.5
export MPI_INC_PATH=/usr/mpi/gcc/openmpi-4.1.9a1/include
export MPI_LIB_PATH=/usr/mpi/gcc/openmpi-4.1.9a1/lib
export NVSHMEM_INC_PATH=/home/muhengl/local/nvshmem-3.6.5-dev/usr/include/nvshmem_13
export NVSHMEM_LIB_PATH=/home/muhengl/local/nvshmem-3.6.5-dev/usr/lib/x86_64-linux-gnu/nvshmem/13

# Optional: skip attention for MoE-only test
export MPK_SKIP_ATTN=${MPK_SKIP_ATTN:-1}
# MPK_AR_LOCAL_COPY=1: bypass NVSHMEM AllReduce (output=input) to test without comms
export MPK_AR_LOCAL_COPY=${MPK_AR_LOCAL_COPY:-0}
# MPK_SKIP_ALLREDUCE=1: skip allreduce tasks entirely in builder (no AR blocks in kernel)
export MPK_SKIP_ALLREDUCE=${MPK_SKIP_ALLREDUCE:-0}
# MPK_NO_RESIDUAL=1: skip residual connections (match reference)
export MPK_NO_RESIDUAL=${MPK_NO_RESIDUAL:-0}

echo "=== Environment ==="
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "NVSHMEM_INC_PATH=$NVSHMEM_INC_PATH"
echo "NVSHMEM_LIB_PATH=$NVSHMEM_LIB_PATH"
echo "MPI_INC_PATH=$MPI_INC_PATH"
echo "MPI_LIB_PATH=$MPI_LIB_PATH"
echo "MPK_SKIP_ATTN=$MPK_SKIP_ATTN"
echo "=================="

mpirun --allow-run-as-root -np 2 \
    -x CUDA_VISIBLE_DEVICES \
    -x LD_LIBRARY_PATH \
    -x LD_PRELOAD \
    -x PATH \
    -x MPK_SKIP_ATTN \
    -x MPK_AR_LOCAL_COPY \
    -x MPK_SKIP_ALLREDUCE \
    -x MPK_NO_RESIDUAL \
    -x MPK_NO_NVSHMEM \
    -x MPI_INC_PATH \
    -x MPI_LIB_PATH \
    -x NVSHMEM_INC_PATH \
    -x NVSHMEM_LIB_PATH \
    python demo/deepseek_v3/demo.py \
    --model-path /raid/catalyst/models/DeepSeek-V3 \
    --use-mirage --correctness --layers 3 \
    --max-num-batched-tokens 1 --max-seq-length 512 \
    "$@"
