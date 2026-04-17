#!/bin/bash
# TP=4 correctness test for DeepSeek V3 on Mirage MPK
# Usage: bash test_deepseek_v3_tp.sh [LAYERS] [MAX_SEQ]
#
# Prerequisites: mpi4py installed, NVSHMEM available, 4 free GPUs
# This script:
# 1. Runs single-GPU reference on GPU 0 to get expected output
# 2. Runs TP=4 MPK on GPUs 0-3 via mpirun
# 3. Compares tokens

set -e

LAYERS="${1:-0,1,2,3,4,5,6,7,8,9}"  # first 10 layers by default
MAX_SEQ="${2:-15}"
MTP_FLAG="${3:---mtp}"  # pass "--mtp" or ""
SPEC_TOKENS="${4:-1}"
MODEL_PATH="/raid/catalyst/models/DeepSeek-V3"
DEMO="/home/muhengl/mirage/demo/deepseek_v3/demo.py"
VENV="/raid/user_data/muhengl/.venv"
MPI_BIN="/usr/mpi/gcc/openmpi-4.1.9a1/bin"
MPI_LIB="/usr/mpi/gcc/openmpi-4.1.9a1/lib"

export PATH="${MPI_BIN}:${PATH}"
export LD_LIBRARY_PATH="${MPI_LIB}:${LD_LIBRARY_PATH}"

echo "============================================================"
echo "DeepSeek V3 TP=4 Correctness Test"
echo "  Layers: ${LAYERS}"
echo "  Max seq: ${MAX_SEQ}"
echo "  MTP: ${MTP_FLAG} spec=${SPEC_TOKENS}"
echo "============================================================"

# Step 1: Single-GPU reference (use GPU 0)
echo ""
echo "[Step 1] Running single-GPU reference on GPU 0..."
source "${VENV}/bin/activate"
cd /dev/shm/mh_jit

CUDA_VISIBLE_DEVICES=0 MPK_NO_RESIDUAL=1 timeout 600 python -u "${DEMO}" \
    --model-path "${MODEL_PATH}" \
    --use-mirage --correctness \
    --layers "${LAYERS}" \
    --max-seq-length "${MAX_SEQ}" \
    --max-num-pages 4 \
    --max-num-batched-tokens 1 \
    ${MTP_FLAG:+${MTP_FLAG} --num-speculative-tokens ${SPEC_TOKENS}} \
    2>&1 | tee /dev/shm/tp_ref.log

REF_TOKEN=$(grep "MPK output token:" /dev/shm/tp_ref.log | awk '{print $NF}')
REF_COSINE=$(grep "cosine=" /dev/shm/tp_ref.log | tail -1 | grep -o 'cosine=[^ ]*' | head -1)
echo ""
echo "Reference: token=${REF_TOKEN} ${REF_COSINE}"

# Step 2: TP=4 MPK via mpirun
echo ""
echo "[Step 2] Running TP=4 MPK on GPUs 0-3..."
CUDA_VISIBLE_DEVICES=0,1,2,3 MPK_NO_RESIDUAL=1 \
    mpirun --allow-run-as-root -np 4 \
    --mca btl_tcp_if_include lo \
    python -u "${DEMO}" \
    --model-path "${MODEL_PATH}" \
    --use-mirage --correctness \
    --layers "${LAYERS}" \
    --max-seq-length "${MAX_SEQ}" \
    --max-num-pages 4 \
    --max-num-batched-tokens 1 \
    ${MTP_FLAG:+${MTP_FLAG} --num-speculative-tokens ${SPEC_TOKENS}} \
    2>&1 | tee /dev/shm/tp_mpk.log

TP_TOKEN=$(grep "MPK output token:" /dev/shm/tp_mpk.log | awk '{print $NF}')
TP_COSINE=$(grep "cosine=" /dev/shm/tp_mpk.log | tail -1 | grep -o 'cosine=[^ ]*' | head -1)
echo ""
echo "============================================================"
echo "Results:"
echo "  Single-GPU: token=${REF_TOKEN} ${REF_COSINE}"
echo "  TP=4:       token=${TP_TOKEN} ${TP_COSINE}"
if [ "${REF_TOKEN}" = "${TP_TOKEN}" ]; then
    echo "  STATUS: PASS (tokens match)"
else
    echo "  STATUS: FAIL (tokens differ)"
fi
echo "============================================================"
