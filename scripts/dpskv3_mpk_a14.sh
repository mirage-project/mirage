#!/bin/bash
# Run MPK demo on the A14 super-long workload (16K prompt + 256 decode).
# This is a pure perf test — no token-id comparison expected. The user
# wanted to see if MPK can sustain the full 16K prefill on layers 0-19
# at TP=4 EP=2 in "几十秒" (tens of seconds).
#
# Usage:
#   bash scripts/dpskv3_mpk_a14.sh [--gpus 1,3,4,5] [--mtp 0|1]

set -uo pipefail

GPUS="${GPUS:-1,3,4,5}"
MTP=0
LAYERS=0-19
TP=4
EP=2
PROMPT_LEN=16384
DECODE=256
MBT=128
MODEL_PATH="${MODEL_PATH:-/raid/catalyst/models/DeepSeek-V3}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus) GPUS="$2"; shift 2;;
        --mtp) MTP="$2"; shift 2;;
        --layers) LAYERS="$2"; shift 2;;
        --prompt-len) PROMPT_LEN="$2"; shift 2;;
        --decode) DECODE="$2"; shift 2;;
        --mbt) MBT="$2"; shift 2;;
        *) echo "Unknown arg: $1" >&2; exit 1;;
    esac
done

REPO=/home/muhengl/mirage
VENV=/raid/user_data/muhengl/.venv

OUT_BASE="outputs/dpskv3_mpk_A14_mtp${MTP}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT_BASE"
LOG="$OUT_BASE/run.log"
TOKENS="$OUT_BASE/tokens.json"

SEQ=$(( PROMPT_LEN + DECODE + 32 ))
PAGES=$(( ((SEQ + 127) / 128) + 4 ))

echo "[A14] prompt=$PROMPT_LEN decode=$DECODE mtp=$MTP seq=$SEQ pages=$PAGES gpus=$GPUS"
echo "[A14] out=$OUT_BASE"

MTP_ARG=""
if (( MTP > 0 )); then
    MTP_ARG="--mtp $MTP"
fi

CUDA_VISIBLE_DEVICES="$GPUS" \
LD_LIBRARY_PATH=/home/muhengl/local/nvshmem-3.6.5-dev/usr/lib/x86_64-linux-gnu/nvshmem/13:/usr/mpi/gcc/openmpi-4.1.9a1/lib \
LD_PRELOAD=/home/muhengl/local/nvshmem-3.6.5-extract/usr/lib/x86_64-linux-gnu/nvshmem/13/libnvshmem_host.so.3.6.5 \
NVSHMEM_SYMMETRIC_SIZE=4294967296 \
PATH=/usr/local/cuda-13.2/bin:/usr/mpi/gcc/openmpi-4.1.9a1/bin:$PATH \
MPI_INC_PATH=/usr/mpi/gcc/openmpi-4.1.9a1/include \
MPI_LIB_PATH=/usr/mpi/gcc/openmpi-4.1.9a1/lib \
NVSHMEM_INC_PATH=/home/muhengl/local/nvshmem-3.6.5-dev/usr/include/nvshmem_13 \
NVSHMEM_LIB_PATH=/home/muhengl/local/nvshmem-3.6.5-dev/usr/lib/x86_64-linux-gnu/nvshmem/13 \
timeout 3600 mpirun --allow-run-as-root -np "$TP" \
    -x CUDA_VISIBLE_DEVICES -x LD_LIBRARY_PATH -x LD_PRELOAD -x PATH \
    -x MPI_INC_PATH -x MPI_LIB_PATH -x NVSHMEM_INC_PATH -x NVSHMEM_LIB_PATH \
    -x NVSHMEM_SYMMETRIC_SIZE \
    "$VENV/bin/python" "$REPO/demo/deepseek_v3/demo.py" \
        --model-path "$MODEL_PATH" --use-mirage \
        --layers "$LAYERS" \
        --max-num-batched-tokens "$MBT" \
        --max-num-batched-requests 1 \
        --prompt-length "$PROMPT_LEN" \
        --max-new-tokens "$DECODE" \
        --max-seq-length "$SEQ" \
        --max-num-pages "$PAGES" \
        --page-size 128 \
        --ep-size "$EP" \
        --ignore-eos \
        --save-tokens "$TOKENS" \
        $MTP_ARG \
        > "$LOG" 2>&1
RC=$?
echo "[A14] rc=$RC log=$LOG"
grep -E "per-token latency|prompt length" "$LOG" | tail -3
