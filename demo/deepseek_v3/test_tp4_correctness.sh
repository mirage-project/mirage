#!/bin/bash
# TP=4 correctness + high-load test for DeepSeek V3 (40 layers)
#
# Usage:
#   bash demo/deepseek_v3/test_tp4_correctness.sh
#   GPUS=4,5,6,7 bash demo/deepseek_v3/test_tp4_correctness.sh
#
# Scans for 4 idle GPUs (<500 MB used) unless GPUS is set explicitly.

set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/raid/catalyst/models/DeepSeek-V3}"
TP=4
LAYERS="0-39"
MEM_THRESHOLD=500  # MiB — GPU with usage above this is considered busy

# ── GPU selection ────────────────────────────────────────────────
if [[ -z "${GPUS:-}" ]]; then
    echo "Scanning for $TP idle GPUs (memory < ${MEM_THRESHOLD} MiB)..."
    IDLE_GPUS=()
    while IFS=', ' read -r idx mem util; do
        mem_val=${mem%% *}
        if (( mem_val < MEM_THRESHOLD )); then
            IDLE_GPUS+=("$idx")
        fi
    done < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
                        --format=csv,noheader,nounits)

    if (( ${#IDLE_GPUS[@]} < TP )); then
        echo "FATAL: Only ${#IDLE_GPUS[@]} idle GPUs found, need $TP."
        echo "Idle GPUs: ${IDLE_GPUS[*]:-none}"
        nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
        exit 1
    fi

    # Take the first TP idle GPUs
    SELECTED=("${IDLE_GPUS[@]:0:$TP}")
    GPUS=$(IFS=,; echo "${SELECTED[*]}")
    echo "Selected GPUs: $GPUS"
else
    echo "Using user-specified GPUs: $GPUS"
fi

export CUDA_VISIBLE_DEVICES="$GPUS"

# ── Environment ──────────────────────────────────────────────────
source /raid/user_data/muhengl/.venv/bin/activate
export PATH=/usr/mpi/gcc/openmpi-4.1.9a1/bin:$PATH
export LD_LIBRARY_PATH=/home/muhengl/local/nvshmem-3.6.5-dev/usr/lib/x86_64-linux-gnu/nvshmem/13:/usr/mpi/gcc/openmpi-4.1.9a1/lib:${LD_LIBRARY_PATH:-}
export LD_PRELOAD=/home/muhengl/local/nvshmem-3.6.5-extract/usr/lib/x86_64-linux-gnu/nvshmem/13/libnvshmem_host.so.3.6.5
export MPI_INC_PATH=/usr/mpi/gcc/openmpi-4.1.9a1/include
export MPI_LIB_PATH=/usr/mpi/gcc/openmpi-4.1.9a1/lib
export NVSHMEM_INC_PATH=/home/muhengl/local/nvshmem-3.6.5-dev/usr/include/nvshmem_13
export NVSHMEM_LIB_PATH=/home/muhengl/local/nvshmem-3.6.5-dev/usr/lib/x86_64-linux-gnu/nvshmem/13

MPI_ENV_ARGS=(
    -x CUDA_VISIBLE_DEVICES -x LD_LIBRARY_PATH -x LD_PRELOAD -x PATH
    -x MPI_INC_PATH -x MPI_LIB_PATH -x NVSHMEM_INC_PATH -x NVSHMEM_LIB_PATH
    -x MPK_SKIP_ATTN -x MPK_FUSE_RESIDUAL
)

run_test() {
    local desc="$1"; shift
    echo ""
    echo "================================================================"
    echo "TEST: $desc"
    echo "================================================================"
    local start_ts
    start_ts=$(date +%s)
    if mpirun --allow-run-as-root -np $TP "${MPI_ENV_ARGS[@]}" "$@"; then
        local elapsed=$(( $(date +%s) - start_ts ))
        echo "PASS: $desc  (${elapsed}s)"
    else
        local elapsed=$(( $(date +%s) - start_ts ))
        echo "FAIL: $desc  (${elapsed}s, exit code $?)"
        return 1
    fi
}

PASS=0
FAIL=0

run_and_track() {
    if run_test "$@"; then
        PASS=$((PASS + 1))
    else
        FAIL=$((FAIL + 1))
    fi
}

# ── Test 1: Single MoE layer, skip attn (MLP/MoE sanity check) ──
export MPK_SKIP_ATTN=1
run_and_track "TP=$TP MoE layer 3, skip attn (MLP/MoE only)" \
    python demo/deepseek_v3/demo.py \
    --model-path "$MODEL_PATH" --use-mirage --correctness --layers 3 \
    --max-num-batched-tokens 1 --max-seq-length 512

# ── Test 2: Single MoE layer, WITH MLA (full layer correctness) ─
unset MPK_SKIP_ATTN
run_and_track "TP=$TP MoE layer 3, with MLA (full layer)" \
    python demo/deepseek_v3/demo.py \
    --model-path "$MODEL_PATH" --use-mirage --correctness --layers 3 \
    --max-num-batched-tokens 1 --max-seq-length 512

# ── Test 3: Dense layer 0, with MLA ─────────────────────────────
run_and_track "TP=$TP dense layer 0, with MLA" \
    python demo/deepseek_v3/demo.py \
    --model-path "$MODEL_PATH" --use-mirage --correctness --layers 0 \
    --max-num-batched-tokens 1 --max-seq-length 512

# ── Test 4: 40 layers, short decode ─────────────────────────────
run_and_track "TP=$TP 40 layers, batch=1, seq=1024, 32 tokens" \
    python demo/deepseek_v3/demo.py \
    --model-path "$MODEL_PATH" --use-mirage --layers "$LAYERS" \
    --max-num-batched-tokens 1 --max-seq-length 1024 \
    --max-num-pages 16 --max-new-tokens 32

# ── Test 5: 40 layers, longer decode ────────────────────────────
run_and_track "TP=$TP 40 layers, batch=1, seq=4096, 256 tokens" \
    python demo/deepseek_v3/demo.py \
    --model-path "$MODEL_PATH" --use-mirage --layers "$LAYERS" \
    --max-num-batched-tokens 1 --max-seq-length 4096 \
    --max-num-pages 64 --max-new-tokens 256

# ── Test 6: Higher batch size ────────────────────────────────────
run_and_track "TP=$TP 40 layers, batch=8, seq=4096, 128 tokens" \
    python demo/deepseek_v3/demo.py \
    --model-path "$MODEL_PATH" --use-mirage --layers "$LAYERS" \
    --max-num-batched-tokens 8 --max-seq-length 4096 \
    --max-num-pages 64 --max-new-tokens 128

# ── Test 7: MTP decode ──────────────────────────────────────────
run_and_track "TP=$TP 40 layers, batch=1, seq=4096, MTP spec=3" \
    python demo/deepseek_v3/demo.py \
    --model-path "$MODEL_PATH" --use-mirage --layers "$LAYERS" \
    --mtp --num-speculative-tokens 3 \
    --max-num-batched-tokens 1 --max-seq-length 4096 \
    --max-num-pages 64 --max-new-tokens 128

# ── Summary ──────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "RESULTS: $PASS passed, $FAIL failed (total $((PASS + FAIL)))"
echo "================================================================"
[[ $FAIL -eq 0 ]] || exit 1
