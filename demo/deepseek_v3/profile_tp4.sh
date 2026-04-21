#!/bin/bash
# TP=4 profiling script for DeepSeek V3 (40 layers)
#
# Runs a matrix of workload configs and saves nsys profiles to ~/profile/{config}/
#
# Usage:
#   bash demo/deepseek_v3/profile_tp4.sh
#   GPUS=4,5,6,7 bash demo/deepseek_v3/profile_tp4.sh
#
# Configs: (batch, input_seq, decode_tokens, mtp_spec)
#   batch=1:  input=1K,4K,64K × decode=256 × mtp=0,3
#   batch=8:  input=1K,4K     × decode=256 × mtp=0,3
#   batch=32: input=1K        × decode=256 × mtp=0,3

set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/raid/catalyst/models/DeepSeek-V3}"
PROFILE_DIR="${PROFILE_DIR:-$HOME/profile}"
TP=4
LAYERS="0-39"
MEM_THRESHOLD=500

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
        nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
        exit 1
    fi

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
)

# ── Workload matrix ─────────────────────────────────────────────
# Format: "batch input_seq decode_tokens mtp_spec"
CONFIGS=(
    # batch=1
    "1 1024 256 0"
    "1 1024 256 3"
    "1 4096 256 0"
    "1 4096 256 3"
    "1 65536 256 0"
    "1 65536 256 3"
    # batch=8
    "8 1024 256 0"
    "8 1024 256 3"
    "8 4096 256 0"
    "8 4096 256 3"
    # batch=32
    "32 1024 256 0"
    "32 1024 256 3"
)

max_pages_for_seq() {
    local seq=$1 page_size=128
    echo $(( (seq + page_size - 1) / page_size + 8 ))
}

run_profile() {
    local batch=$1 input_seq=$2 decode=$3 mtp_spec=$4

    local mtp_tag="nomtp"
    local mtp_args=()
    if (( mtp_spec > 0 )); then
        mtp_tag="mtp${mtp_spec}"
        mtp_args=(--mtp --num-speculative-tokens "$mtp_spec")
    fi

    local config_name="tp${TP}_b${batch}_s${input_seq}_d${decode}_${mtp_tag}"
    local out_dir="${PROFILE_DIR}/${config_name}"
    mkdir -p "$out_dir"

    local max_pages
    max_pages=$(max_pages_for_seq "$input_seq")

    local trace_name="${out_dir}/trace"

    echo ""
    echo "================================================================"
    echo "PROFILE: $config_name"
    echo "  batch=$batch input_seq=$input_seq decode=$decode mtp=$mtp_spec"
    echo "  max_pages=$max_pages output=$out_dir"
    echo "================================================================"

    local start_ts
    start_ts=$(date +%s)

    # Run with nsys for system-level profiling
    nsys profile \
        --output "${out_dir}/nsys_report" \
        --force-overwrite true \
        --trace cuda,nvtx,osrt \
        --cuda-memory-usage true \
        mpirun --allow-run-as-root -np $TP "${MPI_ENV_ARGS[@]}" \
        python demo/deepseek_v3/demo.py \
        --model-path "$MODEL_PATH" --use-mirage --layers "$LAYERS" \
        --profiling --trace-name "$trace_name" \
        --max-num-batched-tokens "$batch" \
        --max-seq-length "$input_seq" \
        --max-num-pages "$max_pages" \
        --max-new-tokens "$decode" \
        "${mtp_args[@]}" \
        2>&1 | tee "${out_dir}/stdout.log"

    local elapsed=$(( $(date +%s) - start_ts ))
    echo "Done: $config_name (${elapsed}s)"

    # Extract per-token latency from output
    grep -o 'per-token latency.*ms' "${out_dir}/stdout.log" > "${out_dir}/latency.txt" 2>/dev/null || true
}

# ── Run all configs ──────────────────────────────────────────────
echo "Profile output directory: $PROFILE_DIR"
echo "Total configs: ${#CONFIGS[@]}"

for cfg in "${CONFIGS[@]}"; do
    read -r batch input_seq decode mtp_spec <<< "$cfg"
    run_profile "$batch" "$input_seq" "$decode" "$mtp_spec"
done

echo ""
echo "================================================================"
echo "All profiles saved to $PROFILE_DIR"
ls -la "$PROFILE_DIR"/*/latency.txt 2>/dev/null || echo "(no latency files yet)"
echo "================================================================"
