#!/bin/bash
# TP=4 profiling for layers 0-10 (11 layers), tuned for Perfetto analysis.
#
# Produces one .perfetto-trace per rank per config under $PROFILE_DIR/{config}/.
# Load files at https://ui.perfetto.dev to inspect per-SM task timelines.
#
# Usage:
#   bash demo/deepseek_v3/profile_tp4_layers0-10.sh
#   GPUS=4,5,6,7 bash demo/deepseek_v3/profile_tp4_layers0-10.sh
#
# Decode length kept short (16 steps) so the 3000*128 uint64 profiler buffer
# allocated in demo.py doesn't overflow — each task+rank writes ~2-3 uint64
# entries per event, and we emit events for every worker step across 11 layers.
#
# Configs cover the three regimes worth inspecting:
#   1) batch=1  s=4K  decode=16 nomtp — decode-heavy steady state, short prefill
#   2) batch=1  s=4K  decode=16 mtp=3 — MTP draft/verify loop overhead
#   3) batch=8  s=1K  decode=16 nomtp — multi-request batch contention

set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/raid/catalyst/models/DeepSeek-V3}"
PROFILE_DIR="${PROFILE_DIR:-$HOME/profile/tp4_layers0-10}"
TP=4
LAYERS="0-10"
MEM_THRESHOLD="${MEM_THRESHOLD:-500}"
UTIL_THRESHOLD="${UTIL_THRESHOLD:-5}"
MAX_WAIT="${MAX_WAIT:-7200}"
POLL_INTERVAL="${POLL_INTERVAL:-30}"

find_idle_gpus() {
    local idle=()
    while IFS=', ' read -r idx mem util; do
        mem_val=${mem%% *}
        util_val=${util%% *}
        if (( mem_val < MEM_THRESHOLD )) && (( util_val < UTIL_THRESHOLD )); then
            idle+=("$idx")
        fi
    done < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
                        --format=csv,noheader,nounits)
    echo "${idle[@]}"
}

if [[ -z "${GPUS:-}" ]]; then
    echo "Polling for $TP idle GPUs (mem<${MEM_THRESHOLD}MiB, util<${UTIL_THRESHOLD}%, max wait ${MAX_WAIT}s)..."
    start_wait=$(date +%s)
    while :; do
        IDLE_GPUS=($(find_idle_gpus))
        n=${#IDLE_GPUS[@]}
        elapsed=$(( $(date +%s) - start_wait ))
        if (( n >= TP )); then
            SELECTED=("${IDLE_GPUS[@]:0:$TP}")
            GPUS=$(IFS=,; echo "${SELECTED[*]}")
            echo "Found $TP idle GPUs after ${elapsed}s: $GPUS"
            break
        fi
        if (( elapsed >= MAX_WAIT )); then
            echo "FATAL: timed out after ${elapsed}s; only $n/$TP idle."
            nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
            exit 1
        fi
        echo "  waiting... idle=[${IDLE_GPUS[*]}] ($n/$TP)"
        sleep "$POLL_INTERVAL"
    done
else
    echo "Using user-specified GPUs: $GPUS"
fi

export CUDA_VISIBLE_DEVICES="$GPUS"

# Environment (matches profile_tp4.sh / demo/deepseek_v3/readme.md)
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

# Workload matrix: "batch prompt_len decode_tokens mtp_spec"
CONFIGS=(
    "1 4096 16 0"
    "1 4096 16 3"
    "8 1024 16 0"
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
        mtp_args=(--mtp "$mtp_spec")
    fi

    local config_name="tp${TP}_l0-10_b${batch}_s${input_seq}_d${decode}_${mtp_tag}"
    local out_dir="${PROFILE_DIR}/${config_name}"
    mkdir -p "$out_dir"

    local seq_needed=$(( input_seq + decode + 32 ))
    local max_pages
    max_pages=$(max_pages_for_seq "$seq_needed")

    local trace_name="${out_dir}/trace"

    echo ""
    echo "================================================================"
    echo "PROFILE: $config_name"
    echo "  batch=$batch prompt_len=$input_seq decode=$decode mtp=$mtp_spec"
    echo "  max_seq_length=$seq_needed max_pages=$max_pages output=$out_dir"
    echo "================================================================"

    local start_ts=$(date +%s)

    mpirun --allow-run-as-root -np $TP "${MPI_ENV_ARGS[@]}" \
        python demo/deepseek_v3/demo.py \
        --model-path "$MODEL_PATH" --use-mirage --layers "$LAYERS" \
        --profiling --trace-name "$trace_name" \
        --max-num-batched-tokens "$batch" \
        --prompt-length "$input_seq" \
        --max-seq-length "$seq_needed" \
        --max-num-pages "$max_pages" \
        --max-new-tokens "$decode" \
        "${mtp_args[@]}" \
        2>&1 | tee "${out_dir}/stdout.log"

    local elapsed=$(( $(date +%s) - start_ts ))
    echo "Done: $config_name (${elapsed}s)"
    grep -o 'per-token latency.*ms' "${out_dir}/stdout.log" > "${out_dir}/latency.txt" 2>/dev/null || true
    ls -la "$out_dir"/*.perfetto-trace 2>/dev/null || echo "  (no perfetto traces written)"
}

echo "Profile output directory: $PROFILE_DIR"
echo "Total configs: ${#CONFIGS[@]}"

for cfg in "${CONFIGS[@]}"; do
    read -r batch input_seq decode mtp_spec <<< "$cfg"
    run_profile "$batch" "$input_seq" "$decode" "$mtp_spec"
done

echo ""
echo "================================================================"
echo "All profiles saved to $PROFILE_DIR"
find "$PROFILE_DIR" -maxdepth 2 -name "*.perfetto-trace" -printf "%p  %s bytes\n" 2>/dev/null
echo "================================================================"
echo ""
echo "To analyze: open https://ui.perfetto.dev and load the .perfetto-trace files."
