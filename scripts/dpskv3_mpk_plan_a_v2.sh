#!/bin/bash
# Run MPK side of Plan A v2 sweep — one demo invocation per workload.
#
# Reads tests/dpskv3_reference/plan_a_v2.json and runs the DeepSeek V3
# demo with --use-mirage for each entry, saving tokens to
# <out_root>/<tag>_mtp<N>/tokens.json so it lines up with the
# reference batched runner's directory layout.
#
# IMPORTANT: each MPK run does its own nvcc compile + weight load.
# Total time is roughly ~5-6 min/workload × 26 workloads ≈ 2.5h.
#
# Usage:
#   bash scripts/dpskv3_mpk_plan_a_v2.sh \
#       --out-dir outputs/dpskv3_mpk_plan_a_v2_<ts> \
#       --gpus 1,3,4,5
#
# Optional:
#   --workloads A1,A4,A11    subset filter (matches "tag" field)
#   --layers 0-19            override layer range
#   --tp 4 --ep 2            parallelism config

set -uo pipefail

WL_JSON=/home/muhengl/mirage/tests/dpskv3_reference/plan_a_v2.json
WL_FILTER=""
LAYERS=0-19
TP=4
EP=2
GPUS="1,3,4,5"
OUT_ROOT=""
MODEL_PATH="${MODEL_PATH:-/raid/catalyst/models/DeepSeek-V3}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --workloads) WL_FILTER="$2"; shift 2;;
        --layers) LAYERS="$2"; shift 2;;
        --tp) TP="$2"; shift 2;;
        --ep) EP="$2"; shift 2;;
        --gpus) GPUS="$2"; shift 2;;
        --out-dir) OUT_ROOT="$2"; shift 2;;
        --plan) WL_JSON="$2"; shift 2;;
        *) echo "Unknown arg: $1" >&2; exit 1;;
    esac
done

if [[ -z "$OUT_ROOT" ]]; then
    OUT_ROOT="outputs/dpskv3_mpk_plan_a_v2_$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p "$OUT_ROOT"
SUMMARY="$OUT_ROOT/summary.txt"
: > "$SUMMARY"

REPO=/home/muhengl/mirage
VENV=/raid/user_data/muhengl/.venv

# Parse plan_a_v2.json into bash-friendly tab-separated rows.
TMP_PLAN=$(mktemp)
python3 - "$WL_JSON" "$WL_FILTER" > "$TMP_PLAN" <<'PYEOF'
import json, sys
path, wl_filter = sys.argv[1], sys.argv[2]
with open(path) as f: data = json.load(f)
allow = set(wl_filter.split(',')) if wl_filter else None
for w in data:
    if allow and w['tag'] not in allow:
        continue
    print('\t'.join([
        w['tag'], str(w['prompt_length']), str(w['decode']),
        str(w.get('mtp', 0)), str(w.get('max_num_batched_tokens', 128)),
    ]))
PYEOF

echo "[mpk-sweep] Plan: $WL_JSON" | tee -a "$SUMMARY"
echo "[mpk-sweep] Out: $OUT_ROOT" | tee -a "$SUMMARY"
echo "[mpk-sweep] GPUs: $GPUS  TP=$TP  EP=$EP  layers=$LAYERS" | tee -a "$SUMMARY"
echo "[mpk-sweep] Workloads:" | tee -a "$SUMMARY"
cat "$TMP_PLAN" | tee -a "$SUMMARY"
echo "" | tee -a "$SUMMARY"

START_ALL=$(date +%s)

while IFS=$'\t' read -r TAG PROMPT_LEN DECODE MTP MBT; do
    SUB="$OUT_ROOT/${TAG}_mtp${MTP}"
    mkdir -p "$SUB"
    LOG="$SUB/run.log"
    TOKENS="$SUB/tokens.json"

    SEQ=$(( PROMPT_LEN + DECODE + 32 ))
    PAGES=$(( ((SEQ + 127) / 128) + 4 ))

    # MPK requires max_seq_length >= prompt + decode. Some workloads have
    # MBT > prompt; demo requires max_num_batched_tokens >= 1.
    echo "[mpk-sweep] === $TAG mtp=$MTP prompt=$PROMPT_LEN decode=$DECODE seq=$SEQ pages=$PAGES ===" | tee -a "$SUMMARY"

    MTP_ARG=""
    if (( MTP > 0 )); then
        MTP_ARG="--mtp $MTP"
    fi

    START=$(date +%s)
    set +e
    CUDA_VISIBLE_DEVICES="$GPUS" \
    LD_LIBRARY_PATH=/home/muhengl/local/nvshmem-3.6.5-dev/usr/lib/x86_64-linux-gnu/nvshmem/13:/usr/mpi/gcc/openmpi-4.1.9a1/lib \
    LD_PRELOAD=/home/muhengl/local/nvshmem-3.6.5-extract/usr/lib/x86_64-linux-gnu/nvshmem/13/libnvshmem_host.so.3.6.5 \
    NVSHMEM_SYMMETRIC_SIZE=4294967296 \
    PATH=/usr/local/cuda-13.2/bin:/usr/mpi/gcc/openmpi-4.1.9a1/bin:$PATH \
    MPI_INC_PATH=/usr/mpi/gcc/openmpi-4.1.9a1/include \
    MPI_LIB_PATH=/usr/mpi/gcc/openmpi-4.1.9a1/lib \
    NVSHMEM_INC_PATH=/home/muhengl/local/nvshmem-3.6.5-dev/usr/include/nvshmem_13 \
    NVSHMEM_LIB_PATH=/home/muhengl/local/nvshmem-3.6.5-dev/usr/lib/x86_64-linux-gnu/nvshmem/13 \
    timeout 1800 mpirun --allow-run-as-root -np "$TP" \
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
    set -e
    EL=$(( $(date +%s) - START ))
    if (( RC == 0 )) && [[ -f "$TOKENS" ]]; then
        echo "[mpk-sweep] $TAG mtp=$MTP DONE in ${EL}s" | tee -a "$SUMMARY"
        grep -E "per-token latency" "$LOG" | tail -1 | tee -a "$SUMMARY"
    else
        echo "[mpk-sweep] $TAG mtp=$MTP FAIL rc=$RC after ${EL}s" | tee -a "$SUMMARY"
        tail -15 "$LOG" | tee -a "$SUMMARY"
    fi
done < "$TMP_PLAN"

ELAPSED=$(( $(date +%s) - START_ALL ))
echo "" | tee -a "$SUMMARY"
echo "[mpk-sweep] DONE in ${ELAPSED}s ($(( ELAPSED / 60 )) min)" | tee -a "$SUMMARY"
rm -f "$TMP_PLAN"
