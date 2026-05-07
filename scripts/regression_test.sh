#!/bin/bash
# scripts/regression_test.sh
#
# Pre-commit regression suite for DeepSeek V3 + Qwen3.
#
# 4 workloads, MTP on/off for DeepSeek (Qwen3 has no MTP):
#   A: DeepSeek prefill perfetto  (TP=2, prompt 200, max_seq 256, mbt 128)
#   B: DeepSeek decode  perfetto  (TP=2, prompt 1, mbt 1, decode 32)
#   C: Qwen3 torch-vs-MPK token compare (existing CI smoke test)
#   D: DeepSeek TP=4 EP=2 layers 0-19 short-decode (correctness probe)
#
# Each DeepSeek workload runs twice: --mtp 0 and --mtp 2 (with --num-speculative-tokens 3).
#
# Usage:
#   bash scripts/regression_test.sh                     # full
#   bash scripts/regression_test.sh --quick             # A + C only (cheapest)
#   bash scripts/regression_test.sh --workload=A,C      # selected
#   bash scripts/regression_test.sh --no-mtp            # skip MTP-on variants
#
# Override env (sensible defaults for this machine):
#   MIRAGE_VENV     venv to use; auto-detects $REPO/.venv (if mirage installed)
#                   then falls back to /raid/user_data/muhengl/.venv.
#   MODEL_PATH      DeepSeek base path (default /raid/catalyst/models/DeepSeek-V3)
#   QWEN_MODEL      Qwen3 hf model id (default Qwen/Qwen3-8B; honored by demo)
#   MIRAGE_REPO     repo root (default /home/muhengl/mirage)
#   MPI_HOME        OpenMPI prefix (default /usr/mpi/gcc/openmpi-4.1.9a1)
#   NVSHMEM_HOME    NVSHMEM 3.6.5 prefix (default /home/muhengl/local/nvshmem-3.6.5-dev/usr)
#   NVSHMEM_PRELOAD libnvshmem_host.so to LD_PRELOAD
#   OUT_DIR         output dir (default outputs/regression_<timestamp>)
#   GPUS            comma-separated GPU indices (auto-polls otherwise)
#   EXCLUDE_GPUS    comma-separated indices to never use (default 4 — flaky on this host)
#   QUICK_DECODE    decode tokens for workload B in --quick mode (default 8)
#   FULL_DECODE     decode tokens for workload B in full mode  (default 32)
#
# Output structure: $OUT_DIR/
#   summary.txt              one line per workload+MTP-mode
#   {tag}.log                full stdout/stderr per run
#   {tag}_rank{N}.perfetto-trace  (where applicable)
#
# This script:
#   - polls for clean GPUs (mem<500, util<5%) before each launch
#   - times each workload, parses per-token latency from demo output
#   - returns non-zero if ANY workload failed/hung
#   - is incremental-friendly: re-running with same OUT_DIR re-uses
#     finished workload logs (skip if log exists and contains a result line)

set -euo pipefail

# ---------- arg parsing ----------
QUICK=0
NO_MTP=0
WORKLOAD_FILTER=""
for arg in "$@"; do
    case "$arg" in
        --quick) QUICK=1;;
        --no-mtp) NO_MTP=1;;
        --workload=*) WORKLOAD_FILTER="${arg#--workload=}";;
        -h|--help)
            sed -n '/^#/p' "$0" | sed 's/^# \?//' | head -50
            exit 0;;
        *) echo "Unknown arg: $arg" >&2; exit 1;;
    esac
done

want() {
    local w=$1
    [[ -z "$WORKLOAD_FILTER" ]] && return 0
    [[ ",$WORKLOAD_FILTER," == *",$w,"* ]]
}

# ---------- defaults ----------
MIRAGE_REPO="${MIRAGE_REPO:-/home/muhengl/mirage}"
MODEL_PATH="${MODEL_PATH:-/raid/catalyst/models/DeepSeek-V3}"
QWEN_MODEL="${QWEN_MODEL:-Qwen/Qwen3-8B}"
MPI_HOME="${MPI_HOME:-/usr/mpi/gcc/openmpi-4.1.9a1}"
NVSHMEM_HOME="${NVSHMEM_HOME:-/home/muhengl/local/nvshmem-3.6.5-dev/usr}"
NVSHMEM_PRELOAD="${NVSHMEM_PRELOAD:-/home/muhengl/local/nvshmem-3.6.5-extract/usr/lib/x86_64-linux-gnu/nvshmem/13/libnvshmem_host.so.3.6.5}"
OUT_DIR="${OUT_DIR:-$MIRAGE_REPO/outputs/regression_$(date +%Y%m%d_%H%M%S)}"
EXCLUDE_GPUS="${EXCLUDE_GPUS:-4}"
QUICK_DECODE="${QUICK_DECODE:-8}"
FULL_DECODE="${FULL_DECODE:-32}"

# ---------- venv auto-detection ----------
# Prefer the new $REPO/.venv if mirage is importable; else fall back.
detect_venv() {
    local primary="${1:-}"
    [[ -n "$primary" ]] && [[ -x "$primary/bin/python" ]] && return 0
    if [[ -x "$MIRAGE_REPO/.venv/bin/python" ]]; then
        if "$MIRAGE_REPO/.venv/bin/python" -c "import mirage" 2>/dev/null; then
            echo "$MIRAGE_REPO/.venv"
            return 0
        fi
    fi
    if [[ -x /raid/user_data/muhengl/.venv/bin/python ]]; then
        if /raid/user_data/muhengl/.venv/bin/python -c "import mirage" 2>/dev/null; then
            echo /raid/user_data/muhengl/.venv
            return 0
        fi
    fi
    return 1
}

if [[ -n "${MIRAGE_VENV:-}" ]]; then
    VENV="$MIRAGE_VENV"
else
    VENV="$(detect_venv)"
fi
if [[ -z "$VENV" ]] || ! [[ -x "$VENV/bin/python" ]]; then
    echo "FATAL: no usable mirage venv found." >&2
    echo "  Tried: \$MIRAGE_VENV, $MIRAGE_REPO/.venv, /raid/user_data/muhengl/.venv" >&2
    exit 2
fi
PY="$VENV/bin/python"
echo "Using venv: $VENV"
echo "Using python: $PY"
"$PY" -c "import torch, transformers, mirage; print('  torch=' + torch.__version__, 'transformers=' + transformers.__version__, 'mirage=' + mirage.__file__)"

mkdir -p "$OUT_DIR"
SUMMARY="$OUT_DIR/summary.txt"
: > "$SUMMARY"
echo "OUT_DIR=$OUT_DIR"
echo "MODEL_PATH=$MODEL_PATH"

# ---------- env exports ----------
export PATH="$MPI_HOME/bin:$PATH"
export MPI_INC_PATH="$MPI_HOME/include"
export MPI_LIB_PATH="$MPI_HOME/lib"
export NVSHMEM_INC_PATH="$NVSHMEM_HOME/include/nvshmem_13"
export NVSHMEM_LIB_PATH="$NVSHMEM_HOME/lib/x86_64-linux-gnu/nvshmem/13"
export LD_LIBRARY_PATH="$NVSHMEM_LIB_PATH:$MPI_HOME/lib:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="$NVSHMEM_PRELOAD"
export NVSHMEM_SYMMETRIC_SIZE=4294967296
# CUDA toolchain for any nvcc the runtime may invoke
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
export PATH="$CUDA_HOME/bin:$PATH"

MPI_ENV_ARGS=(
    -x CUDA_VISIBLE_DEVICES -x LD_LIBRARY_PATH -x LD_PRELOAD -x PATH
    -x MPI_INC_PATH -x MPI_LIB_PATH -x NVSHMEM_INC_PATH -x NVSHMEM_LIB_PATH
    -x NVSHMEM_SYMMETRIC_SIZE -x CUDA_HOME
)

# ---------- GPU polling ----------
gpu_excluded() {
    local idx="$1"
    local excl=" ${EXCLUDE_GPUS//,/ } "
    [[ "$excl" == *" $idx "* ]]
}
find_clean_gpus() {
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
               --format=csv,noheader,nounits 2>/dev/null \
        | awk -F', ' '$2+0 < 500 && $3+0 < 5 { print $1 }'
}
acquire_gpus() {
    local need=$1 max_wait=${2:-1200}
    if [[ -n "${GPUS:-}" ]]; then
        echo "Using user-specified GPUS=$GPUS for need=$need"
        export CUDA_VISIBLE_DEVICES="$GPUS"
        return 0
    fi
    local start=$(date +%s)
    while true; do
        local clean
        mapfile -t clean < <(find_clean_gpus)
        local pick=()
        for idx in "${clean[@]}"; do
            if ! gpu_excluded "$idx"; then pick+=("$idx"); fi
            if (( ${#pick[@]} >= need )); then break; fi
        done
        if (( ${#pick[@]} >= need )); then
            export CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${pick[*]}")"
            echo "Acquired $need GPUs: $CUDA_VISIBLE_DEVICES (waited $(( $(date +%s) - start ))s)"
            return 0
        fi
        local elapsed=$(( $(date +%s) - start ))
        if (( elapsed > max_wait )); then
            echo "FATAL: timed out after ${elapsed}s waiting for $need clean GPUs."
            nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv | head
            return 1
        fi
        sleep 30
    done
}

# ---------- run helper ----------
# run_workload TAG TP MBT MBR PROMPT DECODE LAYERS EP SEQ PAGES MTP EXTRA_ARGS
run_workload() {
    local tag=$1 tp=$2 mbt=$3 mbr=$4 prompt=$5 decode=$6 layers=$7 ep=$8 seq=$9 pages=${10} mtp=${11} extra="${12:-}"
    local log="$OUT_DIR/${tag}.log"
    local mtp_args=""
    if (( mtp > 0 )); then
        mtp_args="--mtp $mtp --num-speculative-tokens 3"
    fi
    if [[ -f "$log" ]] && grep -qE "REGRESSION_RESULT=(PASS|FAIL|HANG)" "$log"; then
        local prev=$(grep -oE "REGRESSION_RESULT=[A-Z]+" "$log" | tail -1)
        echo "[$tag] CACHED $prev (log $log)" | tee -a "$SUMMARY"
        return 0
    fi
    if ! acquire_gpus "$tp"; then
        echo "[$tag] SKIP (no $tp GPUs)" | tee -a "$SUMMARY"
        return 1
    fi
    echo ""
    echo "================================================================"
    echo "[$tag] tp=$tp layers=$layers ep=$ep mbt=$mbt mbr=$mbr prompt=$prompt decode=$decode mtp=$mtp"
    echo "      seq=$seq pages=$pages GPUS=$CUDA_VISIBLE_DEVICES log=$log"
    echo "================================================================"
    local start=$(date +%s)
    local rc=0
    set +e
    timeout 1800 mpirun --allow-run-as-root -np "$tp" "${MPI_ENV_ARGS[@]}" \
        "$PY" "$MIRAGE_REPO/demo/deepseek_v3/demo.py" \
            --model-path "$MODEL_PATH" --use-mirage \
            --layers "$layers" \
            --max-num-batched-tokens "$mbt" \
            --max-num-batched-requests "$mbr" \
            --prompt-length "$prompt" \
            --max-new-tokens "$decode" \
            --max-seq-length "$seq" \
            --max-num-pages "$pages" \
            --page-size 128 \
            --ep-size "$ep" \
            --ignore-eos \
            --trace-name "$OUT_DIR/${tag}_trace" \
            --profiling \
            $mtp_args \
            $extra \
            > "$log" 2>&1
    rc=$?
    set -e
    local elapsed=$(( $(date +%s) - start ))
    local result lat
    lat=$(grep -oE "per-token latency[^:]*: *[0-9.]+ *ms" "$log" | tail -1 | awk '{print $(NF-1)" "$NF}')
    if (( rc == 124 )); then
        result="HANG"
    elif (( rc != 0 )) || [[ -z "$lat" ]]; then
        result="FAIL"
    else
        result="PASS"
    fi
    echo "REGRESSION_RESULT=$result" >> "$log"
    printf '[%s] %-4s %4ds  %s\n' "$tag" "$result" "$elapsed" "${lat:--}" | tee -a "$SUMMARY"
}

run_qwen3() {
    local tag=$1
    local log="$OUT_DIR/${tag}.log"
    if [[ -f "$log" ]] && grep -qE "REGRESSION_RESULT=(PASS|FAIL|HANG)" "$log"; then
        local prev=$(grep -oE "REGRESSION_RESULT=[A-Z]+" "$log" | tail -1)
        echo "[$tag] CACHED $prev (log $log)" | tee -a "$SUMMARY"
        return 0
    fi
    if ! acquire_gpus 1; then
        echo "[$tag] SKIP (no 1 GPU)" | tee -a "$SUMMARY"
        return 1
    fi
    echo ""
    echo "================================================================"
    echo "[$tag] qwen3 torch-vs-MPK comparison (single GPU $CUDA_VISIBLE_DEVICES)"
    echo "      log=$log"
    echo "================================================================"
    local start=$(date +%s)
    local rc=0
    set +e
    (
        cd "$MIRAGE_REPO"
        # The CI script uses bare `python`; force ours via PATH for this subshell.
        PATH="$VENV/bin:$PATH" \
        MIRAGE_HOME="$MIRAGE_REPO" \
        timeout 1200 bash tests/ci-tests/run_ci_tests_qwen3.sh \
            > "$log" 2>&1
    )
    rc=$?
    set -e
    local elapsed=$(( $(date +%s) - start ))
    local result
    if (( rc == 124 )); then result="HANG"
    elif (( rc != 0 )); then result="FAIL"
    else result="PASS"
    fi
    echo "REGRESSION_RESULT=$result" >> "$log"
    printf '[%s] %-4s %4ds\n' "$tag" "$result" "$elapsed" | tee -a "$SUMMARY"
}

# ---------- workload definitions ----------
# pages = ceil(seq / 128) * mbr + 4 slack
calc_pages() {
    local seq=$1 mbr=$2
    echo $(( ((seq + 127) / 128) * mbr + 4 ))
}

declare -a TODO_MTP_MODES
if (( NO_MTP == 1 )); then TODO_MTP_MODES=(0); else TODO_MTP_MODES=(0 2); fi

# Workload A — prefill perfetto
if want A; then
    SEQ_A=256
    PAGES_A=$(calc_pages "$SEQ_A" 1)
    for m in "${TODO_MTP_MODES[@]}"; do
        run_workload "A_prefill_mtp$m"  2  128 1  200  1   0-3   1  "$SEQ_A"  "$PAGES_A"  "$m"  ""  || true
    done
fi

# Workload B — decode perfetto
if want B; then
    DEC_B=$(( QUICK == 1 ? QUICK_DECODE : FULL_DECODE ))
    SEQ_B=$(( 1 + DEC_B + 16 ))
    PAGES_B=$(calc_pages "$SEQ_B" 1)
    for m in "${TODO_MTP_MODES[@]}"; do
        run_workload "B_decode_mtp$m"   2  1   1  1    "$DEC_B"  0-3   1  "$SEQ_B"  "$PAGES_B"  "$m"  ""  || true
    done
fi

# Workload C — qwen3 torch vs MPK
if want C; then
    run_qwen3 "C_qwen3" || true
fi

# Workload D — DeepSeek TP=4 EP=2 layers 0-19 short decode
# Skip in --quick because TP=4 acquisition is slower.
if want D && (( QUICK == 0 )); then
    SEQ_D=512
    PAGES_D=$(calc_pages "$SEQ_D" 1)
    for m in "${TODO_MTP_MODES[@]}"; do
        run_workload "D_tp4_ep2_l20_mtp$m"  4  1   1  1    8     0-19  2  "$SEQ_D"  "$PAGES_D"  "$m"  ""  || true
    done
fi

echo ""
echo "================================================================"
echo "REGRESSION SUMMARY ($OUT_DIR/summary.txt)"
echo "================================================================"
cat "$SUMMARY"

# Non-zero exit if any FAIL/HANG
if grep -qE " (FAIL|HANG) " "$SUMMARY"; then
    exit 1
fi
exit 0
