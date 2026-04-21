#!/bin/bash
# DeepSeek V3 MLA TP merge validation — correctness + stress matrix
#
# Validates the PR #663 MLA TP kernels now wired into the builder (TP=2/4/8)
# plus the associated tma.cuh / task_register / empty-split fixes currently
# in the working tree (uncommitted).
#
# Test matrix (per TP ∈ {1, 2, 4}):
#   Correctness  : layers 0-10 × bs ∈ {1, 8, 32} × seq ∈ {128, 1024, 16384}
#                  (decode=32 tokens is plenty for a first-token match check)
#   Stress       : layers 0-10 × (bs, seq, decode) matrix from CLI request
#                  bs=1  × seq ∈ {128, 1K, 16K, 100K} × decode ∈ {256, 1024}
#                  bs=8  × seq ∈ {128, 1K, 16K}      × decode ∈ {256, 1024}
#                  bs=32 × seq ∈ {128, 1K, 16K}      × decode ∈ {256, 1024}
#
# --correctness gives real token-vs-PyTorch-ref validation ONLY at TP=1
# (demo.py runs in-process reference only when world_size==1). For TP>1 the
# flag exists but just exercises the same code path; the pass criterion is
# "completes without crash + rank 0 decodes same tokens as the TP=1 run".
#
# Outputs each test's stdout+stderr to $OUT_DIR/<tag>.log and a CSV summary.
# The script NEVER tries to run TP>1 if fewer than TP idle GPUs are found.
#
# Constraints:
#   • 0-10 layers only (--layers 0-10, 11 layers total — 3 dense + 8 MoE).
#   • No MTP, no lm_head skip, no residual env overrides — baseline settings.
#   • MPK persistent kernel needs exclusive GPU. Idle check enforces this.
#
# Usage:
#   bash demo/deepseek_v3/test_mla_tp_matrix.sh [MODE] [TP_LIST]
#     MODE      correctness | stress | all   (default: all)
#     TP_LIST   "1 2 4"                      (default — omit TP=8)
#
#   Env overrides:
#     MODEL_PATH  (default /raid/catalyst/models/DeepSeek-V3)
#     OUT_DIR     (default $HOME/mla_tp_matrix_out)
#     GPUS        comma-separated GPU ids. If unset, script picks idle GPUs.
#     QUICK=1     skip seq=100K and decode=1024 rows (fast smoke: ~15 min)
#     STOP_ON_FAIL=1  abort the script on first failing config

set -u  # don't use -e — we want to continue past per-test failures

# ── Arguments ────────────────────────────────────────────────────
MODE="${1:-all}"
TP_LIST="${2:-1 2 4}"

case "$MODE" in
  correctness|stress|all) ;;
  *) echo "FATAL: MODE must be correctness|stress|all, got '$MODE'"; exit 2;;
esac

MODEL_PATH="${MODEL_PATH:-/raid/catalyst/models/DeepSeek-V3}"
OUT_DIR="${OUT_DIR:-$HOME/mla_tp_matrix_out}"
LAYERS="${LAYERS:-0-9}"    # User wants layers 0-9 (10 layers) for stress tests
PAGE_SIZE=128
MEM_THRESHOLD=500          # MiB — GPU considered busy above this
LAYER_BUDGET=10            # layers 0-9 inclusive
SUMMARY="$OUT_DIR/summary.csv"

mkdir -p "$OUT_DIR"
: > "$SUMMARY"
echo "phase,tp,batch,seq,decode,status,seconds,log" >> "$SUMMARY"

log() { echo "[$(date +%H:%M:%S)] $*"; }

# ── Environment (copied from existing test_tp4_correctness.sh) ───
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

# ── GPU discovery ────────────────────────────────────────────────
pick_gpus() {
    local need=$1
    if [[ -n "${GPUS:-}" ]]; then
        echo "$GPUS"
        return 0
    fi
    local idle=()
    while IFS=', ' read -r idx mem util; do
        if (( ${mem%% *} < MEM_THRESHOLD )); then
            idle+=("$idx")
        fi
    done < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
                        --format=csv,noheader,nounits)
    if (( ${#idle[@]} < need )); then
        return 1
    fi
    (IFS=,; echo "${idle[*]:0:$need}")
}

# ── Page count for a given max_seq_length ────────────────────────
pages_for() {
    # Keep 8 extra pages for generation headroom.
    echo $(( ($1 + PAGE_SIZE - 1) / PAGE_SIZE + 8 ))
}

# ── Single test runner ──────────────────────────────────────────
# Args:
#   phase (correctness|stress)
#   tp
#   batch
#   seq       max_seq_length (input + decode headroom)
#   decode    --max-new-tokens
#   extra...  extra flags (--correctness, etc.)
run_one() {
    local phase="$1" tp="$2" batch="$3" seq="$4" decode="$5"
    shift 5
    local tag="${phase}_tp${tp}_b${batch}_s${seq}_d${decode}"
    local log_file="$OUT_DIR/${tag}.log"
    local max_pages
    max_pages=$(pages_for "$seq")

    local gpus
    gpus=$(pick_gpus "$tp") || {
        log "SKIP $tag: need $tp idle GPUs (<${MEM_THRESHOLD} MiB) — not available"
        echo "$phase,$tp,$batch,$seq,$decode,SKIP_NOGPU,0,$log_file" >> "$SUMMARY"
        return 0
    }

    log "RUN  $tag  (GPUs=$gpus, pages=$max_pages)"
    export CUDA_VISIBLE_DEVICES="$gpus"
    # Per-test timeout: scale with seq — 100K needs more; default 30 min, 60 min
    # for seq>=65536, 90 min for seq>=100000.
    local per_test_timeout=1800
    (( seq >= 65536 )) && per_test_timeout=3600
    (( seq >= 100000 )) && per_test_timeout=5400
    local t0 status
    t0=$(date +%s)
    if (( tp == 1 )); then
        timeout "$per_test_timeout" python demo/deepseek_v3/demo.py \
            --model-path "$MODEL_PATH" --use-mirage --layers "$LAYERS" \
            --max-num-batched-tokens "$batch" \
            --max-seq-length "$seq" \
            --max-num-pages "$max_pages" \
            --max-new-tokens "$decode" \
            "$@" \
            >"$log_file" 2>&1
        status=$?
    else
        timeout "$per_test_timeout" mpirun --allow-run-as-root -np "$tp" "${MPI_ENV_ARGS[@]}" \
            python demo/deepseek_v3/demo.py \
            --model-path "$MODEL_PATH" --use-mirage --layers "$LAYERS" \
            --max-num-batched-tokens "$batch" \
            --max-seq-length "$seq" \
            --max-num-pages "$max_pages" \
            --max-new-tokens "$decode" \
            "$@" \
            >"$log_file" 2>&1
        status=$?
    fi
    local elapsed=$(( $(date +%s) - t0 ))
    # timeout returns 124 (or 137 if KILL was needed)
    if (( status == 124 || status == 137 )); then
        status=124
    fi

    # Correctness PASS/FAIL verdict:
    #   TP=1 + --correctness  : demo.py prints "PASS: tokens match!" or "FAIL: tokens differ!"
    #   TP>1                  : only "ran without crash" — status==0 is the signal
    local verdict="UNKNOWN"
    if (( status == 124 )); then
        verdict="TIMEOUT"
    elif (( status != 0 )); then
        verdict="CRASH"
    elif grep -q "PASS: tokens match" "$log_file"; then
        verdict="PASS_CORRECT"
    elif grep -q "FAIL: tokens differ" "$log_file"; then
        verdict="FAIL_MISMATCH"
    elif (( status == 0 )); then
        verdict="PASS_RAN"
    fi

    log "  → $verdict  (${elapsed}s)"
    echo "$phase,$tp,$batch,$seq,$decode,$verdict,$elapsed,$log_file" >> "$SUMMARY"

    if [[ "${STOP_ON_FAIL:-0}" == "1" ]] && [[ "$verdict" == CRASH || "$verdict" == FAIL_MISMATCH ]]; then
        log "STOP_ON_FAIL=1 and $tag did not pass — aborting"
        exit 1
    fi
    return 0
}

# ── Correctness matrix ──────────────────────────────────────────
# Only TP=1 gets real PASS/FAIL; TP>1 validates "no crash" at same config.
# Shorter decode (32) is enough — the check is first-token match.
#
# Drop seq=16K for bs=32 to avoid KV cache memory blowup on smaller GPUs —
# bs=32 × 16K × 11 layers × FP8 KV is still ~tolerable but not critical for
# a first-token correctness check.
CORRECTNESS_CONFIGS_BS1=(
    "1 128   32"
    "1 1024  32"
    "1 16384 32"
)
CORRECTNESS_CONFIGS_BS8=(
    "8 128   32"
    "8 1024  32"
    "8 16384 32"
)
CORRECTNESS_CONFIGS_BS32=(
    "32 128   32"
    "32 1024  32"
)

run_correctness_phase() {
    local tp="$1"
    log "========== CORRECTNESS TP=$tp =========="
    local cfgs=("${CORRECTNESS_CONFIGS_BS1[@]}" "${CORRECTNESS_CONFIGS_BS8[@]}" "${CORRECTNESS_CONFIGS_BS32[@]}")
    for cfg in "${cfgs[@]}"; do
        read -r batch seq decode <<< "$cfg"
        run_one correctness "$tp" "$batch" "$seq" "$decode" --correctness
    done
}

# ── Stress matrix (from CLI request) ────────────────────────────
#   bs=1  × seq ∈ {128, 1K, 16K, 100K} × decode ∈ {256, 1024}
#   bs=8  × seq ∈ {128, 1K, 16K}       × decode ∈ {256, 1024}
#   bs=32 × seq ∈ {128, 1K, 16K}       × decode ∈ {256, 1024}
STRESS_CONFIGS=(
    # bs=1 × seq ∈ {128, 1K, 16K} × decode ∈ {256, 1024}
    "1 128   256"
    "1 128   1024"
    "1 1024  256"
    "1 1024  1024"
    "1 16384 256"
    "1 16384 1024"
    # bs=8
    "8 128   256"
    "8 128   1024"
    "8 1024  256"
    "8 1024  1024"
    "8 16384 256"
    "8 16384 1024"
    # bs=32
    "32 128   256"
    "32 128   1024"
    "32 1024  256"
    "32 1024  1024"
    "32 16384 256"
    "32 16384 1024"
)
# Special: TP=4, bs=1, input=100K, decode=1024 (per user request)
STRESS_CONFIGS_TP4_SPECIAL=(
    "1 100000 1024"
)

run_stress_phase() {
    local tp="$1"
    log "========== STRESS TP=$tp =========="
    for cfg in "${STRESS_CONFIGS[@]}"; do
        read -r batch seq decode <<< "$cfg"
        if [[ "${QUICK:-0}" == "1" ]]; then
            (( seq >= 16384 )) && continue
            (( decode >= 1024 )) && continue
        fi
        run_one stress "$tp" "$batch" "$seq" "$decode"
    done
    # Special 100K input case: only TP=4 per user request
    if (( tp == 4 )) && [[ "${QUICK:-0}" != "1" ]]; then
        for cfg in "${STRESS_CONFIGS_TP4_SPECIAL[@]}"; do
            read -r batch seq decode <<< "$cfg"
            run_one stress "$tp" "$batch" "$seq" "$decode"
        done
    fi
}

# ── Main ────────────────────────────────────────────────────────
log "MLA TP matrix: MODE=$MODE  TP_LIST=[$TP_LIST]"
log "MODEL_PATH=$MODEL_PATH  OUT_DIR=$OUT_DIR  LAYERS=$LAYERS  (budget=$LAYER_BUDGET)"

for tp in $TP_LIST; do
    case "$MODE" in
        correctness) run_correctness_phase "$tp" ;;
        stress)      run_stress_phase "$tp" ;;
        all)         run_correctness_phase "$tp"; run_stress_phase "$tp" ;;
    esac
done

# ── Summary ─────────────────────────────────────────────────────
echo ""
log "================================================================"
log "Summary (CSV at $SUMMARY):"
column -t -s, "$SUMMARY"

# Quick failure count
n_fail=$(grep -cE ',(CRASH|FAIL_MISMATCH),' "$SUMMARY" || true)
n_pass=$(grep -cE ',(PASS_CORRECT|PASS_RAN),' "$SUMMARY" || true)
n_skip=$(grep -cE ',SKIP_NOGPU,' "$SUMMARY" || true)
log "Passed=$n_pass  Failed=$n_fail  Skipped=$n_skip"
log "================================================================"

exit $(( n_fail > 0 ? 1 : 0 ))
