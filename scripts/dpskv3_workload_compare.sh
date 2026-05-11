#!/bin/bash
# Run a single Plan A v2 workload on both MPK and the PyTorch reference,
# then diff the saved token IDs.
#
# Usage:
#   bash scripts/dpskv3_workload_compare.sh \
#       --tag A1 --prompt-len 100 --decode 32 --mtp 0 \
#       --layers 0-19 --tp 4 --ep 2
#
# Defaults: tp=4, ep=2, layers=0-19, model_path=/raid/catalyst/models/DeepSeek-V3
#
# Output: outputs/wlcompare_<tag>_<ts>/
#   mpk/      — demo's --save-tokens output
#   ref/      — reference dump
#   compare.json — per-workload diff report
#   summary.txt  — one-line PASS/FAIL + latencies
#
# Note: requires both MPK and reference to be runnable on the same 4 GPUs.
# Reference uses torchrun. MPK uses mpirun. They CANNOT run simultaneously
# on the same GPUs — this script runs them sequentially.

set -euo pipefail

# ---- defaults ----
TAG=""
PROMPT_LEN=100
DECODE=32
MTP=0
LAYERS_MAIN="0-19"
TP=4
EP=2
MBT=128
# Max concurrent requests in a single MPK iteration. Default 1; >1 stresses
# the gather/chunked-prefill path across distinct request slots.
MBR=1
# FP8-faithful reference: route the PyTorch reference's FP8-eligible
# linears through a quantize-then-matmul simulation so numerics match
# MPK's hardware FP8 GEMM. Otherwise reference uses BF16-dequant weights
# and diverges by FP8 activation-quantization noise.
FP8_FAITHFUL=0
MODEL_PATH="${MODEL_PATH:-/raid/catalyst/models/DeepSeek-V3}"
PROMPT_TEXT="${PROMPT_TEXT:-Give me a short introduction to large language model.}"
GPUS="${GPUS:-0,1,3,5}"

# ---- arg parse ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --tag) TAG="$2"; shift 2;;
        --prompt-len) PROMPT_LEN="$2"; shift 2;;
        --decode) DECODE="$2"; shift 2;;
        --mtp) MTP="$2"; shift 2;;
        --layers) LAYERS_MAIN="$2"; shift 2;;
        --tp) TP="$2"; shift 2;;
        --ep) EP="$2"; shift 2;;
        --mbt) MBT="$2"; shift 2;;
        --mbr) MBR="$2"; shift 2;;
        --gpus) GPUS="$2"; shift 2;;
        --fp8-faithful) FP8_FAITHFUL=1; shift;;
        --skip-mpk) SKIP_MPK=1; shift;;
        --skip-ref) SKIP_REF=1; shift;;
        *) echo "Unknown arg: $1" >&2; exit 1;;
    esac
done

if [[ -z "$TAG" ]]; then
    echo "Required: --tag" >&2; exit 1
fi

TS=$(date +%Y%m%d_%H%M%S)
OUT_BASE="${OUT_BASE:-outputs/wlcompare_${TAG}_mtp${MTP}_${TS}}"
mkdir -p "$OUT_BASE/mpk" "$OUT_BASE/ref"
SUMMARY="$OUT_BASE/summary.txt"
COMPARE="$OUT_BASE/compare.json"

REPO=/home/muhengl/mirage
VENV=/home/muhengl/mirage/.venv

# Page calc (page_size=128, max_seq = prompt_len + decode + slack)
SEQ=$(( PROMPT_LEN + DECODE + 32 ))
PAGES=$(( ((SEQ + 127) / 128) + 4 ))

echo "============================================================" | tee -a "$SUMMARY"
echo "[$TAG] prompt=$PROMPT_LEN decode=$DECODE mtp=$MTP tp=$TP ep=$EP layers=$LAYERS_MAIN" | tee -a "$SUMMARY"
echo "  seq=$SEQ pages=$PAGES mbt=$MBT mbr=$MBR" | tee -a "$SUMMARY"
echo "  gpus=$GPUS  out=$OUT_BASE" | tee -a "$SUMMARY"
echo "============================================================" | tee -a "$SUMMARY"

# =========================================================================
# Phase 1: MPK side
# =========================================================================
if [[ -z "${SKIP_MPK:-}" ]]; then
    echo "[$TAG] MPK side (TP=$TP) ..." | tee -a "$SUMMARY"
    MTP_ARG=""
    if (( MTP > 0 )); then
        MTP_ARG="--mtp $MTP"
    fi
    MPK_LOG="$OUT_BASE/mpk/run.log"
    MPK_TOKENS="$OUT_BASE/mpk/tokens.json"
    START_MPK=$(date +%s)
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
    timeout 1800 mpirun --allow-run-as-root -np $TP \
        -x CUDA_VISIBLE_DEVICES -x LD_LIBRARY_PATH -x LD_PRELOAD -x PATH \
        -x MPI_INC_PATH -x MPI_LIB_PATH -x NVSHMEM_INC_PATH -x NVSHMEM_LIB_PATH \
        -x NVSHMEM_SYMMETRIC_SIZE \
        "$VENV/bin/python" "$REPO/demo/deepseek_v3/demo.py" \
            --model-path "$MODEL_PATH" --use-mirage \
            --layers "$LAYERS_MAIN" \
            --max-num-batched-tokens "$MBT" \
            --max-num-batched-requests "$MBR" \
            --prompt-length "$PROMPT_LEN" \
            --max-new-tokens "$DECODE" \
            --max-seq-length "$SEQ" \
            --max-num-pages "$PAGES" \
            --page-size 128 \
            --ep-size "$EP" \
            --ignore-eos \
            --save-tokens "$MPK_TOKENS" \
            $MTP_ARG \
            > "$MPK_LOG" 2>&1
    MPK_RC=$?
    set -e
    MPK_EL=$(( $(date +%s) - START_MPK ))
    if (( MPK_RC == 0 )); then
        echo "[$TAG] MPK PASS in ${MPK_EL}s" | tee -a "$SUMMARY"
        grep -E "per-token latency" "$MPK_LOG" | tail -1 | tee -a "$SUMMARY"
    else
        echo "[$TAG] MPK FAIL rc=$MPK_RC after ${MPK_EL}s" | tee -a "$SUMMARY"
        tail -10 "$MPK_LOG" | tee -a "$SUMMARY"
    fi
else
    echo "[$TAG] MPK side: SKIPPED" | tee -a "$SUMMARY"
fi

# =========================================================================
# Phase 2: Reference side
# =========================================================================
if [[ -z "${SKIP_REF:-}" ]]; then
    echo "[$TAG] Reference side (TP=$TP EP=$EP) ..." | tee -a "$SUMMARY"
    MTP_ARG=""
    if (( MTP > 0 )); then
        MTP_ARG="--enable-mtp --spec-length $MTP"
    fi
    REF_LOG="$OUT_BASE/ref/run.log"
    START_REF=$(date +%s)

    # Get the same prompt MPK used (first PROMPT_LEN tokens of full prompt).
    # Simplest: pass the same prompt text; demo.py and reference both
    # tokenise via AutoTokenizer with `trust_remote_code=True`. Reference
    # encodes the full text and uses the first PROMPT_LEN tokens.
    # Note: MPK demo uses fixed --prompt-length to truncate/pad; reference
    # uses the actual tokenized length. For EXACT alignment we'd need to
    # share the token IDs, but for now both encode the same text so the
    # token IDs match for short prompts; for long prompts this assumption
    # may break.

    FP8_ARG=""
    if (( FP8_FAITHFUL == 1 )); then
        FP8_ARG="--fp8-faithful"
    fi
    set +e
    CUDA_VISIBLE_DEVICES="$GPUS" \
    PATH="$VENV/bin:/usr/local/cuda-13.2/bin:/usr/mpi/gcc/openmpi-4.1.9a1/bin:$PATH" \
    timeout 3600 "$VENV/bin/torchrun" \
        --nproc_per_node="$TP" \
        --master_addr=127.0.0.1 \
        --master_port=29504 \
        "$REPO/tests/dpskv3_reference/runner_distributed.py" \
            --model-path "$MODEL_PATH" \
            --layers "$LAYERS_MAIN" \
            --tp-size "$TP" --ep-size "$EP" \
            --prompt-length "$PROMPT_LEN" \
            --max-new-tokens "$DECODE" \
            --max-num-batched-tokens "$MBT" \
            --dump-dir "$OUT_BASE/ref" \
            $MTP_ARG $FP8_ARG \
            > "$REF_LOG" 2>&1
    REF_RC=$?
    set -e
    REF_EL=$(( $(date +%s) - START_REF ))
    if (( REF_RC == 0 )); then
        echo "[$TAG] REF PASS in ${REF_EL}s" | tee -a "$SUMMARY"
        grep -E "PREFILL_MS|DECODE_TPOT_MS" "$REF_LOG" | tee -a "$SUMMARY"
    else
        echo "[$TAG] REF FAIL rc=$REF_RC after ${REF_EL}s" | tee -a "$SUMMARY"
        tail -15 "$REF_LOG" | tee -a "$SUMMARY"
    fi
else
    echo "[$TAG] REF side: SKIPPED" | tee -a "$SUMMARY"
fi

# =========================================================================
# Phase 3: Compare
# =========================================================================
echo "[$TAG] Comparing tokens..." | tee -a "$SUMMARY"
if [[ -f "$OUT_BASE/mpk/tokens.json" ]] && [[ -f "$OUT_BASE/ref/tokens.json" ]]; then
    "$VENV/bin/python" -m tests.dpskv3_reference.comparator \
        --reference "$OUT_BASE/ref" \
        --mpk "$OUT_BASE/mpk" \
        --out "$COMPARE" 2>&1 | tee -a "$SUMMARY"

    # Quick pass/fail summary line. Use Python interpolation {status}, not
    # ${} bash expansion, so `set -u` doesn't trip on the local Python var.
    "$VENV/bin/python" -c "
import json, sys
with open('$COMPARE') as f: r = json.load(f)
match = r.get('tokens_match')
nc = r.get('tokens_n_compared', 0)
fm = r.get('tokens_first_mismatch')
ref_h = r.get('tokens_ref_head', [])
mpk_h = r.get('tokens_mpk_head', [])
status = 'PASS' if match else 'FAIL'
print(f'[$TAG] TOKENS_{status}  n={nc}  first_mismatch={fm}')
if not match:
    print(f'  ref head: {ref_h}')
    print(f'  mpk head: {mpk_h}')
" | tee -a "$SUMMARY"
else
    echo "[$TAG] tokens.json missing on one side; cannot compare." | tee -a "$SUMMARY"
    [[ ! -f "$OUT_BASE/mpk/tokens.json" ]] && echo "  missing: $OUT_BASE/mpk/tokens.json" | tee -a "$SUMMARY"
    [[ ! -f "$OUT_BASE/ref/tokens.json" ]] && echo "  missing: $OUT_BASE/ref/tokens.json" | tee -a "$SUMMARY"
fi

echo "[$TAG] DONE  out=$OUT_BASE" | tee -a "$SUMMARY"
