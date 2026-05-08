#!/bin/bash
# Plan A v2 end-to-end driver:
#   (assume reference batched sweep already ran)
#   1. Run MPK side over plan_a_v2.json
#   2. Compare each workload's tokens.json vs the reference output
#   3. Emit summary.md / summary.json
#
# Usage:
#   bash scripts/dpskv3_plan_a_v2_pipeline.sh \
#       --ref-dir outputs/dpskv3_ref_plan_a_v2_<ts> \
#       [--mpk-dir outputs/dpskv3_mpk_plan_a_v2_<ts>] \
#       [--gpus 1,3,4,5] [--workloads A1,A4,A11]
#
# If --mpk-dir is omitted, a fresh outputs/dpskv3_mpk_plan_a_v2_<ts>
# is created. If you re-run with --mpk-dir <existing>, individual
# workloads that already have tokens.json are NOT skipped — the demo
# overwrites them.

set -uo pipefail

REF_DIR=""
MPK_DIR=""
GPUS="1,3,4,5"
WL_FILTER=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ref-dir) REF_DIR="$2"; shift 2;;
        --mpk-dir) MPK_DIR="$2"; shift 2;;
        --gpus) GPUS="$2"; shift 2;;
        --workloads) WL_FILTER="$2"; shift 2;;
        *) echo "Unknown arg: $1" >&2; exit 1;;
    esac
done

if [[ -z "$REF_DIR" ]]; then
    echo "Required: --ref-dir <reference batched sweep output>" >&2
    exit 1
fi
if [[ ! -d "$REF_DIR" ]]; then
    echo "Reference dir not found: $REF_DIR" >&2
    exit 1
fi

if [[ -z "$MPK_DIR" ]]; then
    MPK_DIR="outputs/dpskv3_mpk_plan_a_v2_$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p "$MPK_DIR"

CMP_DIR="outputs/plan_a_v2_compare_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$CMP_DIR"

echo "[pipeline] ref=$REF_DIR"
echo "[pipeline] mpk=$MPK_DIR"
echo "[pipeline] cmp=$CMP_DIR"

WL_FLAG=()
if [[ -n "$WL_FILTER" ]]; then
    WL_FLAG=(--workloads "$WL_FILTER")
fi

echo "[pipeline] Step 1: MPK sweep"
bash /home/muhengl/mirage/scripts/dpskv3_mpk_plan_a_v2.sh \
    --out-dir "$MPK_DIR" --gpus "$GPUS" \
    "${WL_FLAG[@]}" \
    | tee -a "$CMP_DIR/pipeline.log"

echo "[pipeline] Step 2: Comparator"
/raid/user_data/muhengl/.venv/bin/python \
    /home/muhengl/mirage/scripts/dpskv3_compare_plan_a_v2.py \
    --ref "$REF_DIR" --mpk "$MPK_DIR" --out "$CMP_DIR" \
    | tee -a "$CMP_DIR/pipeline.log"

echo "[pipeline] DONE"
echo "  Summary markdown: $CMP_DIR/summary.md"
echo "  Summary JSON:     $CMP_DIR/summary.json"
