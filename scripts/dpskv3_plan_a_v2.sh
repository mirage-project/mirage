#!/bin/bash
# Plan A v2: 14 workloads × MTP on/off = 28 (MPK + reference) pairs.
# Runs sequentially. Reports prefill / decode TPOT + token alignment.
#
# Usage:
#   bash scripts/dpskv3_plan_a_v2.sh [--quick] [--workloads A1,A4,A11]
#
# --quick: only A1, A4, A11, A14 (one short, one medium, one boundary, super-long)
# --workloads: comma-separated subset
# --no-mtp: skip MTP=2 variants
# --no-mpk / --no-ref: skip one side (for debugging)
#
# Each workload's output goes to outputs/wlcompare_<tag>_<mtp>_<ts>/.

set -uo pipefail

QUICK=""
WL_FILTER=""
NO_MTP=""
EXTRA=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --quick) QUICK=1; shift;;
        --no-mtp) NO_MTP=1; shift;;
        --workloads) WL_FILTER="$2"; shift 2;;
        --no-mpk) EXTRA+=" --skip-mpk"; shift;;
        --no-ref) EXTRA+=" --skip-ref"; shift;;
        *) echo "Unknown arg: $1" >&2; exit 1;;
    esac
done

# Workload table:    tag prompt decode mtp_setting (used regardless if MTP loop runs)
declare -A WL_PROMPT WL_DECODE
WLS=(A1 A2 A3 A4 A5 A6 A7 A8 A9 A10 A11 A12 A13 A14)
WL_PROMPT[A1]=100;  WL_DECODE[A1]=32     # single page baseline
WL_PROMPT[A2]=200;  WL_DECODE[A2]=32     # crosses 1 page boundary
WL_PROMPT[A3]=256;  WL_DECODE[A3]=32     # exactly 2 pages
WL_PROMPT[A4]=384;  WL_DECODE[A4]=32     # 3 page boundaries
WL_PROMPT[A5]=500;  WL_DECODE[A5]=32     # 4 pages
WL_PROMPT[A6]=768;  WL_DECODE[A6]=32     # 6 pages
WL_PROMPT[A7]=1024; WL_DECODE[A7]=32     # 8 pages, long prefill
WL_PROMPT[A8]=2048; WL_DECODE[A8]=32     # 16 pages
WL_PROMPT[A9]=64;   WL_DECODE[A9]=128    # decode crosses page (prefill < 1 page)
WL_PROMPT[A10]=100; WL_DECODE[A10]=200   # decode spans many pages
WL_PROMPT[A11]=127; WL_DECODE[A11]=32    # prefill fills page 0, decode goes into page 1
WL_PROMPT[A12]=129; WL_DECODE[A12]=32    # prefill just past page boundary
WL_PROMPT[A13]=256; WL_DECODE[A13]=256   # long prefill + long decode
WL_PROMPT[A14]=16384; WL_DECODE[A14]=256 # super-long (the user's perf-check target)

if [[ "$QUICK" == 1 ]]; then
    WLS=(A1 A4 A11 A14)
fi
if [[ -n "$WL_FILTER" ]]; then
    IFS=',' read -ra WLS <<< "$WL_FILTER"
fi

MTPS=(0)
if [[ -z "$NO_MTP" ]]; then
    MTPS=(0 2)
fi

OUT_ROOT="${OUT_ROOT:-outputs/plan_a_v2_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT_ROOT"
SUMMARY="$OUT_ROOT/all_summary.txt"
: > "$SUMMARY"

echo "Plan A v2 sweep starting at $(date)" | tee -a "$SUMMARY"
echo "Workloads: ${WLS[*]}" | tee -a "$SUMMARY"
echo "MTP modes: ${MTPS[*]}" | tee -a "$SUMMARY"
echo "Output root: $OUT_ROOT" | tee -a "$SUMMARY"
echo "" | tee -a "$SUMMARY"

START_ALL=$(date +%s)

for tag in "${WLS[@]}"; do
    P=${WL_PROMPT[$tag]}
    D=${WL_DECODE[$tag]}
    for mtp in "${MTPS[@]}"; do
        echo "" | tee -a "$SUMMARY"
        echo "============== $tag mtp=$mtp prompt=$P decode=$D ==============" | tee -a "$SUMMARY"
        out_sub="$OUT_ROOT/${tag}_mtp${mtp}"
        OUT_BASE="$out_sub" bash /home/muhengl/mirage/scripts/dpskv3_workload_compare.sh \
            --tag "$tag" --prompt-len "$P" --decode "$D" --mtp "$mtp" \
            $EXTRA \
            2>&1 | tee -a "$SUMMARY"
    done
done

ELAPSED=$(( $(date +%s) - START_ALL ))
echo "" | tee -a "$SUMMARY"
echo "Plan A v2 sweep DONE in ${ELAPSED}s ($(( ELAPSED / 60 )) min)" | tee -a "$SUMMARY"
echo "Summary: $SUMMARY" | tee -a "$SUMMARY"
echo "" | tee -a "$SUMMARY"
echo "Token alignment summary:" | tee -a "$SUMMARY"
grep -E "TOKENS_(PASS|FAIL)" "$SUMMARY" | tee -a "$SUMMARY"
