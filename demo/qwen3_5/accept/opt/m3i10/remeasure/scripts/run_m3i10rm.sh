#!/usr/bin/env bash
# M3-I10 matched-geometry re-measure capture driver (runs ON the B200 box,
# under the GPU guard, in the ISOLATED mirage-rm clone).
#
#   bash gpu_guard_m3i10rm.sh 1,7 -- bash run_m3i10rm.sh <arm> <mode>
#
# arm  = A (matched geometry, msl=1280, synthetic 256-tok prompts, 96 decode
#          steps) | B (continuity, msl=132, AC-3 prompts, 64 new tokens --
#          M3-I1's exact invocation)
# mode = all | prof | noprof
#
# Reuses M3-I1/M3-I8's capture pipeline pattern (profile_wave.py, one wave per
# PROCESS -- HAZARD-WAVE-RESET) but points $OPT at the ISOLATED mirage-rm
# clone (which carries the ONE code change: --synthetic-prompt-len /
# --synthetic-seed on profile_wave.py) instead of the shared m3i1/opt copy, so
# this never touches ~/mpk-qwen35/mirage or any sibling clone. Tree runs at
# HEAD defaults (gate_padding_rows ON) -- no arm-snapshot swapping.
set -uo pipefail

M=$HOME/mpk-qwen35/m3i10-remeasure
MIRAGE=$HOME/mpk-qwen35/mirage-rm
OPT=$MIRAGE/demo/qwen3_5/accept/opt
export PATH=/usr/local/cuda-12.8/bin:$PATH
export HF_HOME=$HOME/mpk-qwen35/hf
export MPK_ACCEPT_DIR=$MIRAGE/demo/qwen3_5/accept
export PYTHONUNBUFFERED=1
PY=$HOME/mpk-qwen35/venv-rm/bin/python
REPS=${REPS:-3}
ARM=${1:?arm (A|B)}
MODE=${2:-all}
SEED_BASE=20260725

mkdir -p "$M/prof_$ARM" "$M/noprof_$ARM" "$M/logs"

AVAIL=$(df -BG --output=avail /raid | tail -1 | tr -dc '0-9')
echo "df /raid avail=${AVAIL}G"
if [ "${AVAIL:-0}" -lt 10 ]; then
  echo "REFUSING: /raid headroom ${AVAIL}G < 10G" >&2; exit 96
fi

echo "=== arm=$ARM mirage-rm HEAD ==="
git -C "$MIRAGE" rev-parse HEAD
git -C "$MIRAGE" status --short | head -5
grep -n "^MOE_GATE_PADDING_ROWS" "$MIRAGE/python/mirage/mpk/models/qwen3_5/builder.py" \
  | tee "$M/logs/arm_${ARM}_sha256.txt"
sha256sum "$OPT/profile_wave.py" "$MIRAGE/python/mirage/mpk/models/qwen3_5/builder.py" \
  | tee -a "$M/logs/arm_${ARM}_sha256.txt"

# M3-I8's ascending-length AC-3 prompt sets (arm B / continuity only).
ids_for_bs() {
  case "$1" in
    1)  echo "p06-poem" ;;
    2)  echo "p06-poem,p01-history" ;;
    4)  echo "p06-poem,p01-history,p04-chinese,p09-translate" ;;
    8)  echo "p06-poem,p01-history,p04-chinese,p09-translate,p07-format,p05-cuda,p08-science,p10-logic" ;;
    16) echo "p06-poem,p01-history,p04-chinese,p09-translate,p07-format,p05-cuda,p08-science,p10-logic,p03-python,p02-math" ;;
    *)  echo "BAD" ;;
  esac
}

geom_flags() {
  local BS=$1 REP=$2
  if [ "$ARM" = "A" ]; then
    local SEED=$((SEED_BASE + BS*1000 + REP))
    # NOTE (root-caused empirically, see logs/ROOT_CAUSE_msl.txt): MPK's
    # MODE_OFFLINE retires a request on step+step_advance+1 >= max_seq_length
    # (persistent_kernel.cuh) -- max_seq_length is what actually gates decode
    # LENGTH, unlike vLLM where max_model_len is a capacity ceiling separate
    # from --output-len. The spec's literal "--max-seq-length 1280" (vLLM's
    # 256+1024 capacity number) makes MPK decode ~1023 steps, not 96 -- a 16x
    # blowup and a ~10x wave-time miss against the spec's own ~1.1min budget,
    # and INCONSISTENT with the spec's own sec-4 "profiled decode window sits
    # at context 257-352" claim, which only holds for a 96-step decode. Using
    # prompt_len(256) + 96 + 1 = 353 instead reproduces exactly the spec's
    # stated 112/224/352-iteration, context-257-352, ~1.1min design -- 1280
    # was likely a slip conflating vLLM's max_model_len with MPK's same-named
    # but differently-behaving max_seq_length. bs-independent: every arm-A
    # request is uniformly 256 synthetic tokens regardless of batch size.
    echo "--max-seq-length 353 --max-new-tokens 96 --mbt 16 --page-size 256 --synthetic-prompt-len 256 --synthetic-seed $SEED --slots 96000000"
  else
    echo "--max-seq-length 132 --max-new-tokens 64 --mbt 16 --page-size 256 --prompt-ids $(ids_for_bs "$BS") --slots 48000000"
  fi
}

run_profiled() {
  local BS=$1 REP=$2
  local KDIR=$M/kernel_${ARM}_bs${BS}_prof
  local RK=""; [ -f "$KDIR/task_graph_rank0.json" ] && RK="--reuse-kernel"
  local RAW=""; [ "$REP" -eq 0 ] && RAW="--save-raw"
  echo "===== [$ARM] PROFILED bs=$BS rep=$REP ${RK:-(compile)} $(date -Is) ====="
  # shellcheck disable=SC2046
  $PY -u "$OPT/profile_wave.py" --batch-size "$BS" \
      $(geom_flags "$BS" "$REP") \
      --out-dir "$M/prof_$ARM" --kernel-dir "$KDIR" --rep "$REP" $RK $RAW \
      > "$M/logs/${ARM}_prof_bs${BS}_rep${REP}.log" 2>&1
  echo "rc=$? $(grep -h 'profiler:\|wall=' "$M/logs/${ARM}_prof_bs${BS}_rep${REP}.log" | tail -2)"
}

run_noprof() {
  local BS=$1 REP=$2
  local KDIR=$M/kernel_${ARM}_bs${BS}_noprof
  local RK=""; [ -f "$KDIR/task_graph_rank0.json" ] && RK="--reuse-kernel"
  echo "===== [$ARM] UNPROFILED bs=$BS rep=$REP ${RK:-(compile)} $(date -Is) ====="
  # shellcheck disable=SC2046
  $PY -u "$OPT/profile_wave.py" --batch-size "$BS" \
      $(geom_flags "$BS" "$REP") \
      --out-dir "$M/noprof_$ARM" --kernel-dir "$KDIR" --rep "$REP" --no-profiler $RK \
      > "$M/logs/${ARM}_noprof_bs${BS}_rep${REP}.log" 2>&1
  echo "rc=$? $(grep -h 'wall=' "$M/logs/${ARM}_noprof_bs${BS}_rep${REP}.log" | tail -1)"
}

for BS in ${BSLIST:-1 8 16}; do
  case "$MODE" in
    all|prof) for R in $(seq 0 $((REPS - 1))); do run_profiled "$BS" "$R"; done ;;
  esac
  case "$MODE" in
    all|noprof) for R in $(seq 0 $((REPS - 1))); do run_noprof "$BS" "$R"; done ;;
  esac
done
echo "M3I10RM_DRIVER_EXIT=0 arm=$ARM mode=$MODE $(date -Is)"
