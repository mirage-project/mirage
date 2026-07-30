#!/usr/bin/env bash
# M4-status: profiled LATE-CONTEXT capture (msl=897) on the combined tree, at
# M4-I5's OWN windows, so cp_decompose.py's output is directly comparable to its
# published 7957.5 / 8240.9 / 8642.0 us paths.
#
# WHY msl=897 AND WHY THESE WINDOWS.  M3-I7's basis note is explicit: at msl=353
# the exact admission replay has no prefill-free regime wider than five live
# requests at bs8 or bs16, so that geometry cannot express a bs8 or bs16 decode
# step at all.  M4-I5's basis is msl=897 = 256-token synthetic prompt + 640
# decode steps, mbt=16, page 256, and its windows are m3i7/window_plan.py's:
# bs1 [288,384), bs8 [365,461), bs16 [720,733).  Passing those exact windows to
# width.py removes the geometry/window difference from the comparison; the
# msl=353 bs1 capture (committed separately) is the same-geometry control
# against M4-I6's two arms.
#
# --slots 200000000 is M3-I7's late-capture value (gate_i7.sh PHASE late): at
# 640 decode steps the default 48M slots overflows and PROFILER_CAN_WRITE
# silently truncates the tail, which is the defect M4-I5 root-caused for bs16.
set -uo pipefail
export PATH=/usr/local/cuda-12.8/bin:$PATH
export HF_HOME=$HOME/mpk-qwen35/hf
export PYTHONUNBUFFERED=1
GPU="${CUDA_VISIBLE_DEVICES:?the guard must pin CUDA_VISIBLE_DEVICES}"

T=${T:-$HOME/mpk-qwen35/final-gate/tree-6741b4ad8aae}
PY=$HOME/mpk-qwen35/venv-rm/bin/python
M=${M:-/var/tmp/m4status_prof_late}
OPT=$T/demo/qwen3_5/accept/opt
SLOTS="${SLOTS:-200000000}"
MSL=897
NEWTOK=640
# "bs:lo,hi" -- M4-I5's own windows
CELLS="${CELLS:-1:288,384 8:365,461 16:720,733}"
REP=0
SEED_BASE=20260730
mkdir -p "$M/logs" "$M/stage" "$M/prof_head"

echo "########## M4-status LATE profiled capture  gpu=$GPU  $(date -Is) ##########"
echo "tree: $T ($(git -C "$T" rev-parse --short HEAD))  slots=$SLOTS"
md5sum "$T"/python/mirage/core.cpython-*.so
df -BG /var/tmp | tail -1

used_on_pinned () {
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F',' -v g="$GPU" '{gsub(/ /,"",$1)} $1+0==g+0 {gsub(/ /,"",$2); print $2+0}'
}
drain () {
  local i used
  for i in $(seq 1 90); do
    used=$(used_on_pinned); [ "${used:-9999}" -lt 500 ] && return 0; sleep 5
  done
  echo "    ABORT: device $GPU held ${used}MiB after 450s"; exit 97
}

for CELL in $CELLS; do
  BS="${CELL%%:*}"; WIN="${CELL#*:}"
  TAG=head_late_bs${BS}
  OD=$M/prof_head
  KDIR=$M/kernel_late_bs${BS}
  if [ ! -f "$OD/raw_bs${BS}_rep${REP}.npz" ]; then
    SEED=$((SEED_BASE + BS*1000 + REP))
    RK=""; [ -f "$KDIR/task_graph_rank0.json" ] && RK="--reuse-kernel"
    drain
    BEFORE=$(used_on_pinned)
    echo "--- $TAG msl=$MSL newtok=$NEWTOK window=$WIN gpu_before=${BEFORE}MiB $(date -Is) ---"
    MPK_ACCEPT_DIR="$T/demo/qwen3_5/accept" PYTHONPATH="$T/python" \
    timeout 9000 "$PY" -u "$OPT/profile_wave.py" \
        --batch-size "$BS" --max-seq-length "$MSL" --max-new-tokens "$NEWTOK" \
        --mbt 16 --page-size 256 --synthetic-prompt-len 256 \
        --synthetic-seed "$SEED" --out-dir "$OD" --kernel-dir "$KDIR" \
        --rep "$REP" --slots "$SLOTS" --save-raw $RK > "$M/logs/${TAG}.log" 2>&1
    echo "  rc=$? gpu_after=$(used_on_pinned)MiB"
    grep -hE "wall=|profiler:" "$M/logs/${TAG}.log" | tail -2 | sed 's/^/    /'
  else
    echo "--- $TAG raw cached ---"
  fi
  RAW=$OD/raw_bs${BS}_rep${REP}.npz
  META=$OD/meta_bs${BS}_rep${REP}.json
  NAMES=$OD/task_names.json
  GRAPH=$KDIR/task_graph_rank0.json
  for f in "$RAW" "$META" "$NAMES" "$GRAPH"; do
    [ -f "$f" ] || { echo "  $TAG: MISSING $f -- cp skipped"; continue 2; }
  done
  ls -la --time-style=+%F_%T "$RAW" | sed 's/^/    /'
  "$PY" -u "$OPT/m4i5/scripts/width.py" "$RAW" "$META" "$NAMES" --graph "$GRAPH" \
      --window "$WIN" --out "$M/stage/width_${TAG}.json" \
      > "$M/logs/width_${TAG}.log" 2>&1
  echo "  width rc=$? (window $WIN)"
  "$PY" -u "$OPT/m4i5/scripts/cp_decompose.py" "$GRAPH" "$M/stage/width_${TAG}.json" \
      --names "$NAMES" --ferret "$OPT/m3i10/ferret_targets.json" --weight levelmax \
      --out "$M/stage/cp_${TAG}.json" > "$M/logs/cp_${TAG}.log" 2>&1
  echo "  cp_decompose rc=$?"
  grep -aE "critical path|cp_max|step" "$M/logs/cp_${TAG}.log" | head -4 | sed 's/^/    /'
  # the raws are 0.7-1.5 GB each and /var/tmp is at 94% -- drop each one as soon
  # as its width+cp are derived (both are committed; the raw is regenerable).
  rm -f "$RAW"
  df -BG /var/tmp | tail -1
done
echo "M4STATUS_LATE_PROF_DONE $(date -Is)"
