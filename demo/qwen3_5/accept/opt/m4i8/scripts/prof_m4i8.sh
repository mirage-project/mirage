#!/usr/bin/env bash
# M4-I8: profiled LATE-CONTEXT capture (msl=897) on MY OWN clone at HEAD
# (5756c789), at M4-I5's / M4-status's OWN windows, so width.py + cp_decompose.py
# reproduce the published 4112.0 / 4802.3 / 5482.1 us paths and the
# 5791.8 / 8350.6 / 10308.7 us steps as a same-basis control.
#
# DIFFERENCE FROM prof_m4status_late.sh: the raws are KEPT. This issue needs the
# per-task begin/end stream itself (sched_gap.py), not just the width/cp digests,
# so the 0.9-1.5 GB npz per cell stays on /var/tmp until the derivation is done.
# --slots 200000000 is M3-I7's late value; m4status measured fill 37.9/48.7/61.5%
# at 640 steps with anchor_qc PASS at all three, so it is sized for this geometry.
set -uo pipefail
export PATH=/usr/local/cuda-12.8/bin:$PATH
export HF_HOME=$HOME/mpk-qwen35/hf
export PYTHONUNBUFFERED=1
GPU="${CUDA_VISIBLE_DEVICES:?the guard must pin CUDA_VISIBLE_DEVICES}"

T=${T:-$HOME/mpk-qwen35/mirage-m4i8}
PY=$HOME/mpk-qwen35/venv-rm/bin/python
M=${M:-/var/tmp/m4i8_prof}
OPT=$T/demo/qwen3_5/accept/opt
SLOTS="${SLOTS:-200000000}"
MSL=897
NEWTOK=640
CELLS="${CELLS:-1:288,384 8:365,461 16:720,733}"
REP=0
SEED_BASE=20260730
mkdir -p "$M/logs" "$M/stage" "$M/prof"

echo "########## M4-I8 profiled capture  gpu=$GPU  $(date -Is) ##########"
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
  TAG=bs${BS}
  OD=$M/prof
  KDIR=$M/kernel_bs${BS}
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
    [ -f "$f" ] || { echo "  $TAG: MISSING $f -- derivation skipped"; continue 2; }
  done
  ls -la --time-style=+%F_%T "$RAW" | sed 's/^/    /'
  "$PY" -u "$OPT/m4i5/scripts/width.py" "$RAW" "$META" "$NAMES" --graph "$GRAPH" \
      --window "$WIN" --out "$M/stage/width_${TAG}.json" \
      > "$M/logs/width_${TAG}.log" 2>&1
  echo "  width rc=$? (window $WIN)"
  "$PY" -c "
import json;d=json.load(open('$M/stage/width_${TAG}.json'))
print('    step_us=%.1f n_it=%d anchor=%s worst_rel_err=%s exact_prefix=%s dropped=%s/%s workbound=%.1f'%(
 d['step_us'],d['n_iterations'],d['anchor_qc']['verdict'],d['anchor_qc']['worst_rel_err'],
 d['anchor_qc'].get('exact_prefix_iterations'),d['dropped_begin'],d['dropped_end'],d['machine']['work_bound_us']))"
  "$PY" -u "$OPT/m4i5/scripts/cp_decompose.py" "$GRAPH" "$M/stage/width_${TAG}.json" \
      --names "$NAMES" --ferret "$OPT/m3i10/ferret_targets.json" --weight levelmax \
      --out "$M/stage/cp_${TAG}.json" > "$M/logs/cp_${TAG}.log" 2>&1
  echo "  cp_decompose rc=$?"
  grep -aE "cp_max|measured step" "$M/logs/cp_${TAG}.log" | head -3 | sed 's/^/    /'
  df -BG /var/tmp | tail -1
done
echo "M4I8_PROF_DONE $(date -Is)"
