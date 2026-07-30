#!/usr/bin/env bash
# M4-I9: profiled LATE-CONTEXT capture of ARM F (MPK_FUSE_SILU_QUANT=1) at
# M4-I8's OWN geometry and windows, so cp_exact can be re-derived on the same
# basis and the acceptance question -- DID THE CHAIN ACTUALLY GET SHORTER? -- is
# answered by a difference of two like-for-like numbers rather than by a model.
#
# An e2e win with an UNCHANGED chain would mean something other than the chain
# moved and the attribution is wrong. Predicted (fuse_model.py on M4-I8's arm-A
# buffers): cp_exact 4130.7 -> 3953.4 / 5275.7 -> 5086.7 / 5638.7 -> 5454.2 us at
# bs 1/8/16, i.e. -177.3 / -189.0 / -184.4, with 40 chain records removed.
#
# The raws are KEPT until the derivation is done (0.9-1.5 GB per cell); they are
# regenerable and are not committed. --slots 200000000 is M3-I7's late value.
set -uo pipefail
export PATH=/usr/local/cuda-12.8/bin:$PATH
export HF_HOME=$HOME/mpk-qwen35/hf
export PYTHONUNBUFFERED=1
GPU="${CUDA_VISIBLE_DEVICES:?the guard must pin CUDA_VISIBLE_DEVICES}"

T=${T:-$HOME/mpk-qwen35/mirage-m4i9}
ARM=${ARM:-F}
case "$ARM" in
  A) ARM_ENV="" ;;
  F) ARM_ENV="MPK_FUSE_SILU_QUANT=1" ;;
  N) ARM_ENV="MPK_FUSE_NORM_QUANT=1" ;;
  G) ARM_ENV="MPK_FUSE_RECUR_QUANT=1" ;;
  S) ARM_ENV="MPK_FUSE_SILU_QUANT=1 MPK_FUSE_NORM_QUANT=1 MPK_FUSE_RECUR_QUANT=1" ;;
  *) echo "unknown ARM=$ARM"; exit 2 ;;
esac
PY=$HOME/mpk-qwen35/venv-rm/bin/python
M=${M:-/var/tmp/m4i9_prof_$ARM}
OPT=$T/demo/qwen3_5/accept/opt
SLOTS="${SLOTS:-200000000}"
MSL=897
NEWTOK=640
CELLS="${CELLS:-1:288,384 8:365,461 16:720,733}"
REP=0
SEED_BASE=20260730
mkdir -p "$M/logs" "$M/stage" "$M/prof"

echo "########## M4-I9 profiled capture arm=$ARM ($ARM_ENV)  gpu=$GPU  $(date -Is) ##########"
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
    env $ARM_ENV MPK_ACCEPT_DIR="$T/demo/qwen3_5/accept" PYTHONPATH="$T/python" \
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
  df -BG /var/tmp | tail -1
done
echo "M4I9_PROF_DONE $(date -Is)"
