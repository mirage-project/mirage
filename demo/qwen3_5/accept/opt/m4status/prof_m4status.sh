#!/usr/bin/env bash
# M4-status: profiled wave capture on the COMBINED tree, so M4-I5's
# cp_decompose.py can be re-run against a current path.
#
# Basis choice, and why it differs per batch size:
#   bs1  -- msl=353 / 96 decode steps.  This is M4-I6's precedent
#           (opt/m4i6/scripts/stage_wallspan.sh + critpath_m4i6.sh); its arm-A
#           re-derivation landed within 0.7% of M4-I5's 7957.5 us bs1 path, so
#           the cheap geometry is licensed at bs1.
#   bs8/16 -- msl=897 / 640 decode steps.  M3-I7's basis note is explicit that at
#           msl=353 the admission replay has NO prefill-free regime wider than
#           five live requests at bs8/bs16, so that geometry cannot express a
#           bs8 or bs16 decode step at all.  M4-I5's own basis is msl=897.
#
# Single arm: HEAD (the gate's own freshly built clone).  No A/B here -- the A/B
# baselines live in the m4i5/m4i6 artefacts already.
set -uo pipefail
export PATH=/usr/local/cuda-12.8/bin:$PATH
export HF_HOME=$HOME/mpk-qwen35/hf
export PYTHONUNBUFFERED=1
GPU="${CUDA_VISIBLE_DEVICES:?the guard must pin CUDA_VISIBLE_DEVICES}"

T=${T:-$HOME/mpk-qwen35/final-gate/tree-6741b4ad8aae}
PY=$HOME/mpk-qwen35/venv-rm/bin/python
M=${M:-/var/tmp/m4status_prof}
OPT=$T/demo/qwen3_5/accept/opt
# "bs:msl:newtok" triples
CELLS="${CELLS:-1:353:96}"
REP="${REP:-0}"
SEED_BASE=20260730
mkdir -p "$M/logs" "$M/stage" "$M/prof_head"

echo "########## M4-status profiled capture  gpu=$GPU  $(date -Is) ##########"
echo "tree: $T ($(git -C "$T" rev-parse --short HEAD))"
md5sum "$T"/python/mirage/core.cpython-*.so
df -BG /var/tmp | tail -1

used_on_pinned () {
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F',' -v g="$GPU" '{gsub(/ /,"",$1)} $1+0==g+0 {gsub(/ /,"",$2); print $2+0}'
}
drain () {   # our own previous rep must be gone before the next claim
  local i used
  for i in $(seq 1 90); do
    used=$(used_on_pinned); [ "${used:-9999}" -lt 500 ] && return 0; sleep 5
  done
  echo "    ABORT: device $GPU held ${used}MiB after 450s"; exit 97
}

for CELL in $CELLS; do
  BS="${CELL%%:*}"; REST="${CELL#*:}"; MSL="${REST%%:*}"; NEWTOK="${REST##*:}"
  TAG=head_bs${BS}_msl${MSL}_rep${REP}
  OD=$M/prof_head
  if [ ! -f "$OD/raw_bs${BS}_rep${REP}.npz" ]; then
    KDIR=$M/kernel_prof_bs${BS}_msl${MSL}
    SEED=$((SEED_BASE + BS*1000 + REP))
    RK=""; [ -f "$KDIR/task_graph_rank0.json" ] && RK="--reuse-kernel"
    drain
    BEFORE=$(used_on_pinned)
    echo "--- profiled run $TAG msl=$MSL newtok=$NEWTOK gpu_before=${BEFORE}MiB $(date -Is) ---"
    MPK_ACCEPT_DIR="$T/demo/qwen3_5/accept" PYTHONPATH="$T/python" \
    timeout 7200 "$PY" -u "$OPT/profile_wave.py" \
        --batch-size "$BS" --max-seq-length "$MSL" --max-new-tokens "$NEWTOK" \
        --mbt 16 --page-size 256 --synthetic-prompt-len 256 \
        --synthetic-seed "$SEED" --out-dir "$OD" --kernel-dir "$KDIR" \
        --rep "$REP" --save-raw $RK > "$M/logs/${TAG}.log" 2>&1
    echo "  rc=$? gpu_after=$(used_on_pinned)MiB $(grep -h 'wall=' "$M/logs/${TAG}.log" | tail -1)"
    tail -4 "$M/logs/${TAG}.log" | sed 's/^/    /'
  else
    echo "--- $TAG raw cached ---"
  fi
  RAW=$OD/raw_bs${BS}_rep${REP}.npz
  META=$OD/meta_bs${BS}_rep${REP}.json
  NAMES=$OD/task_names.json
  KDIR=$M/kernel_prof_bs${BS}_msl${MSL}
  GRAPH=$KDIR/task_graph_rank0.json
  for f in "$RAW" "$META" "$NAMES" "$GRAPH"; do
    [ -f "$f" ] || { echo "  $TAG: MISSING $f -- cp skipped"; continue 2; }
  done
  ( cd "$OPT" && MPK_ACCEPT_DIR="$T/demo/qwen3_5/accept" PYTHONPATH="$T/python" \
    "$PY" -u concurrency.py "$RAW" "$META" "$NAMES" "$M/stage/conc_${TAG}.json" ) \
    > "$M/logs/conc_${TAG}.log" 2>&1
  echo "  concurrency rc=$?"
  "$PY" -u "$OPT/m4i5/scripts/width.py" "$RAW" "$META" "$NAMES" --graph "$GRAPH" \
      --out "$M/stage/width_${TAG}.json" > "$M/logs/width_${TAG}.log" 2>&1
  echo "  width rc=$?"
  "$PY" -u "$OPT/m4i5/scripts/cp_decompose.py" "$GRAPH" "$M/stage/width_${TAG}.json" \
      --names "$NAMES" --ferret "$OPT/m3i10/ferret_targets.json" --weight levelmax \
      --out "$M/stage/cp_${TAG}.json" > "$M/logs/cp_${TAG}.log" 2>&1
  echo "  cp_decompose rc=$?"
  tail -6 "$M/logs/cp_${TAG}.log" | sed 's/^/    /'
done
echo "M4STATUS_PROF_DONE $(date -Is)"
df -BG /var/tmp | tail -1
