#!/usr/bin/env bash
# M4-I7: the dense-fp8 STAGE WALLSPAN, before vs after, and the width residual.
#
# WHY THIS AND NOT JUST THE e2e A/B. M3-I7's re-derived table put dense fp8 at
# 2.07x slower than vLLM as a STAGE, while the ferret kernel now measures at
# PARITY standalone (min_ratio 1.011). If the integrated e2e win is much smaller
# than that, the difference has to live somewhere, and the only two candidates
# are (i) the stage is a small share of the step, and (ii) the stage does not get
# the machine's full width. Separating them is what tells M4 where the rest of
# the gap is -- M4-I5 owns width, so this number is the handoff.
#
# concurrency.py reports, per task type, over ONE steady-window iteration:
#   total_us                 sum of per-task durations (the stage's work)
#   wall_span_us             union of those intervals (the stage's WALLSPAN --
#                            what the step actually pays)
#   mean_concurrency_during  how many of the 128 worker CTAs were inside a task
#
# The width residual for the stage is then
#   wall_span_us - total_us / NW
# i.e. what the stage costs beyond a perfectly-wide execution of the same work.
#
# The profiler perturbs timing, so these are DIAGNOSTIC attributions; the
# performance claim itself comes from the --no-profiler A/B in sweep_fp8.sh.
set -uo pipefail
export PATH=/usr/local/cuda-12.8/bin:$PATH
export HF_HOME=$HOME/mpk-qwen35/hf
export PYTHONUNBUFFERED=1
GPU="${CUDA_VISIBLE_DEVICES:?the guard must pin CUDA_VISIBLE_DEVICES}"

T=${T:-$HOME/mpk-qwen35/mirage-m4i7}
ACC=$T/demo/qwen3_5/accept
OPT=$ACC/opt
export MPK_ACCEPT_DIR=$ACC
export PYTHONPATH=$T/python
PY=$HOME/mpk-qwen35/venv-rm/bin/python
M=${M:-/var/tmp/m4i7_prof}
BSLIST="${BSLIST:-1 16}"
ARMS="${ARMS:-A B}"
MSL=353
NEWTOK=96
SEED_BASE=20260730
mkdir -p "$M/logs" "$M/stage"

echo "########## M4-I7 stage wallspan  gpu=$GPU  $(date -Is) ##########"
echo "tree: $T  HEAD=$(git -C "$T" rev-parse --short HEAD)"
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

for BS in $BSLIST; do
  for ARM in $ARMS; do
    TAG=${ARM}_bs${BS}
    OD=$M/prof_${ARM}
    mkdir -p "$OD"
    if [ ! -f "$OD/raw_bs${BS}_rep0.npz" ]; then
      KDIR=$M/kernel_${ARM}_bs${BS}
      SEED=$((SEED_BASE + BS*1000))
      RK=""; [ -f "$KDIR/task_graph_rank0.json" ] && RK="--reuse-kernel"
      drain
      ENVA=""; [ "$ARM" = A ] && ENVA="MPK_MOE_BLOCKSCALE_BASELINE=1"
      echo "--- profiled run $TAG (${ENVA:-default}) $(date -Is) ---"
      env $ENVA timeout 5400 "$PY" -u "$OPT/profile_wave.py" \
          --batch-size "$BS" --max-seq-length "$MSL" --max-new-tokens "$NEWTOK" \
          --mbt 16 --page-size 256 --synthetic-prompt-len 256 \
          --synthetic-seed "$SEED" --out-dir "$OD" --kernel-dir "$KDIR" \
          --rep 0 --save-raw $RK > "$M/logs/${TAG}.log" 2>&1
      echo "  rc=$? $(grep -h 'wall=' "$M/logs/${TAG}.log" | tail -1)"
    else
      echo "--- $TAG raw cached ---"
    fi
    RAW=$OD/raw_bs${BS}_rep0.npz
    META=$OD/meta_bs${BS}_rep0.json
    NAMES=$OD/task_names.json
    if [ -f "$RAW" ] && [ -f "$META" ] && [ -f "$NAMES" ]; then
      ( cd "$OPT" && "$PY" -u concurrency.py "$RAW" "$META" "$NAMES" \
          "$M/stage/conc_${TAG}.json" ) > "$M/logs/conc_${TAG}.log" 2>&1
      echo "  concurrency rc=$?  -> $M/stage/conc_${TAG}.json"
    else
      echo "  MISSING raw/meta/names for $TAG:"; ls -la "$OD" | head -8
    fi
  done
done
echo "STAGE_WALLSPAN_DONE $(date -Is)"
df -BG /var/tmp | tail -1
