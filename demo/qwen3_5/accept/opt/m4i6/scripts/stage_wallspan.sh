#!/usr/bin/env bash
# M4-I6: the router STAGE WALLSPAN and in-MPK PER-TASK TIME, before vs after.
#
# WHY THIS AND NOT JUST THE e2e A/B. M4-I5's critical-path decomposition put
# MOE_TOPK_SOFTMAX_SM100 at 842.1 us of the 7957.5 us bs1 path (10.58%), at
# T = 21.053 us/task with live/lvl = 1.0 -- the MOST SERIALIZED stage in the
# graph, one task per layer with nothing to overlap it against -- and said it must
# reach 3.697 us/task at bs1 to hold up its end of the five-stage parity scenario.
# The e2e A/B says whether the step got faster; only this says whether the STAGE
# moved to where the model needs it, and how much of the 842 us came back.
#
# concurrency.py reports, per task type, over ONE steady-window iteration:
#   total_us                 sum of per-task durations (the stage's work)
#   wall_span_us             union of those intervals (the stage's WALLSPAN)
#   mean_concurrency_during  how many of the 128 worker CTAs were inside a task
#
# M4-I2's lesson on reading these: standalone gains translate imperfectly. There
# the integrated work went UP 1.24x while concurrency rose, so the wallspan still
# improved 1.9x. For THIS stage that mechanism cannot apply -- one task per layer
# per step means concurrency is pinned at 1.0 and there is no packing to gain, so
# wallspan can only follow per-task time. That makes the comparison unusually
# direct, and it is also why a per-task regression here would be unrecoverable.
#
# The profiler perturbs timing, so these are DIAGNOSTIC attributions; the
# performance claim itself comes from the --no-profiler A/B in sweep_router.sh.
set -uo pipefail
export PATH=/usr/local/cuda-12.8/bin:$PATH
export HF_HOME=$HOME/mpk-qwen35/hf
export PYTHONUNBUFFERED=1
GPU="${CUDA_VISIBLE_DEVICES:?the guard must pin CUDA_VISIBLE_DEVICES}"

TA=${TA:-$HOME/mpk-qwen35/mirage-m4i6-base}
TB=${TB:-$HOME/mpk-qwen35/mirage-m4i6}
PY=$HOME/mpk-qwen35/venv-rm/bin/python
M=${M:-/var/tmp/m4i6_prof}
BSLIST="${BSLIST:-1 8 16}"
ARMS="${ARMS:-A B}"
# REP exists so the non-router stages' apparent drift can be RE-MEASURED. The
# first pass (rep 0) showed the router's path contribution falling 445.3 us while
# the whole path fell only 354.4 us, with the 90.9 us difference sitting in the
# OTHER stages' per-task times (W13 +53.6, ATTN +14.9, W2 +14.4 us on the path).
# Two mechanisms fit: single-rep profiled variance, or a real shared-budget tax
# from the 238->255 register + 4-byte spill gate 2 measured. A second independent
# rep of both arms tells them apart -- variance will not reproduce the same sign
# pattern, a budget tax will.
REP="${REP:-0}"
MSL=353
NEWTOK=96
SEED_BASE=20260730
mkdir -p "$M/logs" "$M/stage"

echo "########## M4-I6 stage wallspan  gpu=$GPU  $(date -Is) ##########"
echo "arm A: $TA ($(git -C "$TA" rev-parse --short HEAD))"
echo "arm B: $TB ($(git -C "$TB" rev-parse --short HEAD))"
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
    TAG=${ARM}_bs${BS}_rep${REP}
    T="$TB"; [ "$ARM" = A ] && T="$TA"
    OPT=$T/demo/qwen3_5/accept/opt
    OD=$M/prof_${ARM}
    mkdir -p "$OD"
    if [ ! -f "$OD/raw_bs${BS}_rep${REP}.npz" ]; then
      # KERNEL DIR PER (arm, bs) AND PER PROFILED/UNPROFILED LANE: the profiled
      # build carries -DMPK_ENABLE_PROFILING, which the reuse cache does not
      # record, so sharing a dir with sweep_router.sh would reload the wrong
      # binary and print "compatibility check passed" (add-mpk-task).
      KDIR=$M/kernel_prof_${ARM}_bs${BS}
      SEED=$((SEED_BASE + BS*1000 + REP))
      RK=""; [ -f "$KDIR/task_graph_rank0.json" ] && RK="--reuse-kernel"
      drain
      echo "--- profiled run $TAG (tree $(basename "$T")) $(date -Is) ---"
      MPK_ACCEPT_DIR="$T/demo/qwen3_5/accept" PYTHONPATH="$T/python" \
      timeout 5400 "$PY" -u "$OPT/profile_wave.py" \
          --batch-size "$BS" --max-seq-length "$MSL" --max-new-tokens "$NEWTOK" \
          --mbt 16 --page-size 256 --synthetic-prompt-len 256 \
          --synthetic-seed "$SEED" --out-dir "$OD" --kernel-dir "$KDIR" \
          --rep "$REP" --save-raw $RK > "$M/logs/${TAG}.log" 2>&1
      echo "  rc=$? $(grep -h 'wall=' "$M/logs/${TAG}.log" | tail -1)"
    else
      echo "--- $TAG raw cached ---"
    fi
    RAW=$OD/raw_bs${BS}_rep${REP}.npz
    META=$OD/meta_bs${BS}_rep${REP}.json
    NAMES=$OD/task_names.json
    if [ -f "$RAW" ] && [ -f "$META" ] && [ -f "$NAMES" ]; then
      ( cd "$OPT" && MPK_ACCEPT_DIR="$T/demo/qwen3_5/accept" PYTHONPATH="$T/python" \
        "$PY" -u concurrency.py "$RAW" "$META" "$NAMES" \
          "$M/stage/conc_${TAG}.json" ) > "$M/logs/conc_${TAG}.log" 2>&1
      echo "  concurrency rc=$?  -> $M/stage/conc_${TAG}.json"
    else
      echo "  MISSING raw/meta/names for $TAG:"; ls -la "$OD" | head -8
    fi
  done
done
echo "STAGE_WALLSPAN_DONE $(date -Is)"
df -BG /var/tmp | tail -1
