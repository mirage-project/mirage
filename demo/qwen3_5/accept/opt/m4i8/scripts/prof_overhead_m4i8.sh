#!/usr/bin/env bash
# M4-I8: how much of the measured dispatch-latency terms is the PROFILER itself?
#
# WHY THIS CONTROL IS OWED. sched_gap.py's data_gap and resource_gap are measured
# BETWEEN profiler timestamps: resource_gap is END(task_i) -> BEGIN(task_i+1) on
# one worker, and each of those markers costs a %globaltimer read plus a store.
# On Blackwell a %globaltimer read is hundreds of ns, so part of every gap is the
# instrument, not the runtime -- and that part does NOT exist in the shipped
# kernel. M4-I5 reported the aggregate ratio (profiled steps x1.112-1.123 vs the
# unprofiled wall) but at a DIFFERENT geometry, so it is quoted, not assumed.
#
# This runs the SAME geometry as prof_m4i8.sh (msl=897, 640 decode steps,
# mbt=16, same seeds) with --no-profiler, so ms_per_decode_step is directly
# comparable to the profiled window's step_us and the inflation is measured
# rather than modelled. A kernel dir per (mode, bs): --slots is baked into the
# compiled kernel, so a profiled run cannot share a dir with an unprofiled one.
set -uo pipefail
export PATH=/usr/local/cuda-12.8/bin:$PATH
export HF_HOME=$HOME/mpk-qwen35/hf
export PYTHONUNBUFFERED=1
GPU="${CUDA_VISIBLE_DEVICES:?the guard must pin CUDA_VISIBLE_DEVICES}"
T=${T:-$HOME/mpk-qwen35/mirage-m4i8}
PY=$HOME/mpk-qwen35/venv-rm/bin/python
M=${M:-/var/tmp/m4i8_povh}
OPT=$T/demo/qwen3_5/accept/opt
MSL=897; NEWTOK=640; REPS="${REPS:-0 1 2}"
BSLIST="${BSLIST:-1 8 16}"
SEED_BASE=20260730
ARMS="${ARMS:-A S}"
mkdir -p "$M/logs"
echo "########## M4-I8 profiler-overhead control  gpu=$GPU  $(date -Is) ##########"
echo "tree: $T ($(git -C "$T" rev-parse --short HEAD))  msl=$MSL newtok=$NEWTOK"
used_on_pinned () {
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F',' -v g="$GPU" '{gsub(/ /,"",$1)} $1+0==g+0 {gsub(/ /,"",$2); print $2+0}'
}
drain () { local i used; for i in $(seq 1 90); do used=$(used_on_pinned);
  [ "${used:-9999}" -lt 500 ] && return 0; sleep 5; done
  echo "    ABORT: device $GPU held ${used}MiB after 450s"; exit 97; }
arm_env () { case "$1" in A) echo "";; S) echo "MPK_EVENT_WAIT_GPU_SCOPE=1";;
             O) echo "MPK_WORKER_OOO_POP=1";; esac; }
for BS in $BSLIST; do
  for REP in $REPS; do
    for ARM in $ARMS; do
      TAG=noprof_${ARM}_bs${BS}_rep${REP}
      OD=$M/out_${ARM}; KDIR=$M/kernel_${ARM}_bs${BS}
      [ -f "$OD/meta_bs${BS}_rep${REP}_${ARM}.json" ] && { echo "  [$TAG] cached"; continue; }
      mkdir -p "$OD"
      SEED=$((SEED_BASE + BS*1000 + REP))
      RK=""; [ -f "$KDIR/task_graph_rank0.json" ] && RK="--reuse-kernel"
      drain
      ENVA=$(arm_env "$ARM")
      env $ENVA MPK_ACCEPT_DIR="$T/demo/qwen3_5/accept" PYTHONPATH="$T/python" \
        timeout 9000 "$PY" -u "$OPT/profile_wave.py" \
          --batch-size "$BS" --max-seq-length "$MSL" --max-new-tokens "$NEWTOK" \
          --mbt 16 --page-size 256 --synthetic-prompt-len 256 \
          --synthetic-seed "$SEED" --out-dir "$OD" --kernel-dir "$KDIR" \
          --rep "$REP" --no-profiler $RK > "$M/logs/${TAG}.log" 2>&1
      echo "  [$TAG] rc=$? ${ENVA:-default} $(grep -h 'wall=' "$M/logs/${TAG}.log" | tail -1)"
      f=$OD/meta_bs${BS}_rep${REP}.json
      [ -f "$f" ] && mv "$f" "$OD/meta_bs${BS}_rep${REP}_${ARM}.json"
      g=$OD/tokens_bs${BS}_rep${REP}.json
      [ -f "$g" ] && mv "$g" "$OD/tokens_bs${BS}_rep${REP}_${ARM}.json"
    done
  done
done
echo "PROF_OVERHEAD_DONE $(date -Is)"
