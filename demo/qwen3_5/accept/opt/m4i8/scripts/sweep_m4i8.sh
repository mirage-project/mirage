#!/usr/bin/env bash
# M4-I8: the scheduler A/B -- arms INTERLEAVED per (bs, rep) inside ONE GPU claim
# so drift or a co-tenant hits every arm equally. Protocol copied verbatim from
# m4i7/scripts/sweep_moe_m4i7.sh (which came from m4i5 <- m3i6a, the version that
# survived two review rounds).
#
#   arm A (base) no -D                        -- HEAD, both knobs off.
#   arm S (new)  MPK_EVENT_WAIT_GPU_SCOPE=1   -- ld.acquire.gpu instead of
#                ld.acquire.sys on the local-event counter spin.
#   arm O (new)  MPK_WORKER_OOO_POP=1         -- out-of-order pop inside the
#                loaded task-desc buffer.
#
# PRE-REGISTERED PREDICTION (sched_gap.py, dependency-invariant-verified sims,
# relative to the slide_1 model of HEAD): arm S targets the 18-28% that the
# zero-latency counterfactual attributes to dispatch/poll latency, so it should
# move the step by a clearly measurable amount; arm O's realisable variant
# (batch_8) predicts only -0.3% at bs1, so arm O is expected to be a NULL and is
# run at bs1 as the simulator's falsifier. If arm O comes out large, the ranking
# in the decomposition is wrong and has to be redone.
#
# A KERNEL DIR PER (arm, bs) IS MANDATORY: the arm is a -D in the generated TU,
# so two arms sharing --kernel-dir under --reuse-kernel would run ONE binary and
# report themselves identical while nothing changed (M3-I7 defect 3).
#
# Geometry B, the AC-4-shaped primary: synthetic 256-token prompts, msl=353,
# 96 decode steps, mbt=16.
set -uo pipefail
export PATH=/usr/local/cuda-12.8/bin:$PATH
export HF_HOME=$HOME/mpk-qwen35/hf
export PYTHONUNBUFFERED=1
GPU="${CUDA_VISIBLE_DEVICES:?the guard must pin CUDA_VISIBLE_DEVICES}"

T=${T:-$HOME/mpk-qwen35/mirage-m4i8}
ACC=$T/demo/qwen3_5/accept
OPT=$ACC/opt
export MPK_ACCEPT_DIR=$ACC
export PYTHONPATH=$T/python
PY=$HOME/mpk-qwen35/venv-rm/bin/python
M=${M:-/var/tmp/m4i8_sweep}
SEED_BASE=20260730
ARMS="${ARMS:-A S}"
REPS="${REPS:-0 1 2}"
BSLIST="${BSLIST:-1 2 4 8 16}"
MSL="${MSL:-353}"
NEWTOK="${NEWTOK:-96}"
mkdir -p "$M/logs" "$M/audit"

echo "########## M4-I8 scheduler A/B  gpu=$GPU  $(date -Is) ##########"
echo "tree: $T  HEAD=$(git -C "$T" rev-parse --short HEAD)"
echo "arms: $ARMS  reps: $REPS  bs: $BSLIST  msl=$MSL newtok=$NEWTOK  scratch: $M"
"$PY" -c "import sys;sys.path.insert(0,'$T/python');import mirage,os;print('mirage from',os.path.realpath(mirage.__file__))"
md5sum "$T"/python/mirage/core.cpython-*.so
df -BG /var/tmp | tail -1
AVAIL=$(df -BG --output=avail /var/tmp | tail -1 | tr -dc '0-9')
[ "${AVAIL:-0}" -lt 20 ] && { echo "REFUSING: /var/tmp headroom ${AVAIL}G < 20G" >&2; exit 96; }

smi () { nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits; }
used_on_pinned () {
  smi | awk -F',' -v g="$GPU" '{gsub(/ /,"",$1)} $1+0==g+0 {gsub(/ /,"",$2); print $2+0}'
}
drain () {
  local i used
  for i in $(seq 1 90); do
    used=$(used_on_pinned); [ "${used:-9999}" -lt 500 ] && return 0; sleep 5
  done
  echo "    ABORT: device $GPU held ${used}MiB by a co-tenant after 450s --"
  echo "    releasing the claim so the guard can re-probe. Completed reps kept."
  exit 97
}

arm_env () {
  case "$1" in
    A) echo "" ;;
    S) echo "MPK_EVENT_WAIT_GPU_SCOPE=1" ;;
    O) echo "MPK_WORKER_OOO_POP=1" ;;
    SO) echo "MPK_EVENT_WAIT_GPU_SCOPE=1 MPK_WORKER_OOO_POP=1" ;;
    *) echo "BADARM"; return 1 ;;
  esac
}

run () {
  local ARM="$1" BS="$2" REP="$3"
  local TAG=${ARM}_bs${BS}_rep${REP}
  local OD=$M/noprof_${ARM}
  [ -f "$OD/meta_bs${BS}_rep${REP}_${ARM}.json" ] && { echo "  [$TAG] cached"; return; }
  local KDIR=$M/kernel_${ARM}_bs${BS}
  local SEED=$((SEED_BASE + BS*1000 + REP))
  local RK=""; [ -f "$KDIR/task_graph_rank0.json" ] && RK="--reuse-kernel"
  mkdir -p "$OD"
  drain
  smi > "$M/audit/gpu_before_${TAG}.txt"
  local ENVA; ENVA=$(arm_env "$ARM")
  env $ENVA timeout 5400 "$PY" -u "$OPT/profile_wave.py" \
      --batch-size "$BS" --max-seq-length "$MSL" --max-new-tokens "$NEWTOK" --mbt 16 \
      --page-size 256 --synthetic-prompt-len 256 --synthetic-seed "$SEED" \
      --out-dir "$OD" --kernel-dir "$KDIR" --rep "$REP" --no-profiler $RK \
      > "$M/logs/${TAG}.log" 2>&1
  local RC=$?
  echo "  [$TAG] rc=$RC ${ENVA:-default} $(grep -h 'wall=' "$M/logs/${TAG}.log" | tail -1)"
  local f=$OD/meta_bs${BS}_rep${REP}.json
  [ -f "$f" ] && mv "$f" "$OD/meta_bs${BS}_rep${REP}_${ARM}.json"
  local g=$OD/tokens_bs${BS}_rep${REP}.json
  [ -f "$g" ] && mv "$g" "$OD/tokens_bs${BS}_rep${REP}_${ARM}.json"
}

for BS in $BSLIST; do
  for REP in $REPS; do
    for ARM in $ARMS; do run "$ARM" "$BS" "$REP"; done
  done
  echo "  == bs$BS complete $(date -Is) =="
done
echo; echo "SWEEP_DONE $(date -Is)"
df -BG /var/tmp | tail -1
