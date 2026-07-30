#!/usr/bin/env bash
# M4-I9: the fusion A/B -- arms INTERLEAVED per (bs, rep) inside ONE GPU claim so
# drift or a co-tenant hits every arm equally. Protocol from
# m4i8/scripts/sweep_m4i8.sh (<- m4i7 <- m4i5 <- m3i6a).
#
#   arm A (base) no env                    -- HEAD, fusion off.
#   arm F (new)  MPK_FUSE_SILU_QUANT=1     -- MoE activation SwiGLU fused into
#                its quantize: one fewer chain record per layer, and the bf16
#                `moe_act` round trip is gone.
#
# PRE-REGISTERED PREDICTION (opt/m4i9 fuse_model.py on M4-I8's own buffers, so
# the durations and the reconstruction are the SAME verified ones): the fusion
# removes 40 chain records carrying 160.1 / 189.4 / 184.2 us of task time and
# 49.7 / 46.4 / 45.2 us of chain gap at bs 1/8/16, so the measured step should
# fall by roughly 210 / 236 / 229 us = 3.6 / 2.9 / 2.3%. cp_exact should fall
# 177.3 / 189.0 / 184.4 us. An e2e win LARGER than that, or a win with an
# unchanged cp_exact, means something other than the chain moved.
#
# A KERNEL DIR PER (arm, bs) IS MANDATORY: the arm changes the GENERATED TU (a
# new task type, one fewer op), so two arms sharing --kernel-dir under
# --reuse-kernel would run ONE binary and report themselves identical while
# nothing changed (M3-I7 defect 3).
#
# Geometry B, the AC-4-shaped primary: synthetic 256-token prompts, msl=353,
# 96 decode steps, mbt=16.
set -uo pipefail
export PATH=/usr/local/cuda-12.8/bin:$PATH
export HF_HOME=$HOME/mpk-qwen35/hf
export PYTHONUNBUFFERED=1
GPU="${CUDA_VISIBLE_DEVICES:?the guard must pin CUDA_VISIBLE_DEVICES}"

T=${T:-$HOME/mpk-qwen35/mirage-m4i9}
ACC=$T/demo/qwen3_5/accept
OPT=$ACC/opt
export MPK_ACCEPT_DIR=$ACC
export PYTHONPATH=$T/python
PY=$HOME/mpk-qwen35/venv-rm/bin/python
M=${M:-/var/tmp/m4i9_sweep}
SEED_BASE=20260730
ARMS="${ARMS:-A F}"
REPS="${REPS:-0 1 2}"
BSLIST="${BSLIST:-1 2 4 8 16}"
MSL="${MSL:-353}"
NEWTOK="${NEWTOK:-96}"
PROF="${PROF:---no-profiler}"
mkdir -p "$M/logs" "$M/audit"

echo "########## M4-I9 fusion A/B  gpu=$GPU  $(date -Is) ##########"
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
    A) echo "" ;;                                     # HEAD, all flags off
    F) echo "MPK_FUSE_SILU_QUANT=1" ;;                # MoE SwiGLU + quantize
    N) echo "MPK_FUSE_NORM_QUANT=1" ;;                # pre-norm + quantize
    C) echo "MPK_FUSE_COMBINE_NORM=1" ;;              # combine + next pre-norm
    G) echo "MPK_FUSE_RECUR_QUANT=1" ;;               # gdn_recurrent + quantize
    S) echo "MPK_FUSE_SILU_QUANT=1 MPK_FUSE_NORM_QUANT=1 MPK_FUSE_RECUR_QUANT=1" ;;
    *) echo "BADARM"; return 1 ;;
  esac
}

run () {
  local ARM="$1" BS="$2" REP="$3"
  local TAG=${ARM}_bs${BS}_rep${REP}
  local OD=$M/out_${ARM}
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
      --out-dir "$OD" --kernel-dir "$KDIR" --rep "$REP" $PROF $RK \
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
