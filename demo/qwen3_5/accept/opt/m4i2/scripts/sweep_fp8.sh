#!/usr/bin/env bash
# M4-I2: the dense-fp8 integration A/B -- two arms, five batch sizes, three reps,
# arms INTERLEAVED per (bs, rep) inside ONE GPU claim so drift or a co-tenant
# hits both equally. Protocol copied from m4i5/scripts/sweep_moe.sh, which
# copied m3i6a/scripts/phase6.sh -- the version that survived two review rounds.
#
#   arm A (base) MPK_FP8_DENSE_BASELINE=1 -- slice 128 + the golden path, i.e.
#                the pre-M4-I2 generated code, from THIS tree.
#   arm B (new)  default -- per-shape slices + the ferret v011 fast path.
#
# A KERNEL DIR PER (arm, bs) IS MANDATORY: the slice is a compile-time template
# argument and the path choice is a -D, so two arms sharing a --kernel-dir under
# --reuse-kernel would run ONE binary and report themselves identical while
# nothing changed (M3-I7 defect 3). Dir names carry the arm AND the bs.
#
# Geometry B, the AC-4-shaped primary: synthetic 256-token prompts, msl=353,
# 96 decode steps.
set -uo pipefail
export PATH=/usr/local/cuda-12.8/bin:$PATH
export HF_HOME=$HOME/mpk-qwen35/hf
export PYTHONUNBUFFERED=1

# The guard exec's us with the device it actually won. The drain gate and every
# audit record derive from THIS value, never from the candidate list.
GPU="${CUDA_VISIBLE_DEVICES:?the guard must pin CUDA_VISIBLE_DEVICES}"

T=${T:-$HOME/mpk-qwen35/mirage-m4i2}
ACC=$T/demo/qwen3_5/accept
OPT=$ACC/opt
export MPK_ACCEPT_DIR=$ACC
export PYTHONPATH=$T/python
PY=$HOME/mpk-qwen35/venv-rm/bin/python
M=${M:-/var/tmp/m4i2_sweep}
SEED_BASE=20260729
ARMS="${ARMS:-A B}"
REPS="${REPS:-0 1 2}"
BSLIST="${BSLIST:-1 2 4 8 16}"
MSL="${MSL:-353}"
NEWTOK="${NEWTOK:-96}"
mkdir -p "$M/logs" "$M/audit"

echo "########## M4-I2 dense-fp8 A/B  gpu=$GPU  $(date -Is) ##########"
echo "tree: $T  HEAD=$(git -C "$T" rev-parse --short HEAD)"
echo "arms: $ARMS  reps: $REPS  bs: $BSLIST  msl=$MSL newtok=$NEWTOK  scratch: $M"
"$PY" -c "import sys;sys.path.insert(0,'$T/python');import mirage,os;print('mirage from',os.path.realpath(mirage.__file__))"
df -BG /var/tmp | tail -1
AVAIL=$(df -BG --output=avail /var/tmp | tail -1 | tr -dc '0-9')
[ "${AVAIL:-0}" -lt 20 ] && { echo "REFUSING: /var/tmp headroom ${AVAIL}G < 20G" >&2; exit 96; }

smi () { nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits; }
used_on_pinned () {
  smi | awk -F',' -v g="$GPU" '{gsub(/ /,"",$1)} $1+0==g+0 {gsub(/ /,"",$2); print $2+0}'
}
drain () {   # the pinned device must be genuinely free before a rep starts
  local i used
  for i in $(seq 1 90); do
    used=$(used_on_pinned)
    [ "${used:-9999}" -lt 500 ] && return 0
    sleep 5
  done
  echo "    ABORT: device $GPU held ${used}MiB by a co-tenant after 450s --"
  echo "    releasing the claim so the guard can re-probe. Completed reps kept."
  exit 97
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
  # gpu_before is derived from THIS run's own pinned device, never a candidate
  # list (M3-I7's phantom-dirty-rep bug).
  smi > "$M/audit/gpu_before_${TAG}.txt"
  local ENVA=""; [ "$ARM" = A ] && ENVA="MPK_FP8_DENSE_BASELINE=1"
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
