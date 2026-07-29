#!/usr/bin/env bash
# M4-I5: the MPK_MOE_N_SPLITS A/B -- three arms (k = 2 base / 4 / 8), five batch
# sizes, three reps, arms INTERLEAVED per (bs, rep) inside one GPU claim so
# drift or a co-tenant hits all three equally.  Protocol copied from
# m3i6a/scripts/phase6.sh, which is the version that survived two review
# rounds.
#
#   geometry B -- matched 256/1024 shape: synthetic 256-token prompts, msl=353
#                 (96 decode steps).  The AC-4-shaped geometry and the primary.
#   geometry C -- deep context: msl=897 (640 decode steps, ctx 257->896).  The
#                 routed-MoE cost is context-INDEPENDENT, so C is a control: the
#                 arm ratio should survive an 8x longer decode window, and a
#                 divergence would mean something other than the MoE moved.
#
# A KERNEL DIR PER (geometry, arm, bs) IS MANDATORY: moe_n_splits is a
# compile-time template argument (per-task OUTPUT_SIZE), and two arms sharing a
# --kernel-dir under --reuse-kernel run ONE binary and report themselves
# identical while nothing changed (M3-I7 defect 3).
set -uo pipefail
export PATH=/usr/local/cuda-12.8/bin:$PATH
export HF_HOME=$HOME/mpk-qwen35/hf
export PYTHONUNBUFFERED=1

# The guard exec's us with the device it actually won.  The drain gate and every
# audit record derive from THIS value, never from the candidate list.
GPU="${CUDA_VISIBLE_DEVICES:?the guard must pin CUDA_VISIBLE_DEVICES}"

T=$HOME/mpk-qwen35/mirage-m4i5
ACC=$T/demo/qwen3_5/accept
OPT=$ACC/opt
export MPK_ACCEPT_DIR=$ACC
export PYTHONPATH=$T/python
PY=$HOME/mpk-qwen35/venv-rm/bin/python
M=/var/tmp/m4i5_sweep
SEED_BASE=20260725
ARMS="${ARMS:-2 4 8}"
REPS="${REPS:-0 1 2}"
BSLIST="${BSLIST:-1 2 4 8 16}"
GEOMS="${GEOMS:-B C}"
mkdir -p "$M/logs" "$M/audit"

echo "########## M4-I5 MPK_MOE_N_SPLITS sweep  gpu=$GPU  $(date -Is) ##########"
echo "tree: $T  HEAD=$(git -C "$T" rev-parse --short HEAD) (+m4i5 overlay)"
echo "arms: $ARMS   reps: $REPS   bs: $BSLIST   geoms: $GEOMS   scratch: $M"
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
  local GEOM="$1" BS="$2" K="$3" REP="$4" MSL="$5"
  local TAG=${GEOM}_k${K}_bs${BS}_rep${REP}
  local OD=$M/noprof${GEOM}_k${K}
  [ -f "$OD/meta_bs${BS}_rep${REP}_k${K}.json" ] && { echo "  [$TAG] cached"; return; }
  local KDIR=$M/kernel_${GEOM}_k${K}_bs${BS}
  local SEED=$((SEED_BASE + BS*1000 + REP))
  local RK=""; [ -f "$KDIR/task_graph_rank0.json" ] && RK="--reuse-kernel"
  mkdir -p "$OD"
  drain
  MPK_MOE_N_SPLITS=$K timeout 3600 "$PY" -u "$OPT/profile_wave.py" \
      --batch-size "$BS" --max-seq-length "$MSL" --max-new-tokens 96 --mbt 16 \
      --page-size 256 --synthetic-prompt-len 256 --synthetic-seed "$SEED" \
      --out-dir "$OD" --kernel-dir "$KDIR" --rep "$REP" --no-profiler $RK \
      > "$M/logs/${TAG}.log" 2>&1
  local RC=$?
  echo "  [$TAG] rc=$RC $(grep -h 'wall=' "$M/logs/${TAG}.log" | tail -1)"
  local f=$OD/meta_bs${BS}_rep${REP}.json
  [ -f "$f" ] && mv "$f" "$OD/meta_bs${BS}_rep${REP}_k${K}.json"
  local g=$OD/tokens_bs${BS}_rep${REP}.json
  [ -f "$g" ] && mv "$g" "$OD/tokens_bs${BS}_rep${REP}_k${K}.json"
  # per-(bs) evidence is pulled by the poller as soon as it lands; a campaign
  # whose per-rep records are deleted is unverifiable (M3-I6a c3).
}

for GEOM in $GEOMS; do
  case $GEOM in B) MSL=353;; C) MSL=897;; *) echo "bad geom $GEOM"; exit 2;; esac
  echo; echo "--- geometry $GEOM (msl=$MSL), arms interleaved per (bs, rep) ---"
  for BS in $BSLIST; do
    for REP in $REPS; do
      for K in $ARMS; do run "$GEOM" "$BS" "$K" "$REP" "$MSL"; done
    done
    echo "  == geometry $GEOM bs$BS complete $(date -Is) =="
  done
done

echo; echo "SWEEP_DONE $(date -Is)"
df -BG /var/tmp | tail -1
