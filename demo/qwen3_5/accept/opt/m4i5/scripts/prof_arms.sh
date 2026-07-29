#!/usr/bin/env bash
# M4-I5: the mechanism check for the falsified prediction.
#
# The wave-depth model said the routed-GEMM span keeps falling from k=4 to k=8.
# Geometry B says otherwise: k=8 beats k=4 at bs1/bs2 but LOSES to it at bs4/bs8
# and loses to the k=2 base at bs8/bs16.  A wall number cannot say why, so this
# captures a profiled wave per arm at the SAME geometry the A/B used and hands it
# to `width.py`, which reports per stage whether the span actually fell and what
# grew instead.
#
# Profiled captures need their OWN kernel dirs: --slots is baked into the kernel
# as MPK_PROFILER_BUFFER_ENTRIES, so a profiled run cannot reuse the
# --no-profiler dirs of the A/B.
set -uo pipefail
export PATH=/usr/local/cuda-12.8/bin:$PATH
export HF_HOME=$HOME/mpk-qwen35/hf
export PYTHONUNBUFFERED=1
GPU="${CUDA_VISIBLE_DEVICES:?the guard must pin CUDA_VISIBLE_DEVICES}"

T=$HOME/mpk-qwen35/mirage-m4i5
ACC=$T/demo/qwen3_5/accept
OPT=$ACC/opt
export MPK_ACCEPT_DIR=$ACC
export PYTHONPATH=$T/python
PY=$HOME/mpk-qwen35/venv-rm/bin/python
M=/var/tmp/m4i5_prof
SEED_BASE=20260725
ARMS="${ARMS:-2 4 8}"
BSLIST="${BSLIST:-8}"
MSL="${MSL:-353}"
SLOTS="${SLOTS:-120000000}"
mkdir -p "$M/logs"

echo "########## M4-I5 profiled arms  gpu=$GPU  msl=$MSL  $(date -Is) ##########"
df -BG /var/tmp | tail -1
AVAIL=$(df -BG --output=avail /var/tmp | tail -1 | tr -dc '0-9')
[ "${AVAIL:-0}" -lt 30 ] && { echo "REFUSING: /var/tmp headroom ${AVAIL}G < 30G" >&2; exit 96; }

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
  for K in $ARMS; do
    TAG=k${K}_bs${BS}
    OD=$M/prof_k${K}
    [ -f "$OD/meta_bs${BS}_rep0_k${K}.json" ] && { echo "  [$TAG] cached"; continue; }
    KDIR=$M/kernel_prof_k${K}_bs${BS}
    SEED=$((SEED_BASE + BS*1000))
    RK=""; [ -f "$KDIR/task_graph_rank0.json" ] && RK="--reuse-kernel"
    mkdir -p "$OD"
    drain
    MPK_MOE_N_SPLITS=$K timeout 5400 "$PY" -u "$OPT/profile_wave.py" \
        --batch-size "$BS" --max-seq-length "$MSL" --max-new-tokens 96 --mbt 16 \
        --page-size 256 --synthetic-prompt-len 256 --synthetic-seed "$SEED" \
        --out-dir "$OD" --kernel-dir "$KDIR" --rep 0 --slots "$SLOTS" --save-raw \
        $RK > "$M/logs/${TAG}.log" 2>&1
    echo "  [$TAG] rc=$? $(grep -h 'wall=\|profiler:' "$M/logs/${TAG}.log" | tail -2 | tr '\n' ' ')"
    for f in meta tokens; do
      [ -f "$OD/${f}_bs${BS}_rep0.json" ] && mv "$OD/${f}_bs${BS}_rep0.json" "$OD/${f}_bs${BS}_rep0_k${K}.json"
    done
    [ -f "$OD/raw_bs${BS}_rep0.npz" ] && mv "$OD/raw_bs${BS}_rep0.npz" "$OD/raw_bs${BS}_rep0_k${K}.npz"
    cp -f "$KDIR/task_graph_rank0.json" "$OD/task_graph_bs${BS}_k${K}.json" 2>/dev/null
    ls -la "$OD" | tail -3
  done
done
echo "PROF_ARMS_DONE $(date -Is)"
df -BG /var/tmp | tail -1
