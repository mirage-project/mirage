#!/usr/bin/env bash
# M3-I6a geometries B and C, RE-RUN with a per-rep GPU DRAIN GATE.
#
# Why: the first geometry-B attempt produced one absurd point (bs1 qp2 rep0 at
# 2658.8 ms against qp4's 1247.9 ms).  `meta.gpu_before` shows 36364 MiB already
# resident on the pinned device at that run's start -- the PREVIOUS rep's
# 34.4 GB process had not finished tearing down, so two MPK megakernels briefly
# shared one GPU.  resources.md says that is invalid (and can deadlock), so the
# fix is structural, not a re-run-and-hope: before every rep, wait for the pinned
# device to drain below 500 MiB, and record gpu_before so any rep that still
# starts dirty can be discarded in analysis rather than silently averaged in.
#
#   geometry B = matched 256/1024, msl=353  (96 decode steps)  -- M3-I10 armA basis
#   geometry C = deep context,     msl=897  (640 decode steps) -- ~97% decode at
#                bs1, so the unprofiled wall ratio IS the decode-step ratio
set -uo pipefail
export PATH=/usr/local/cuda-12.8/bin:$PATH
export HF_HOME=$HOME/mpk-qwen35/hf
export PYTHONUNBUFFERED=1
GPU="${CUDA_VISIBLE_DEVICES:?guard must pin the GPU}"
T=$HOME/mpk-qwen35/mirage-i6a
OPT=$T/demo/qwen3_5/accept/opt
export MPK_ACCEPT_DIR=$T/demo/qwen3_5/accept
export PYTHONPATH=$T/python
PY=$HOME/mpk-qwen35/venv-mpk/bin/python
M=$HOME/mpk-qwen35/i6a
SEED_BASE=20260725
mkdir -p "$M/perf/logs"

drain () {  # wait for the pinned device to be genuinely free again
  local i used
  for i in $(seq 1 60); do
    used=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
           | awk -F',' -v g="$GPU" '{gsub(/ /,"",$1)} $1+0==g+0 {gsub(/ /,"",$2); print $2+0}')
    [ "${used:-9999}" -lt 500 ] && { echo "    drained (${used}MiB after ${i} checks)"; return 0; }
    sleep 5
  done
  echo "    WARNING: device $GPU still at ${used}MiB after 300s"
  return 1
}

one () {  # geom bs qp rep msl outdir
  local GEOM="$1" BS="$2" QP="$3" REP="$4" MSL="$5" OD="$6"
  local KDIR=$M/perf/kernel_${GEOM}_qp${QP}_bs${BS}_noprof
  local TAG=${GEOM}_qp${QP}_bs${BS}_rep${REP}
  local SEED=$((SEED_BASE + BS*1000 + REP))
  local RK=""; [ -f "$KDIR/task_graph_rank0.json" ] && RK="--reuse-kernel"
  drain
  MPK_ATTN_Q_PASS=$QP timeout 3000 $PY -u "$OPT/profile_wave.py" \
      --batch-size "$BS" --max-seq-length "$MSL" --max-new-tokens 96 --mbt 16 \
      --page-size 256 --synthetic-prompt-len 256 --synthetic-seed "$SEED" \
      --out-dir "$OD" --kernel-dir "$KDIR" --rep "$REP" --no-profiler $RK \
      > "$M/perf/logs/${TAG}.log" 2>&1
  echo "  [$TAG] rc=$? $(grep -h 'wall=' "$M/perf/logs/${TAG}.log" | tail -1)"
  local f=$OD/meta_bs${BS}_rep${REP}.json
  [ -f "$f" ] && mv "$f" "$OD/meta_bs${BS}_rep${REP}_qp${QP}.json"
}

echo "########## geometry B+C re-run, drain-gated  gpu=$GPU $(date -Is) ##########"
for QP in 4 2; do
  rm -rf "$M/perf/noprofB_qp${QP}"        # the contaminated arm is discarded whole
  mkdir -p "$M/perf/noprofB_qp${QP}" "$M/perf/noprofC_qp${QP}"
done
echo "--- geometry B: msl=353 ---"
for BS in 1 2 4 8 16; do for QP in 4 2; do for REP in 0 1 2; do
  one B "$BS" "$QP" "$REP" 353 "$M/perf/noprofB_qp${QP}"
done; done; done
echo "--- geometry C: msl=897 ---"
for BS in 1 8 16; do for QP in 4 2; do for REP in 0 1 2; do
  one C "$BS" "$QP" "$REP" 897 "$M/perf/noprofC_qp${QP}"
done; done; done
echo "PHASE5_DONE $(date -Is)"
