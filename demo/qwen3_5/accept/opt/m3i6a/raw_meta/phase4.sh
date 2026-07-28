#!/usr/bin/env bash
# M3-I6a geometry C: UNPROFILED e2e A/B at the DEEP-context regime.
#
# Geometry B (msl=353) only reaches decode context 257-352, and its wave wall is
# increasingly prefill at higher bs (256*bs/16 prefill iterations against 96
# decode steps), so it dilutes a decode-stage win -- the same trap M3-I3's
# i3_medians.py recorded.  msl=897 runs 640 decode steps against 16 prefill
# iterations per request, so at bs1 the wall is ~97% decode and the unprofiled
# wall ratio IS the decode-step ratio, with no profiler in the loop.
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
echo "########## geometry C: msl=897, unprofiled, 3 reps  gpu=$GPU $(date -Is) ##########"
for QP in 4 2; do mkdir -p "$M/perf/noprofC_qp${QP}"; done
for BS in 1 8 16; do
  for QP in 4 2; do
    KDIR=$M/perf/kernel_C_qp${QP}_bs${BS}_noprof
    for REP in 0 1 2; do
      SEED=$((SEED_BASE + BS*1000 + REP))
      RK=""; [ -f "$KDIR/task_graph_rank0.json" ] && RK="--reuse-kernel"
      TAG=C_qp${QP}_bs${BS}_rep${REP}
      MPK_ATTN_Q_PASS=$QP timeout 3000 $PY -u "$OPT/profile_wave.py" \
          --batch-size "$BS" --max-seq-length 897 --max-new-tokens 96 --mbt 16 \
          --page-size 256 --synthetic-prompt-len 256 --synthetic-seed "$SEED" \
          --out-dir "$M/perf/noprofC_qp${QP}" --kernel-dir "$KDIR" \
          --rep "$REP" --no-profiler $RK > "$M/perf/logs/${TAG}.log" 2>&1
      echo "  [$TAG] rc=$? $(grep -h 'wall=' "$M/perf/logs/${TAG}.log" | tail -1)"
      f=$M/perf/noprofC_qp${QP}/meta_bs${BS}_rep${REP}.json
      [ -f "$f" ] && mv "$f" "$M/perf/noprofC_qp${QP}/meta_bs${BS}_rep${REP}_qp${QP}.json"
    done
  done
done
echo "PHASE4_DONE $(date -Is)"
