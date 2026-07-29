#!/usr/bin/env bash
# M3-I11 campaign 2 cold-compile census -- same protocol as the M3-I5c S6 arm
# (run_s6_census.sh): e4_full.py, bs4, msl 1280, 1024 new tokens, ten reference
# prompts, a fresh kernel dir compiled in-process per rep, KV/GDN wave-boundary
# fingerprints attached.
# usage: ARM=fix|ctrl GPU=<id> REPS=<n> TAG=<prefix> bash i11b_census.sh
set -uo pipefail
ARM=${ARM:?}
GPU=${GPU:?}
REPS=${REPS:?}
TAG=${TAG:?}
B=$HOME/mpk-qwen35
MIRAGE=$B/m3i11b-$ARM
PY=$B/venv-rm/bin/python
export ACC=$MIRAGE/demo/qwen3_5/accept
export PYTHONPATH=$MIRAGE/python
export HF_HOME=$B/hf
export PYTHONUNBUFFERED=1
export PATH=/usr/local/cuda-12.8/bin:$PATH
export CUDA_VISIBLE_DEVICES=$GPU
M=$B/m3i11b-out/census
OUT=$M/$TAG
mkdir -p "$OUT" "$M/logs" "$M/kernels"
SCR=$MIRAGE/demo/qwen3_5/accept/opt/m3i11/scripts

echo "=== CENSUS ARM=$ARM GPU=$GPU REPS=$REPS TAG=$TAG $(date -Is) ==="
grep -n "store_async_wait<0>\|tma_store_wait<0>" \
  "$MIRAGE/include/mirage/persistent_kernel/tasks/blackwell/linear_sm100_mpk.cuh"

for r in $(seq 1 "$REPS"); do
  t="${TAG}_c${r}"
  kd="$M/kernels/cold_${t}"
  rm -rf "$kd"
  echo "=== COLD rep $r/$REPS tag=$t $(date -Is)"
  "$PY" "$SCR/e4_full.py" --out "$OUT" --tag "$t" --bs 4 \
      --kernel-dir "$kd" --fresh-compile \
      > "$M/logs/${t}.log" 2>&1
  rc=$?
  echo "$t rc=$rc $(grep -o 'md5=[0-9a-f]*' "$M/logs/${t}.log" | tail -1) $(date -Is)"
  if [ "$rc" != 0 ]; then tail -8 "$M/logs/${t}.log"; fi
  rm -rf "$kd"
  df -h /raid | tail -1
done
echo "=== CENSUS DONE ARM=$ARM TAG=$TAG $(date -Is) ==="
