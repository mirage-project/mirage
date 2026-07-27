#!/usr/bin/env bash
# M3-I11 E6: second GPU + the census's strongest correlate (the rep that
# COMPILED the kernel in-process). Every m3i9 anomaly landed on GPU 1/4/7;
# 3 of the 6 were the rep that compiled.
set -uo pipefail
M=$HOME/mpk-qwen35/m3i11
export ACC=$HOME/mpk-qwen35/mirage/demo/qwen3_5/accept
PY=$HOME/mpk-qwen35/venv-mpk/bin/python
export PATH=/usr/local/cuda-12.8/bin:$PATH
export HF_HOME=$HOME/mpk-qwen35/hf PYTHONUNBUFFERED=1
GPU=${GPU:?}
export CUDA_VISIBLE_DEVICES=$GPU
OUT=$M/out/e6
mkdir -p "$OUT" "$M/logs" "$M/kernels"

warm() {
  local bs="$1"
  local r="$2"
  local tag="g${GPU}_bs${bs}_w${r}"
  "$PY" "$M/e4_full.py" --out "$OUT" --tag "$tag" --bs "$bs" \
      --kernel-dir "$HOME/mpk-qwen35/m3i9/kernels/bs${bs}_msl1280" \
      > "$M/logs/e6_$tag.log" 2>&1
  echo "$tag rc=$? $(grep -o 'md5=[0-9a-f]*' "$M/logs/e6_$tag.log" | tail -1) $(date -Is)"
}

cold() {
  local bs="$1"
  local r="$2"
  local tag="g${GPU}_bs${bs}_c${r}"
  local kd="$M/kernels/cold_${tag}"
  rm -rf "$kd"
  "$PY" "$M/e4_full.py" --out "$OUT" --tag "$tag" --bs "$bs" \
      --kernel-dir "$kd" --fresh-compile \
      > "$M/logs/e6_$tag.log" 2>&1
  echo "$tag rc=$? $(grep -o 'md5=[0-9a-f]*' "$M/logs/e6_$tag.log" | tail -1) $(date -Is)"
  rm -rf "$kd"
}

for r in 1 2 3; do cold 4 "$r"; done
for r in 1 2 3; do cold 8 "$r"; done
for r in 1 2 3 4 5 6; do warm 4 "$r"; done
for r in 1 2 3 4; do warm 8 "$r"; done
echo "=== E6 done $(date -Is)"
