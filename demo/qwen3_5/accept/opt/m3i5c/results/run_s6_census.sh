#!/usr/bin/env bash
set -uo pipefail
M=$HOME/mpk-qwen35/mirage-i5c-run/demo/qwen3_5/accept/opt/m3i11/scripts
export ACC=$HOME/mpk-qwen35/mirage-i5c-run/demo/qwen3_5/accept
export PYTHONPATH=$HOME/mpk-qwen35/mirage-i5c-run/python
PY=$HOME/mpk-qwen35/venv-rm/bin/python
export PATH=/usr/local/cuda-12.8/bin:$PATH
export HF_HOME=$HOME/mpk-qwen35/hf PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=7
OUT=$HOME/mpk-qwen35/m3i5c/s6_census_out
WARM_KD=$HOME/mpk-qwen35/m3i5c/kernels/s6_warm_bs4_msl1280
mkdir -p "$OUT" "$HOME/mpk-qwen35/m3i5c/logs" "$HOME/mpk-qwen35/m3i5c/kernels"

cold() {
  local r="$1"
  local tag="g7_bs4_c${r}"
  local kd="$HOME/mpk-qwen35/m3i5c/kernels/cold_${tag}"
  rm -rf "$kd"
  echo "=== COLD rep $r $(date -Is)"
  "$PY" "$M/e4_full.py" --out "$OUT" --tag "$tag" --bs 4 \
      --kernel-dir "$kd" --fresh-compile \
      > "$HOME/mpk-qwen35/m3i5c/logs/s6_$tag.log" 2>&1
  echo "$tag rc=$? $(grep -o "md5=[0-9a-f]*" "$HOME/mpk-qwen35/m3i5c/logs/s6_$tag.log" | tail -1) $(date -Is)"
  rm -rf "$kd"
}

warm() {
  local r="$1"
  local tag="g7_bs4_w${r}"
  echo "=== WARM rep $r $(date -Is)"
  "$PY" "$M/e4_full.py" --out "$OUT" --tag "$tag" --bs 4 \
      --kernel-dir "$WARM_KD" \
      > "$HOME/mpk-qwen35/m3i5c/logs/s6_$tag.log" 2>&1
  echo "$tag rc=$? $(grep -o "md5=[0-9a-f]*" "$HOME/mpk-qwen35/m3i5c/logs/s6_$tag.log" | tail -1) $(date -Is)"
}

for r in 1 2 3; do cold "$r"; done
for r in 1 2 3 4 5 6 7; do warm "$r"; done
echo "=== S6 CENSUS DONE $(date -Is)"
