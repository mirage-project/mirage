#!/usr/bin/env bash
set -uo pipefail
M=$HOME/mpk-qwen35/m3i11
export ACC=$HOME/mpk-qwen35/mirage/demo/qwen3_5/accept
PY=$HOME/mpk-qwen35/venv-mpk/bin/python
export PATH=/usr/local/cuda-12.8/bin:$PATH
export HF_HOME=$HOME/mpk-qwen35/hf PYTHONUNBUFFERED=1
GPU=${GPU:?}
export CUDA_VISIBLE_DEVICES=$GPU
OUT=$M/out/e7
KD=$M/kernels/expose_bs1_msl1280
mkdir -p "$OUT" "$M/logs" "$M/kernels"
for p in 1 2 3 4 5; do
  echo "=== E7 p$p $(date -Is)"
  "$PY" "$M/e7_router_mask.py" --out "$OUT" --tag "p$p" --waves 2 \
      --kernel-dir "$KD" > "$M/logs/e7_p$p.log" 2>&1
  echo "rc=$? $(date -Is)"
done
echo "=== E7 done $(date -Is)"
