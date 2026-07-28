#!/usr/bin/env bash
set -uo pipefail
M=$HOME/mpk-qwen35/mirage-i5c-run/demo/qwen3_5/accept/opt/m3i11
export ACC=$HOME/mpk-qwen35/mirage-i5c-run/demo/qwen3_5/accept
export PYTHONPATH=$HOME/mpk-qwen35/mirage-i5c-run/python
PY=$HOME/mpk-qwen35/venv-rm/bin/python
export PATH=/usr/local/cuda-12.8/bin:$PATH
export HF_HOME=$HOME/mpk-qwen35/hf PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=5
OUT=$HOME/mpk-qwen35/m3i5c/e7_postfix_out
KD=$HOME/mpk-qwen35/m3i5c/kernels/e7_postfix_bs1_msl1280
mkdir -p "$OUT" "$HOME/mpk-qwen35/m3i5c/logs" "$HOME/mpk-qwen35/m3i5c/kernels"
for p in 1 2 3 4 5; do
  echo "=== E7-POSTFIX p$p $(date -Is)"
  "$PY" "$M/scripts/e7_router_mask.py" --out "$OUT" --tag "p$p" --waves 2 \
      --kernel-dir "$KD" > "$HOME/mpk-qwen35/m3i5c/logs/e7_postfix_p$p.log" 2>&1
  echo "rc=$? $(date -Is)"
done
echo "=== E7-POSTFIX done $(date -Is)"
