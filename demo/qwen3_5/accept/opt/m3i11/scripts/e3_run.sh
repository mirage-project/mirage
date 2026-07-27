#!/usr/bin/env bash
# M3-I11 E3: uninitialised-intermediate causation test + cold-vs-warm isolation.
set -uo pipefail
M=$HOME/mpk-qwen35/m3i11
REPO=$HOME/mpk-qwen35/mirage
export ACC=$REPO/demo/qwen3_5/accept
PY=$HOME/mpk-qwen35/venv-mpk/bin/python
KDIR=${KDIR:-$HOME/mpk-qwen35/m3i9/kernels/bs1_msl1280}
export PATH=/usr/local/cuda-12.8/bin:$PATH
export HF_HOME=$HOME/mpk-qwen35/hf PYTHONUNBUFFERED=1
GPU=${GPU:?set GPU}; export CUDA_VISIBLE_DEVICES=$GPU
OUT=$M/out/e3; mkdir -p "$OUT" "$M/logs" "$M/kernels"
run() {  # run <tag> <extra args...>
  local tag=$1; shift
  echo "=== E3 $tag $(date -Is)"
  "$PY" "$M/e3_churn.py" --waves 2 --out "$OUT" --tag "$tag" \
      --kernel-dir "$KDIR" "$@" > "$M/logs/e3_$tag.log" 2>&1
  echo "rc=$? $(date -Is)"
}
run ctrl_a
run churnAA_a --churn-mb 3072 --churn-fill 0xAA
run churn55_a --churn-mb 3072 --churn-fill 0x55
run churnAA_b --churn-mb 3072 --churn-fill 0xAA
run ctrl_b
echo "=== E3 cold arm (in-process compile, fresh kernel dir) $(date -Is)"
rm -rf "$M/kernels/cold1"
KDIR2=$M/kernels/cold1
"$PY" "$M/e3_churn.py" --waves 2 --out "$OUT" --tag cold1 \
    --kernel-dir "$KDIR2" --fresh-compile > "$M/logs/e3_cold1.log" 2>&1
echo "rc=$? $(date -Is)"
echo "=== E3 done $(date -Is)"
