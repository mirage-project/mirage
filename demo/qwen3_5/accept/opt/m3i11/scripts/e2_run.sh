#!/usr/bin/env bash
# M3-I11 E2: NPROC independent processes x WAVES waves each, KV-cache
# fingerprinted after every wave.  Detects any ULP-level run-to-run difference,
# not just the ~2% that reach an argmax margin.
set -uo pipefail
M=$HOME/mpk-qwen35/m3i11
REPO=$HOME/mpk-qwen35/mirage
export ACC=$REPO/demo/qwen3_5/accept
PY=$HOME/mpk-qwen35/venv-mpk/bin/python
KDIR=${KDIR:-$HOME/mpk-qwen35/m3i9/kernels/bs1_msl1280}
export PATH=/usr/local/cuda-12.8/bin:$PATH
export HF_HOME=$HOME/mpk-qwen35/hf
export PYTHONUNBUFFERED=1
GPU=${GPU:?set GPU}
export CUDA_VISIBLE_DEVICES=$GPU
NPROC=${NPROC:-4}
WAVES=${WAVES:-4}
OUT=${OUT:-$M/out/e2}
PROMPT=${PROMPT:-p03-python}
mkdir -p "$OUT" "$M/logs"
for p in $(seq 1 "$NPROC"); do
  echo "=== E2 process $p/$NPROC on GPU $GPU $(date -Is)"
  "$PY" "$M/e2_fingerprint.py" --waves "$WAVES" --out "$OUT" --tag "p$p" \
      --prompt "$PROMPT" --kernel-dir "$KDIR" \
      > "$M/logs/e2_p$p.log" 2>&1
  echo "rc=$? $(date -Is)"
done
echo "=== E2 done $(date -Is)"
