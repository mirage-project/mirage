#!/usr/bin/env bash
# M3-I11 E5: high-volume detection campaign.
# The census rate is ~2-3% of trajectories, so 40 trajectories refutes nothing.
# This runs ~300 trajectories with the KV fingerprint attached, so every
# perturbation is caught (not only the ~1-in-40 that reaches an argmax margin),
# and a caught event comes with its first perturbed (layer, cache slot).
set -uo pipefail
M=$HOME/mpk-qwen35/m3i11
export ACC=$HOME/mpk-qwen35/mirage/demo/qwen3_5/accept
PY=$HOME/mpk-qwen35/venv-mpk/bin/python
export PATH=/usr/local/cuda-12.8/bin:$PATH
export HF_HOME=$HOME/mpk-qwen35/hf PYTHONUNBUFFERED=1
GPU=${GPU:?}; export CUDA_VISIBLE_DEVICES=$GPU
OUT=$M/out/e5; mkdir -p "$OUT" "$M/logs"
one() {  # one <bs> <rep>
  local bs=$1 r=$2 tag="bs${1}_r${2}"
  "$PY" "$M/e4_full.py" --out "$OUT" --tag "$tag" --bs "$bs" \
      --kernel-dir "$HOME/mpk-qwen35/m3i9/kernels/bs${bs}_msl1280" \
      > "$M/logs/e5_$tag.log" 2>&1
  echo "$tag rc=$? $(grep -o 'md5=[0-9a-f]*' $M/logs/e5_$tag.log | tail -1) $(date -Is)"
}
for r in $(seq 1 20); do one 4 "$r"; done
for r in $(seq 1 10); do one 8 "$r"; done
for r in $(seq 1 6);  do one 1 "$r"; done
echo "=== E5 done $(date -Is)"
