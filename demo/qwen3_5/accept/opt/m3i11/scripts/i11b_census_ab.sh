#!/usr/bin/env bash
# M3-I11 campaign 2 DISCRIMINATOR: cold-compile census, CTRL and FIX
# INTERLEAVED on one GPU in one window.
#
# Protocol identical to the M3-I5c S6 arm (run_s6_census.sh): e4_full.py, bs4,
# msl 1280, 1024 new tokens, the ten reference prompts, a fresh kernel dir
# compiled in-process for every rep, KV/GDN wave-boundary fingerprints attached.
# S6 produced 1 diverging rep in 10 cold reps on GPU7.
#
# Interleaving matters: the census anomalies were seen in a heavily contended
# window, so box load is a plausible modulator. Alternating arms rep-by-rep on
# the same GPU means load drift cannot masquerade as an arm effect.
#
# usage: GPU=<id> PAIRS=<n> EXTRA_FIX=<n> TAG=<prefix> bash i11b_census_ab.sh
set -uo pipefail
GPU=${GPU:?}
PAIRS=${PAIRS:-10}
EXTRA_FIX=${EXTRA_FIX:-5}
TAG=${TAG:?}
B=$HOME/mpk-qwen35
PY=$B/venv-rm/bin/python
export HF_HOME=$B/hf
export PYTHONUNBUFFERED=1
export PATH=/usr/local/cuda-12.8/bin:$PATH
export CUDA_VISIBLE_DEVICES=$GPU
M=$B/m3i11b-out/census
mkdir -p "$M/logs" "$M/kernels" "$M/${TAG}_ctrl" "$M/${TAG}_fix"

echo "=== CENSUS A/B GPU=$GPU PAIRS=$PAIRS EXTRA_FIX=$EXTRA_FIX TAG=$TAG $(date -Is) ==="
for arm in ctrl fix; do
  grep -n "store_async_wait<0>\|tma_store_wait<0>" \
    "$B/m3i11b-$arm/include/mirage/persistent_kernel/tasks/blackwell/linear_sm100_mpk.cuh" \
    | sed "s/^/  $arm: /"
done

one() {  # arm rep
  local arm="$1" r="$2"
  local mirage=$B/m3i11b-$arm
  local t="${TAG}_${arm}_c${r}"
  local kd="$M/kernels/cold_${t}"
  rm -rf "$kd"
  ACC=$mirage/demo/qwen3_5/accept PYTHONPATH=$mirage/python \
    "$PY" "$mirage/demo/qwen3_5/accept/opt/m3i11/scripts/e4_full.py" \
      --out "$M/${TAG}_${arm}" --tag "$t" --bs 4 \
      --kernel-dir "$kd" --fresh-compile \
      > "$M/logs/${t}.log" 2>&1
  local rc=$?
  echo "  $t rc=$rc $(grep -o 'md5=[0-9a-f]*' "$M/logs/${t}.log" | tail -1) $(date -Is)"
  [ "$rc" != 0 ] && tail -6 "$M/logs/${t}.log" | sed 's/^/    /'
  rm -rf "$kd"
}

for r in $(seq 1 "$PAIRS"); do
  echo "=== pair $r/$PAIRS $(date -Is) ==="
  one ctrl "$r"
  one fix  "$r"
  df -h /raid | tail -1
done
for r in $(seq $((PAIRS + 1)) $((PAIRS + EXTRA_FIX))); do
  echo "=== extra fix rep $r $(date -Is) ==="
  one fix "$r"
done
echo "=== CENSUS_AB_DONE TAG=$TAG $(date -Is) ==="
