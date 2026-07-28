#!/usr/bin/env bash
# M3-I6a attention GPU etiquette guard (same contract as opt/gpu_guard_m3i1.sh /
# scripts/gpu_guard_i8.sh): 3 samples 3s apart per candidate, first candidate
# idle on EVERY sample wins, claims .gpu-locks/M3-I6A.lock, pins
# CUDA_VISIBLE_DEVICES, execs the command. Fails closed (rc 97). MPK megakernel
# runs need the GPU EXCLUSIVE -- it can deadlock when shared (resources.md).
set -uo pipefail
CANDIDATES="$1"; shift
if [ "${1:-}" == "--" ]; then shift; fi
SAMPLES=3; SLEEP_S=3
LOCKFILE=$HOME/mpk-qwen35/.gpu-locks/M3-I6A.lock
mkdir -p "$(dirname "$LOCKFILE")"
IFS=',' read -ra CANDS <<< "$CANDIDATES"
for gpu in "${CANDS[@]}"; do
  echo "=== probing GPU $gpu ($SAMPLES samples, ${SLEEP_S}s apart) ==="
  ok=1
  for s in $(seq 1 $SAMPLES); do
    ROW=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | awk -F',' -v g="$gpu" '{gsub(/ /,"",$1)} $1+0==g+0')
    USED=$(echo "$ROW" | awk -F',' '{gsub(/ /,"",$2); print $2+0}')
    UTIL=$(echo "$ROW" | awk -F',' '{gsub(/ /,"",$3); print $3+0}')
    echo "  sample $s: GPU $gpu used=${USED}MiB util=${UTIL}%"
    if [ "$UTIL" -gt 5 ] || [ "$USED" -gt 500 ]; then ok=0; break; fi
    [ "$s" -lt "$SAMPLES" ] && sleep "$SLEEP_S"
  done
  if [ "$ok" -eq 1 ]; then
    echo "GPU $gpu stable-idle across $SAMPLES samples -- claiming it (M3-I6A)."
    echo "M3-I6A $gpu $(date -Iseconds) pid=$$" > "$LOCKFILE"
    export CUDA_VISIBLE_DEVICES="$gpu"
    echo "=== running (CUDA_VISIBLE_DEVICES=$gpu): $* ==="
    exec "$@"
  fi
  echo "GPU $gpu not stable, trying next candidate."
done
echo "REFUSING: no candidate GPU ($CANDIDATES) was stable-idle." >&2
exit 97
