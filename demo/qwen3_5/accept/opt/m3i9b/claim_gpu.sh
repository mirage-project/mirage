#!/usr/bin/env bash
# M3-I9b GPU etiquette guard, PRINT-ONLY variant of opt/m3i9/gpu_guard_m3i9.sh:
# 3 samples 3s apart per candidate, first candidate idle on EVERY sample wins,
# claims .gpu-locks/M3-I9b.lock and PRINTS the index on stdout (so a multi-run
# plan can claim once and hold, instead of exec'ing a single command).
# Fails closed (rc 97, nothing on stdout).
set -uo pipefail
CANDIDATES="$1"
SAMPLES=3; SLEEP_S=3
LOCKFILE=$HOME/mpk-qwen35/.gpu-locks/M3-I9b.lock
mkdir -p "$(dirname "$LOCKFILE")"
IFS=',' read -ra CANDS <<< "$CANDIDATES"
for gpu in "${CANDS[@]}"; do
  echo "=== probing GPU $gpu ($SAMPLES samples, ${SLEEP_S}s apart) ===" >&2
  ok=1
  for s in $(seq 1 $SAMPLES); do
    ROW=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | awk -F',' -v g="$gpu" '{gsub(/ /,"",$1)} $1+0==g+0')
    USED=$(echo "$ROW" | awk -F',' '{gsub(/ /,"",$2); print $2+0}')
    UTIL=$(echo "$ROW" | awk -F',' '{gsub(/ /,"",$3); print $3+0}')
    echo "  sample $s: GPU $gpu used=${USED}MiB util=${UTIL}%" >&2
    if [ "$UTIL" -gt 5 ] || [ "$USED" -gt 500 ]; then ok=0; break; fi
    [ "$s" -lt "$SAMPLES" ] && sleep "$SLEEP_S"
  done
  if [ "$ok" -eq 1 ]; then
    echo "M3-I9b $gpu $(date -Iseconds) pid=$$" > "$LOCKFILE"
    echo "GPU $gpu stable-idle across $SAMPLES samples -- claimed (M3-I9b)." >&2
    echo "$gpu"
    exit 0
  fi
  echo "GPU $gpu not stable, trying next candidate." >&2
done
echo "REFUSING: no candidate GPU ($CANDIDATES) was stable-idle." >&2
exit 97
