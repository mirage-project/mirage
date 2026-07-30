#!/usr/bin/env bash
# M4-I6 GPU etiquette guard -- same contract as opt/m4i2/scripts/gpu_guard_m4i2.sh
# and opt/m4i5/scripts/gpu_guard_m4i5.sh: 3 samples 3s apart per candidate, the
# first candidate idle on EVERY sample wins, claim a lock, pin
# CUDA_VISIBLE_DEVICES, exec the command. Fails closed (rc 97). An MPK megakernel
# run needs the GPU EXCLUSIVE -- it can deadlock when shared, and a single read
# at claim time is not evidence of an idle device.
#
# TWO FERRET CHAINS ARE LIVE on this box (moe-fp8-grouped on workspace3,
# moe-router-topk on workspace5, 24h long-haul). They do NOT read these locks:
# ferret/pick_gpu.sh takes the LOWEST-memory GPU not already pinned by a running
# cc-run process, scanning nvidia-smi in INDEX order with a strict `<`, so ties
# among idle devices resolve to the LOWEST index. Candidates below therefore
# start from the HIGH end -- that is the part of the device list ferret reaches
# last -- and the caller must still audit gpu_before from each run's own log
# line, because the guard cannot stop a ferret episode that launches into the
# window between our claim and our first allocation.
set -uo pipefail
CANDIDATES="$1"; shift
if [ "${1:-}" == "--" ]; then shift; fi
SAMPLES=3; SLEEP_S=3
LOCKFILE=$HOME/mpk-qwen35/.gpu-locks/M4-I6.lock
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
    echo "GPU $gpu stable-idle across $SAMPLES samples -- claiming it (M4-I6)."
    echo "M4-I6 $gpu $(date -Iseconds) pid=$$" > "$LOCKFILE"
    export CUDA_VISIBLE_DEVICES="$gpu"
    echo "=== running (CUDA_VISIBLE_DEVICES=$gpu): $* ==="
    exec "$@"
  fi
  echo "GPU $gpu not stable, trying next candidate."
done
echo "REFUSING: no candidate GPU ($CANDIDATES) was stable-idle." >&2
exit 97
