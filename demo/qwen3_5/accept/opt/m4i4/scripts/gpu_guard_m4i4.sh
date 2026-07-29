#!/usr/bin/env bash
# M4-I4 GPU etiquette guard -- the standing 3-sample contract (opt/m3i1,
# opt/m3i6a, opt/m3i7): 3 samples 3 s apart per candidate, the first candidate
# idle on EVERY sample wins, claims its own lock file, pins
# CUDA_VISIBLE_DEVICES + the foreign floor, then execs. Fails closed (rc 97).
#
# Three ferret chains (ferret/workspace2,3,4) and an M4-I1 gate builder are
# sharing this box, so the candidate list is ordered away from whatever they
# claimed and the lock file is named per issue, never overwritten.
#
# MAXMEM is the per-sample resident-memory ceiling; see gpu_guard_i7.sh's header
# for why 500 MiB is not always satisfiable on this box and what raising it means.
set -uo pipefail
CANDIDATES="$1"; shift
if [ "${1:-}" == "--" ]; then shift; fi
SAMPLES=3; SLEEP_S=3; MAXMEM="${MAXMEM:-500}"
LOCKFILE=$HOME/mpk-qwen35/.gpu-locks/M4-I4.lock
mkdir -p "$(dirname "$LOCKFILE")"
IFS=',' read -ra CANDS <<< "$CANDIDATES"
for gpu in "${CANDS[@]}"; do
  echo "=== probing GPU $gpu ($SAMPLES samples, ${SLEEP_S}s apart, max ${MAXMEM}MiB) ==="
  ok=1; floor=999999
  for s in $(seq 1 $SAMPLES); do
    ROW=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits \
          | awk -F',' -v g="$gpu" '{gsub(/ /,"",$1)} $1+0==g+0')
    USED=$(echo "$ROW" | awk -F',' '{gsub(/ /,"",$2); print $2+0}')
    UTIL=$(echo "$ROW" | awk -F',' '{gsub(/ /,"",$3); print $3+0}')
    echo "  sample $s: GPU $gpu used=${USED}MiB util=${UTIL}%"
    if [ "$UTIL" -gt 5 ] || [ "$USED" -gt "$MAXMEM" ]; then ok=0; break; fi
    [ "$USED" -lt "$floor" ] && floor=$USED
    [ "$s" -lt "$SAMPLES" ] && sleep "$SLEEP_S"
  done
  if [ "$ok" -eq 1 ]; then
    echo "GPU $gpu stable-idle across $SAMPLES samples -- claiming it (M4-I4)."
    echo "foreign resident floor on this device at claim time: ${floor}MiB"
    echo "M4-I4 $gpu $(date -Iseconds) pid=$$ floor=${floor}MiB" > "$LOCKFILE"
    export CUDA_VISIBLE_DEVICES="$gpu"
    export MPK_M4I4_GPU_FLOOR="$floor"
    export MPK_M4I4_GPU_PHYS="$gpu"
    echo "=== running (CUDA_VISIBLE_DEVICES=$gpu): $* ==="
    exec "$@"
  fi
  echo "GPU $gpu not stable, trying next candidate."
done
echo "REFUSING: no candidate GPU ($CANDIDATES) was stable-idle." >&2
exit 97
