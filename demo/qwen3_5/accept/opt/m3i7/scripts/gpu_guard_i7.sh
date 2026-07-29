#!/usr/bin/env bash
# M3-I7 GPU etiquette guard -- same contract as opt/gpu_guard_m3i1.sh and
# opt/m3i6a/raw_meta/gpu_guard_i6a.sh: 3 samples 3s apart per candidate, the
# first candidate idle on EVERY sample wins, claims .gpu-locks/M3-I7.lock, pins
# CUDA_VISIBLE_DEVICES, execs the command. Fails closed (rc 97). MPK's
# megakernel claims all 148 SMs and spin-waits, so a co-tenant block is a
# self-sustaining deadlock, not just noise (SM-RESIDENCY LAW, M3-I2a).
#
# The guard proves idleness at CLAIM time only -- M3-I6a caught a previous rep's
# 34 GB process still tearing down and producing a fake 2.1x regression, so the
# gate script ALSO drains before every rep and records gpu_before per run.
#
# MAXMEM (default 500 MiB) is the per-sample resident-memory ceiling. Raising it
# is a DOCUMENTED measurement decision, not a loosened rule: on 2026-07-28 three
# foreign users parked ~920 MiB of idle CUDA context on GPUs 2-6 simultaneously,
# so no device on the box could satisfy 500 MiB and the gate would have starved.
# An idle context launches no blocks, so it does not violate the SM-residency
# law; what it does do is make "the device is clean" unprovable, which is why
# every run still records gpu_before and the analysis discards any rep whose
# device grew beyond the recorded floor. Never raise MAXMEM above a level you
# have separately confirmed to be idle context (util 0 %) rather than live work.
set -uo pipefail
CANDIDATES="$1"; shift
if [ "${1:-}" == "--" ]; then shift; fi
SAMPLES=3; SLEEP_S=3; MAXMEM="${MAXMEM:-500}"
LOCKFILE=$HOME/mpk-qwen35/.gpu-locks/M3-I7.lock
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
    echo "GPU $gpu stable-idle across $SAMPLES samples -- claiming it (M3-I7)."
    echo "foreign resident floor on this device at claim time: ${floor}MiB"
    echo "M3-I7 $gpu $(date -Iseconds) pid=$$ floor=${floor}MiB" > "$LOCKFILE"
    export CUDA_VISIBLE_DEVICES="$gpu"
    export MPK_I7_GPU_FLOOR="$floor"
    echo "=== running (CUDA_VISIBLE_DEVICES=$gpu): $* ==="
    exec "$@"
  fi
  echo "GPU $gpu not stable, trying next candidate."
done
echo "REFUSING: no candidate GPU ($CANDIDATES) was stable-idle." >&2
exit 97
