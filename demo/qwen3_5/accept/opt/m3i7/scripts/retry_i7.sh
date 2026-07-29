#!/usr/bin/env bash
# M3-I7: retry the 3-sample free-GPU guard until it wins, then run the gate.
# The box is shared and the guard fails closed rather than sharing a device
# (SM-RESIDENCY LAW), so a single attempt often loses the lottery.
#   PHASES="ac3 perfA" CANDS=5,6,2,3 bash retry_i7.sh
set -uo pipefail
S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
for i in $(seq 1 "${TRIES:-120}"); do
  echo "===== claim attempt $i/${TRIES:-120} $(date -Is) ====="
  PHASES="${PHASES:-ac3 perfA perfM prof late}" REPS="${REPS:-3}" \
    MIN_RAID_G="${MIN_RAID_G:-4}" MAXMEM="${MAXMEM:-500}" \
    bash "$S/gpu_guard_i7.sh" "${CANDS:-5,6,2,3,0,4,7,1}" -- bash "$S/gate_i7.sh"
  rc=$?
  [ "$rc" -ne 97 ] && { echo "===== gate exited rc=$rc $(date -Is) ====="; exit "$rc"; }
  sleep "${SLEEP_S:-60}"
done
echo "GAVE UP" >&2; exit 97
