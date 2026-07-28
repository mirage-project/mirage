#!/usr/bin/env bash
# M3-I6a: retry the 3-sample free-GPU guard until it wins, then run $1.
set -uo pipefail
M=$HOME/mpk-qwen35/i6a
BODY="${1:?body script}"
for i in $(seq 1 "${TRIES:-90}"); do
  echo "===== claim attempt $i/$((${TRIES:-90})) $(date -Is) ====="
  bash "$M/gpu_guard_i6a.sh" "${CANDS:-6,3,1,0,2,7,4,5}" -- bash "$M/$BODY"
  rc=$?
  [ "$rc" -ne 97 ] && { echo "===== body exited rc=$rc $(date -Is) ====="; exit "$rc"; }
  sleep "${SLEEP_S:-60}"
done
echo "GAVE UP" >&2; exit 97
