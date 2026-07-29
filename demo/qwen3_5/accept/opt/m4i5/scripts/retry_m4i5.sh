#!/usr/bin/env bash
# M4-I5: retry the 3-sample free-GPU guard until it wins, then run the body.
# The body is resumable (it skips any rep whose meta already exists), so a
# rc=97 release mid-campaign costs nothing but the current rep.
set -uo pipefail
M=$HOME/mpk-qwen35/m4i5
BODY="${1:?body script}"
for i in $(seq 1 "${TRIES:-200}"); do
  echo "===== claim attempt $i/${TRIES:-200} $(date -Is) ====="
  bash "$M/gpu_guard_m4i5.sh" "${CANDS:-0,1,2,5,4,3,7,6}" -- bash "$M/$BODY"
  rc=$?
  [ "$rc" -ne 97 ] && { echo "===== body exited rc=$rc $(date -Is) ====="; exit "$rc"; }
  sleep "${SLEEP_S:-60}"
done
echo "GAVE UP" >&2; exit 97
