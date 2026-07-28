#!/usr/bin/env bash
# M3-I6a: retry the 3-sample free-GPU guard until it wins, then exec the gates.
# The box is contended (another builder is cycling GPUs), and the guard fails
# closed rather than sharing, so a single attempt usually loses the lottery.
# Etiquette is unchanged: still 3 samples 3 s apart, still <=500 MiB and <=5%
# util on EVERY sample, still one claim at a time.
set -uo pipefail
M=$HOME/mpk-qwen35/i6a
CANDS="${CANDS:-6,3,1,0,2,7,4,5}"
TRIES="${TRIES:-60}"
SLEEP_S="${SLEEP_S:-60}"
for i in $(seq 1 "$TRIES"); do
  echo "===== claim attempt $i/$TRIES $(date -Is) ====="
  bash "$M/gpu_guard_i6a.sh" "$CANDS" -- bash "$M/gate_all.sh"
  rc=$?
  if [ "$rc" -ne 97 ]; then
    echo "===== gates exited rc=$rc $(date -Is) ====="
    exit "$rc"
  fi
  echo "no free GPU; retrying in ${SLEEP_S}s"
  sleep "$SLEEP_S"
done
echo "GAVE UP after $TRIES attempts" >&2
exit 97
