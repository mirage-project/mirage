#!/usr/bin/env bash
# M4-I4 launcher: start the campaign detached under ONE GPU claim, so a single
# claim covers every phase and every arm (interleaving the arms inside one window
# is the point -- M3-I7's bs4/bs8 A/B was assembled across two windows).
# Writes CAMPAIGN.done with the exit code when it finishes, so a poller can tell
# "still running" from "finished".
set -uo pipefail
export B=$HOME/mpk-qwen35
export S=$B/m4i4/scripts
LOG=$B/m4i4/campaign.log
export CANDS
export PHASES="${PHASES:-geomA geomM ac3gate}"
export REPS="${REPS:-3}"
export BSS="${BSS:-1 2 4 8 16}"
export ARMS="${ARMS:-none auto}"
export GATE_REPS="${GATE_REPS:-3}"
CANDS="${CANDS:-6,5,3,2,1,0,7,4}"
rm -f "$B/m4i4/CAMPAIGN.done"
inner () {
  bash "$S/gpu_guard_m4i4.sh" "$CANDS" -- bash "$S/run_m4i4.sh"
  echo "EXIT=$?" > "$B/m4i4/CAMPAIGN.done"
}
export -f inner
nohup setsid bash -c inner > "$LOG" 2>&1 < /dev/null &
echo "launched pid=$! log=$LOG phases='$PHASES' reps=$REPS bss='$BSS' arms='$ARMS'"
