#!/usr/bin/env bash
# M4-status combined-tree measurement: the M4-I1 gate at all five pinned batch
# sizes on HEAD 6741b4ad, declared NON-BINDING (status measurement, not AC-6).
#
# --box-root is passed ABSOLUTE on purpose.  The gate's default box root is the
# literal string '$HOME/mpk-qwen35' ("expanded ON the box"), but every ssh call
# site single-quotes it, so the remote shell never expands it, while the
# heredoc'd work scripts double-quote it and DO expand it on the box.  The two
# halves then disagree: mkdir creates a literal '$HOME' directory while the work
# script redirects its stage log into the real path, which does not exist, so the
# remote tmux session dies at the redirect and the driver polls forever.  An
# absolute box root removes the remote expansion entirely.
RUNDIR=/home/catalyst/m4status-gate/20260730T101424Z
mkdir -p "$RUNDIR"
cd /home/catalyst/project/demo/qwen3_5/accept || exit 9
bash final.sh --non-binding \
  --agent-root /home/catalyst/agent \
  --box-root /home/muhengl/mpk-qwen35 \
  --candidates 5,2,3,0,6,1,7,4 \
  --stage-timeout 39600 \
  --poll-seconds 30 \
  --out "$RUNDIR" > "$RUNDIR/driver.log" 2>&1
echo "GATE_DRIVER_EXIT=$?" >> "$RUNDIR/driver.log"
date -Is >> "$RUNDIR/driver.log"
