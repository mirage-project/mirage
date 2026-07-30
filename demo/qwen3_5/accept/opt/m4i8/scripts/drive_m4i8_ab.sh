#!/usr/bin/env bash
# M4-I8: one GPU claim, two passes -- the A/S sweep over all five batch sizes,
# then arm O at bs1/bs16 as the simulator's falsifier.
set -uo pipefail
S=$HOME/mpk-qwen35/mirage-m4i8/demo/qwen3_5/accept/opt/m4i8/scripts
ARMS="A S" BSLIST="1 2 4 8 16" bash "$S/sweep_m4i8.sh"; RC1=$?
echo "PASS1_RC=$RC1"
[ "$RC1" -eq 0 ] || exit "$RC1"
ARMS="O" BSLIST="1 16" bash "$S/sweep_m4i8.sh"; RC2=$?
echo "PASS2_RC=$RC2"
echo "M4I8_AB_DONE $(date -Is)"
