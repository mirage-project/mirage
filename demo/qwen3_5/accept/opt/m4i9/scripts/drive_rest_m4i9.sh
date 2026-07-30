#!/usr/bin/env bash
# M4-I9 continuation: everything left, chained inside ONE GPU claim so the box's
# co-tenants get probed once rather than five times.
#   1. flag C's unit/oracle, both nvcc lanes (no megakernel involved)
#   2. profiled captures of arms N, G and the three-flag stack S, at M4-I8's own
#      geometry and windows, for the per-flag AND STACKED cp_exact re-derivation
#   3. AC-3 at all five batch sizes for N, G and S
set -uo pipefail
S=${S:-$HOME/mpk-qwen35/mirage-m4i9/demo/qwen3_5/accept/opt/m4i9/scripts}
echo "########## M4-I9 remaining gates, gpu=${CUDA_VISIBLE_DEVICES:-?} $(date -Is)"

bash "$S/gate_unit_gdn_m4i9.sh"; echo "GATE2C_RC=$?"

for A in N G S; do
  ARM=$A bash "$S/prof_m4i9.sh"; echo "PROF_${A}_RC=$?"
done

for A in S N G; do
  ARM=$A bash "$S/gate_ac3_m4i9.sh"; echo "AC3_${A}_RC=$?"
done
echo "M4I9_REST_DONE $(date -Is)"
