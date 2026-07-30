#!/usr/bin/env bash
# M4-I8: run sched_gap.py (exact step decomposition + policy simulations) over
# the three profiled cells.  CPU only -- no GPU claim needed.
set -uo pipefail
M=${M:-/var/tmp/m4i8_prof}
T=${T:-$HOME/mpk-qwen35/mirage-m4i8}
S=$T/demo/qwen3_5/accept/opt/m4i8/scripts
PY=$HOME/mpk-qwen35/venv-rm/bin/python
CELLS="${CELLS:-1:288,384 8:365,461 16:720,733}"
mkdir -p "$M/stage" "$M/logs"
for CELL in $CELLS; do
  BS="${CELL%%:*}"; WIN="${CELL#*:}"
  echo "########## gap bs$BS window $WIN $(date -Is) ##########"
  timeout 7200 "$PY" -u "$S/sched_gap.py" \
      "$M/prof/raw_bs${BS}_rep0.npz" "$M/prof/meta_bs${BS}_rep0.json" \
      "$M/prof/task_names.json" --graph "$M/kernel_bs${BS}/task_graph_rank0.json" \
      --window "$WIN" --iters 2 --sim --out "$M/stage/gap_bs${BS}.json"
  echo "  rc=$?"
done
echo "M4I8_DERIVE_DONE $(date -Is)"
