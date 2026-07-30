#!/usr/bin/env bash
# M4-I6: re-run M4-I5's OWN critical-path decomposer on both arms' profiled raws.
#
# WHY. The stage table says the router's WALLSPAN fell 448.6 us at bs1 while the
# STEP fell only 322.0 us -- a 72% pass-through. That gap needs a mechanism, not a
# shrug (first-principles: a result you cannot explain you cannot trust). Two
# candidate mechanisms:
#   (i)  the saving is real but the step is not the path -- M4-I5 measured
#        step/cp = 1.23-1.47x, so scheduling slack absorbs part of any path
#        saving; or
#   (ii) the router was NOT 100% on the path for all 40 layers, so part of its
#        wallspan reduction was never on the critical chain to begin with.
# Re-deriving the path with M4-I5's cp_decompose.py on BOTH arms distinguishes
# them: it reports the router's path tasks and path microseconds directly, so
# arm A's number can be checked against M4-I5's recorded 842.1 us and arm B's is
# the measured after.
#
# CPU-only: reads the saved .npz/.json from stage_wallspan.sh, no GPU claim.
set -uo pipefail
export PYTHONUNBUFFERED=1
TA=${TA:-$HOME/mpk-qwen35/mirage-m4i6-base}
TB=${TB:-$HOME/mpk-qwen35/mirage-m4i6}
PY=$HOME/mpk-qwen35/venv-rm/bin/python
M=${M:-/var/tmp/m4i6_prof}
OUT=${OUT:-$HOME/mpk-qwen35/m4i6/critpath}
BSLIST="${BSLIST:-1 8 16}"
# M4-I5's scripts live in the tree; the ferret per-call floors are the pinned
# basis both it and this use.
S5=$TB/demo/qwen3_5/accept/opt/m4i5/scripts
FERRET=$TB/demo/qwen3_5/accept/opt/m3i10/ferret_targets.json
mkdir -p "$OUT"

echo "########## M4-I6 critical-path re-decomposition  $(date -Is) ##########"
[ -f "$S5/width.py" ] || { echo "REFUSING: $S5/width.py missing"; exit 3; }
[ -f "$FERRET" ] || { echo "REFUSING: $FERRET missing"; exit 3; }

for BS in $BSLIST; do
  for ARM in A B; do
    T="$TB"; [ "$ARM" = A ] && T="$TA"
    OD=$M/prof_${ARM}
    RAW=$OD/raw_bs${BS}_rep0.npz
    META=$OD/meta_bs${BS}_rep0.json
    NAMES=$OD/task_names.json
    KD=$M/kernel_prof_${ARM}_bs${BS}
    GRAPH=$KD/task_graph_rank0.json
    for f in "$RAW" "$META" "$NAMES" "$GRAPH"; do
      [ -f "$f" ] || { echo "  ${ARM}_bs${BS}: MISSING $f -- skipped"; continue 2; }
    done
    echo "--- ${ARM}_bs${BS} ---"
    "$PY" -u "$S5/width.py" "$RAW" "$META" "$NAMES" --graph "$GRAPH" \
        --out "$OUT/width_${ARM}_bs${BS}.json" > "$OUT/width_${ARM}_bs${BS}.log" 2>&1
    echo "  width rc=$?"
    "$PY" -u "$S5/cp_decompose.py" "$GRAPH" "$OUT/width_${ARM}_bs${BS}.json" \
        --names "$NAMES" --ferret "$FERRET" --weight levelmax \
        --out "$OUT/cp_${ARM}_bs${BS}.json" > "$OUT/cp_${ARM}_bs${BS}.log" 2>&1
    echo "  cp_decompose rc=$?"
    "$PY" - "$OUT/cp_${ARM}_bs${BS}.json" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
cp = d.get("cp_us") or d.get("cp_max_us") or d.get("path_us")
print(f"    cp = {cp}")
comp = d.get("composition") or d.get("by_stage") or []
if isinstance(comp, dict):
    comp = [dict(task_type=k, **v) for k, v in comp.items()]
for r in comp:
    tt = r.get("task_type") or r.get("stage")
    if tt and "TOPK_SOFTMAX" in tt:
        print(f"    ROUTER on path: tasks={r.get('path_tasks') or r.get('n')} "
              f"us={r.get('us_on_path') or r.get('total_us')} "
              f"share={r.get('share') or r.get('pct')}")
EOF
  done
done
echo "CRITPATH_M4I6_DONE $(date -Is)"
