#!/usr/bin/env bash
# M3-I7 -- anchor-QC the re-derived capture, then regenerate ferret_targets.json
# so its PRIMARY fields are the integrated-HEAD numbers.
#
# Anchor QC first, on purpose: it checks the per-step task-type COUNTS the trace
# implies against the counts the compiled task graph actually contains, and
# refuses above --frac-err-threshold. A stage table whose task counts disagree
# with the graph is measuring the wrong window, and that is precisely the failure
# mode this issue found in the previous basis -- so the assertion runs before any
# number is published, not after.
#
# Then the generator (opt/m3i10/scripts/regenerate_ferret_v2.py, unforked) is
# pointed at this issue's comparison with --history-key history_m3i10, which
# preserves the layer being displaced while leaving history_m3i1 untouched.
set -uo pipefail
ACC="${ACC:-/home/catalyst/project/demo/qwen3_5/accept}"
RAW="${RAW:-/home/catalyst/mpk-artifacts/m3i7/late_raw}"
BOX="${BOX:-/home/catalyst/mpk-artifacts/m3i7/box}"
W="${W:-/home/catalyst/mpk-artifacts/m3i7/stage}"
OPT=$ACC/opt
PY="${PY:-python3}"
mkdir -p "$W/qc" "$W/logs"

for BS in 1 8 16; do
  WARM=$($PY -c "import json;print(json.load(open('$W/window_plan.json'))['$BS']['warm_iters'])")
  SPAN=$($PY -c "import json;print(json.load(open('$W/window_plan.json'))['$BS']['span'])")
  echo "===== anchor QC bs$BS (warm=$WARM span=$SPAN) ====="
  ( cd "$OPT" && M3I10RM_OPT_DIR="$OPT" $PY -u "$OPT/m3i10/remeasure/scripts/anchor_qc.py" \
      --raw "$RAW/raw_bs${BS}_rep0.npz" \
      --meta "$BOX/prof/prof_Alate/meta_bs${BS}_rep0.json" \
      --names "$BOX/prof/prof_Alate/task_names.json" \
      --graph "$BOX/graphs/task_graph_bs${BS}.json" \
      --out "$W/qc/armL_bs${BS}_rep0_qc.json" \
      --warm-iters "$WARM" --steady-iters "$SPAN" ) 2>&1 | tail -12
  echo "  rc=${PIPESTATUS[0]}"
done

$PY "$OPT/m3i7/scripts/qc_summary.py" --qc-dir "$W/qc" --prefix armL \
   --out "$W/qc/anchor_qc_summary.json" | tail -20

echo "===== regenerate ferret_targets.json ====="
$PY "$OPT/m3i10/scripts/regenerate_ferret_v2.py" \
  --comparison "$W/armL_m3i10/tables/comparison_by_stage.csv" \
  --qc-dir "$W/qc" \
  --qc-prefix armL \
  --qc-summary "$W/qc/anchor_qc_summary.json" \
  --latectx-dir "$W/armL/tables" \
  --single-basis \
  --history-key history_m3i10 \
  --generated-utc 2026-07-29 \
  --generator "demo/qwen3_5/accept/opt/m3i7/scripts/regen_targets.sh -> opt/m3i10/scripts/regenerate_ferret_v2.py" \
  --basis-mpk "$(cat "$ACC/opt/m3i7/basis_mpk.txt")" \
  --basis-caveat "$(cat "$ACC/opt/m3i7/basis_caveat.txt")" \
  --latectx-rel "opt/m3i7/stage/armL/ (raws: /home/catalyst/mpk-artifacts/m3i7/late_raw/)" \
  --artifact-rel "opt/m3i7/stage/ (derived tables in-tree; raw npz at /home/catalyst/mpk-artifacts/m3i7/late_raw/)" \
  --tiers-run "M3-I7 milestone gate: integrated-HEAD late-context capture (msl=897), 3 unprofiled reps + 1 profiled rep per bs" \
  | tee "$W/logs/regen_targets.txt"
