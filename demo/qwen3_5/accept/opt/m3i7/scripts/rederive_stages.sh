#!/usr/bin/env bash
# M3-I7 -- re-derive the per-stage MPK-vs-vLLM comparison at integrated HEAD.
#
# Runs OFF the GPU box: the profiler raws are archived under
# /home/catalyst/mpk-artifacts/m3i7/late_raw/ and everything below is CPU work
# over those buffers, using the SAME committed pipeline (parse_profile.py,
# concurrency.py, analyze_armA.py, build_comparison_armA.py) so the result is
# reproducible from the archived npz alone.
#
# WHY THE BASIS MOVED FROM msl=353 TO msl=897 (the substantive finding):
# comparison_by_stage.csv's MPK column is a ONE-STEP measurement taken inside
# the "steady window" schedule_sim picks. At msl=353 (M3-I10 arm A's basis) no
# prefill-free window wider than FIVE live requests exists at bs8 or bs16 --
# the first request retires (iteration 112) long before the last finishes
# prefilling (iteration 128 at bs8, 256 at bs16). Enumerating every prefill-free
# run of the exact admission replay:
#
#   msl=353  bs8 : widest = [160,167) 5 live      bs16: widest = [336,344) 5 live
#   msl=897  bs8 : widest = [170,656) 8 live      bs16: widest = [720,733) 12 live
#              bs1: [16,656) 1 live (the whole decode)
#
# So msl=353 cannot express a bs8 or bs16 decode step at all, and the committed
# armA capture in fact fell through parse_profile's `hi <= lo` guard to the
# last-eight-iterations fallback -- its bs8/bs16 attribution rows record
# regime (1 live, 0 prefill, 1 decode, 1 token) and tokens_per_step=1. msl=897
# gives a genuine full-width 8-live decode step at bs8 and 12/16 at bs16, and
# it samples the context band (556-896) the vLLM reference table was itself
# captured at -- which is exactly the argument M3-I10 already accepted for the
# attention row. This issue applies it to every stage instead of one.
#
# WINDOW ALIGNMENT: per-stage wall spans come from concurrency.py, which
# measures ONE iteration at the midpoint of the raw steady window; step_us comes
# from parse_profile.py, which measures [lo+warm, lo+warm+steady). Left at the
# defaults those are different iterations at different contexts, so the totals
# row would divide a late-context numerator by an early-context denominator.
# WARM below centres parse's window on concurrency's midpoint, per bs.
set -uo pipefail
ACC="${ACC:-/home/catalyst/project/demo/qwen3_5/accept}"
RAW="${RAW:-/home/catalyst/mpk-artifacts/m3i7/late_raw}"
BOX="${BOX:-/home/catalyst/mpk-artifacts/m3i7/box}"
W="${W:-/home/catalyst/mpk-artifacts/m3i7/stage}"
OPT=$ACC/opt
PY="${PY:-python3}"
mkdir -p "$W/armL/tables" "$W/armL/meta" "$W/armL/meta_noprof" "$W/qc" "$W/logs"

# per-bs: midpoint of the widest prefill-free window, and the warm offset that
# centres a 96-iteration parse window on it (derived by scripts/window_plan.py,
# which prints the enumeration above and is re-run below as a live assertion).
$PY "$OPT/m3i7/scripts/window_plan.py" --msl 897 --prompt-len 256 --mbt 16 \
    --span 96 --out "$W/window_plan.json" | tee "$W/logs/window_plan.txt"

for BS in 1 8 16; do
  WARM=$($PY -c "import json;print(json.load(open('$W/window_plan.json'))['$BS']['warm_iters'])")
  SPAN=$($PY -c "import json;print(json.load(open('$W/window_plan.json'))['$BS']['span'])")
  echo "===== bs$BS  warm=$WARM span=$SPAN ====="
  ( cd "$OPT" && $PY -u parse_profile.py \
      --raw "$RAW/raw_bs${BS}_rep0.npz" \
      --meta "$BOX/prof/prof_Alate/meta_bs${BS}_rep0.json" \
      --names "$BOX/prof/prof_Alate/task_names.json" \
      --out-prefix "$W/armL/tables/bs${BS}" \
      --warm-iters "$WARM" --steady-iters "$SPAN" --no-perfetto \
    ) > "$W/armL/tables/bs${BS}_parse.log" 2>&1
  echo "  parse rc=$?"
  ( cd "$OPT" && $PY -u concurrency.py \
      "$RAW/raw_bs${BS}_rep0.npz" \
      "$BOX/prof/prof_Alate/meta_bs${BS}_rep0.json" \
      "$BOX/prof/prof_Alate/task_names.json" \
      "$W/armL/tables/bs${BS}_concurrency.json" ) > "$W/armL/tables/bs${BS}_conc.log" 2>&1
  echo "  conc  rc=$?"
  $PY - "$W/armL/tables/bs${BS}_attrib.json" "$W/armL/tables/bs${BS}_concurrency.json" <<'EOF'
import json, sys
s = json.load(open(sys.argv[1]))["summary"]
c = json.load(open(sys.argv[2]))
print(f"  step_us={s['step_us']:.1f} window={s['steady_window']} "
      f"regime={s['steady_regime_live_prefill_decode_tokens']} "
      f"agrees={s['schedule_model_agrees']} | conc iteration={c['iteration']} "
      f"step_us={c['step_us']:.1f} regime={c['regime']}")
EOF
  cp -f "$BOX/prof/prof_Alate/meta_bs${BS}_rep0.json" "$W/armL/meta/"
  for R in 0 1 2; do
    f=$BOX/prof/noprof_Alate/meta_bs${BS}_rep${R}.json
    [ -f "$f" ] && cp -f "$f" "$W/armL/meta_noprof/"
  done
done

cp -f "$OPT/m3i10/remeasure/scripts/analyze_armA.py" "$W/armL/analyze.py"
( cd "$W/armL" && $PY analyze.py ) | tee "$W/logs/analyze.txt"

M3I10RM_DIR="$W" M3I10RM_ARM=armL \
  M3I10RM_VLLM_TABLES="$OPT/m3i10/tables" \
  $PY "$OPT/m3i10/remeasure/scripts/build_comparison_armA.py" | tee "$W/logs/comparison.txt"

echo "=== outputs ==="
ls "$W/armL" "$W/armL_m3i10/tables" 2>/dev/null
