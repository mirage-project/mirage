#!/usr/bin/env bash
# M3-I9 validation plan -- WRITTEN, NOT ARMED.
#
# All 8 B200s are contended.  M3-I2b owns the next window and M3-I8 the one
# after; this is third.  The coordinator sequences windows, so this script
# refuses to start until it is explicitly armed AND both prior windows have
# released their locks:
#
#     M3I9_ARMED=1 nohup bash ~/mpk-qwen35/m3i9/plan_m3i9.sh \
#         > ~/mpk-qwen35/m3i9/logs/plan.log 2>&1 &
#
# Stage order is VALUE order and, unusually, the first two stages need NO source
# change at all -- the issue's central mechanism claim is settled before anyone
# compiles a modified runtime:
#
#  0  BASELINE + THE FREE FALSIFIER.  Re-run the bs16 AC-3 wave exactly as
#     shipped.  `dup_checks.first_divergence` is now recorded, and
#     predictions.md registers six per-slot bounds (60/54/46/35/19/- for slots
#     10..15).  Compaction predicts them; the competing "different prefill
#     chunking" explanation predicts ~0 for all six.  Costs one ordinary run.
#  1  NEGATIVE CONTROL.  Same wave at --max-seq-length 212, where the replay
#     says zero reported windows straddle a migration.  The six duplicate pairs
#     must come back identical:true.  One megakernel JIT, no source change.
#  2  COST-LAW CHECK.  --slot-order sorted-padded at bs16, no rebuild.  The
#     model says 179 iterations / 4214 ms (+11.4%).  This is the only cheap test
#     of the `a + b*max_chunk + c*n_live` law OUTSIDE the data it was fit on --
#     if it misses, every predicted policy delta in the ranking is suspect and
#     the runtime change should not be built yet.
#  3  THE RUNTIME KNOB.  Build with MPK_MAX_TOKENS_PER_REQUEST (README.md
#     "the runtime knob"), default = MPK_MAX_NUM_BATCHED_TOKENS.  Codegen
#     identity first: default-off must regenerate the pre-M3-I9 graph byte for
#     byte.  Then cap = max(1, mbt/bs).
#  4  AC-3 GATE on the cap: bs16 first (the one that changes), then the full
#     sweep + per-case byte diff vs the committed report at e51cb86.  The six
#     duplicate pairs are predicted to flip to identical:true -- nothing else in
#     the tree has ever moved them.
#  5  PERF bs16 base vs cap, >=3 reps, the pinned statistical rule.  Predicted
#     203 -> 131 iterations, 4566.5 -> 2825 ms unprofiled (+61.6%).
#  6  PERF bs 1/2/4/8 -- the predicted SMALL LOSSES (-0.0/-0.6/-2.2/-2.7%).
#     They are part of the claim; a large loss there falsifies the cost law as
#     surely as a missing win at bs16.
#  7  MATCHED GEOMETRY (remeasure-protocol.md): 256/1024, msl=1280, base and
#     cap, all five bs.  Feeds M4.  Runs LAST because it is the longest and it
#     depends on stages 4-6 having passed.
#  8  analysis + backlog.json update.
set -uo pipefail

M=$HOME/mpk-qwen35/m3i9
REPO=${REPO:-$HOME/mpk-qwen35/mirage}
ACC=$REPO/demo/qwen3_5/accept
CANDS=${CANDS:-6,0,1,2,3,4,5,7}
STAGES=${STAGES:-0,1,2,3,4,5,6,7,8}
REPS=${REPS:-3}
export PATH=/usr/local/cuda-12.8/bin:$PATH
PY=$HOME/mpk-qwen35/venv-mpk/bin/python
BASE_SHA=e51cb86603380057eb94488f5171d99bc85cf959

# ------------------------------------------------------------------ arming
if [ "${M3I9_ARMED:-0}" != "1" ]; then
  cat >&2 <<'EOF'
REFUSING: M3-I9's window is not armed.

This capture is prepared and parked.  M3-I2b owns the next GPU window and M3-I8
the one after; the coordinator sequences them.  Re-run with M3I9_ARMED=1 once
both have completed and their arms have been removed from the shared clone.
EOF
  exit 98
fi

# ------------------------------------------------------------- interlocks
for L in "$HOME/mpk-qwen35/.gpu-locks/"M3-I2b.lock "$HOME/mpk-qwen35/.gpu-locks/"M3-I8.lock; do
  if [ -f "$L" ]; then
    echo "REFUSING: $L still exists -- that issue's window has not been released." >&2
    exit 98
  fi
done
for P in "$HOME/mpk-qwen35/m3i2b/logs/plan.pid" "$HOME/mpk-qwen35/m3i8/logs/plan.pid"; do
  if [ -f "$P" ] && kill -0 "$(cat "$P")" 2>/dev/null; then
    echo "REFUSING: $(dirname "$(dirname "$P")") plan is still running (pid $(cat "$P"))." >&2
    exit 98
  fi
done
# The prior windows stage ARMS into the shared clone.  A leftover arm would make
# every number here a measurement of somebody else's change.
if ! git -C "$REPO" diff --quiet || ! git -C "$REPO" diff --cached --quiet; then
  echo "REFUSING: $REPO has uncommitted changes -- a prior window's arm may still be staged." >&2
  git -C "$REPO" status --short >&2
  exit 98
fi
echo "clone HEAD: $(git -C "$REPO" rev-parse HEAD)"
mkdir -p "$M/logs" "$M/out" "$M/kernels"
echo $$ > "$M/logs/plan.pid"
trap 'rm -f "$M/logs/plan.pid"' EXIT

has() { case ",$STAGES," in *,$1,*) return 0;; *) return 1;; esac; }
guard() { bash "$ACC/opt/m3i9/gpu_guard_m3i9.sh" "$CANDS" -- "$@"; }

run_ac3() {   # run_ac3 <tag> <bs> <msl> [extra args...]
  local tag=$1 bs=$2 msl=$3; shift 3
  guard "$PY" "$ACC/mpk_engine_run.py" \
      --batch-size "$bs" --max-seq-length "$msl" --max-new-tokens 64 \
      --reference "$ACC/reference/reference_outputs.json" \
      --kernel-dir "$M/kernels/bs${bs}_msl${msl}" --reuse-kernel \
      --out-dir "$M/out/$tag" --dump-name "bs${bs}.json" "$@"
}

# ---------------------------------------------------------------- stage 0
if has 0; then
  echo "=== stage 0: shipped baseline + the free falsifier (bs16) $(date -Is)"
  run_ac3 s0_base 16 132
  "$PY" "$ACC/opt/m3i9/analyze_m3i9.py" --check-divergence "$M/out/s0_base/timings_bs16.json" \
      | tee "$M/logs/s0_divergence.txt"
fi

# ---------------------------------------------------------------- stage 1
if has 1; then
  echo "=== stage 1: negative control, msl=212 (predicted 293 iters, 0 straddles) $(date -Is)"
  run_ac3 s1_msl212 16 212
  "$PY" "$ACC/opt/m3i9/analyze_m3i9.py" --check-isolation "$M/out/s1_msl212/timings_bs16.json" \
      | tee "$M/logs/s1_isolation.txt"
fi

# ---------------------------------------------------------------- stage 2
if has 2; then
  echo "=== stage 2: cost-law check, --slot-order sorted-padded (predicted 4214 ms) $(date -Is)"
  for r in $(seq 1 "$REPS"); do
    run_ac3 "s2_sorted_rep$r" 16 132 --slot-order sorted-padded
  done
  "$PY" "$ACC/opt/m3i9/analyze_m3i9.py" --check-costlaw "$M/out" | tee "$M/logs/s2_costlaw.txt"
fi

# ---------------------------------------------------------------- stage 3+
if has 3; then
  cat <<'EOF'
=== stage 3: the runtime knob
  NOT SCRIPTED.  It is a source change (opt/m3i9/README.md, "the runtime knob":
  one define, one min(), one Python kwarg) and it must be reviewed and landed by
  the issue owner, not materialised by a capture script.  Land it, then:
    - codegen identity: default-off must regenerate the pre-M3-I9 task graph
      byte for byte (diff task_graph_rank0.json)
    - re-run stages 4-7 with STAGES=4,5,6,7,8
EOF
  exit 0
fi

# ---------------------------------------------------------------- stage 4
if has 4; then
  echo "=== stage 4: AC-3 gate under the cap $(date -Is)"
  run_ac3 s4_cap16 16 132 --per-request-token-cap 1
  for bs in 1 2 4 8; do run_ac3 "s4_cap$bs" "$bs" 132 --per-request-token-cap auto; done
  "$PY" "$ACC/harness/run_ac3.py" --engine-dump-dir "$M/out" \
      --reference "$ACC/reference/reference_outputs.json" \
      --out "$M/out/run_report_cap.json" | tee "$M/logs/s4_ac3.txt"
  "$PY" "$ACC/opt/m3i9/analyze_m3i9.py" --byte-diff "$M/out/run_report_cap.json" \
      --against "$ACC/results/run_report_all_bs.json" --base-sha "$BASE_SHA" \
      | tee "$M/logs/s4_bytediff.txt"
fi

# ---------------------------------------------------------------- stage 5/6
if has 5 || has 6; then
  echo "=== stage 5/6: perf, base vs cap, $REPS reps $(date -Is)"
  BSL=""; has 5 && BSL="16"; has 6 && BSL="$BSL 1 2 4 8"
  for bs in $BSL; do
    for arm in base cap; do
      for r in $(seq 1 "$REPS"); do
        if [ "$arm" = base ]; then
          run_ac3 "s5_${arm}_bs${bs}_rep$r" "$bs" 132
        else
          run_ac3 "s5_${arm}_bs${bs}_rep$r" "$bs" 132 --per-request-token-cap auto
        fi
      done
    done
  done
  "$PY" "$ACC/opt/m3i9/analyze_m3i9.py" --perf "$M/out" --reps "$REPS" | tee "$M/logs/s5_perf.txt"
fi

# ---------------------------------------------------------------- stage 7
if has 7; then
  echo "=== stage 7: matched geometry 256/1024 (remeasure-protocol.md) $(date -Is)"
  "$PY" "$ACC/opt/m3i9/make_synthetic_prompts.py" --input-len 256 --out "$M/out/synthetic256.jsonl"
  for bs in 1 2 4 8 16; do
    for arm in base cap; do
      for r in $(seq 1 "$REPS"); do
        EXTRA=""; [ "$arm" = cap ] && EXTRA="--per-request-token-cap auto"
        guard "$PY" "$ACC/mpk_engine_run.py" \
            --batch-size "$bs" --max-seq-length 1280 --max-new-tokens 1024 \
            --prompts-file "$M/out/synthetic256.jsonl" \
            --reference "$ACC/reference/reference_outputs.json" \
            --kernel-dir "$M/kernels/bs${bs}_msl1280" --reuse-kernel \
            --out-dir "$M/out/s7_${arm}_bs${bs}_rep$r" --dump-name "bs${bs}.json" $EXTRA
      done
    done
  done
  "$PY" "$ACC/opt/m3i9/analyze_m3i9.py" --matched "$M/out" --reps "$REPS" \
      --vllm "$ACC/baselines/vllm-0.25.1-20260725" | tee "$M/logs/s7_matched.txt"
fi

# ---------------------------------------------------------------- stage 8
if has 8; then
  echo "=== stage 8: analysis $(date -Is)"
  "$PY" "$ACC/opt/m3i9/analyze_m3i9.py" --all "$M/out" --predictions "$ACC/opt/m3i9/predictions.md" \
      | tee "$M/logs/s8_summary.txt"
fi
echo "=== plan complete $(date -Is)"
