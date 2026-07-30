#!/usr/bin/env bash
# M4-I9 GATE 3 -- FULL AC-3 at all five batch sizes with ARM S enabled.
#
# The pinned harness is harness/gate_ac3_stable.sh: 10 pinned prompts, msl 132,
# 64 new tokens, a COLD kernel compile for every rep, >=REPS fingerprint-identical
# reps per bs, and per-case token byte-identity against the committed baseline
# results/dumps_final. Quarantined reps are kept and replaced, never deleted.
#
# THE CLAIM THIS GATE TESTS IS THE STRONG ONE. Arm F merges the MoE activation
# SwiGLU into its quantize. Gate 2 already proved the merged task byte-identical
# to the shipped pair at the kernel level in both nvcc lanes, so the expected
# result here is byte-identical dumps at every bs -- through 40 real layers, on
# the real checkpoint, with the megakernel's own -use_fast_math. Under the
# re-pinned AC-3 (goal.md 2026-07-29) bit-exactness is a diagnostic rather than
# the pass condition, but a NON-identical dump would mean the construction
# argument is wrong somewhere the unit test does not reach, and the arm would
# have to be withdrawn rather than waived.
#
# It also exercises the one thing the unit test cannot: the GRAPH change. Arm F
# emits one fewer op per layer, a new task type, and a task at a finer grid, so
# this gate covers the annotated-graph rewrite, the event fan-in counts and the
# task-to-worker assignment as well as the arithmetic.
set -uo pipefail
export PATH=/usr/local/cuda-12.8/bin:$PATH
export HF_HOME=$HOME/mpk-qwen35/hf
export PYTHONUNBUFFERED=1
GPU="${CUDA_VISIBLE_DEVICES:?the guard must pin CUDA_VISIBLE_DEVICES}"

T=${T:-$HOME/mpk-qwen35/mirage-m4i9}
ACC=$T/demo/qwen3_5/accept
export MPK_ACCEPT_DIR=$ACC
export PYTHONPATH=$T/python
PY=${PY:-$HOME/mpk-qwen35/venv-rm/bin/python}
M=${M:-/var/tmp/m4i9_ac3}
REPS=${REPS:-3}
BSLIST=${BSLIST:-1,2,4,8,16}
export MPK_FUSE_SILU_QUANT=${MPK_FUSE_SILU_QUANT:-1}
mkdir -p "$M"

echo "########## M4-I9 AC-3 (arm S)  gpu=$GPU  $(date -Is) ##########"
echo "tree: $T  HEAD=$(git -C "$T" rev-parse --short HEAD)"
echo "fused kernel: $(sha256sum "$T/include/mirage/persistent_kernel/tasks/blackwell/moe_silu_mul_quantize_fp8_sm100.cuh")"
echo "core.so: $(md5sum "$T"/python/mirage/core.cpython-*.so)"
"$PY" -c "import sys;sys.path.insert(0,'$T/python');import mirage,os;print('mirage from',os.path.realpath(mirage.__file__))"
echo "MPK_FUSE_SILU_QUANT=$MPK_FUSE_SILU_QUANT"
df -BG /var/tmp | tail -1

echo
echo "===== stage 1: the pinned cold-rep stability gate ====="
PY="$PY" bash "$ACC/harness/gate_ac3_stable.sh" \
    --out "$M/sweep" --reps "$REPS" --batch-sizes "$BSLIST" \
    --python "$PY" 2>&1 | tail -60
echo "GATE_AC3_STABLE_EXIT=${PIPESTATUS[0]}"

echo
echo "===== stage 1 verdict + per-rep record ====="
"$PY" - "$M/sweep/gate_ac3_stable.json" <<'PYEOF'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception as e:
    print("could not read the gate report:", e); raise SystemExit(0)
print("verdict:", d.get("verdict"), " binding:", d.get("binding"))
for k in ("reps_required", "divergence_rate", "quarantined", "lost_reps",
          "scored_reps", "tokens_verdict"):
    if k in d:
        print(f"  {k}: {d[k]}")
per = d.get("per_bs") or d.get("by_bs") or {}
for bs, v in sorted(per.items(), key=lambda x: int(str(x[0]).lstrip('bs'))):
    if isinstance(v, dict):
        print(f"  bs{bs}: " + ", ".join(
            f"{kk}={vv}" for kk, vv in v.items()
            if kk in ("verdict", "accepted", "attempts", "quarantined",
                      "tokens_identical", "cases_identical", "fingerprint")))
PYEOF

echo
echo "===== stage 2: the re-pinned three-part report ====="
ASSM="$M/assembled"
rm -rf "$ASSM"; mkdir -p "$ASSM"
for BS in 1 2 4 8 16; do
  for d in "$M"/sweep/reps/bs${BS}_r*; do
    [ -f "$d/bs${BS}.json" ] || continue
    cp -f "$d/bs${BS}.json" "$ASSM/" 2>/dev/null
    cp -f "$d/timings_bs${BS}.json" "$ASSM/" 2>/dev/null
    echo "  bs$BS <- $(basename "$d")"
    break
  done
done
ls "$ASSM" | sed 's/^/    /'
if ls "$ASSM"/bs*.json >/dev/null 2>&1; then
  "$PY" -u "$ACC/harness/run_ac3.py" --engine-dump-dir "$ASSM" \
      --batch-sizes 1,2,4,8,16 --output-json "$M/run_report_m4i9.json" 2>&1 | tail -30
  echo "RUN_AC3_EXIT=${PIPESTATUS[0]}"
  if [ -f "$ACC/opt/m4i4/scripts/ac3_repin_report.py" ]; then
    "$PY" -u "$ACC/opt/m4i4/scripts/ac3_repin_report.py" \
        --tree "m4i9=$ASSM" --out-json "$M/repin_m4i9.json" 2>&1 | tail -45
    echo "REPIN_EXIT=${PIPESTATUS[0]}"
  fi
else
  echo "NO DUMPS FOUND -- stage 2 cannot run; see stage 1 output"
fi
echo "AC3_M4I9_DONE $(date -Is)"
df -BG /var/tmp | tail -1
