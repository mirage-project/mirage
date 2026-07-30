#!/usr/bin/env bash
# M4-I7 GATE 3 -- FULL AC-3 at all five batch sizes on the integrated tree.
#
# The pinned harness is harness/gate_ac3_stable.sh: 10 pinned prompts, msl 132,
# 64 new tokens, a COLD kernel compile for every rep, >=REPS fingerprint-identical
# reps per bs, and per-case token byte-identity against the committed baseline
# results/dumps_final. Quarantined reps are kept and replaced, never deleted.
#
# Under the RE-PINNED AC-3 (goal.md, 2026-07-29) the pass condition is coherence
# + a >=90% agreement floor, with bit-exactness REPORTED as a diagnostic rather
# than required. The claim this issue makes is the STRONG one -- the ferret port
# is bit-exact against the golden path (Gate 1: 160 arms, both nvcc flag
# lanes, both data regimes, PATH 0/1/2 and the dispatcher at the MPK geometry)
# -- so the expected result is byte-identical dumps, and the per-case byte diff
# is the evidence. If a case is byte-identical to the ADJUDICATED baseline then
# coherence and agreement transfer by identity and no HF model load is owed;
# anything that is NOT byte-identical owes a real measurement, which
# ac3_repin_report.py reports as NOT-EVALUATED rather than assuming.
#
# Arm: DEFAULT (the shipped configuration). MPK_MOE_BLOCKSCALE_BASELINE is deliberately
# unset -- the baseline arm's generated megakernel source is byte-identical to the
# pre-M4-I7 tree's (verified: same TU sha256), so it is the committed state and
# needs no re-litigation here.
set -uo pipefail
export PATH=/usr/local/cuda-12.8/bin:$PATH
export HF_HOME=$HOME/mpk-qwen35/hf
export PYTHONUNBUFFERED=1
GPU="${CUDA_VISIBLE_DEVICES:?the guard must pin CUDA_VISIBLE_DEVICES}"

T=${T:-$HOME/mpk-qwen35/mirage-m4i7}
ACC=$T/demo/qwen3_5/accept
export MPK_ACCEPT_DIR=$ACC
export PYTHONPATH=$T/python
PY=${PY:-$HOME/mpk-qwen35/venv-rm/bin/python}
M=${M:-/var/tmp/m4i7_ac3}
REPS=${REPS:-3}
BSLIST=${BSLIST:-1,2,4,8,16}
mkdir -p "$M"

echo "########## M4-I7 AC-3 (shipped arm)  gpu=$GPU  $(date -Is) ##########"
echo "tree: $T  HEAD=$(git -C "$T" rev-parse --short HEAD)"
echo "tasks 241/242 kernel:"; sha256sum "$T/include/mirage/persistent_kernel/tasks/blackwell/moe_fp8_blockscale_sm100.cuh"
"$PY" -c "import sys;sys.path.insert(0,'$T/python');import mirage,os;print('mirage from',os.path.realpath(mirage.__file__))"
echo "MPK_MOE_BLOCKSCALE_BASELINE=${MPK_MOE_BLOCKSCALE_BASELINE:-<unset, shipped arm>}"
df -BG /var/tmp | tail -1

echo
echo "===== stage 1: the pinned cold-rep stability gate ====="
PY="$PY" bash "$ACC/harness/gate_ac3_stable.sh" \
    --out "$M/sweep" --reps "$REPS" --batch-sizes "$BSLIST" \
    --python "$PY" 2>&1 | tail -60
echo "GATE_AC3_STABLE_EXIT=${PIPESTATUS[0]}"

echo
echo "===== stage 1 verdict + per-rep record ====="
"$PY" - "$M/sweep/gate_ac3_stable.json" <<'EOF'
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
EOF

echo
echo "===== stage 2: the re-pinned three-part report ====="
# The stability gate writes one dump tree per accepted rep under reps/<tag>/.
# Every accepted rep at a bs is fingerprint-identical, so any accepted rep is a
# valid representative; assemble a bs -> first-accepted-rep tree and score that.
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
      --batch-sizes 1,2,4,8,16 --output-json "$M/run_report_m4i7.json" \
      2>&1 | tail -30
  echo "RUN_AC3_EXIT=${PIPESTATUS[0]}"
  if [ -f "$ACC/opt/m4i4/scripts/ac3_repin_report.py" ]; then
    "$PY" -u "$ACC/opt/m4i4/scripts/ac3_repin_report.py" \
        --tree "m4i7=$ASSM" --out-json "$M/repin_m4i7.json" 2>&1 | tail -45
    echo "REPIN_EXIT=${PIPESTATUS[0]}"
  fi
else
  echo "NO DUMPS FOUND -- stage 2 cannot run; see stage 1 output"
fi
echo "AC3_M4I7_DONE $(date -Is)"
df -BG /var/tmp | tail -1
