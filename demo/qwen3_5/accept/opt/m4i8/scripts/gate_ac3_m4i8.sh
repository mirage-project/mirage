#!/usr/bin/env bash
# M4-I8 GATE 3 -- FULL AC-3 at all five batch sizes with ARM S enabled.
#
# The pinned harness is harness/gate_ac3_stable.sh: 10 pinned prompts, msl 132,
# 64 new tokens, a COLD kernel compile for every rep, >=REPS fingerprint-identical
# reps per bs, and per-case token byte-identity against the committed baseline
# results/dumps_final. Quarantined reps are kept and replaced, never deleted.
#
# THE CLAIM THIS GATE TESTS IS THE STRONG ONE. Arm S changes the memory SCOPE of
# one load in a spin-wait -- ld.acquire.sys -> ld.acquire.gpu on a local event
# counter. It changes no arithmetic, no dtype, no task-to-data mapping, and no
# task ordering; the value read is the same monotone counter and the predicate is
# the same. So the expected result is byte-identical dumps at every bs, exactly
# as under the M4-I7 precedent for a change that only redistributes work, and the
# per-case byte diff is the evidence. Under the re-pinned AC-3 (goal.md
# 2026-07-29) bit-exactness is a diagnostic rather than the pass condition, but a
# NON-identical dump here would mean the scope argument is wrong and the arm has
# to be withdrawn, not waived.
set -uo pipefail
export PATH=/usr/local/cuda-12.8/bin:$PATH
export HF_HOME=$HOME/mpk-qwen35/hf
export PYTHONUNBUFFERED=1
GPU="${CUDA_VISIBLE_DEVICES:?the guard must pin CUDA_VISIBLE_DEVICES}"

T=${T:-$HOME/mpk-qwen35/mirage-m4i8}
ACC=$T/demo/qwen3_5/accept
export MPK_ACCEPT_DIR=$ACC
export PYTHONPATH=$T/python
PY=${PY:-$HOME/mpk-qwen35/venv-rm/bin/python}
M=${M:-/var/tmp/m4i8_ac3}
REPS=${REPS:-3}
BSLIST=${BSLIST:-1,2,4,8,16}
export MPK_EVENT_WAIT_GPU_SCOPE=${MPK_EVENT_WAIT_GPU_SCOPE:-1}
mkdir -p "$M"

echo "########## M4-I8 AC-3 (arm S)  gpu=$GPU  $(date -Is) ##########"
echo "tree: $T  HEAD=$(git -C "$T" rev-parse --short HEAD)"
echo "worker loop: $(sha256sum "$T/include/mirage/persistent_kernel/persistent_kernel.cuh")"
"$PY" -c "import sys;sys.path.insert(0,'$T/python');import mirage,os;print('mirage from',os.path.realpath(mirage.__file__))"
echo "MPK_EVENT_WAIT_GPU_SCOPE=$MPK_EVENT_WAIT_GPU_SCOPE  MPK_WORKER_OOO_POP=${MPK_WORKER_OOO_POP:-<unset>}"
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
      --batch-sizes 1,2,4,8,16 --output-json "$M/run_report_m4i8.json" 2>&1 | tail -30
  echo "RUN_AC3_EXIT=${PIPESTATUS[0]}"
  if [ -f "$ACC/opt/m4i4/scripts/ac3_repin_report.py" ]; then
    "$PY" -u "$ACC/opt/m4i4/scripts/ac3_repin_report.py" \
        --tree "m4i8=$ASSM" --out-json "$M/repin_m4i8.json" 2>&1 | tail -45
    echo "REPIN_EXIT=${PIPESTATUS[0]}"
  fi
else
  echo "NO DUMPS FOUND -- stage 2 cannot run; see stage 1 output"
fi
echo "AC3_M4I8_DONE $(date -Is)"
df -BG /var/tmp | tail -1
