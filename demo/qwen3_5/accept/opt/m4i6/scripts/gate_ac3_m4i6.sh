#!/usr/bin/env bash
# M4-I6 GATE 3 -- FULL AC-3 at all five batch sizes on the integrated tree.
#
# The pinned harness is harness/gate_ac3_stable.sh: 10 pinned prompts, msl 132,
# 64 new tokens, a COLD kernel compile for every rep, >=REPS fingerprint-identical
# reps per bs, and per-case token byte-identity against the committed baseline
# results/dumps_final. Quarantined reps are kept and replaced, never deleted.
#
# Under the RE-PINNED AC-3 (goal.md, 2026-07-29) the pass condition is coherence
# + a >=90% agreement floor, with bit-exactness REPORTED as a diagnostic rather
# than required. This issue's claim is the STRONG one for every LIVE row -- the
# ferret harness required position-exact agreement with the frozen golden on
# topk_weights (live rows), mpk_routing_indices and mpk_active_expert_ids across
# all five configs, and gate 1 re-proved it in-repo at experts 128 and 256 with
# HF-bit-exact bf16 weights -- so the expected result is byte-identical dumps.
#
# WHERE THE CANDIDATE DELIBERATELY DIFFERS, and why AC-3 is the arbiter: rows at
# or above num_active_rows now write ZERO top-k weights instead of the softmax of
# their residue logits. Those weights ARE read -- mul_sum_add_sm100.cuh:34-45
# loops row_idx over the whole BATCH_SIZE and multiplies d_weight[row*NUM_TOPK+j]
# unconditionally -- but only into PADDING rows of moe_out, and M2's AC-3
# established live-row independence from padding rows empirically (builder.py's
# gate_padding_rows note: every prompt's 64 token ids byte-identical at bs
# 1/2/4/8/16, i.e. identical whether the other rows hold residue or 15 different
# live prompts). A byte-identical dump here is that argument's confirmation; a
# non-identical dump would falsify it and owes a root-cause, not a tolerance.
set -uo pipefail
export PATH=/usr/local/cuda-12.8/bin:$PATH
export HF_HOME=$HOME/mpk-qwen35/hf
export PYTHONUNBUFFERED=1
GPU="${CUDA_VISIBLE_DEVICES:?the guard must pin CUDA_VISIBLE_DEVICES}"

T=${T:-$HOME/mpk-qwen35/mirage-m4i6}
ACC=$T/demo/qwen3_5/accept
export MPK_ACCEPT_DIR=$ACC
export PYTHONPATH=$T/python
PY=${PY:-$HOME/mpk-qwen35/venv-rm/bin/python}
M=${M:-/var/tmp/m4i6_ac3}
REPS=${REPS:-3}
BSLIST=${BSLIST:-1,2,4,8,16}
mkdir -p "$M"

echo "########## M4-I6 AC-3  gpu=$GPU  $(date -Is) ##########"
echo "tree: $T  HEAD=$(git -C "$T" rev-parse --short HEAD)"
echo "task260 kernel:"; sha256sum "$T/include/mirage/persistent_kernel/tasks/blackwell/topk_softmax_sm100.cuh"
"$PY" -c "import sys;sys.path.insert(0,'$T/python');import mirage,os;print('mirage from',os.path.realpath(mirage.__file__))"
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
      --batch-sizes 1,2,4,8,16 --output-json "$M/run_report_m4i6.json" \
      2>&1 | tail -30
  echo "RUN_AC3_EXIT=${PIPESTATUS[0]}"
  if [ -f "$ACC/opt/m4i4/scripts/ac3_repin_report.py" ]; then
    "$PY" -u "$ACC/opt/m4i4/scripts/ac3_repin_report.py" \
        --tree "m4i6=$ASSM" --out-json "$M/repin_m4i6.json" 2>&1 | tail -45
    echo "REPIN_EXIT=${PIPESTATUS[0]}"
  fi
else
  echo "NO DUMPS FOUND -- stage 2 cannot run; see stage 1 output"
fi
echo "AC3_M4I6_DONE $(date -Is)"
df -BG /var/tmp | tail -1
