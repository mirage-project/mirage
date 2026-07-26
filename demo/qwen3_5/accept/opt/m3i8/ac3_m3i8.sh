#!/usr/bin/env bash
# M3-I8 correctness gate. Run under the GPU guard, arm already installed.
#
#   bash gpu_guard_m3i8.sh <cands> -- bash ac3_m3i8.sh <arm>
#
# 1. MECHANISM ORACLE: read the router's own activated-expert mask
#    (mask_probe.py). The gated arm must satisfy the HARD cap
#    activated <= min(256, topk * live_rows) at every layer and every bs. That
#    is a bound, not a fit -- if it fails the change did not take effect and
#    every downstream number is meaningless.
# 2. INERT-AT-bs16 ORACLE: at bs16 every row of the mbt=16 batch is live, so
#    the gate has nothing to gate. AC-3 at bs16 must be byte-identical to base.
#    This is the cheapest possible test that the change touches ONLY padding.
# 3. full AC-3 sweep at all five batch sizes + per-case byte-diff vs the
#    committed M2 dumps (results/dumps_final) + the AC-3 report itself;
# 4. Qwen3-8B CI as a smoke test.
set -uo pipefail
ARM=${1:?arm}
M=$HOME/mpk-qwen35/m3i8
MIRAGE=$HOME/mpk-qwen35/mirage
export PATH=/usr/local/cuda-12.8/bin:$PATH
export HF_HOME=$HOME/mpk-qwen35/hf
export PYTHONUNBUFFERED=1
PY=$HOME/mpk-qwen35/venv-mpk/bin/python
DUMPS=$M/ac3_dumps_$ARM
mkdir -p "$DUMPS" "$M/logs" "$M/masks"

echo "##### 0. arm=$ARM sha256"
sha256sum "$MIRAGE/python/mirage/mpk/persistent_kernel.py" \
          "$MIRAGE/python/mirage/mpk/models/qwen3_5/builder.py"
grep -n "^MOE_GATE_PADDING_ROWS\|self.moe_n_splits" \
     "$MIRAGE/python/mirage/mpk/models/qwen3_5/builder.py"

echo "##### 1. MECHANISM ORACLE: activated-expert mask per bs $(date -Is)"
FAILED=0
for BS in 1 2 4 8 16; do
  $PY -u "$MIRAGE/demo/qwen3_5/accept/opt/m3i8/mask_probe.py" \
      --batch-size $BS --out "$M/masks/mask_${ARM}_bs${BS}.json" \
      --kernel-dir "$M/kernel_${ARM}_mask_bs$BS" \
      > "$M/logs/${ARM}_mask_bs${BS}.log" 2>&1
  RC=$?
  echo "##### mask bs=$BS rc=$RC"
  grep -E "activated_mean|activated_max|hard_cap|cap_respected|rows_marked" \
       "$M/logs/${ARM}_mask_bs${BS}.log" | head -8
  [ "$RC" -ne 0 ] && FAILED=1
done
if [ "$FAILED" -ne 0 ]; then
  echo "MECHANISM ORACLE FAILED -- stopping before AC-3" >&2; exit 94
fi

echo "##### 2. INERT-AT-bs16 ORACLE $(date -Is)"
cd "$MIRAGE/demo/qwen3_5/accept"
$PY -u mpk_engine_run.py --batch-size 16 --out-dir "$DUMPS" \
    --kernel-dir "$M/kernel_${ARM}_ac3_bs16" --max-seq-length 132 \
    > "$M/logs/${ARM}_ac3_bs16.log" 2>&1
echo "##### rc=$? bs=16"; tail -3 "$M/logs/${ARM}_ac3_bs16.log"
$PY -u $HOME/mpk-qwen35/m3i2a/bytediff.py \
    "$MIRAGE/demo/qwen3_5/accept/results/dumps_final" "$DUMPS" 16 \
    > "$M/bytediff_${ARM}_bs16.json"
$PY - <<EOF
import json, sys
d = json.load(open("$M/bytediff_${ARM}_bs16.json"))
bad = {k: v for k, v in d["per_case"].items() if v != "identical"}
print("bs16 identical:", d["identical"], "changed:", bad or "none")
if bad:
    print("INERT ORACLE FAILED: the gate moved bytes where every row is live",
          file=sys.stderr)
    sys.exit(93)
EOF
[ $? -ne 0 ] && { echo "stopping: bs16 is not inert" >&2; exit 93; }

echo "##### 3. AC-3 SWEEP remaining bs $(date -Is)"
for BS in 1 2 4 8; do
  echo "##### sweep bs=$BS $(date -Is)"
  $PY -u mpk_engine_run.py --batch-size $BS \
    --out-dir "$DUMPS" \
    --kernel-dir "$M/kernel_${ARM}_ac3_bs$BS" \
    --max-seq-length 132 > "$M/logs/${ARM}_ac3_bs${BS}.log" 2>&1
  echo "##### rc=$? bs=$BS $(date -Is)"; tail -3 "$M/logs/${ARM}_ac3_bs${BS}.log"
done

echo "##### 4. AC-3 GATE $(date -Is)"
$PY -u harness/run_ac3.py --engine-dump-dir "$DUMPS" --batch-sizes 1,2,4,8,16 \
  --output-json "$M/run_report_${ARM}.json" > "$M/logs/${ARM}_ac3_gate.log" 2>&1
echo "##### ac3 rc=$?"; tail -20 "$M/logs/${ARM}_ac3_gate.log"

echo "##### 5. PER-CASE BYTE DIFF vs committed dumps_final $(date -Is)"
$PY -u $HOME/mpk-qwen35/m3i2a/bytediff.py \
  "$MIRAGE/demo/qwen3_5/accept/results/dumps_final" \
  "$DUMPS" 1,2,4,8,16 > "$M/bytediff_${ARM}.json"
echo "##### bytediff rc=$?"
$PY - <<EOF
import json
d = json.load(open("$M/bytediff_${ARM}.json"))
print("identical:", d["identical"], "missing:", d["missing"])
print("counts:", json.dumps(d["counts"]))
bad = {k: v for k, v in d["per_case"].items() if v != "identical"}
print("CHANGED:", json.dumps(bad, indent=1) if bad else "none")
EOF

echo "##### 6. Qwen3-8B CI smoke $(date -Is)"
cd "$MIRAGE/demo/qwen3"
$PY -u demo.py --use-mirage --max-new-tokens 50 \
  --max-num-batched-tokens 8 --max-num-batched-requests 1 \
  --output-dir "$M/ci_kernel_$ARM" \
  --save-tokens "$M/ci_${ARM}.json" > "$M/logs/${ARM}_ci.log" 2>&1
echo "##### ci rc=$?"; grep -E "per-token latency|Error|Traceback" "$M/logs/${ARM}_ci.log" | tail -5
echo "##### AC3_M3I8_DONE arm=$ARM $(date -Is)"
