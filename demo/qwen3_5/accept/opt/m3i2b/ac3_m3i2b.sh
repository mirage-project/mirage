#!/usr/bin/env bash
# M3-I2b correctness gate. Run under the GPU guard, arm already installed.
#
#   bash gpu_guard_m3i2b.sh <cands> -- bash ac3_m3i2b.sh <arm>
#
# 1. focused oracle: the touched code path (quantize_fp8_layer row_partition),
#    2-D and 3-D, byte-compared against BOTH the whole-tensor grid and the
#    PyTorch reference primitive, inside the real compile->dispatch pipeline;
# 2. full AC-3 sweep at all five batch sizes, per-case byte-diff vs the
#    committed M2 dumps (results/dumps_final) + the AC-3 report itself;
# 3. Qwen3-8B CI as a smoke test.
set -uo pipefail
ARM=${1:?arm}
M=$HOME/mpk-qwen35/m3i2b
MIRAGE=$HOME/mpk-qwen35/mirage
export PATH=/usr/local/cuda-12.8/bin:$PATH
export HF_HOME=$HOME/mpk-qwen35/hf
export PYTHONUNBUFFERED=1
PY=$HOME/mpk-qwen35/venv-mpk/bin/python
DUMPS=$M/ac3_dumps_$ARM
mkdir -p "$DUMPS" "$M/logs"

echo "##### 0. arm=$ARM sha256"
sha256sum "$MIRAGE/python/mirage/mpk/persistent_kernel.py" \
          "$MIRAGE/python/mirage/mpk/models/qwen3_5/builder.py"

echo "##### 1. ORACLE: quantize row_partition (2-D + 3-D) $(date -Is)"
cd "$MIRAGE"
$PY -u tests/runtime_python/blackwell/sm100_linear_fp8_blockscale/test_linear_fp8_blockscale_testmode.py \
  > "$M/logs/${ARM}_oracle_quantize.log" 2>&1
ORC=$?
echo "##### oracle rc=$ORC"; tail -12 "$M/logs/${ARM}_oracle_quantize.log"
if [ "$ORC" -ne 0 ]; then echo "ORACLE FAILED -- stopping before AC-3" >&2; exit "$ORC"; fi

echo "##### 2. AC-3 SWEEP all bs $(date -Is)"
cd "$MIRAGE/demo/qwen3_5/accept"
for BS in 1 2 4 8 16; do
  echo "##### sweep bs=$BS $(date -Is)"
  $PY -u mpk_engine_run.py --batch-size $BS \
    --out-dir "$DUMPS" \
    --kernel-dir "$M/kernel_${ARM}_ac3_bs$BS" \
    --max-seq-length 132 > "$M/logs/${ARM}_ac3_bs${BS}.log" 2>&1
  echo "##### rc=$? bs=$BS $(date -Is)"; tail -3 "$M/logs/${ARM}_ac3_bs${BS}.log"
done

echo "##### 3. AC-3 GATE $(date -Is)"
$PY -u harness/run_ac3.py --engine-dump-dir "$DUMPS" --batch-sizes 1,2,4,8,16 \
  --output-json "$M/run_report_${ARM}.json" > "$M/logs/${ARM}_ac3_gate.log" 2>&1
echo "##### ac3 rc=$?"; tail -20 "$M/logs/${ARM}_ac3_gate.log"

echo "##### 4. PER-CASE BYTE DIFF vs committed dumps_final $(date -Is)"
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

echo "##### 5. Qwen3-8B CI smoke $(date -Is)"
cd "$MIRAGE/demo/qwen3"
$PY -u demo.py --use-mirage --max-new-tokens 50 \
  --max-num-batched-tokens 8 --max-num-batched-requests 1 \
  --output-dir "$M/ci_kernel_$ARM" \
  --save-tokens "$M/ci_${ARM}.json" > "$M/logs/${ARM}_ci.log" 2>&1
echo "##### ci rc=$?"; grep -E "per-token latency|Error|Traceback" "$M/logs/${ARM}_ci.log" | tail -5
echo "##### AC3_M3I2B_DONE arm=$ARM $(date -Is)"
