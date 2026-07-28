#!/usr/bin/env bash
# M3-I6a gate 1: the attention unit / oracle / test-mode instruments.
#
# The pass-size change is claimed to be BIT-EXACT, not merely close: a query
# row's softmax/accumulate depends only on its own q and the KV stream, and the
# KV stream is replayed identically per pass, so splitting the rows differently
# cannot change any row's arithmetic or its order.  `test_attention_qwen35_qloop`
# is the pre-existing instrument for exactly that claim (§B pass-size invariance
# at the Qwen3.5 shape); M3-I6a adds §B2 for the MULTI-pass regime the production
# change actually moves through (an mbt=16 chunk goes 4 passes -> 8 passes).
#
# $1 = CUDA device index (caller holds the GPU lock).
set -uo pipefail
export PATH=/usr/local/cuda-12.8/bin:$PATH
export HF_HOME=$HOME/mpk-qwen35/hf
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${1:-6}"
T=$HOME/mpk-qwen35/mirage-i6a
D=$T/tests/runtime_python/blackwell/sm100_attention_qwen35
OUT=$HOME/mpk-qwen35/i6a/gates
VENV=$HOME/mpk-qwen35/venv-mpk
PY=$VENV/bin/python
ORACLE=$HOME/mpk-qwen35/oracle-work/dumps
mkdir -p "$OUT"

cd "$D" || exit 1
echo "=== build the attention wrapper extension $(date -Is) ==="
rm -rf build ./*.so
PYTHONPATH=$T/python $PY setup.py build_ext --inplace > "$OUT/unit_build.log" 2>&1
echo "build rc=$?"; tail -4 "$OUT/unit_build.log"; ls -la ./*.so 2>&1 | head -3

echo
echo "=== A. q-loop pass-size equivalence (bit-exactness gate) ==="
PYTHONPATH=$T/python $PY test_attention_qwen35_qloop.py \
    --out "$OUT/qloop_result.json" > "$OUT/qloop.log" 2>&1
echo "qloop rc=$?"
grep -E "^qwen35" "$OUT/qloop.log" | head -60
echo "--- summary ---"
tail -6 "$OUT/qloop.log"
$PY - "$OUT/qloop_result.json" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
rows = d["rows"]
bad = [r for r in rows if r["bit_identical"] != r["expected_match"]]
print(f"qloop rows={len(rows)} mismatched_expectation={len(bad)} "
      f"failures={len(d['failures'])}")
mp = [r for r in rows if r["case"].startswith("qwen35-multipass")]
print(f"  new B2 multipass rows={len(mp)} "
      f"all_bit_identical={all(r['bit_identical'] for r in mp)}")
for r in bad:
    print("  BAD:", r["case"], r["bit_identical"], r["expected_match"], r["max_abs"])
EOF

echo
echo "=== B. HF oracle, control (max_tokens/q_pass = 4) vs candidate (= 2) ==="
for MT in 4 2; do
  QP=$MT
  PYTHONPATH=$T/python $PY test_attention_qwen35_oracle.py \
      --oracle-dir "$ORACLE" --max-tokens "$MT" --q-pass "$QP" \
      --out "$OUT/oracle_mt${MT}.json" > "$OUT/oracle_mt${MT}.log" 2>&1
  echo "--- oracle max_tokens=$MT q_pass=$QP rc=$? ---"
  tail -14 "$OUT/oracle_mt${MT}.log"
done

echo
echo "=== C. test-mode pipeline (full MPK compile + dispatch), q_pass 4 and 2 ==="
for QP in 4 2; do
  rm -rf test_output_attention_qwen35 2>/dev/null
  ATTN_TESTMODE_Q_PASS=$QP PYTHONPATH=$T/python timeout 1800 $PY \
      test_attention_qwen35_testmode.py > "$OUT/testmode_qp${QP}.log" 2>&1
  echo "--- testmode q_pass=$QP rc=$? ---"
  tail -12 "$OUT/testmode_qp${QP}.log"
done
echo "GATE_UNIT_DONE $(date -Is)"
