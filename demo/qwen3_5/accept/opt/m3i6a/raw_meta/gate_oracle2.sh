#!/usr/bin/env bash
# M3-I6a: rebuild the wrapper (now with the ungated max_tokens=2/1 cases) and
# re-run the HF oracle at the candidate pass size.
set -uo pipefail
export PATH=/usr/local/cuda-12.8/bin:$PATH
export HF_HOME=$HOME/mpk-qwen35/hf
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${1:-6}}"
T=$HOME/mpk-qwen35/mirage-i6a
Dd=$T/tests/runtime_python/blackwell/sm100_attention_qwen35
OUT=$HOME/mpk-qwen35/i6a/gates
PY=$HOME/mpk-qwen35/venv-mpk/bin/python
ORACLE=$HOME/mpk-qwen35/oracle-work/dumps
cd "$Dd" || exit 1
rm -rf build ./*.so
PYTHONPATH=$T/python $PY setup.py build_ext --inplace > "$OUT/unit_build2.log" 2>&1
echo "rebuild rc=$?"; tail -3 "$OUT/unit_build2.log"
for MT in 4 2 1; do
  PYTHONPATH=$T/python $PY test_attention_qwen35_oracle.py --oracle-dir "$ORACLE" \
      --max-tokens "$MT" --q-pass "$MT" --out "$OUT/oracle_mt${MT}.json" \
      > "$OUT/oracle_mt${MT}.log" 2>&1
  echo "=== oracle max_tokens=$MT q_pass=$MT rc=$? ==="
  grep -E "^\[(decode|prefill)\]" "$OUT/oracle_mt${MT}.log" | head -24
  tail -3 "$OUT/oracle_mt${MT}.log"
done
echo "--- cross-arm identity: does the ungated/gated kernel output MATCH across pass sizes? ---"
$PY - "$OUT" <<'PYEOF'
import json, sys, os
base = sys.argv[1]
ref = None
for mt in (4, 2, 1):
    p = os.path.join(base, f"oracle_mt{mt}.json")
    if not os.path.exists(p):
        print(f"mt={mt}: MISSING"); continue
    d = json.load(open(p))
    rows = {r["case"]: (r["bit_identical"], round(r["max_abs"], 12))
            for r in d["rows"]}
    print(f"mt={mt}: rows={len(rows)} failures={len(d.get('failures', []))}")
    if ref is None:
        ref = rows; refmt = mt; continue
    diff = {k: (ref.get(k), v) for k, v in rows.items() if ref.get(k) != v}
    print(f"  vs mt={refmt}: {len(diff)} row(s) with a different verdict/max_abs")
    for k, v in list(diff.items())[:8]:
        print(f"    {k}: mt{refmt}={v[0]} mt{mt}={v[1]}")
PYEOF
echo "GATE_ORACLE2_DONE $(date -Is)"
