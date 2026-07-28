#!/bin/bash
# Oracle gate in BOTH nvcc flag lanes. The SHIPPED megakernel uses
# -use_fast_math, so the split-vs-golden equality must hold there; the
# no-fast-math lane is the one the oracle's own EXACT-ORDER references were
# built for, so it is the clean all-PASS control.
set -uo pipefail
export PATH=/usr/local/cuda-12.8/bin:$PATH
export HF_HOME=/home/muhengl/mpk-qwen35/hf
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${1:-7}"
VENV=/home/muhengl/mpk-qwen35/venv-rm
D=/home/muhengl/mpk-qwen35/mirage-rm/tests/runtime_python/blackwell/sm100_gdn_recurrent
OUT=/home/muhengl/mpk-qwen35/m3i3
mkdir -p "$OUT"

run_lane () {
  local lane="$1"; local fm="$2"
  cd "$D"
  rm -rf build *.so
  GDN_TEST_FAST_MATH="$fm" "$VENV/bin/python" setup.py build_ext --inplace \
      > "$OUT/build_${lane}.log" 2>&1
  echo "=== LANE $lane (GDN_TEST_FAST_MATH=$fm) build rc=$? ==="
  grep -c "use_fast_math" "$OUT/build_${lane}.log" | sed "s/^/  use_fast_math occurrences: /"
  "$VENV/bin/python" test_gdn_recurrent_oracle.py > "$OUT/oracle_${lane}.log" 2>&1
  echo "  oracle rc=$?"
  echo "  --- split arms (must ALL be BIT-EXACT) ---"
  grep -E "^split=" "$OUT/oracle_${lane}.log"
  echo "  --- summary ---"
  grep -cE "BIT-EXACT" "$OUT/oracle_${lane}.log" | sed "s/^/  bit-exact lines: /"
  tail -2 "$OUT/oracle_${lane}.log"
  echo
  # unit gate too
  "$VENV/bin/python" test_gdn_recurrent.py > "$OUT/unit_${lane}.log" 2>&1
  echo "  unit rc=$?  $(tail -1 "$OUT/unit_${lane}.log")"
  echo "  unit FAILs: $(grep -c '    FAIL' "$OUT/unit_${lane}.log")"
  echo
}

run_lane nofastmath 0
run_lane fastmath 1
echo "ALL LANES DONE"
