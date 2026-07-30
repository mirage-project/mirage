#!/usr/bin/env bash
# M4-I9 GATE 2c -- unit/oracle for FLAG C (the GDN recurrence's fused quantize),
# BOTH nvcc flag lanes.
#
# This flag is the one whose quantize arithmetic is HAND-WRITTEN rather than a
# call to the shared `per_token_group_quantize_fp8_task_impl`: in the recurrence
# epilogue the 128 values live one-per-thread across four warps, so the amax
# needs a block reduction instead of the impl's warp shuffle. The claim that a
# different reduction SHAPE still gives identical bytes rests on `fmaxf` being
# exact and order-independent, and this is the gate that tests it.
#
# Lane A: no -use_fast_math. Lane B: -use_fast_math (GDN_TEST_FAST_MATH=1) --
# what the megakernel ships, and it rewrites `rsqrtf`, `expf` and the divisions
# in BOTH the epilogue and the quantize.
set -uo pipefail
B=$HOME/mpk-qwen35
D=${T:-$B/mirage-m4i9}
M=$B/m4i9
PY=$B/venv-rm/bin/python
TD=$D/tests/runtime_python/blackwell/sm100_gdn_recurrent
CU=${CU:-/usr/local/cuda-13.0}       # torch cpp_extension refuses a mismatch
export PATH=$CU/bin:$PATH
export CUDA_HOME=$CU
export HF_HOME=$B/hf PYTHONUNBUFFERED=1
mkdir -p "$M/gate2c"

echo "### GPU claim: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader \
  | tee "$M/gate2c/gpu_before.txt"

run_lane () {
  local lane="$1" fmenv="$2"
  echo
  echo "############################################################"
  echo "### LANE $lane   GDN_TEST_FAST_MATH=${fmenv:-0}   $(date -Is)"
  echo "############################################################"
  cd "$TD"
  rm -rf build ./*.so 2>/dev/null
  if [ -n "$fmenv" ]; then export GDN_TEST_FAST_MATH=1; else unset GDN_TEST_FAST_MATH; fi
  "$PY" setup.py build_ext --inplace > "$M/gate2c/build_$lane.log" 2>&1
  local rc=$?
  echo "BUILD_EXIT=$rc"
  if [ $rc -ne 0 ]; then grep -iE "error" "$M/gate2c/build_$lane.log" | head -20; return 1; fi
  grep -c 'use_fast_math' "$M/gate2c/build_$lane.log" \
    | sed "s/^/  -use_fast_math occurrences in build log: /"

  echo "--- flag C: fused recurrence+quantize vs recurrence + standalone quantize ---"
  "$PY" test_gdn_recurrent_fusedq.py 2>&1 | tee "$M/gate2c/fusedq_$lane.log"
  echo "FUSEDQ_EXIT=${PIPESTATUS[0]}"

  echo "--- pre-existing split-vs-golden bit-exactness (regression check) ---"
  "$PY" test_gdn_recurrent.py > "$M/gate2c/split_$lane.log" 2>&1
  echo "SPLIT_EXIT=$?"; tail -8 "$M/gate2c/split_$lane.log"
}

run_lane nofastmath ""
run_lane fastmath   "1"

echo
echo "=== GATE2C_DONE $(date -Is) ==="
