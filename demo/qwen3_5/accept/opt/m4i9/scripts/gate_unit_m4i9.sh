#!/usr/bin/env bash
# M4-I9 GATE 2 -- unit/oracle for the affected tasks, BOTH nvcc flag lanes.
#
# Lane A: no -use_fast_math (the test extension's historical default).
# Lane B: -use_fast_math via MOE_TEST_FAST_MATH=1 -- what the MEGAKERNEL ships
#         (persistent_kernel.py), so a gate run only in lane A would not be
#         validating the shipped arithmetic. It matters here specifically:
#         -use_fast_math rewrites `expf` (the SwiGLU sigmoid) and the `/` in
#         both the SwiGLU and the fp8 rescale.
#
# Instruments, per lane:
#   test_moe_silu_quant_fused.py    fused task vs the SHIPPED unfused pair,
#       byte-identical fp8 AND fp32 scales required, 8 shapes x 4 value scales.
#   test_quantize_fp8_f32scale_moe.py  the pre-existing quantize test (the half
#       the fusion absorbs), so the gate also proves the fusion did not perturb
#       the standalone task it shares a header with.
set -uo pipefail
B=$HOME/mpk-qwen35
D=${T:-$B/mirage-m4i9}
M=$B/m4i9
PY=$B/venv-rm/bin/python
TD=$D/tests/runtime_python/blackwell/sm100_fp8_moe_qwen35
# The TORCH cpp_extension build refuses a CUDA/torch mismatch, and this venv's
# torch is 2.13.0+cu130 -- so the TEST extension is built with CUDA 13.0. That
# is NOT a change of what ships: the MEGAKERNEL JIT still compiles with
# CUDA 12.8 (persistent_kernel.py + the ptxas gate). The lane that matters for
# this gate is -use_fast_math, and both arms live in the SAME TU either way.
CU=${CU:-/usr/local/cuda-13.0}
export PATH=$CU/bin:$PATH
export CUDA_HOME=$CU
export HF_HOME=$B/hf PYTHONUNBUFFERED=1
mkdir -p "$M/gate2"

echo "### GPU claim: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader \
  | tee "$M/gate2/gpu_before.txt"

run_lane () {
  local lane="$1" fmenv="$2"
  echo
  echo "############################################################"
  echo "### LANE $lane   MOE_TEST_FAST_MATH=${fmenv:-0}   $(date -Is)"
  echo "############################################################"
  cd "$TD"
  rm -rf build ./*.so 2>/dev/null
  if [ -n "$fmenv" ]; then export MOE_TEST_FAST_MATH=1; else unset MOE_TEST_FAST_MATH; fi
  "$PY" setup.py build_ext --inplace > "$M/gate2/build_$lane.log" 2>&1
  local rc=$?
  echo "BUILD_EXIT=$rc"
  if [ $rc -ne 0 ]; then tail -40 "$M/gate2/build_$lane.log"; return 1; fi
  grep -c 'use_fast_math' "$M/gate2/build_$lane.log" \
    | sed "s/^/  -use_fast_math occurrences in build log: /"
  "$CU/bin/nvcc" --version | tail -2 | sed 's/^/  /'""

  echo "--- fused SwiGLU+quantize vs the shipped unfused pair ---"
  "$PY" test_moe_silu_quant_fused.py 2>&1 | tee "$M/gate2/fused_$lane.log"
  echo "FUSED_EXIT=${PIPESTATUS[0]}"

  echo "--- pre-existing standalone fp32-scale quantize test ---"
  "$PY" test_quantize_fp8_f32scale_moe.py > "$M/gate2/quant_$lane.log" 2>&1
  echo "QUANT_EXIT=$?"; tail -12 "$M/gate2/quant_$lane.log"
}

# CONTROL for the pre-existing standalone quantize test: under -use_fast_math
# `group_max / 448.0f` becomes a reciprocal-multiply, so its fp32 scale differs
# from torch's `absmax/448` by ~4e-12 and its bit-exact assert fires -- while its
# fp8 VALUES stay 0-ULP. That is a property of the flag, not of M4-I9, and the
# claim needs a control rather than an assertion: the SAME test is run in the
# SAME lane against a PRISTINE tree (mirage-m4i8, whose tests/ is untouched by
# this issue) and must fail identically.
control_lane () {
  local CT=${CTREE:-$B/mirage-m4i8}
  echo
  echo "############################################################"
  echo "### CONTROL: pristine tree $CT, fastmath lane   $(date -Is)"
  echo "############################################################"
  echo "control tree HEAD=$(git -C "$CT" rev-parse --short HEAD)"
  git -C "$CT" status --short tests/ | head -3
  cd "$CT/tests/runtime_python/blackwell/sm100_fp8_moe_qwen35" || return 1
  rm -rf build ./*.so 2>/dev/null
  export MOE_TEST_FAST_MATH=1
  "$PY" setup.py build_ext --inplace > "$M/gate2/build_control.log" 2>&1
  echo "BUILD_EXIT=$?"
  "$PY" test_quantize_fp8_f32scale_moe.py > "$M/gate2/quant_control.log" 2>&1
  echo "CONTROL_QUANT_EXIT=$?"; tail -6 "$M/gate2/quant_control.log"
}

run_lane nofastmath ""
run_lane fastmath   "1"
control_lane

echo
echo "=== GATE2_DONE $(date -Is) ==="
