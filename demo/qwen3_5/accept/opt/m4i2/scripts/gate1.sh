#!/usr/bin/env bash
# M4-I2 GATE 1 -- the dense-path unit/oracle instruments, BOTH nvcc flag lanes.
#
# Lane naming: "nofastmath" / "fastmath". The MEGAKERNEL ships -use_fast_math
# (persistent_kernel.py), so a gate run only without it would not be validating
# the shipped arithmetic.
#
# 1a STANDALONE (primary): scripts/bitexact_standalone.cu, torch-free, built by
#    the SHIPPED toolchain (nvcc 12.8 -- the megakernel JIT takes nvcc off PATH
#    and every driver here pins 12.8). Golden vs ferret-fast, whole projection,
#    bitwise, 6 shipped shapes x 5 bs x 2 data regimes.
# 1b PYBIND (secondary): tests/runtime_python/blackwell/sm100_linear_fp8_blockscale.
#    This box's torch is 2.13.0+cu130 and torch.utils.cpp_extension refuses a
#    CUDA major mismatch, so this harness can only build under nvcc 13.0. It is
#    still run in both flag lanes because it carries the pre-existing tolerance
#    test and the real-checkpoint test, but 1a is what certifies 12.8.
set -uo pipefail
B=$HOME/mpk-qwen35
D=${D:-$B/mirage-m4i2}
M=${M:-$B/m4i2}
PY=$B/venv-rm/bin/python
T=$D/tests/runtime_python/blackwell/sm100_linear_fp8_blockscale
S=$D/demo/qwen3_5/accept/opt/m4i2/scripts
export HF_HOME=$B/hf PYTHONUNBUFFERED=1
mkdir -p "$M/gate1"

echo "### GPU claim: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}  $(date -Is)"
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader \
  | tee "$M/gate1/gpu_before.txt"

INC="-I$D/include/mirage/persistent_kernel -I$D/include/mirage/persistent_kernel/tasks -I$D/include -I$D/deps/cutlass/include -I$D/deps/cutlass/tools/util/include"
DEFS="-DMIRAGE_GRACE_BLACKWELL -DMIRAGE_BACKEND_USE_CUDA -DMIRAGE_FINGERPRINT_USE_CUDA -DMPK_TARGET_CC=100"

echo
echo "############ 1a STANDALONE (nvcc 12.8, the shipped toolchain) ############"
NVCC128=/usr/local/cuda-12.8/bin/nvcc
$NVCC128 --version | tail -2
for lane in nofastmath fastmath; do
  FM=""; [ "$lane" = fastmath ] && FM="-use_fast_math"
  echo "--- lane $lane  (flags: -O3 -arch=sm_100a $FM) ---"
  $NVCC128 -O3 -std=c++17 -arch=sm_100a $DEFS $INC $FM \
      --expt-relaxed-constexpr -Xptxas -v \
      "$S/bitexact_standalone.cu" -o "$M/gate1/bitexact_$lane" \
      > "$M/gate1/build_standalone_$lane.log" 2>&1
  rc=$?; echo "BUILD_EXIT=$rc"
  if [ $rc -ne 0 ]; then tail -30 "$M/gate1/build_standalone_$lane.log"; continue; fi
  "$M/gate1/bitexact_$lane" > "$M/gate1/standalone_$lane.log" 2>&1
  echo "RUN_EXIT=$?"
  grep -E 'GATE1_STANDALONE|regime [ER]:' "$M/gate1/standalone_$lane.log"
  grep -c 'BIT-EXACT' "$M/gate1/standalone_$lane.log" | sed 's/^/  BIT-EXACT lines: /'
  grep 'DIFF' "$M/gate1/standalone_$lane.log" | head -8
done

echo
echo "############ 1b PYBIND harness (nvcc 13.0, torch-matched) ############"
run_pybind () {
  local lane="$1" fmenv="$2"
  echo "--- lane $lane ---"
  cd "$T" || return 1
  rm -rf build ./*.so 2>/dev/null
  if [ -n "$fmenv" ]; then export FP8BS_TEST_FAST_MATH=1; else unset FP8BS_TEST_FAST_MATH; fi
  CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH \
    "$PY" setup.py build_ext --inplace > "$M/gate1/build_pybind_$lane.log" 2>&1
  local rc=$?; echo "BUILD_EXIT=$rc"
  if [ $rc -ne 0 ]; then tail -25 "$M/gate1/build_pybind_$lane.log"; return 1; fi
  "$PY" test_linear_fp8_blockscale_bitexact.py > "$M/gate1/pybind_bitexact_$lane.log" 2>&1
  echo "BITEXACT_EXIT=$?"; tail -4 "$M/gate1/pybind_bitexact_$lane.log"
  "$PY" test_linear_fp8_blockscale.py > "$M/gate1/pybind_unit_$lane.log" 2>&1
  echo "UNIT_EXIT=$?"; tail -4 "$M/gate1/pybind_unit_$lane.log"
  if [ -n "${QWEN35_SNAPSHOT:-}" ]; then
    "$PY" test_linear_fp8_blockscale_ckpt.py > "$M/gate1/pybind_ckpt_$lane.log" 2>&1
    echo "CKPT_EXIT=$?"; tail -4 "$M/gate1/pybind_ckpt_$lane.log"
  else
    echo "CKPT: SKIPPED (QWEN35_SNAPSHOT unset)"
  fi
}
run_pybind nofastmath ""
run_pybind fastmath   "1"

echo
echo "=== GATE1_DONE $(date -Is) ==="
