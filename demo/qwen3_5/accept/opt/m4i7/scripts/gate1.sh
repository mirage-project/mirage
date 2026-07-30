#!/usr/bin/env bash
# M4-I7 gate 1: build + run the torch-free bit-exact harness in BOTH nvcc flag
# lanes with the SHIPPED JIT toolchain (12.8) and shipped budget macros.
set -uo pipefail
D=$HOME/mpk-qwen35/mirage-m4i7
M=$HOME/mpk-qwen35/m4i7
OUT=${OUT:-$M/gate1}
export PATH=/usr/local/cuda-12.8/bin:$PATH
mkdir -p "$OUT"
INC="-I$D/include/mirage/persistent_kernel -I$D/include/mirage/persistent_kernel/tasks -I$D/include"
# Exactly the macro set persistent_kernel.py passes for the shipped megakernel.
DEFS="-DMIRAGE_GRACE_BLACKWELL -DMPK_TARGET_CC=100 -DMODE_OFFLINE -DMIRAGE_BACKEND_USE_CUDA"
rc=0
for lane in nofast fastmath; do
  EXTRA=""
  [ "$lane" = fastmath ] && EXTRA="-use_fast_math"
  echo "=== build lane=$lane $(date -Is) ==="
  nvcc -O3 -std=c++17 -gencode=arch=compute_100a,code=sm_100a \
       --expt-relaxed-constexpr $DEFS $EXTRA $INC \
       -Xptxas -v \
       -o "$OUT/bitexact_$lane" "$M/scripts/bitexact_standalone.cu" \
       > "$OUT/build_$lane.log" 2>&1
  b=$?
  echo "BUILD_EXIT=$b"
  if [ $b -ne 0 ]; then tail -40 "$OUT/build_$lane.log"; rc=1; continue; fi
  grep -E "registers|spill|smem" "$OUT/build_$lane.log" | head -20
  echo "=== run lane=$lane ==="
  "$OUT/bitexact_$lane" > "$OUT/run_$lane.log" 2>&1
  r=$?
  echo "RUN_EXIT=$r"
  tail -4 "$OUT/run_$lane.log"
  grep -c "MISMATCH\|FAIL" "$OUT/run_$lane.log"
  [ $r -ne 0 ] && rc=1
done
echo "GATE1_RC=$rc"
exit $rc
