#!/usr/bin/env bash
# M3-I6a compile-only resource probe. NO GPU claim, NO launch.
# Flags mirror the megakernel JIT (python/mirage/mpk/persistent_kernel.py
# _get_compile_command, target_cc=100): -O3 -use_fast_math -lineinfo
# -gencode=arch=compute_100a,code=sm_100a -DMPK_ENABLE_TMA
# -DMIRAGE_GRACE_BLACKWELL -DMODE_OFFLINE -DMIRAGE_BACKEND_USE_CUDA
# -DMPK_TARGET_CC=100 --expt-relaxed-constexpr, plus -Xptxas -v.
set -u
export PATH=/usr/local/cuda-12.8/bin:$PATH
T=$HOME/mpk-qwen35/mirage-i6a
S=$HOME/mpk-qwen35/i6a
OUT=$S/regprobe
mkdir -p "$OUT"

INC="-I$T/include -I$T/include/mirage/persistent_kernel \
 -I$T/include/mirage/persistent_kernel/tasks \
 -I$T/deps/cutlass/include -I$T/deps/cutlass/tools/util/include \
 -I$T/deps/json/include"

BASE="-O3 -std=c++17 -lineinfo -gencode=arch=compute_100a,code=sm_100a \
 -DMPK_ENABLE_TMA -DMIRAGE_GRACE_BLACKWELL -DMIRAGE_BACKEND_USE_CUDA \
 -DMIRAGE_FINGERPRINT_USE_CUDA -DMPK_TARGET_CC=100 -DMODE_OFFLINE \
 -DMPK_MAX_NUM_BATCHED_REQUESTS=16 -DMPK_MAX_NUM_BATCHED_TOKENS=16 \
 -DMPK_MAX_NUM_PAGES=1024 -DMPK_PAGE_SIZE=256 -DMPK_MAX_SEQ_LENGTH=897 \
 -DMAX_WORKER_PER_SCHEDULER=1 -DMIRAGE_USE_CUTLASS_KERNEL=0 \
 --expt-relaxed-constexpr -Xcudafe --diag_suppress=177"

echo "### nvcc: $(nvcc --version | tail -2 | head -1)"
echo "### tree: $T ($(cd $T && git log --oneline -1))"
echo

for FM in fast nofast; do
  EXTRA=""
  [ "$FM" = "fast" ] && EXTRA="-use_fast_math"
  for Q in 1 2 3 4 6 8; do
    TAG="${FM}_q${Q}"
    nvcc $BASE $EXTRA -DI6A_QPASS=$Q $INC -cubin \
      "$S/tu_i6a_attn.cu" -o "$OUT/$TAG.cubin" \
      -Xptxas -v > "$OUT/$TAG.log" 2>&1
    RC=$?
    echo "=== Q_PASS=$Q  fastmath=$FM  rc=$RC ==="
    if [ "$RC" -ne 0 ]; then
      grep -E "error|Error" "$OUT/$TAG.log" | head -4
    fi
    grep -E "Function properties for|used [0-9]+ registers|stack frame|spill|smem" \
      "$OUT/$TAG.log" | sed 's/^/    /' | head -12
    rm -f "$OUT/$TAG.cubin"
  done
done
echo
echo "REGPROBE_DONE $(date -Is)"
