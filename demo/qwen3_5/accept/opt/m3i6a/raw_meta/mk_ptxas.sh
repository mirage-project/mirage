#!/usr/bin/env bash
# M3-I6a: ptxas resources of the ACTUAL generated megakernel at each Q_PASS.
#
# Why this and not just the isolated-TU probe: MPK compiles ONE __global__
# (`persistent_kernel`, __launch_bounds__(256,1)) with every task body inlined,
# so ptxas allocates a SINGLE register budget and a SINGLE per-thread stack
# frame for all of them.  If the attention body's Q_PASS=4 accumulator is what
# pushes that budget to the wall, then lowering Q_PASS should relieve EVERY task
# family -- which is exactly what the wave measurement shows (dense-fp8 and GDN
# wallspans dropped too, though neither kernel changed).  This measures the
# shared budget directly.
#
# Compile-only: no CUDA API call, no launch, no GPU claim.  Flags are lifted
# verbatim from the JIT command recorded in each run log.
set -u
export PATH=/usr/local/cuda-12.8/bin:$PATH
M=$HOME/mpk-qwen35/i6a
T=$HOME/mpk-qwen35/mirage-i6a
OUT=$M/mkptxas
mkdir -p "$OUT"

INC="-I/usr/include/python3.12 -I$T/include -I$T/include/mirage/persistent_kernel \
 -I$T/deps/cutlass/include -I$T/deps/cutlass/tools/util/include \
 -I$T/deps/json/include"

for QP in 4 2 1; do
  KDIR=$M/kernel_qp${QP}_bs1_msl897
  CU=$KDIR/test_rank0.cu
  if [ ! -f "$CU" ]; then echo "=== Q_PASS=$QP: no $CU, skipped ==="; continue; fi
  echo "=== Q_PASS=$QP  ($CU) ==="
  nvcc "$CU" -O3 -lineinfo $INC \
    -DMAX_WORKER_PER_SCHEDULER=1 -DMIRAGE_USE_CUTLASS_KERNEL=0 \
    -gencode=arch=compute_100a,code=sm_100a \
    -DMPK_ENABLE_TMA -DMIRAGE_GRACE_BLACKWELL \
    -DMPK_TARGET_CC=100 -DMIRAGE_BACKEND_USE_CUDA -DMODE_OFFLINE \
    -DMPK_MAX_NUM_BATCHED_REQUESTS=1 -DMPK_MAX_NUM_BATCHED_TOKENS=16 \
    -DMPK_MAX_NUM_PAGES=64 -DMPK_PAGE_SIZE=256 -DMPK_MAX_SEQ_LENGTH=897 \
    -DMPK_ENABLE_PROFILING -DMPK_PROFILER_BUFFER_ENTRIES=260000000 \
    -shared -std=c++17 -rdc=false -use_fast_math -lcuda -lcudart -lstdc++fs \
    -Xcompiler=-fPIC --expt-relaxed-constexpr -Xptxas -v \
    -o "$OUT/qp${QP}.so" > "$OUT/qp${QP}.log" 2>&1
  echo "  rc=$?"
  grep -E "Compiling entry function '_Z17persistent_kernel|Function properties for _Z17persistent_kernel|bytes stack frame|Used [0-9]+ registers" \
    "$OUT/qp${QP}.log" | head -12 | sed 's/^/    /'
  rm -f "$OUT/qp${QP}.so"
done
echo "MK_PTXAS_DONE $(date -Is)"
