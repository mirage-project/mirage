#!/usr/bin/env bash
# M4-I5 prediction P6: the megakernel's SHARED register budget at each
# moe_n_splits.
#
# Why it has to be checked: MPK compiles ONE __global__ (`persistent_kernel`,
# __launch_bounds__(256,1)) with every task body inlined, so ptxas allocates a
# SINGLE register budget and a SINGLE per-thread stack frame for all of them
# (M3-I6a found the attention accumulator spilling and taxing dense-fp8 and GDN,
# neither of which it touches).  `moe_n_splits` shrinks the routed GEMM's
# per-task NUM_N_BLOCKS, hence its accumulator, so the budget must not get
# WORSE -- and if it got worse that would be an alternative explanation for
# M3-I8's bs8 grid-widen regression that has nothing to do with waves.
#
# Compile-only: no CUDA API call, no launch, no GPU claim.  Flags are lifted
# verbatim from m3i6a/scripts/mk_ptxas.sh, which lifted them from the JIT
# command in the run logs.
set -u
export PATH=/usr/local/cuda-12.8/bin:$PATH
M=$HOME/mpk-qwen35/m4i5
T=$HOME/mpk-qwen35/mirage-m4i5
S=/var/tmp/m4i5_sweep
OUT=$M/mkptxas
mkdir -p "$OUT"

INC="-I/usr/include/python3.12 -I$T/include -I$T/include/mirage/persistent_kernel \
 -I$T/deps/cutlass/include -I$T/deps/cutlass/tools/util/include \
 -I$T/deps/json/include"

for K in 2 4 8; do
  KDIR=$S/kernel_B_k${K}_bs1
  CU=$KDIR/test_rank0.cu
  if [ ! -f "$CU" ]; then echo "=== moe_n_splits=$K: no $CU, skipped ==="; continue; fi
  echo "=== moe_n_splits=$K  ($CU) ==="
  nvcc "$CU" -O3 -lineinfo $INC \
    -DMAX_WORKER_PER_SCHEDULER=1 -DMIRAGE_USE_CUTLASS_KERNEL=0 \
    -gencode=arch=compute_100a,code=sm_100a \
    -DMPK_ENABLE_TMA -DMIRAGE_GRACE_BLACKWELL \
    -DMPK_TARGET_CC=100 -DMIRAGE_BACKEND_USE_CUDA -DMODE_OFFLINE \
    -DMPK_MAX_NUM_BATCHED_REQUESTS=1 -DMPK_MAX_NUM_BATCHED_TOKENS=16 \
    -DMPK_MAX_NUM_PAGES=64 -DMPK_PAGE_SIZE=256 -DMPK_MAX_SEQ_LENGTH=353 \
    -shared -std=c++17 -rdc=false -use_fast_math -lcuda -lcudart -lstdc++fs \
    -Xcompiler=-fPIC --expt-relaxed-constexpr -Xptxas -v \
    -o "$OUT/k${K}.so" > "$OUT/k${K}.log" 2>&1
  echo "  rc=$?"
  grep -E "Compiling entry function '_Z17persistent_kernel|Function properties for _Z17persistent_kernel|bytes stack frame|bytes spill|Used [0-9]+ registers" \
    "$OUT/k${K}.log" | head -12 | sed 's/^/    /'
  # the routed GEMM's own template args, so the split is visible in the source
  grep -o "moe_fp8_blockscale_task_impl<[^;]*" "$CU" | head -2 | sed 's/^/    TPL: /'
  rm -f "$OUT/k${K}.so"
done
echo "MK_PTXAS_DONE $(date -Is)"
