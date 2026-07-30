#!/usr/bin/env bash
# M4-I9 REGISTER GATE -- the megakernel's SHARED register budget, before/after
# the SwiGLU+quantize fusion, plus a proof the fused task actually landed.
#
# WHY THIS IS THE GATE THAT CAN VOID THE ISSUE: MPK compiles ONE __global__
# (`persistent_kernel`, __launch_bounds__(256,1)) with EVERY task body inlined,
# so ptxas allocates a SINGLE register budget and a SINGLE stack frame for all
# of them. HEAD already sits at 255 registers. Fusion RAISES per-task register
# pressure by construction (two task bodies' live ranges overlap), and a fusion
# that spills taxes every stage it does not touch -- M3-I6a is the precedent
# (attention's Q_PASS=4 accumulator was the ONLY spiller and cost the whole
# step ~9-10%), and M4-I6 handed back 21% of its own win the same way.
#
# UNLIKE M4-I8's arms this is NOT a -D: the fusion changes the GENERATED TU (a
# new task type, one fewer op), so the two arms are two DIFFERENT TUs and both
# must be compiled with identical flags and identical geometry. The register
# probe therefore cannot precede the implementation; it is the first gate run
# after it, before any e2e or AC-3 time is spent.
#
# Compile-only: no CUDA API call, no launch, no GPU claim.
set -u
export PATH=/usr/local/cuda-12.8/bin:$PATH
M=${M:-$HOME/mpk-qwen35/m4i9}
S=${S:-/var/tmp/m4i9_sweep}
T=${T:-$HOME/mpk-qwen35/mirage-m4i9}
BS=${BS:-1}
OUT=$M/ptxas
mkdir -p "$OUT"

inc () {
  echo "-I/usr/include/python3.12 -I$T/include -I$T/include/mirage/persistent_kernel \
 -I$T/deps/cutlass/include -I$T/deps/cutlass/tools/util/include -I$T/deps/json/include"
}
report () {
  grep -E "Compiling entry function '_Z17persistent_kernel|bytes stack frame|bytes spill|Used [0-9]+ registers|bytes smem" \
    "$1" | sed 's/^/    /'
}
one () {              # $1 = label, $2 = kernel dir
  local LBL="$1" KD="$2"; shift 2
  local CU="$KD/test_rank0.cu"
  echo "=== $LBL  TU=$CU ==="
  [ -f "$CU" ] || { echo "  MISSING TU"; return 1; }
  sha256sum "$CU" | sed 's/^/    sha256: /'
  wc -l < "$CU" | sed 's/^/    TU lines: /'
  nvcc "$CU" -O3 -lineinfo $(inc) "$@" \
    -DMAX_WORKER_PER_SCHEDULER=1 -DMIRAGE_USE_CUTLASS_KERNEL=0 \
    -gencode=arch=compute_100a,code=sm_100a \
    -DMPK_ENABLE_TMA -DMIRAGE_GRACE_BLACKWELL \
    -DMPK_TARGET_CC=100 -DMIRAGE_BACKEND_USE_CUDA -DMODE_OFFLINE \
    -DMPK_MAX_NUM_BATCHED_REQUESTS=$BS -DMPK_MAX_NUM_BATCHED_TOKENS=16 \
    -DMPK_MAX_NUM_PAGES=64 -DMPK_PAGE_SIZE=256 -DMPK_MAX_SEQ_LENGTH=353 \
    -shared -std=c++17 -rdc=false -use_fast_math -lcuda -lcudart -lstdc++fs \
    -Xcompiler=-fPIC --expt-relaxed-constexpr -Xptxas -v \
    -o "$OUT/$LBL.so" > "$OUT/$LBL.log" 2>&1
  echo "  rc=$?"
  report "$OUT/$LBL.log"
  cuobjdump -sass "$OUT/$LBL.so" > "$OUT/$LBL.sass" 2>/dev/null
  printf "    SASS lines=%s  total_LD=%s\n" \
    "$(wc -l < "$OUT/$LBL.sass")" \
    "$(grep -cE '^\s+/\*[0-9a-f]+\*/\s+LD' "$OUT/$LBL.sass")"
  printf "    TU task-type census: 118(silu)=%s 275(quantize)=%s 243(fused)=%s\n" \
    "$(grep -c 'TASK_SILU_MUL' "$CU")" \
    "$(grep -c 'TASK_QUANTIZE_FP8_SM100' "$CU")" \
    "$(grep -c 'TASK_MOE_SILU_MUL_QUANTIZE_FP8_SM100' "$CU")"
  printf "    impl calls: silu+q=%s rms+q=%s comb+rms=%s recur+q=%s | silu=%s rms=%s quant=%s\n" \
    "$(grep -c 'moe_silu_mul_quantize_fp8_task_impl' "$CU")" \
    "$(grep -c 'rms_norm_quantize_fp8_task_impl' "$CU")" \
    "$(grep -c 'moe_mul_sum_add_rmsnorm_task_impl' "$CU")" \
    "$(grep -c 'gdn_recurrent_quantize_fp8_task_impl' "$CU")" \
    "$(grep -c 'silu_mul_task_impl<' "$CU")" \
    "$(grep -c 'rms_norm_hopper_impl' "$CU")" \
    "$(grep -c 'per_token_group_quantize_fp8_task_impl' "$CU")"
  rm -f "$OUT/$LBL.so"
}

echo "########## M4-I9 ptxas -v, bs=$BS  $(date -Is) ##########"
nvcc --version | tail -2
echo "tree: $T HEAD=$(git -C "$T" rev-parse --short HEAD)"
ARMS=${ARMS:-A F}
for a in $ARMS; do one "$a" "$S/kernel_${a}_bs${BS}"; done

echo
echo "=== summary: registers / stack / spills, both arms ==="
pk () {
  awk '/Compiling entry function ._Z17persistent_kernel/{f=1}
       f && /bytes stack frame/{sf=$0}
       f && /Used [0-9]+ registers/{print sf"|"$0; exit}' "$1"
}
for f in $ARMS; do
  [ -f "$OUT/$f.log" ] || continue
  printf "  %-3s %s\n" "$f" "$(pk "$OUT/$f.log")"
done
echo "MK_PTXAS_M4I9_DONE $(date -Is)"
