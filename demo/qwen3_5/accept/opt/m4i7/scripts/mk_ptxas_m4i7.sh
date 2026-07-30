#!/usr/bin/env bash
# M4-I7 GATE 2 -- the megakernel's SHARED register budget, before vs after.
#
# WHY IT IS A GATE AND NOT A CURIOSITY: MPK compiles ONE __global__
# (`persistent_kernel`, __launch_bounds__(256,1)) with EVERY task body inlined,
# so ptxas allocates a SINGLE register budget and a SINGLE per-thread stack frame
# for all of them. M3-I6a found the attention accumulator spilling and taxing
# dense-fp8 and GDN, stages it does not touch. So a MoE win that raises register
# pressure can REGRESS every other stage, and the win would be a mirage.
#
# The v012 fast body is exactly the shape of change that does this: three
# __noinline__ PATH bodies, mbarrier state, and up to 1 KiB bulk staging. The
# __noinline__ is load-bearing -- the ferret run measured a ~20% regression on
# the UNTOUCHED legacy path when all three bodies were inlined into one dispatch
# function (codegen pollution). ptxas is where that would show.
#
# THE CONTROL IS TIGHTER HERE THAN IN M4-I2, because it can be. M4-I2's arm-A
# knob changed a TEMPLATE ARGUMENT, so its generated TU differed from HEAD's and
# a TU diff was the control. M4-I7 changes only the HEADER BODY and a -D, so the
# generated TU is byte-identical in every arm -- which means ONE TU can be
# compiled several ways and the register budget compared with nothing else moving:
#
#   head   pre-M4-I7 header, no -D          -- the BEFORE
#   armA   M4-I7 header + BASELINE=1        -- must equal head; this is what
#                                              licenses arm A as the e2e baseline
#   armB   M4-I7 header, no -D  (SHIPPED)   -- the AFTER
#   armB_p{0,1,2}  MPK_MOE_PATH_POLICY pinned -- each fetch path's own cost
#
# Compile-only: no CUDA API call, no launch, no GPU claim. Flags are lifted from
# the JIT command in the run logs (the same set m4i2/m4i5 used).
set -u
export PATH=/usr/local/cuda-12.8/bin:$PATH
M=${M:-$HOME/mpk-qwen35/m4i7}
S=${S:-/var/tmp/m4i7_sweep}
T=${T:-$HOME/mpk-qwen35/mirage-m4i7}
BASE_REV=${BASE_REV:-5e48eaab}
BS=${BS:-1}
KD=${KD:-$S/kernel_B_bs${BS}}          # any arm's dir: the TU is arm-independent
OUT=$M/ptxas
mkdir -p "$OUT"

CU="$KD/test_rank0.cu"
if [ ! -f "$CU" ]; then echo "MISSING TU: $CU"; exit 2; fi

# A parallel include tree whose ONLY difference is the pre-M4-I7 MoE header.
# Hardlinked, so it costs no space and cannot drift from the clone.
# NOTE: it must live on the SAME filesystem as $T -- /var/tmp is md0 and the
# clone is md1, and `cp -al` across filesystems cannot hardlink.
BI=$M/base_include
rm -rf "$BI"; mkdir -p "$BI"
cp -al "$T/include" "$BI/include" || { echo "hardlink copy failed"; exit 2; }
[ -f "$BI/include/mirage/persistent_kernel/persistent_kernel.cuh" ] || {
  echo "base include tree is incomplete"; exit 2; }
REL=include/mirage/persistent_kernel/tasks/blackwell/moe_fp8_blockscale_sm100.cuh
rm -f "$BI/$REL"
git -C "$T" show "$BASE_REV:$REL" > "$BI/$REL" || { echo "cannot fetch base header"; exit 2; }
echo "base header ($BASE_REV): $(sha256sum "$BI/$REL" | cut -c1-16) $(wc -l < "$BI/$REL") lines"
echo "M4-I7 header:            $(sha256sum "$T/$REL" | cut -c1-16) $(wc -l < "$T/$REL") lines"

inc () {   # $1 = include root (the clone, or the base-header tree)
  echo "-I/usr/include/python3.12 -I$1/include -I$1/include/mirage/persistent_kernel \
 -I$T/deps/cutlass/include -I$T/deps/cutlass/tools/util/include -I$T/deps/json/include"
}

report () {
  grep -E "Compiling entry function '_Z17persistent_kernel|bytes stack frame|bytes spill|Used [0-9]+ registers|bytes smem" \
    "$1" | sed 's/^/    /'
}

one () {              # $1 = label, $2 = include root, $3... = extra -D
  local LBL="$1" ROOT="$2"; shift 2
  echo "=== $LBL  (include root $ROOT; extra: ${*:-none}) ==="
  nvcc "$CU" -O3 -lineinfo $(inc "$ROOT") "$@" \
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
  rm -f "$OUT/$LBL.so"
}

echo "########## M4-I7 ptxas -v, bs=$BS  $(date -Is) ##########"
nvcc --version | tail -2
echo "ONE TU, compiled several ways: $CU"
sha256sum "$CU" | sed 's/^/  TU sha256: /'
grep -o "moe_fp8_blockscale_task_impl<[^>]*>" "$CU" | sort | uniq -c | sort -rn \
  | head -6 | sed 's/^/  TPL: /'

one head "$BI"
one armA "$T" -DMPK_MOE_BLOCKSCALE_BASELINE=1
one armB "$T"
for p in 0 1 2; do one "armB_p$p" "$T" "-DMPK_MOE_PATH_POLICY=$p"; done

echo
echo "=== summary: registers / stack / spills, all arms, one TU ==="
# Parse the persistent_kernel ENTRY block specifically. Grepping the first
# "Used N registers" in the log picks up a tiny helper (4 registers) and would
# report every arm as identical no matter what the megakernel did.
pk () {   # $1 = log, $2 = awk field expression
  awk '/Compiling entry function ._Z17persistent_kernel/{f=1}
       f && /bytes stack frame/{sf=$0}
       f && /Used [0-9]+ registers/{print sf"|"$0; exit}' "$1"
}
for f in head armA armB armB_p0 armB_p1 armB_p2; do
  [ -f "$OUT/$f.log" ] || continue
  printf "  %-9s %s\n" "$f" "$(pk "$OUT/$f.log")"
done
echo
H=$(pk "$OUT/head.log" 2>/dev/null)
A=$(pk "$OUT/armA.log" 2>/dev/null)
if [ -n "$H" ] && [ "$H" = "$A" ]; then
  echo "CONTROL PASS: armA reproduces head exactly -- the baseline knob is"
  echo "  $H"
  echo "  a faithful stand-in, which is what licenses arm A as the e2e baseline."
else
  echo "CONTROL FAIL: head='$H' armA='$A'"
fi
echo "MK_PTXAS_M4I7_DONE $(date -Is)"
