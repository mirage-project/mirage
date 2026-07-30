#!/usr/bin/env bash
# M4-I8 GATE 2 -- the megakernel's SHARED register budget, plus a SASS-level
# proof that each arm's -D actually changed the emitted instruction.
#
# WHY THE REGISTER BUDGET IS A GATE: MPK compiles ONE __global__
# (`persistent_kernel`, __launch_bounds__(256,1)) with EVERY task body inlined,
# so ptxas allocates a SINGLE register budget and a SINGLE stack frame for all
# of them. HEAD already sits at 255 registers with a 4 B spill (M4-I6's router),
# so any arm that costs registers taxes every stage it does not touch --
# M4-I6 handed back 21% of its own win exactly that way.
#
# Both arms are -D only and neither touches the generated code, so ONE TU is
# compiled four ways with nothing else moving:
#   base   no -D                                            -- the BEFORE
#   S      -DMPK_EVENT_WAIT_GPU_SCOPE=1                      -- scope fix
#   O      -DMPK_WORKER_OOO_POP=1                            -- out-of-order pop
#   SO     both
#
# GATE 2b, THE FLAG-LANDED PROOF: arm S changes one load's memory scope, which
# ptxas -v cannot show (a scope qualifier costs no registers). So the SASS is
# dumped and the STRONG.SYS / STRONG.GPU load counts are compared. If arm S does
# not remove a STRONG.SYS load, its -D never reached the compile and any e2e
# delta is something else -- the M3-I7 defect-3 failure mode in a new costume.
#
# Compile-only: no CUDA API call, no launch, no GPU claim.
set -u
export PATH=/usr/local/cuda-12.8/bin:$PATH
M=${M:-$HOME/mpk-qwen35/m4i8}
S=${S:-/var/tmp/m4i8_sweep}
T=${T:-$HOME/mpk-qwen35/mirage-m4i8}
BS=${BS:-1}
KD=${KD:-$S/kernel_A_bs${BS}}       # any arm's dir: the TU is arm-independent
OUT=$M/ptxas
mkdir -p "$OUT"
CU="$KD/test_rank0.cu"
[ -f "$CU" ] || { echo "MISSING TU: $CU"; exit 2; }

inc () {
  echo "-I/usr/include/python3.12 -I$T/include -I$T/include/mirage/persistent_kernel \
 -I$T/deps/cutlass/include -I$T/deps/cutlass/tools/util/include -I$T/deps/json/include"
}
report () {
  grep -E "Compiling entry function '_Z17persistent_kernel|bytes stack frame|bytes spill|Used [0-9]+ registers|bytes smem" \
    "$1" | sed 's/^/    /'
}
one () {              # $1 = label, $2... = extra -D
  local LBL="$1"; shift
  echo "=== $LBL  (extra: ${*:-none}) ==="
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
  printf "    SASS  LD*.STRONG.SYS=%s  LD*.STRONG.GPU=%s  total_LD=%s  lines=%s\n" \
    "$(grep -cE 'LD[A-Z.]*\.STRONG\.SYS' "$OUT/$LBL.sass")" \
    "$(grep -cE 'LD[A-Z.]*\.STRONG\.GPU' "$OUT/$LBL.sass")" \
    "$(grep -cE '^\s+/\*[0-9a-f]+\*/\s+LD' "$OUT/$LBL.sass")" \
    "$(wc -l < "$OUT/$LBL.sass")"
  rm -f "$OUT/$LBL.so"
}

echo "########## M4-I8 ptxas -v + SASS, bs=$BS  $(date -Is) ##########"
nvcc --version | tail -2
echo "ONE TU, compiled four ways: $CU"
sha256sum "$CU" | sed 's/^/  TU sha256: /'
echo "worker-loop header: $(sha256sum "$T/include/mirage/persistent_kernel/persistent_kernel.cuh" | cut -c1-16)"

one base
one S  -DMPK_EVENT_WAIT_GPU_SCOPE=1
one O  -DMPK_WORKER_OOO_POP=1
one SO -DMPK_EVENT_WAIT_GPU_SCOPE=1 -DMPK_WORKER_OOO_POP=1

echo
echo "=== summary: registers / stack / spills, all arms, one TU ==="
pk () {
  awk '/Compiling entry function ._Z17persistent_kernel/{f=1}
       f && /bytes stack frame/{sf=$0}
       f && /Used [0-9]+ registers/{print sf"|"$0; exit}' "$1"
}
for f in base S O SO; do
  [ -f "$OUT/$f.log" ] || continue
  printf "  %-5s %s\n" "$f" "$(pk "$OUT/$f.log")"
done
echo
echo "=== GATE 2b: did each -D reach the compile? ==="
SYS_BASE=$(grep -cE 'LD[A-Z.]*\.STRONG\.SYS' "$OUT/base.sass")
SYS_S=$(grep -cE 'LD[A-Z.]*\.STRONG\.SYS' "$OUT/S.sass")
echo "  base STRONG.SYS loads = $SYS_BASE ; arm S = $SYS_S"
if [ "$SYS_S" -lt "$SYS_BASE" ]; then
  echo "  ARM S FLAG LANDED: the .sys load is gone from the SASS."
else
  echo "  ARM S FLAG DID NOT LAND -- refuse to attribute any e2e delta to it."
fi
# arm O adds code, so its SASS must differ from base in size
echo "  SASS line counts: base=$(wc -l < "$OUT/base.sass") S=$(wc -l < "$OUT/S.sass") O=$(wc -l < "$OUT/O.sass") SO=$(wc -l < "$OUT/SO.sass")"
for f in S O SO; do
  if diff -q "$OUT/base.sass" "$OUT/$f.sass" >/dev/null; then
    echo "  arm $f: SASS IDENTICAL to base -- flag did not land"
  else
    echo "  arm $f: SASS differs from base ($(diff "$OUT/base.sass" "$OUT/$f.sass" | grep -c '^[<>]') changed lines)"
  fi
done
echo "MK_PTXAS_M4I8_DONE $(date -Is)"
