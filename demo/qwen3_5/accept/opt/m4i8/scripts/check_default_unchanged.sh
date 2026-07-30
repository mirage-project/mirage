#!/usr/bin/env bash
# M4-I8 GATE 1c -- is the SHIPPED DEFAULT (both knobs off) unchanged?
#
# Both arms live behind #ifdef, but arm O also renamed an index in code that is
# OUTSIDE any #ifdef: `task_ids[queue_pos]` became `task_ids[slot]`, with
# `int const slot = queue_pos;` in the #else branch.  That is semantically a
# no-op, and "semantically a no-op" is exactly the claim a reviewer should not
# have to take on trust.  So compile the same TU against (a) this tree's header
# and (b) the header at HEAD before M4-I8, with NO -D, and diff the SASS.
# The header carries 31 assert() calls, each embedding __LINE__, and the #ifdef
# blocks shift those line numbers -- so the SASS is NOT expected to be
# byte-identical.  The gate is therefore: every difference must be an immediate
# constant (an __LINE__ value), and the persistent_kernel entry's registers,
# barriers, stack frame, smem and spills must be identical.  Any opcode,
# control-flow or scheduling difference fails.
#
# The base header is materialised into a hardlinked parallel include tree so it
# costs no space and cannot drift (pattern from mk_ptxas_m4i7.sh).  The tree MUST
# be on the same filesystem as the clone -- cp -al cannot hardlink across md0/md1.
set -u
export PATH=/usr/local/cuda-12.8/bin:$PATH
M=${M:-$HOME/mpk-qwen35/m4i8}
S=${S:-/var/tmp/m4i8_sweep}
T=${T:-$HOME/mpk-qwen35/mirage-m4i8}
BASE_REV=${BASE_REV:-5756c789}
BS=${BS:-1}
KD=${KD:-$S/kernel_A_bs${BS}}
OUT=$M/ptxas
mkdir -p "$OUT"
CU="$KD/test_rank0.cu"
[ -f "$CU" ] || { echo "MISSING TU: $CU"; exit 2; }
REL=include/mirage/persistent_kernel/persistent_kernel.cuh

BI=$M/base_include
rm -rf "$BI"; mkdir -p "$BI"
cp -al "$T/include" "$BI/include" || { echo "hardlink copy failed"; exit 2; }
rm -f "$BI/$REL"
git -C "$T" show "$BASE_REV:$REL" > "$BI/$REL" || { echo "cannot fetch base header"; exit 2; }
echo "pre-M4-I8 header ($BASE_REV): $(sha256sum "$BI/$REL" | cut -c1-16) $(wc -l < "$BI/$REL") lines"
echo "M4-I8 header:                 $(sha256sum "$T/$REL" | cut -c1-16) $(wc -l < "$T/$REL") lines"
echo "source diff (should be #ifdef blocks + the slot rename only):"
diff <(git -C "$T" show "$BASE_REV:$REL") "$T/$REL" | grep -cE '^[<>]' | sed 's/^/  changed lines: /'

one () {   # $1 = label, $2 = include root
  nvcc "$CU" -O3 -lineinfo \
    -I/usr/include/python3.12 -I"$2/include" -I"$2/include/mirage/persistent_kernel" \
    -I"$T/deps/cutlass/include" -I"$T/deps/cutlass/tools/util/include" -I"$T/deps/json/include" \
    -DMAX_WORKER_PER_SCHEDULER=1 -DMIRAGE_USE_CUTLASS_KERNEL=0 \
    -gencode=arch=compute_100a,code=sm_100a -DMPK_ENABLE_TMA -DMIRAGE_GRACE_BLACKWELL \
    -DMPK_TARGET_CC=100 -DMIRAGE_BACKEND_USE_CUDA -DMODE_OFFLINE \
    -DMPK_MAX_NUM_BATCHED_REQUESTS=$BS -DMPK_MAX_NUM_BATCHED_TOKENS=16 \
    -DMPK_MAX_NUM_PAGES=64 -DMPK_PAGE_SIZE=256 -DMPK_MAX_SEQ_LENGTH=353 \
    -shared -std=c++17 -rdc=false -use_fast_math -lcuda -lcudart -lstdc++fs \
    -Xcompiler=-fPIC --expt-relaxed-constexpr -Xptxas -v \
    -o "$OUT/$1.so" > "$OUT/$1.log" 2>&1
  echo "  $1 rc=$?"
  cuobjdump -sass "$OUT/$1.so" > "$OUT/$1.sass" 2>/dev/null
  rm -f "$OUT/$1.so"
}
echo "compiling one TU two ways, NO -D:"
one prem4i8_default "$BI"
one m4i8_default    "$T"
echo
DD=$OUT/default_sass_diff.txt
diff "$OUT/prem4i8_default.sass" "$OUT/m4i8_default.sass" | grep -E "^[<>]" > "$DD"
N=$(wc -l < "$DD")
# The header holds 31 assert() calls, each embedding __LINE__, and the #ifdef
# blocks shift those line numbers.  ptxas materialises such a constant as an
# IMAD.MOV.U32 / MOV / HFMA2 immediate, so a pure line-number shift shows up as
# immediate-only differences.  Anything else is a real codegen change.
OTHER=$(grep -vcE "IMAD\.MOV\.U32 R[0-9]+, RZ, RZ, 0x|MOV R[0-9]+, 0x|HFMA2 R[0-9]+, -RZ, RZ," "$DD")
pk () { awk '/Compiling entry function ._Z17persistent_kernel/{f=1}
             f && /bytes stack frame/{sf=$0}
             f && /Used [0-9]+ registers/{print sf" | "$0; exit}' "$1"; }
PRE=$(pk "$OUT/prem4i8_default.log"); POST=$(pk "$OUT/m4i8_default.log")
echo "  SASS changed lines: $N   non-immediate (real codegen) differences: $OTHER"
echo "  persistent_kernel entry, pre : $PRE"
echo "  persistent_kernel entry, post: $POST"
if [ "$OTHER" -eq 0 ] && [ "$PRE" = "$POST" ]; then
  echo "GATE 1c PASS: the default arm differs only in assert() __LINE__ immediates"
  echo "  ($N such constants shifted by the #ifdef blocks); the persistent_kernel"
  echo "  entry is identical in registers, barriers, stack frame, smem and spills,"
  echo "  and no opcode, control-flow or scheduling difference exists."
else
  echo "GATE 1c FAIL: the default arm's codegen CHANGED -- M4-I8 altered shipped behaviour"
  grep -vE "IMAD\.MOV\.U32 R[0-9]+, RZ, RZ, 0x|MOV R[0-9]+, 0x|HFMA2 R[0-9]+, -RZ, RZ," "$DD" | head -20
fi
grep -hE "Used [0-9]+ registers.*smem" "$OUT/prem4i8_default.log" "$OUT/m4i8_default.log" | sed 's/^/  /'
echo "CHECK_DEFAULT_DONE $(date -Is)"
