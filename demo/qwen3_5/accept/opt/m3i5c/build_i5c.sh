#!/usr/bin/env bash
# M3-I5c compile-only gate. NO CUDA API call, NO launch, NO GPU claim.
# nvcc -c / -cubin / cuobjdump / ptxas -v only. Same flag sets and TUs as the
# M3-I5b gate (build.sh there), plus tu_i5c_ep.cu for the runtime
# start/end-expert, blockDim<NUM_EXPERTS shape.
set -u
export PATH=/usr/local/cuda-12.8/bin:$PATH
S=$HOME/mpk-qwen35/scratch-i5c
CUT=$HOME/mpk-qwen35/scratch-i5b/cutlass   # read-only, the pinned f3fde58 copy
OUT=$S/out
rm -rf "$OUT"; mkdir -p "$OUT"

BASE="-O3 -std=c++17 -gencode=arch=compute_100a,code=sm_100a \
 -DMPK_ENABLE_TMA -DMIRAGE_GRACE_BLACKWELL -DMIRAGE_BACKEND_USE_CUDA \
 -DMIRAGE_FINGERPRINT_USE_CUDA -DMPK_TARGET_CC=100 -DMODE_OFFLINE \
 --expt-relaxed-constexpr -Xcudafe --diag_suppress=177"

TUS="tu_small tu_odd_sigmoid tu_big tu_i5c_ep"

echo "### nvcc: $(which nvcc)"
nvcc --version | tail -2
echo "### cutlass: $CUT"
echo "### base flags: $BASE"
echo "### TUs: $TUS"
echo

RC_TOTAL=0
NCOMP=0
NOK=0
for FM in fast nofast; do
  EXTRA=""
  [ "$FM" = "fast" ] && EXTRA="-use_fast_math"
  for TREE in pre post; do
    INC="-I$S/$TREE/include/mirage/persistent_kernel \
         -I$S/$TREE/include/mirage/persistent_kernel/tasks \
         -I$S/$TREE/include \
         -I$CUT/include -I$CUT/tools/util/include"
    for TU in $TUS; do
      TAG="${FM}_${TREE}_${TU}"
      nvcc $BASE $EXTRA $INC -c "$S/$TU.cu" -o "$OUT/$TAG.o" \
          2> "$OUT/$TAG.compile.log"
      RC=$?; NCOMP=$((NCOMP+1)); [ "$RC" -eq 0 ] && NOK=$((NOK+1))
      echo "### rc=$RC  nvcc -c      $TAG"
      [ "$RC" -ne 0 ] && { RC_TOTAL=1; tail -30 "$OUT/$TAG.compile.log"; }
      rm -f "$OUT/$TAG.o"

      nvcc $BASE $EXTRA $INC -cubin "$S/$TU.cu" -o "$OUT/$TAG.cubin" \
          2>> "$OUT/$TAG.compile.log"
      RC=$?; NCOMP=$((NCOMP+1)); [ "$RC" -eq 0 ] && NOK=$((NOK+1))
      echo "### rc=$RC  nvcc -cubin  $TAG"
      [ "$RC" -ne 0 ] && { RC_TOTAL=1; tail -30 "$OUT/$TAG.compile.log"; }

      # any warning at all is worth seeing: a new barrier in a loop is exactly
      # the kind of thing nvcc warns about if it thinks it is divergent
      if [ -s "$OUT/$TAG.compile.log" ]; then
        echo "    --- diagnostics ---"
        sed 's/^/    /' "$OUT/$TAG.compile.log" | head -20
      fi

      if [ -f "$OUT/$TAG.cubin" ]; then
        cuobjdump -sass "$OUT/$TAG.cubin" > "$OUT/$TAG.sass"
      fi
    done
  done
done

echo
echo "==================== COMPILE MATRIX: $NOK/$NCOMP rc=0 ===================="
echo
echo "========= PER-FUNCTION VALUE-OP IDENTITY / ATOMIC / BARRIER CHECK ========"
FPRC=0
for FM in fast nofast; do
  for TU in $TUS; do
    echo
    echo "----- $FM / $TU -----"
    python3 "$S/fpcheck.py" "$OUT/${FM}_pre_${TU}.sass" "$OUT/${FM}_post_${TU}.sass"
    [ $? -ne 0 ] && FPRC=1
  done
done

echo
echo "==================== ptxas -v RESOURCES (pre/post) ===================="
python3 "$S/resources_i5c.py"

echo
echo "==================== BUILD_RC=$RC_TOTAL FPCHECK_RC=$FPRC ===================="
rm -f "$OUT"/*.cubin
