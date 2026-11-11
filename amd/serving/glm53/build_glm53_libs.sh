# build_glm53_libs.sh "T ..." [KINDS]: GLM-5.3 plain-decode server libraries (8 GPUs, TP 8) per batch size T from a GLM-5.3 tree
# (CPU work: amd/tests/test_glm53_pair_layer.py --codegen --serving, then amd/tools/gen_rank_lib.py): $DLIB/<kind>_t<T>/lib.so for
# the kinds g53full (layer 6: full indexer + MoE), g53shared (layer 7: shared + MoE), g53dense (layer 0: full indexer + dense MLP).
# Run with OMP_NUM_THREADS capped (many builds at once).   env: TREE DLIB EXTRA_FL (more --flag=-D.. for every build)
TS=${1:-"1 2 4 8 16 32 64"}; KINDS=${2:-"g53full g53shared g53dense"}
export GPU_ARCHS=gfx950; . /mpk-amd/env/bin/activate
TREE=${TREE:?}; D=${DLIB:?}; mkdir -p $D
export MPK_ARS_LAND_MIN_T=1 MK_BACKEND=hip MIRAGE_ROOT=$TREE PYTHONPATH=$TREE/python OMP_NUM_THREADS=${OMP_NUM_THREADS:-6}; cd $TREE
one() {   # one() T KIND: codegen + rank library; a line per library
  local T=$1 K=$2 CG=$D/cg_$2_t$1 OUT=$D/$2_t$1 L
  case $K in g53full) L=6 ;; g53shared) L=7 ;; g53dense) L=0 ;; esac
  # (glm53_layers.G53_FLAGS / G53_ENV: the harness adds them itself; gen_rank_lib needs the same flags)
  local FL="--flag=-DSTATIC_ARS_LAND_MIN_T=1 --flag=-DSTATIC_RBLK_PAR=1 --flag=-DSTATIC_GME_W13DB=1 --flag=-DSTATIC_GME_W2R=128 --flag=-DSTATIC_GME_W2XQ=2 --flag=-DSTATIC_GME_CUPOST=1 ${EXTRA_FL:-}"
  MPK_TOKENS=$T timeout 1800 python amd/tests/test_glm53_pair_layer.py --tokens $T --layer $L --codegen --serving --out $CG $FL > $CG.log 2>&1
  grep -q "hipcc exit 0" $CG.log || { echo "T $T $K CODEGEN FAILED: $(grep -m3 -E 'rror' $CG.log | cut -c1-300)"; return 1; }
  MPK_TOKENS=$T timeout 1800 python amd/tools/gen_rank_lib.py --layer-cu $CG/layer.cu --kind $K --out $OUT $FL > $OUT.log 2>&1 \
    && echo "T $T $K OK $(md5sum $OUT/lib.so | cut -c1-10)" \
    || echo "T $T $K LIB FAILED: $(grep -m3 -E 'rror|Error' $OUT.log | cut -c1-300)"
}
for T in $TS; do for K in $KINDS; do one $T $K & done; done
wait
