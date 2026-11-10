# build_glm_libs.sh "T ..." [KINDS]: GLM-5.3-Flash plain-decode server libraries (4 GPUs, TP4) per batch size T from a GLM tree
# (CPU work: the harnesses' --codegen with the server's shapes, then amd/tools/gen_rank_lib.py): $DLIB/<kind>_t<T>/lib.so for
# the kinds gkda (KDA sublayer), gdsa (DSA sublayer, req rows of 8192 positions), gmoe (MoE sublayer), gdense (dense MLP),
# gkdamoe (a KDA layer with its MoE MLP as one kernel)
TS=${1:-"1 2 4 8 16 32 64"}; KINDS=${2:-"gkda gdsa gmoe gdense"}
export GPU_ARCHS=gfx950; . /mpk-amd/env/bin/activate
TREE=${TREE:-/mpk-amd/glm/mirage_glm5}; D=${DLIB:-/mpk-amd/glm/dlib}; mkdir -p $D
export MK_BACKEND=hip MIRAGE_ROOT=$TREE PYTHONPATH=$TREE/python; cd $TREE
one() {   # one() T KIND: codegen + rank library; a line per library
  local T=$1 K=$2 CG=$D/cg_$2_t$1 OUT=$D/$2_t$1
  case $K in
    # gkda: requests per glm_kda_decode task (MI355X 11-07 glmkda6: T 64 R 1 / 2 / 4 167.8 / 152.7 / 148.4 us, T 32 R 1 / 2
    # 103.6 / 100.9, T 8 R 2 slower 67.8 vs 60.8)
    gkda) TEST="amd/tests/test_glm_kda_layer.py --rpt $([ $T -ge 64 ] && echo 4 || ([ $T -ge 32 ] && echo 2 || echo 1))" ;;
    # gdsa: sparse-attention splits 8 at T <= 8, else 4 (MI355X 11-08 glmdsa11, --ctx 512: T 1 4 / 8 / 16 71.4 / 66.3 / 66.2 us,
    # T 8 98.6 / 92.4 / 98.2; T 32 2 / 4 / 8 138.3 / 123.7 / 131.2, T 64 192.6 / 181.2 / 199.2)
    gdsa) TEST="amd/tests/test_glm_dsa_layer.py --maxctx 8192 --splits $([ $T -le 8 ] && echo 8 || echo 4)" ;;
    gmoe) TEST="amd/tests/test_glm_moe_layer.py" ;;
    gdense) TEST="amd/tests/test_glm_dense_layer.py" ;;
    # gkdamoe: one kernel per layer (KDA + MoE, MPK_SPLIT_LOOP at the MoE part's first node, set by the harness; MPK_GLM_PAIR=1)
    gkdamoe) TEST="amd/tests/test_glm_pair_layer.py --layer 4 --mode pair" ;;
    gkdamoelast) TEST="amd/tests/test_glm_pair_layer.py --layer 4 --mode pair --last" ;;
    gkdadense) TEST="amd/tests/test_glm_pair_layer.py --layer 0 --mode pair --mlp dense" ;;
    gkdamoe2) TEST="amd/tests/test_glm_chain_layer.py --layer 4 --layers 2" ;;
    gkdamoe3) TEST="amd/tests/test_glm_chain_layer.py --layer 4 --layers 3" ;;
    gdsamoe) TEST="amd/tests/test_glm_dsa_layer.py --maxctx 8192 --splits $([ $T -le 8 ] && echo 8 || echo 4) --pair-moe" ;;
  esac
  local FL="" KQ=0
  case $K in gkdamoe2|gkdamoe3) FL="--flag=-DSTATIC_MAX_RESETS=48" ;; esac   # (the chains' reset buffers: core.cuh MAX_RESETS)
  # gkdamoe T >= 32: STATIC_GME_WBL2 (the KDA part's dirty state lines written back at the experts' start; MI355X 11-08 glmab32 bs 32
  # 12125.0 / 12172.9 vs 12281.7 / 12219.6 us without)
  if [ ${K%last} = gkdamoe ] && [ $T -ge 32 ]; then FL="$FL --flag=-DSTATIC_GME_WBL2=1"; fi
  # gkdamoe(last) T >= 32: the KDA state stores nt sc1 (STATIC_DRD_STATE_WT=2; with STATIC_DRD_COAL whole lines - the experts then meet
  # fewer dirty lines; MI355X 11-09 glmab71, A B A B: bs 32 span 11299 / 10744 vs 11762 / 11575 us, bs 64 15370 / 15424 vs 15170 / 15808)
  if [ ${K%last} = gkdamoe ] && [ $T -ge 32 ]; then FL="$FL --flag=-DSTATIC_DRD_STATE_WT=2"; fi
  # gkdamoe3 T >= 16: WBL2 too (each layer's experts after its KDA part; MI355X 11-09 glmab47 bs 16 9773.3 / 9204.5 vs 9820.2 / 9679.6 us)
  if [ $K = gkdamoe3 ] && [ $T -ge 16 ]; then FL="$FL --flag=-DSTATIC_GME_WBL2=1"; fi
  FL="$FL ${EXTRA_FL:-}"   # (EXTRA_FL: more --flag=-D.. for this build, e.g. a probe in one kind's libraries)
  # chains: layers 1 .. project with hybrid gemms (CHAIN_HYB=0: off); the marker file chain_hyb tells the hook to shuffle their wproj
  # (MI355X 11-09 glmchain6: chain 3 T 16 -12.9, T 4 -13.5 us)
  local HYB=0; case $K in gkdamoe2|gkdamoe3) HYB=${CHAIN_HYB:-1} ;; esac
  MPK_CHAIN_HYB=$HYB MPK_GME_KQ=$KQ MPK_TOKENS=$T timeout 1800 python $TEST --tokens $T --gpus 4 --codegen --serving --out $CG $FL > $CG.log 2>&1
  grep -q "hipcc exit 0" $CG.log || { echo "T $T $K CODEGEN FAILED: $(grep -m3 -E 'rror' $CG.log | cut -c1-300)"; return 1; }
  MPK_GME_KQ=$KQ MPK_TOKENS=$T timeout 1800 python amd/tools/gen_rank_lib.py --layer-cu $CG/layer.cu --kind $K --out $OUT $FL > $OUT.log 2>&1 \
    && { rm -f $OUT/chain_hyb; [ $HYB = 1 ] && touch $OUT/chain_hyb; echo "T $T $K OK $(md5sum $OUT/lib.so | cut -c1-10) hyb $HYB"; } \
    || echo "T $T $K LIB FAILED: $(grep -m3 -E 'rror|Error' $OUT.log | cut -c1-300)"
}
for T in $TS; do for K in $KINDS; do one $T $K & done; done
wait
