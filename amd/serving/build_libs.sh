# build_libs.sh T [PARTS]: plain-decode server libraries for batch size T from mirage_bs (CPU only): /mpk-amd/dlib/{kdad,mla,moe}_t$T,
# fused (MoE + the next layer's aggregation 1 + its attention part): moeaggkdad_t$T, moeaggmla_t$T
# (the verify build's GEMM grids and flags; KDA requests per recurrence task and MLA KV splits per T). PARTS: "kda mla moe" (default), fkda, fmla
T=$1; PARTS=${2:-kda mla moe}
export GPU_ARCHS=gfx950 MPK_RANK_LIB_ASM=1 MPK_TOKENS=$T; . /mpk-amd/env/bin/activate
TREE=${TREE:-/mpk-amd/mirage_bs}   # (TREE / DLIB: another tree and output directory, for A/B builds)
export MK_BACKEND=hip MIRAGE_ROOT=$TREE PYTHONPATH=$TREE/python; cd $TREE
D=${DLIB:-/mpk-amd/dlib}; mkdir -p $D
strip_s() { grep -v "^\s*\.loc\|^\s*\.file\|\.ident\|hip_cuid\|^\s*;" $1 | sed "s/;.*//" | md5sum | cut -c1-12; }
res() { echo "T $T $1 lib.s $(strip_s $2/lib.s) $(grep -h "; ScratchSize:\|; NumVgprs:" $2/lib.s 2>/dev/null | head -2 | tr -s " " | tr "\n" " ")"; }
KG='{"21":[100,7,1],"24":[112,1,1]}'; KFL="--flag=-DSTATIC_GEMM_LDSW=3 --flag=-DSTATIC_GEMM_STAGES=16"
MG='{"29":[34,7,1],"31":[144,1,1],"32":[24,7,1],"37":[112,1,1]}'; MFL="--flag=-DSTATIC_GEMM_LDSW=3 --flag=-DSTATIC_AR_POST_NOWAIT=1 --flag=-DSTATIC_ARG_RELOAD --flag=-DSTATIC_GEMM_STAGES=16"
# T <= 32: MLA KV chunks by LDS DMA (dec80, fused MLA e2e: T 1 108.0 / 108.4 vs 116.3 / 119.0 us, T 4 121.4 / 123.5 vs 128.0 / 130.6,
# T 8 141.8 / 141.4 vs 147.2 / 151.1, T 16 183.1 / 183.1 vs 188.8 / 191.2, T 32 276.6 / 275.6 vs 279.8 / 279.6)
# (T 64, dec93: fused MLA 437.8 / 436.4 vs 451.7 / 449.5 us: every T)
MFL="$MFL --flag=-DSTATIC_MLA_KV_DMA=1"
MOEP="--fold --pack 34:4 --regrid 37:64,1,1:36 --apart 39,37,38 --regrid 28:56,8,1 --regrid 40:64,1,1:36,37,38,39"
MF="--flag=-DSTATIC_TASK_PREFETCH --flag=-DSTATIC_ARG_RELOAD --flag=-DSTATIC_GEMM_LANE_ASM=1"
RPT=$([ $T -ge 64 ] && echo 4 || ([ $T -ge 32 ] && echo 2 || echo 1)); KVS=$([ $T -le 4 ] && echo 16 || ([ $T -le 32 ] && echo 8 || echo 2))   # (MLA KV splits; dec66 / dec68, fused MLA e2e: T 4 8 splits 139 vs 4 150 us, 16 vs 8 1-2 us less at T 1..4; T 8 8 vs 4 148 vs 156, 16 tie; T 16 8 vs 4 188 vs 197; T 32 281 vs 287; T 64 4 vs 2 no change)
# per size (the paper's per-batch-size choices): T <= 4 the MoE's W13 items one per CU (few items; dec15: T 1 78.6 -> 73.4 us,
# T 8 89.8 -> 100.6); T <= 8 the fused KDA projection as the verify step's (hybrid 128-row tasks with half the weights in LDS,
# padded activation rows, the next task's first 256 KB pre-read into L2; node 68 = the projection in the fused graph)
[ $T -le 4 ] && export MPK_MOE_W13_CU=1
# T >= 16: the GEMMs' activation slices by LDS DMA (dec16: MoE T 32 shared down S tokens 16..23 off with the register fill,
# right with the DMA fill)
[ $T -ge 16 ] && MF="$MF --flag=-DSTATIC_GEMM_ACT_DMA=1"
# T <= 8 (KDA) / 16 (MLA): the attention's first node off moe_agg1's CUs (dec55, fused e2e: KDA T 1 101 vs 112 us, T 2 110 vs
# 118, T 4 124 vs 130, T 8 138 vs 143, T 16 no gain; MLA T 1..16 1-4 us less)
FMENV="MPK_ATTN_OFF_AGG=$([ $T -le 16 ] && echo 1 || echo 0)"
# T >= 16, fused MLA: o_proj waits for mla_vgate's count and fills its activation by DMA (MPK_MLA_OPROJ_POLLED=0; dec125: T 32 268.0 / 268.8
# vs 271.8 / 273.1 us, T 64 384.6 / 381.8 vs 384.7 / 385.8, T 16 175.2 / 174.5 vs 175.4 / 176.4; trees from mirage_bs_t)
[ $T -ge 32 ] && FMENV="$FMENV MPK_MLA_OPROJ_POLLED=0"   # (T 16: srvd31 moeaggmla 182.3 vs 179.1 us)
if [ $T -le 8 ]; then FKG='{"21":[50,7,1],"24":[112,1,1]}'; FKENV="MPK_KDA_PROJ_HYB=1 MPK_ATTN_OFF_AGG=1"
  FKFL="--flag=-DSTATIC_GEMM_ACT_PAD=1 --flag=-DSTATIC_GEMM_LDSW=3 --flag=-DSTATIC_GEMM_STAGES=16 --flag=-DSTATIC_GEMM_L2WARM=256 --flag=-DSTATIC_GEMM_L2WARM_NODES=68"
else FKG=$KG; FKENV="MPK_KDA_PROJ_HYB=0"; FKFL=$KFL; fi
FMFL="--flag=-DSTATIC_GEMM_ACT_PAD=1 $MFL --flag=-DSTATIC_MAX_RESETS=28"
[ $T -ge 16 ] && FMFL="$MFL --flag=-DSTATIC_MAX_RESETS=28"   # (padded rows turn the DMA fill off)
# T >= 32: topk_route's tasks right after the router's, over its 98 CUs (tokens_plan stacked them 8 per CU on 8 CUs at T = 64;
# dec37: T = 64 342.3 -> 304.8 us, T = 32 191.7 -> 174.7)
if [ $T -ge 16 ]; then MOEP="$MOEP --follow 27:26"; export MPK_MOE_FOLLOW="27:26"; MOEP=${MOEP/37:64,1,1:36/37:128,1,1:36}; export MPK_MOE_R_TASKS=128; fi
# T <= 2: the shared down 2 per CU (dec46, fused MoE + KDA: T = 1 120.0 -> 112.1 us, T = 2 123.5 -> 119.4; T = 16: R 128 tasks +
# topk after the router 185.2 -> 178.3)
if [ $T -le 2 ]; then MOEP=${MOEP/--pack 34:4/--pack 34:2}; export MPK_MOE_PACK=2; fi
# T = 1: the shared down unpacked (dec65 / dec68, fused e2e: KDA 96.3 / 96.9 vs 100.2 / 99.2 us, MLA 117.6 / 114.4 vs 119.0 /
# 119.8; T 2 no change)
if [ $T -eq 1 ]; then MOEP=${MOEP/--pack 34:2/--pack 34:1}; export MPK_MOE_PACK=1; fi
# T <= 2: a wave holds one W2 item at a time (dec62, fused e2e: KDA T 1 100.8 / 100.4 vs 101.8 / 106.3 us, T 2 107.3 / 108.9 vs
# 110.9 / 110.7, MLA T 1 122.3 / 119.6 vs 124.6 / 122.6; T 4 no change)
[ $T -le 2 ] && MF="$MF --flag=-DSTATIC_W2_ONE=1"
# T = 2 / 4: T = 8's routing table (dec67, fused KDA e2e: T 4 121.0 / 121.0 vs 122.8 / 125.8 us, T 2 107.1 / 108.7 vs 109.5 /
# 109.0; T 1 / 16 no change)
[ $T -eq 2 -o $T -eq 4 ] && MF="$MF --flag=-DSTATIC_FAST_TABLE=1"
# T 16, fused builds: the same table (dec379 / srvd82, -0.8..-1.25 us per fused kernel; the plain moe / moeagg T 16 libraries without it)
[ $T -eq 16 ] && { FKFL="$FKFL --flag=-DSTATIC_FAST_TABLE=1"; FMFL="$FMFL --flag=-DSTATIC_FAST_TABLE=1"; }
# T = 4: the next W13 unit claimed and its weights issued before the current one's MFMAs (W13CU builds; dec75, fused KDA 120.7 /
# 119.8 vs 124.0 / 121.3 us); the fused MLA with W13 units per wave and the shared down unpacked (dec73 / dec76: 128.2 / 129.1 vs
# 134.0 / 137.5); T = 8: the fused KDA with the shared down 2 per CU (dec76: 134.4 / 134.8 vs 134.9 / 136.5)
[ $T -eq 4 ] && FMENV="$FMENV MPK_MOE_PACK=1 MPK_MOE_W13_CU=0"
# T = 4, fused KDA: W13 units per wave + the shared down unpacked, as the fused MLA (dec115: 108.0 / 108.6 vs 113.0 / 116.3 us; per
# wave alone 116.0 / 116.2, unpacked alone 112.4 / 114.7; T 1 / 2 per wave slower: dec116)
[ $T -eq 4 ] && FKENV="$FKENV MPK_MOE_PACK=1 MPK_MOE_W13_CU=0"
# T <= 4 (W13CU builds): the experts' queues interleaved by groups of 2 experts (STATIC_MOE_ILV=2, in place of T 4's W13CU_PF;
# dec105, fused e2e: KDA T 1 91.1 / 92.9 vs 98.5 / 97.6 us, T 2 100.1 / 101.7 vs 105.4 / 107.9, T 4 117.5 / 115.5 vs 118.4 / 119.6;
# MLA T 1 104.2 / 103.7 vs 109.7 / 110.1, T 2 113.3 / 113.8 vs 118.3 / 118.1)
[ $T -le 4 ] && MF="$MF --flag=-DSTATIC_MOE_ILV=2"
# T = 4, the per-wave W13 builds (the fused MLA): the same interleave per wave by groups of 4 experts (STATIC_MOE_ILVW=4; dec107:
# 118.0 / 117.4 vs 123.3 / 122.1 us; T 8 / 16 slower)
[ $T -eq 4 ] && MF="$MF --flag=-DSTATIC_MOE_ILVW=4"
# T = 8 / 16: groups of 8 / 12 experts (dec109, fused e2e: KDA T 8 127.8 / 129.5 vs 134.8 / 132.8 us, MLA T 8 136.7 / 138.1 vs
# 139.3 / 142.6; KDA T 16 172.1 / 172.6 vs 176.4 / 177.7; T 16 MLA not measured)
[ $T -eq 8 ] && MF="$MF --flag=-DSTATIC_MOE_ILVW=8"
[ $T -eq 16 ] && MF="$MF --flag=-DSTATIC_MOE_ILVW=12"
[ $T -eq 8 ] && FKENV="$FKENV MPK_MOE_PACK=2"
# T >= 8: allreduce_send polls all of a slice's R block counters at once (dec83, fused KDA e2e: T 64 412.9 / 407.3 vs 417.4 /
# 414.3 us, T 16 177.8 / 179.5 vs 181.6 / 178.8, T 8 137.1 / 133.8 vs 138.2 / 136.3; MoE T 64 295.8 / 291.7 vs 299.9 / 298.2)
# (T 1 / 2 / 4, dec93: 96.6 / 95.7 vs 97.8 / 97.2 us, 108.4 / 108.5 vs 109.4 / 109.2, 117.8 / 117.1 vs 117.6 / 118.3: every T)
MF="$MF --flag=-DSTATIC_RBLK_PAR=1"
# T = 32: the CSR table's pairs polled with a thread's loads in flight together (dec84, fused KDA e2e: 284.4 / 285.8 vs 290.9 /
# 290.9 us; T 64 fused 411.8 / 414.0 vs 412.8 / 413.4, MoE alone no gain)
[ $T -eq 32 ] && MF="$MF --flag=-DSTATIC_CSR_PAIRS_PAR=1"
# T >= 32 (dec40 / dec41 / dec43, KDA decode alone: T = 64 182.8 -> 139.8 us, T = 32 116.5 -> 95.6): the projection 50 x 7
# (128-row tasks), short_conv_decode over 2 token groups, o_proj 56 tasks, 1 request per recurrence task, the DMA fill;
# the MoE's R allreduce_send at 128 tasks (dec42: T = 64 308.1 -> 293.5, T = 32 203.9 -> 196.4)
if [ $T -ge 32 ]; then KG='{"21":[50,7,1],"22":[100,2,1],"24":[56,1,1]}'; FKG=$KG; RPT=1; KFL="$KFL --flag=-DSTATIC_GEMM_ACT_DMA=1"; FKFL=$KFL
  MOEP=${MOEP/37:64,1,1:36/37:128,1,1:36}; export MPK_MOE_R_TASKS=128; fi
# T = 32: 4 requests per recurrence task (dec112, fused KDA e2e: 295.5 / 289.8 vs 302.4 / 302.2 us with 1; T 64 1 stays: 2 / 4
# 401.7 / 401.8, 403.6 / 400.2 vs 398.4 / 394.7)
[ $T -eq 32 ] && RPT=4
# T >= 32: the KDA recurrence as (head, request) pairs per task, every state loaded at the task's start (MPK_KDA_FLAT=P,
# delta_rule_decode.cuh FLAT; dec191, fused KDA e2e share 0.4: T 64 P 3 374.4 / 374.6 / 373.4 vs 389.5 / 390.0 / 387.8 us, T 32 P 2
# 279.7 / 277.3 / 282.3 vs R 4 286.4 / 289.3 / 294.7) - tree mirage_bs_au+
[ $T -ge 32 ] && export MPK_KDA_FLAT=$([ $T -ge 64 ] && echo 3 || echo 2)
# T 32: every pair's state written back after the count (STATIC_DRD_FLAT_LATE; dec205, means of 3: fused KDA 272.5 vs 278.1 us; T 64
# 364.2 vs 361.6 - not there) - tree mirage_bs_ay+
[ $T -eq 32 ] && { FKFL="$FKFL --flag=-DSTATIC_DRD_FLAT_LATE=1"; KFL="$KFL --flag=-DSTATIC_DRD_FLAT_LATE=1"; }
# T 64 fused KDA: o_proj in 3 K parts (combine finish_rs, MPK_OPROJ_FRS; the 64 x 1536 activation did not fit one task's LDS) and the
# activation rows padded (STATIC_GEMM_ACT_PAD: the B reads' LDS bank conflicts; register fill) - dec195, share 0.4: 366.7 vs 380.2 us
# (dec192 padded rows alone -8.5; T 32 both lose: +3.3 / +4.7) - tree mirage_bs_av+
[ $T -ge 64 ] && { FKENV="$FKENV MPK_OPROJ_FRS=1"; FKG='{"21":[50,7,1],"22":[100,2,1],"24":[56,3,1]}'; FKFL="$FKFL --flag=-DSTATIC_GEMM_ACT_PAD=1"; }
# T >= 16: a K-part GEMM's finish (latent_up, the T 64 o_proj) shared by every part's task of the row block (STATIC_GEMM_COOP_FINISH;
# dec198, means of 3: fused MLA T 64 372.7 vs 381.6 us, fused KDA T 32 277.0 vs 280.4, T 16 179.6 vs 181.0, T 64 with o_proj 56 x 3
# 358.8 vs 374.6) - tree mirage_bs_ax+
[ $T -ge 16 ] && MF="$MF --flag=-DSTATIC_GEMM_COOP_FINISH=1"
# T >= 32: the experts' z_q staging by LDS DMA after one check of every scale row, sum_quant_send pushing a block's scales after its
# acknowledged bytes (STATIC_CSR_ZDMA + STATIC_SQS_SCALES_LAST; dec223, means of 3: MoE alone T 64 249.6 vs 260.4 us, T 32 171.1 vs
# 178.7, fused KDA T 64 357.8 vs 362.4, T 32 268.9 vs 271.6, MLA T 64 tie) - tree mirage_bs_be+
[ $T -ge 32 ] && MF="$MF --flag=-DSTATIC_CSR_ZDMA=1 --flag=-DSTATIC_SQS_SCALES_LAST=1"
# T 8 / 16 fused: latent_up also off the CUs of S's send (MPK_LUP_APART_EXTRA=35; dec230, true recipe: MLA T 8 138.0 vs 140.3 us, KDA
# T 8 129.4 vs 130.2, KDA T 16 180.1 vs 181.5, MLA T 16 181.9 vs 182.5) - tree mirage_bs_bf+
[ $T -eq 8 -o $T -eq 16 ] && export MPK_LUP_APART_EXTRA=35
# (T 2 / 4: the fused MLA only - dec231, MLA T 2 107.2 vs 108.8 us, T 4 117.1 vs 119.3; KDA ties)
[ $T -eq 2 -o $T -eq 4 ] && FMENV="$FMENV MPK_LUP_APART_EXTRA=35"
# T 64 fused MLA: o_proj in 2 K parts (finish_rs; dec201, means of 3: 112 x 2 + coop finish 363.9 vs 379.9 us, 56 x 4 alone 367.0;
# T 32 112 x 2 284.8 vs 282.9) - its own grids (the unfused MLA's push_all o_proj has one K part)
FMG=$MG; [ $T -ge 64 ] && { FMENV="$FMENV MPK_OPROJ_FRS=1"; FMG='{"29":[34,7,1],"31":[144,1,1],"32":[24,7,1],"37":[112,2,1]}'; }
# T 8 / 16 fused MLA: mla_vgate over 2 / 4 token groups per head (grid 12 x G; dec216 T 8 G 2 140.7 vs 143.5 us, dec215 T 16 G 4 182.7
# vs 184.9; T 32 / 64 ties)
[ $T -eq 8 ] && FMG='{"29":[34,7,1],"31":[144,1,1],"32":[24,7,1],"37":[112,1,1],"36":[12,2,1]}'
[ $T -eq 16 ] && FMG='{"29":[34,7,1],"31":[144,1,1],"32":[24,7,1],"37":[112,1,1],"36":[12,4,1]}'
# (not used: the projection's L2 pre-read at T >= 16 - dec79's T 32 difference was spread: with the flag the T 32 fused lib.s is
# identical, the pre-read's conditions do not hold there)
# T >= 32: the fused builds' MoE tail not folded - R summed once per token (sum_rmsnorm, over the experts' CUs after R's send),
# latent_up's last K part pushing, S sent right after the shared down, sum_gpus over the experts' CUs after sum_rmsnorm
# (test_kda_moe_layer.py MPK_MOE_NFOLD; one more in-kernel reset: STATIC_MAX_RESETS 32, last on the line). dec92, fused e2e: KDA T 64
# 392.6 / 397.2 vs 432.2 / 434.4 us, T 32 303.2 / 304.6 vs 309.1 / 308.6; MLA T 64 422.5 / 423.5 vs 451.6 / 451.0, T 32 299.8 /
# 294.2 vs 303.0 / 300.8 (MoE alone dec88: T 16 slower, folded kept there)
# T >= 32 (fused builds only: the standalone kdad / mla not measured with it): o_proj's rows only to their GPU + attn_residual over each GPU's columns (MPK_AR_RS; dec111, fused e2e: KDA T 64 390.5 /
# 384.2 vs 402.1 / 394.4 us, MLA T 64 377.9 / 378.4 vs 401.7 / 406.6, KDA T 32 288.1 / 295.2 vs 290.1 / 301.9, MLA T 32 277.6 / 288.5
# vs 286.6 / 297.4)
if [ $T -ge 32 ]; then export MPK_MOE_NFOLD=1 MPK_MOE_SEND_FOLLOW=1 MPK_MOE_FOLLOW_LATE="38:36:37 39:36:38" MPK_AR_RS=1
  FKFL="$FKFL --flag=-DSTATIC_MAX_RESETS=32"; FMFL="$FMFL --flag=-DSTATIC_MAX_RESETS=32"; fi
# T = 8 / 16, fused KDA: the conv folded into the recurrence (MPK_KDA_CONV_IN_REC=1: no conv node, the KDA graph one node shorter -
# o_proj is "23" in the grids; trees from mirage_bs_u). T 8 with the projection rows ordered q / k / v, f_a / beta, g
# (MPK_KDA_SPLIT_PROJ=2). dec131: T 8 125.8 / 124.5 vs 131.2 / 131.6 us (fold alone 124.0 / 127.2); dec128: T 16 171.1 / 170.6 vs
# 172.2 / 172.3; T 1 / 2 / 4 ties
# T 8: the recurrence (node 69 in the CIR graph) off the CUs with two projection tasks (dec134: 126.0 / 122.9 / 123.2 vs 125.5 / 123.6 / 130.0)
[ $T -eq 8 ] && { FKENV="$FKENV MPK_KDA_CONV_IN_REC=1 MPK_KDA_SPLIT_PROJ=2 MPK_ATTN_OFF2=69:68"; FKG='{"21":[50,7,1],"23":[112,1,1]}'; }
# T 16 (dec240 / dec241, fused KDA, means of 4): the projection as 175 HYB4 tasks (512 KB, weights in before x) off latent_up's and
# moe_agg1's CUs in one pass, sum_gpus one per CU after the R / S sends, the conv in the recurrence: 169.5 vs 181.6 us (needs the
# MPK_ATTN_OFF_SET harness, tree mirage_bs_bh)
[ $T -eq 16 ] && { FKENV="$FKENV MPK_KDA_PROJ_HYB=4 MPK_ATTN_OFF_SET=68:39+48 MPK_MOE_FOLLOW_LATE=38:37:35:39 MPK_KDA_CONV_IN_REC=1"; FKG='{"21":[25,7,1],"23":[112,1,1]}'; }
# T 8 / 16 fused KDA (conv in the recurrence): its q / k / v, f_a and beta polled at once, the conv window first (STATIC_DRD_CIR_PAR;
# dec242 means of 4: T 16 168.1 vs 169.0, T 8 127.2 vs 128.1 us; tree mirage_bs_bk)
[ $T -eq 8 -o $T -eq 16 ] && FKFL="$FKFL --flag=-DSTATIC_DRD_CIR_PAR=1"
# T 8 / 16 fused KDA: the recurrence issues its conv window, conv weights and first K-part loads (and g's) before its state, g taken at
# the input stage (STATIC_DRD_CIR_EARLY + STATIC_DRD_G_EARLY; dec283 T 16 8 rounds 161.1 vs 162.45 us, dec287 T 8 125.25 vs 126.9) - tree mirage_bs_bx+
[ $T -eq 8 -o $T -eq 16 ] && FKFL="$FKFL --flag=-DSTATIC_DRD_CIR_EARLY=1 --flag=-DSTATIC_DRD_G_EARLY=1"
# T 32 fused KDA / MLA: sum_quant_send, sum_gpus and sum_rmsnorm as functions of their own (MPK_NOINLINE_TASKS: the kernel's register
# allocation does not span them; VGPR spills KDA 62 -> 0, MLA 73 -> 0; dec291 KDA 250.7 vs 256.6, dec292 MLA 248.8 vs 256.7 us; T 64 and
# KDA T 16 lose) - tree mirage_bs_ca+
[ $T -eq 32 ] && { FKENV="$FKENV MPK_NOINLINE_TASKS=sum_quant_send,sum_gpus,sum_rmsnorm"; FMENV="$FMENV MPK_NOINLINE_TASKS=sum_quant_send,sum_gpus,sum_rmsnorm"; }
# T 64: sum_rmsnorm + sum_gpus only (with sum_quant_send it lost ~10 us; dec294 KDA 336.4 vs 340.7, MLA 344.9 vs 348.3 us)
[ $T -ge 64 ] && { FKENV="$FKENV MPK_NOINLINE_TASKS=sum_rmsnorm,sum_gpus"; FMENV="$FMENV MPK_NOINLINE_TASKS=sum_rmsnorm,sum_gpus"; }
# every T, fused KDA / MLA: the next launch's exchange resets only on the CUs whose list starts with moe_experts (node 36), after
# kernel_begin's barrier, not waited for (STATIC_RESET_FIRST_NODE; dec252 / dec253, all checks pass: fused KDA T 1 -2.5, 2 -3.5,
# 4 -1.9, 8 -1.7, 16 -2.7, 32 -2.5, 64 -4.9 us; MLA -0.4 .. -5.6; tree mirage_bs_bm)
FKFL="$FKFL --flag=-DSTATIC_RESET_FIRST_NODE=36"; FMFL="$FMFL --flag=-DSTATIC_RESET_FIRST_NODE=36"
# T <= 16 fused KDA / MLA (latent_up with the RMSNorm folded): each R poll pass reloads its rows in one burst, then checks
# (STATIC_GEMM_NORM_POLL2; dec265, all checks pass: KDA T 16 -3.4, 8 -2.1, 4 -1.5; MLA -5.8 / -1.6 / -1.3 us; tree mirage_bs_bp)
[ $T -le 16 ] && { FKFL="$FKFL --flag=-DSTATIC_GEMM_NORM_POLL2=1"; FMFL="$FMFL --flag=-DSTATIC_GEMM_NORM_POLL2=1"; }
# T >= 32 fused KDA / MLA (unfolded tail, reduce-scatter attn_residual): sum_rmsnorm lands a thread's vectors in one burst
# (STATIC_SRN_POLL2) and attn_residual RS polls in bursts (STATIC_AR_POLL2); dec268, all checks pass: KDA T 32 -7.2, 64 -12.9;
# MLA -12.9 / -8.8 us (tree mirage_bs_br)
[ $T -ge 32 ] && { FKFL="$FKFL --flag=-DSTATIC_SRN_POLL2=1 --flag=-DSTATIC_AR_POLL2=1"; FMFL="$FMFL --flag=-DSTATIC_SRN_POLL2=1 --flag=-DSTATIC_AR_POLL2=1"; }
# (T 16 CIR: srvd31 moeaggkdad 180.7 vs 179.1 us - not used)
# fused MLA T 8: the shared down 2 per CU (dec138: 134.9 / 133.9 / 136.0 vs 137.6 / 137.7 / 137.1 us; dec115 135.0 / 136.6 vs 137.6 /
# 137.8); T 4: attn_residual in 14 chunks of 512 columns (MPK_AR_CHUNKS; trees from mirage_bs_v; dec138: 117.1 / 116.7 / 116.6 vs
# 119.3 / 117.8 / 117.4, dec137 116.6 / 116.2 vs 119.7 / 116.4)
[ $T -eq 8 ] && FMENV="$FMENV MPK_MOE_PACK=2"
[ $T -eq 4 ] && FMENV="$FMENV MPK_AR_CHUNKS=14"
# T = 1 / 2 / 8 / 32 fused MLA: the decode combine spread over the KV splits (MPK_MLA_COMB_SPREAD; dec163, fused MLA e2e means of 3
# rounds: T 8 135.0 vs 136.5 us, T 1 103.6 vs 105.1, T 2 110.3 vs 114.2, T 32 255.3 vs 259.5; T 4 / 16 / 64 ties) - tree mirage_bs_ah+
[ $T -eq 1 -o $T -eq 2 -o $T -eq 8 -o $T -eq 32 ] && FMENV="$FMENV MPK_MLA_COMB_SPREAD=1"
# MOE_AITER (env, default 1 since 10-24 srvd64; MOE_AITER= turns it off): the experts read AITER's layout (STATIC_MOE_AITER; the
# server with MPK_MOE_AITER=1 keeps SGLang's experts and prefills with AITER: GSM8K 208.7 vs 215.4 s) - tree mirage_bs_cd+
# T = 64 fused KDA: the projections second-round tasks weights pre-read into the L2 by the CUs starting with moe_experts
# (STATIC_EARLY_L2_SECOND; dec333, means of 4: 323.4 vs 326.0 us, every round; T 32 mixed) - tree mirage_bs_cl+
[ $T -ge 64 ] && FKFL="$FKFL --flag=-DSTATIC_EARLY_L2=512 --flag=-DSTATIC_EARLY_L2_NODES=69 --flag=-DSTATIC_EARLY_L2_SECOND=1"
# fused KDA T <= 32 / fused MLA T <= 16: every task thread-index read through a volatile v_mov (STATIC_TID_ASM=3: no per-lane value
# hoisted out of the task loop; SGPR spills KDA T 16 184 -> 71, T 16 scratch 32 B -> 0; dec339-345, means of 3-4: KDA T 1 -2.1,
# 2 -1.0, 4 -0.5, 8 -3.8, 16 -4.0, 32 -3.4; MLA T 1 -1.8, 2 -2.0, 4 -3.3, 8 -4.2, 16 -3.1 us; KDA T 64 +3.8 and MLA T >= 32 ties:
# off there) - tree mirage_bs_co+
[ $T -le 32 ] && FKFL="$FKFL --flag=-DSTATIC_TID_ASM=3"; [ $T -le 16 ] && FMFL="$FMFL --flag=-DSTATIC_TID_ASM=3"
MOE_AITER=${MOE_AITER-1}; [ -n "$MOE_AITER" ] && { MF="$MF --flag=-DSTATIC_MOE_AITER=1"; export MPK_MOE_AITER=1; }
# ---- adopted 10-26 .. 10-27 (paired in-server A/B, amd_runs/2026-10-22/STATUS_decode.md); fused builds only ----
# T >= 16: moe_agg1 loads 8 bank rows in flight (STATIC_AGG1_RB=8, srvd130), the task list entry by one scalar load (STATIC_TASK_SLOAD,
# srvd132: bs 64 -5); fused KDA: the recurrence's coalesced state columns (STATIC_DRD_COAL, srvd131 / srvd133)
if [ $T -ge 16 ]; then FKFL="$FKFL --flag=-DSTATIC_AGG1_RB=8 --flag=-DSTATIC_TASK_SLOAD=1 --flag=-DSTATIC_DRD_COAL=1"; FMFL="$FMFL --flag=-DSTATIC_AGG1_RB=8 --flag=-DSTATIC_TASK_SLOAD=1"; fi
# T 16: a W2 item's h_q before its weights (STATIC_MOE_W2_HQ_FIRST, dlib61: -5 per kernel); fused MLA: the decode combine spread and the
# shared gate_up off the router / latent_down CUs (MPK_FUSED_APART, dec27x: -1.3)
[ $T -eq 16 ] && { FKFL="$FKFL --flag=-DSTATIC_MOE_W2_HQ_FIRST=1"; FMFL="$FMFL --flag=-DSTATIC_MOE_W2_HQ_FIRST=1"; FMENV="$FMENV MPK_MLA_COMB_SPREAD=1 MPK_FUSED_APART=30:26,28"; }
# T >= 32: the CSR W2 item's h_q first (STATIC_CSR_W2_HQ_FIRST, srvd142) and no W2 steals from other dies (STATIC_CSR_W2_NOSTEAL,
# srvd163: bs 64 -7.5 .. -10); attn_residual RS pushes u before the dots' hop (STATIC_AR_U_EARLY=2, srvd188) on fused KDA T 32 / 64, MLA T 32
if [ $T -ge 32 ]; then H="--flag=-DSTATIC_CSR_W2_HQ_FIRST=1 --flag=-DSTATIC_CSR_W2_NOSTEAL=1"; FKFL="$FKFL $H --flag=-DSTATIC_AR_U_EARLY=2"; FMFL="$FMFL $H"
  [ $T -eq 32 ] && FMFL="$FMFL --flag=-DSTATIC_AR_U_EARLY=2"; fi
# T 32 fused KDA: sum_rmsnorm one task per token (srvd158) and z_q in 4 token groups (srvd161); fused MLA: o_proj 112 x 2 (srvd140)
[ $T -eq 32 ] && { FKENV="$FKENV MPK_MOE_REGRID=38:1,32,1:36,37;29:28,4,1:28"; FMENV="$FMENV MPK_OPROJ_FRS=1"; FMG='{"29":[34,7,1],"31":[144,1,1],"32":[24,7,1],"37":[112,2,1]}'; }
# T 64 fused KDA: sum_rmsnorm per token, cooperative z_q (MPK_ZQ_FINISH=2 + STATIC_GEMM_QUANT_COOP=2, srvd170), h_q die-local
# (STATIC_CSR_HQ_LOCAL, srvd181), the last pair's state held in LDS (STATIC_DRD_HOLD=72), R reduce-scattered (STATIC_MOE_R_RS=2, srvd187);
# fused MLA: padded activation rows filled by DMA (dec408), the combine spread, sum_rmsnorm per token, h_q die-local
[ $T -ge 64 ] && { FKENV="$FKENV MPK_MOE_REGRID=38:1,64,1:36,37 MPK_ZQ_FINISH=2 MPK_LDOWN_GRID=28,8,1"
  FKFL="$FKFL --flag=-DSTATIC_GEMM_QUANT_COOP=2 --flag=-DSTATIC_CSR_HQ_LOCAL=1 --flag=-DSTATIC_DRD_HOLD=72 --flag=-DSTATIC_MOE_R_RS=2"
  FMENV="$FMENV MPK_MLA_COMB_SPREAD=1 MPK_MOE_REGRID=38:1,64,1:36,37"; FMFL="$FMFL --flag=-DSTATIC_GEMM_ACT_PAD=1 --flag=-DSTATIC_GEMM_ACT_DMA_PAD=1 --flag=-DSTATIC_CSR_HQ_LOCAL=1"; }
for p in $PARTS; do case $p in
  kda) env -u MPK_AR_RS python amd/tests/test_kda_decode.py --codegen --tokens $T --rpt $RPT --gpus 8 --gemm-shuffled --grids $KG --out $D/kcg_t$T 2>&1 | grep -E "hipcc|rror" | head -3
       python amd/tools/gen_rank_lib.py --layer-cu $D/kcg_t$T/layer.cu --kind kdad --out $D/kdad_t$T $KFL 2>&1 | grep -E "lib.so:|rror" | head -3; res kdad $D/kdad_t$T ;;
  mla) env -u MPK_AR_RS python amd/tests/test_mla_decode.py --codegen --tokens $T --kv-splits $KVS --gpus 8 --gemm-shuffled --grids $MG --out $D/mcg_t$T 2>&1 | grep -E "hipcc|rror" | head -3
       python amd/tools/gen_rank_lib.py --layer-cu $D/mcg_t$T/layer.cu --kind mla --out $D/mla_t$T $MFL 2>&1 | grep -E "lib.so:|rror" | head -3; res mla $D/mla_t$T ;;
  moe) python amd/tests/test_layer_one_gpu.py --codegen --gpus 8 --gemm-shuffled $MOEP --out $D/ocg_t$T 2>&1 | grep -E "hipcc|rror" | head -3
       python amd/tools/gen_rank_lib.py --layer-cu $D/ocg_t$T/layer.cu --kind moe --out $D/moe_t$T $MF 2>&1 | grep -E "lib.so:|rror" | head -3; res moe $D/moe_t$T ;;
  magg) python amd/tests/test_layer_one_gpu.py --codegen --gpus 8 --gemm-shuffled $MOEP --agg 3,0,2 --out $D/acg_t$T 2>&1 | grep -E "hipcc|rror" | head -3   # (the last MoE layer + the model's final read: MPK_AMD_DEC_MAGG)
       python amd/tools/gen_rank_lib.py --layer-cu $D/acg_t$T/layer.cu --kind moeagg --out $D/moeagg_t$T $MF 2>&1 | grep -E "lib.so:|rror" | head -3; res moeagg $D/moeagg_t$T ;;
  fkda) env $FKENV python amd/tests/test_kda_moe_layer.py --codegen --decode --tokens $T --rpt $RPT --gpus 8 --gemm-shuffled --order moe --today --grids $FKG --agg 3,0,2 $MF $FKFL --out $D/fkcg_t$T 2>&1 | grep -E "hipcc|rror|order moe|OFF_AGG" | head -4
       python amd/tools/gen_rank_lib.py --layer-cu $D/fkcg_t$T/layer.cu --kind moeaggkdad --out $D/moeaggkdad_t$T $MF $FKFL 2>&1 | grep -E "lib.so:|rror" | head -3; res moeaggkdad $D/moeaggkdad_t$T ;;
  fmla) env $FMENV python amd/tests/test_kda_moe_layer.py --codegen --decode --tokens $T --attn mla --kv-splits $KVS --gpus 8 --gemm-shuffled --order moe --today --grids $FMG --agg 3,0,2 $MF $FMFL --out $D/fmcg_t$T 2>&1 | grep -E "hipcc|rror|order moe|OFF_AGG" | head -4
       python amd/tools/gen_rank_lib.py --layer-cu $D/fmcg_t$T/layer.cu --kind moeaggmla --out $D/moeaggmla_t$T $MF $FMFL 2>&1 | grep -E "lib.so:|rror" | head -3; res moeaggmla $D/moeaggmla_t$T ;;
esac; done
echo DLIBDONE $T
