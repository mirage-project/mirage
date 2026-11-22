# STATIC_FLASH64_TWOSHOT: the Flash layer tail's all-reduce as a two-shot by token (T 64, 4 GPUs)

Example: GLM-5.3-Flash, TP 4 (GPUs 0..3), T = 64 tokens, H = 4096, HC = 4 streams. The owner of token t is GPU
t / 16 (AR_TOWN = T / GPUS = 16): GPU 0 owns tokens 0..15, GPU 1 16..31, GPU 2 32..47, GPU 3 48..63.

```
// the layer tails (every Flash layer: one allreduce_send, then its readers)
//   DSA / KDA attention: o_proj (fixed point) -> allreduce_send (FX: task x = columns 128 x .. 128 x + 127, all T)
//                        -> hc_post_mix (task s = columns 32 s .. 32 s + 31 of every token, grid 128) -> hc_pre
//                        (or, attention "post only": -> hc_post)
//   MoE:                 scaled_moe_experts -> allreduce_send (a token's routed rows added)
//                        -> hc_post (task (q, t), grid (2, 64): token t, columns 2048 q .. 2048 q + 2047)   [end mhc]
//                        or -> hc_post_mix -> hc_pre (the next layer's attention mHC pre)                [end mhc_pre]
// hc_pre reads only this GPU's R' and mixes: no cross-GPU data. The all-reduced value feeds hc_post / hc_post_mix only.

// allreduce_send buffer, bytes GPUS T H 2 = 4 x 64 x 4096 x 2 = 2 MB a set, 0xFF-filled
//   one-shot:  [GPUS][T][H]: slot r = GPU r's partial of every token, multicast to every GPU
//   two-shot:  [0, 512 KB)      owner landing [GPUS][16][H]: slot r, row j = GPU r's partial of the owner's token
//                               16 owner + j (each GPU's copy: its own 16 tokens), one unicast store per 16 B
//              [512 KB, 1 MB)   gathered rows [T][H] bf16: bf16(sum over the GPUs), multicast by the owner
//              [1 MB, 2 MB)     unused

// allreduce_send on GPU 1, token 37 (owner GPU 2, j = 5), vector c (8 columns):
st.relaxed.sys  peer[2] + 1 * (16 * 4096 * 2) + (5 * 512 + c) * 16  <-  bf16(GPU 1's partial)   // ar_twoshot_send

// hc_post_mix task s = 7 (columns 224..255) on GPU 2, lane for token 37, vector v (columns 224 + 8 v ..):
//   1. owner (37 / 16 == 2): land the 4 slots of row 5, vector 7 * 4 + v, add in fp32 in GPU order 0, 1, 2, 3
//      (allreduce_land_bf16, the one-shot's code), multicast bf16(sum) to gathered row 37 of every GPU, re-arm the slots
//   2. every GPU, every token: poll gathered row t, vector 7 * 4 + v (one slot), re-arm it; out = that bf16 value
// lanes for tokens 0..15, 16..31, 48..63 of the same task do step 2 only (the owners' task 7 does their step 1)
// hc_post task (q, t): the same per (token t, its 256 vectors): GPU t / 16 does step 1, every GPU step 2
```

Values: the one-shot computes `bf16(fp32 sum of the 4 bf16 partials in GPU order)` on every GPU and both hc_post and
hc_post_mix round the sum to bf16 before they use it. The two-shot computes the same sum with the same code on the owner
and every GPU reads that bf16 value: R', the mixes and x of hc_pre are bit-identical to the one-shot.

Waits: hc_post_mix task s on GPU g ends only after task s on every other GPU did step 1 (hc_post: task (q, t) after the
owner's task (q, t)); step 1 waits only for the sends (deps of the task). Same lists on every GPU: no circle.
compile_plan checks the GPUs' lists together (static_schedule.check_twoshot_deadlock_free) when the two-shot is on.

How this differs from GLM-5.3's STATIC_GLM64_TWOSHOT (mirage_twoshot f3b9b28; 8 GPUs, owner t / 8):
- the readers: GLM's residual_add (LAND) and sum_gpus; Flash's hc_post and hc_post_mix (GLM's commit static-asserts
  them off). hc_post_mix's slice (32 columns of every token) already holds vectors of every owner's tokens, so it
  needs no token-interleaved order; a lane does step 1 for its own vector when this GPU owns its token, then step 2.
- sum_gpus is ported as in GLM's commit; sum_rmsnorm keeps the one-shot layout (a static assert asks for
  -DSTATIC_FLASH64_TWOSHOT=0); Flash does not use either.

Switch: STATIC_FLASH64_TWOSHOT (core.cuh), default 1 at T 64 on 4 GPUs, else 0; -DSTATIC_FLASH64_TWOSHOT=0 builds the
one-shot (the same code as before). The host sets G::peer: the harness (one process) every GPU's copy of the
multicast region (mc_alloc), a serving rank (init_rank) every rank's copy from torch symmetric memory's buffer_ptrs.

## The A/B (box, 4 GPUs; build on the box: its nvcc 13.0)

Harness (one process, GPUs 0..3, the in-kernel all-reduce over them; `-DSTATIC_RESET_IN_KERNEL` as the server builds, so
the re-arms run; each pair: the default build saves the outputs, the one-shot build must give the same bits):

```
export STATIC_TOKENS=64 CUDA_VISIBLE_DEVICES=0,1,2,3 M=/raid/catalyst/models/GLM-5.3-Flash R=--nvcc-flag=-DSTATIC_RESET_IN_KERNEL
O=--nvcc-flag=-DSTATIC_FLASH64_TWOSHOT=0
python demo/glm53_flash/moe_demo.py --model-dir $M --layer 3 --tp 4 --gpus 4 --end mhc $R --out /tmp/f2s_moe2 --save-y /tmp/f2s_moe.pt
python demo/glm53_flash/moe_demo.py --model-dir $M --layer 3 --tp 4 --gpus 4 --end mhc $R $O --out /tmp/f2s_moe1 --same-y /tmp/f2s_moe.pt
python demo/glm53_flash/moe_demo.py --model-dir $M --layer 3 --tp 4 --gpus 4 --end mhc_pre $R --out /tmp/f2s_moep2 --save-y /tmp/f2s_moep.pt
python demo/glm53_flash/moe_demo.py --model-dir $M --layer 3 --tp 4 --gpus 4 --end mhc_pre $R $O --out /tmp/f2s_moep1 --same-y /tmp/f2s_moep.pt
python demo/glm53_flash/dsa_fused_demo.py --model-dir $M --layer 3 --tp 4 --gpus 4 $R --out /tmp/f2s_dsa2 --save-y /tmp/f2s_dsa.pt
python demo/glm53_flash/dsa_fused_demo.py --model-dir $M --layer 3 --tp 4 --gpus 4 $R $O --out /tmp/f2s_dsa1 --same-y /tmp/f2s_dsa.pt
python demo/glm53_flash/kda_fused_demo.py --model-dir $M --layer 0 --tp 4 --gpus 4 $R --out /tmp/f2s_kda2 --save-y /tmp/f2s_kda.pt
python demo/glm53_flash/kda_fused_demo.py --model-dir $M --layer 0 --tp 4 --gpus 4 $R $O --out /tmp/f2s_kda1 --same-y /tmp/f2s_kda.pt
```

Each run ends with PASS / FAIL (the reference checks, every GPU equal, the last timed launch equal to the first, and with
--same-y the saved outputs bit for bit) and prints `span ... median` over the GPUs (512 MB read before each launch).

Server (sglang_dsa/sgl_chain.py, every DSA / KDA / MoE layer static; FLASH_TWOSHOT=0 builds the T 64 layers one-shot):

```
cd <run dir> && CUDA_VISIBLE_DEVICES=0,1,2,3 python <tree>/demo/glm53_flash/sglang_dsa/sgl_chain.py check_t64_bs64_attn check_t64_bs64_moe4l prof_all3ppf_off_bs64_p256 prof_all3ppf_off_bs64_p4000
cd <other run dir> && FLASH_TWOSHOT=0 CUDA_VISIBLE_DEVICES=0,1,2,3 python <tree>/demo/glm53_flash/sglang_dsa/sgl_chain.py prof_all3ppf_off_bs64_p256 prof_all3ppf_off_bs64_p4000
```
