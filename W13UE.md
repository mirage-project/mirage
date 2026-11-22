# STATIC_MOE64_W13UE: T = 64 W13 with UE8M0 scales applied by the MMA (scaled_moe_experts)

Example (GLM-5.3 TP8 rank 0, layer 3, T = 64, LAT 6144 = K tiles 0..47, IR 256 = items mt 0..3; x with D = 1
distinct row, so slot 0 = expert 17 has all 64 tokens):

```
host:   w13_sf[17][0] = [kt 0..47][gate, up], each 2^e (pack_w13(ue8m0=True));  z_q scales xs[kt][t] = 2^e (group_quant ue8m0)
entry:  (slot 0, items mt 0, 1); item mt 0 = piece row 0: rows 0..63 gate, 64..127 up of features 0..63
ring:   stage s = 0..23 holds K tiles 2s, 2s+1 (two 16 KB pieces + z_q 64 rows x 2 x 128 B); issuer 0 owns s even, issuer 1 s odd
issuer: lane s loads float4 {gate(2s), up(2s), gate(2s+1), up(2s+1)} once; waits acc_empty of the item's tick a = wtt % 2
  stage s = 4 (ring stage g = 4, st = g % 3 = 1), piece at = 0: kt = 8, scale slot k = 2 st + at = 2
    smem OFF_UE13 + 2*512 + 16 l = {eg(8), eg(8), eu(8), eu(8)}             (lane l = rows l, l+32 gate; l+64, l+96 up)
    smem OFF_WSF  + 2*512 + 16 l = {ex(t=l, 8), ex(t=l+32, 8), 127, 127}     (rows 64..127: past N = 64)
    fence.proxy.async; cp_sf -> TMEM col 384 + 8*2 = 400 (weight side), 404 (z_q side)
    4 x mma_mx (kind::mxf8f6f4, N 64) into issuer 0's accumulator, cols 64a .. 64a+63; the first of the item: no accumulate
  after its last stage (s = 22 / 23): commit acc_full[a]
epilogue: v[t] = TMEM[row][64a + t] + TMEM[row][128 + 64a + t], t = 0..63; arrive acc_empty[a]
          then h = SiLU(gate) up, h_q, CHQ[0] += 2 per 128 features (unchanged)
before:   48 per-K-tile accumulators per item, 12 ticks; the epilogue read 48 x 64 columns and did 48 x 64 FMAs with fp32
          sw x sx per thread ('W13: scales -> drained' 19.8 us at D = 1, 13.7 at D = 64, measured with W13C off)
now:      1 tick per item, the epilogue reads 2 x 64 columns and adds them
```

## 1. Switches

| name | where | values |
|---|---|---|
| `STATIC_MOE64_W13UE` | nvcc `-D` (scaled_moe_experts.cuh:162) | 1 default: the MMA path where the layer has option 8; 0: the fp32-scale W13 on the same inputs |
| layer option 8 | `scaled_moe_experts_layer(w13_ue8m0=True)` (params[2] bit 3) | W13 weights requantized, z_q from `group_quant_layer(ue8m0=True)` |
| `MIRAGE_STATIC_MOE64_W13UE` | env of demo/glm53_moe/demo.py and the decoder | default 1 at T = 64: option 8 + requantized W13 + UE8M0 x; 0: none of it (the old inputs) |
| `STATIC_MOE64_W13C` | nvcc `-D` | `Slots::W13C` is forced off when W13UE is on; with `-DSTATIC_MOE64_W13UE=0` it follows its own macro (default 1), so pass `-DSTATIC_MOE64_W13C=0` in both A/B builds, as the server does |

`Slots::W13UE = STATIC_MOE64_W13UE != 0 && option 8 && Slots::W2C` (T = 64, every row live, many-token epilogue, not
SINGLE). A step with an idle row (run-time live table) runs `SlotsOld<RES>`: W13UE off, the fp32-scale W13 on the same
power-of-two inputs. T != 64, Flash (its own demo, no option 8) and K3 (moe_experts.cuh) are unchanged: PTX identical to
the parent for every layer without option 8.

## 2. Files and lines

| what | file:line |
|---|---|
| macro and its comment | include/mirage/static_megakernel/tasks/scaled_moe_experts.cuh:144-163 |
| `Slots::W13UE_OPT`, `W13UE`, W13C off under it | scaled_moe_experts.cuh:327-337 |
| `SlotsOld::W13UE = false` | scaled_moe_experts.cuh:405 |
| `Smem::OFF_UE13` (+3 KB at the end, only with W13UE) | scaled_moe_experts.cuh:1023-1029 |
| `Smem::W13UE_COL` (TMEM 384..431) and its asserts | scaled_moe_experts.cuh:1094-1107 |
| `ue_word`, `issue_w13_ue`, `drain_w13_ue` | scaled_moe_experts.cuh:1312-1471 |
| issuer: W13UE item, 1 tick | scaled_moe_experts.cuh:2384-2400 |
| epilogue: barrier instead of `stage_w13_scales`, `drain_w13_ue`, 1 tick | scaled_moe_experts.cuh:2732-2737, 2758-2759, 2924 |
| option 8 needs a UE8M0 group_quant (static_assert) | scaled_moe_experts.cuh:3177 |
| `scaled_moe_experts_layer(w13_ue8m0=)` and its checks, opts bit 8 | python/mirage/mpk/static_megakernel.py:699-702, 791-796, 806-807 |
| `pack_w13(ue8m0=True)` | python/mirage/mpk/fp8_fp32scale_weights.py:31-46 |
| `layer_tensors(w13_ue8m0=)` | demo/glm53_moe/weights.py:95-103, 128 |
| demo: `W13_UE8M0`, MOE_OPTS, weights, group_quant ue8m0 | demo/glm53_moe/demo.py:102-103, 122, 188 |
| demo: reference with power-of-two x scales | demo/glm53_moe/demo.py:331-342, 527 |
| decoder: weights with `moe_demo.W13_UE8M0` | demo/glm53_model/decoder.py:179 |
| CPU numerics (requant vs SGLang, chunk + MMA simulation, error on real weights) | moe64_w13ue_numerics.py |
| CPU protocol model (scale slots, tcgen05 order, ticks) | moe64_w13ue_model.py |

## 3. Box A/B (one GPU, harness)

Inputs: x with D distinct rows, row t = distinct row t % D (D = 1: every slot of the 9 has 64 tokens; D = 64: random
routing). Make the files once:

```
for D in 1 4 16 64; do python -c "
import sys, torch
D, T, H = int(sys.argv[1]), 64, 6144
r = torch.randn(D, H, generator=torch.Generator().manual_seed(1)).to(torch.bfloat16)
r[torch.arange(T) % D].contiguous().view(torch.int16).numpy().tofile(sys.argv[2])" $D $HOME/mk/x_d$D.bin; done
```

Runs (tree `$HOME/mk/mirage_w13ue`, the environment of glmbox/jobs/ue8m0_quick.sh: venv, `MIRAGE_Z3_LIB`,
`LD_LIBRARY_PATH`, `CUDA_HOME`), one GPU each, per D:

```
cd $TREE/demo/glm53_moe
COMMON="--checkpoint /dev/shm/glm53 --layer 3 --gpus 1 --tp 8 --shared-in-bank --reps 20 --check-ref --prof-experts \
        --nvcc-flag=-DSTATIC_RESET_IN_KERNEL --nvcc-flag=-DSTATIC_MOE64_W13C=0 --x $HOME/mk/x_d$D.bin"
P="PYTHONPATH=$TREE/python:$TREE:$TREE/demo/glm53_model STATIC_TOKENS=64"
# old: the inputs and code before this change (fp32 W13 scales, amax / 448 x scales)
env $P MIRAGE_STATIC_MOE64_W13UE=0 CUDA_VISIBLE_DEVICES=0 python -u demo.py $COMMON --out $OUT/old_d$D > $OUT/old_d$D.log 2>&1
# off: the same power-of-two inputs as on, the fp32-scale W13 code
env $P CUDA_VISIBLE_DEVICES=1 python -u demo.py $COMMON --nvcc-flag=-DSTATIC_MOE64_W13UE=0 --save-y $OUT/off_d$D.pt \
    --out $OUT/off_d$D > $OUT/off_d$D.log 2>&1
# on: the MMA applies the scales
env $P CUDA_VISIBLE_DEVICES=2 python -u demo.py $COMMON --out $OUT/on_d$D > $OUT/on_d$D.log 2>&1
# on vs off, same inputs (run after off): relRMS of y and the routed rows (one fp32 accumulator vs per-K-tile promotion)
env $P CUDA_VISIBLE_DEVICES=2 python -u demo.py $COMMON --diff-y $OUT/off_d$D.pt --out $OUT/ondiff_d$D > $OUT/ondiff_d$D.log 2>&1
```

Read from each log: `span min .. median ..`, `PASS` / `FAIL` and the `reference ... routed rows relRMS` line (FAIL above
5e-3; the line says `x scales True` for off / on), `W13_UE8M0 ...: True/False`, and the `experts prof` rows
`W13 item (seen-done)`, `W13: scales -> drained` (with W13UE: the wait for the item's MMAs + one TMEM read),
`W13: drained -> h_q`, `last W13 done`, `W2 issuer 1: wait h_q`, `W13 items`. Baseline from the task (W13C off, before
this change): D = 1 span 104.1, W13 item 31.3 (19.8 + 10.8), last W13 done 81.8; D = 64 span 140.2, item 15.9, last W13
done 86.4. Server (only after the harness passes): the decoder takes W13_UE8M0 from the same env (default on); the
server's `MIRAGE_STATIC_EXTRA_FLAGS` already carries `-DSTATIC_MOE64_W13C=0`.

## 4. What reviewers must check

1. Weight-side scale rows (issue_w13_ue): lane l word c is row l + 32 c (CUTLASS's SM100 SfAtom, the layout W2UE's and
   scaled_gemm_tile's z_q-side chunks already use on the box), so words 0, 1 = gate rows 0..63 and 2, 3 = up rows
   64..127. In this tree the weight side was so far always the constant chunk; this is the first chunk with different
   rows. A wrong mapping mixes gate and up scales; --check-ref fails then (moe64_w13ue_numerics.py control `gate_up`:
   relative error 1.1).
2. Scale slot reuse: slots k = 2 (g % SMAX) + at in shared memory (OFF_UE13, OFF_WSF) and TMEM (W13UE_COL + 8 k), written
   only after the stage's `full`; that implies both issuers' arrivals on the stage's `empty` of use g - 3 (the owner's a
   commit), so its copies and MMAs are done (tcgen05.mma -> tcgen05.cp is not an ordered pair). moe64_w13ue_model.py:
   2e4 runs pass; controls fail.
3. Accumulator ticks: one per item in the issuers (`wtt0 += 1`) and the epilogue (`wtt0 += RES::W13UE ? 1 : ROUNDS`);
   each issuer waits acc_empty before its first MMA and commits acc_full once (count 2).
4. TMEM 384..431 is used by nothing else in the expert phase: W13 / W2 accumulators 0..255, W2UE chunks 256..267 and
   320..331, scaled_gemm_tile's UE8M0 chunks 448..495 (other tasks, before the expert node).
5. OFF_WSF is free in the expert phase (the e4m3 loader passes WSF = false; the W13C / W2 rings and the W2UE constant
   chunk end below it: static_asserts).
6. Shared memory: GLM-5.3 dynamic 199040 -> 202112 B from the base (launch 203776) + static 18432 (MoE-only build);
   the server's MoE layer kernel must stay at or under 232448 B in all (ptxas prints its static part). Flash (static
   34816 B) would exceed it by 2 KB if it opted in; it does not.
7. The epilogue's `bar.sync 1, 128` replaces stage_w13_scales' barrier (it ordered the previous item's acc_s reads before
   this item's writes).
8. Build identity was checked on the PTX: nvcc 13.3's ptxas here gives a different SASS for the same source from run to
   run (sm_100a and sm_103a alike), so SASS comparisons are not meaningful. No PTX instruction is new to this tree
   (tcgen05.cp 32x128b.warpx4 and tcgen05.mma kind::mxf8f6f4.block_scale as W2UE uses them), so the box's nvcc 13.0
   should take it; build on the box before a server run.
9. Numerics change (stated): W13 weights requantized as SGLang does for DeepGEMM, x's scales powers of two, all 48 K
   tiles in one fp32 accumulator per issuer. Routed rows vs the exact product, real layer 3, 5 experts, x = N(0, 1):
   fp32-scale kernel 3.71e-2, W2UE 4.82e-2, W2UE + W13UE 5.59e-2 (medians; moe64_w13ue_numerics.py).

## 5. Not done

- Lever B (STATIC_MOE64_W13SPLIT: a slot of many tokens in more W13 entries). A W13 entry is the two items of a 128-feature
  group because h_q's scale is per (token, 128 features); one item per entry puts the two items on two SMs, which then
  must exchange their per-token amax (and h) through global memory before either writes h_q, as the T <= 32 split mode
  does (h_quant_split_tok: scratch [SPLIT_SLOTS][T][IR] fp32, pair counters, an acq_rel add). At T = 64 that needs a
  version writing W2C's h_q planes with power-of-two scales, the scratch and counters in scaled_moe_experts_layer
  (can_split is T <= 32 now), and CHQ[slot] still reaching MT13 only after the second item of every group wrote its
  h_q (the W2 loaders wait for MT13). Not implemented.
- No GPU run; no compile of the decoder's real graphs (only the synthetic MoE-only kernels t64lv / t64pv / tf64v and the
  timing build).

## 6. Review round 1 (two reviews of 6af560f..aacd79b): no defect found, no code change

Both reviews (one on the MMA / TMEM / scale-factor contract, one on numerics and wiring) report no defect. Each point was
checked again in the code:

| point | where | result |
|---|---|---|
| idesc: `idesc_mx(IDESC_MX_E4M3, j)`, N = TW = 64 fixed, so W2UE's `idesc_mx_e4m3_n` is not needed | issue_w13_ue | holds |
| chunk rows: lane l word c = row l + 32 c; weight side words 0, 1 gate, 2, 3 up (GateUpScales `row >= 64`); z_q side words 0, 1 tokens l, l + 32, words 2, 3 past N hold 127 | issue_w13_ue st.shared.v4 | holds |
| scale pointer `W13SF + ex (IR/128) KT_LAT 2 + GateUpScales::first(mt, KT_LAT, 0)`, float4 per lane = {g(2s), u(2s), g(2s+1), u(2s+1)}, 16-B aligned (KT_LAT even) | issuer W13UE branch, stage_w13_scales | holds |
| xs [KT_LAT][T] written by wait_z_and_pairs, then `__syncthreads` before the issuers read it | run_scaled_moe_experts | holds |
| first MMA of each issuer does not accumulate (`s == iss && at == 0`) | issue_w13_ue | holds |
| barrier counts: empty 2 (owner's commit + other's arrive), acc_full 2 (runtime.cuh:593, 638), acc_empty 128 after fence_before | issue_w13_ue, drain_w13_ue | holds |
| ticks: +1 an item in issuers and epilogue; the other `PromoRounds` increments only on the non-W13UE path | issuer, epilogue | holds |
| scale slot k = 2 (g % SMAX) + at reused only after full(g), which follows empty(g - SMAX) (both issuers) | moe64_w13ue_model.py: MODEL OK, controls fail (rerun 2026-11-22) | holds |
| OFF_WSF + 6 x 512 B = RING_BYTES - OFF_WSF (config.cuh:87-88): the e4m3 loader passes WSF = false; W2 ring, h_q buffers, W13C ring and the W2UE constant chunk end below OFF_WSF (static_asserts) | config.cuh, Smem | holds |
| TMEM 384..431 clear of accumulators 0..255 and W2UE chunks 256..267 / 320..331 (static_assert); SlotsOld's piece-1 columns are used only in a SlotsOld step, never with a W13UE item in flight | Smem::W13UE_COL | holds |
| epilogue `bar.sync 1, 128` keeps stage_w13_scales' ordering of acc_s reads; nothing reads OFF_BS under W13UE | epilogue | holds |
| reference: `ref_rows(x_ue8m0=W13_UE8M0)` follows the graph's group_quant param, not the nvcc switch, so `-DSTATIC_MOE64_W13UE=0` has the same reference | demo.py | holds |
| non-bank shared expert (`_shared_branch`, T <= 8 in the decoder): `scaled_gemm_tile_layer` reads z_q's fp32 scales unless its own `ue8m0=True`; at T <= 8 W13_UE8M0 is False anyway | demo.py, static_megakernel.py:446 | holds |

Open (box only, not a defect): the decoder's MoE layer kernel total shared memory. From the source: MoE-only build
203776 (launch) + 18432 (static, includes group_quant's `scale_s[T]`) = 222208 B; the decoder adds row_rmsnorm's
`red[8]` (32 B) = 222240 B, 10208 B under 232448. ptxas on the box prints the static part; check it with the first
decoder build.
