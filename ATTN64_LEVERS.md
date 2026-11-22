# GLM-5.3 attention at T = 64: the levers on branch glm64-attn

Example (layer 7, a shared layer, T = 64, every seq_lens 300, TOPK 2048, S = 8 sparse_mla splits, TP8 rank 0):

```
before (6af560f)                                  after (all levers on)
dsa_index_score / topk (layer 6): 3 x 1024 tasks  every task returns at entry (all seq_lens <= 2048)    L1
  each: load seq_lens[t], __syncthreads, add 1
sparse_mla split s of token t: 4 tiles (256 rows)  3 tiles (192 rows): splits 0, 1 = 192 + 108 rows     L3
  splits 2..7 empty: __syncthreads + add 1         splits 2..7 return at entry, no add                  L2a
dsa_mla_merge waits counter + t == 8               waits counter + t == mla_live_splits(300) == 2       L2a
q_a | kv_a, o_proj: 3 of 4 ring stages pre-wait   4 of 4 (STATIC_RING_STAGES 4)                        L4
dsa_q_absorb grid [64, 1]: 64 tasks of 32 tokens  [64, 2]: 128 tasks of 16 tokens                      L6
```

L5 (o_absorb + group_quant fusion) is not done.

## Switches

| lever | switch | default | old behavior |
|---|---|---|---|
| L1 | `STATIC_ATTN64_IDXSKIP` (dsa_common.cuh) | 1 at T = 64 | `-DSTATIC_ATTN64_IDXSKIP=0` |
| L2a | `STATIC_ATTN64_MERGENS` (dsa_common.cuh) | 1 at T = 64 | `-DSTATIC_ATTN64_MERGENS=0` |
| L3 | `STATIC_ATTN64_MLALIVE` (dsa_common.cuh) | 1 at T = 64 | `-DSTATIC_ATTN64_MLALIVE=0` |
| L4 | env `MIRAGE_ATTN64_RING` (demo/glm53_dsa/demo.py) | 4 | `MIRAGE_ATTN64_RING=3` |
| L6 | env `MIRAGE_ATTN64_QABS_Y` (demo/glm53_dsa/demo.py) | 2 | `MIRAGE_ATTN64_QABS_Y=1` |

- The `-D` switches go through `MIRAGE_STATIC_EXTRA_FLAGS` in the server (decoder.py SERVING) and `--nvcc-flag=-D...` in
  the demo. The env switches are read when `demo/glm53_dsa/demo.py` is imported, so the server (decoder.py imports it)
  takes them from its own environment.
- All five off (`MIRAGE_ATTN64_RING=3 MIRAGE_ATTN64_QABS_Y=1` and the three `-D...=0`): the server's T = 64 full and
  shared attention layers compile to the same SASS as 6af560f (checked on CPU, below).
- T != 64: every lever is gated on T == 64 (`constexpr` in the headers, `T == 64` in Python); the generated layer.cu of
  T = 8 and T = 32 (all four decoder kinds) is byte-identical to 6af560f's, and their SASS is identical.

## Per lever

### L1 `STATIC_ATTN64_IDXSKIP` (commit 4795427)

Files:
- include/mirage/static_megakernel/dsa_common.cuh:74-115 (switch, `ATTN64_IDXSKIP`, `idx_all_short`)
- include/mirage/static_megakernel/core.cuh:435-440 (`KernelLocals::idx_short`), :670-672 (zeroed in `kernel_begin`)
- include/mirage/static_megakernel/tasks/dsa_index_score.cuh:405-412 (early return; header note :6-8)
- include/mirage/static_megakernel/tasks/dsa_index_topk.cuh:340-349 (early return; header note :34-37)
- python/mirage/mpk/static_schedule.py:247-253 (`#define STATIC_DSA_IDX_SHORT 1` in a T = 64 layer with a unit-1
  dsa_index_topk node; no other layer gets the field, so MoE / dense / Flash / T < 64 KernelLocals is unchanged)

What a reviewer must check:
- The waiters on the three counters, in the server's full layer (decoder `attn_full`, T = 64, rr plan; layer.cu node
  ids from the CPU generation): score node 41 (counter 288) is waited on only by topk count node 43
  (`dsa_index_topk.cuh` MODE 0: `n > TOPK && !empty`); count node 43 (counter 352) only by topk write node 44 (MODE 1:
  `n > TOPK && c * 1024 < n`); write node 44 (counter 416) only by sparse_mla node 46 (`IX`, `idx_t != nullptr`, i.e.
  `seq_lens[t] > TOPK` for KPOOL 1). With every `seq_lens[t] <= 2048` none of these waits happens. No other node in
  the layer names these nodes as an input.
- Readers of `indices` when all tokens are short: sparse_mla of the full layer and of the later shared layers
  (`idx_t == nullptr` when `sl <= TOPK`, sparse_mla.cuh:552-555); the engine side (decoder.py, sglang_glm53/) does not
  read `StaticDecoder.indices` (it only passes it to the kernels). So topk write leaves idx as it was. A later step
  with one token > TOPK runs the full path, which writes all TOPK slots of that token.
- `idx_all_short` is called by all 256 threads at a task entry (it contains `__syncthreads_and`), only after
  `kernel_begin`'s `griddepcontrol.wait` (seq_lens is a host-written input of this step). The answer is cached per SM
  in `L.idx_short` keyed by K (score: SKIPK, topk: TOPK; Python enforces skip_k == TOPK when skip_k > 0).
- Flash (FLASH score, UNIT 4 topk) is excluded by `!FLASH` / `UNIT == 1` and by the generator's unit-1 test.

### L2a `STATIC_ATTN64_MERGENS` (commit 52470d0)

Files:
- include/mirage/static_megakernel/dsa_common.cuh:117-130 (switch, `MLA_KV_TILE`), :145-166 (`mla_split_tiles`,
  `mla_live_splits`)
- include/mirage/static_megakernel/tasks/sparse_mla.cuh:514-536 (per from `mla_split_tiles`; empty split returns
  before the `__syncthreads` with no counter add)
- include/mirage/static_megakernel/tasks/dsa_mla_merge.cuh:50-84 (`ns_of` from `mla_live_splits`; thread 0 loads the
  G seq_lens, then waits for `ns[i]` instead of S)

What a reviewer must check:
- The empty test in sparse_mla is `s >= dsa::mla_live_splits<TOPK, S>(live)` and merge's wait and reads use the same
  `mla_live_splits<TOPK, S>(live)` with the same `live = min(max(seq_lens[t], 0), TOPK)` and the same S (merge's
  `SM::x`) and TOPK (`SM::v[1]`). The old empty test `first * 64 >= live` is still below it and is equivalent
  (`s * per * 64 >= live` iff `s >= ceil(live / (per * 64))`; checked for live 0..2048 on CPU).
- Nothing else waits on a KPOOL-1 sparse_mla counter: dsa_o_absorb's MERGE path static_asserts `OL::v[7] > 1`
  (KPOOL > 1). Counters are zeroed every launch (host.cuh reset list), so a per-token target that changes between
  launches is fine. A token with seq_lens 0 gets ns 0: the merge does not wait and writes zeros, as before.

### L3 `STATIC_ATTN64_MLALIVE` (commit f17dc9e)

Files:
- include/mirage/static_megakernel/dsa_common.cuh:132-143 (switch, `MLA_LIVE_SPLITS = NSM / T` = 2), :153-158
  (`mla_split_tiles` = `max(1, min(TOPK / 64 / S, ceil(ceil(live / 64) / 2)))`)
- include/mirage/static_megakernel/tasks/sparse_mla.cuh:515-523 (`LIVE_SPLITS` also on with MLALIVE)
- include/mirage/static_megakernel/tasks/dsa_mla_merge.cuh:7-16, :50-56 (`ns_of` from the helper with MLALIVE; the
  wait for ns only with MERGENS (`WAIT_LIVE`))

What a reviewer must check:
- KPOOL 1 only (`LIVE_SPLITS` needs `KPOOL == 1`; merge needs `SEQ >= 0`, which python gives only for KPOOL 1).
- At most TOPK / 64 / S tiles a split: seq_lens >= 2048 keeps 8 splits of 4 tiles; 513 .. 2047 positions keep 4 tiles
  a split; <= 512 positions get 2 splits (64 tokens x 2 = 128 tasks). Coverage of [0, live) by the non-empty splits
  and ns <= S checked for live 0..2048 on CPU.
- sparse_mla's `early` copy (`(first + 1) * 64 < live`) uses the new `first`: tile `first` still ends before the last
  valid slot (the only cache row this step writes).
- Numerics: the split boundaries move (300: 256 + 44 -> 192 + 108 rows), so the merge's fp32 sum differs in order;
  y may differ in the last bits. With MERGENS off the empty splits still add 1 and the merge waits for S.

### L4 `MIRAGE_ATTN64_RING` (commit 0c230e3)

Files:
- demo/glm53_dsa/demo.py:88-93 (env), :150-151 (`mpk.ring_stages = 4` at T = 64 in `build_graph`; decoder.py
  `_graph` calls this `build_graph`, so the server gets it)

What a reviewer must check:
- The rr plan grids and task lists are unchanged: the CPU generation of the server's T = 64 graphs differs only in
  `#define STATIC_RING_STAGES 4` and the `SMAX == 4` check of layer.cu (the only ring-dependent Python default,
  `gemm_tile_layer`'s input_polled parts, has no node in the attention graph).
- Shared memory: the launch's `STATIC_SMEM_BYTES` goes from 173056 to 205824 (printed from the compiled layer), under
  the 227 KB static_assert (static_schedule.py smem_code). sparse_mla (173 KB Smem from the ring base) is unchanged.
- MoE / dense kernels are separate graphs and keep SMAX 3.

### L6 `MIRAGE_ATTN64_QABS_Y` (commit 673eb9e)

Files:
- demo/glm53_dsa/demo.py:94-97 (env), :257-264 (`rr_plan`: dsa_q_absorb `[2 * NH * 4, ATTN64_QABS_Y]` at T = 64)
- python/mirage/mpk/static_megakernel.py:1159-1160 (docstring), :1176-1179 (`changeable_grid_dims` axis 1, `(1, 1,
  None, 2)`, only at T == 64 with ROPE == 64, so `compiler.apply_grids` accepts y = 2; Flash's ROPE 0 node is unchanged)

What a reviewer must check:
- The server's plan: at T = 64 decoder.py sets `PLANS["attn_full"] = PLANS["attn_shared"] = None` (decoder.py:63-66,
  also with MIRAGE_STATIC_PLAN=hand), so `_plan` uses `dsa_demo.rr_plan` (decoder.py:280-282). There is no T = 64 hand
  plan file. A plan file given by `MIRAGE_STATIC_PLAN_ATTN_FULL / _SHARED` keeps its own grids.
- dsa_q_absorb.cuh: TH = 2, y = 2, TT = 16 (static_assert `TT % 8 == 0`); the q_pe loop covers t0 + 4 w + cb for w <
  4 (16 tokens). sparse_mla waits for `Q::x * Q::y * Q::z` = 128 (the generated layer's Q is `StaticNode<64, 2, 1,
  ...>`).

## Checks run here (no GPU; local nvcc 13.3, the box has 13.0)

- Graphs and layer.cu generated on CPU with the mock core of the earlier session
  (scratchpad glm8/cpu_check/check_graphs.py, fake_mirage.py): decoder graphs at T = 8, 32, 64 (ALLREDUCE_IN_KERNEL 1,
  and 0 at T = 64), and the one-GPU demo graphs at T = 64 (full / shared, key_scale ue8m0 / fp32, rr plan).
- `nvcc -O3 -std=c++17 -gencode arch=compute_103a,code=sm_103a -DSTATIC_RESET_IN_KERNEL -DSTATIC_PDL_TRIGGER=2 -c` of
  the server's T = 64 full and shared layers for every switch combination; the demo's T = 64 layers without the serving
  flags and with -DSTATIC_TIMING_BUILD.
- SASS (cuobjdump, addresses stripped): all off == 6af560f for the T = 64 full and shared layers; each switch at 0 ==
  the build without that commit; T = 8 / 32 attention, T = 32 MoE, T = 64 dense identical to 6af560f. The T = 64 MoE
  layer is not comparable this way: two builds of the same source with the same headers give different SASS (register
  numbers). Its preprocessed source differs from 6af560f's only by the new unused declarations of dsa_common.cuh.
- `python3 tests/python/test_static_twoshot_order.py` (static_schedule's two-shot order check): OK after each commit.
- No code uses an nvcc-13.3-only feature (`__syncthreads_and`, `if constexpr`, the existing PTX), but the build has not
  been run with nvcc 13.0: run build_decode on the box first.

## The 1-GPU harness A/B

T = 64 has no hand plan file (demo.py `plan_file("hand", ...)` would look for `hand_plan_<kind>_T64.json`, which does not
exist): use `--plan rr`. The demo's y options are `--save-y DIR`, `--compare-y DIR` (relRMS and bit-equal of y of the
first launch, and bit-equal of every launch's y in DIR/launches_layer<L>.pt), `--vary-x`, `--residual random`,
`--next-norm`; `--record FILE` for cost records. There is no `--diff-y` or `--check-ref` in this demo. Give `-D` flags
as `--nvcc-flag=-D...` (with `=`), and a separate `--out` per variant.

```
S300=$(printf '300 %.0s' $(seq 64))
cd ~/megakernel/mirage_twoshot
# old (all five off): SASS of 6af560f
MIRAGE_ATTN64_RING=3 MIRAGE_ATTN64_QABS_Y=1 STATIC_TOKENS=64 python demo/glm53_dsa/demo.py --checkpoint zai-org/GLM-5.3 \
  --layers 6 7 --tp 8 --rank 0 --plan rr --seq-lens $S300 --reps 20 --out build/a64_old \
  --nvcc-flag=-DSTATIC_ATTN64_IDXSKIP=0 --nvcc-flag=-DSTATIC_ATTN64_MERGENS=0 --nvcc-flag=-DSTATIC_ATTN64_MLALIVE=0 \
  --save-y y_old
# new (defaults)
STATIC_TOKENS=64 python demo/glm53_dsa/demo.py --checkpoint zai-org/GLM-5.3 --layers 6 7 --tp 8 --rank 0 --plan rr \
  --seq-lens $S300 --reps 20 --out build/a64_new --compare-y y_old
# one lever off (e.g. L3) against the defaults
STATIC_TOKENS=64 python demo/glm53_dsa/demo.py ... --out build/a64_noL3 --nvcc-flag=-DSTATIC_ATTN64_MLALIVE=0
# the indexer's real path (seq_lens > 2048: L1 inactive, L3 keeps 4 tiles): the demo's default --seq-lens
STATIC_TOKENS=64 python demo/glm53_dsa/demo.py --checkpoint zai-org/GLM-5.3 --layers 6 7 --tp 8 --rank 0 --plan rr \
  --reps 20 --out build/a64_old_d [old switches] --save-y y_old_d
STATIC_TOKENS=64 python demo/glm53_dsa/demo.py ... --out build/a64_new_d --compare-y y_old_d
```

Expected y (from reading the code, not run): L1, L2a, L4, L6 bit-equal to the old build (L4: runtime.cuh issue_job
assigns a job's stage s to issuer s & 1 whatever SMAX is; L6: the same per-token MMA columns); L3 may differ in the last bits
(relRMS printed by `--compare-y`). The span per layer is the line `layer L (...): median span X us over 20 launches`.
The server: `MIRAGE_STATIC_EXTRA_FLAGS="-DSTATIC_ATTN64_..=0"` and the two env variables in the server's environment.
