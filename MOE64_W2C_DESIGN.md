# STATIC_MOE64_W2C: T = 64 W2 over a slot's own tokens (scaled_moe_experts)

Example (GLM-5.3 TP8 rank 0, T = 64, IR = 256, LAT = 6144): slot 5 = expert 17 has tokens {3, 40, 41}.

```
build_table (TOKEN_ORDER, every SM):   tmask[5] = bit 3 | bit 40 | bit 41
                                       list of slot 5 = start[5] + {0: token 3, 1: token 40, 2: token 41}
W13 epilogue (hq_out, PL = HQP):       h_q plane kt, row 5*64 + c, byte hq_swz(f, row)   (c = 0, 1, 2)
W2 entry (slot 5, tiles ot, ot + 1):   ng = ceil(3 / 16) = 1
  w2_load_hq:  expect 2 x 2048 B; bulk_g2s(buf + kt*8192, plane kt + 5*64*128, 2048)   (kt = 0, 1)
  issuer:      mma_f8_i(..., idesc_f8_n(16)) x 4 per kt          (was N = 64)
  epilogue:    entry start[5] + c reads TMEM column c             (was column = token id)
```

## 1. What changed and why

At T = 64 every W2 entry loaded 64 h_q rows of its slot (16 KB, the h_q tensor map's box {128, 64, KT2}) and ran
the W2 MMA N = 64 wide, while a slot has ~3.7-4.5 tokens at bs 64 (real routing, measured server state given in
the task). With W2C:

| | old (`-DSTATIC_MOE64_W2C=0`) | new (default) |
|---|---|---|
| a slot's list order | shared-atomic order (differs per SM) | token order (`build_table<RES, true>`) |
| h_q layout | `[slot T + token][IR]` | two planes `[KT2][NSLOT T][128 B]`, row `slot T + c`, chunks pre-swizzled |
| h_q load per entry | 1 TMA, `KT2 x 64 x 128 B` = 16 KB | `KT2` bulk copies of `16 ng x 128 B`; 4 KB at ng = 1 |
| W2 MMA N | 64 | `16 ng`, ng = ceil(ntok / 16) |
| W2 epilogue TMEM column | token id | place c in the slot's list |
| h_q scales | `[slot][token][KT2]` | unchanged |

`ng` for GLM-5.3 at bs 64: 1 for every routed slot with <= 16 tokens, 4 for the shared expert in the bank (64
tokens). The h_q buffer keeps its size (`NSLOT T IR` bytes = `KT2` planes of `NSLOT T` rows): no Python change.

Conditions (`Slots::W2C`): `STATIC_MOE64_W2C != 0 && T == 64 && LIVE == T && MANY_EPI && !SINGLE`. LIVE == T: a
token is at most once in a slot (a 64-bit mask per slot; idle rows of a verify step put a token several times on the
shared experts). MANY_EPI and !SINGLE: the only h_q writer is then `h_quant_group_tok<.., H = true>` (static_assert).
LIVE is compile-time; rows made idle at run time by the decoder's live table send the step to the old code (section 8).

## 2. Grouping tokens by token id, not by per-SM table order

Each SM builds its own routing table; the old step 3 placed a pair at `start + atomicAdd(fill)`, so slot 5's list
order differs between the SM that runs its W13 entries (writer of h_q) and the SM that runs a W2 entry (reader). The
K3 trap (skill section 6) applies directly: a token's h_q row must not depend on that order. TOKEN_ORDER:

1. step 2 zeroes `tmask[slot]` (64-bit, in the W13 accumulator area `Smem::OFF_ACC`: no W13 item has run when
   build_table does; NSLOT x 8 B = 2304 B for GLM-5.3, 4864 B for Flash, area 32 KB);
2. every pair: `red.shared.or.b32` of bit `t & 31` into word `2 slot + (t >> 5)`; `__syncthreads`;
3. every pair: `pos = start[slot] + popc(tmask[slot] & ((1 << t) - 1))`.

`start` and `ntok` were already deterministic (prefix sums of counts); with 3. the whole list is identical on every
SM. The W13 epilogue's loop index j over the list is the token's place c; the W2 epilogue's `i - start[slot]` is the
same c. Values are unchanged by the order: every per-token computation (amax, scale, e4m3, the W2 rows) is indexed
by token, and each output row (t, k) is written once.

## 3. Shared memory / TMEM budget at T = 64 (GLM-5.3 TP8, from the standalone build)

- dynamic (`Smem::BYTES`, unchanged): GLM-5.3 199040 B -> launch 200704 B, + static 18432 B = 219136 B; Flash
  (LAT 4096, bank NE 608, NPAIR 1152) 194816 B -> launch 196608 B, + static 34816 B = 231424 B; limit 232448 B.
  Static shared memory is the same in the old and new builds (ptxas): no new `__shared__` arrays, `tmask` lives in
  `OFF_ACC`.
- W2 ring: unchanged (3 stages x 33792 B of weights, two h_q buffers of 16 KB after them, inside the W13 ring area).
  The h_q buffers could shrink (ng <= 4 still needs 16 KB for the shared expert's slot), so no stage is gained.
- TMEM: unchanged columns; an N = 16 ng MMA writes columns `0 .. 16 ng - 1` of each K half's 64; the epilogue
  reads columns `< ntok` only.

## 4. Barrier protocol

No barrier, count, owner or parity changes. The only protocol-relevant change is the tx byte count of the h_q
buffer's `full` barrier (`w2hq_bar(b)`): `expect(KT2 x 16 ng x 128)` followed by exactly `KT2` bulk copies of
`16 ng x 128` bytes on that barrier, issued by the same thread right after the expect (as the old single TMA). The
K3 hang rules (skill section 5: two waiters on one barrier, two issuers / two loaders alternating, release counts)
are untouched: the h_q buffer is still loaded by the loader of the entry's first tile, waited by issuer 0 only, freed
by issuer 0's commit; `w2hq_bar(2 + b)` parity waits by the loaders are unchanged.

Rows `ntok .. 16 ng - 1` of the copy are the same slot's stale plane rows (never another slot's: a slot owns 64
rows), read by MMA columns nobody reads: no data race with another slot's W13 writer.

Cross-SM memory order: unchanged. The W13 epilogue's generic stores -> `bar.sync` -> thread 64's CHQ release; the W2
loader's CHQ acquire (`w2_arm`) -> `fence.proxy.async.global` -> the bulk copies (async proxy, as the TMA before).

CPU model: `moe64_w2c_model.py` (routing with skewed popularity, two SMs' tables, the writer's plane bytes, the
bulk copy into a 1024-B aligned buffer, the MMA's swizzled row read, the epilogue column, the tx count vs the
expect). Negative controls that must fail: the old atomic list order, no swizzle, plane rows from `start[slot]`
(not a multiple of 8), `ng = floor(ntok / 16)`, the old 16 KB expect.

## 5. Bit identity

Column c of the N = 16 ng MMA is the dot product of the same A rows with the same B row (the same h_q bytes of the
same token) as column `token` of the N = 64 MMA; the epilogue then runs the same FP operations in the same order.
Expected: y and the routed rows bit-equal. This relies on the tensor core's per-element sum not depending on N
(K3's W2C kept y bit-equal at N = 16); the box job checks it (`--same-y` both ways).

## 6. Expected saving, and its bound

Per layer at bs 64 on GLM-5.3 TP8 (assume ~130 live routed slots + the shared one; n2 = 131 x 48 = 6288 W2 tiles,
~3200 W2 entries, 21.6 a SM):

- h_q bytes into shared memory: old 16 KB an entry -> new 4 KB (ng = 1; 16 KB for the shared expert's 48 tiles):
  ~51 MB -> ~13 MB a layer (from L2), next to ~197 MB of W2 weights from HBM (131 x 1.5 MB).
- If the W2 phase is bound by what lands per SM (TMA bytes into shared memory, L2 or HBM alike): 1 - 210/248 = 15 %
  of the W2 phase. SGLang's W2 is 38.0 us a layer; at a W2 phase of ~40 us the bound is ~6 us a layer.
- If it is bound by HBM alone (h_q from L2 free), ~0. The per-entry h_q wait (on the issuer's path: issuer 0 waits
  `w2hq_bar` before the entry's first MMA) shrinks by the 12 KB transfer: ~0.05-0.1 us an entry at 120-250 GB/s a
  SM from L2, ~1-2 us a layer over 21.6 entries.
- MMA: 8 MMAs of N 64 -> N 16 a tile (~256 -> ~64 tensor clocks at ~8k FP8 MAC/clock/SM; a tile's 32 KB of weights
  land in ~0.7 us at 7 TB/s / 148 SMs): not on the critical path by this estimate, no gain counted.

So the estimate is 1-2 us a layer, bound ~6 us (x 75 layers: ~75-150 us a step, bound ~450). The target (1.05 x
SGLang = 16550 us a step from 17796) needs -1246 us a step = -16.6 us a MoE layer; W2C alone does not reach it. None
of these numbers is measured; the box job's prof rows (`W2 done-done`, `W2: wait acc full`, `W2 loaders: wait CHQ`)
give the split.

## 7. W13C (STATIC_MOE64_W13C, commits b8545e5, 263cc53)

Example (GLM-5.3 TP8 rank 0, T = 64, KT_LAT 48): the shared slot 130 has 64 tokens, slot 5 has tokens {3, 40, 41}.

```
w13c_groups (every SM):  slot 130: ng 4, extra groups (130, 1) (130, 2) (130, 3) at queue places M .. M + 2
entry (slot 5, q 0, items 0..1):  n 3, nq 1, rows word = 3 | 40 << 8 | 41 << 16 | 3 << 24
w13c_load stage s (slot g % 4):  expect 32768 + 1 x 1024; tma3 W13X2 (2 pieces); per K tile 2 s + kt:
                                  tma_gather4(ZQR, dst + 32768 + kt 2048, col (2 s + kt) 128, rows word)
w13c_issue:  mma_f8_i x 4 a piece, idesc_f8_n(16), columns as issue_promoted
drain_promoted_c:  v[c] += sw x xs[atom][tok[c]] x acc[c], c < 16 (atoms in issue_promoted's order)
epilogue:  h_q plane row 5 T + 16 q + c (hq_out c0 = 16 q); CHQ[5] += 2 per 128 features: MT13 per group
W2 (w2_arm):  waits CHQ[slot] >= MT13 ceil(ntok / 16)
```

| | W2C only (`-DSTATIC_MOE64_W13C=0`) | W2C + W13C (default) |
|---|---|---|
| W13 entry | (slot, two items) | (slot, group of <= 16 places, two items) |
| ring | layer-wide: 3 stages of 48 KB (32 KB + 64 z_q rows) | own: 4 stages of 36 KB (32 KB + 2 x 16 rows), same 144 KB |
| z_q rows a stage | 1 TMA box {128, 64, 2} = 16 KB | 2 x ceil(n / 4) tile::gather4 of 4 rows = 1..4 KB |
| barriers | rt.full0 / empty0 [3] | w13c_full [4] count 1, w13c_empty [4] count 2 (OFF_W2BAR + 2304) |
| MMA N | 64 | 16 |
| drain | T-wide (n > 8) or per token (n <= 8) | 16 columns, scale of tok[c] |
| CHQ per slot | MT13 | MT13 ceil(ntok / 16) |

Inputs: map slot 7 (Slots ZQR) = z_q as `act8 <L> <T> 1` (box {128, 1}, SW128) in exchange set 0, slot 8 set 1;
scaled_moe_experts_layer declares them for T = 64 layers of W2C's conditions; a layer without them compiles to the W2C
code (SASS-identical). Shared memory: dynamic and static unchanged (GLM-5.3 199040 -> launch 200704 + static 18432 in the
MoE-only build, 19456 in the GLM-5.3 layer kernel; Flash 196608 + 34816 = 231424 of 232448). The OFF_W2BAR area holds the
8 barriers (+2304) and the extra groups' table (+2368, NXG = NPAIR / 16 + 1 ints: 132 B GLM-5.3, 292 B Flash) inside its
3072 B.

Protocol: slot g % 4 always has the loader g % 2 (24 / 16 stages an item, even), its `empty` waits are on its own slots
(runtime.cuh's parity rule); every issuer waits every stage's `full` in order; the owner commits `empty`, the other
arrives (count 2, as the layer ring); acc ticks as issue_promoted. Warp 0's switch to W2 waits warp 7's stage count, then
each slot's last `empty` phase. CPU model `moe64_w13c_model.py` (2e4 runs a bank of the data part and of the ring with
roles at bursty random speeds): passes; controls that fail: odd depth 3, `empty` count 1, the switch on warp 0's slots
only, the switch without warp 7's count with 4 accumulator ticks; data controls: atomic order, CHQ MT13, nq floor, h_q
rows without 16 q, gather rows without 16 q, scales by place. Found by the model: at ACC_STAGES 2 the switch's wait for
warp 7's count is implied (issuer 0 starts a round only after the round two ticks back was drained, which follows issuer
1's MMAs of it); the wait stays.

Builds (nvcc 13.3, sm_100a): W13C 255 registers, 0 spills, 0-B stack (W2C only: 256-B stack: the T-wide path's local
array); T < 64, LIVE < T, the layer without map slot 7, and `-DSTATIC_MOE64_W13C=0`: SASS identical to the parent.

Not measured yet: the gain (bound in this section's first version: 16 KB -> ~4 KB of z_q a stage, a 4th stage; ~17-21
us a layer if W13 is bound by the bytes landing per SM). Box job: moe64_ab2.sh (old / w2c / new).

## 8. Run-time idle rows: the whole step takes the old code

Example (GLM-5.3 TP8, T = 64, K = 9, the decoder's live table with rows 0..62 live): token 63's 9 pairs all name the
shared bank expert (topk_route idle_pair: 8 routed pairs at weight 0, the shared pair at weight 1).

```
run_scaled_moe_experts, after wait_z_and_pairs (every SM, the same pairs):
  rows_once<RES>():  thread 63: e[0..8] = sel[567..575] = {256, 256, ..., 256} -> dup = 1;  __syncthreads_or -> false
  -> scaled_moe_table_and_queue<SlotsOld<RES>, OUT>   (W2C = W13C = false: the code of -DSTATIC_MOE64_W2C=0)
live 64 (no idle row): rows_once -> true -> scaled_moe_table_and_queue<RES, OUT>  (W2C, W13C as sections 1-7)
```

`SlotsOld<RES>` derives from `RES`: the routing table's `__shared__` arrays (RES's member functions) and `Smem` are
RES's own, so static and dynamic shared memory are unchanged (GLM-5.3 18432 B static in the MoE-only build, 19456 B in
the layer kernel; Flash 34816 B). `h_quant_group_tok`'s `red_s` is one array for both codes: the h_q plane row count
is a run-time argument (`pl`, a constant at each call), not a template argument. The old code is the same source
instantiated with W2C off; the `-DSTATIC_MOE64_W2C=0` build and T < 64 builds are SASS-identical to the parent.

The run-time dedup this replaces (commit 0c29da4: build_table's claim masks, the W2 epilogue over all of a slot's
pairs with a binary search for the column of a repeated token) cost +19 us a layer at live 64, where no row is idle
(box: W2C 135.85 us at a795180 vs 154.9 us at 0c29da4, old 136.95 / 137.2). Cause, from the SASS (sm_100a, nvcc
13.3, MoE-only build, W2C without W13C): the epilogue's `col(i, t)` lambda put a token load, a branch and an inlined
binary-search loop (with BSSY / BSYNC reconvergence) in front of each of the 16 single-column TMEM loads of an
8-token step: the loop body grew from 592 B (16 back-to-back LDTM, one wait) to 5280 B, its tail from 944 B to
5552 B, run for every W2 tile (~42 an SM) by the 4 epilogue warps. The fallback keeps the a795180 epilogue (560 B).

The live-63 mismatch of 0c29da4 (y and routed rows not bit-equal to old) is in the idle row only, by the code: the
old code's T-wide W13 epilogue applies SiLU in place to the second item's h (`h_silu_tok`, dst = up = h2, 4 list
entries a step: loads, then stores). A token listed n times in a slot gets SiLU once per step that holds it: 1 for a
live token, up to 9 for an idle token, a count that depends on the SM's atomic list order (moe64_w2c_model.py: two
SMs' orders give different counts for ~40 % of the idle tokens of the T-wide slots). The dedup code listed the idle
token once (SiLU once), so its idle row differed from old's; the live rows are computed per token with the same
operations in both. This is also why old itself is not run-to-run bit-equal at live 1 (63 idle tokens). With the
fallback, a step with idle rows runs the old code, so its live rows are the old code's; demo.py --same-y now compares
the live rows as well (whole tensors are still printed).

CPU models: moe64_w2c_model.py checks rows_once == (no idle row) in every run, the W2C path for live steps and the old
layout (atomic order, rows [slot T + token], column = token, every R row once) for idle steps, and the in-place SiLU
counts above (live tokens: exactly once); control no_fallback (the W2C table with idle rows) fails every run.
moe64_w13c_model.py: the same decision, its data checks for live steps, control no_fallback.
