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

## 7. W13C (not implemented in this branch)

Measured shape of the W13 phase at T = 64: a W13 ring stage is 32 KB of weights + 16 KB of z_q (64 rows x 2 K tiles
x 128 B, config.cuh `A_STAGE = TW 256`), 24 stages an item (KT_LAT 48), the MMA N = 64. z_q is a third of the bytes
a stage lands (from L2), and the 48 KB stage limits the ring to 3 stages (SMAX 3, smem). The bigger lever by the
same bound as section 6: 16 KB -> 4 KB a stage is -25 % of the W13 landing bytes (W13 ~70-85 us a layer: bound
~17-21 us), and 36 KB stages give a 4th stage in the same 144 KB.

What a GLM W13C needs, mapped from K3's W13C / W13C2 / W13C4:

1. a 2D z_q row map per exchange set (K3: `act8 <K bytes> <rows> 1`, box 1 row) for `tile::gather4`; Python
   `scaled_moe_experts_layer` map slots 7 / 8 (T = 64 only), host.cuh map kind if `act8` has no 1-row box;
2. a W13 loader in scaled_moe_experts.cuh (not runtime.cuh `load_job`, shared by every GEMM): weights by warp 0 /
   warp 7 as now, z_q rows of the slot's tokens by `ceil(n / 4)` gather4 a K tile (K3 W13C4), expect bytes from n;
3. slots of more than 16 tokens (the shared expert's: 64) in groups of 16 (K3: extra groups `n_live + q`, W13 items
   per group, CHQ per group) or kept on the N = 64 path; the ring geometry must be one for all items;
4. `issue_promoted` with a per-item idesc (N = 16), `drain_promoted` reading 16 columns, its z_q scales
   `xs[atom][tok[c]]` (fp32 in shared memory: no TMEM scale copy at T = 64, simpler than K3's W13C2 scale buffers);
5. the HEAVY epilogue's h buffers indexed by place c instead of token (its wide path assumes token columns);
6. a new ring protocol if the stage count or loaders change (4 stages, two loaders: K3's W13C2 rule "slot s loaded
   by warp s % 2, waited / committed by issuer s % 2", `full` count 2) -> a CPU ring model with negative controls
   before any box run.
