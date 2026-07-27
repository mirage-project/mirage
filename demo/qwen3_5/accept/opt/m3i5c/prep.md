# M3-I5c — Phase-7 compaction race: fix, evidence, and the window plan

**Status: prepared, not validated on hardware.** The B200 was busy, so everything below is a
source-level argument, a compile-only result, or a host model. No GPU was claimed and no kernel
was launched. Section 5 is the pre-registered window plan.

The change replaces the in-place `atomicAdd` scatter that compacts `mpk_active_expert_ids` in
both SM100 routers with a barrier-separated prefix-count compaction. Output order becomes
strictly ascending by expert id, which was previously atomicAdd arrival order.

---

## 1. The defect

Both routers end by compacting a sparse mark array into a dense list. Phase 0 writes `-1` into
every slot and Phase 5/7 writes `mark[e] = e` for each expert some row's top-k selected. The
compaction was:

```cpp
for (int expert = start_expert + threadIdx.x; expert < end_expert; expert += blockDim.x) {
  int const local_expert = expert - start_expert;
  int const mark = mpk_active_expert_ids[local_expert];      // read slot j
  if (mark >= 0) {
    int const pos = atomicAdd(mpk_active_expert_ids + NUM_EXPERTS, 1);
    mpk_active_expert_ids[pos] = expert;                     // write slot pos
  }
}
```

Compacted entries land in slots `[0, n_active)`, which **alias** the marks of experts
`[0, n_active)`, and nothing orders thread *j*'s read of slot *j* against another thread's write
of slot *j*. Two independent defects sit on top of each other:

**(1) The race**, present even at `blockDim.x == NUM_EXPERTS` where each thread makes one pass.
Every scatter stores a **non-negative** id, so the corruption is one-sided:

- an **inactive** expert *j* whose slot was overwritten passes `mark >= 0` and appends **itself**
  (`expert`, not `mark`) to the list → a phantom expert and an inflated count;
- an **active** expert is never lost, because no scatter ever stores a negative value;
- with enough phantoms `pos` reaches `NUM_EXPERTS` and **clobbers the counter itself**.

The consequence at graph level is not a crash: the grouped GEMM streams weights for an expert no
token routed to, gathers zero rows for it (`num_rows == 0` → `continue`), and discards the work.
So the observable symptom is wasted weight traffic and a wrong `mask[NUM_EXPERTS]`, which is
exactly what the existing `mask[NUM_EXPERTS] == unique(ids)` assertions detect.

**(2) A guaranteed miscount when `blockDim.x < NUM_EXPERTS`** and the grid-stride loop makes more
than one pass: a thread's own pass-*p* scatter can land on a slot it reads in a later pass. That
is arithmetic, not a scheduling accident. Found by M3-I9b. No shipped graph launches either
router with `blockDim.x < NUM_EXPERTS` (`block_dim=(256,1,1)`, `NUM_EXPERTS` 128 or 256), so the
single-pass shape was hiding defect (2) entirely and merely thinning defect (1).

Sites: `topk_softmax_sm100.cuh:372-382` and `topk_sigmoid_sm100.cuh:382-392` at
65e42ee8.

---

## 2. The fix

A barrier-separated prefix-count compaction, one tile of `blockDim.x` experts at a time, carrying
the running base in a register. Identical code in both files.

```
base_t = #active in [0, t*B)                    (block-uniform, in a register)
rank   = base_t + #{ j in [t*B, local_expert) : mark[j] >= 0 }
mpk_active_expert_ids[rank] = expert
```

```cpp
if (mpk_active_expert_ids != nullptr) {
  int const num_local_experts = end_expert - start_expert;
  int const block_size = static_cast<int>(blockDim.x);
  int base = 0;
  for (int tile_base = 0; tile_base < num_local_experts; tile_base += block_size) {
    int const tile_end = (tile_base + block_size < num_local_experts)
                             ? (tile_base + block_size) : num_local_experts;
    int const local_expert = tile_base + static_cast<int>(threadIdx.x);
    bool is_active = false;
    int rank_in_tile = 0, tile_count = 0;
    for (int j = tile_base; j < tile_end; ++j) {
      if (mpk_active_expert_ids[j] >= 0) {
        ++tile_count;
        if (j < local_expert) { ++rank_in_tile; }
        if (j == local_expert) { is_active = true; }
      }
    }
    __syncthreads();            // every read of this tile's marks precedes every write
    if (is_active) {
      mpk_active_expert_ids[base + rank_in_tile] = start_expert + local_expert;
    }
    base += tile_count;
  }
  if (threadIdx.x == 0) { mpk_active_expert_ids[NUM_EXPERTS] = base; }
}
```

### 2.1 Why it is race-free

No assumption about warp size, `blockDim.x`, `NUM_EXPERTS`, the number of active experts, or how
many row tiles produced the marks.

1. **Tile *t* reads only slots `[t*B, min((t+1)*B, n_local))`** — its own marks. Nothing else.
2. **Every write of tile *t* targets a slot `< base_{t+1} <= min((t+1)*B, n_local)`**, because at
   most one active expert exists per slot, so `base_{t+1} = #active in [0, tile_end) <= tile_end`.
   Therefore **no tile can touch a later tile's marks**.
3. **Tile *t+1*'s writes cannot race tile *t*'s reads.** A thread reaching tile *t+1*'s barrier
   implies every thread already passed tile *t*'s barrier, hence finished tile *t*'s reads.
   Barriers are matched in program order and every thread executes the same sequence.
4. **The one `__syncthreads()` inside the tile separates that tile's reads from that tile's
   writes** — the only remaining overlap (a write can land inside the tile's own range when
   `base_t + rank_in_tile >= tile_base`, e.g. when every expert is active and the compaction is
   the identity).
5. **The barrier is reached by every thread.** `mpk_active_expert_ids`, `start_expert`,
   `end_expert` and `blockDim.x` are all block-uniform, so the trip count is uniform and no thread
   can skip it. This was checked explicitly because a divergent `__syncthreads()` inside a loop is
   the obvious way to get this wrong; nvcc emitted no diagnostic on any of the 32 compiles in §3.
6. **The count slot needs no barrier of its own.** It is written once, by thread 0, after the
   loop; compacted entries only ever occupy slots `< n_local <= NUM_EXPERTS`.

A second barrier after the writes is *not* needed and is deliberately absent: tile *t*'s writes go
to slots `< tile_end`, and tile *t+1* reads from `tile_end` upward — disjoint by (2).

### 2.2 Why it is deterministic

`rank` is a pure prefix count over the mark array, so the emitted list is strictly ascending in
expert id under every schedule. The `atomicAdd` — the only source of run-to-run permutation in
this task — is gone, and §3.2 confirms zero atomics survive in the compiled router.

### 2.3 Cost

`n_local` L1-resident broadcast loads per thread (256 at the shipped shape; every lane of a warp
reads the same address, so the warp issues one transaction). The router is **0.085% of measured
per-step worker time** — 564 µs of 664,964 µs at bs1, `opt/pertask_by_bs.csv`,
`TASK_MOE_TOPK_SOFTMAX_SM100` 40 tasks/step at mean 14.1 µs. A scan that costs a few hundred
nanoseconds per task is not worth trading correctness or a shared-memory allocation for. **No
shared memory is used**, so the megakernel's smem budget and worker occupancy are untouched — the
reason a `cub::BlockScan` or a ballot-plus-scratch scheme was rejected despite being O(1) per
thread.

### 2.4 Files changed

| file | change |
|---|---|
| `include/mirage/persistent_kernel/tasks/blackwell/topk_softmax_sm100.cuh` | compaction replaced; the full argument in comments |
| `include/mirage/persistent_kernel/tasks/blackwell/topk_sigmoid_sm100.cuh` | same code, comment cross-references the softmax sibling |
| `demo/qwen3_5/accept/opt/m3i5c/compaction_model.py` | new — host model + race detector (§4) |
| `demo/qwen3_5/accept/opt/m3i5c/stress_compaction.py` | new — pre-registered window stress (§5 P1) |
| `demo/qwen3_5/accept/opt/m3i5c/prep.md` | this file |

No codegen parameter, no kernel-dir knob, no test, and no builder was touched. `git diff -w -U0`
over `include/` shows exactly two hunks per file, both inside the compaction block; **no
floating-point statement was added, removed, or reordered anywhere**.

---

## 3. Compile-check evidence (B200, no GPU touched)

Scratch tree `~/mpk-qwen35/scratch-i5c/`, fresh. `pre` = `git archive HEAD include` at 65e42ee8,
`post` = the working tree. CUTLASS headers referenced read-only from the pinned f3fde58 copy at
`~/mpk-qwen35/scratch-i5b/cutlass`. No CUDA API call, no launch, no GPU claim.

Provenance: the `pre` router headers hash to `bc7f6859…` / `29b40066…`, byte-identical to the
hashes M3-I5b recorded for the tree it landed. The `post` headers hash to `0caa1c35…`
(softmax) / `b081de87…` (sigmoid), which is what this commit contains.

Four TUs, each instantiating the routers with `num_rows` as an integer **literal**, as the
megakernel sees it. The first three are M3-I5b's, unchanged, so the comparison is apples to
apples; `tu_i5c_ep.cu` is new and passes `start_expert`/`end_expert` as **runtime** values with
`__launch_bounds__` below `NUM_EXPERTS`, i.e. the multi-tile expert-parallel shape defect (2)
lives in.

### 3.1 The matrix — 32/32 clean

`nvcc -c` and `nvcc -cubin`, `sm_100a`, CUDA 12.8 (V12.8.93), for
4 TUs × {pre, post} × {fast, nofast} = **32 compiles, all rc=0, zero diagnostics**. `fast` carries
`-use_fast_math` (the shipped JIT default, `persistent_kernel.py:290`); `nofast` is the
`MPK_NO_FAST_MATH=1` lane. Flags otherwise identical to the M3-I5b gate.

### 3.2 Per-function SASS: the numerics did not move

M3-I5b could claim byte-identical SASS. **M3-I5c cannot and does not** — the fix intends to change
code. The narrower claim, which is what bit-exactness for the consumers actually rests on, is that
the *routing arithmetic* is untouched. Checked per function
(`~/mpk-qwen35/scratch-i5c/fpcheck.py`):

- **A.** every value-producing FP instruction (`FADD/FMUL/FFMA/FMNMX/FMNMX3/FSEL/FSETP/MUFU.EX2`,
  genuine `MUFU.RCP`, conversions) is unchanged in count;
- **B.** every reduction shuffle (`SHFL.BFLY`) is unchanged in count;
- **C.** POST contains **zero** atomics;
- **D.** POST contains **exactly one more** `BAR.SYNC` than PRE.

**Result: 70/72 functions PASS**, across both flag sets. Every shipped or candidate instantiation
passes, including:

| instantiation | FP ops | SHFL | atomics | BAR |
|---|---|---|---|---|
| `k_softmax_gated<16,256,16>` — **the shipped Qwen3.5 router** | SAME (250) | SAME (32) | 5→0 | 2→3 |
| `k_softmax<16,256,{1,15,16}>`, `k_softmax<8,128,{1,15,16}>`, `k_softmax<8,256,{1,7,8}>` | SAME | SAME | 5→0 | 2→3 |
| `k_sigmoid<{2,4,8}>` — DeepSeek-V3 | SAME (476) | SAME (168) | 5→0 | 2→3 |
| `k_sigmoid_odd<{1,3,7}>`, `k_sigmoid_big<{9,16,17,64}>` | SAME | SAME | 5→0 | 2→3 |
| `k_softmax_big_gated<16,256,{32,64,128}>` — **the mbt candidates** | SAME | SAME | 5→0 | 2→3 |
| `k_softmax_ep<…,{64,128,256}>`, `k_sigmoid_ep<{64,256}>` — runtime start/end expert | SAME | SAME | 1→0 | 2→3 |

Three SASS idioms are deliberately not counted as arithmetic, each verified by reading operands in
this build rather than assumed:

- `FLO.U32` — integer find-leading-one. It is part of nvcc's **warp-aggregated atomic** sequence
  (`VOTEU.ANY` + `POPC` + `FLO.U32` leader election + `SHFL.IDX` broadcast + one `ATOMG`), which
  is why the pre trees show 5 `ATOMG.E.ADD.STRONG.GPU`, 5 `SHFL.IDX` and 5 `VOTEU.ANY` that all
  vanish together. `SHFL.IDX` is *not* a router reduction; the reductions are `SHFL.BFLY` and
  those are unchanged.
- `HFMA2 Rn, -RZ, RZ, imm` — ptxas's 32-bit immediate-materialisation MOV idiom. Every `HFMA2` in
  the post dumps has that form; neither router has any fp16 arithmetic.
- `I2F.U32.RP → MUFU.RCP → F2I.FTZ.U32.TRUNC.NTZ` — unsigned integer division by a runtime value.
  Each occurrence consumes one `I2F.U32.RP` and one `MUFU.RCP`, so genuine reciprocals (the
  kernels' `1.f/row_sum`) are counted as `#MUFU.RCP − #I2F.U32.RP`: **2 pre, 2 post** in the
  shipped function.

**The two failures are the same instantiation under both flag sets: `k_softmax_big<8,128,17>`**
(+1 `FSEL`, +2 `FSETP`). Root-caused rather than waived: the multiset of distinct FP instruction
*forms* is identical (19 pre / 19 post under `fast`, 32 / 32 under `nofast`) and **every surplus
POST instruction is a duplicate of a form already present in PRE**
(`~/mpk-qwen35/scratch-i5c/dupcheck.py`, verdict recorded for both flag sets). The compiler
duplicated an existing guarded block while restructuring the 2-tile / 1-row-tail loop; the rest of
the delta is `.reuse` operand-cache hints migrating between otherwise identical lines. No shipped
graph or mbt candidate builds `<8,128,17>` — 128 experts is the Qwen3 30B-A3B shape and 17 rows is
an M3-I5b exploratory row count.

### 3.3 Resources (`ptxas -v`, pre/post)

Zero spill stores and zero spill loads everywhere, pre and post. The shipped router **improves**:
`k_softmax_gated<16,256,16>` goes 40 → 32 registers with no stack frame. All sigmoid
instantiations keep their pre-existing 96-byte stack frame (the `all_group_scores` /
`group_selected` local arrays), unchanged.

**One regression, recorded not waived:** `k_softmax_big<16,256,17>` gains a 64-byte stack frame
(0 → 64) with still zero spills — a `float row_chunk[16]` demoted to local memory by the register
allocator. Same 17-row shape as the §3.2 outlier. Not shipped and not an mbt candidate: the
candidates `k_softmax_big_gated<16,256,{32,64,128}>` are all 0/0.

### 3.4 CPU checks

- `demo/qwen3_5/accept/opt/m3i8/static_checks.py` → `ALL STATIC CHECKS PASS`, rc=0.
- `ast.parse` clean on both new Python files.
- **No CPU-runnable unit test exists for these kernels** — every test in
  `tests/runtime_python/blackwell/sm100_moe*/` builds a CUDA extension and needs a device. That
  gap is what §4 fills; nothing was invented to paper over it.

---

## 4. Host model of the algorithm (CPU, `compaction_model.py`)

The defect is schedule-dependent, so one GPU run proves little either way. The model simulates the
block at the level the bug lives at — individual global accesses by individual threads, interleaved
by a scheduler, with `__syncthreads()` as the only ordering primitive. Both thread bodies are
statement-for-statement transliterations of the CUDA.

Two instruments:

1. a **barrier-interval race detector** — inside any interval between two block-wide barriers, no
   address may be written by one thread and touched by another. That is exactly the happens-before
   relation CUDA gives a block synchronising only with `__syncthreads()`, so "no conflict" is a
   statement about **all** interleavings of that interval, not a sample;
2. the **output** under four concrete schedules (`sequential`, `reverse`, `roundrobin`, and two
   random), checked for set, count and ascending order.

2310 cases: 8 configurations (`NUM_EXPERTS` ∈ {8,16,64,128,256} × `blockDim.x` ∈
{1,2,3,4,5,7,8,16,17,32,33,64,128,256} including `blockDim < NUM_EXPERTS`, `blockDim > NUM_EXPERTS`
and `start_expert != 0`) × 14 mark patterns (none, all, first, last, halves, alternating,
randomised densities) × 5 schedules.

```
POST-FIX (per-tile prefix count + barrier)
  barrier-interval races detected : 0 / 2310
  wrong set / count / order       : 0 / 2310
  identical output across all 5 schedules per case: yes
  output order                    : strictly ascending by expert id

PRE-FIX control (in-place atomicAdd scatter) -- SET+count only
  barrier-interval races detected : 1571 / 2310
  wrong under sequential          : 93 / 462  (20%)
  wrong under reverse             : 246 / 462 (53%)
  wrong under roundrobin          : 0 / 462   (0%)
  wrong under random              : 207 / 924 (22%)

ABLATION: the fix with its __syncthreads() deleted
  barrier-interval races detected : 1765 / 2310
  wrong set / count / order       : 556 / 2310

ALL CHECKS PASS
```

Three things make this more than a green tick:

- the **pre-fix control** fails, so the detector is not blind. It also explains the field
  behaviour: a near-lockstep `roundrobin` schedule gets the right answer 100% of the time, which
  is why the race has been latent in shipped code.
- the **ablation** — the fix with only its `__syncthreads()` removed — races on 1765 cases and
  produces wrong output on 556, so the barrier is demonstrably the thing carrying the argument,
  not decoration.
- the model asserts on its own controls: if the pre-fix arm ever came out clean, or the ablation
  ever came out clean, the run exits non-zero.

`python3 demo/qwen3_5/accept/opt/m3i5c/compaction_model.py` — rc=0, ~4 min, no dependencies.

---

## 5. Consumers, and why ascending order is free

**Every reader of the buffer indexes its output by the expert ID, never by the list position.**
Enumerated exhaustively over `blackwell/`, `hopper/`, `ampere/` and all model builders.

### 5.1 Device readers

Every one has the same loop shape —
`for (pos = expert_offset; pos < num_activated; pos += EXPERT_STRIDE) { expert = mask[pos]; … }` —
and `pos` appears exactly once, to fetch `expert`.

| | reader | count / ids | output addressed by | registered tasks |
|---|---|---|---|---|
| R1 | `blackwell/moe_fp8_blockscale_sm100.cuh:194,196-197` | both | `:373-376` via `smem_tok`/`smem_slot`, built from `d_routing[expert*BATCH_SIZE+t]` (`:205`) | 241/242 — **the shipped Qwen3.5 path** |
| R2 | `blackwell/fp8_group_gemm_sm100.cuh:773-774` | both; ids re-read by 4 warp roles (`:820` DMA, `:1111` MMA, `:1346` scale, `:1487` epilogue) | `:1542` `mOutput(n_idx, topk_idx-1, m_idx)`, `topk_idx` from `tRoutingIndex(n_idx)` (`:1539`) | 248/249 UE8M0 |
| R3 | `blackwell/moe_linear_sm100.cuh:438-439` | ids `:489`/`:715`; MMA `:580-582` count-only | `:793`, id-addressed | 254/255 BF16 sm100 |
| R4 | `hopper/moe_linear_swapAB_hopper.cuh:349-350` | ids `:365`/`:463` | `:570`, id-addressed | 161/162 sm90 |
| R5 | `ampere/moe_linear.cuh:311-312,318` | — | position appears only under `DEBUG_LOG` | **not a registered task** (`MIRAGE_UNIT_TEST` only) |

Zero atomics and zero `RED.*` in R1–R4 (grep-confirmed), and accumulators are reset per output
tile (`moe_fp8_blockscale_sm100.cuh:277-283`, `moe_linear_sm100.cuh:610`,
`moe_linear_swapAB_hopper.cuh:490-492`), so there is no cross-expert reduction whose association
order could shift.

For R1 concretely:

```cpp
int const num_activated = d_mask[NUM_EXPERTS];
for (int ae = expert_offset; ae < num_activated; ae += expert_stride) {
  int const expert = d_mask[ae];
  ...
  int const slot = d_routing[(size_t)expert * BATCH_SIZE + t];       // by ID
  ...
  orow = ((size_t)smem_tok[row] * NUM_TOPK + smem_slot[row]) * <stride>;
  d_output[orow + col] = T(acc[...]);                                 // (token, slot)
```

### 5.2 CTA → position, and coverage

`task.task_metadata.expert_offset = bid.x` (`src/runtime.cc:354`, behind an 8-way task-type guard
at `:346-353`). The stride is `bgraph.grid_dim.x` for the blockscale and UE8M0 readers
(`src/kernel/task_register.cc:3243` and `:3011`; intent comment at `:3008-3010`) and **hard-coded**
for two others (`:2776` sm100 BF16, 10/8; `:3431` sm90, 5/4). Coverage of `[0, num_activated)` is
exact iff `grid.x >= stride`, which holds at every call site.

On the **shipped Qwen3.5 path `grid.x == stride` exactly** — `builder.py:589`
`grid_x = min(num_experts, mbt*topk)`, passed at `:597`/`:614` — so the CTAs partition the list one
per activated expert with no redundancy.

*Footnote, pre-existing and unrelated to this change:* `deepseek_v3/builder.py:1036`/`:1112` pass
`grid=(256,1,1)` into the hard-coded-stride-10/8 readers, so CTAs 10..255 redundantly recompute
positions. Idempotent, and unaffected by list order.

### 5.3 Why permuting the list changes nothing

1. The CTAs **partition** `[0, num_activated)`, so every position is visited exactly once whatever
   the order.
2. Permuting therefore only **re-assigns experts to CTAs**. No CTA's arithmetic changes: each
   rebuilds its row gather from `d_routing[expert]` from scratch.
3. Output cells are **disjoint across experts** — for a token `t`, each of its `TOPK` slots is
   filled by exactly one expert — and there is no accumulation, atomic, or split-k across experts
   in this stage.

So the SET is what matters, the set is unchanged, and **ascending order changes no output byte**.
No consumer is order-sensitive, so order determinism is not *required* here; it is taken because
it is free and it removes a real run-to-run nondeterminism source from the engine while M3-I11
root-causes engine nondeterminism separately.

### 5.4 Host readers

Nothing anywhere compares the compacted list as an ordered sequence.

- **SET-based:** `test_gate_topk.py:227-240` (`index_fill_` `:231`, `torch.equal` `:233`),
  `test_gate_topk_sigmoid.py:141-142`, `test_topk_sigmoid_testmode.py:205-206`.
- **COUNT-based:** `test_router_oracle.py:87` (`mask[NUM_EXPERTS] == unique(ids)`; `:79` also
  checks the per-token id set), `probes/moe/p5_router_semantics.py:229-231`, `mask_probe.py:106`.
- **PRODUCERS that already build ASCENDING lists**, i.e. the suite is already exercising exactly
  the order this fix now produces: `bench_fp8_moe_gemm.py:100-107`,
  `test_fp8_moe_gemm.py:113-116`, `test_moe_block_oracle.py:156`,
  `test_moe_w13_linear_testmode.py:62-69`. `test_w13_linear.py:38` and `test_w2_linear.py:38`
  reference-implement the consumer *assuming* ascending positions.

Line numbers are against the **current working tree**, because two other agents are concurrently
reworking `sm100_moe/test_gate_topk.py` and `sm100_moe_block_qwen35/test_router_oracle.py` into
tie-aware form. That rework **keeps** the `mask[NUM_EXPERTS] == unique(ids)` detector and keeps the
mask comparison set-based, so it composes with this change rather than colliding with it.

### 5.5 The aliasing, restated

`topk_softmax_sm100.cuh:376` read a mark from the same `[0, NUM_EXPERTS)` region that `:379` wrote
compacted ids into (sigmoid: `:386` vs `:389`). A clobbered slot reads back as active, so the list
can gain a duplicate-in-effect entry and the count can inflate. Computing ranks from a prefix count
before any write — and separating the two with a barrier — removes the aliasing hazard as a side
effect of removing the race.

---

## 6. Pre-registered predictions and falsifiers for the GPU window

Run in this order. `PY=$HOME/mpk-qwen35/venv-mpk/bin/python`, `MIRAGE=$HOME/mpk-qwen35/mirage`,
`export PATH=/usr/local/cuda-12.8/bin:$PATH`, `export HF_HOME=$HOME/mpk-qwen35/hf`, one GPU claimed
under the 3-sample stability guard.

### P0 — the existing detectors, cheapest first

```
cd $MIRAGE/tests/runtime_python/blackwell/sm100_moe              && $PY setup.py build_ext --inplace && $PY test_gate_topk.py
cd $MIRAGE/tests/runtime_python/blackwell/sm100_moe_sigmoid      && $PY setup.py build_ext --inplace && $PY test_gate_topk_sigmoid.py
cd $MIRAGE/tests/runtime_python/blackwell/sm100_moe_sigmoid      && $PY test_topk_sigmoid_testmode.py
cd $MIRAGE/tests/runtime_python/blackwell/sm100_moe_block_qwen35 && $PY setup.py build_ext --inplace && $PY test_router_oracle.py
```

**Predicted:** all four exit 0. The load-bearing assertions are
`mask[NUM_EXPERTS] == unique(ids)` in `test_router_oracle.py` and the reconstructed-mask comparison
in `test_gate_topk.py` / `test_gate_topk_sigmoid.py`.
**Falsifier:** any mask/count assertion failure means the compaction is wrong in a way neither the
model nor the SASS check caught, and the change is void — root-cause, do not retry.

### P1 — the race-targeted stress (the issue's own acceptance clause)

```
cd $MIRAGE/tests/runtime_python/blackwell/sm100_moe_block_qwen35 && $PY setup.py build_ext --inplace
$PY $MIRAGE/demo/qwen3_5/accept/opt/m3i5c/stress_compaction.py --iters 2000 --rows 16 \
    --out $HOME/mpk-qwen35/m3i5c/stress_rows16.json
```

2000 iterations at the shipped max rows, sweeping logit spread so the active count sweeps from
sparse to near-saturated (the pre-fix race's worst case is a full mark array). Four independent
per-iteration checks: C1 count-vs-unique, C2 set-vs-torch-oracle, C3 **strictly ascending**,
C4 same-input replay is byte-identical.

**Predicted:** 0 failures on all four counters, `n_active_max` close to `min(rows*8, 256)`.
**Falsifiers, each distinct:**
- C1 or C2 non-zero → the compaction is still wrong; the fix is void.
- C3 non-zero → the prefix count is not producing ascending output, i.e. §2.2 is wrong. This one
  can only fail if the barrier argument is wrong, since ascending is what a correct prefix count
  *is*.
- C4 non-zero with C1–C3 clean → the router has a second, independent nondeterminism source that
  is not the compaction. That is a **finding for M3-I11**, not a failure of this issue — record it
  and hand it over rather than reverting.

Then repeat at the mbt candidates, which is the prerequisite finding 5 of the M3-I5b prep-review
attached to any `mbt > 16` default change:

```
for R in 32 64 128; do $PY .../stress_compaction.py --iters 1000 --rows $R \
    --out $HOME/mpk-qwen35/m3i5c/stress_rows$R.json; done
```

**Predicted:** 0 failures at every row count — the fix's correctness argument has no row-count
dependence, since marks from multiple row tiles are just more marks. **Falsifier:** any failure
that appears only above 16 rows would mean the row-tile loop and the compaction interact, which
§2.1 says they cannot.

### P2 — per-case AC-3 byte-diff at the DEFAULT config (the non-regression gate)

Unchanged config, `mbt=16`, the M3-I2b protocol:

```
cd $MIRAGE/demo/qwen3_5/accept
for BS in 1 2 4 8 16; do
  $PY -u mpk_engine_run.py --batch-size $BS --max-seq-length 132 \
      --out-dir $HOME/mpk-qwen35/m3i5c/ac3_dumps \
      --kernel-dir $HOME/mpk-qwen35/m3i5c/kernel_ac3_bs$BS
done
$PY -u harness/run_ac3.py --engine-dump-dir $HOME/mpk-qwen35/m3i5c/ac3_dumps \
    --batch-sizes 1,2,4,8,16 --output-json $HOME/mpk-qwen35/m3i5c/run_report.json
$PY -u $HOME/mpk-qwen35/m3i2a/bytediff.py \
    $MIRAGE/demo/qwen3_5/accept/results/dumps_final \
    $HOME/mpk-qwen35/m3i5c/ac3_dumps 1,2,4,8,16
```

**Predicted:** `identical` for every (prompt, bs) case at all five batch sizes. The strong form,
on three independent grounds: the compacted SET is unchanged (§2), the routing arithmetic's SASS is
unchanged (§3.2), and the only consumer is order-insensitive (§5).
**Falsifier:** ANY token change. Not a tolerance question — it would mean one of those three
grounds is false. The correct response is cast-position root-cause, not a waiver.

Two caveats worth stating before the run:

- If the baseline `dumps_final` was itself produced under a schedule where the race fired, P2 could
  differ *because the baseline is wrong*, not the fix. Distinguish by re-running the **pre-change**
  commit twice: if the two pre-change runs disagree with each other, the baseline is the problem
  and the correct gate is post-change-vs-post-change plus P0/P1.
- A `dumps_final` mismatch that reproduces identically on repeat is a real regression; a mismatch
  that varies run to run points at M3-I11's territory.

### P3 — CI smoke

```
cd $MIRAGE/demo/qwen3 && $PY -u demo.py --use-mirage --max-new-tokens 50 \
    --max-num-batched-tokens 8 --max-num-batched-requests 1
```

Qwen3-8B is a 128-expert softmax-router graph — `k_softmax<8,128,8>`, FP-op-identical per §3.2.
**Predicted:** unchanged 50-token output. **Falsifier:** any token change.

### P4 — router cost (a measurement, not a gate)

The scan is O(`NUM_EXPERTS`) per thread where the old code was O(1), so it should be measured
rather than assumed away.

```
cd $MIRAGE/demo/qwen3_5/accept && <the opt/meta_noprof profiling lane, bs1 and bs16, >=3 reps>
```

**Predicted:** `TASK_MOE_TOPK_SOFTMAX_SM100` mean task time rises by **< 1.5 µs** from the 14.1 µs
bs1 baseline (`opt/pertask_by_bs.csv`), i.e. under +11% on a stage that is 0.085% of per-step
worker time → **< 0.01% of the step**, inside run-to-run dispersion.
**Falsifier:** a router mean above ~20 µs would mean the broadcast-load model is wrong (e.g. the
loads are not being coalesced into one transaction per warp). The fix would stay — correctness is
not negotiable against 0.085% — but the ballot-plus-scratch O(1) variant considered and rejected in
§2.3 would become worth revisiting.

---

## 7. What this prep does NOT establish

- Nothing here is hardware-validated. §2 and §5 are source-level, §3 is compile-only, §4 is a host
  model of the algorithm rather than of the compiled kernel.
- `k_softmax_big<8,128,17>` is not FP-op-identical (§3.2), and `k_softmax_big<16,256,17>` gains a
  64-byte stack frame (§3.3). Both are argued benign from the SASS and neither is shipped or an mbt
  candidate, but neither was executed.
- The model proves the ALGORITHM race-free under a barrier-only memory model. It does not prove the
  *compiled kernel* is race-free — that would need `compute-sanitizer --tool racecheck`, which is a
  GPU tool. **Worth adding to the window** if a device is free:
  `compute-sanitizer --tool racecheck $PY test_router_oracle.py`.
- The sigmoid router (DeepSeek-V3) gets the identical code and identical compile evidence, but no
  DeepSeek-V3 end-to-end run exists in this project. Its coverage is P0's
  `test_gate_topk_sigmoid.py` only.
- P1's `stress_compaction.py` was written on CPU and has never run. It depends on the
  `runtime_kernel_blackwell_moe_block_qwen35` extension, which two other agents are concurrently
  editing tests around; the pybind signature it calls
  (`topk_softmax_sm100(gating, weights, routing, active, vpt, round_weights)`) is from the current
  working tree.
