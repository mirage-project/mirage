# M3-I5b — router row loop: prep, bit-exactness argument, and the window plan

**Status: prepared, not validated on hardware.** The M3-I8 window owns the B200, so everything
below is either a source-level argument or a compile-only result. Nothing here was run on a GPU.

The change removes the MoE router's silent row cap in both SM100 routers and lifts the builder
assert that existed only to fence it off. `max_num_batched_tokens` is left at 16 on purpose — see
§5, the A/B is a separate measurement, not a default flip.

---

## 1. Mechanism

Both routers map one query row to one (warp, sub-group) and derive `thread_row` from `threadIdx`
alone. There was no loop, so a block covered exactly

```
ROWS_PER_CTA = WARPS_PER_CTA * ROWS_PER_WARP = 8 * (WARP_SIZE * VPT / NUM_EXPERTS)
```

rows and `if (thread_row < num_rows)` silently discarded the rest. A second task instance would
have recomputed the same rows, not the next slice, so the cap was per-graph, not per-task.

| kernel | shipped shape | VPT | THREADS_PER_ROW | ROWS_PER_WARP | **rows dropped past** |
|---|---|---|---|---|---|
| `topk_softmax_sm100` | Qwen3.5, 256 experts | 16 | 16 | 2 | **16** |
| `topk_softmax_sm100` | 256 experts, batch<=8 | 8 | 32 | 1 | 8 |
| `topk_softmax_sm100` | Qwen3 30B-A3B, 128 experts | 8 | 16 | 2 | 16 |
| `topk_sigmoid_sm100` | DeepSeek-V3, 256 experts | 8 | 32 | 1 | **8** |

Consequence at the graph level: `topk_w` and `mpk_routing_indices` stay ZERO for the surplus rows,
so those tokens lose their routed experts at all 40 layers while the shared expert and the residual
keep flowing — a quiet quality loss, not a crash. M2-I9 hit it at `mbt=128` (rows 16+ degraded,
every AC-3 prompt diverged at generated position 0) and fenced it with a builder assert.

The fix is a row-tile loop around the existing body:

```cpp
static constexpr int ROWS_PER_CTA = WARPS_PER_CTA * ROWS_PER_WARP;
for (int row_tile_base = 0; row_tile_base < num_rows; row_tile_base += ROWS_PER_CTA) {
  int const thread_row = row_tile_base + warp_base_row + thread_row_in_warp;
  uint32_t warp_mask = /* ...unchanged formula, keyed on thread_row and num_rows... */;
  if (thread_row < num_rows) {
    /* body verbatim */
  }
}
__syncthreads();
```

Files:

| file | change |
|---|---|
| `include/mirage/persistent_kernel/tasks/blackwell/topk_softmax_sm100.cuh` | row-tile loop |
| `include/mirage/persistent_kernel/tasks/blackwell/topk_sigmoid_sm100.cuh` | row-tile loop **+ warp-mask fix** (§2.3) |
| `src/kernel/task_register.cc` | drop `assert(rows_per_task >= batch_size)`; VPT selection unchanged |
| `python/mirage/mpk/models/qwen3_5/builder.py` | `MOE_ROUTER_MAX_ROWS_PER_TASK` (a cap) -> `MOE_ROUTER_ROWS_PER_TILE` (a cost unit); the `mbt <=` assert is gone |
| `python/mirage/mpk/persistent_kernel.py` | docstring: rows-per-pass, not capacity |
| `tests/.../sm100_moe_block_qwen35/test_router_oracle.py` | section 4 **inverted** (see §4.P0) |
| `tests/.../sm100_moe/test_gate_topk.py` | 1 row -> {1,8,9,16,17,33} |
| `tests/.../sm100_moe_sigmoid/test_gate_topk_sigmoid.py` | batch sizes {1,2,4,8} -> {1,2,4,8,9,16,17,32} |
| `tests/.../sm100_moe_sigmoid/test_topk_sigmoid_testmode.py` | batch 8 -> {8,17}, CLI-overridable |
| `demo/qwen3_5/accept/mpk_engine_run.py`, `demo/qwen3_5/demo.py` | comment/help only; **defaults unchanged at 16** |

### Nothing else was load-bearing on the cap

Checked, with the negative result recorded so it is not re-checked:

- **Shared memory:** neither kernel declares any (`grep __shared__` -> no hits). The file header
  says so explicitly: rows are shared inside a warp so no inter-warp communication is needed.
- **Fixed-size arrays keyed on rows:** none. `all_group_scores[NUM_GROUPS]` /
  `group_selected[NUM_GROUPS]` in the sigmoid kernel are per-thread and sized by expert groups.
- **Register pressure / spills:** measured, §3.3. Zero spills everywhere, pre and post.
- **Scheduler task count:** the router is `grid_dim=(1,1,1)` at every `mbt`, so raising `mbt` adds
  passes inside one task, not tasks. (Other stages *do* scale with `mbt` — that is §5, not a cap.)
- **`MPK_MAX_NUM_BATCHED_TOKENS`:** consumed only by `prepare_next_batch` admission arithmetic
  (`persistent_kernel.cuh:324-651`) and `admission_policy.h`; no array is sized by it.

---

## 2. Bit-exact for `num_rows <= ROWS_PER_CTA`, by construction

### 2.1 The argument (softmax)

Three independent reasons, all source-level:

1. **Trip count 1.** `ceil(num_rows / ROWS_PER_CTA) == 1` whenever `num_rows <= ROWS_PER_CTA`, so
   `row_tile_base` is identically `0` and `thread_row = 0 + warp_base_row + thread_row_in_warp`
   *is* the pre-change expression.
2. **The body is the pre-change body verbatim.** `git diff -w` on the kernel shows only the added
   loop header and comments — every operation, operand, order, and shuffle mask inside is
   unchanged. `live_rows` moved above the loop but is loop-invariant and its expression is
   untouched.
3. **`num_rows` is a compile-time literal at the real call site.** `task_register.cc` emits
   `batch_size` as an integer literal into the generated `topk_softmax_task_impl(...)` call, and
   the function is `__device__ __forceinline__`, so the loop constant-folds in the megakernel
   exactly as it does in the check TU of §3.

Above the cap the same reasoning gives correctness rather than identity: no reduction,
accumulation, or communication crosses a tile. Every reduction (max, sum, argmax) is a shuffle
within one row's sub-group; each tile writes a disjoint row slice of `output` and
`mpk_routing_indices`. The one cross-tile write is
`mpk_active_expert_ids[local_expert] = local_expert`, an idempotent mark that is order-independent
and is compacted after the loop behind the same single `__syncthreads()` as before.

### 2.2 The argument (sigmoid)

Identical (1)/(2)/(3). The extra per-row state — group top-2, group scores, the selected-group set
(Phases 2-4) — lives in registers and is rebuilt from scratch on each trip, so no tile can observe
another tile's groups.

### 2.3 The one deliberate behaviour change: the sigmoid warp mask

The pre-change sigmoid mask was

```cpp
uint32_t const warp_mask = (num_rows % 2 == 1 && thread_row == num_rows - 1)
                               ? 0x0000ffff : 0xffffffff;
```

At DeepSeek-V3's shape `THREADS_PER_ROW == 32`, so a row spans the whole warp and restricting to
lanes 0-15 leaves lanes 16-31 calling `__shfl_xor_sync` with a mask that excludes them — undefined
per the CUDA C++ programming guide. It is also wrong once tiles exist: an odd `num_rows` above
`ROWS_PER_CTA` leaves a *full-width* row in a later tile.

It is replaced with the softmax sibling's form, which is compile-time-disabled unless a warp
really does hold two sub-groups. For `THREADS_PER_ROW == 32` that is `0xffffffff` unconditionally.

**Scope of the change:** exactly the odd-`num_rows` DeepSeek-V3 launches. Even `num_rows`
(the only sizes any committed test or graph used besides 1) is bit-identical — confirmed by SASS
in §3.2. `num_rows == 1` is the one previously-exercised odd case (`test_gate_topk_sigmoid.py`
`BATCH_SIZES[0]`); it passed only because the hardware happened to do the defined thing.

---

## 3. Compile-check evidence (B200, no GPU touched)

Scratch tree `~/mpk-qwen35/scratch-i5b/` — a fresh directory, not the window's checkout. The
pre-change tree came from `git archive HEAD include`, the post-change tree from the working tree;
CUTLASS headers were **copied** (read-only) out of `~/mpk-qwen35/mirage-clean/deps/cutlass`.
No CUDA API call, no launch, no python on the box beyond the two SASS-diff scripts.

The compiled `post` tree is byte-identical to what this commit lands — verified by sha256:

```
bc7f6859f4b12ceea766891b33bbf3a2aa0fba1736eb4bea29ff23e951e231e0  topk_softmax_sm100.cuh
29b40066c8d746fa53ad2588618b8abff0bf29ec96c4579d07b7423e8ca4565e  topk_sigmoid_sm100.cuh
```

### 3.1 Commands

```
export PATH=/usr/local/cuda-12.8/bin:$PATH        # nvcc 12.8, V12.8.93
BASE="-O3 -std=c++17 -gencode=arch=compute_100a,code=sm_100a \
  -DMPK_ENABLE_TMA -DMIRAGE_GRACE_BLACKWELL -DMIRAGE_BACKEND_USE_CUDA \
  -DMIRAGE_FINGERPRINT_USE_CUDA -DMPK_TARGET_CC=100 -DMODE_OFFLINE \
  --expt-relaxed-constexpr -Xcudafe --diag_suppress=177"
# two flag sets: `fast` adds -use_fast_math (the shipped JIT default,
# persistent_kernel.py:290); `nofast` is the MPK_NO_FAST_MATH=1 lane.
nvcc $BASE [$FAST] -I<tree>/include/mirage/persistent_kernel \
     -I<tree>/include/mirage/persistent_kernel/tasks -I<tree>/include \
     -I<cutlass>/include -I<cutlass>/tools/util/include \
     {-c | -cubin} tu_{small,odd_sigmoid,big}.cu -o ...
cuobjdump -sass <cubin>
```

Three TUs, each instantiating the kernels with `num_rows` as an integer **literal**, matching how
the megakernel sees it:

- `tu_small.cu` — every `num_rows <= ROWS_PER_CTA` case, i.e. everything the pre-change kernel
  already handled. Includes the shipped Qwen3.5 router with the M3-I8 live-row gate on.
- `tu_odd_sigmoid.cu` — DeepSeek-V3 sigmoid at `num_rows` 1/3/7 (the §2.3 UB fix).
- `tu_big.cu` — the rows the pre-change kernels dropped: 9/16/17/33/64/128/129 across all shapes,
  plus the `mbt` candidates of §5 with the M3-I8 gate on.

### 3.2 Result — 24/24 compiles clean, shipped instantiations byte-identical

**`nvcc -c` and `nvcc -cubin`: rc=0 for all 3 TUs x {pre, post} x {fast, nofast} = 24 compiles.**

`tu_small`, per-function SASS after stripping instruction addresses (script:
`scratch-i5b/perfunc.py`):

| flag set | functions identical | differing |
|---|---|---|
| `fast` (shipped) | **13 / 14** | `k_softmax<VPT=8, E=256, rows=1>` |
| `nofast` | **13 / 14** | `k_softmax<VPT=8, E=256, rows=1>` |

Byte-identical SASS under **both** flag sets, including every instantiation any committed graph
builds:

- `k_softmax_gated<16, 256, 16>` — **the shipped Qwen3.5 router** (`MOE_GATE_PADDING_ROWS=True`)
- `k_softmax<16, 256, 16>` — the `MOE_GATE_PADDING_ROWS=False` A/B arm
- `k_softmax<16, 256, {1, 15}>`, `k_softmax<8, 128, {1, 15, 16}>` (Qwen3 30B-A3B shape),
  `k_softmax<8, 256, {7, 8}>`
- `k_sigmoid<{2, 4, 8}>` — DeepSeek-V3 at even row counts

The single outlier, `k_softmax<VPT=8, E=256, rows=1>`, is reachable only from the `sm100_moe`
unit-test wrapper (`TopkConstants` picks VPT=8; the Qwen3.5 build resolves to VPT=16, and DSV3 uses
the sigmoid router). It was analysed instruction by instruction
(`scratch-i5b/{opcheck,editscript,numcheck}.py`) — under **both** flag sets **every** value- or
memory-affecting opcode count is equal:

```
fast:  FADD.FTZ 42/42  FMUL.FTZ 53/53  FMNMX.FTZ 11/11  FMNMX3.FTZ 3/3  FSEL 16/16
       FSETP.{GEU,GT,NEU} 9/9 7/7 9/9  MUFU.EX2 16/16  MUFU.RCP 6/6
       SHFL.BFLY 40/40  SHFL.IDX 5/5  LDG.E 34/34  STG.E 58/58
       ATOMG.E.ADD 5/5  BAR.SYNC 2/2  VOTEU.ANY 5/5
       only difference: LDC.64 16->13 / LDCU.64 11->13  (kernel-param constant loads,
       regular vs uniform datapath)
```

The residual ordering difference is the eight independent softmax lanes' `FADD (x-max)` and
`FMUL (*log2e)` being interleaved differently — each lane's `FADD -> FMUL -> MUFU.EX2` chain is
intact. That is scheduling of independent chains, not reassociation. Recorded as a known,
non-shipped delta rather than claimed as identity.

`tu_odd_sigmoid` and `tu_big` differ in every function, which is the point: those are the cases the
pre-change kernels got wrong.

### 3.3 Resources (`ptxas -v`, pre/post)

| TU | registers | spills | stack |
|---|---|---|---|
| `tu_small` (all 13 fns) | **identical** (32 or 39/40) | 0 / 0 | identical |
| `tu_big` softmax | 39->40 worst case | 0 / 0 | 0 / 0 |
| `tu_big` sigmoid | 32->38/39 | 0 / 0 | 96 / 96 (pre-existing local arrays) |

No spills anywhere. Occupancy for every shipped instantiation is unchanged by construction (same
SASS).

### 3.4 CPU checks run green

- `python3 demo/qwen3_5/accept/opt/m3i8/static_checks.py` -> `ALL STATIC CHECKS PASS`, rc=0, before
  and after the change (S3 skipped: no compiled graph locally, as designed).
- `ast.parse` on all eight edited Python files.

**No CPU-runnable unit test exists for these two kernels.** Every test in
`tests/runtime_python/blackwell/sm100_moe*/` builds a CUDA extension or a `PersistentKernel` and
needs a device. `static_checks.py` is a source-level gate, not a kernel test. Nothing was invented
to fill the gap.

---

## 4. Pre-registered predictions and falsifiers for the GPU window

Run in this order; each has a stated falsifier. `PY=$HOME/mpk-qwen35/venv-mpk/bin/python`,
`MIRAGE=$HOME/mpk-qwen35/mirage`, `export PATH=/usr/local/cuda-12.8/bin:$PATH`,
`export HF_HOME=$HOME/mpk-qwen35/hf`, under the usual GPU guard.

### P0 — the unit tests, cheapest first

```
cd $MIRAGE/tests/runtime_python/blackwell/sm100_moe          && $PY setup.py build_ext --inplace && $PY test_gate_topk.py
cd $MIRAGE/tests/runtime_python/blackwell/sm100_moe_sigmoid  && $PY setup.py build_ext --inplace && $PY test_gate_topk_sigmoid.py
cd $MIRAGE/tests/runtime_python/blackwell/sm100_moe_sigmoid  && $PY test_topk_sigmoid_testmode.py
cd $MIRAGE/tests/runtime_python/blackwell/sm100_moe_block_qwen35 && $PY setup.py build_ext --inplace && $PY test_router_oracle.py
```

**Predicted:** all four exit 0.

- `test_router_oracle.py` section 4 is **inverted** by this change. It used to *assert the bug*:
  `assert int(r8[:, 8:].sum().item()) == 0` ("VPT=8 is expected to leave rows 8..15 unrouted").
  It now asserts that VPT=8 and VPT=16 agree bit-for-bit on all 16 rows, and adds 1/7/9/17/33-row
  cases at both VPTs.
- Sections 1-3 of that test (HF-oracle expert sets, fp32 weights, bf16 bit-exact
  `topk_renorm_weights`) run on <=16-row dumps and are the direct check of §2's identity claim.
  **Falsifier:** any section 1-3 failure means the loop is not bit-exact and the change is void.
- `test_gate_topk_sigmoid.py` at `batch_size=1` is the §2.3 UB case. **Falsifier:** if it regresses,
  the mask rewrite is wrong (not merely the old behaviour restored).
- These four need only a device, not the checkpoint — run them first, they are minutes.

### P1 — the live >16-row probe, against a committed pre-change artifact

`results/probe_prefill.py` already takes `--mbt` and dumps the argmax of every prefill row. The
committed `results/probe_prefill_mbt32.json` is a **pre-change run at mbt=32** on `p01-history`
(prompt_len 30) and shows the cap directly: rows 0-15 hold plausible next-token predictions, rows
16-29 collapse to the degenerate token 248046, and `row_with_reference_first_token` is `null`.

```
cd $MIRAGE/demo/qwen3_5/accept/results
$PY probe_prefill.py --prompt-id p01-history --mbt 32 --rows 16 \
    --out probe_prefill_mbt32_i5b.json
```

**Predicted (P1):** rows 16-29 stop being 248046, and
`row_with_reference_first_token == 29` (`= plen - 1`), i.e. the last prefill row now predicts the
reference's first token 90700. **Falsifier:** rows 16+ still degenerate -> the loop does not
actually route the later tiles in the real graph (as opposed to the check TU).

This is the single highest-information run in the plan: it is one prefill pass, it has a committed
pre-change baseline in the tree, and it fails loudly.

### P2 — per-case AC-3 byte-diff at the DEFAULT config (the non-regression gate)

Unchanged config, `mbt=16`, exactly the M3-I2b protocol:

```
cd $MIRAGE/demo/qwen3_5/accept
for BS in 1 2 4 8 16; do
  $PY -u mpk_engine_run.py --batch-size $BS --max-seq-length 132 \
      --out-dir  $HOME/mpk-qwen35/m3i5b/ac3_dumps \
      --kernel-dir $HOME/mpk-qwen35/m3i5b/kernel_ac3_bs$BS
done
$PY -u harness/run_ac3.py --engine-dump-dir $HOME/mpk-qwen35/m3i5b/ac3_dumps \
    --batch-sizes 1,2,4,8,16 --output-json $HOME/mpk-qwen35/m3i5b/run_report.json
$PY -u $HOME/mpk-qwen35/m3i2a/bytediff.py \
    $MIRAGE/demo/qwen3_5/accept/results/dumps_final \
    $HOME/mpk-qwen35/m3i5b/ac3_dumps 1,2,4,8,16
```

**Predicted (P2):** `identical` for every (prompt, bs) case at all five batch sizes — the strong
form, because §2 and §3.2 say the shipped router SASS did not move a byte. **Falsifier:** ANY token
change. That is not a tolerance question: it would mean the `<=16`-row path is not identical after
all, and the correct response is cast-position root-cause (the M2 rule), not a waiver.

### P3 — CI smoke

```
cd $MIRAGE/demo/qwen3 && $PY -u demo.py --use-mirage --max-new-tokens 50 \
    --max-num-batched-tokens 8 --max-num-batched-requests 1
```

Qwen3-8B is a 128-expert softmax-router graph at `mbt=8`, i.e. `k_softmax<8,128,8>` —
byte-identical SASS per §3.2. **Predicted:** unchanged 50-token output. **Falsifier:** any token
change (would contradict §3.2 directly).

### P4 — the mbt A/B (see §5 for why it is a question, not a landing)

Two arms, same commit, differing only in `--mbt`. Nothing in the tree changes between arms — `mbt`
is already a CLI knob, which is why no default is touched here.

```
# arm A (baseline, today's default)
for BS in 1 2 4 8 16; do
  $PY -u mpk_engine_run.py --batch-size $BS --mbt 16 --max-seq-length 132 \
      --out-dir $HOME/mpk-qwen35/m3i5b/ab/mbt16 \
      --kernel-dir $HOME/mpk-qwen35/m3i5b/ab/kernel_mbt16_bs$BS
done
# arm B / C: --mbt 32, then --mbt 64, same loop, separate out-dir + kernel-dir
```

Protocol, per the M3 milestone rule and the I6/M3-I2b machinery: warmup + **>=3 reps per cell**,
median + dispersion, no profiler (`meta_noprof` lane), one GPU claimed under
`opt/m3i2b/gpu_guard_m3i2b.sh`-style 3-sample stability check. Report per-bs median tok/s with
dispersion, not a single run.

**Pre-registered predictions:**

- **P4a (correctness first).** At every `mbt`, the AC-3 per-case byte-diff versus `dumps_final`
  stays `identical`. Rationale: a live row's top-k is a reduction over its own 256 logits, and the
  grouped GEMM gathers rows in ascending token order, so changing how many padding rows exist does
  not move a live row's arithmetic — the same argument M3-I8 already validated at bs 1..16.
  **Falsifier:** any token change at mbt>16 means `mbt` is *not* a pure scheduling knob and the A/B
  stops there.
- **P4b (the router's own cost).** Router worker time scales ~linearly in `mbt`. Measured baseline
  (`opt/pertask_by_bs.csv`, `TASK_MOE_TOPK_SOFTMAX_SM100`): 40 tasks/step, mean 14.1-15.8 us,
  564-631 us/step total, concurrency 9.0. Both the row loop and the Phase-0 zeroing
  (`NUM_EXPERTS * num_rows` int writes) are O(rows), so predict 2.0x +/- 0.3 at mbt=32 and
  4.0x +/- 0.6 at mbt=64, i.e. ~1.2 ms and ~2.5 ms per step. **Falsifier:** superlinear growth
  (would indicate the loop, not the arithmetic, is the cost).
- **P4c (the reason to do it at all) — already computed, CPU only.** `opt/schedule_sim.py` replays
  admission exactly (109/109/109/111/203 predicted == observed at bs 1/2/4/8/16), so the iteration
  count at other `mbt` is a *derived* number, not a guess. Sweeping the committed
  `opt/meta/meta_bs*_rep0.json`:

  | bs | iters @16 | @32 | @64 | decode_full @16 -> @32 | Delta iters @32 |
  |---:|---:|---:|---:|---:|---:|
  | 1  | 109 | 108 | 108 | 107 -> 107 | **-0.9%** |
  | 2  | 109 | 108 | 108 | 101 -> 101 | -0.9% |
  | 4  | 109 | 108 | 108 | 96 -> 98   | -0.9% |
  | 8  | 111 | 108 | 108 | 82 -> 88   | -2.7% |
  | 16 | 203 | 123 | 114 | **0 -> 54** | **-39.4%** (-43.8% @64) |

  ```
  cd demo/qwen3_5/accept/opt && python3 -c "
  import json,sys; sys.path.insert(0,'.')
  from schedule_sim import simulate,label
  m=json.load(open('meta/meta_bs16_rep0.json')); p=m['prompt_lens']
  s=p+[p[i%len(p)] for i in range(len(p),m['batch_size'])]
  print([ (mbt, simulate(s,mbt,m['max_seq_length'])['n_iterations']) for mbt in (16,32,64)])"
  ```

  The gain is **entirely at bs16** and matches backlog rank 4's +44% wave-level ceiling.
  **Falsifier:** if the measured bs16 wave time at mbt=32 does not fall at all, the iteration model
  is not the binding mechanism and rank 4 is mis-attributed.
- **P4d (the cost that could eat it).** Every per-token stage computes `mbt` rows whether or not
  `mbt` tokens exist. After M3-I2b, `quantize_fp8` is row-split (`grid_dim=(mbt,1,1)`,
  `QUANTIZE_ROW_SPLIT`), so its task count is proportional to `mbt`: 3840 tasks/step at mbt=16 ->
  ~15360 at mbt=64, on a stage that is already 29.7% of the bs1 step wall. The dense fp8 GEMMs
  (86-87 ms worker time, flat across bs) are also mbt-shaped. Combined with P4c, which says bs1-bs4
  buy **0.9% fewer iterations** for that extra per-iteration work, predict **mbt>16 is a net LOSS
  at bs1-bs4**, roughly break-even at bs8, and a win only at bs16. **Falsifier:** a bs1 improvement
  at mbt=32 would mean the padding cost model is wrong and the whole M3 flat-cost story needs
  revisiting.

**Disposition rule.** This issue lands the row loop. The `mbt` default moves only if P4a holds and
a bs-weighted median improvement survives the >=3-rep dispersion — otherwise the terminal
disposition is *rejected-with-evidence* or *blocked-with-reason*, recorded in `backlog.json`, and
the default stays 16.

---

## 5. `mbt` — the knob, what the loop unlocks, and the open question

**Knob locations** (none changed):

| where | value | note |
|---|---|---|
| `demo/qwen3_5/accept/mpk_engine_run.py:127` `DEFAULT_MBT` | 16 | AC-3 / opt harness default; `--mbt` overrides |
| `demo/qwen3_5/demo.py:43` `--max-num-batched-tokens` | 16 | demo default |
| `persistent_kernel.py:320` `-DMPK_MAX_NUM_BATCHED_TOKENS=<mbt>` | compile-time | the only place it reaches the runtime |
| `persistent_kernel.cuh:324-651`, `admission_policy.h` | consumer | greedy slot-order admission budget |

**What the row loop unlocks.** Before: `mbt <= 16`, asserted, because a wider chunk silently
corrupted routing. After: any `mbt`, at `ceil(mbt / 16)` router passes. The remaining hard bounds
are unrelated to the router — `mbt >= mbr` (surplus requests would stall) and the memory of the
`[mbt, hidden]` / `[mbt, topk, inter]` intermediates. Candidate values: **32, 64, 128** (2, 4, 8
router passes). The AC-3 prompts are 24-68 tokens, so `mbt >= 68` is the first value that makes
every AC-3 prefill single-chunk.

**The profitability question, stated so it can be answered.** Raising `mbt` trades

- **gain:** admission headroom. At bs16 `mbt == batch size`, so decode saturates the budget and
  prefill starves one token at a time; the wave never reaches a decode steady state (108 of 203
  iterations mixed, 95 draining, first request retires at iteration 101). Backlog rank 4.
- **cost:** every mbt-shaped stage does more padding work — quantize (row-split, so O(mbt) tasks),
  the dense fp8 GEMMs, the MoE grid, and now the router itself at O(mbt).

M3-I8 partially decouples the largest of these: with pad-row gating on, expert *activation*
follows live rows, not `mbt`. That is what makes the question worth asking now and not before.
It is genuinely open which side wins at each batch size, and P4c is checkable in the simulator
before any GPU time is spent.

---

## 6. What this prep does NOT establish

- Nothing here is hardware-validated. Every claim in §2 is source-level; §3 is compile-only.
- `k_softmax<VPT=8, E=256, rows=1>` is not SASS-identical (§3.2). Argued equivalent from identical
  value-opcode multisets; not proven by execution.
- The §2.3 sigmoid mask change is a real behaviour change for odd-row DeepSeek-V3 launches. The
  old behaviour was undefined, so "unchanged" was never available; P0 is the check.
- **Adjacent hazard, pre-existing, NOT fixed here:** the Phase-7 compaction in both kernels reads
  `mpk_active_expert_ids[local_expert]` and then writes `mpk_active_expert_ids[pos]` with `pos`
  from an `atomicAdd`, with no `__syncthreads()` between the read and the write. A thread whose
  mark has not been read yet can have it clobbered by another thread's compacted write. It is
  order-dependent, not row-loop-specific, and more rows mean more marks and more opportunities.
  Flagged for its own issue; the `mask[NUM_EXPERTS] == unique(ids)` assertion in
  `test_router_oracle.py` and `test_gate_topk_sigmoid.py` is what would catch it.

---

## 7. Prep review outcome (codex gpt-5.6-sol, 2026-07-27)

`PREP-REVIEW: PASS` — full reply in `review-prep-codex.txt`. Zero blocking findings; the
bit-exactness argument, the sigmoid mask semantics (including the tile-boundary odd-row case),
the no-divergent-barrier check, and the race triage were each independently confirmed.

**Binding constraint from finding 5:** any `mbt > 16` DEFAULT change now requires either
M3-I5c (the Phase-7 compaction race) fixed and validated first, or the window adds a
high-repetition compaction stress test at the proposed maximum row count. P4's disposition
rule inherits this prerequisite.

---

## 8. mbt terminal disposition (2026-07-28): REJECTED-WITH-EVIDENCE (superseded)

The P4 A/B is dispositioned without a GPU run, on the pre-registered models plus what landed
since: (a) P4c's exact-replay simulation put the ENTIRE mbt>16 gain at bs16 (-39.4%
iterations at mbt=32; bs1-8 gains <= 2.7%); (b) M3-I9's admission cap landed bs16-
conditionally and delivers MORE than mbt would (203 -> 131 iterations, +84.2% at the AC-3
geometry, +14.1% matched e2e) with ZERO of the O(mbt) padding cost mbt adds to every
mbt-shaped stage; (c) P4d's padding-cost model predicts a net LOSS at bs1-4 — and its key
premise was later confirmed by measurement (quantize row-split landed; the remaining
mbt-shaped stages persist). The M3-I5c stress prerequisite (finding-5) is now MET
(5000/5000 across rows 16-128), so this rejection is on the merits, not on safety.
REOPEN CONDITION: a workload class needing >16-token prefill chunks per request (beyond
AC-4's pinned geometry) — the row loop already enables it and this protocol stands ready.
