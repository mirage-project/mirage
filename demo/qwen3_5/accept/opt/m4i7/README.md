# M4-I7 — the ferret MoE grouped-GEMM winner, integrated

Tasks **241** (`TASK_MOE_W13_FP8_BLOCKSCALE_SM100`) and **242**
(`TASK_MOE_W2_FP8_BLOCKSCALE_SM100`) now run the ferret
`moe-fp8-grouped-vllm-beat` winner from `~/mpk-qwen35/ferret/workspace3` tag
**v012** (`c8b5b24`, "w2-only clamped-PATH2 2-wave dispatch, 84.7KB alloc,
2 CTAs/SM"). Both tasks instantiate one template in one file, so this is a single
header change plus a dispatcher.

**The loop is running again on workspace3** (re-seeded onto a cross-expert
tile-packing axis). This port takes the **v012 tag blob** — `git show
v012:kernel.cu`, 1561 lines, md5 `9387953d5dc0bc6061031d76b553f219` — never the
worktree file, and nothing was written into the workspace.

## Why integrate a candidate that did not clear its bar

v012's `min_ratio` is **0.801** against a 1.3333 target, so it does **not** beat
FlashInfer/vLLM standalone. Verified, reading the ratio line rather than the raw
pairs (the metric is throughput-style, higher is better — reading the pairs as
latency inverts the conclusion):

```
$ python3 -m ferret.state ferret/workspace3 ferret/workspace3/task.yaml
  score : 0.801 (via min_ratio)   worst_config : w13_bs2
  w13  98.9 / 80.1 / 94.5 / 86.9 / 85.6 %   at bs 1/2/4/8/16
  w2   96.1 / 157.3 / 152.5 / 118.6 / 100.3 %
```

Two w2 configs already beat vLLM by >50%. More to the point, the stage it feeds
is 42.6% of the bs1 critical path (w13 27.6% + w2 15.0%, `opt/m4i5/`), and
M4-I5's decomposition put those stages at span/perfect-pack ratios of **7.36**
and **7.07** — overwhelmingly **width**-bound, not kernel-bound. That is the
prediction this integration was taken on, and it is what the measurements below
confirm.

### One correction to the launch brief's framing, stated up front

The brief's "~1.76x faster than what MPK ships today" comes from v012's 0.801
against v001's 0.456. That comparison is **not** apples-to-apples: the ferret
loop's **v006 was a measurement-only tag** (candidate scoring moved from
per-launch `cudaEvent` windows, which carry this box's ~6.17 us submission floor,
to device `%globaltimer` wall spans; and the reference's L2 flush moved from
write-only to read-only). It moved `min_ratio` 0.476 → 0.692 with no device-code
change. So roughly a third of the 0.456 → 0.801 travel is instrument, not kernel.
The kernel-only travel in wall-span terms is closer to **1.6x at w13_bs1** and
much less elsewhere.

None of that changes what to do, because the in-MPK A/B is the arbiter and it is
reported below — but the 1.76x figure should not be quoted.

## What landed, and the one structural difference

The pre-M4-I7 body is preserved **byte-for-byte** as
`kernel::golden::moe_fp8_blockscale_task_impl`. Not asserted, *proven*:
`scripts/check_golden.py` re-extracts the frozen region from HEAD and from
`git show 5e48eaab:<path>` and compares — 13169 bytes, sha256
`298aa9c455f4e7885f9ef86af45259a8448a6f427b9c0b70987f180de12033e3`, and the
header advertises that same hash. It also refuses a golden region that has grown
any reference to the fast machinery.

`scripts/gen_header.py` builds the shipped header from three provable pieces (the
frozen region, the v012 `cand` region, and the new dispatcher) and prints the
byte counts and hashes it copied. The v012 region is verbatim apart from:

- the constants namespace renamed `moe_fp8_blockscale` → `moe_fp8_blockscale_fast`
  so it can coexist with the frozen golden one (5 occurrences);
- the vendored `.cg` cp.async / mbarrier / bulk helpers moved from `kernel::`
  scope into that namespace, so the megakernel TU — which concatenates every task
  header — gains **no** new name at `kernel::` scope;
- the smem `static_assert` retargeted from `smem_bytes` (an upper bound) to
  `smem_bytes_k` (the clamped layout the kernel actually addresses);
- its dispatcher **replaced**. That is the structural difference.

### The structural difference: MPK's parallel unit is the emitted task, not the SM

The fast body walks a **flattened** work space —
`wi ∈ [expert_offset, num_activated · NUM_N_BLOCKS)` step `expert_stride`, with
`ae = wi / NUM_N_BLOCKS`, `nb = wi % NUM_N_BLOCKS` — where the golden body walked
experts and, inside each, all N blocks serially.

This is the load-bearing MPK decision, and the mirror of M4-I2's per-shape N
slice. `task_register.cc` emits `expert_stride = grid.x = min(num_experts,
mbt·topk) = 128` tasks per (layer, stage, N-split), but the golden body gave each
task one expert, so only `num_activated` of those 128 ever had work and the rest
exited immediately. Flattening spreads the same work over
`num_activated · NUM_N_BLOCKS` of the **already-emitted** tasks.

It is coverage-correct for any `expert_offset ∈ [0, expert_stride)`
(⋃ₒ {o, o+s, o+2s, …} = [0, total)), each work item writes a disjoint set of
output elements, and the per-column K accumulation order is untouched — so it is
bit-identical, which Gate 1 measures rather than assumes.

**It is a different lever from `MPK_MOE_N_SPLITS`,** and strictly better on two
counts. That knob multiplies the *emitted* count too (256 → 1024 at k=8), so
every worker pays ~7 dead dispatches per level instead of 1, and it shrinks the N
tile. M4-I5 measured it at ×1.11 at bs1 and a **regression** at bs16, and left it
default-off. Flattening costs no extra dispatch and keeps TILE_N at 128 on the
paths that use it. `moe_n_splits` stays at its shipped 2 here; the two levers are
independent and this issue does not touch the knob.

### The fetch-path rule: swept, not reasoned

v012 chooses among three fetch paths — PATH 0 (golden fetch: one K tile per
stage, 16 B `cp.async`, plus the measured w13-only `#pragma unroll`), PATH 1
(4 K tiles per stage, 512 B `cp.async.bulk` weight rows, TILE_N=128), PATH 2
(8 K tiles clamped to K, 1 KiB bulk rows, TILE_N=64). Its gate compared work
items against `%nsmid`, because in the ferret harness **one CTA ran one work
item**, so "does the grid fit one wave" decided whether the wide-smem layouts'
1-CTA/SM residency was free.

**In MPK that denominator is meaningless.** There is exactly one persistent worker
per SM and each owns the whole dynamic smem budget (205,824 B), so residency is
fixed at 1 CTA/SM whichever path runs and the wide layouts cost nothing. Which
left the choice genuinely open — so it was **measured**, each path pinned as a
`-D` (`MPK_MOE_PATH_POLICY`), arm B only, 3 reps, arms in one GPU claim
(`tables/path_policy.txt`):

| path | bs1 per-rep ms | bs1 med | bs16 per-rep ms | bs16 med |
|---|---|---|---|---|
| PATH 0 | 825.7 / 826.3 / 826.2 | 826.2 | 3452.0 / 3497.3 / 3434.0 | 3452.0 |
| PATH 1 | 823.9 / 823.6 / 824.6 | **823.9** | 3371.2 / 3408.7 / 3355.6 | **3371.2** |
| PATH 2 | 837.1 / 837.8 / 842.8 | 837.8 | 3724.1 / 3783.1 / 3699.4 | 3724.1 |

Two results, and the second one changed the design.

1. **PATH 1 does dominate PATH 0 in MPK**, as predicted: +0.28% at bs1 and
   +2.34% at bs16. The ferret run only ever preferred PATH 0 to protect 4–5
   CTAs/SM of residency, which does not exist here.
2. **PATH 2 loses at every measured batch size** — −1.66% at bs1 and −9.48% at
   bs16. Its whole premise was halving per-CTA weight bytes to recruit a second
   wave of CTAs; in MPK the task count is fixed by the graph and the flattened
   work space already saturates it, so halving the tile only doubles the
   per-item gathers, A re-fetches and epilogues for the same MMAs.

So the **shipped rule is simply PATH 1 when admissible, else PATH 0** — no
runtime mask read, no branch on the task's critical path. An earlier iteration
shipped an adaptive `num_activated · OUTPUT_SIZE/64 ≤ expert_stride` gate that
admitted PATH 2 for w13 at bs1; the sweep retired it. Worth recording *why* that
gate looked fine locally and was still wrong: with PATH 2 on w13, that stage's
own wallspan at bs1 was **better** (588.8 us vs 673.1 us) while the whole step
was **worse**, because PATH 2's extra per-item work is worker time taken from
every other stage in the megakernel. A locally optimal stage is not a faster step.

PATH 2 stays built and bit-exact (Gate 1 covers it) and reachable through
`MPK_MOE_PATH_POLICY=2`, so the sweep can be repeated if the geometry changes —
a larger `moe_n_splits` would shrink `OUTPUT_SIZE` and change the trade.

Two fail-closed guards ride along, because the failure mode of getting this wrong
is a silent wrong answer rather than an error:

- **Admissibility is compile-time and the PATH argument is sanitised.**
  `path_admissible()` reproduces every `static_assert` inside `moe_impl_path`, and
  `safe_path()` maps an inadmissible path to 0 *at the template argument* rather
  than guarding the call. This is M4-I2's lesson applied structurally: nvcc is not
  guaranteed to leave a discarded `if constexpr` branch uninstantiated, so the
  safety rests on the dispatcher's reachability `static_assert`, which no
  compiler's instantiation eagerness can affect.
- **`%dynamic_smem_size` is re-read at run time.** A standalone launcher can hand
  the task less smem than the wide layouts need (the pybind wrapper used to size
  its launch off the golden layout alone), and a wide layout on a short
  allocation would write past the arena. The device follows the allocation
  exactly and degrades to the narrow path. `launch_smem_bytes()` is the public
  helper launchers should size with, and the wrapper now does.

### Prefill stays on the proven path

`fast_path_ok()` requires `BATCH_SIZE ≤ 16` — exactly the regime the ferret run
validated (the shipped decode `max_num_batched_tokens`). Prefill instantiates the
same template with the full batched-token count and takes the golden body, which
AC-5 depends on. The generated TU confirms the decode instantiations are
`<bfloat16, 16, 8, 256, 512, 1024, 2048, true>` (w13) and
`<..., 1024, 2048, 512, false>` (w2), i.e. `moe_n_splits = 2` and `BATCH_SIZE = 16`.

## The five non-negotiables, and how each was verified

**1. Preserved fp32 block scales on the warp-MMA path — no drift to a requant
path.** Verified three ways. Structurally: every path calls the same
`mma_m16n8k32_e4m3_f32` (unscaled `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32`,
FP32 accumulate) and folds `d_input_scale[...] * b_scale_row[kt]` — both float32,
both read straight from the checkpoint's `weight_scale_inv` and the fp32-scale
quantizer — into an FP32 accumulator once per 128-element K tile. No `>> 23`,
no ue8m0, no per-row collapse, no `repeat_interleave`. Mechanically: Gate 1 is
bit-exact against the golden body, which is itself the named fail-closed
alternative to the ue8m0 grouped kernel that FLOORS exponents. And behaviourally:
the pre-existing **scale-consumption control** passes in both flag lanes — it
multiplies one expert's weight block scale by **1.3**, a non-power-of-two, and
requires the output to follow exactly, which a kernel that had drifted onto a
ue8m0 path provably cannot do.

**2. The M3-I8 gating contract.** Activation still follows live rows: the gather
reads `d_routing[expert·BATCH_SIZE + t]` and admits only `slot > 0`, and the
router (`topk_softmax_sm100.cuh`) is what writes non-routing values for padding
rows — the grouped GEMM never took `num_active_rows` and still does not, so the
contract is unchanged by construction. The task-level early exit for dead groups
**strengthens**: `num_rows == 0 → continue` survives verbatim, and the new
`wi < total_work` bound means a task whose offset is past the live work does
nothing at all rather than walking the expert list. Gate 1's coverage audit
checks the other half of the contract directly: with a 0xEE poison fill, every
routed `(token, slot)` row must be written and every **non**-routed row must
still read 0xEE. 0 unwritten, 0 clobbered across all 160 arms.

**3. The integer-division scale-slice hazard (M4-I2 on task 279).** **Not
reachable here, and that is a property of the design rather than luck.**
`OUTPUT_SIZE % BLOCK_N == 0` still holds (static_assert), so the per-task N slice
is unchanged and `weight_scale`'s grid split stays the exact division
`task_register.cc` already asserts (`dim(1) · 128 == output_size`), with
`moe_fp8_blockscale_layer` asserting `weight_scale.dim(1) · 128 ==
weight_fp8.dim(1)` above it. `moe_n_splits` is untouched at 2, and the builder
already refuses a value that does not divide `N/128`. PATH 2's TILE_N=64
subdivides only *inside* a task, **below** the scale block: it reads the row of
its containing 128-column block (`n0 / BLOCK_N`), so the scale a column sees does
not depend on the tiling — and `BLOCK_N % path_tile_n(p) == 0` is an
admissibility condition, so a tiling that straddled a scale block could not
compile. Gate 1 runs the `moe_n_splits = 2` geometry explicitly (`OUT_N=512`
w13 / `OUT_N=1024` w2, two slices, pointer-offset per slice) alongside the full-N
one, so the sliced path is measured and not assumed. Row replication was read
(`opt/m4i2/`) before designing and is **not needed**: this task never asks for a
sub-block slice.

**4. Golden preserved byte-for-byte with a fallback for what the candidate does
not cover.** `check_golden.py` (above). Fallback coverage: prefill
(`BATCH_SIZE > 16`), any shape where all three paths are inadmissible, and
`MPK_MOE_BLOCKSCALE_BASELINE=1`. The dispatcher `static_assert`s that a shape the
fast paths reject is one the golden path can run, so a mis-wired caller fails the
**build**, not a numeric check at run time.

**5. `-Xptxas -v` on the generated megakernel TU, before vs after.** Gate 2 below.
On v012's "84.7KB alloc / 2 CTAs per SM" occupancy claim: it does **not** transfer,
and the integration does not rely on it. MPK runs one 256-thread worker per SM
with the full dynamic smem budget by design, so there is no second resident CTA to
win — which is exactly why the ferret dispatcher's residency-based gate had to be
replaced with the `expert_stride` rule. The clamped 84,660 B layout is still what
w2's PATH 2 addresses (that part is real and is what `smem_bytes_k` accounts), it
just buys width in MPK rather than occupancy.

## Provenance

Isolated clone `~/mpk-qwen35/mirage-m4i7` at `origin/qwen3-5_support` =
**`5e48eaab`**, with its **own freshly built** C++ extension
(`scripts/setup_m4i7.sh`; the stale-extension trap). Deliberately **pre-M4-I6**:
the concurrent router integration landed in the shared working tree while this
issue ran, and keeping the measurement base at `5e48eaab` makes the A/B
single-variable. The cross-issue register interaction is measured separately
below.

## Gate 1 — bit-exactness, both nvcc flag lanes

`scripts/bitexact_standalone.cu`, torch-free so it can be built by the **shipped**
JIT toolchain (nvcc 12.8; the box's torch is `+cu130` and
`torch.utils.cpp_extension` refuses a CUDA-major mismatch, so the pybind harness
can only certify 13.0 — M4-I2 hit the same wall).

Each case computes the same output twice and compares **bitwise**: golden at one
CTA / `expert_stride = 1`, against each `moe_impl_path<PATH>` and the shipped
dispatcher at the **MPK dispatch geometry** (`expert_offset = blockIdx.x`,
`expert_stride = gridDim.x = 128`, which is what `task_register.cc` emits). A
0xEE poison fill plus the coverage audit means an unwritten element cannot pass
by matching a zeroed buffer.

Arms = {PATH 0, PATH 1, PATH 2, dispatcher} × {w13, w2} × {full-N,
`moe_n_splits`=2} × n_live {1, 2, 4, 8, 16} × regimes {**E** exact-by-construction
small-integer fp8 with n/8 scales, so bit-exactness is independent of FMA
contraction and a mismatch is a real port defect; **R** full-range random, which
additionally requires identical compiler contraction}.

| lane | nvcc | arms | result |
|---|---|---|---|
| no fast-math | **12.8**, the shipped JIT toolchain | 80 | **80/80 bit-exact**, 0 unwritten, 0 clobbered |
| `-use_fast_math` (**what ships**) | **12.8** | 80 | **80/80 bit-exact**, 0 unwritten, 0 clobbered |

The two lanes' run logs are byte-identical (md5 `04a0ff82df73bcf31d8fed1ee4739d3a`),
and the gate was re-run unchanged after the dispatch rule was simplified
(`gates/gate1/run_*.log` are the final, post-simplification logs).
Measured layouts, on the real 205,824 B budget: w13 PATH0 41,652 / PATH1 152,244 /
PATH2 166,580 B; w2 PATH0 41,652 / PATH1 152,244 / PATH2 (K-clamped) 84,660 B.

### Gate 1b — the pre-existing unit and oracle instruments

Both flag lanes, via a `MOE_TEST_FAST_MATH` switch added to both MoE test setups
(they had none, so they were only ever validating the non-shipped arithmetic).

| test | no fast-math | fast-math |
|---|---|---|
| `test_moe_fp8_blockscale` (floor + **1.3× scale-consumption control** + routing control) | PASS | PASS |
| `test_moe_fp8_blockscale_ckpt` (real checkpoint) | PASS | PASS |
| `test_moe_block_oracle` | PASS | PASS |
| `test_router_oracle` | PASS | PASS |
| `test_sigmoid_gate_mul_add` | PASS | PASS |
| `test_quantize_fp8_f32scale_moe` | PASS | **FAIL — pre-existing, proven** |

### Two instrument defects this gate exposed

1. **Both MoE test setups were compiling against the wrong shared-memory budget.**
   `MPK_TARGET_CC` was undefined, which preprocesses as 0, so `runtime_header.h`
   selected the 163 KiB fallback (163·1024 − 6·1024 = 163,840 B) instead of
   B200's real 205,824 B. Invisible while the only path needed 41 KiB; PATH 2
   needs 166,580 B, so the wrong constant made an **admissible path fail to
   build**. Fixed as `sm100_linear_fp8_blockscale/setup.py` already had to
   (`MPK_TARGET_CC=100` + `MODE_OFFLINE`) — in `sm100_fp8_moe_qwen35` and
   `sm100_moe_block_qwen35`. **10 of the 14 blackwell test setups still lack
   these**; out of scope here, and left for the coordinator as M4-I2 did.
2. **`test_quantize_fp8_f32scale_moe` fails under `-use_fast_math`, and it is not
   mine.** It asserts the quantizer's fp32 group scales equal `absmax/448`, and
   `-use_fast_math` implies `-prec-div=false`, turning that fp32 division into a
   reciprocal multiply — max |delta| **3.638e-12**. The fp8 **values** stay
   bit-identical (ULP histogram `{0: 2048}`). Proven pre-existing rather than
   argued: with the pre-M4-I7 kernel and wrapper restored and *only* the flag-lane
   patch applied, the identical assertion fires with the identical delta
   (`gate1b/base_test_quantize_fastmath.log`). It surfaced because this setup had
   no fast-math lane until now. Consequence worth recording: the shipped
   megakernel's fp32 activation scales differ from a non-fast-math build's by
   ~1 ULP, which is orthogonal to this issue and is covered end-to-end by AC-3.

## Gate 2 — `-Xptxas -v` on the generated megakernel TU

MPK inlines every task into ONE `__global__`, so ptxas allocates a **single**
register budget and stack frame for all of them: a MoE win that raised pressure
would tax every other stage, which is how M3-I6a's finding worked. The v012 body
is exactly the shape of change that does this — three PATH bodies, mbarrier state,
1 KiB staging.

The control is tighter here than M4-I2's, because it can be. M4-I2's arm-A knob
changed a template argument, so its TU differed from HEAD's. M4-I7 changes only
the header body and a `-D`, so the generated TU is byte-identical in every arm —
which means **one TU compiled six ways**, with nothing else moving. bs1 TU
(sha256 `30fba221…`), shipped JIT flags, `persistent_kernel` entry:

| arm | registers | stack | spill st | spill ld | barriers | smem |
|---|---|---|---|---|---|---|
| head (pre-M4-I7 header) | 238 | 144 B | **0** | **0** | 16 | 5824 B |
| armA (`MPK_MOE_BLOCKSCALE_BASELINE=1`) | 238 | 144 B | **0** | **0** | 16 | 5824 B |
| armB (**shipped**) | **236** | 144 B | **0** | **0** | 16 | 5824 B |
| armB PATH 0 pinned | 238 | 144 B | 0 | 0 | 16 | 5824 B |
| armB PATH 1 pinned | 238 | 144 B | 0 | 0 | 16 | 5824 B |
| armB PATH 2 pinned | 238 | 144 B | 0 | 0 | 16 | 5824 B |

**armA reproduces head exactly**, which is what licenses arm A as the e2e
baseline. The shipped arm is 2 registers *below* head with no spill introduced:
the fast body's own frame sits under the megakernel's ceiling, which some other
task sets. `__noinline__` on the three PATH bodies is load-bearing and stays —
inlining them was the ~20% codegen-pollution regression on the *untouched* legacy
path that the ferret run measured.

**Cross-issue check, because the budget is shared.** M4-I6's concurrent router
integration takes the same TU to **255 registers with a 4 B spill**. Compiling one
TU against M4-I7's MoE header **and** M4-I6's router header gives **255 registers,
112 B stack, 4 B spill stores, 4 B spill loads, 5856 B smem** — *identical to
M4-I6 alone* (`gates/ptxas/combined_i6_i7.log`). The two integrations do **not**
compound register pressure; the ceiling is entirely M4-I6's.

## Gate 4 — per-bs e2e A/B

Geometry B (the AC-4-shaped primary): synthetic 256-token prompts, `msl=353`, 96
decode steps, `mbt=16`. Three reps per (arm, bs), arms **interleaved per (bs, rep)
inside one GPU claim** so drift or a co-tenant hits both equally, a kernel dir per
(arm, bs) because the arm is a `-D` in the TU (two arms sharing a `--kernel-dir`
under `--reuse-kernel` would run one binary and report themselves identical —
M3-I7 defect 3).

30 runs, all on GPU 4. Drain gate before every rep; the audit derives the pinned
device from each run's **own** recorded `cuda_visible_devices` and its **own**
`gpu_before`, never from the guard's candidate list (M3-I7's phantom-dirty-rep
bug). **0 dirty, 0 unauditable**; observed pinned-device floors 108–131 MiB.

| bs | A base (per-rep ms) | B new (per-rep ms) | A med | B med | speedup |
|---|---|---|---|---|---|
| 1 | 1095.5 / 1093.8 / 1095.6 | 824.3 / 824.1 / 824.3 | 1095.5 | 824.3 | **1.3291×** |
| 2 | 1266.7 / 1273.5 / 1267.4 | 992.0 / 1006.9 / 997.8 | 1267.4 | 997.8 | **1.2702×** |
| 4 | 1616.0 / 1627.4 / 1624.1 | 1310.1 / 1372.9 / 1366.0 | 1624.1 | 1366.0 | **1.1890×** |
| 8 | 2507.6 / 2498.8 / 2508.6 | 2215.6 / 2143.1 / 2172.8 | 2507.6 | 2172.8 | **1.1541×** |
| 16 | 3825.4 / 3842.0 / 3825.3 | 3385.9 / 3424.0 / 3371.8 | 3825.4 | 3385.9 | **1.1298×** |

Per-rep ranges are 0.2–72.6 ms, i.e. under 3.4% everywhere and under 0.1% at bs1.

Two reps needed re-running, both recorded rather than quietly replaced.
`(A, bs8, rep2)` started with 1028 MiB held by a co-tenant that arrived between
the drain gate and `profile_wave`'s own snapshot (its original wall, 2501.1 ms,
sat between the other two A reps, so it had not moved the median). And **all of
arm B was re-run** after the path sweep retired the adaptive PATH 2 gate, so the
table above is the shipped code; the pre-simplification arm B measured 1.3284 /
1.2388 / 1.1830 / 1.1502 / 1.1264×.

**A second correctness signal, for free.** `tokens_sha256` from the two arms
matched in **all 15 (bs, rep) pairs** — Gate 1 proved the paths bit-exact on
synthetic tensors, and this is the same property through 40 real layers on real
checkpoint weights.

## The stage wallspan, the mechanism, and the path recovery

Profiled runs (profiler ON, so these are **diagnostic attributions**; the perf
claim is the `--no-profiler` A/B above), one steady-window decode iteration,
`concurrency.py`.

| stage | bs | arm | step us | work us | **WALLSPAN us** | span/step | mean conc (of 128) |
|---|---|---|---|---|---|---|---|
| w13 (241) | 1 | A | 8670.6 | 40 840.0 | **2340.7** | 0.270 | 94.2 |
| w13 (241) | 1 | B | 6072.9 | 41 104.3 | **673.1** | 0.111 | 83.9 |
| w13 (241) | 16 | A | 11 095.1 | 184 872.4 | **2864.3** | 0.258 | 59.0 |
| w13 (241) | 16 | B | 9984.2 | 149 756.0 | **1934.4** | 0.194 | 87.7 |
| w2 (242) | 1 | A | 8670.6 | 23 726.9 | **1306.2** | 0.151 | 68.3 |
| w2 (242) | 1 | B | 6072.9 | 29 556.8 | **397.7** | 0.065 | 88.5 |
| w2 (242) | 16 | A | 11 095.1 | 93 941.8 | **1566.6** | 0.141 | 56.4 |
| w2 (242) | 16 | B | 9984.2 | 92 602.5 | **1185.1** | 0.119 | 92.7 |

**Stage wallspan: w13 2340.7 → 673.1 us (3.478×) and w2 1306.2 → 397.7 us
(3.285×) at bs1; w13 1.481× and w2 1.322× at bs16.**

### Which mechanism, with numbers

**At bs1 it is almost entirely PACKING.** The stage's *work* barely moves — w13
40 840 → 41 104 us (×0.994, flat) and w2 23 727 → 29 557 us (×0.80, up 1.25×) —
while the wallspan drops **3.5×** and **3.3×**. The task count in the window is
**identical** between arms (10 240 = 40 layers × grid.x 128 × grid.y 2), because
the emitted count is fixed by the graph. What changed is how many of those tasks
have work: flattening turns dead tasks into live ones, so the same MMAs are spread
over ~`NUM_N_BLOCKS`× more workers. Mean concurrency during the w2 stage goes
68.3 → 88.5 of 128.

The **pinned-path sweep is the clean confirmation**: PATH 0 — the *golden fetch*,
16 B `cp.async`, no bulk staging at all — pinned at bs1 gives 826.2 ms against the
shipped 824.3 ms. So **at bs1 the fetch path is worth 0.3% and the flattening is
worth the other 32.6%.** Nothing about the kernel's memory pipeline explains the
bs1 win; the dispatch geometry does.

This is M4-I2's precedent (work ×1.24 up, concurrency 70.8 → 113.0, span 1.896×)
in a sharper form: same lever, larger effect, and this time with a control that
isolates it.

**At bs16 the two effects separate and both are real.** w13's work *falls* ×1.234
— there the bulk-fetch staging does contribute a genuine per-task kernel gain
(PATH 1 beats PATH 0 by 2.34% e2e) — while mean concurrency still rises
59.0 → 87.7, so the 1.481× span gain is part kernel, part width. w2's work is flat
(×1.014) and its 1.322× is pure width.

### The critical-path recovery

M4-I5 put the bs1 contributions at **2194.6 us (w13)** and **1191.0 us (w2)**.
Against the arm-A spans measured here (2340.7 / 1306.2 us — the same stages under
this issue's `msl=353` geometry, within 7–10% of M4-I5's figures):

| stage | M4-I5 bs1 contribution | recovered | share of M4-I5's figure |
|---|---|---|---|
| w13 | 2194.6 us | **−1667.7 us** | **76.0%** |
| w2 | 1191.0 us | **−908.5 us** | **76.3%** |
| both | 3385.6 us | **−2576.2 us** | **76.1%** |

And the attribution closes: the profiled step fell 8670.6 → 6072.9 us, i.e.
**−2597.7 us**, of which these two stages' wallspan accounts for **−2576.2 us —
99.2%**. Essentially the whole bs1 step gain is these two stages, which is what
licenses reading the e2e number as this change's effect rather than drift.

At bs16 the same arithmetic: w13 −929.9 us and w2 −381.5 us against a step that
fell 1110.9 us — 1311.4 us of stage gain for 1110.9 us of step, i.e. the two
stages overlap each other and with the rest of the step, so their gains do not
simply add there.

**What is left in these stages.** After the change, w13 is 11.1% of the bs1 step
and w2 is 6.5%; driving *both* to zero would still leave ~5.0 ms of the 6.1 ms
step standing. The width residual (`wallspan − work/128`) is 351.9 us for w13
(52.3% of its remaining wallspan) and 166.8 us for w2 (41.9%) at bs1, and 764.4 /
461.7 us (39.5% / 39.0%) at bs16. Read together with the sweep: at bs1 what is
left is *width*, not kernel — half of w13's remaining 673 us is arrival spread and
tail, which is M4-I5's arrival-spread floor (~33 us per level) and not something a
faster inner loop touches. bs16 has both width and kernel headroom left.

## Gate 3 — FULL AC-3 at all five batch sizes

`harness/gate_ac3_stable.sh` under the **re-pinned** criteria (goal.md,
2026-07-29), shipped arm, 10 pinned prompts, `msl=132`, 64 new tokens, a **cold
kernel compile for every rep**, 3 fingerprint-consistent reps per bs.

**Verdict STABLE.** 15 accepted, **2 quarantined** (both at bs1, kept in the
record and replaced), fingerprint divergence rate **11.8% of scored reps** — in
line with M3-I11 campaign 2's measured 10–16% for the cold-compile class and
with M4-I0's known engine nondeterminism, i.e. pre-existing and not introduced
here. All reps on GPU 4.

| bs | verdict | accepted | quarantined | bit-exact vs `dumps_final` | agreement ≥ 90% |
|---|---|---|---|---|---|
| 1 | STABLE | 3/3 | 2 | **10/10** | 10/10, worst 0.9375 |
| 2 | STABLE | 3/3 | 0 | **10/10** | 10/10, worst 0.9375 |
| 4 | STABLE | 3/3 | 0 | **10/10** | 10/10, worst 0.9375 |
| 8 | STABLE | 3/3 | 0 | **10/10** | 10/10, worst 0.9375 |
| 16 | STABLE | 3/3 | 0 | **10/10** | 10/10, worst 0.9375 |

Under the three-part re-pinned criterion, at every batch size:

- **(a) coherence** — repetition ok; byte-soup not evaluated (no tokenizer in
  this venv). Perplexity **transfers by identity**: every case is byte-identical
  to the adjudicated baseline, so it is literally the same continuation that was
  already scored.
- **(b) agreement floor** — 10/10 cases ≥ 90% at every bs; worst 0.9375
  (`p06-poem`).
- **(c) no silent degradation** — **bit-exact 10/10 at every bs**. Exactness is
  the expected result here (Gate 1 proves the paths bit-exact) and it held, so
  there is nothing to explain.

The single divergence from the HF reference is `p06-poem` position 60 at every
bs, and it is **the same token the committed baseline emits** — the M2-adjudicated
numeric-precision tie. `repin_m4i7.json` marks it `same-as-baseline
[known-adjudicated]`. Not introduced by this change.

## Terminal disposition

**INTEGRATED.** The candidate did not clear its own standalone bar (`min_ratio`
0.801 against 1.3333) and it still produced the largest single in-MPK e2e gain of
the campaign so far — **1.3291× at bs1**, against M4-I2's 1.100× — because the
stage it feeds was width-bound and the port's work-item flattening is a width
lever the standalone metric structurally could not express: in the ferret harness
one CTA already ran one work item, so there was no dead-task slack to reclaim
there and the change was worth nothing to the score it was optimised against.

### What this leaves for M4

- **These two stages are close to done at low bs.** w13 is 11.1% of the bs1 step
  and w2 6.5%; both at zero would still leave 5.0 of 6.1 ms. Over half of w13's
  remaining bs1 wallspan is width residual, i.e. M4-I5's ~33 us/level arrival
  spread, which no inner-loop work touches.
- **bs16 still has both.** 39–40% width residual and a live kernel margin
  (PATH 1 beat PATH 0 by 2.34% there, so the fetch pipeline is still binding).
- **The flattening pattern generalises.** Any MPK task whose grid is sized by a
  worst-case activation count and whose body walks a serial inner dimension is
  leaving emitted tasks dead. That is a cheap, bit-exact lever and it is worth
  auditing the other grouped/routed tasks for it.
- **`MPK_MOE_N_SPLITS` is now a *worse* lever than before**, not a better one:
  flattening already recruits the tasks the knob would have created, at no dead-
  dispatch cost. It should stay default-off, and M4-I5's ×1.11 should not be
  re-litigated on top of this.
