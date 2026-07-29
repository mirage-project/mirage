# M4-I5 — graph width: what it is worth, measured

**Disposition: the knob lands DEFAULT-OFF; graph width is NOT the AC-4 residual.**

Width is real and large — at bs1 the megakernel runs 80 % of a decode step with
exactly one stage active on 18 % of the machine. But it is worth far less than
the packing arithmetic says. The best merge-free split available, applied to the
largest width residual in the graph, measures **×1.11 / ×1.08 / ×1.07 / ×1.04 at
bs 1/2/4/8 and ×0.99 at bs16** — against a model that predicted ×1.34 at bs1 —
and the gap between model and measurement is itself the finding: a grid split
recovers workers but is **not work-conserving**, and a dependency level has an
arrival-spread floor the split cannot cross.

The number that reframes AC-4 is not a width number at all. The compiled task
graph's **critical path** — the longest dependency chain, weighted by measured
per-task times — is 4.55 / 5.39 / 6.09 ms at bs 1/8/16, i.e. **46–50 % of the
current step and 1.14–1.30× vLLM's entire step**. At infinite width and perfect
packing MPK would still be slower than vLLM at every batch size. The residual is
the chain, not the fan-out.

Basis: integrated HEAD (no `src/`, `include/` or `python/` change between
`c80ebd68` and `01a54ad9` — `git diff --name-only` over those trees is empty), so
the retained M3-I7 profiler buffers are current for graph structure. A/B and
gates ran in an isolated clone with its own freshly built extension.

---

## 1. The per-stage width table

`scripts/width.py`, over the retained M3-I7 buffers (msl=897, bs 1/8/16, the
context band the vLLM reference was captured at). Windows are M3-I7's:
`[288,384)`, `[365,461)`, `[720,733)`. Full tables in `tables/width_bs{1,8,16}.json`.

**`live/lvl` is the width number**: how many tasks of that stage are available at
one dependency level. A stage with 16 live tasks per level cannot use more than
16 of 128 workers, whatever the scheduler does. `sole` is the wall time the stage
is the ONLY thing running; `idle` is that time weighted by the machine it leaves
idle — width residual no other stage can hide.

### bs1 — step 9822.8 µs, occupancy 0.18

| stage | live/lvl | lvls | T_live µs | span µs | % step | span/pack | sole µs | idle µs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dense fp8 (279) | 32.5 | 160 | 16.3 | 2805.9 | 28.6 | 4.24 | 1768.8 | 1345.8 |
| MoE w13 (241) | **16.0** | 40 | 54.9 | 2346.0 | 23.9 | 7.36 | 2304.9 | **1992.3** |
| MoE w2 (242) | **16.0** | 40 | 29.8 | 1316.9 | 13.4 | 7.07 | 1201.4 | 1047.8 |
| MoE router (260) | **1.0** | 40 | 21.1 | 842.1 | 8.6 | 128.0 | 450.6 | 447.1 |
| dense bf16 + lm_head (253) | 32.1 | 71 | 12.9 | 830.4 | 8.5 | 3.63 | 206.5 | 66.6 |
| full attention (257) | **1.0** | 20 | 60.5 | 607.4 | 6.2 | 64.3 | 603.2 | 593.8 |
| quantize fp8 (275) | 0.12 | 950 | 4.2 | 538.5 | 5.5 | 9.20 | 215.9 | 191.3 |
| MoE combine (261) | 16.0 | 40 | 6.0 | 261.7 | 2.7 | 8.73 | 229.7 | 201.9 |
| GDN gate (238) | 9.5 | 40 | 4.6 | 255.2 | 2.6 | 12.7 | 233.2 | 213.8 |
| GDN recurrent (237) | **128.0** | 30 | 5.5 | 219.6 | 2.2 | **1.33** | 210.5 | 50.6 |
| GDN conv1d (234) | 8.0 | 30 | 5.3 | 172.2 | 1.8 | 17.4 | 166.5 | 156.9 |
| RMS norm (154) | — | 681 | 1.4 | 182.3 | 1.9 | 12.9 | 114.1 | 103.5 |
| **all stages** | | **2347** | | | | | **7915.3** | **6579.9** |

Three readings, all measured:

* **80.6 % of the step (7915 of 9823 µs) has exactly ONE stage running.** The
  megakernel is, at the stage level, essentially serial — one stage at a time,
  each on 1–33 of 128 workers. 6580 µs of the step is idle machine attributable
  to a single narrow stage.
* **7761 µs (79 %) of the step runs on ≤ 16 of 128 workers.** Total task time is
  226.9 ms, so the work bound is 1773 µs: the step is **5.5× its arithmetic**.
* **M3-I3's GDN split is what a fixed stage looks like.** 128 live tasks per
  level, span/perfect-pack 1.33 — against 7.36 for MoE w13. It is the only stage
  in the graph that is packed.

### bs8 (step 10728.1, occupancy 0.35) and bs16 (step 12662.3, occupancy 0.46)

Ranked by `idle`, i.e. by width residual:

| stage | bs1 idle | bs8 idle | bs16 idle | bs8 live/lvl | bs16 live/lvl |
|---|---:|---:|---:|---:|---:|
| MoE w13 | 1992.3 | 1191.6 | 694.1 | 65.7 | 121.2 |
| dense fp8 | 1345.8 | **1340.2** | **1347.3** | 32.5 | 32.5 |
| MoE w2 | 1047.8 | 636.2 | 349.7 | 65.7 | 121.2 |
| attention | 593.8 | 539.8 | 620.7 | 8.0 | 12.0 |
| MoE router | 447.1 | 532.2 | 602.4 | 1.0 | 1.0 |
| 16-wide glue (275/261/238/154) | 710.5 | 706.8 | 682.3 | ~16 | ~16 |
| total sole-idle | 6579.9 | 5398.8 | 5126.1 | | |

* The **MoE width residual is a small-batch phenomenon**: 16 live tasks per level
  at bs1, 65.7 at bs8, 121.2 of 128 at bs16. By bs16 it is packed.
* **Dense fp8 (279) and the router (260) are batch-INDEPENDENT**: 1346/1340/1347
  and 447/532/602 µs of idle machine at bs 1/8/16. They are the width residual
  that does not go away with batch, and both need a cross-task reduction to fix.
* bs1 is the binding batch size for AC-4 (2.79× vLLM against 2.25× at bs16), and
  the largest bs1 residual with a merge-free fix is the MoE pair, 3040 µs.

### Admissible split factor, and what limits it

Source-derived, in `scripts/ceiling.py`'s `SPLITS` table (each entry a citation):

| stage | knob | now | max | merge? | limit |
|---|---|---:|---:|---|---|
| MoE w13 (241) | `moe_n_splits` = grid.y | 2 | 8 | **no** | `OUTPUT_SIZE % 128 == 0` (`moe_fp8_blockscale_sm100.cuh:134`); w13 N = 1024 |
| MoE w2 (242) | same knob | 2 | 16 | **no** | same; w2 N = 2048. Shared knob ⇒ 8 |
| dense fp8 (279) | `fp8_grid(N) = N/128` | — | **at max** | yes | per-task N must be a whole 128-row scale block; finer means splitting K = a cross-task reduction |
| dense bf16 (253) | `grid_for_rmsnorm_linear_layer` | — | **at max** | yes | hardcoded 96/64 or N/256 in a util shared with every MPK model; the 256 cap is a deliberate nondeterminism workaround |
| router (260) | `grid_dim=(1,1,1)` | 1 | 16 (rows) | yes | `routing`/`moe_mask` are a compaction across experts over all rows |
| attention (257) | `grid_dim=(mbr, kv_heads)` | 1 | ~8 | yes | split-KV needs a partial-output merge |
| quantize/combine/gate | `grid_dim=(mbt,1,1)` | 1 | ~8 | no | already one task per token row; tasks are 4–6 µs so per-task cost dominates |
| RMS norm (154) | — | — | 1 | yes | needs the whole row's sum of squares |
| GDN recurrent (237) | `gdn_split` grid.z | 4 | **4** | yes | `MAX_INPUTS_PER_TASK` is 7 with 6 in + 1 out used; partials and the counter already share a buffer |
| GDN conv1d (234) | `gdn_conv_channel_blocks` | 8 | 16 | no | channels independent; `conv_dim/blocks` must stay a whole tile |

Only **four** stages admit a merge-free widening, and one knob covers the two
that matter.

### Anchor QC — and the bs16 row M3-I7 excluded is admissible

The compiled graph is static: the same task list runs every iteration and only
durations vary. So the test with no free parameters is *does every iteration of
the window contain exactly the static call-site count of every task type*.
**PASS at bs1, bs8 and bs16.**

That resolves M3-I7's bs16 failure. Its `max_frac_err = 0.4437` is the
**fractional part** of a run-averaged tasks-per-step, so it asked "is this an
integer", not "is it the right integer". `scripts/anchor_bs16.py` locates the
cause: `nblocks = 208`, `ngroups = 1`, 200 M profiler slots ⇒ 961 538 events per
track, and **all 128 worker tracks are at that cap**. `PROFILER_CAN_WRITE`
(`profiler.h`) then silently drops events, so the last 36 of 1003 iterations are
truncated. Iterations 0–966 are exact in every watched task type, and the window
`[720,733)` is exact in all 13 iterations with a constant trace-derived
live-slot count of 12. Evidence: `tables/anchor_bs16.json`.

One consequence for the other M3-I7 item: because the tail is truncated, the
trace's 1004-iteration count at bs16 is a **floor**, not a measurement. The
recorded "`schedule_sim` predicts 1360, the trace has 1004" divergence is
therefore **not established by that capture** and needs an unprofiled iteration
count to settle.

---

## 2. The ceiling, as a bound with its assumptions

Three bounds, `scripts/ceiling.py` + `scripts/critpath.py`, all from the compiled
graph plus measured per-task times.

| bs | step µs | work bound (task-µs/128) | **critical path** | CP as % of step | vLLM's whole step |
|---:|---:|---:|---:|---:|---:|
| 1 | 9822.8 | 1773.0 | **4554.3** | 46.4 % | 3503 |
| 8 | 10728.1 | 3741.0 | **5394.2** | 50.3 % | 4727 |
| 16 | 12662.3 | 5800.9 | **6085.5** | 48.1 % | 5301 |

`critpath.py` walks the DAG (task T gated by `dependent_event`, triggering
`trigger_event`; 0 topological violations, ids are emitted in dependency order)
and takes the longest weighted chain. **At every batch size the chain alone is
longer than vLLM's entire decode step** (1.30× / 1.14× / 1.15×). Applying every
admissible split to the chain's per-task times brings it to 3804 / 4166 / 4328 µs
— still 8.6 % above vLLM at bs1, below it at bs8/bs16.

The packing ceiling, from the wave-depth dispatch model (M3-I8's: a level costs
`max_worker(live·T_live + dead·T_dead)`, calibrated per stage on its measured
span, calibration factors 1.06–1.36 printed in `tables/ceiling.json`), with
**measured** fixed-cost terms — M3-I6a's attention intercept, this run's own
dead-task cost — and source-derived admissibility:

| bs | merge-free splits only | every admissible split |
|---:|---|---|
| 1 | ×1.34 … ×1.36 | ×1.48 … ×1.60 |
| 8 | ×1.08 … ×1.09 | ×1.17 … ×1.23 |
| 16 | ×1.01 … ×1.01 | ×1.09 … ×1.13 |

Ranges are (Δstep = full span delta) … (Δstep = span delta × the stage's measured
sole-occupancy fraction). **Assumptions, stated because §4 falsifies two of
them:** per-task time divides by the split factor; the split is
work-conserving apart from the dead-task cost; a stage's span delta passes
through to the step in proportion to its sole occupancy. Gaps needed for AC-4
are ×2.79 / ×2.38 / ×2.25, so **width does not close AC-4 at any batch size**
even at this optimistic ceiling.

---

## 3. What was prototyped, and why

**`MPK_MOE_N_SPLITS`** — grid.y on both grouped MoE GEMMs, default unchanged at
2, env-overridable before build, legality asserted in `_build_moe` against the
kernel's own `OUTPUT_SIZE % 128 == 0`. One line of behaviour change.

Chosen because it is the largest merge-free width residual at the batch size AC-4
binds hardest (bs1: 3040 µs of a 9823 µs step, 46 % of the whole width residual),
and the cheapest thing to falsify.

**The M3-I3 soundness test, applied — does it need a barrier? NO.** grid.y
partitions the OUTPUT COLUMNS. Each task owns a disjoint output range, the whole
K reduction stays inside one task, and each n-block's accumulator was already
independent. There is no cross-task reduction, hence no arrival counter, no
epilogue election, and no co-residency requirement — the property a persistent
work-queue scheduler cannot guarantee is not needed here. Bit-exact by
construction for the same reason, which §5 confirms.

**This re-tests a rejected lever, and says in advance what is different.** M3-I8
staged this knob and its disposition is *rejected-with-evidence* — but the
evidence is `moe_n_splits = 4` at bs8 (+19.6 % against v1's +25.1 %), and
`moe_n_splits = 8` was measured at bs1 only. M3-I8's own mechanism for that
regression ("splitting returns bs8 to two waves") is a worker-depth argument, and
worked through it predicts k=4 is the *worst* choice and k=8 is a different bet.
The predictions were committed before the run (`predictions.md`) with their
falsifiers.

---

## 4. The A/B — and the predictions it falsified

Geometry B (the matched 256/1024 shape: synthetic 256-token prompts, msl=353, 96
decode steps), five batch sizes, three arms **interleaved per (bs, rep) inside
one GPU claim** so drift or a co-tenant hits all three equally. **45 reps, 0
discarded.** A kernel dir per (arm, bs) — the split is a compile-time template
argument, and two arms sharing one would report themselves identical while the
binary never changed (M3-I7 defect 3). Every rep drain-gated below 500 MiB and
audited from its **own** `meta.cuda_visible_devices` + `meta.gpu_before`.
`scripts/sweep_tables.py` re-derives the table and every discard decision from
`raw/`.

Wall ms, median of 3; `sp%` is rep spread as a percentage of the median;
**ratios below 1.000 are faster than the k=2 base**.

| bs | k=2 (base) | sp% | k=4 | sp% | k=8 | sp% | k4/k2 | k8/k2 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1206.1 | 0.12 | 1082.7 | 0.48 | **1074.2** | 0.15 | 0.8977 | **0.8906** |
| 2 | 1536.5 | 0.24 | 1416.9 | 0.55 | **1412.4** | 0.63 | 0.9222 | **0.9192** |
| 4 | 1747.3 | 0.23 | **1626.5** | 0.26 | 1719.0 | 3.45 | **0.9309** | 0.9838 |
| 8 | 2674.0 | 0.21 | **2584.8** | 3.66 | 2688.1 | 2.32 | **0.9666** | 1.0053 |
| 16 | **4120.8** | 0.42 | 4161.6 | 2.48 | 4252.1 | 1.89 | 1.0099 | 1.0319 |

Per-rep values: `tables/sweep_geomB.json` (`per_rep_wall_ms`,
`per_rep_mib_before`, `per_rep_device`, `per_rep_tokens_sha256` for all 45).

Against `predictions.md`:

* **P1 FALSIFIED.** k=8 was predicted to beat k=4 at bs8. It loses (1.0053 vs
  0.9666) and is worse than the base.
* **P2 holds.** k=8 is the best arm at bs1 — but by 0.7 %, not the wide margin
  predicted.
* **P3 FALSIFIED in magnitude.** bs8 k=8 predicted ×1.057; measured ×0.995.
* **P4 holds, and its optimistic branch is refuted.** bs16 regresses (+1.0 % at
  k=4, +3.2 % at k=8). The model's alternative — that a genuinely 16-live bs16
  step would gain ≈24 % — is wrong: this geometry runs the shipped cap with 16
  live slots and still regresses.
* **P6 holds.** `-Xptxas -v` on the generated megakernel TU: **238 registers,
  144 B stack frame, 0 spill stores, 0 spill loads — identical at k = 2/4/8**
  (`gates/ptxas/`). The regressions are not register pressure, and the template
  args confirm the split is in the binary: w13 `OUTPUT_SIZE` 512 → 256 → 128, w2
  1024 → 512 → 256.

### Why the model was wrong — measured, not guessed

Three profiled arms at bs1, same geometry, `tables/prof/width_bs1_k{2,4,8}.json`.
Anchor QC PASS on all three. The profiled steps reproduce the unprofiled walls
(10882 → 9786 → 9748 µs, ×1.112 / ×1.116 against the wall's ×1.114 / ×1.123).

| quantity | k=2 | k=4 | k=8 | ideal at k=8 |
|---|---:|---:|---:|---:|
| w13 live/level | 26.0 | 52.1 | **104.1** | 104.1 |
| w13 `T_live` µs | 56.06 | 29.24 | 15.77 | 14.02 |
| w13 span µs | 2384.9 | 1630.8 | 1516.9 | — |
| w13 total task-µs | 63 772 | 70 628 | **84 307** | 63 772 |
| w2 span µs | 1344.0 | 1016.4 | **1142.1** | — |
| w2 total task-µs | 35 925 | 42 622 | **55 355** | 35 925 |
| work bound (all stages) µs | 2202.5 | 2310.3 | 2517.0 | 2202.5 |
| **step µs** | 10 882.0 | 9786.0 | 9748.4 | — |

1. **The width IS recovered.** Live tasks per level double exactly, 26 → 52 →
   104 of 128. The split does what it claims.
2. **Per-task time does not quite halve.** ×0.522 and ×0.539 per doubling for
   w13, ×0.535 / ×0.558 for w2 — 4–6 % of each doubling lost to the smaller N
   tile.
3. **The split is NOT work-conserving.** Total worker time grows 32 % (w13) and
   54 % (w2) from k=2 to k=8, and the whole step's work bound rises 14 %. Part is
   (2); the rest is dead-task dispatch — emitted tasks per level go 256 → 1024
   while live goes 26 → 104, so each worker pays 7 dead tasks per level at ~0.5 µs
   instead of 1.
4. **A level has an arrival-spread floor the split cannot cross.**
   `scripts/arrival_skew.py` measures `max(begin) − min(begin)` over one level's
   tasks: w13 **58.2 → 36.5 → 32.6 µs**, w2 32.3 → 23.4 → 25.5 µs. Compare the
   span per level: 59.6 → 40.8 → 37.9 µs. **The level's cost is its arrival
   spread**, and from k=4 to k=8 the serial worker time per level halves
   (30.8 → 19.3 µs) while the span falls 7 %. That floor is why w2's span goes
   back UP at k=8: the spread stops shrinking while the work keeps growing.

So the corrected picture: a merge-free grid split buys width, pays for it in
total work, and bottoms out against a per-level arrival spread of ~33 µs at bs1.
The measured ×1.11 against the model's ×1.34 is that correction. Scaled across
the merge-free ceiling of §2, the realistic merge-free width ceiling at bs1 is
**×1.11–×1.15, not ×1.34**.

---

## 5. AC-3

`scripts/gate_ac3_m4i5.sh`, arm `MPK_MOE_N_SPLITS=4` (the best-performing value),
full sweep at bs 1/2/4/8/16 on the 10 pinned reference prompts, msl=132. The
compiled split is audited from the generated TU per batch size.

* **Per-case byte diff vs the committed `results/dumps_final`: `identical: true`,
  10/10 at every batch size, `CHANGED: none`** (`gates/ac3/bytediff_k4.json`).
* **The harness verdict is identical to the committed baseline field for field**:
  same `overall_pass`, same 45/50 per-case verdicts with **zero differences**,
  the same 5 waiver records (`p06-poem` position 60 at every bs, reference top-1
  `31000` at logit 21.0 tying top-2 `81316`, engine argmax `40581` = the
  reference's own top-3). That is the numeric-precision tie M2 adjudicated and it
  is present in the committed report too.
* Under the re-pinned AC-3: coherence is satisfied trivially (tokens identical to
  the adjudicated baseline), agreement is 63/64 positions on the one non-exact
  case — 98.4 %, above the 90 % floor — and the divergence is a documented
  reference-side near-tie with margin 0.0. **Bit-exactness holds, not merely
  reported.** The strict-identity harness's `overall_pass: false` is the
  pre-existing tie, superseded as a pass condition by the 2026-07-29 re-pin.

The split contributes **zero** correctness change, which is what §3's
disjoint-columns argument requires.

---

## 6. Terminal disposition

**The knob lands, default-off. The width programme does not close AC-4.**

* `MPK_MOE_N_SPLITS` ships with default 2 — byte-identical behaviour to HEAD — so
  nothing changes unless it is set. It is the cheapest available AC-4 increment at
  small batch: **+11.1 % / +8.8 % / +7.4 % / +3.5 % at bs 1/2/4/8 with k=4**, and
  **−1.0 % at bs16**, so an unconditional default is not adoptable. A
  bs-conditional rule (k=4 for bs ≤ 8, k=2 at bs16) is adoptable and is left to
  the coordinator as an M4 decision, not taken here: it needs the AC-5 prefill
  arm re-measured, since geometry B's wall mixes 16 prefill iterations into 96
  decode steps.
* `moe_n_splits = 8` is **rejected with evidence**: worse than k=4 at bs ≥ 4 and
  worse than the base at bs ≥ 8, mechanism in §4.
* **Graph width is not the AC-4 residual.** Ranked by what is left:
  1. **The critical path.** 4.55/5.39/6.09 ms against vLLM's 3.50/4.73/5.30 ms
     whole step. This is the binding structural fact and no split touches it. The
     levers are per-task throughput on the chain (dense fp8, MoE GEMM kernels) and
     FEWER, BIGGER tasks — the opposite of splitting. 2347 dependency levels per
     step is itself a target.
  2. **Dense fp8 (279)** — 1346 µs of idle machine at *every* batch size, already
     at the finest legal N split. Needs split-K with an atomic merge (the M3-I3
     idiom is admissible) or a coarser scale layout. Largest remaining
     batch-independent width residual.
  3. **The MoE router (260)** — 1 task per layer, 447–602 µs of idle machine,
     batch-independent, and the worst ratio of any non-trivial stage. Splittable
     over the 16 router rows with a last-arriving-task compaction epilogue.
  4. **Attention split-KV (257)** — 594–621 µs of idle machine; M3-I6a's model
     (`29.6 + 0.0536·ctx/k + merge`) is confirmed by this basis to 1 % (measured
     59.8 µs/task at ctx ≈ 575 against 60.4 predicted), but §4's arrival-spread
     floor applies and the 29.6 µs fixed term dominates past k≈4.
  5. **The 16-wide `grid_dim=(mbt,1,1)` glue** (275/261/238/154) — ~700 µs of idle
     machine at every batch size, merge-free, but tasks are already 4–6 µs so §4's
     costs bite immediately. Low expected value.
* **A harness defect found and fixed on the way**: `profile_wave.py` was dead at
  every batch size where the shipped admission policy resolves to a cap (bs ≥ 4)
  because `mpk_engine_run._cap_kwargs` probed `mirage.PersistentKernel` — the
  attribute `profile_wave.py` itself monkeypatches for the duration of that call.
  Nothing had profiled bs ≥ 4 since the probe landed at `348a601a`.
* **Not run:** geometry C (deep context). The routed-MoE cost is
  context-independent and geometry B is decisive and internally consistent
  (profiled steps reproduce unprofiled walls to 1 %); a second geometry would not
  change a disposition that already rests on a measured mechanism.

---

## 7. Layout and reproduction

| path | what |
|---|---|
| `predictions.md` | pre-registered predictions + falsifiers + voiding rules, committed before the run |
| `tables/width_bs{1,8,16}.json` | the per-stage width table, anchor QC, concurrency bands |
| `tables/anchor_bs16.json` | the bs16 anchor-QC resolution (profiler tail truncation) |
| `tables/ceiling.json` | the wave-depth ceiling model, per-stage calibration, admissibility |
| `tables/critpath_bs*.json` | DAG critical path, baseline and with splits |
| `tables/sweep_geomB.json`, `.csv` | the A/B: all 45 per-rep walls, device audits, token hashes |
| `tables/prof/width_bs1_k{2,4,8}.json` | the profiled mechanism check |
| `tables/prof/skew_bs1.json`, `sched_bs1.json` | arrival spread per level; scheduler-CTA time |
| `raw/noprofB_k{2,4,8}/` | **all 45 per-rep metas + token dumps** — `sweep_tables.py` reproduces §4 from these alone |
| `raw/prof/` | the profiled arms' metas, token dumps and logs |
| `gates/ac3/` | AC-3 run report, per-case byte diff, per-bs logs, per-bs device audits |
| `gates/ptxas/` | `-Xptxas -v` on the generated megakernel TU at k = 2/4/8 |
| `scripts/` | every script used |

The three bs1 profiler buffers behind §4's mechanism check are retained off-repo
at **`/home/catalyst/mpk-artifacts/m4i5/prof_raw/raw_bs1_rep0_k{2,4,8}.npz`**
(566 MB); the bs8 pair stays on catalyst-B200 under `/var/tmp/m4i5_prof/`, and the
M3-I7 buffers the §1 width table rests on are at
`/home/catalyst/mpk-artifacts/m3i7/late_raw/`. Everything in `tables/` regenerates
from those with the committed scripts; §4's A/B table regenerates from `raw/`
alone, with no GPU — verified, and every published field is identical (only each
rep record's `path` differs, scratch vs repo).

```bash
# CPU-only, no GPU: the width table, the ceiling, the critical path
cd demo/qwen3_5/accept/opt
python3 m4i5/scripts/width.py /home/catalyst/mpk-artifacts/m3i7/late_raw/raw_bs1_rep0.npz \
    m3i7/raw_meta/prof_prof_Alate/meta_bs1_rep0.json meta/task_names.json \
    --graph /home/catalyst/mpk-artifacts/m3i7/box/graphs/task_graph_bs1.json \
    --window 288,384 --out /tmp/w1.json
python3 m4i5/scripts/ceiling.py m4i5/tables/width_bs{1,8,16}.json --out /tmp/c.json
python3 m4i5/scripts/critpath.py /home/catalyst/mpk-artifacts/m3i7/box/graphs/task_graph_bs1.json \
    m4i5/tables/width_bs1.json
python3 m4i5/scripts/sweep_tables.py m4i5/raw --out /tmp/ab.json   # reproduces §4

# on catalyst-B200, isolated clone + fresh extension
BASE=6bd3ffd7 PINNED=<sha> bash m4i5/scripts/setup_m4i5.sh
GEOMS=B bash m4i5/scripts/retry_m4i5.sh sweep_moe.sh      # the A/B, ~45 min on one GPU
K=4 bash m4i5/scripts/retry_m4i5.sh gate_ac3_m4i5.sh      # AC-3 on the split arm
bash m4i5/scripts/mk_ptxas_m4i5.sh                        # compile-only, no GPU
BSLIST=1 bash m4i5/scripts/retry_m4i5.sh prof_arms.sh     # the profiled mechanism check
```
