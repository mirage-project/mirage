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
per-task times — is **7.96 / 8.24 / 8.64 ms at bs 1/8/16, i.e. 68–81 % of the
current step and 1.63–2.27× vLLM's entire step**. At infinite width and perfect
packing MPK would still be 1.6–2.3× slower than vLLM. The residual is the chain,
not the fan-out. §7 decomposes it and answers what has to get faster, and by how
much, for AC-4 to be arithmetically reachable.

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

| bs | step µs | work bound (task-µs/128) | **critical path** | CP as % of step | vLLM's whole step | CP / vLLM step |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 9822.8 | 1773.0 | **7957.5** | 81.0 % | 3503 | **2.27×** |
| 8 | 10728.1 | 3741.0 | **8240.9** | 76.8 % | 4727 | **1.74×** |
| 16 | 12662.3 | 5800.9 | **8642.0** | 68.2 % | 5301 | **1.63×** |

`critpath.py` walks the DAG (task T gated by `dependent_event`, triggering
`trigger_event`; 0 topological violations, ids are emitted in dependency order)
and takes the longest weighted chain. **At every batch size the chain alone is
1.6–2.3× vLLM's entire decode step.** Applying every admissible split to the
chain's per-task times brings it to 4827 / 5063 / 5349 µs — still above vLLM at
bs1 and bs8, marginally below at bs16.

> **Correction (this issue's own first pass reported 4554 / 5394 / 6086 µs.)** That
> weighting charged each task the live/dead EXPECTED value, which understates
> every level mixing live and dead tasks — the routed MoE GEMMs are 6 % live at
> bs1, so their chain contribution came out ~14× too small. **Every event in this
> graph has `num_triggers == n_producers` — verified, 2277 of 2277 at all three
> batch sizes** — so every event is a full fan-in barrier and a level costs its
> SLOWEST producer, i.e. `T_live`. Both weightings are emitted
> (`cp_max_us` and `cp_expected_weighting_us`) so the correction is auditable.
> The direction of the conclusion does not change; its size roughly doubles. §7
> is the decomposition.

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
  1. **The critical path.** 7.96/8.24/8.64 ms against vLLM's 3.50/4.73/5.30 ms
     whole step (§2, §7). This is the binding structural fact and no split
     touches it. The chain is **59 % MoE block at bs1** — router + w13 + w2 alone
     are 4228 µs of 7958 (53 %) — so the binding AC-4 lever is per-task LATENCY
     on the MoE chain, which is what the running MoE ferret loop attacks. §7 has
     the per-stage targets and which of them are known-achievable.
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

## 7. What the critical path is MADE OF, and the AC-4 feasibility arithmetic

`scripts/cp_decompose.py`. The chain is recovered by recording, for every event,
which producer set its ready time, then walking back from the finishing task; each
path task's layer is read off its own `layer_<i>_...` tensor names, so the
per-layer structure is exact rather than inferred. Self-check: the unscaled length
equals `critpath.py`'s `cp_max_us` to 0.1 µs at all three batch sizes.

**The coordinator's back-of-envelope was arithmetically exact — only the
denominator was wrong.** router 21.05 + w13 54.87 + w2 29.77 = 105.69 µs/layer ×
40 layers = 4227.6 µs, and the three stages measure 842.1 + 2194.6 + 1191.0 =
4227.7 µs on the path. **No double counting**: each appears exactly 40 times, once
per layer, confirmed by the per-layer chain below. It is **53 % of the path**, not
93 %, because the path is 7958 µs and not the understated 4554.

### (a) Composition by stage


**bs1 — cp = 7957.5 µs, 595 tasks over 40 layers, 81.0 % of the measured 9822.8 µs step**

| stage | path tasks | µs on path | % of cp | µs/path task | measured T_live |
|---|---:|---:|---:|---:|---:|
| MOE_W13_FP8_BLOCKSCALE_SM100 | 40 | 2194.6 | 27.58 | 54.866 | 54.866 |
| LINEAR_FP8_BLOCKSCALE_SM100 | 80 | 1301.8 | 16.36 | 16.272 | 16.272 |
| MOE_W2_FP8_BLOCKSCALE_SM100 | 40 | 1191.0 | 14.97 | 29.774 | 29.774 |
| MOE_TOPK_SOFTMAX_SM100 | 40 | 842.1 | 10.58 | 21.053 | 21.053 |
| ATTN_SM100 | 10 | 605.0 | 7.6 | 60.502 | 60.502 |
| LINEAR_SM100 | 41 | 527.1 | 6.62 | 12.857 | 12.857 |
| QUANTIZE_FP8_SM100 | 120 | 504.0 | 6.33 | 4.2 | 4.2 |
| MOE_MUL_SUM_ADD_SM100 | 40 | 239.9 | 3.01 | 5.997 | 5.997 |
| GDN_RECURRENT_SM100 | 30 | 165.4 | 2.08 | 5.515 | 5.515 |
| GDN_CONV1D_SM100 | 30 | 158.0 | 1.99 | 5.266 | 5.266 |
| RMS_NORM_HOPPER | 81 | 113.4 | 1.43 | 1.4 | 1.4 |
| EMBEDDING | 1 | 57.9 | 0.73 | 57.911 | 57.911 |
| SILU_MUL | 40 | 49.2 | 0.62 | 1.231 | 1.231 |
| ARGMAX_PARTIAL_SM100 | 1 | 5.6 | 0.07 | 5.604 | 5.604 |
| ARGMAX_REDUCE_SM100 | 1 | 2.4 | 0.03 | 2.377 | 2.377 |

**bs8 — cp = 8240.9 µs, 595 tasks over 40 layers, 76.8 % of the measured 10728.1 µs step**

| stage | path tasks | µs on path | % of cp | µs/path task | measured T_live |
|---|---:|---:|---:|---:|---:|
| MOE_W13_FP8_BLOCKSCALE_SM100 | 40 | 2278.6 | 27.65 | 56.966 | 56.966 |
| LINEAR_FP8_BLOCKSCALE_SM100 | 80 | 1301.4 | 15.79 | 16.268 | 16.268 |
| MOE_W2_FP8_BLOCKSCALE_SM100 | 40 | 1218.9 | 14.79 | 30.473 | 30.473 |
| MOE_TOPK_SOFTMAX_SM100 | 40 | 903.3 | 10.96 | 22.582 | 22.582 |
| ATTN_SM100 | 10 | 574.7 | 6.97 | 57.465 | 57.465 |
| LINEAR_SM100 | 41 | 534.2 | 6.48 | 13.03 | 13.03 |
| QUANTIZE_FP8_SM100 | 120 | 510.2 | 6.19 | 4.252 | 4.252 |
| GDN_RECURRENT_SM100 | 30 | 309.1 | 3.75 | 10.304 | 10.304 |
| MOE_MUL_SUM_ADD_SM100 | 40 | 217.2 | 2.64 | 5.429 | 5.429 |
| GDN_CONV1D_SM100 | 30 | 155.7 | 1.89 | 5.189 | 5.189 |
| RMS_NORM_HOPPER | 81 | 108.0 | 1.31 | 1.333 | 1.333 |
| SILU_MUL | 40 | 45.5 | 0.55 | 1.137 | 1.137 |
| EMBEDDING | 1 | 43.8 | 0.53 | 43.763 | 43.763 |
| ARGMAX_PARTIAL_SM100 | 1 | 31.8 | 0.39 | 31.764 | 31.764 |
| ARGMAX_REDUCE_SM100 | 1 | 8.6 | 0.1 | 8.593 | 8.593 |

**bs16 — cp = 8642.0 µs, 595 tasks over 40 layers, 68.2 % of the measured 12662.3 µs step**

| stage | path tasks | µs on path | % of cp | µs/path task | measured T_live |
|---|---:|---:|---:|---:|---:|
| MOE_W13_FP8_BLOCKSCALE_SM100 | 40 | 2430.4 | 28.12 | 60.759 | 60.759 |
| LINEAR_FP8_BLOCKSCALE_SM100 | 80 | 1295.4 | 14.99 | 16.193 | 16.193 |
| MOE_W2_FP8_BLOCKSCALE_SM100 | 40 | 1209.0 | 13.99 | 30.225 | 30.225 |
| MOE_TOPK_SOFTMAX_SM100 | 40 | 982.1 | 11.36 | 24.552 | 24.552 |
| ATTN_SM100 | 10 | 623.0 | 7.21 | 62.303 | 62.303 |
| LINEAR_SM100 | 41 | 549.4 | 6.36 | 13.4 | 13.4 |
| QUANTIZE_FP8_SM100 | 120 | 513.1 | 5.94 | 4.276 | 4.276 |
| GDN_RECURRENT_SM100 | 30 | 306.1 | 3.54 | 10.202 | 10.202 |
| MOE_MUL_SUM_ADD_SM100 | 40 | 200.1 | 2.32 | 5.003 | 5.003 |
| SILU_MUL | 40 | 166.0 | 1.92 | 4.15 | 4.15 |
| GDN_CONV1D_SM100 | 30 | 154.6 | 1.79 | 5.153 | 5.153 |
| RMS_NORM_HOPPER | 81 | 114.5 | 1.33 | 1.414 | 1.414 |
| ARGMAX_PARTIAL_SM100 | 1 | 46.8 | 0.54 | 46.751 | 46.751 |
| EMBEDDING | 1 | 38.6 | 0.45 | 38.612 | 38.612 |
| ARGMAX_REDUCE_SM100 | 1 | 12.9 | 0.15 | 12.857 | 12.857 |

The **MoE block** (router → w13 → SiLU-mul → w2 → combine) is
4516.8 µs of the bs1 path = **57 %**, and adding the
MoE activation quantize (40 of the 120 quantize path tasks, 168 µs) takes it to
59 %. So **the coordinator's hypothesis is right
in substance**: the MoE chain dominates, and the binding AC-4 lever is per-task
latency on it. The two corrections to the sizing are that dense fp8 (79 with 279
appearing **80** times — twice per layer, in-proj and out-proj) is the second
largest single contributor at 16 %, and that the MoE GEMMs are not 93 % of the
chain because the chain is twice as long as first reported.

### (b) Per-layer chain structure

Exact, from tensor names. 595 path tasks over 40 layers; the histogram of path
tasks per layer is `{14: 10, 15: 29, 16: 1}` — 10 full-attention layers of 14 and
30 GDN layers of 15 (one carries an extra RMS norm). Identical at all three batch
sizes.


**GDN layer (15–16 path tasks):**

`RMS_NORM → QUANTIZE_FP8 → LINEAR_FP8(in_proj) → GDN_CONV1D → GDN_RECURRENT →
QUANTIZE_FP8 → LINEAR_FP8(out_proj) → RMS_NORM → LINEAR_SM100(router gate) →
MOE_TOPK_SOFTMAX → MOE_W13 → SILU_MUL → QUANTIZE_FP8 → MOE_W2 → MOE_MUL_SUM_ADD
→ RMS_NORM`

**Full-attention layer (14 path tasks):**

`QUANTIZE_FP8 → LINEAR_FP8(qkv) → ATTN_SM100 → QUANTIZE_FP8 →
LINEAR_FP8(out_proj) → RMS_NORM → LINEAR_SM100 → MOE_TOPK_SOFTMAX → MOE_W13 →
SILU_MUL → QUANTIZE_FP8 → MOE_W2 → MOE_MUL_SUM_ADD → RMS_NORM`

Both layer types carry the **same 7-task MoE tail**, which is why the MoE block
is 40× on the chain regardless of layer type, and why it dominates. The GDN and
attention heads differ but are the cheap end: GDN's conv1d + recurrent is
5.27 + 5.51 µs at bs1 against the MoE tail's ~112 µs.

### (c) Sensitivity — cp when one stage reaches a multiple of vLLM's per-call time

Floors are the MEASURED `vllm_us_per_call` from `opt/m3i10/ferret_targets.json`,
per batch size. Tasks 279 and 253 have no per-call number because vLLM fuses
them, so their floor is the derived `vllm_us_per_step / sites_per_step` and is
labelled as such in the JSON. A stage where MPK is already at or below vLLM
(GDN recurrent at bs16: 10.20 µs against 15.43) is skipped rather than
"brought to parity", which would be a regression.


**bs1** (base cp 7957.5 µs)

| stage | measured T µs | vLLM µs/call | cp @ 2×vLLM | cp @ parity | cp @ 0.7×vLLM |
|---|---:|---:|---:|---:|---:|
| MOE_W13_FP8_BLOCKSCALE_SM100 | 54.866 | 8.414 | 6436.0 | **6099.4** | 5998.4 |
| LINEAR_FP8_BLOCKSCALE_SM100 | 16.272 | 8.459 | 8009.2 | **7332.5** | 7129.4 |
| MOE_W2_FP8_BLOCKSCALE_SM100 | 29.774 | 7.517 | 7367.9 | **7067.2** | 6977.0 |
| MOE_TOPK_SOFTMAX_SM100 | 21.053 | 3.697 | 7411.1 | **7263.2** | 7218.9 |
| ATTN_SM100 | 60.502 | 9.425 | 7541.0 | **7446.7** | 7418.4 |
| LINEAR_SM100 | 12.857 | 8.581 | 8134.0 | **7782.2** | 7676.6 |

**bs8** (base cp 8240.9 µs)

| stage | measured T µs | vLLM µs/call | cp @ 2×vLLM | cp @ parity | cp @ 0.7×vLLM |
|---|---:|---:|---:|---:|---:|
| MOE_W13_FP8_BLOCKSCALE_SM100 | 56.966 | 21.996 | 7722.0 | **6842.1** | 6578.2 |
| LINEAR_FP8_BLOCKSCALE_SM100 | 16.268 | 8.967 | 8374.2 | **7656.8** | 7441.6 |
| MOE_W2_FP8_BLOCKSCALE_SM100 | 30.473 | 17.477 | 8420.2 | **7721.1** | 7511.4 |
| MOE_TOPK_SOFTMAX_SM100 | 22.582 | 4.602 | 7705.8 | **7521.7** | 7466.5 |
| ATTN_SM100 | 57.465 | 9.237 | 7851.0 | **7758.6** | 7730.9 |
| LINEAR_SM100 | 13.03 | 8.829 | 8430.7 | **8068.7** | 7960.1 |

**bs16** (base cp 8642.0 µs)

| stage | measured T µs | vLLM µs/call | cp @ 2×vLLM | cp @ parity | cp @ 0.7×vLLM |
|---|---:|---:|---:|---:|---:|
| MOE_W13_FP8_BLOCKSCALE_SM100 | 60.759 | 28.081 | 8458.1 | **7334.8** | 6997.9 |
| LINEAR_FP8_BLOCKSCALE_SM100 | 16.193 | 8.569 | 8717.5 | **8032.0** | 7826.3 |
| MOE_W2_FP8_BLOCKSCALE_SM100 | 30.225 | 18.898 | 8944.8 | **8188.9** | 7962.1 |
| MOE_TOPK_SOFTMAX_SM100 | 24.552 | 5.955 | 8136.3 | **7898.1** | 7826.6 |
| ATTN_SM100 | 62.303 | 10.659 | 8232.1 | **8125.5** | 8093.5 |
| LINEAR_SM100 | 13.4 | 8.904 | 8822.7 | **8457.6** | 8348.1 |

**cp is a MAX over paths, so it is not additive under perturbation.** Cheapening
one stage can expose a different, longer chain: the naive sum of the three MoE
gains at bs1 is 3442 µs, but recomputing gives 4868 µs rather than 4515. Every
number in these tables is a recomputation, never a subtraction.

### The feasibility answer

vLLM's whole decode step is 3503 / 4727 / 5301 µs at bs 1/8/16 (bs ÷ its measured
decode tok/s, M3-I7 §2b). Stages added greedily by measured gain, cp recomputed at
each step:


**bs1** — cp must fall 4454.5 µs (7957.5 → below 3503.0)

| + stage at vLLM parity | path tasks | must reach µs/task | cp µs | under vLLM step? |
|---|---:|---|---:|---|
| MOE_W13_FP8_BLOCKSCALE_SM100 | 40 | **54.87 → 8.414** | 6099.4 | no |
| MOE_W2_FP8_BLOCKSCALE_SM100 | 40 | **29.77 → 7.517** | 5209.1 | no |
| MOE_TOPK_SOFTMAX_SM100 | 40 | **21.05 → 3.697** | 4867.8 | no |
| LINEAR_FP8_BLOCKSCALE_SM100 | 80 | **16.27 → 8.459** | 3889.9 | no |
| ATTN_SM100 | 10 | **60.5 → 9.425** | 3379.1 | **YES** |
| LINEAR_SM100 | 41 | **12.86 → 8.581** | 3203.8 | **YES** |
| GDN_CONV1D_SM100 | 30 | **5.27 → 2.987** | 3135.4 | **YES** |
| GDN_RECURRENT_SM100 | 30 | **5.51 → 5.455** | 3133.6 | **YES** |

Minimal sufficient set: **5 stages** — MOE_W13_FP8_BLOCKSCALE_SM100, MOE_W2_FP8_BLOCKSCALE_SM100, MOE_TOPK_SOFTMAX_SM100, LINEAR_FP8_BLOCKSCALE_SM100, ATTN_SM100.
Measured step / cp = **1.234×**, so meeting the chain floor with that packing factor unchanged gives a step of **3868.2 µs** (still above vLLM's 3503.0).

**bs8** — cp must fall 3513.9 µs (8240.9 → below 4727.0)

| + stage at vLLM parity | path tasks | must reach µs/task | cp µs | under vLLM step? |
|---|---:|---|---:|---|
| MOE_W13_FP8_BLOCKSCALE_SM100 | 40 | **56.97 → 21.996** | 6842.1 | no |
| MOE_TOPK_SOFTMAX_SM100 | 40 | **22.58 → 4.602** | 6122.9 | no |
| LINEAR_FP8_BLOCKSCALE_SM100 | 80 | **16.27 → 8.967** | 5538.8 | no |
| MOE_W2_FP8_BLOCKSCALE_SM100 | 40 | **30.47 → 17.477** | 5019.0 | no |
| ATTN_SM100 | 10 | **57.47 → 9.237** | 4536.7 | **YES** |
| LINEAR_SM100 | 41 | **13.03 → 8.829** | 4364.5 | **YES** |
| GDN_CONV1D_SM100 | 30 | **5.19 → 3.056** | 4300.5 | **YES** |
| GDN_RECURRENT_SM100 | 30 | **10.3 → 9.014** | 4261.8 | **YES** |

Minimal sufficient set: **5 stages** — MOE_W13_FP8_BLOCKSCALE_SM100, MOE_TOPK_SOFTMAX_SM100, LINEAR_FP8_BLOCKSCALE_SM100, MOE_W2_FP8_BLOCKSCALE_SM100, ATTN_SM100.
Measured step / cp = **1.302×**, so meeting the chain floor with that packing factor unchanged gives a step of **5548.0 µs** (still above vLLM's 4727.0).

**bs16** — cp must fall 3341.0 µs (8642.0 → below 5301.0)

| + stage at vLLM parity | path tasks | must reach µs/task | cp µs | under vLLM step? |
|---|---:|---|---:|---|
| MOE_W13_FP8_BLOCKSCALE_SM100 | 40 | **60.76 → 28.081** | 7334.8 | no |
| MOE_TOPK_SOFTMAX_SM100 | 40 | **24.55 → 5.955** | 6591.0 | no |
| LINEAR_FP8_BLOCKSCALE_SM100 | 80 | **16.19 → 8.569** | 5981.0 | no |
| ATTN_SM100 | 10 | **62.3 → 10.659** | 5464.6 | no |
| MOE_W2_FP8_BLOCKSCALE_SM100 | 40 | **30.23 → 18.898** | 5011.5 | **YES** |
| LINEAR_SM100 | 41 | **13.4 → 8.904** | 4827.1 | **YES** |
| GDN_CONV1D_SM100 | 30 | **5.15 → 3.187** | 4768.2 | **YES** |

Minimal sufficient set: **5 stages** — MOE_W13_FP8_BLOCKSCALE_SM100, MOE_TOPK_SOFTMAX_SM100, LINEAR_FP8_BLOCKSCALE_SM100, ATTN_SM100, MOE_W2_FP8_BLOCKSCALE_SM100.
Measured step / cp = **1.465×**, so meeting the chain floor with that packing factor unchanged gives a step of **6986.4 µs** (still above vLLM's 5301.0).

**So, stated as the coordinator asked — "stage X must reach Y µs/task":**

| stage | bs1 | bs8 | bs16 | known achievable? |
|---|---|---|---|---|
| MoE w13 (241) | 54.87 → **8.41** | 56.97 → **22.00** | 60.76 → **28.08** | **NOT yet** — the ferret MoE loop is at 0.456 of vLLM's throughput, i.e. this is the open target |
| MoE w2 (242) | 29.77 → **7.52** | 30.47 → **17.48** | 30.23 → **18.90** | same loop, same status |
| MoE router (260) | 21.05 → **3.70** | 22.58 → **4.60** | 24.55 → **5.96** | untried; M3-I5c knowingly cost this task +51–61 % for a compaction fix and it has never been re-costed |
| dense fp8 (279) | 16.27 → **8.46** | 16.27 → **8.97** | 16.19 → **8.57** | **YES** — the ferret dense-fp8 winner crossed parity and landed at `eee0fe66` (v011), after this basis was captured |
| attention (257) | 60.50 → **9.43** | 57.47 → **9.24** | 62.30 → **10.66** | partly — M3-I6a took it 8.09→4.50× and its split-KV model is confirmed here to 1 %, but the 29.6 µs fixed term floors it near 33 µs, so **9.4 µs needs a kernel change, not a split** |
| dense bf16 (253) | 12.86 → **8.58** | 13.03 → **8.83** | 13.40 → **8.90** | untried; 1.37–1.44× is the smallest ratio in the graph |
| GDN conv1d (234) | 5.27 → **2.99** | 5.19 → **3.06** | 5.15 → **3.19** | plausible — the GDN recurrent loop reached parity, so the family is tractable |
| GDN recurrent (237) | 5.51 → 5.46 | 10.30 → 9.01 | **already ahead** (10.20 vs 15.43) | **YES, done** — M3-I3 |

**The answer to "is AC-4 arithmetically reachable".** Yes on the chain, no on the
chain alone:

1. **Five stages at vLLM parity clear the chain floor at every batch size** —
   w13, w2, router, dense fp8 and attention. One of those five (dense fp8) is
   already done. Three of the five are one ferret programme (the MoE chain).
2. **No single stage suffices anywhere.** Taking w13 to *zero* removes only
   2195 µs of the bs1 chain, leaving 5763 µs against vLLM's 3503.
3. **The chain floor is not the step.** The measured step is 1.23× / 1.30× /
   1.47× cp. Meeting the five-stage floor with that packing factor unchanged
   gives 3868 / 5548 / 6986 µs — **still above vLLM at every batch size.**
   So latency work and width work are each necessary and neither is sufficient:
   the ferret loops have to close the per-task gap *and* the packing factor has
   to come down toward 1.0, which is what §1's 80 %-one-stage-at-a-time number
   describes.
4. **Corollary the coordinator asked about directly:** width work and dense work
   cannot move AC-4 while the MoE chain stands — the MoE block is 59 % of the
   bs1 chain, and even zeroing everything else leaves it at 4685 µs, above
   vLLM's whole step. But the converse is equally true: closing the MoE chain
   alone leaves cp at 4868 µs, also above 3503. **Both, or neither.**

---

## 8. Layout and reproduction

| path | what |
|---|---|
| `predictions.md` | pre-registered predictions + falsifiers + voiding rules, committed before the run |
| `tables/width_bs{1,8,16}.json` | the per-stage width table, anchor QC, concurrency bands |
| `tables/anchor_bs16.json` | the bs16 anchor-QC resolution (profiler tail truncation) |
| `tables/ceiling.json` | the wave-depth ceiling model, per-stage calibration, admissibility |
| `tables/critpath_bs*.json` | DAG critical path, both weightings, baseline and with splits |
| `tables/cp_decompose_bs{1,8,16}.json` | **§7: the path's composition, per-layer chain, sensitivity, and the AC-4 feasibility solve** |
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
python3 m4i5/scripts/cp_decompose.py /home/catalyst/mpk-artifacts/m3i7/box/graphs/task_graph_bs1.json \
    m4i5/tables/width_bs1.json --names meta/task_names.json \
    --ferret m3i10/ferret_targets.json --out /tmp/cp1.json    # section 7
python3 m4i5/scripts/sweep_tables.py m4i5/raw --out /tmp/ab.json   # reproduces §4

# on catalyst-B200, isolated clone + fresh extension
BASE=6bd3ffd7 PINNED=<sha> bash m4i5/scripts/setup_m4i5.sh
GEOMS=B bash m4i5/scripts/retry_m4i5.sh sweep_moe.sh      # the A/B, ~45 min on one GPU
K=4 bash m4i5/scripts/retry_m4i5.sh gate_ac3_m4i5.sh      # AC-3 on the split arm
bash m4i5/scripts/mk_ptxas_m4i5.sh                        # compile-only, no GPU
BSLIST=1 bash m4i5/scripts/retry_m4i5.sh prof_arms.sh     # the profiled mechanism check
```
