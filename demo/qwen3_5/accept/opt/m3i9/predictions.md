# M3-I9 — predictions, registered before any GPU time

Prep mode: every number below is a prediction from `protocol_sim.py` (schedule replay) and
`cost_model.py` (fit on M3-I1's 638 measured iterations). Nothing here has been measured.
`plan_m3i9.sh` is the capture that settles it. Written at `46872ad`, `mbt = 16`, AC-3 geometry
unless the row says 256/1024.

## 0. The claims, ranked by how badly a miss hurts

| # | claim | falsifier |
|---|---|---|
| C1 | live-slot compaction fires in the *shipped* wave protocol: 1 / 12 / 69 migrations at bs 4 / 8 / 16 | the runtime prints `request_ids` per iteration and the sequence never changes |
| C2 | the bs16 duplicate-slot `identical: false` signal IS that compaction, not a numerics/isolation defect | raise `max_seq_length` so no slot retires inside any reported window; if the six pairs still disagree, C2 is dead |
| C3 | prefill chunk decomposition does not change the answer | any two non-straddling placements of one prompt with different chunkings that disagree |
| C4 | `per_request_token_cap = 1` gives 131 iterations at bs16 (from 203) and removes every migration | replay the built kernel's own iteration count; `!= 131` kills it |
| C5 | that is +61.5% wave time at bs16, not backlog #4's +44% | measured bs16 wave outside 2.75–3.05 s (unprofiled) |
| C6 | it is bit-exact: all 25 (prompt, bs) sequences unchanged | one token differs |
| C7 | the cost law is `a + b·max_chunk + c·n_live`, i.e. cost tracks the LARGEST slot chunk, not the token total | `--slot-order sorted-padded` at bs16 lands outside 4.05–4.35 s |
| C8 | at 256/1024 the same fix is worth ~1.52× and takes straddling requests from 15/16 to 0 | replayed iteration count at that geometry `!= 1279` under the cap |

## 1. Policy ranking (predicted, profiled clock)

Model closure first: it reproduces all five measured profiled wave times to ≤0.04%
(`python3 cost_model.py`). Ratios below cancel the 2.85–3.59% profiling overhead.

### AC-3 geometry, bs16 (measured baseline 4695 ms profiled / 4566.5 ms unprofiled)

| rank | policy | lane | iters | pred ms | vs today | migrations | straddling |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | **per-request cap = 1** | runtime | **131** | **2906** | **1.615×** | **0** | **0** |
| 2 | hold-decode until all prefilled | runtime | 143 | 3561 | 1.318× | 36 | 0 |
| 3 | mbt=32 + cap=2 | runtime (needs I5b) | 119 | 3577 | 1.313× | 32 | 0 |
| 4 | mbt=32 | runtime (needs I5b) | 123 | 4063 | 1.155× | 50 | 6 |
| 5 | slot order = ascending-padded | **python/adapter** | 179 | 4214 | 1.114× | 91 | 4 |
| 6 | per-request cap = 2 | runtime | 201 | 4332 | 1.084× | 65 | 5 |
| 7 | slot order = descending | python/adapter | 192 | 4500 | 1.043× | 119 | 15 |
| 8 | per-request cap = 4 | runtime | 204 | 4500 | 1.043× | 71 | 5 |
| 9 | *today* | shipped | 203 | 4694 | 1.000× | 69 | 6 |
| 10 | mbt=64 + cap=4 | runtime (needs I5b) | 113 | 5148 | 0.912× | 32 | 0 |
| 11 | mbt=64 | runtime (needs I5b) | 114 | 5442 | 0.863× | 38 | 5 |

### 256/1024 geometry, bs16

| rank | policy | iters | pred s | vs today | migrations | straddling |
|---:|---|---:|---:|---:|---:|---:|
| 1 | **per-request cap = 1** | **1279** | **28.4** | **1.521×** | **0** | **0** |
| 2 | hold-decode until all prefilled | 1279 | 31.9 | 1.354× | 0 | 0 |
| 3 | mbt=32 + cap=2 | 1151 | 34.5 | 1.250× | 0 | 0 |
| 7 | *today* | 1887 | 43.2 | 1.000× | 120 | **15 of 16** |
| 8 | slot order (either) | 1887 | 43.2 | 1.000× | 120 | 15 |

Slot ordering is worth exactly nothing at 256/1024 — every prompt is the same length. The
absolute seconds are a **lower bound** (the step cost model is fit at ≤132 context; §6 of
`remeasure-protocol.md` predicts +14% at 1280). Ratios are the claim.

### mbt sensitivity — the extrapolation, stated

No measured iteration exists at `mbt != 16` (the MoE router 16-row cap, M2-I9). `delta` is the
share of the step that scales with `mbt`; the source audit in `cost_model.py` puts it at 0.35
with a 0.15–0.55 band.

| policy | delta=0.15 | delta=0.35 | delta=0.55 |
|---|---:|---:|---:|
| mbt=32 | 1.333× | 1.155× | 1.019× |
| mbt=32 + cap=2 | 1.538× | 1.313× | 1.145× |
| mbt=64 + cap=4 | 1.284× | 0.912× | 0.707× |
| **mbt=16 + cap=1** | **1.615×** | **1.615×** | **1.615×** |

The winning policy is the only candidate that needs no extrapolation at all: it is priced
entirely on one directly measured regime (16 slots × 1 token = 22003 µs, n=18, spread 0.49%).

### per batch size — the cap must be bound to `bs`

| bs | cap = max(1, mbt//bs) | iters | pred | migrations |
|---:|---:|---|---:|---|
| 1 | 16 | 109 → 109 | +0.0% | 0 → 0 |
| 2 | 8 | 109 → 110 | −0.6% | 0 → 0 |
| 4 | 4 | 109 → 113 | −2.2% | 1 → 0 |
| 8 | 2 | 111 → 119 | −2.7% | 12 → 0 |
| 16 | 1 | 203 → **131** | **+61.5%** | 69 → 0 |

A global cap of 1 would be a 19% regression at bs1. Bound to `bs`, the perf cost at bs 2/4/8 is
0.6–2.7% and it buys zero migrations at every batch size. **Registered call: take the small
loss.** A protocol whose correctness depends on "the corrupted tokens happen to fall outside the
reported window" is not a protocol, and at 256/1024 that luck runs out (15 of 16 exposed).

## 2. Backlog #4's +44%, re-derived

```
backlog delta_basis : 36 prefill iters @ 25.5 ms + 107 decode iters @ 22.0 ms
                      36*25.5 + 107*22.0                    = 3272 ms
                      4695.2 / 3272                         = 1.435x = +43.5%   ("+44%")
                      quoted tok/s 234 -> 327                = +39.7%
```

Three defects, two of which point the same way:

**(a) mixed basis.** +43.5% divides the **profiled** 4695.2 ms wave; 234.2 tok/s is the
**unprofiled** 4566.5 ms wave. Same schedule quoted on two clocks. Consistent accounting gives
+43.5% (profiled) or +39.7% (unprofiled-vs-projection) — not both.

**(b) the prefill iterations are mispriced.** 25.5 ms is the *mean of the 108 measured mixed
iterations*, and 95% of those are starved steps that move one token per slot — the cheap kind.
A prefill iteration that actually delivers 16 tokens delivers them to **one** slot, and the
fitted law prices that at `18917 + 982.4·16 + 142.7·16 = 36.9 ms`. The backlog's own
143-iteration schedule, correctly priced:

```
36*36.9 + 107*22.2                                          = 3703 ms = +26.8%
```

**(c) 143 iterations is neither the floor nor reachable.** The floor at `mbt=16, bs=16`: every
request must walk step 0 → 131, at most 16 tokens per iteration across the whole batch, so
`16*131/16 = 131` iterations. And no *ordering* reaches even 143: `prepare_next_batch` gives the
budget to the lowest live slot first, so a slot that finished prefill immediately starts eating
one token per iteration and the `j`-th slot to finish only gets `mbt − j` — the harmonic
blow-up that produced 203 in the first place. `per_request_token_cap = 1` attains the floor
exactly, and every one of its 131 iterations is the *measured* 22003 µs step:

```
131 * 22.183 ms                                             = 2906 ms = +61.5%
unprofiled: 4566.5 -> 2825 ms                               = +61.6%
```

**Verdict: +44% is not reproducible as stated. The same schedule, correctly priced, is +26.8%;
the reachable optimum is +61.5%.** The *mechanism* in backlog #4 was also wrong — "`mbt=16`
equals the batch size so decode saturates admission" reads as "raise `mbt`", which needs I5b's
router row loop and costs step time (the table above: `mbt=32` alone is +15.5% central, and
negative at `delta=0.55`). The real defect is that **one slot may take the whole budget**.

**Metric caveat, inherited by anything that quotes the M2 wave row.**
`tokens_per_s = len(wave) * max_decode_steps / wall` (`mpk_engine_run.py:385`). At bs16 that is
`10 * 107 / 4.5686 = 234.2`: ten **distinct** prompts (six slots are duplicates and excluded)
and 107 steps (the `max_seq_length` tail, not the 64 reported tokens). It is a self-consistent
wall-clock proxy for a correctness harness; it is not the same quantity as vLLM's 16-request
decode throughput, so `3018.1 / 233.9 = 12.9×` is not a like-for-like ratio. Policy *ratios* are
unaffected. `remeasure-protocol.md` §4 re-derives the field at the pinned geometry.

## 3. Bit-exactness of the cap — adjudicated

`per_request_token_cap` changes **only how a prompt is chopped across iterations**. It changes
no arithmetic, no cast position, no reduction order within a token's own computation, and no
kernel. The question is whether chopping matters.

**It does not, and the committed report already proves it.** Nine of the ten AC-3 prompts are
prefilled with two to four *different* decompositions across batch sizes — p02-math as
`[16,16,16,16,4]` at bs1, `[4,15,15,15,15,4]` at bs2/4/8 and `[6,7,7,7,7,7,7,7,7,6]` at bs16;
p07-format as `[16,16,1]` at bs1/2/4 and `[11,12,10]` at bs8/16 — in placements that never
migrate, and M2's committed report has all 25 (prompt, batch size) sequences byte-identical
(`compaction_audit.py`, the H2 control). The cap simply selects the all-ones decomposition.

Two honest residuals:

- **Extrapolation.** Chunk sizes 1 through 16 all appear in the clean corpus (`[16,16,1]` ends
  in a 1-token chunk), but an *all*-ones prefill for a whole prompt does not. If a prefill-only
  code path is entered on `qo_len > 1` and takes a different reduction order, the cap would
  route every token through the `qo_len == 1` path instead. That is a *narrowing* of paths, not
  a widening, and the `qo_len == 1` path is the one M2 proved bit-exact against HF — but it is
  a prediction, and stage 2 of the plan is what settles it.
- **Only 4.7% of today's bs16 prefill tokens are already delivered one at a time** (27 of 569).
  The "it is already the common path" intuition is false; checked, and rejected.

Predicted outcome: **all 25 cases byte-identical, and the six bs16 duplicate-slot
`identical: false` results flip to `true`** — the latter is the sharper test, because nothing
else in the tree has ever moved it.

## 4. Falsifier — six ordered numbers, and it costs no GPU time at all

H1 (compaction) and H2 (chunk decomposition) both predict "all six bs16 duplicate pairs
disagree", so the boolean `identical` cannot separate them and never could. **Where** they
diverge can. Under H1 a copy must agree with its original right up to the token produced by the
iteration that migrated its slot; under H2 the copy's prefill is different from token zero, so
generated index 0 already differs.

`dup_checks` now records `first_divergence` (`mpk_engine_run.py`), so the next bs16 AC-3 run
that happens for *any* reason — I2b's, I8's, or stage 3 of this plan — settles it. Registered
prediction, from `protocol_sim.py` (`step` of each slot at iteration 101, the iteration that
migrates all six):

| slot | prompt | plen | step at migration | H1: `first_divergence` must be ≥ | H2 predicts |
|---:|---|---:|---:|---:|---:|
| 10 | p06-poem | 24 | 83 | **60** | ~0 |
| 11 | p01-history | 30 | 83 | **54** | ~0 |
| 12 | p04-chinese | 32 | 77 | **46** | ~0 |
| 13 | p09-translate | 32 | 66 | **35** | ~0 |
| 14 | p07-format | 33 | 51 | **19** | ~0 |
| 15 | p05-cuda | 36 | 19 (still prefilling) | 0 | ~0 |

Slot 15 does not discriminate — it is migrated mid-prefill, so both hypotheses allow index 0.
Slots 10–14 do, and they carry a **strictly decreasing** ordering that H2 has no mechanism to
produce. Pass condition: `first_divergence` ≥ the bound for all of 10–14, monotone decreasing.
Any value materially below its bound refutes H1, and with it the anchor under §3's
bit-exactness argument — stop and root-cause before building anything.

Second, independent, one extra run and no source change (stage 1): re-run the same bs16 wave at
`--max-seq-length 212`. Retirement moves to step 211, so the first migration lands at iteration
181 while the last reported window closes at 180: **zero straddling slots**, predicted 293
iterations. If the six pairs come back `identical: true`, C1+C2 hold on measured ground. This
also supplies the negative control that `compaction_audit.py` currently lacks — 14 of 14
agreeing, but not one predicted-*safe* duplicate in the whole corpus. (It costs one megakernel
JIT, 1–10 min, because `max_seq_length` is baked into the graph; no source changes. It is not a
perfectly single-variable test: raising the retirement step also lengthens slot 15's prefill
tail, so read it together with C3, which is what makes that harmless.)
