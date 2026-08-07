# E7 — why W13's megakernel body win is −16.08% at the median but only −9.01% at the mean

Date: 2026-08-06 (box time). Analysis-only pass over ALREADY-CAPTURED E6 data; no GPU work, no
rebuilds, no new captures. Resumed session: the extraction + most analyses were produced by a
prior agent (mtimes 12:07–12:31); this session verified them against the raw captures
(`e7_verify.log`, all checks PASS), added the population start-shift and normal-window analyses,
and wrote this report. Interpretation is out of scope (goes to cross-review); every sentence
below is arithmetic on the captured data.

**Answer in one line:** the 7.07 pp mean-vs-median split is NOT mainly a size-dependent win:
1.96 pp is an estimator artifact of ratio-of-medians on W13's bimodal mix, 0.12 pp is size-profile
weighting, and 4.99 pp is a within-size tail — of which 4.14 pp is a single event class: 8.74% of
paired instances whose ON body jumps a tight +7.4–9.9 µs quantum above its own id-median while the
SAME (task id, iteration) input runs normally in OFF. These events are input-independent
(mechanism A insufficient) and are NOT accompanied by elevated co-tenant density at onset or over
their would-be-normal window (mechanism B in its "denser windows" placement sense: refuted), but
they are 8.3x rarer in windows with zero other-type overlap in both arms (a position-conditional
association the data cannot causally resolve; the separating captures are listed in §8).

Pairing basis: routing seed-pinned, tokens byte-identical, task graph sha identical across arms →
(task id, iteration) pairs exactly. 352 iterations captured; E6's window [2,350) used throughout;
live = body bracket ≥ 4 µs; `ok` = ring/profiler timestamp containment (0 failures out of 14.4 M).
W13: 3,534,896 exact pairs (live flags agree across arms on ALL 3,604,480 (id,it) — 0
disagreements); W2: 3,549,875 pairs (13,645 in-window live disagreements). Executing worker is
identical across arms for 100.0% of pairs; the per-(iteration, scheduler-group) queue rotation is
identical in both arms (group 0 rotated by +2 in all 352 iterations, all other groups direct —
verified element-for-element against the static schedule).

## 1. Paired per-instance body deltas (analysis 1) — `e7_paired.json`

Per-pair delta = 100·(body_ON − body_OFF)/body_OFF on exactly-paired (id, iteration).

| family | n pairs | p10 | p25 | p50 | p75 | p90 | mean | frac pairs regressed (Δ>0) |
|---|---|---|---|---|---|---|---|---|
| **W13** | 3,534,896 | −20.22% | −17.09% | **−14.12%** | −10.97% | **+17.70%** | **−9.36%** | 11.51% |
| W2 (control) | 3,549,875 | −25.06% | −21.12% | −16.77% | −10.48% | +9.77% | −12.62% | 17.43% |

Reconciliation with E6 (exact): recomputing E6's per-arm aggregates from the joined data gives
W13 mean 18.087→16.457 µs = **−9.01%** and median 14.528→12.192 µs = **−16.08%** — identical to
E6's REPORT numbers; the paired-selection aggregates are the same (−9.01 / −16.08). W2: −13.49 /
−13.45 (paired; per-arm −13.27 / −13.45). So the split is fully reproduced by the pairing, and
W13's p90 pair at **+17.7%** vs W2's +9.8% is the tail asymmetry to explain.

In µs: W13 median pair −2.27 µs, p90 pair **+4.48 µs**, mean −1.63 µs.

## 2. Test (A): size dependence (analysis 2) — `e7_paired.json`, `e7_quantify.json`

Pairs binned by OFF body decile (work-size proxy). "clean" = pairs with no excursion in either
arm (excursion = body > id-median + 4 µs, see §4).

**W13** (OFF body is bimodal: deciles 1–4 ≈ 11.2–11.9 µs small mode, 5–6 ≈ 14–15 µs, 7–10 ≈
23–27 µs big mode):

| decile | n | OFF med µs | pair Δ med | pair Δ mean | clean-pair Δ med | clean Δ mean | ON-only exc rate | OFF-time share | contrib to mean Δ |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 296,828 | 11.23 | −15.43% | −15.07% | −15.43% | −15.10% | 0.05% | 5.21% | −0.79 pp |
| 2 | 388,006 | 11.39 | −15.54% | −13.85% | −15.60% | −15.13% | 1.88% | 6.92% | −0.96 pp |
| 3 | 353,985 | 11.62 | −15.83% | −9.29% | −16.12% | −15.16% | 8.78% | 6.43% | −0.60 pp |
| 4 | 374,100 | 11.94 | −16.03% | −1.66% | −17.12% | −14.56% | **20.75%** | 7.12% | −0.12 pp |
| 5 | 354,042 | 14.11 | −19.44% | −13.74% | −19.72% | −16.92% | 5.46% | 7.79% | −1.07 pp |
| 6 | 349,802 | 15.04 | −18.26% | −8.60% | −20.68% | −13.90% | 10.27% | 9.35% | −0.80 pp |
| 7 | 348,976 | 23.33 | −11.84% | −9.11% | −11.96% | −11.62% | 5.32% | 12.88% | −1.17 pp |
| 8 | 351,433 | 25.34 | −11.41% | −8.63% | −11.88% | −11.61% | 7.95% | 13.92% | −1.20 pp |
| 9 | 359,297 | 25.92 | −11.68% | −8.16% | −12.06% | −12.11% | 10.66% | 14.56% | −1.19 pp |
| 10 | 358,427 | 26.59 | −12.18% | −7.04% | −13.01% | −12.87% | 14.71% | 15.81% | −1.11 pp |

Size dependence is real at the typical level: big-mode deciles 7–10 (57.2% of OFF body time)
improve −11.9 to −13.0% (clean median) vs −15.4 to −20.7% for deciles 1–6. But its contribution
to the mean-median split is small: weighting each decile's per-pair MEDIAN delta by its OFF-time
share gives C = −14.00%, vs the plain per-pair median −14.12% — size-profile weighting moves the
typical win by only **+0.12 pp**. The step from E6's −16.08% to the per-pair median −14.12%
(**1.96 pp**) is a pure estimator artifact (ratio of aggregate medians vs median of per-pair
ratios on a bimodal mix; exhibit: per-iteration-decile aggregate-median deltas swing from −18% to
+25% across the decode while the per-pair median stays −13.5 to −14.9%, `e7_paired.json
iteration_deciles`). The rest of the gap, mean −9.01% vs C −14.00% = **+4.99 pp**, is
within-size-decile tail asymmetry (`e7_decomp.json`: 2.08 = 1.96+0.12, 4.99, sum 7.07).

**Within-id vs across-id variance split (the brief's direct test)** — `e7_paired.json`:

| quantity | W13 OFF | W13 ON | W2 OFF | W2 ON |
|---|---|---|---|---|
| pooled IQR (µs) | 13.76 | 12.35 | 4.93 | 4.03 |
| within-id IQR, median id (µs) | **0.58** | 0.87 | 3.08 | 2.88 |
| within-id IQR p90 (µs) | 11.46 | 10.83 | 3.74 | 3.66 |
| across-id IQR of id-medians (µs) | 13.76 | 12.26 | 3.99 | 2.93 |
| variance share within-id | 36.2% | **45.8%** | 33.8% | 49.8% |
| variance share between-id | 63.8% | 54.2% | 66.2% | 50.2% |

Same-id-same-work verification: id→expert is static (task graph, sha-identical across arms);
67.8% of W13 ids keep OFF body IQR < 1 µs across all ~348 iterations (30.1% ≥ 4 µs — per-step
routing volume genuinely varies for those); the within-id residual OFF↔ON correlation is
**0.825** (between-id 0.989) — what varies across iterations for an id is pinned to
(id, iteration) and shared across the two independent runs, i.e. input-side, not run noise.

**Arithmetic sentence (A):** size explains the −16.08→−14.0 move (2.08 pp, of which 1.96 pp is
the estimator itself) but cannot explain the remaining 4.99 pp: ids whose OFF body is stable to
<1 µs across 348 iterations still regress >+10% in ~10% of their iterations (median id 9.8%, p90
21.8%, and 10,232/10,240 ids regress at least once — spread, not id-anchored), and the ON side of
the variance split rises from 36.2%→45.8% while the per-(id,it) input is byte-identical — a
work-size-only mechanism predicts none of that.

## 3. Test (B): co-tenancy of the tail (analysis 3) — `e7_overlap.json`, `e7_startshift.json`, `e7_first10.json`, `e7_normwin.json`

Per-instance overlap features from the realized schedule (E6's own profiler pairs; every one of
the 14.4 M task pairs categorized): mean live count and coverage of other-same-family tasks
(`own`), any-other-type tasks (`oth`, for W13 this includes W2), scheduler/bookkeeping tracks
(`sched`), instantaneous counts at the instance's begin, stage-span position, wave rank.

Correlations of per-pair delta vs environment are weak for W13 (all |pearson| ≤ 0.27; largest is
the duration-confounded on.end_pos +0.27 and on.own_meancnt −0.18): no overlap feature linearly
explains the tail. Key bucketed views:

**W13 delta by ON whole-window other-type mean count** (40% of W13 windows have exactly zero):

| bucket | n | oth mean cnt | Δ med | Δ mean |
|---|---|---|---|---|
| =0 | 1,413,954 | 0.00 | −15.43% | −14.63% |
| 5 | 352,396 | 0.28 | −14.61% | −6.78% |
| 6 | 354,562 | 0.54 | −11.63% | −1.88% |
| 7 | 353,510 | 0.74 | −11.51% | −2.75% |
| 8 | 353,485 | 0.97 | −11.55% | **+2.84%** |
| 9 | 353,469 | 1.44 | −14.17% | −11.01% |
| 10 | 353,520 | 1.92 | −15.82% | −15.51% |

Non-monotone: the worst mean deltas sit at INTERMEDIATE whole-window overlap (buckets 5–8), and
the densest windows (bucket 10) improve as well as zero-overlap ones. The same table on the OFF
arm's own overlap is flat (−6.9 to −10.9% mean) — the association is specific to the ON window.

**Isolation split** (`e7_startshift.json`): windows with ZERO other-type overlap in BOTH arms =
35.35% of W13 pairs: per-pair mean −14.98%, p90 −9.31% (regression tail absent), >+10% regression
rate 2.23%, ON-only excursion rate **1.53%**; windows with some overlap: regression rate 16.56%,
excursion rate **12.68%** (8.3x), mean −6.29%. 93.8% of all ON-only excursions occur in the
64.65% of windows with some other-type overlap. (The zero-overlap subset's ratio-of-medians is
+27.24% — a bimodal composition artifact, OFF median 16.10 µs vs ON median 20.48 µs land on
different modes; its per-pair median is −15.23%. `e7_quantify.json`.)

**But density at execution time is NOT elevated for tail events:**
- First 10 µs from begin (`e7_first10.json`): excursed W13 instances average 0.451 other-type
  co-tenants vs 0.558 for others; 65.5% of excursions have ZERO other-type overlap in their first
  10 µs (others 59.6%).
- Whole would-be-normal window [begin, begin + id-median-ON-body) (`e7_normwin.json`): excursed
  0.605 vs others 0.775 mean other-type count; zero-fraction 43.5% vs 40.2%; own-family 115.2 vs
  113.5 (by size-quintile the pattern is mixed: q1–q3 lower-or-equal for excursions, q5 slightly
  higher 0.38 vs 0.20).
- Instantaneous other-type count at begin: excursed 0.524 vs others 0.603.
- Scheduler proximity: sched mean count 0.0178 (excursed) vs 0.0186 (others) — flat; correlation
  with delta +0.007 (W13).

**Stage position:** excursions start slightly EARLIER in the stage span (start_pos 0.185 vs
0.232; wave rank 118.9 vs 127.7) — not in the drain/boundary zone; drain-wave fraction is
identical across arms per pair (0.350 vs 0.350) because placement is deterministic. The
`on.end_last_decile` zone's mean +1.19% is duration-confounded (an extended instance ends later)
and is not treated as evidence.

**Start-shift analysis (does ON move starts into denser windows?)** — `e7_startshift.json`:
same executing worker 100.0%; wave-rank correlation 0.908 across arms; start-in-iteration shift
median −26.8 µs (p10 −65.5, p90 +54.7) — ON starts earlier, consistent with upstream stages
finishing faster; shift is uncorrelated with the pair delta (pearson +0.014, spearman −0.033;
bucket medians flat −13.3 to −15.1% across all shift deciles); the begin-time other-type density
CHANGE (ON−OFF) is slightly negative (mean −0.13, p50 0) and uncorrelated with delta (−0.001);
whole-window Δoth mean +0.019. Excursion-vs-others shift is identical (−27.4 vs −26.8 µs median,
`e7_tail.json`). The ON arm does NOT shift instances into denser co-tenant windows.

**Structure of the tail events** (`e7_tail.json`, `e7_adjacency.json`): per-id-median residuals
with a 4 µs threshold give OFF-excursion rate 8.44%, ON 15.87%; only-ON 308,917 (8.74%), only-OFF
46,145 (1.31%), both 252,047 (7.13%). Shared (both-arm) excursions cluster heavily by
(stage, iteration) cell (variance/binomial = 63.6 — token-burst-like, input-side, and they do NOT
hurt the delta: their pair Δ median is −11.7%); ON-only excursions cluster far less (7.2), have
no same-worker adjacency (lift 0.98–1.12 at lags 1–5), are spread across all 348 iterations
(top-10% iterations hold only 11.8%), across workers (max rate 9.9% vs base 8.7%), and mildly
concentrate by stage (top stage 13.7% vs base 8.7%). Their residual is a tight quantum: p10–p90 =
**+7.4 to +9.9 µs** (median +8.26); the excursed body lands at 20.93 µs ≈ the big mode, from an
id-median of 11.87 µs ≈ the small mode. ENTRY (T0→T1) and EXIT (T2→T3) medians are unchanged for
regressed pairs (0.224/0.064 µs vs 0.160/0.064) — the +8 µs is entirely in the BODY phase.
Highest incidence at the small/big mode boundary (size-decile 4: 20.75%; `e7_quantify.json`).

**Arithmetic sentence (B):** the tail is not a placement/density effect — the ON arm keeps
worker, order, and near-identical start environments (§ above), and tail events show LOWER
other-type density at onset (0.45 vs 0.56), over their would-be-normal window (0.61 vs 0.78), and
at begin (0.52 vs 0.60) — but tail events are 8.3x depleted in windows with zero other-type
overlap in both arms, so the +8 µs event class is CONDITIONAL on other-type activity existing
somewhere in the instance's realized window while NOT scaling with its density; a
whole-window-overlap causal reading is confounded by the extension itself (the +8 µs catches
co-tenants mechanically), and the data cannot separate "other-type presence causes the event"
from "the schedule region that hosts other-type presence hosts the event".

## 4. Reconciliation (analysis 4) — `e7_decomp.json`, `e7_quantify.json`

Exact chain for W13's 7.07 pp mean-vs-median gap (−9.01 vs −16.08):

| component | pp | source |
|---|---|---|
| ratio-of-medians vs median-of-ratios on the bimodal mix | **1.96** | −16.08 → −14.12 |
| size-profile weighting of typical (median) per-decile deltas | **0.12** | −14.12 → C = −14.00 |
| within-size tail asymmetry | **4.99** | C → mean −9.01 |
| — of which the ON-only +8 µs excursion class (8.74% of pairs) | **4.14** | counterfactual: pulling only those pairs back to their id-median ON body gives mean −13.15% |
| — remaining within-decile skew | 0.85 | residual |

Additive check by excursion class (sums to −9.01 exactly): no-excursion pairs contribute
−10.66 pp, ON-only +2.74 pp, both −0.87 pp, OFF-only −0.23 pp. Removing all excursion classes on
both sides gives mean −13.01% / aggregate-median −19.68%.

**Split sentence:** mechanism (A) work-size-dependent win exists (big-mode typical −12 to −13%
vs small-mode −15 to −21%) but supplies only ~2.1 pp of the 7.07 pp split — and 1.96 pp of that
is the median estimator itself, not physics; the dominating 4.99 pp is an ON-arm-specific,
input-independent +8 µs event class (4.14 pp), which is neither pure (A) — the same (id,it) input
runs normally in OFF and excursion incidence is non-monotone in size — nor confirmed (B) — no
elevated co-tenant density at onset/normal-window, no start densification — but is
position-conditional on other-type activity within the window (8.3x). The data cannot separate a
same-window co-tenant CAUSE from a shared-position confound; §8 lists the separating captures.

## 5. W2 control (analysis 5) — `e7_paired.json`, `e7_decomp.json`, `e7_quantify.json`

W2 has NO aggregate split (mean −13.49 vs median −13.45, gap −0.05 pp) — but not because it lacks
tail events: it has 4.67% ON-only excursions (+4.99 µs median residual, +0.69 pp contribution;
counterfactual removal moves the mean −13.49→−15.56%). The gap closes by CANCELLATION: W2's size
profile runs the opposite way (its biggest decile improves MOST, clean median −20.9%, vs −13.8%
for decile 1), giving size-profile −3.89 pp against tail +3.84 pp (`e7_decomp.json`). W2's OFF
body is unimodal (deciles 7.0→17.5 µs, no mode gap), so the estimator artifact is absent
(aggregate median −13.45 vs per-pair median −16.77 differ for the usual mean-of-ratios reasons
but the two aggregate estimators coincide). Its overlap features correlate more with delta
(on.oth_meancnt spearman +0.41; W2 windows are 97.5% non-isolated) yet the same onset/normal-
window tests point the same way as W13 (excursed 2.29 vs others 3.41 normal-window other-type
count). This is consistent with the E6 fact that only W13 shows the mean-median split.

## 6. Verification of the resumed artifacts — `e7_verify.log`

- All 13 input checksums in `INPUTS.sha256` re-verified byte-identical at resume.
- Fresh rejoin from RAW profiler npz + RAW ring for workers {0,29,77,135} x both arms x both
  families: counts, monotone begins, T0 containment mod 2^32, and exact equality of
  begin/end/d1/d2/d3 vs the extracted npz — all PASS.
- Static-schedule cross-check on ALL 14.4 M rows: group(executing worker) == group(queue owner);
  rows where they differ form exactly 352 rotated (iteration, group-0) cells per arm per family,
  all 3-worker groups, owner_idx == (worker_idx+2)%3 constant per cell — PASS (the one initial
  FAIL was this verifier's own inverted assertion, corrected in `e7_verify_v3fix.py`; the data
  was right).
- Headline recompute with fresh selection code (live recomputed from tlive, fresh percentile /
  variance / decomposition / excursion code): E6 per-arm aggregates, paired aggregates, per-pair
  percentiles, frac regressed, decile medians, variance shares, C and tail components, excursion
  counts — all equal to the prior agent's JSONs to the printed precision — PASS.

## 7. Files

Scripts + outputs (all under /var/tmp/e7_pairing/): `e7_extract.py` (+`.log`,
`e7_extract_meta.json`, per-arm npz), `e7_paired.py` (+`.json/.log`), `e7_decomp.py` (+`.json`),
`e7_overlap.py` (+`.json/.log`, `_ov.npz` feature files), `e7_tail.py` (+`.json/.log`),
`e7_adjacency.py` (+`.json`), `e7_first10.py` (+`.json`), `e7_verify.py` + `e7_verify_v3fix.py`
(+`e7_verify.log`), `e7_startshift.py` (+`.json/.log`), `e7_normwin.py` (+`.json/.log`),
`e7_quantify.py` (+`.json/.log`).

Inputs consumed (sha256/mtime in `INPUTS.sha256`; re-verified this session): the E6 phase rings
(`e6_off.bin`, `e6_on.bin`), E6 profiler npz + meta (both arms), the task graph json (identical
sha both arms), `trace_lib.py`, E6 `analysis.json` + `REPORT.md`, and the schedule-method
references `sched_gap.py` + `sched_gap_realized.py`. The part1 rings and `width_realized.py` were
NOT needed: E6's own profiler npz carries the realized schedule (as the brief preferred).

## 8. What this data cannot separate, and the capture that would

The remaining ambiguity is confined to the +8 µs ON-only event class: same-window other-type
presence gates it (8.3x), but its instantaneous density does not scale it, and the whole-window
association is partially self-caused (the extension catches co-tenants). Captures that would
separate cause from position:
1. Per-instance `%smid` stamps (one register read per task) → SM-local co-residency instead of
   device-global counts; directly tests whether an other-type CTA (or scheduler CTA) shares the
   SM during the excursion.
2. An ON-arm replay with stage serialization (drain between W13 and neighbor stages): excursions
   vanishing under isolation ⇒ window co-tenancy causal; persisting ⇒ intrinsic to the v024 body
   under MPK residency (memory-system or clock state), not task co-tenancy.
3. A per-(expert, iteration) routed-token-count dump would close the last input-side loophole
   directly (today it is inferred, tightly, from OFF-body stability + cross-arm residual
   correlation 0.825).
