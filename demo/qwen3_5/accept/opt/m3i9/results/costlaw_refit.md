# C7 root cause — the cost law's shape held; its anchor was two windows stale

**Verdict: not a functional-form failure. PROCEED with stages 3–8 at cap = 1, after re-registering
C5's band.**

The law `iter_us = a + b·max_chunk + c·n_live` was fit on M3-I1's 202 measured bs16 iterations and
then used to predict an **absolute** wall time for a run made in the M3-I9 window. Between those
two windows the binary got ~19 % faster on the *identical* schedule. Once predictions and
measurements are compared inside one window — as ratios — the law's held-out error is **−0.90 %**
on sorted-padded and **−0.92 %** on the msl=212 control. Every candidate functional form,
including a constant-cost null that ignores iteration shape entirely, misses the absolute number by
+20.5 % to +23.3 %. That is the signature of a scale error, not a shape error.

Reproduce: `python3 costlaw_refit.py` (output pinned in `costlaw_refit.log`, machine-readable in
`costlaw_refit.json`). CPU only.

---

## 1. The miss, decomposed

The window compared a prediction anchored on one binary against a measurement from another:

| | iterations | measured wall (unprofiled) | source |
|---|---:|---:|---|
| M3-I1, shipped order, msl 132 | 203 | **4566.5 ms** | `opt/attribution.csv` (`wave_wall_ms_unprofiled`) |
| M3-I9 stage 0, shipped order, msl 132 | 203 | **3689.27 ms** | `results/out/s0_base/timings_bs16.json` |

Same 16 prompts, same slot order, same `mbt`, same `max_seq_length`, same 107 decode steps, same
203-iteration schedule. **−19.21 %.** Nobody in the window compared stage 0 against the baseline it
was standing in for, so a 19 % anchor error was carried straight into stage 2's verdict.

That fully covers the reported miss: predicted 4214 ms profiled, measured 3437 ms profiled, ratio
0.8156; the baseline ratio is 0.8083. The two agree to 0.9 %.

### Independent confirmation, one window earlier

M3-I8's own analyzer laid the same quantity side by side (`m3i8/results/analyze_final2.log`,
"Step time and decode throughput"; `I1_STEP` in `analyze_m3i8.py`). Its *base* arm — with I8's own
change proven inert at bs16 (base 18736 vs v1 18739 µs) — already measured:

| bs | M3-I1 step µs | M3-I8 base step µs | Δ rel | Δ abs |
|---:|---:|---:|---:|---:|
| 1 | 15264 | 11715 | −23.3 % | 3549 µs |
| 2 | 15648 | 12474 | −20.3 % | 3174 µs |
| 4 | 15645 | 11868 | −24.1 % | 3777 µs |
| 8 | 18618 | 15610 | −16.2 % | 3008 µs |
| 16 | 22005 | 18736 | −14.9 % | 3269 µs |

The **absolute** saving is near-constant (3008–3777 µs, mean 3355) while the relative saving falls
with batch size. That is the signature of a fixed per-iteration cost being removed.

**Leading candidate: `624e8e1`** — "quantize row-partition fix", landed between the M3-I1 window and
the I8/I9 windows, default-ON for the four qwen3.5 sites. Its commit message: `quantize_fp8_layer`
registered `input_map=(-1,-1,-1)`, so all 16 tasks quantized the whole tensor — 124800/133120
redundant row-quantizations per step, "29.7 % of the bs1 step". Measured bs1 saving is 23.3 %:
same mechanism, same order, honestly a bit less. This is a strong candidate, not a proof — no
one-variable A/B of `624e8e1` was ever run. **Falsifier:** rebuild with `row_partition` off and
re-measure the bs16 wave; if it does not return to ≈4.5 s the cause is elsewhere (box clocks,
co-tenancy, toolkit) and only the recommendation in §5 about same-window controls survives.

---

## 2. Refit table — fit on shipped-order only, scored on same-window held-out ratios

Fit set: `opt/tables/bs16_iters.csv`, M3-I1's 202 profiled iterations. Held-out: the M3-I9 window's
own runs, scored as **ratios against stage 0**, which cancels any binary/clock scale:

- sorted-padded / shipped = 3341.49 / 3689.27 = **0.90573** (median of 3 reps, spread 0.41 %)
- msl 212 / msl 132, shipped = 5282.53 / 3689.27 = **1.43186**

| model | R² | med \|res\| µs | r_sorted err | r_msl212 err | absolute err |
|---|---:|---:|---:|---:|---:|
| M0 constant | 0.000 | 2660 | −2.65 % | +0.80 % | **+20.5 %** |
| M1 a + b·chunk | 0.914 | 906 | −1.57 % | −2.11 % | +21.8 % |
| M2 a + c·live | 0.048 | 2116 | −1.96 % | +2.01 % | +21.4 % |
| M3 a + d·tokens | 0.510 | 1295 | −0.33 % | +2.87 % | +23.3 % |
| **M4 shipped: a + b·chunk + c·live** | **0.960** | **269** | **−0.90 %** | **−0.92 %** | +22.6 % |
| M5 + tokens | 0.961 | 252 | −0.90 % | −0.95 % | +22.6 % |
| M6 + n_prefill | 0.961 | 296 | −0.90 % | −0.57 % | +22.6 % |
| M7 + prefill dummy | 0.961 | 260 | −0.86 % | −1.61 % | +22.7 % |
| M8 a + b·log₂chunk + c·live | 0.924 | 577 | −0.69 % | −2.72 % | +22.9 % |
| M9 a + b·√chunk + c·live | 0.957 | 355 | −0.79 % | −1.85 % | +22.8 % |
| M10 a + b·⌈chunk/2⌉ + c·live | 0.951 | 498 | −0.95 % | −0.34 % | +22.5 % |
| M11 a + b·⌈chunk/4⌉ + c·live | 0.884 | 602 | −0.96 % | +0.51 % | +22.5 % |
| M12 + excess_tokens (tokens − max_chunk) | 0.961 | 252 | −0.90 % | −0.95 % | +22.6 % |
| M13 + live×chunk | 0.964 | 231 | −0.88 % | −1.26 % | +22.6 % |

Reading it:

- **The absolute column is flat at ~+22 % for every model, including M0.** A functional-form
  failure cannot do that; a stale scale does exactly that.
- **The four hypothesised form failures are all refuted.** Concavity in `max_chunk` (M8/M9/M10/M11)
  fits *worse* in-sample than the linear term and is no better held-out. Prefill/decode overlap
  (M12) and a token-total term (M5) add nothing (R² 0.9598 → 0.9609). A regime split (M6/M7) adds
  nothing. Only M13's interaction term improves in-sample at all (0.9598 → 0.9636), and it is
  slightly worse on the msl=212 held-out point.
- **Best model is still M4**, the shipped law — it is the only one under 1 % on *both* held-out
  ratios. M5/M6/M12 tie it.

### The residual −0.9 % is the anchor, not the shape

Both held-out ratios err by the same −0.9 %, in the same direction, on schedules that differ in
opposite directions from the fit set (sorted-padded is 46.4 % `max_chunk=1`, shipped is 56.7 %,
msl 212 is 72.0 %). A shape defect would have to change sign between them. A single explanation
covers both: **stage 0 is n = 1**, and a ~0.9 % fast rep biases both ratios high by 0.9 %. Stage 2's
3 reps spread 0.41 %, so that is well inside run-to-run noise.

An exactly-identified two-parameter re-anchor (`s` on the flat+live terms, `t` on the chunk term)
solved from the two msl=132 totals returns `s = 0.695, t = 1.554` — a nonsense pair, because
sorted-padded's mean `max_chunk` (3.363) is barely different from shipped's (3.103) and the system
is near-collinear. A three-constraint affine re-anchor (`new = α + β·old`, pinned to I8's measured
step) returns `β = −0.039` and is **refuted on held-out**: it predicts sorted-padded at 3250 ms vs
3341 measured (−2.73 %), worse than the plain uniform rescale (−0.90 %). Both are reported so the
band in §3 is honest, neither is adopted.

---

## 3. Re-derived cap predictions, re-anchored on the same-window control

All figures unprofiled ms, anchored on stage 0 = 3689.27 ms. Iteration counts and migration counts
are the simulator's, and the simulator is *trusted*: it predicted stage 2's 179 iterations, 91
migrations and 4 straddling slots exactly (`results/out/s2_sorted_rep1/timings_bs16.json`).

| policy | iters | migrations | straddling | predicted ms | vs shipped |
|---|---:|---:|---:|---:|---:|
| shipped (control) | 203 | 69 | 6 | 3689 (measured) | 1.000× |
| **cap = 1** | **131** | **0** | **0** | **2160 – 2390** | **1.55 – 1.71×** |
| cap = 2 | 201 | 65 | 5 | 3341 – 3413 | 1.081 – 1.104× |
| cap = 4 | 204 | 71 | 5 | 3510 – 3559 | 1.037 – 1.051× |
| cap = 8 | 203 | 73 | 6 | 3626 – 3653 | 1.010 – 1.017× |
| hold-decode | 143 | — | — | 2694 – 2828 | 1.30 – 1.37× |
| sorted-padded | 179 | 91 | 4 | 3310 – 3315 (meas. 3341) | 1.11× |

Bands are the spread over the credible model set (R² ≥ 0.95 **and** both held-out errors ≤ 2 %:
M4, M5, M6, M7, M9, M10, M13). The cap = 1 row additionally folds in anchor A below, which is the
one estimate that uses no model at all.

### cap = 1 priced four independent ways

The window report worried that cap = 1 is "an even more extreme extrapolation". **It is the
opposite.** Under cap = 1 every slot takes exactly one token per iteration, all 16 slots advance in
lockstep and all retire together at step 131, so **all 131 iterations have the identical shape
`(max_chunk = 1, n_live = 16)`** — and that shape is the single most densely measured point in the
fit corpus: 19 of M3-I1's 202 bs16 iterations, mean 22002.1 µs, spread 0.49 %. cap = 1 requires no
extrapolation in either regressor; it can be priced by table lookup.

| | cap = 1 (ms) | vs shipped |
|---|---:|---:|
| A direct measured regime: 131 × I8's 18736 µs | 2386 | 1.546× |
| B shape-free null: 131/203 × stage 0 | 2381 | 1.550× |
| C affine re-anchor (α, β) | 2385 | 1.547× |
| D uniform rescale of the M3-I1 law (M4) | 2284 | 1.615× |
| credible-model-set spread | 2158 – 2327 | 1.586 – 1.710× |

A, B and C cluster at 2381–2386 ms; D and the model set run lower. The gap is a real, unresolved
5 %: the uniform rescale implies the `(1, 16)` iteration costs 22002 × 0.8083 = 17784 µs on the
current binary, while I8 measured 18736 µs. Cross-window difference (I8 window vs I9 window) is the
most likely cause, and it is what §5's measurement settles.

**Registered band: 2.16 – 2.39 s unprofiled, central 2.28 – 2.39 s (+54 % to +62 %).**

### Sensitivity — the optimum did not move

cap = 2 buys 8–10 % by flattening chunks but leaves **65 migrations and 5 straddling slots**;
cap = 4 and cap = 8 raise the migration count *above* shipped (71 and 73 vs 69) and are worth only
1–5 % on perf. Only cap = 1 reaches zero migrations and zero straddling, and it is also the perf
winner by roughly six times the margin of the next-best cap value. There is no setting of the knob
for which the ranking flips — the sensitivity sweep was worth running and it came back negative.

---

## 4. What stands regardless of pricing

The cap's **correctness** case is untouched by any of this and was independently confirmed in the
same window:

- Stage 0's duplicate-slot first-divergence test passed on all five discriminating slots, strictly
  decreasing 60/58/46/35/19 against bounds 60/54/46/35/19 (`results/logs/s0_divergence.txt`) —
  live-slot compaction is real, not a numerics artifact.
- Stage 1 (`--max-seq-length 212`) returned all six duplicate pairs `identical: true` with zero
  straddling slots, the predicted negative control (`results/out/s1_msl212/timings_bs16.json`).
- cap = 1 is the only policy in the whole ranking with 0 migrations and 0 straddling requests.

Everything in this document is about the **perf claim and the GPU-spend justification** only.

---

## 5. Recommendation

**PROCEED to stages 3–8 with cap = 1**, subject to three changes to the plan:

1. **Re-register C5.** Its pre-registered falsifier is "measured bs16 wave outside 2.75–3.05 s
   (unprofiled)". That band was built on the stale 4566.5 ms anchor and is now *above* the
   corrected prediction — a working cap = 1 will land at 2.16–2.39 s and **falsify C5 as written**.
   Replace with **2.10 – 2.55 s**, and state the claim as a ratio: `shipped_wall / cap1_wall`
   in **1.50 – 1.75×**.
2. **Every stage carries a same-window shipped-order control.** The claim is the ratio to that
   control, never an absolute ms against a table from a previous window. Run the control with
   **≥ 3 reps** — stage 0's n = 1 is the largest single source of uncertainty in §3 and costs ~7 s
   of GPU time to remove.
3. **One extra measurement, ~1 GPU-minute, that closes the last 5 %:** a single *profiled*
   `profile_wave.py --batch-size 16` shipped-order rep in the cap window (`--save-raw`, exactly the
   capture `run_m3i8.sh:69-85` already does). It yields per-iteration `iter_us` on the current
   binary, which (a) directly measures the `(max_chunk=1, n_live=16)` step that prices cap = 1,
   deciding between the 2386 ms and 2284 ms anchors, and (b) lets `cost_model.py`'s `--fit` re-derive
   `(a, b, c)` for the current binary so the ranking is priced on the machine it will run on. Do
   this **before** stage 4, not after.

Also worth doing while the plan is open: the +61.6 % headline (`4566.5 → 2825 ms`) in
`predictions.md` §2 and `plan_m3i9_as_run.sh:39` should be restated as `3689 → 2160–2390 ms`. The
*relative* claim survives almost intact; the absolute pair in the plan is stale on both sides and
will mislead the next reader the same way it misled stage 2.
