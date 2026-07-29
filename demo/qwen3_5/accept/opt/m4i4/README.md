# M4-I4 — the prefill / AC-5 track: landing the admission cap

Two things happened here. The admission-cap policy stopped being a convention and became code
(`accept/admission_policy.py`), and it was re-measured properly — both geometries, both arms
interleaved inside one GPU claim, a compiled kernel per arm, three reps everywhere.

The measurement moved the policy: **the cap is now on from bs2 up**, not from bs4, and the
mechanism recorded in `bench-protocol.md` was wrong below bs16.

Tree under test: `348a601a` (the policy landing) in an isolated clone
`~/mpk-qwen35/mirage-m4i4` with its own freshly built C++ extension
(`.memory/main/b200-env.md`, STALE-EXTENSION TRAP). Provenance in
`raw/provenance.txt`; per-run device state in `raw/audit/gpu_audit.txt`; drivers in
`scripts/`.

---

## 1. Where the policy lives now

`demo/qwen3_5/accept/admission_policy.py`. One module, one rule, stated as arithmetic:

    cap = auto = max(1, mbt // batch_size),  on for batch_size >= CAP_MIN_BATCH_SIZE (= 2)

Everything else derives from it and nothing else states it:

| consumer | how it derives |
|---|---|
| `mpk_engine_run.py` | adapter default and CLI default are `policy`; `_cap_kwargs` and the compaction replay call `resolve_int`; the timings artifact records the requested AND compiled value |
| `harness/gate_ac3_stable.{sh,py}` | default `policy`, so the stability gate certifies what the runtime ships; `run_meta.json` embeds the policy summary |
| `opt/profile_wave.py` | passthrough; records both values in its per-run meta |
| `docs/qwen35/bench-protocol.md` | points at the module and prints it, instead of restating the table |
| `opt/m3i11/scripts/e{2,3,4,7}*.py` | pinned to `none` — their committed census basis is uncapped and the default flip must not silently invalidate it |

`--per-request-token-cap` takes `policy | none | auto | <int>`. `none` reproduces every
pre-policy artifact, `results/dumps_final` included.

Why this mattered: the policy had been written down twice and the two statements disagreed for
two days (M3-I9's "bs16 only" and M3-I7's "bs4/8/16"), while the actual behaviour of any run was
whatever its driver happened to pass. `harness/tests/test_admission_policy.py` now pins the
resolution semantics, asserts that `max(1, mbt // bs)` appears in no other entry point, and
pins the mechanism claim from the CPU replay so a future change to either has to face it.

---

## 2. Performance — the A/B at both geometries

Both arms of every cell ran inside one GPU claim on GPU 6, alternated per (rep, bs), each arm
with its own kernel directory keyed by its compiled cap value and carrying a `cap.txt` the
driver refuses to contradict. Every run is drain-gated and its device state recorded before and
after. **Every arm is n=3 and none is discarded.**

### 2a. AC-3 geometry (msl=132, 64 new tokens, the 10 pinned prompts) — total wave wall

`tables/geomA.csv`

| bs | cap | uncapped ms | range | capped ms | range | speedup | replay iters u/c |
|----|----:|------------:|------:|----------:|------:|--------:|-----------------:|
| 1  | 16 | 9181.7 | 0.03 % | 9182.0 | 0.02 % | 1.000 | 956/956 = 1.000x |
| 2  | 8  | 4974.3 | 0.04 % | 4963.1 | 0.10 % | 1.002 | 498/505 = 0.986x |
| 4  | 4  | 3251.0 | 0.06 % | 3194.9 | 0.06 % | 1.018 | 308/318 = 0.969x |
| 8  | 2  | 2909.5 | 0.04 % | 2496.1 | 0.06 % | 1.166 | 228/228 = 1.000x |
| 16 | 1  | 3306.2 | 0.06 % | 1572.1 | 0.14 % | 2.103 | 203/131 = 1.550x |

bs16's 2.103x reproduces M3-I7's independently captured 2.103x (3307.9 → 1573.1 ms) to three
digits, on a different day, a different clone and a different kernel build.

### 2b. Pinned 256/1024 geometry — the M4-binding table

`tables/geomM.csv`. Decode is the prefill-subtracted slope
`bs·(D_full − D_pre)/(wall_full − wall_pre)`, vLLM's own tokens-÷-decode-window definition.

| bs | arm | cap | e2e s | range | prefill s | range | pre/e2e | decode tok/s | replay it |
|----|-----|----:|------:|------:|----------:|------:|--------:|-------------:|----------:|
| 1  | uncapped | – | 10.255 | 0.05 % | 0.311 | 3.15 % | 3.0 % | 102.7 | 1039 |
| 1  | capped | 16 | 10.255 | 0.09 % | 0.311 | 21.0 % | 3.0 % | 102.7 | 1039 |
| 2  | uncapped | – | 10.799 | 0.81 % | 0.614 | 0.33 % | 5.7 % | 200.5 | 1057 |
| 2  | capped | 8 | 10.653 | 0.47 % | 0.479 | 0.37 % | 4.5 % | 200.7 | 1055 |
| 4  | uncapped | – | 11.590 | 0.57 % | 1.190 | 0.20 % | 10.3 % | 392.7 | 1094 |
| 4  | capped | 4 | 11.146 | 0.50 % | 0.822 | 0.48 % | 7.4 % | 395.6 | 1087 |
| 8  | uncapped | – | 14.424 | 0.85 % | 2.938 | 0.23 % | 20.4 % | 711.1 | 1193 |
| 8  | capped | 2 | 12.641 | 0.41 % | 1.701 | 0.09 % | 13.5 % | 746.7 | 1151 |
| 16 | uncapped | – | 28.294 | 5.56 % | 8.252 | 0.10 % | 29.2 % | 815.1 | 1887 |
| 16 | capped | 1 | 15.426 | 6.14 % | 3.040 | 0.19 % | 19.7 % | 1318.9 | 1279 |

Cap effect, same window:

| bs | prefill | decode | e2e |
|----|--------:|-------:|----:|
| 1  | 1.000x | +0.0 % | +0.0 % |
| 2  | 1.283x | +0.1 % | +1.4 % |
| 4  | 1.447x | +0.7 % | +4.0 % |
| 8  | 1.727x | +5.0 % | +14.1 % |
| 16 | 2.714x | +61.8 % | +83.4 % |

bs4/bs8/bs16 reproduce M3-I7's +3.9 % / +14.0 % / +86.0 % — this time with both arms in the same
window, which M3-I7's bs4/bs8 rows were not (its uncapped arm came from the perfM sweep and its
capped arm from a later capsweep probe).

Two dispersion notes, stated rather than smoothed. bs16's full-run arm carries a ~6 % range in
both arms — one rep near 16.1 s / 29.8 s against two near 15.2 s / 28.2 s — the same bimodality
M3-I7 recorded at this cell; the medians are the lower cluster and the effect is 1.83x, far
outside it. bs1's capped prefill arm has one 375.7 ms rep against 310.4/311.1 ms; the median is
311.1 ms, identical to the uncapped median, and the cell is a provable no-op anyway.

### 2c. bs2, which the previous policy excluded

The exclusion argument was that `auto` = 8 is "at or near the uncapped budget, so there is
nothing to gain". The measurement disagrees, and the per-rep values are the reason to believe it
— the two arms do not overlap on either metric:

| metric | uncapped reps | capped reps |
|---|---|---|
| prefill ms | 615.0  613.9  612.9 | 480.1  478.3  478.6 |
| e2e ms | 10842.5  10798.8  10755.6 | 10662.2  10652.7  10612.3 |

1.283x on prefill and +1.4 % e2e at 256/1024; +0.2 % at the AC-3 geometry, also non-overlapping
(4973.7/4975.6/4974.3 ms against 4963.1/4961.4/4966.6 ms). The AC-3 geometry's number is smaller
for the obvious reason: its prompts are 24–68 tokens against 64 decode steps, so there is much
less prefill in the wave for the cap to act on. `CAP_MIN_BATCH_SIZE` is therefore 2.

bs1 stays uncapped, and not as a judgement call: `auto` at bs1 is `mbt`, so the extra `min()` in
`prepare_next_batch` can never fire. It measures at exactly 1.000x at both geometries with
prefill medians equal to 0.1 ms.

### 2d. The mechanism, corrected

`bench-protocol.md` attributed the win to admission SERIALISATION and quoted the replay's
iteration counts (1887 uncapped vs 1279 capped at bs16). **That is the bs16 term only.** The
replay and the stopwatch disagree everywhere else:

| bs | replay iters u/c (256/1024) | measured e2e | replay iters u/c (msl=132) | measured wall |
|----|---:|---:|---:|---:|
| 2  | 1.002x | 1.014x | **0.986x** | 1.002x |
| 4  | 1.006x | 1.040x | **0.969x** | 1.018x |
| 8  | 1.036x | 1.141x | **1.000x** | 1.166x |
| 16 | 1.475x | 1.834x | 1.550x | 2.103x |

At bs4 the capped arm needs *more* iterations at the AC-3 geometry and is still faster; at bs8
it needs exactly as many and is 1.17x faster. So iteration count cannot be the mechanism below
bs16, and even at bs16 it accounts for 1.55x of the measured 2.10x.

The term that actually pays is **graph width per iteration**. The cap drops the widest per-slot
chunk from `mbt` to `mbt/bs`, so instead of one request contributing a 16-token chunk while the
others idle, `bs` requests each contribute `16/bs` — the same tokens, a wider task graph. MPK's
iteration cost is set by the widest per-slot chunk rather than the token total
(`opt/m3i9/cost_model.py`), and the megakernel has 128 workers to fill. Both terms are
prefill-side, which is why M3-I9's decode-side reason for excluding bs<16 never settled the
question.

The consequence for the next lever: what the cap bought was PARALLELISM inside a fixed
16-token budget, and at bs16 that budget is now spread one token per request. There is no width
left to extract this way at any batch size — the remaining prefill lever is the budget itself
(`mbt`), which M3-I5b rejected on decode evidence that does not bind the prefill phase.

---

## 3. AC-5 — the position, measured, not tuned

`tables/m4i4_tables.json`, section `ac5`. vLLM numbers are read from the committed baseline
artifacts (`baselines/vllm-0.25.1-20260725/full/summary.json` for bs1/2/4, the two-boot merges
for bs8/bs16), never hardcoded in the analysis.

| bs | mpk e2e s (landed) | vLLM e2e s | AC-5 ratio | bound 1.25x | uncapped ratio |
|----|-------------------:|-----------:|-----------:|:-----------:|---------------:|
| 1  | 10.255 | 3.602 | **2.847** | FAIL | 2.847 |
| 2  | 10.653 | 3.886 | **2.741** | FAIL | 2.779 |
| 4  | 11.146 | 4.455 | **2.502** | FAIL | 2.602 |
| 8  | 12.641 | 4.953 | **2.552** | FAIL | 2.912 |
| 16 | 15.426 | 5.568 | **2.771** | FAIL | 5.082 |

AC-5 fails at every batch size, and the cap does not change that. Nothing here was tuned toward
the bound; the bound is a pinned goal criterion and this is where we stand against it.

**Why it fails is the useful part.** AC-5 is an end-to-end bound, so it is dominated by the
decode gap (AC-4: 2.3–2.8x), not by prefill. The right question is how much of AC-5's 25 % slack
prefill spends. Rearranging the bound: the decode window may be at most `1.25·V − P`, so at the
landed prefill cost `P` the decode throughput AC-4 must reach for AC-5 to hold is

| bs | P (mpk prefill s) | max decode window s | required decode tok/s | ÷ vLLM decode | prefill budget at decode parity | prefill must fall |
|----|------------------:|--------------------:|----------------------:|--------------:|-------------------------------:|------------------:|
| 1  | 0.311 | 4.191 | 243.6  | **0.853** | 0.926 s | — |
| 2  | 0.479 | 4.379 | 466.3  | **0.880** | 1.003 s | — |
| 4  | 0.822 | 4.746 | 860.5  | **0.921** | 1.197 s | — |
| 8  | 1.701 | 4.490 | 1819.0 | **1.075** | 1.366 s | 1.25x |
| 16 | 3.040 | 3.920 | 4167.8 | **1.381** | 1.547 s | 1.96x |

Read it as a joint constraint on (prefill, decode):

- **bs1, bs2, bs4: prefill is no longer binding on AC-5.** AC-5 holds at 85–92 % of vLLM's
  decode throughput, so AC-4 (strictly faster than vLLM) *implies* AC-5 at these sizes.
- **bs8 and bs16: prefill still taxes the decode margin.** AC-4 has to beat vLLM's decode by
  7.5 % (bs8) and 38.1 % (bs16) before AC-5 follows. Equivalently, at decode parity prefill
  would have to fall a further 1.25x / 1.96x.

What the cap changed, same arithmetic on the uncapped arm:

| bs | required ÷ vLLM decode, uncapped | capped | prefill must fall, uncapped → capped |
|----|--------------------------------:|-------:|-------------------------------------:|
| 1  | 0.853 | 0.853 | — → — |
| 2  | 0.908 | 0.880 | — → — |
| 4  | **0.998** | 0.921 | 0.99x → — |
| 8  | 1.483 | 1.075 | 2.15x → 1.25x |
| 16 | **unsatisfiable** | 1.381 | 5.33x → 1.96x |

Two of those deserve naming. At bs4 the uncapped arm sat at 0.998 — AC-5 required *exactly*
vLLM's decode throughput, so AC-4 and AC-5 were the same bar to within measurement error; the
cap opened 8 % of daylight. At bs16 the uncapped arm was **unsatisfiable at any decode speed**:
prefill alone (8.252 s) exceeded 1.25 x vLLM's entire end-to-end time (6.960 s), so no decode
kernel could have made AC-5 pass. The cap is what makes bs16 AC-5 reachable at all.

For scale: vLLM's own prefill, implied as its e2e minus its decode window at its measured
throughput, is 0.026/0.032/0.084/0.127/0.155 s. Ours after the cap is 12x/15x/10x/13x/20x that.

---

## 4. Correctness — the re-pinned AC-3 at all five batch sizes

Two independent passes. Both use the re-pinned criteria (`goal.md` AC-3, re-pinned 2026-07-29:
coherence + a 90 % top-1 agreement floor + no silent degradation, with bit-exactness as a
reported diagnostic rather than a pass condition). The scoring primitives come from M4-I1's
`accept/final/ac3_criteria.py` when it is in the tree — `scripts/ac3_repin_report.py` imports it
rather than keeping a second implementation.

### 4a. Warm reps from the perf campaign — 300 cases

Every geomA run is itself an AC-3 run (msl=132, 64 new tokens, the 10 pinned prompts), so the
correctness evidence and the wave-wall A/B come from the same runs. 2 arms x 3 reps x 5 bs x
10 prompts:

| criterion | result |
|---|---|
| (c) bit-exactness vs `results/dumps_final` | **300/300 byte-identical**, both arms |
| (b) agreement floor >= 90 % | **300/300 pass**; worst case 0.9375 (`p06-poem`, 60 of 64) |
| (a) degenerate repetition | 300/300 pass |
| (a) perplexity within 1.5x | transfers by identity — every case is byte-identical to the adjudicated baseline, so the continuation, and therefore its perplexity, is the same sequence |
| first divergences | exactly one class: `p06-poem` position 60, at every bs, in every rep, in BOTH arms, and in the committed baseline |

Classification of the one first divergence: **known / adjudicated, not a mechanism and not
unexplained.** It is the M2-adjudicated numeric tie — reference top-1 `31000` and top-2 `81316`
both at logit 21.0 (margin 0.0), engine argmax `40581` which is the reference's own top-3 at
20.875. It is present in `results/dumps_final` itself and in the uncapped arm, so it is not
something the cap did. The three positions after it are post-divergence and are not independent
evidence: greedy decode conditions on its own output, so the reference's top-1 there is
conditioned on a prefix this sequence does not have.

**No token changed anywhere.** There is nothing to classify as near-tie / mechanism /
unexplained, so nothing stops the landing.

Worth recording for the M4 gate: the adjudicated baseline's own worst-case agreement is 0.9375,
which leaves only 4 of 64 positions of slack under a 90 % floor on `p06-poem`. The floor is not
loose at that prompt.

### 4b. Cold-compile fingerprint gate at the shipped policy — STABLE

`harness/gate_ac3_stable.sh` with its new default (`--per-request-token-cap policy`), 3
fingerprint-consistent cold reps per bs at all five batch sizes. This is the pass that certifies
the LANDED configuration with a cold compile per rep — the class M3-I11 campaign 2 measured at
10–16 % state divergence — rather than warm kernel reuse. `gates/gate_ac3_stable.json`.

**Verdict STABLE at every batch size.** 15 reps launched, 15 scored, 15 accepted, 0 quarantined,
0 run errors, fingerprint divergence rate **0.0 %**, token divergence rate 0.0 %, 0 reps starting
on a non-clean device, all on GPU 6.

| bs | verdict | accepted | quarantined | consensus `state_sig` | M4-I0's pinned (uncapped) |
|----|---------|---------:|------------:|---|---|
| 1  | STABLE | 3/3 | 0 | `f66643d43adada64` | `f66643d43adada64` — **equal** |
| 2  | STABLE | 3/3 | 0 | `35629913b83a31ed` | `0449b5655fa57c9b` |
| 4  | STABLE | 3/3 | 0 | `f93138846d1d653d` | `68e7a9fb004338df` |
| 8  | STABLE | 3/3 | 0 | `7fba0ce871626e3c` | `1e305b6d61e9e263` |
| 16 | STABLE | 3/3 | 0 | `317dc00900bfbc49` | `c91b76a10b2430eb` |

### 4c. Why the state fingerprint moves while the tokens do not

The tokens are byte-identical and the state fingerprint is not, at exactly the four batch sizes
where the cap binds. That needed a mechanism, not a shrug — it is the "no silent degradation"
clause's whole job.

**It is HAZARD-COMPACTION slot migration, not numerics.** Evidence, in three steps.

1. **Which tensors moved.** `scripts/fp_decompose.py` compares two reps key by key
   (`w<N>_{k,v,conv,rec}` per wave boundary, `tok_<pid>` per prompt). Capped against uncapped on
   the same tree, same GPU, same day: at bs16 all four state tensors of the single wave boundary
   differ and all 10 token arrays are identical; at bs2, of the FIVE wave boundaries, **only
   `w1` differs** — `w0`, `w2`, `w3`, `w4` are bit-identical in k, v, conv and rec. Control: two
   capped bs16 reps are bit-identical in every fingerprinted tensor, so the detector is not
   noisy.
2. **Which wave that is.** The adapter's own admission replay records live-slot migrations per
   wave. At bs2, the uncapped arm migrates exactly one live slot and it does so in **wave 1**;
   the capped arm migrates none. The one wave boundary whose state differs is the one wave that
   had a migration. Waves with zero migrations in both arms are bit-identical.
3. **The counts, per arm.** Total live-slot migrations, and the `straddling_slots` subset the
   hazard can actually reach:

   | bs | migrations uncapped | capped | straddling uncapped | capped |
   |----|--------------------:|-------:|--------------------:|-------:|
   | 1  | 0  | 0 | 0 | 0 |
   | 2  | 1  | 0 | 0 | 0 |
   | 4  | 4  | 1 | 2 | 0 |
   | 8  | 32 | 3 | 6 | 0 |
   | 16 | 69 | 0 | 6 | 0 |

   bs1 is the only batch size whose migration count is unchanged, and it is the only one whose
   fingerprint equals M4-I0's uncapped pinned value — across a different day, clone and kernel
   build. That is both the positive control for the detector and the negative control for the
   cause.

So the differing bytes are the slot-indexed GDN conv/recurrent pool entries and the paged-KV
layout that a compaction shuffled, belonging to requests that had already retired. This is
consistent with M3-I9b's finding that chunk boundaries are bit-transparent in the current
kernels — the state at every wave boundary with no migration is bit-identical here too — and it
strengthens the case for the cap rather than weakening it: the capped arm has strictly fewer
migrations at every batch size and **zero straddling slots everywhere**, which is the exposure
M2-I9 raised and M3-I9 measured.

**Cross-issue hazard, for M4-I1.** M4-I0 recommended that `final.sh` pin the expected per-bs
fingerprint as an absolute assertion instead of voting within a window. Its pinned values were
captured **uncapped**. Pinning them now would fail a correct engine at bs2/4/8/16. The
signatures of the SHIPPED policy are the table in 4b, and they should be re-derived from a gate
run at `--per-request-token-cap policy` whenever the policy changes — the gate already records
`admission_policy` in its `run_meta.json` so a mismatch is diagnosable rather than mysterious.

---

## 5. WY/UT chunked prefill — terminal disposition

**REJECTED-WITH-EVIDENCE, second time, on a new basis. Not implemented, and not recommended
next.** M3-I6b rejected it because its stated precondition (more tokens per iteration) went the
other way. That reason is now stale — the cap has landed and prefill has been measured properly
— so the disposition is re-derived rather than inherited.

Three findings, in the order that decides it.

**(1) Prefill is still binding at bs8 and bs16, so the lever is not dead on relevance.** After
the cap, prefill is 13.5 % / 19.7 % of e2e at bs8/bs16 and it still costs AC-5 a 1.075x / 1.381x
decode requirement (section 3). A 1.96x prefill speedup at bs16 would make AC-5 follow from
decode parity. So "prefill no longer matters" would be false.

**(2) But WY/UT is not the shape of the remaining prefill cost.** The lever is a
*chunked-matmul* GDN prefill algorithm: it wins by doing the delta-rule recurrence as
block-parallel matmuls over a chunk of many tokens at once. The number of tokens in a chunk is
`mbt/bs` under the landed policy — **1 token per request at bs16, 2 at bs8**. The cap we just
landed *reduced* the per-request chunk to the minimum, and it did so because narrow chunks are
what makes MPK fast here (section 2d). A chunked-matmul algorithm at chunk size 1 is the
sequential recurrence with extra bookkeeping. Its precondition has not merely failed to arrive;
the winning direction is the opposite one, and we now have the measurement that says so rather
than an argument.

**(3) Its numerics gate does not inherit from M3-I9b, so it is not cheap either.** M3-I9b proved
chunk boundaries are bit-transparent in the CURRENT kernels — identical logit rows and
bit-identical per-layer GDN conv / fp32 recurrent state at bs1/bs4 cap-vs-base, H1–H4 refuted
with source citations. That result is about re-chopping the *same* per-token sequential
recurrence. WY/UT is a different algorithm: a block-parallel delta rule with a different
summation order and different intermediate precision. Bit-transparency cannot transfer across an
algorithm change, so WY/UT would need its own bit-exactness gate against the HF chunk algorithm
as oracle, plus its own AC-3 sweep. This campaign's own evidence shows what that costs: the
cap — a one-`min` change with a *proven* transparency argument — still needed 300 cases to
certify.

**What would re-open it.** Only a rise in tokens-per-request-per-iteration, i.e. `mbt` going up
(the budget, not its division). If a future issue raises `mbt` for the prefill phase — the
untaken lever M3-I5b rejected on decode evidence that does not bind prefill — then chunk sizes
grow and WY/UT becomes a real question again. It should be sequenced strictly after that, never
before, and it must carry its own oracle gate.

---

## 6. Backlog

`opt/backlog.json`: the prefill lever's `first_step` is discharged and its measured numbers are
recorded; the WY/UT entry carries the re-derived disposition above. The next prefill step, and
the only one this campaign leaves open, is `mbt` for the prefill phase.

---

## 7. Defects this landing introduced, and who caught them

Recorded because the fix belongs with the change, not with the finder.

1. `harness/gate_ac3_stable.py` kept a reference to a local `cap` variable that the policy
   refactor had removed, so every gate rep died with `NameError` while writing its metadata.
   Caught by **M4-I1's** first real gate run and fixed there; this campaign's own `ac3gate` phase
   hit it too and was re-run after the fix.
2. `mpk_engine_run.py`'s knob-presence probe read `mirage.PersistentKernel`, a package attribute
   that `opt/profile_wave.py` monkeypatches for the duration of the very call being probed, so
   `profile_wave.py` raised `NotImplementedError` at exactly the batch sizes where the policy
   resolves to a cap. Caught by **M4-I5**; fixed to interrogate the class.

Both are consequences of moving the decision out of the call sites, and both were only reachable
because the default now resolves to a cap where it previously resolved to `None`.

---

## Artifacts

- `tables/` — `geomA.csv`, `geomM.csv`, the full `m4i4_tables.json` (both geometries, AC-5, the
  AC-5-as-decode-requirement table, per-rep values, the compiled-cap integrity check) and the
  printed report `m4i4_tables.txt`
- `gates/` — the re-pinned AC-3 report over all six warm trees
  (`ac3_repin_geomA.{txt,json}`) and the cold fingerprint gate
  (`gate_ac3_stable.json`)
- `raw/ac3gate/`, `raw/ac3gate_nocap/` — the cold gate's per-rep records
  (`meta_*.json`, dumps, timings, its launch ledger and per-rep logs) and the uncapped
  bs2/bs16 control reps the fingerprint decomposition in 4c is built from. The `fp_*.npz`
  fingerprint arrays stay on the box under `~/mpk-qwen35/m4i4/{ac3gate,ac3gate_nocap}/reps/`;
  every signature they produce is already in the committed JSON
- `raw/geomA/`, `raw/geomM/` — every run's token dump and timings artifact (the timings carry
  the requested and compiled cap, the admission replay, and the compaction audit)
- `raw/audit/` — per-run device state before and after, and the per-phase CUDA-context device
  probe; `raw/run_index.txt` — every run with its arm, kernel dir, cold/reuse status and wall
- `raw/prompts/` — the 256-token synthetic prompt files, generated by the pinned baseline
  sampler and seed
- `scripts/` — setup, GPU guard, launcher, campaign driver, redeploy, and the two analyses
