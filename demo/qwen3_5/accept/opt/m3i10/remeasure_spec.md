# Re-measure spec — regenerating the MPK side of the M3-I10 correspondence

**EXECUTED 2026-07-27.** Tier 1 (arm A, matched geometry) + tier 2 (arm B, continuity) + a
late-context addendum (§4d) all ran, on an isolated `mirage-rm` clone + `venv-rm`, current HEAD
(`4e26acf`, gate ON). Two execution bugs were found and fixed before the numbers could be
trusted — **both change what is written below relative to what was originally spec'd**, so read
§3/§4 as corrected, not historical:

1. **§4(a)'s `--max-seq-length 1280` does not produce a 96-decode-step wave** — it produces
   ~1023. MPK's `MODE_OFFLINE` retirement (`step + step_advance + 1 >= max_seq_length`) makes
   `max_seq_length` gate decode *length* directly, unlike vLLM where `max_model_len` is a
   capacity ceiling independent of `--output-len`. `1280` was vLLM's 256+1024 capacity number,
   carried over as if MPK's same-named parameter meant the same thing. Corrected value:
   `max_seq_length = prompt_len + decode_steps + 1 = 256 + 96 + 1 = 353`. Full derivation,
   the three-way internal-consistency check that caught it (this doc's own §3 iteration counts,
   §4(a)'s context claim, and §8's wave-time budget all only hold at 96 real decode steps), and
   empirical confirmation:
   `opt/m3i10/remeasure/logs/ROOT_CAUSE_msl.txt`.
2. **`schedule_sim.py`'s `steady_window()` can select the prefill regime instead of decode** —
   its tie-break ranks "most tokens/step" above "is this decode", which inverts whenever
   `batch_size <= mbt` with a long uniform prompt (every arm-A run at bs 1 and 8): prefill's
   mbt=16-bounded tokens/step beats decode's bs-bounded tokens/step, so the "steady window"
   silently landed entirely inside the 16-iteration prefill phase at bs1. Fixed with a corrected
   copy, **not a git-tree edit** — `opt/m3i10/remeasure/opt_fixed/schedule_sim.py` (one tie-break
   term prepended: prefer any prefill-free regime outright before falling through to the original
   ranking). Verified as a strict improvement: no regression on the AC-3 geometry it was tuned on
   (arm B's full-vs-steady crosscheck tightens to −1.1 %), and it correctly recovers the
   decode-only window at arm A bs 1/8/16. **This is a latent bug in the committed
   `parse_profile.py`/`analyze.py` pipeline itself** (item 6.5.1 — lands via a separate follow-up
   issue, tracked informally as "I7" in the remeasure's own memory note; not yet folded into the
   git-tree `schedule_sim.py`). Any future capture with `batch_size <= mbt` should use the fixed
   copy, not the committed one, until that lands.

Results, regenerated tables, the `profile_wave.py` patch, and every analysis script used are all
under `opt/m3i10/remeasure/` (raw profiler buffers >20 MB pointed to
`/home/catalyst/mpk-artifacts/m3i10-remeasure/`, also retained on the capture box). The
regenerated `ferret_targets.json` and this file are the two git-tracked artifacts this closure
updates in place.

## 1. Why the MPK column is stale

The vLLM column of `tables/comparison_by_stage.csv` was measured in this issue, at the pinned
256/1024 workload, on the binding engine identity. The MPK column is **M3-I1's capture**, and it
is stale in two independent ways:

| | M3-I1 capture (what the table uses) | HEAD today |
|---|---|---|
| geometry | AC-3: `max_seq_length=132`, 24–68-token prompts, 64 new tokens | the pinned workload is 256 in / 1024 out |
| MoE router gating | `MOE_GATE_PADDING_ROWS` did not exist | `builder.py:148 MOE_GATE_PADDING_ROWS = True` — **default-ON** since `96eff01` |
| quantize / task width | pre-M3-I2b | M3-I2b landed the quantize redundancy fix (93.75 % of that stage's work was redundant) and the narrow-stage widening |

So two of the top four ferret ranks — quantize (rank 2) and MoE w13/w2 (ranks 3, 5) — are ranked
against an MPK baseline that the tree has already moved past. M3-I8's own A/B says the gate cuts
`moe_w13` per-layer wall span from 76.8 µs to 34.8 µs at bs1 (`m3i8/results/analyze_final2.log`),
which is roughly a 2× move on a stage currently ranked 3rd.

## 2. What has to come out the other end

Regenerated, at the pinned geometry on current HEAD:

- `opt/pertask_by_bs.csv` — per task type: count, Σ worker µs, **wall span**, concurrency, per bs.
  Wall span is the column the comparison joins on.
- `opt/attribution.csv`, `opt/layer_type_by_bs.csv`, `opt/tables/bs<N>_{attrib,concurrency}.json`.
- **New, required by ferret rank 9:** a per-**call-site** split of task 253 (`in_proj_ba` ×30 /
  MoE router gate ×40 / `lm_head` ×1) and of task 279 (its 6 GEMM shapes). The rank-9 target is
  currently stated on the family sum only, and `lm_head` inside it is already at 84 % of the HBM
  roof — without the split a ferret run could aim at the part with no headroom. See §5.

## 3. The run matrix

Three tiers. Tier 1 is what the finding actually requires; 2 and 3 are worth their cost if the
window is already open.

| tier | arm | geometry | bs | reps | processes | why |
|---|---|---|---|---|---|---|
| **1 (required)** | A — matched | `msl=353` (corrected; see status block above — **not** 1280), 256-token prompts, 96 decode steps | 1, 8, 16 | 3 profiled + 3 unprofiled | 18 | makes the MPK column like-for-like with the vLLM column |
| 2 (recommended) | B — continuity | `msl=132`, AC-3 prompts, 64 new tokens (M3-I1's exact invocation) | 1, 8, 16 | 3 + 3 | 18 | **separates the code delta from the geometry delta.** Without arm B a moved number cannot be attributed to the I8 gate / I2b vs to the longer context |
| 3 (optional) | A + B | as above | 2, 4 | 3 + 3 | 24 | restores M3-I1's 5-point table so the two are directly diffable — **not run** (box volatility already spent the discretionary margin on two GPU-residency wedges; see ROOT_CAUSE_msl.txt and the remeasure's memory note for the wedge diagnosis) |
| 4 (closure, added post-c2) | A-late — recentred context | `msl=897` (256-token prompt + 640 decode steps + 1), final-96-iteration window (context ≈801–896) | 1, 8, 16 | 1 profiled only | 3 | closes the context-window mismatch between arm A's own decode window (context 257–352) and the vLLM reference table's (556–896) — see §4(d) |

Arm B is the one people skip and then regret: arm A alone changes two variables at once.

## 4. Capture invocation

Reuse `m3i8/run_m3i8.sh` verbatim as the driver pattern — it already wraps M3-I1's
`profile_wave.py` / `parse_profile.py` / `concurrency.py` instrument, enforces one wave per
process (HAZARD-WAVE-RESET), does the `/raid` headroom refusal, reuses compiled kernels across
reps via `--reuse-kernel`, and records an arm sha256 manifest. Two changes:

**(a) Geometry flags.** `profile_wave.py` already exposes everything needed — no new plumbing:

```bash
# arm A, matched geometry (msl CORRECTED from the original 1280 -- see status
# block at the top of this file and ROOT_CAUSE_msl.txt. General form:
# max_seq_length = synthetic_prompt_len + desired_decode_steps + 1, NOT
# vLLM's prompt+output capacity sum -- MPK's max_seq_length gates decode
# LENGTH directly, it is not a capacity ceiling like vLLM's max_model_len.)
$PY -u "$OPT/profile_wave.py" --batch-size $BS \
    --max-seq-length 353 --max-new-tokens 96 --mbt 16 --page-size 256 \
    --synthetic-prompt-len 256 --synthetic-seed $((20260725 + BS*1000 + REP)) \
    --out-dir "$M/prof_A" --kernel-dir "$M/kernel_A_bs${BS}_prof" \
    --rep $REP --slots 96000000 --reuse-kernel [--save-raw]
```

**(b) One small code change — synthetic prompts.** `profile_wave.py` takes `--prompt-ids` from
the AC-3 reference set, whose prompts are 24–68 tokens; arm A needs 256. Add
`--synthetic-prompt-len` / `--synthetic-seed` implementing *exactly* `bench_vllm.py`'s
`build_synthetic_prompts` (`rng = random.Random(seed)`, `ids = [rng.randrange(0,
tokenizer.vocab_size) for _ in range(n)]`). Use the **same seed formula the vLLM side used**,
`20260725 + bs*1000 + rep`, so both engines consume literally the same token ids. Arm B keeps
`--prompt-ids` and the AC-3 set, because that is what the byte-identity check needs.

**(c) Profiler buffer.** `--slots 96000000`, not M3-I1's 48 M. Arm A's wave is prefill + decode:
at bs16 that is `4096/16 = 256` prefill iterations + 96 decode = 352 iterations, and M3-I1
measured ~121.7 k events/iteration at bs16 (24.7 M events / 203 iterations) ⇒ **~42.8 M events**,
which is 89 % of a 48 M buffer. M3-I1's validity rule is *zero dangling profiler events*, so run
with headroom. 96 M slots ≈ 1.15 GB of device buffer at M3-I1's ~12 bytes/event — trivial against
183 GB.

Do **not** try to capture a full 1024-token decode: that is ~16× the events (150–400 M) for no
extra information, since the comparison only needs a steady-decode slice. ~~96 decode steps at
`msl=1280` is the MPK analogue~~ **corrected**: 96 decode steps at `msl=353` (see status block —
`msl=1280` does not yield 96 decode steps at all, it yields ~1023) is the MPK analogue of the
50-step kineto windows used on the vLLM side.

**Context caveat that survives this design.** Arm A's profiled decode window sits at context
257–352, while the vLLM windows sat at 556–896. Only attention is context-sensitive and this issue
already bounded it: the same FMHA kernel costs 8.706 µs/call at ctx ≈ 260 vs 9.425 µs/call at ctx
556–896, **+8.3 %** (`tables/prefill_bs1_kernels.csv` vs `tables/bs1_kernels.csv`). Apply that as
a correction to the attention row rather than paying 16× the events to avoid it.

**UPDATE (post-c2, §4(d) below): the +8.3 % single-kernel correction understated the effect.**
Arm A's own matched-geometry attention ratio came in at 5.3–6.0× (up from the old AC-3-geometry
capture's 3.1–3.8×) — a ~50 % increase, not the ~8 % the single-FMHA-call correction implied. A
dedicated late-context capture (§4(d)) was run to measure this directly rather than keep
extrapolating from one kernel call, and the corrected attention row uses that capture's numbers,
not the +8.3 % arithmetic correction.

### 4(d). Closure capture — recentring the decode window into the vLLM reference's context band

Codex c2 flagged that arm A's own decode window (context 257–352) still does not match the
vLLM reference table's sampled context (556–896) — exactly the stage the remeasure's own
attention finding is about, so the mismatch could itself be inflating (or masking) that finding.
Closure, cheapest correct form (MPK's profiler buffer is instrumented in-kernel with no windowed
schedule like vLLM's torch.profiler `skip_first`/`wait`/`warmup`/`active` — it is all-or-nothing
per wave, so "profile a later slice only" is not available; the only lever is how long the wave
runs):

```bash
# arm A-late: same 256-token synthetic prompt, same seed formula, msl sized so
# the wave's FINAL 96 decode iterations land in context ~[801,896] (inside
# the vLLM reference's 556-896 band) -- prompt_len + D + 1, D=640:
$PY -u "$OPT/profile_wave.py" --batch-size $BS \
    --max-seq-length 897 --max-new-tokens 96 --mbt 16 --page-size 256 \
    --synthetic-prompt-len 256 --synthetic-seed $((20260725 + BS*1000 + 0)) \
    --out-dir "$M/prof_Alate" --kernel-dir "$M/kernel_Alate_bs${BS}_prof" \
    --rep 0 --slots 200000000 --save-raw
```

Analysis takes the **final-96-iteration** window instead of arm A's post-warmup head window:
`parse_profile.py ... --warm-iters 544 --steady-iters 96` (`schedule_sim.steady_window` finds the
full ~640-iteration decode_full run; `warm_iters=544` skips to iteration 544 of it, leaving
exactly the last 96 — the fixed `opt_fixed/schedule_sim.py` from §5 item 3 is required here too,
for the same reason). Profiled only, 1 rep/bs (not the full 3+3): this is a targeted closure
measurement of one stage's context sensitivity plus a spot-check that GDN/GEMM stages stay
context-flat, not the primary statistical deliverable — arm A's own 3-rep captures already showed
sub-0.2 % wave-to-wave spread and event counts are provably seed-independent (same event count
across reps with different synthetic tokens, arm A tier-1 evidence), so 1 rep suffices for this
purpose. Cost: 3 processes, ≈5 GPU-minutes total (all 3 bs compiled + ran in well under 5 minutes;
`--slots 200000000` for headroom against bs16's ~123 M events, up from arm A's ~53 M since the
wave now covers 640 decode + prefill iterations instead of 96).

Verdict: attention's ratio increase is confirmed as a **real, mostly context-driven effect, larger
than the old single-FMHA-kernel +8.3 % correction implied.** Attention wallspan (µs/step), matched
geometry (ctx 257–352) → late-context (ctx ≈801–896 at bs1/bs8; see caveat below for bs16):

| bs | matched-geometry (§4) | late-context (§4d) | further growth |
|---|---:|---:|---:|
| 1 | 757.0 | 1080.5 | **+42.7 %** |
| 8 | 804.0 | 1221.2 | **+51.9 %** |
| 16 | 787.0 | 1509.3 | **+91.8 %** (caveat below) |

bs1/bs8 are clean measurements — bs1 is a single request, bs8 is a full 8-wide `decode_full`
window (iterations 560–656, all 8 concurrent, uniform context). **bs16 has no clean equivalent**:
`schedule_sim.simulate` shows its 12 surviving requests at the chosen window spread across
per-slot context **263–890**, not a tight high-context band — bs16's staggered admission means
its concurrent slots are never all at similar context simultaneously, at any point in the wave
(structurally the same reason `schedule_sim.py`'s docstring already documented "no `decode_full`
phase exists at bs16" — the fixed tie-break in item 2 above changes which regime a bs16 query
lands on, not the underlying absence of one). Treat bs16's +91.8 % as directionally consistent,
not independently precise.

GDN recurrent and dense-fp8 (the two "robust" stages from §9) were spot-checked at the same
late-context capture and stay flat, confirming they generalise to high context too:

| task | bs | matched-geometry | late-context | Δ |
|---|---|---:|---:|---:|
| GDN recurrent (237) | 1/8/16 | 1218.0 / 2467.6 / 4938.0 | 1221.0 / 2467.6 / 4937.3 | +0.2% / +0.02% / −0.01% |
| dense fp8 (279) | 1/8/16 | 2933.2 / 2978.4 / 2990.3 | 2939.2 / 2978.4 / 2990.3 | +0.2% / −0.6% / −0.5% |

MoE w13 (241) is flat at bs1/bs8 (−0.3 % / +3.9 %) but +18.8 % at bs16 — read with the same
reduced-concurrency caveat as attention's bs16 point, not yet attributed cleanly to context vs.
concurrency. Full numbers, regime detail, and the exact per-slot context arrays:
`ferret_targets.json`'s `full attention` row — **late-context is that row's PRIMARY basis as of
the c3 closure** (`mpk_us_per_step` / `ratio_mpk_over_vllm` / `step_gain_if_met_us` are the
801–896-context numbers directly, with `context_band` documenting the band + the bs16 caveat and
`matched_window` preserving the demoted ctx-257–352 numbers this table's own left column shows)
— plus top-level `late_context_addendum` for the GDN/dense-fp8/MoE-w13 spot check; regenerated
tables in `opt/m3i10/remeasure/armAlate/`.

## 5. Normalisation — transplanting the anchor method

The vLLM side could not trust kineto's `ProfilerStep#` markers, because with CUDA graphs the GPU
timeline lags the CPU by ~1 step (~9 % error at a 10-step window). **MPK does not have that
problem** — its profiler buffer is instrumented in-kernel and carries explicit iteration markers.
So the transplant is not the windowing; it is the **QC**:

1. **Anchor = `TASK_BEGIN_TASK_GRAPH` (task type 10)**, which fires exactly once per step
   (`n = 1.0` at every batch size in the current `pertask_by_bs.csv`). Integrate over
   `[first ts, last ts)` ⇒ exactly `count − 1` complete steps.
2. **Assert every task type's per-step count is an integer**, and that it equals the compiled task
   graph's static call-site count. `taskgraph_moe.py` and `taskgraph_quantize.py` already dump
   those counts with no GPU. This is the check M3-I1's parse never made, and it is what caught a
   9 % normalisation error on the vLLM side. Report `max |count/step − round(count/step)|`;
   anything above ~0.02 invalidates the window.
3. Cross-check the anchor window against `parse_profile.py`'s existing `--warm-iters` /
   `--steady-iters` segmentation. For arm A use `--warm-iters 8 --steady-iters 80` (96 decode
   steps, drop the first 8 and the last 8). They should agree to well under 1 %; if they do not,
   the anchor is right and the fixed-window offsets are wrong.
4. Keep emitting **both** conventions per task family — wall span (union) *and* Σ worker time —
   because the comparison joins on wall span but the roofline arithmetic needs total work.

## 6. Analysis — regenerating the tables

All CPU, no GPU:

```bash
# on the box, after capture
for BS in 1 8 16; do
  python opt/parse_profile.py --raw prof_A/raw_bs${BS}_rep0.npz \
      --meta prof_A/meta_bs${BS}_rep0.json --names prof_A/task_names.json \
      --out-prefix tables/bs${BS} --warm-iters 8 --steady-iters 80
  python opt/concurrency.py prof_A/raw_bs${BS}_rep0.npz prof_A/meta_bs${BS}_rep0.json \
      prof_A/task_names.json tables/bs${BS}_concurrency.json
done

# in the repo
python3 opt/analyze.py                              # -> pertask_by_bs.csv, attribution.csv, ...
python3 opt/m3i10/scripts/build_comparison.py       # -> m3i10/tables/comparison_by_stage.csv
python3 opt/m3i10/scripts/extend_ferret.py          # -> m3i10/ferret_targets.json (idempotent)
python3 opt/m3i10/scripts/roofline.py               # -> m3i10/ncu/roofline.csv
```

`build_comparison.py` reads MPK's `wallspan_us_bs*` straight out of `pertask_by_bs.csv`, so
pointing it at the regenerated file is the whole join. `extend_ferret.py` trims and re-appends its
own rows, so it is safe to re-run. Then update `comparison.md` §3 and §8, and re-run the
`tables/`-vs-doc cross-check.

**Also update `MPK_STEP` in `build_comparison.py`** (currently the M3-I1 constants
15264/18618/22005 µs) from the regenerated `attribution.csv` — it is the denominator of the
overall ratio and the only hardcoded MPK number in the pipeline.

## 7. Harvesting the running M3-I9 window instead

**Moot for this execution** (tier 1 was captured directly, not harvested — see status block at
the top), but the criteria below are corrected in place since a future issue may face the same
harvest-or-capture choice and would otherwise inherit the same `msl=1280` mistake this file
originally had.

An M3-I9 resumption window is running stage-7 matched-geometry runs right now. If its output
already satisfies the list below, **tier 1 needs no new GPU time** — check before scheduling.

Reusable if and only if all of these hold:

1. **Raw profiler buffers retained**, not just derived tables: `raw_bs<N>_rep<R>.npz` +
   `meta_bs<N>_rep<R>.json` + `task_names.json`. Perfetto exports alone are not enough — the
   3-iteration Perfetto window is too short to re-derive per-task wall spans.
2. **Geometry is genuinely matched**: ~~`max_seq_length` ≥ 1280~~ **corrected**: `max_seq_length
   == 353` (256-token prompt + 96 decode steps + 1 — see status block; `max_seq_length` gates
   MPK decode length directly, so `>= 1280` would actually mean the SAME bug this file's status
   block documents, not "matched or better") and prompt length 256 recorded in the meta. A run at
   `msl=132` is arm B, not arm A, and does not close the finding.
3. **Gate state recorded**: `MOE_GATE_PADDING_ROWS` value and the commit sha in the run manifest
   (M3-I8's `arm_<arm>_sha256.txt` pattern). A run that does not record it cannot be attributed.
4. **Both profiled and unprofiled reps of the same config.** Profiled-only is not usable: M3-I1's
   validity rule is that profiled vs unprofiled wall time differ by <5 % (it measured 2.85–3.59 %),
   and without the unprofiled arm there is no way to show the instrument did not change the answer.
5. **≥3 reps per (bs, mode)** with event counts bit-identical across reps, and **zero dangling
   profiler events** (the truncation check).
6. **bs coverage ⊇ {1, 8, 16}.** bs1 and bs16 alone would let ranks be recomputed but leaves the
   shared-expert-gate bs8 anomaly (see `ferret_targets.json` → `dispositions`) unresolved.
7. **Exclusive GPU** for every timed rep, with the lock and the 3-sample idle guard recorded.
8. **AC-3 non-regression on the same commit** — token sequences byte-identical to the committed
   reference. A perf table from a run that changed outputs is void.

If items 1–3 hold but 4–6 do not, the window is still worth harvesting as a **single-rep sighting
shot** to see whether the quantize and MoE ranks move at all — that alone would tell the
coordinator whether tier 1 is urgent or merely due.

## 8. Cost

| tier | processes | compiles | wave time | GPU-minutes |
|---|---:|---:|---:|---:|
| 1 — arm A, bs {1,8,16}, 3+3 | 18 | 6 | ~1.1 min | **~75–90** |
| 2 — arm B, bs {1,8,16}, 3+3 | 18 | 6 | ~0.8 min | +~55 |
| 3 — bs {2,4}, both arms | 24 | 8 | ~1.0 min | +~70 |
| all three | 60 | 20 | ~3 min | **~3.5 h** |

Where tier 1's ~90 minutes goes — the wave time is *not* the cost:

- **Kernel compiles ≈ 30 min.** 6 kernel dirs (3 bs × {prof, noprof}); MPK JIT is 1–10+ min each
  (`b200-env.md`). `--reuse-kernel` makes this once-per-dir, not once-per-rep.
- **Per-process fixed cost ≈ 24 min.** One wave per process (HAZARD-WAVE-RESET) ⇒ 18 model loads;
  ~80 s each (torch import + 37 GB checkpoint off /raid).
- **Actual waves ≈ 1.1 min.** bs1 112 iterations, bs8 224, bs16 352, at ~11/13/19 ms per step
  using M3-I8 v1 (gate-ON) step times plus a few percent for the longer context.
- **AC-3 spot check ≈ 6 min**, 3 processes at the AC-3 geometry.
- Remainder is GPU-guard sampling, the `/raid` headroom check, and retry slack.

Disk: raw buffers scale with events — M3-I1's were 111–298 MB each at 9.3–24.7 M events, so arm A
at up to ~43 M events is ~500 MB per saved rep. Only `--save-raw` on rep 0 (M3-I8's pattern) ⇒
~1.5 GB for tier 1. `/raid` had 184 GB free at the end of this issue; the 11 GB
`m3i10-profile/` tree (of which ~9 GB is the SGLang venv and 1.1 GB the chrome traces) can be
reclaimed if needed.

## 9. What the re-measure will and will not change

Stated in advance so the outcome is falsifiable rather than rationalised afterwards.

**Robust — should survive:**

- **GDN recurrent growth ratio.** 7.44× → 9.10× → 10.76× across bs 1/8/16. Neither the I8 gate
  (MoE router) nor I2b (quantize/width) touches task 237, and the vLLM reference is corroborated
  by SGLang running the identical kernel at an identical 164.9 vs 163.6 µs/step. The *absolute*
  ratio may shift a little with the step-time denominator; the **growth with batch** should not.
- **Dense fp8 flat ratio.** ~2.17× / 2.05× / 2.17×, flat across batch, over 160 call sites that
  nothing in I8 or I2b touches. Task 279's wall span is already flat (2936/2947/2973 µs).
- **lm_head at 84 % of the HBM roof** — arithmetic from shapes and a vLLM-side measurement; no MPK
  dependence at all.
- **norms/RoPE/glue as an MPK win** (0.23–0.29×) — a structural fusion advantage, not a tuning
  artifact.

**May reshuffle:**

- **Quantize rank (currently 2).** M3-I2b targeted exactly this stage and found 93.75 % of the
  work redundant. If that landed as measured, task 275's 4540 µs/step at bs1 could fall by most of
  its value and drop the stage several ranks. The *mechanism* conclusion — that the stage moves
  ~1 MB and is 4284× off roofline, so it is a fusion/width problem and not a kernel problem —
  survives regardless, because it is a statement about vLLM's side and about physics.
- **MoE w13 / w2 ranks (currently 3 and 5).** The I8 gate is default-ON at HEAD and its own A/B
  shows moe_w13 per-layer wall span 76.8 → 34.8 µs at bs1. Expect rank 3 to fall at bs1–bs4 and to
  move much less at bs16 (M3-I8: the gate saves no wave at bs16).
- **The shared-expert-gate bs8 point** (1.34×, MPK 596 µs against 428/432 µs at bs1/bs16 with flat
  per-task time). Flagged as a probable M3-I1 capture artifact; the re-measure either confirms it
  or removes it.
- **Every absolute `step_gain_if_met_us`**, and therefore the overall 4.28× / 3.86× / 4.10× step
  ratio, since the MPK step denominator itself moves.

**Ordering claim that should be checked explicitly:** whether GDN recurrent (rank 1) overtakes
quantize and MoE w13 in *absolute* recoverable microseconds at bs1 once I2b and I8 are in. It is
already rank 1 at bs16 by a wide margin; if it becomes rank 1 at every batch size, the ferret
dispatch order simplifies to "GDN recurrent first, unconditionally".
