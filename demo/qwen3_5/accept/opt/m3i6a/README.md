# M3-I6a — full attention: `max_tokens_per_pass` 4 → 2

**Disposition: INTEGRATED.** One value changes (`Qwen35Builder.attn_q_pass`, overridable with
`MPK_ATTN_Q_PASS`). No kernel edit — `attention_sm100.cuh` is byte-identical to HEAD. Tokens are
bit-identical everywhere they were checked: AC-3 50/50 byte-diff vs `results/dumps_final`, 116/116
q-loop invariance rows at `max_abs = 0`, and the HF oracle returns identical verdicts on all 35
comparison rows at pass size 4, 2 and 1.

At the pinned 256-in/1024-out workload, decode-step time falls **9.7 % at bs1 and 10.3 % at bs8**
(profiled, decode context ≈848), and unprofiled end-to-end wall over a 640-step decode falls
**5.8 / 5.5 / 4.6 %** at bs1 / bs8 / bs16.

---

## 1. What was measured, and why the previous ranking was wrong

M3-I1 put this lever at rank 7 and recommended deferring it, on the grounds that
`TASK_ATTN_SM100`'s whole wall span was 513 µs/step at bs1 — 3.4 % of the step. That number came
from the **AC-3 geometry**, where decode context never exceeds 132 tokens. It also, correctly,
said the sweep was only worth a builder call if the stage exceeded ~10 % at the AC-4 256/1024
geometry. It does: at decode context ≈848 the stage is **1447 µs/step at bs1, 13.1 % of the step**,
and it is the only stage whose cost grows with context.

Measurement used one deep-context wave per (batch size, pass size) — `msl=897`
(256-token synthetic prompt + 640 decode steps), the same geometry and seed formula as M3-I10's
late-context closure (`opt/m3i10/remeasure/scripts/run_armA_latectx.sh`). A single such wave walks
context 257 → 896, and because `TASK_BEGIN_TASK_GRAPH` fires exactly once per step,
`scripts/ctx_curve.py` recovers the whole trajectory from that one capture by binning task
instances into sliding 96-iteration windows. Anchor QC is mandatory and ran over the full span of
every capture: `max_frac_err = 0.0000` and anchor count `= 1.0` per step in all six
(`tables/ctx_*.json` → `anchor_qc`). Windows containing prefill slot-iterations, and windows past
the first slot retirement, are excluded from fits — both ends are contaminated otherwise.

Sanity check against the committed basis: at context 336 this pipeline reports 807.5 µs/step at
bs1, and the fit below gives 764 µs at M3-I10 arm A's own context (~304), against arm A's committed
757 µs — 0.9 %. Same convention, same numbers.

## 2. The binding constraint, with numbers

Per-task latency is linear in decode context, tightly (`R² ≥ 0.998` over 10–12 windows):

| bs | pass size | fixed µs | µs per KV token | µs per 64-token KV tile | effective GB/s per task | % of 8 TB/s HBM roof |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4 | 38.77 | 0.12491 | 7.994 | 8.20 | 0.102 |
| 1 | 2 | 29.63 | 0.05356 | 3.428 | 19.12 | 0.239 |
| 1 | 1 | 29.05 | 0.05272 | 3.374 | 19.42 | 0.243 |
| 8 | 4 | 32.36 | 0.13312 | 8.520 | 7.69 | 0.096 |
| 8 | 2 | 26.91 | 0.05191 | 3.323 | 19.72 | 0.247 |
| 8 | 1 | 27.14 | 0.05094 | 3.260 | 20.10 | 0.251 |

**It is not the KV read.** The stage moves 64 × 256 × 2 bytes × 2 (K and V) = 64 KiB per KV tile and
takes 8 µs to do it — 8.2 GB/s, **0.10 % of the HBM roof**. Nothing about this is bandwidth-bound at
either setting.

**It is per-KV-tile instruction work in a 2-wide task family.** At bs1 the stage is exactly 20 task
instances per step = 2 per layer (one per KV head, `grid_dim=(mbr, num_key_value_heads, 1)`), and
wall span per layer equals one task's mean duration — so the stage's cost *is* one worker's serial
walk over the KV history, and only 2 of 128 workers are in it.

**And at pass size 4 that per-tile work was inflated by register spilling across the whole
megakernel.** MPK inlines every task body into one `persistent_kernel`
(`__launch_bounds__(WORKER_NUM_THREADS, 1)`), so ptxas allocates one register budget and one
per-thread stack frame for all of them. Recompiling the *generated* TU with `-Xptxas -v`
(`scripts/mk_ptxas.sh`, `ptxas/megakernel_qp{4,2,1}.txt`):

| pass size | `persistent_kernel` registers | stack frame | spill stores | spill loads |
|---:|---:|---:|---:|---:|
| 4 (shipped) | 255 | 576 B | 780 B | 976 B |
| 2 | 240 | 144 B | **0** | **0** |
| 1 | 238 | 144 B | **0** | **0** |

The attention accumulator was the *only* source of spilling in the megakernel. `task_register.cc`
sets `MAX_TOKENS` from `max_tokens_per_pass`, the kernel derives
`MMA_ITERS_M = ceil(MAX_TOKENS·NUM_QO_PER_KV/16) = ceil(Q_PASS/2)` at 16 q / 2 kv heads, and the
per-thread accumulator is `float o[MMA_ITERS_M][HEAD_DIM/16][8]` = `MMA_ITERS_M · 128` floats — 256
floats (1 KiB) at pass size 4, above the 255-register file. The KV loop's rescale step touches every
`o` element once per tile, so the spill was paid **per KV tile**, which is exactly why the cost grew
with context. `scripts/tu_i6a_attn.cu` + `scripts/probe_regs.sh` confirm the same thresholds in
isolation (240 regs / 0 spill at pass ≤ 2; 255 regs / 820-800 B spill at 4; pass ≥ 6 fails the smem
`static_assert`, reproducing probe P3's ceiling).

The mechanism predicts a side effect that the waves then show: relieving a *shared* budget must
speed up families this knob does not touch. It does — `tables/ctx_sweep.csv`, bs1, steady windows:
dense fp8 2939 → 2811 µs/step (−4.4 %), GDN recurrent 234 → 219 µs/step (−6.2 %). That is why the
step improves by more than attention's own saving.

## 3. Sweep result and pick

Attention wall span, µs/step (`tables/ctx_sweep.csv`):

| bs | ctx | pass 4 | pass 2 | pass 1 | 2/4 | 1/4 | step 4 | step 2 | step 2/4 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 336 | 807.5 | 478.2 | 469.9 | 0.592 | 0.582 | 10379.7 | 9682.7 | 0.933 |
| 1 | 528 | 1047.3 | 581.5 | 571.3 | 0.555 | 0.545 | 10627.2 | 9790.6 | 0.921 |
| 1 | 848 | 1447.2 | 752.2 | 739.4 | 0.520 | 0.511 | 11027.4 | 9960.7 | 0.903 |
| 8 | 407 | 959.8 | 518.8 | 517.6 | 0.540 | 0.539 | 11572.0 | 10655.3 | 0.921 |
| 8 | 789 | 1491.0 | 723.9 | 718.8 | 0.485 | 0.482 | 12099.6 | 10856.7 | 0.897 |

Pass sizes ≥ 6 are inadmissible (the smem `static_assert`), so the admissible set is {1, 2, 4} and
the choice between 1 and 2 is settled by §3a.

Read on its own this profiler view is **ambiguous** between 1 and 2: they have the same register
profile (238 vs 240, both zero spill), pass 1 is ~1.7 % cheaper per decode step, and in
prefill-containing windows it is 20–24 % worse (bs1 window 0–96, 16 prefill slot-iterations: 945.1 µs
at pass 2 against 1174.0 at pass 1; bs8 window 141–237: 886.0 against 1059.2). Two batch sizes of
profiler windows cannot weigh those against each other, which is why the selection rests on the
end-to-end sweep below and not on this table.

## 3a. The complete three-way sweep — how 2 vs 1 is settled

`tables/sweep3_table.txt`, `tables/sweep3_medians.csv`; driver `scripts/phase6.sh`, analyser
`scripts/sweep3.py`. **135 reps — 3 per (geometry, bs, arm), 0 discarded.** All three arms run
back-to-back per (bs, rep) inside one GPU claim at integrated HEAD (`f3606f2c`), so drift or a
co-tenant hits all three equally; `MPK_ATTN_Q_PASS` makes every arm available from one tree, so
nothing changes between arms but that one value. Every rep is drain-gated (pinned device below
500 MiB before it starts) and audited from **its own record** — `meta.cuda_visible_devices` +
`gpu_before` for the `profile_wave` geometries, a per-rep sidecar for the AC-3 geometry — never from
the candidate list the guard was handed, which is not evidence of what actually ran.

The two perf geometries deliberately **bracket** the tradeoff rather than sampling one side of it: B
is the prefill-heavy end (256·bs/16 prefill iterations against 96 decode steps, where pass 1's
doubled pass count should hurt most) and C is the decode-heavy end (16 prefill iterations per request
against 640 decode steps, where pass 1's lower per-KV-token cost should help most).

Ratios are median/median; **`1 vs 2` above 1.000 means pass 1 is slower**. `sp%` is rep spread as a
percentage of the median.

**Geometry A — AC-3 (10 pinned reference prompts, msl=132), Σ wave wall_ms**

| bs | pass 4 | sp% | pass 2 | sp% | pass 1 | sp% | 2 vs 4 | 1 vs 4 | **1 vs 2** |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 9481.3 | 1.58 | 9308.7 | 2.84 | 9266.9 | 0.55 | 0.9818 | 0.9774 | 0.9955 |
| 2 | 5139.4 | 4.52 | 4974.1 | 0.39 | 4983.8 | 0.09 | 0.9678 | 0.9697 | 1.0020 |
| 4 | 3366.7 | 3.04 | 3259.5 | 2.84 | 3305.2 | 0.64 | 0.9681 | 0.9817 | 1.0140 |
| 8 | 2984.8 | 0.05 | 2917.1 | 0.61 | 2954.3 | 0.48 | 0.9773 | 0.9898 | 1.0128 |
| 16 | 3383.1 | 1.25 | 3313.1 | 1.38 | 3327.3 | 1.97 | 0.9793 | 0.9835 | 1.0043 |
| **all** | 24355.3 | | 23772.4 | | 23837.4 | | 0.9761 | 0.9787 | **1.0027** |

Per-wave median ms/decode-step on the same runs puts pass 1 behind pass 2 at *every* batch size,
bs1 included: 1.0029 / 1.0015 / 1.0150 / 1.0126 / 1.0043.

**Geometry B — matched 256/1024 (msl=353, 96 decode steps), unprofiled wave wall_ms**

| bs | pass 4 | sp% | pass 2 | sp% | pass 1 | sp% | 2 vs 4 | 1 vs 4 | **1 vs 2** |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1251.7 | 0.23 | 1205.9 | 0.12 | 1230.6 | 0.17 | 0.9635 | 0.9832 | 1.0205 |
| 2 | 1590.3 | 0.27 | 1540.2 | 0.26 | 1583.0 | 0.57 | 0.9685 | 0.9954 | 1.0277 |
| 4 | 2243.5 | 0.43 | 2182.2 | 0.39 | 2270.8 | 0.41 | 0.9727 | **1.0122** | 1.0406 |
| 8 | 4364.6 | 0.26 | 4272.0 | 0.26 | 4428.7 | 0.27 | 0.9788 | **1.0147** | 1.0367 |
| 16 | 10592.0 | 0.80 | 10458.0 | 0.79 | 10753.0 | 0.76 | 0.9874 | **1.0152** | 1.0282 |
| **all** | 20042.0 | | 19658.4 | | 20266.1 | | 0.9809 | **1.0112** | **1.0309** |

**Geometry C — deep context (msl=897, 640 decode steps, ctx 257→896), unprofiled wave wall_ms**

| bs | pass 4 | sp% | pass 2 | sp% | pass 1 | sp% | 2 vs 4 | 1 vs 4 | **1 vs 2** |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 6889.4 | 0.13 | 6480.4 | 0.16 | 6509.4 | 0.11 | 0.9406 | 0.9448 | 1.0045 |
| 2 | 7360.3 | 0.33 | 6928.5 | 0.32 | 6964.4 | 0.31 | 0.9413 | 0.9462 | 1.0052 |
| 4 | 8136.7 | 0.42 | 7669.5 | 0.31 | 7734.6 | 0.91 | 0.9426 | 0.9506 | 1.0085 |
| 8 | 10759.5 | 0.47 | 10201.6 | 0.43 | 10336.1 | 0.45 | 0.9482 | 0.9606 | 1.0132 |
| 16 | 23252.6 | 3.18 | 22229.0 | 3.30 | 22431.8 | 3.27 | 0.9560 | 0.9647 | 1.0091 |
| **all** | 56398.5 | | 53508.9 | | 53976.2 | | **0.9488** | 0.9571 | **1.0087** |

**Pass size 2 is the pick, and the sweep is unanimous.** Pass 1 is slower than pass 2 in **14 of the
15 (geometry, bs) cells**. The single cell where it is nominally ahead — geometry A bs1, 0.9955 —
sits well inside pass 2's own 2.84 % rep spread there, and the same pair reverses on that geometry's
per-wave ms/decode-step metric (1.0029). The margin is 0.3 % aggregate at the AC-3 geometry, 0.9 % at
the decode-heavy end and **3.1 % at the prefill-heavy end** — the ordering the mechanism predicts,
because what pass 1 buys (a marginally cheaper KV tile) is fixed, while what it costs (twice the
passes, each replaying the KV stream) scales with prefill work.

The sharpest result is one the profiler-only view could not have produced: at geometry B **pass 1 is
worse than the shipped pass 4** at bs 4, 8 and 16 (1.0122 / 1.0147 / 1.0152 against rep spreads
≤ 0.8 %). Choosing 1 would have been a regression at the matched 256/1024 geometry across most of the
batch range, so this was not a cosmetic tie-break. Structurally, pass 2 fills the m16n16k16 tile
exactly (2 tokens × 8 q-heads = 16 rows) while pass 1 leaves half of it idle during prefill, which is
why the measured ordering comes out this way.

This sweep is the **primary** e2e evidence, superseding §5's two-arm tables: same protocol, same
tree, interleaved, fully audited, where §5's geometry A predates the drain gate. It also
independently **reproduces** §5 at a different tree — geometry C pass 2 vs 4 lands at
+6.3 / +5.5 / +4.6 % (bs1 / bs8 / bs16) here against +6.2 / +5.5 / +4.6 % there.

## 4. Gates

**Bit-exactness — `test_attention_qwen35_qloop.py`, 116 rows, 0 failures, every equality row at
`max_abs = 0.000e+00`** (`gates/qloop_result.json`). The pre-existing §B proves pass-size invariance
at the Qwen3.5 shape for T ≤ 4; this issue adds **§B2** for the multi-pass regime the change
actually moves through — T ∈ {8, 13, 16}, i.e. 4→8 and 8→16 passes — all bit-identical to the
production 4-pass form. The counterfactual (attention truncated to the second pass) still
*mis*matches, so the test can still discriminate a per-pass causal bug. This is the right shape of
claim: a query row's arithmetic and its order depend only on its own q and the KV stream, and the KV
stream is replayed identically per pass, so redistributing rows across passes cannot change any
row's result. Not a tolerance argument.

**HF oracle — identical verdicts at pass size 4, 2 and 1 on all 35 comparison rows** (same
`num_diff`, `max_abs`, `mean_abs` to 12 decimals; `gates/oracle_mt{4,2,1}.json`), all three
`rc=0` / "all hard assertions passed". Running the oracle at pass size 2 needed two ungated test
instantiations that did not exist (`CASE(8,2,256,2,0,2)`, `CASE(8,2,256,1,0,1)`) — the oracle always
runs an ungated arm first to isolate `core_attn_out` from the gate epilogue, and aborted with
"unsupported attention_qwen35 config" without them.

**Test-mode pipeline** (full MPK compile + dispatch) passes at pass size 4 and 2; the emitted
template arguments at 2 are `..., 256, 64, 64, 0, 0, 2, 1, 2>`, confirming `MAX_TOKENS` and
`Q_PASS_SIZE` both moved and the gated emission branch is still the one used.

**AC-3, full sweep, all five batch sizes — 50/50 byte-identical vs the committed
`results/dumps_final`** (`gates/bytediff_qp2.json`: `identical: true`, `CHANGED: none`, 10/10 at
each of bs 1, 2, 4, 8, 16). `run_ac3.py` reports the single pre-existing p06-poem position-60
divergence at all five batch sizes with `margin=0.0` — the already-adjudicated M2-era reference
logit tie (`opt/m3i8/results/VALIDATION.md`, and M3-I10 caveat 1 records the identical signature).
The byte-diff against `dumps_final` being exactly identical is the proof this change contributes
nothing to it.

## 5. End-to-end A/B, first pass — 3 reps, median, two arms (superseded by §3a)

Kept because it is the run the mechanism was first confirmed against, and because its geometry-C
numbers are what §3a reproduces at a different tree. For the selection, read §3a.

`tables/perf_medians.txt`. Both arms alternate reps inside one GPU claim, so drift hits both
equally.

| geometry | bs1 | bs2 | bs4 | bs8 | bs16 |
|---|---:|---:|---:|---:|---:|
| A — AC-3 (reference prompts, msl=132), Σ wave wall | +3.1 % | +2.8 % | +2.7 % | +2.5 % | +2.1 % |
| B — matched 256/1024 (msl=353, 96 decode steps) | +3.9 % | +3.0 % | +2.6 % | +2.0 % | +1.4 % |
| C — deep context (msl=897, 640 decode steps, ctx 257→896) | **+6.2 %** | — | — | **+5.5 %** | **+4.6 %** |

Rep spread is ≤ 0.6 % of the median everywhere except geometry C bs16 (3.2 %, and both arms move
together). Geometry A also gives per-wave `ms_per_decode_step`: 10.185 → 9.876 at bs1 (0.9697),
31.542 → 30.878 at bs16 (0.9790).

The three geometries order exactly as the mechanism requires — the win grows with decode context
(A ≈ B < C) and shrinks with batch size, because at higher batch the stage parallelises across
requests while other stages keep their cost. Geometry C at bs1 is 16 prefill iterations against 640
decode steps, so its wall ratio is effectively the decode ratio with no profiler in the loop; it
under-reads the steady-state win because it averages context 257→896, whereas the pinned workload
spends most of 1024 output tokens deeper than that.

**One discarded measurement, and the methodology fix.** The first geometry-B attempt produced one
absurd point — bs1 pass 2 at 2658.8 ms against pass 4's 1251.0 ms, a fake 2.1× regression.
`meta.gpu_before` for that run recorded 36364 MiB already resident on the pinned device: the
previous rep's 34.4 GB process had not finished tearing down, so two MPK megakernels briefly shared
one GPU, which `resources.md` says is invalid and can deadlock. The 3-sample free-GPU guard only
proves the device was idle at *claim* time. The whole arm was discarded and re-run with a per-rep
drain gate (`scripts/phase5.sh`: wait for the pinned device below 500 MiB before every rep) plus a
per-rep `gpu_before` audit in the analyser that discards a dirty start rather than averaging it in.
After the fix the same point reads 1204.1 ms with 0.2 % spread, and geometry C reports **0 dirty
reps**. Geometry A predates the drain gate; it used a different driver whose runs are long enough
that overlap did not occur, its rep spreads are ≤ 0.8 %, and both arms agree with B, but it does not
carry the same per-rep audit.

## 6. What is next, and what this does not fix

The stage is still **5.6× vLLM** at bs1 after the change (752 µs/step against FlashInfer trtllm-gen
FMHA's 133.6, `opt/m3i10/ferret_targets.json` rank 5), and the residual is **width, not bytes**: 2
tasks per layer at bs1, wall span per layer = one task's latency = `29.6 + 0.0536·ctx` µs, running at
0.24 % of the HBM roof. The reference is literally a split-KV kernel
(`fmhaSm100fKernel_…MultiCtasKvCga…ForGen`). Splitting the KV range k ways over cooperating tasks
using M3-I3's separate-task + atomic-last-block idiom (`grid.z` → `merge_task_offset`,
`runtime.cc:451-467`, `task_register.cc:4916-4956`) predicts `29.6 + 0.0536·ctx/k + merge` per layer
— at ctx 848 with k=8 that is ≈39 µs against 75 µs today, after which the 29.6 µs fixed term
dominates and becomes the next target. Recorded as `next_lever` on backlog rank 7. Not attempted
here: this issue's brief asks for the smallest change that captures most of the gap, and a one-value
default that halves the stage with zero token change and zero kernel edit is that change.

Three caveats on the numbers above. (0) Every measurement in this directory was taken on a tree at
`170ab325`. M3-I11's task-terminal TMA fence (`0cdd52f0`) landed concurrently and this commit sits on
top of it, so the **absolute** step and wall times here predate that fix. The A/B is unaffected: both
arms are the same tree, the same windows and the same GPU claim, and the change is a Python default
that cannot interact with a TMA store fence. (1) The attention wall spans measured here are **larger than
`ferret_targets.json`'s** primary basis at the same nominal window — 1447 µs/step at bs1 ctx ~848
against its 1080.5. This pipeline reproduces arm A's matched-geometry point to 0.9 %, so the
convention is not the difference; the likely cause is that the late-context capture predates
M3-I3's GDN split, which landed 4× more GDN tasks per layer and changed how the round-robin packs
workers. Not chased here because the A/B is arm-vs-arm at identical HEAD and identical windows, but
it means attention's *absolute* rank-5 gap at current HEAD is larger than the committed file says,
and both should be re-derived together at the I7 re-rank. (2) bs16 keeps M3-I10's staggered-admission
caveat: no prefill-free full-bs16 window exists at any context, so its per-slot contexts span a
range rather than a band.

## 7. Layout

| path | what |
|---|---|
| `tables/sweep3_table.txt` | **the three-way e2e sweep (pass 4/2/1 × bs 1/2/4/8/16 × 3 geometries), 135 reps, 0 discarded — the selection evidence (§3a)** |
| `tables/sweep3_medians.csv` | the same, machine-readable: medians, rep spreads, n, and all three pairwise ratios |
| `tables/ctx_sweep.csv` | the per-window context sweep, all 6 captures (3 pass sizes × bs 1/8), one row per window |
| `tables/ctx_*.json` | raw `ctx_curve.py` output including each capture's full-span anchor QC |
| `tables/perf_medians.txt` | the FIRST-pass two-arm e2e medians (§5), superseded by `sweep3_table.txt` |
| `ptxas/megakernel_qp{4,2,1}.txt` | `-Xptxas -v` on the generated megakernel TU per pass size |
| `gates/qloop_result.json` | pass-size invariance, 116 rows |
| `gates/oracle_mt{4,2,1}.json` | HF oracle at each pass size |
| `gates/bytediff_qp2.json`, `gates/run_report_qp2.json` | AC-3 per-case byte diff + harness report |
| `scripts/` | every script used: `ctx_curve.py`, `tu_i6a_attn.cu`, `probe_regs.sh`, `mk_ptxas.sh`, `run_ctx.sh`, `gate_*.sh`, `phase*.sh` (`phase6.sh` = the three-way sweep), `sweep3.py` / `sweep3_csv.py`, `perf_medians.py`, `gpu_guard_i6a.sh`, `retry.sh` |

### Provenance

The change landed at `a86b1eb1` on the strength of §§1–5. Cross-provider review then failed it at
c2 on one gap: the acceptance requires a measured pass-size sweep at **all** batch sizes, and pass 1
had only been measured in profiler windows at bs1/bs8, so the choice of 2 over 1 rested on partial
evidence. §3a closes that — it is the whole reason the sweep re-runs all three arms rather than
bolting a third arm onto the earlier tables. (The issue also sat in `todo` without a verdict while
these gaps were being closed; the coordinator has corrected its status to `review`.)

Raw profiler `.npz` (6 × 0.5–1.4 GB) and the kernel dirs stay on catalyst-B200 under
`~/mpk-qwen35/i6a/{prof,perf,ac3}/`; every table above regenerates from them with the committed
scripts.

### Reproduce

```bash
# on catalyst-B200, tree = an isolated clone at this commit, venv-mpk
cd ~/mpk-qwen35/i6a
bash probe_regs.sh                      # compile-only: register/spill vs pass size, no GPU
bash gpu_guard_i6a.sh 6,3,1,0 -- env QPLIST="4 2 1" BSLIST="1 8" bash run_ctx.sh
bash mk_ptxas.sh                        # compile-only: the generated megakernel's shared budget
bash retry.sh phase2.sh                 # oracle at each pass size + full AC-3 sweep + byte diff
bash retry.sh phase5.sh                 # first-pass e2e A/B (two arms), geometries B and C
python3 perf_medians.py ~/mpk-qwen35/i6a

# the selection evidence (SS3a): three arms x 5 batch sizes x 3 geometries, 3 reps,
# arms interleaved per (bs, rep), drain-gated, resumable. Scratch on /var/tmp because
# /raid was at 9G box-wide. ~2.7 h on one exclusive GPU.
CANDS="5,4,1,3" bash retry.sh phase6.sh
python3 sweep3.py /var/tmp/m3i6a_sweep                       # the table
python3 sweep3_csv.py /var/tmp/m3i6a_sweep out.csv           # the machine-readable form
```
