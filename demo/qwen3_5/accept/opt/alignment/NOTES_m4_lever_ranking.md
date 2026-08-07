# M4 lever ranking — which task type can actually pass AC-4, per batch size

Source: `demo/qwen3_5/accept/opt/m4i8/tables/m4i8_summary.txt` §6, computed on the HEAD arm
(floors 4130.7 / 5275.7 / 5976.0 us at bs 1/8/16 vs vLLM 3503.0 / 4727.0 / 5301.0 us).

**Read the counterfactual correctly: each row ZEROES that task type entirely — it is an upper
bound on the lever, not a "bring it to parity" estimate.** I mis-remembered this once as
"attention→parity gives 1.003x"; it is "attention→FREE gives 1.003x", which is a much weaker
claim and it changes the ranking.

| zero this type | bs1 | bs8 | bs16 |
|---|---|---|---|
| **MOE_W13_FP8_BLOCKSCALE** | 1.017 | **0.904** | **0.851** |
| MOE_W2_FP8_BLOCKSCALE | 1.109 | 0.999 | 0.927 |
| LINEAR_FP8_BLOCKSCALE (dense) | 1.016 | 0.989 | 0.950 |
| ATTN_SM100 | **1.003** | 0.994 | 1.097 |
| GDN_RECURRENT | — | 1.043 | 0.998 |
| MOE_TOPK_SOFTMAX | 1.074 | 1.006 | 1.127 |
| QUANTIZE_FP8 | 1.074 | 1.034 | 1.116 |
| LINEAR_SM100 | 1.072 | 1.027 | 1.081 |

## What this settles

1. **W13 is the top lever overall** — the only type that passes with margin at BOTH bs8 (0.904)
   and bs16 (0.851). It contributes ~1463.6 us of cp at bs16, so reaching parity there needs
   roughly a 46% W13 improvement from the HEAD basis (less, now that the fusion stack shipped).
   The MoE ferret loop's worst config is exactly `w13_bs16`, so that loop is correctly slotted —
   its risk is DRIFT onto w13_bs1/bs2, which cannot move min_ratio (see
   `agents/ferret-kernel-agent.md`).
2. **Attention is NOT a bs16 lever at all.** Zeroing it gives 1.097x because the WORK BOUND
   (5814.0 us) becomes binding once its cp contribution is gone — a latency win cannot cross a
   throughput floor. It is the best bs1 lever and a passing bs8 lever, but even entirely free it
   only reaches 1.003x at bs1.
3. **No single lever passes bs1.** The three best (attention 1.003, dense 1.016, W13 1.017) each
   land at or just above parity ALONE, so bs1 requires a COMBINATION — which is also why the
   fusion work mattered there and nowhere else.
4. Corollary for slotting ferret: attention is the weakest of the three plausible levers at bs16
   (useless there) but the STRONGEST at bs1. Prefer W13 for bs8/bs16, attention for bs1.

## INTEGRATION QUEUE 2026-07-31 — sized against the cp contributions above

Two loops have produced value the shipped tree does not have. Sized by what each removes from the
BINDING floor, using the cp contributions in the table above (attention 615.6us of the 4130.7us bs1
path; W13 1463.6us of the 5976.0us bs16 path):

1. **ATTENTION — do first.** 46.7 -> 18.24us in-harness (2.56x, v024). Removing ~61% of 615.6us is
   **~375us off the bs1 floor, ~9%**. Bit-exact on output AND K/V cache in both nvcc lanes, and it
   REDUCES registers 246 -> 227, which relieves the shared __launch_bounds__(256,1) ceiling for
   every other task — a systemic second-order win no other candidate offers. Loop is plateauing
   (0.548 -> 0.563 -> 0.570, 3/6 no-tag rounds), so it is at the right moment.
2. **MoE v022 — second.** Loop CLOSED at 0.932; shipped kernel is v012-based. The v013..v022 delta
   is roughly 5-15% of the w13 stage (its own a044 ablation priced the SWAP_AB regime gate at +4.7%
   on w13_bs16), i.e. **~150us off the bs16 floor, ~2.4%**. Smaller than attention but pure capture.

CAVEATS FOR BOTH. (a) The ws3 extracted deliverable `kernel.cuh` is STALE — dated before v022 was
tagged — so re-extract rather than trusting the file on disk. (b) Loop and tree diverged in KIND:
the integrator layered work-item flattening on top of v012 that the standalone harness cannot see or
reward, so this is a port over the new body, not a file swap. (c) v012 -> v022 is +698/-125 lines.
This is a dedicated block of work, not a monitoring-tick task; several ticks have now accumulated it.

## CORRECTION 2026-07-31 — the "~29.6 us fixed cost" was never itemized, and it is wrong

I wrote here, and repeated to the user, that attention's in-harness bar was structurally unreachable
because a ~29.6 us FIXED cost (task-entry / barrier / dispatch) dwarfed the ~7.5 us target — "29.6 >
7.5 even at k=infinity". workspace6 finally itemized it with a `%globaltimer` phase table at ctx=848
and the story does not survive:

| span | bs1 | phase |
|---|---|---|
| p0->p1 | 0.544 us | task entry + entry barrier — **1.2%, not 64%** |
| p1->p2 | 0.736 us | Q(+gate) cp.async issue |
| p2->p4 | 4.19 us | tile-0 drain + Q/K-norm + RoPE |
| **p4->p5** | **36.06 us** | **KV loop, 14 tiles — 78.6%** |
| p5->p6 | 3.52 us | cross-warp m/d/o merge (un-probed since v004) |
| p6->p7 | 0.83 us | final store |

Genuinely irreducible is ~2.3 us. The gap to FlashInfer is **PARALLELISM**: at bs1 the task runs
grid=(1,2) = TWO CTAs on a 148-SM GPU, while the reference is `...MultiCtasKvCga...` and splits KV
across many. So split-KV — which the task.yaml's own constraints had effectively FENCED OFF while
its hints invited it — is the largest reachable item, and the corrected in-harness best case is
**~8-10 us**, not "unreachable". Also note the synergy with the 136-worker change: a split has more
idle width to claim.

TWO LESSONS. (a) A number quoted repeatedly across documents is not evidence; this one propagated
through task.yaml, a seed, my memory and my reporting to the user without anyone itemizing it —
the second time on this project (cf. the matched-geometry gaps artifact). Demand the breakdown
before letting a number close off a lever. (b) When a spec's constraints forbid what its hints
invite, the agent will obey the constraint and silently skip the best lever; read specs for that
contradiction directly.

## ATTENTION INTEGRATION — CORRECTED 2026-08-05 by codex's own audit of the tags

I reported this queue optimistically twice. The corrections, from `/var/tmp/attn_port/PLAN.md`:

1. **v024 is 18.016 us at bs1, not 18.240** (18.240 was v022/v023). So v006 -> v024 is
   47.424 -> 18.016 = **2.63x**, not "46.7 -> 18.24".
2. **v024 does NOT reduce registers.** Its tag says **228 default / 244 fast-math**, 0 spill. The
   227 figure was v013/v015 DEFAULT lane only. So the "246 -> 227, relieves the shared ceiling for
   every other task" claim I made twice is WRONG for the final kernel — 244 is 11 below the 255
   architectural limit, not comfortably under it. There is no systemic register win to bank.
   [CORRECTED AGAIN 2026-08-06 by the alignment audit: 228 and 244 were TWO DIFFERENT FUNCTIONS
   (precise split-candidate vs golden), not a lane pair — the third life of this number. The real
   integrated K=1 body under the production toolchain is **236 precise / 232 fast, 0 stack,
   0 spill**. Also: the old harness compiled with UNQUALIFIED `nvcc` = CUDA 13.2.51 + C++17 while
   production JIT pins `/usr/local/cuda-12.8/bin/nvcc` + C++20 — same source differs 64 vs 75
   regs across toolchains, so NO cross-toolchain register comparison is like-for-like. My earlier
   flag-audit claim that "-std matches" was wrong.]
3. **The device-only subset is worth ~3.296 us (11.2%), not 2.63x.** Directly measured v006->v009
   is 47.424 -> 44.128. The big wins are TMA: v010 (-7.776), v013 (-7.392), v017 (-1.120), and they
   need HOST-SIDE descriptor plumbing. Later device-only tags (v012/v014/v015/v016/v018/v019/v020)
   had their deltas measured ON TOP OF TMA and cannot be honestly added to a no-TMA build without a
   fresh ablation.
4. **The 16-arg ABI blocks TMA, but it is not absolute** — my ruling. `TaskDesc::input_tma_desc_ptrs`
   already exists and `tma.cuh`'s switches already dispatch descriptor creation; `TASK_ATTN_SM100`
   simply has no cases in either (`tma.cuh:273+`, `:1465+`). Approved design: keep the current entry
   point as a compatibility wrapper, add a TMA-capable internal overload that generated task code
   calls. That is Stage 2.
5. **EXTRACTION TRAP:** the standalone harness default `KV_SPLITS_CFG=0` selects 14/7/4 splits, and
   those builds REPORT OUTPUT MISMATCH — they were never the tagged score. A mechanical extraction
   would carry them in. Strip the split template param, partial buffers, counter,
   `split_merge_and_store` and every `K_SPLITS>1` branch; do NOT merely default it to 1.
6. **BIT-EXACT IS NOT RACE-FREE:** v022 passed 180/180 value checks while racecheck found **1812
   hazards**; v023 restored a post-loop 256-thread rendezvous because `s_o_buffer` aliases the K/V
   arena. Any port must run `compute-sanitizer --tool racecheck`, not just value comparison.

So attention's integrable value TODAY is ~11% of its headline, and the rest is a real MPK-side
change. That makes MoE v022 (~2.4% of the bs16 floor, pure capture) comparable value at lower risk —
size them together rather than assuming attention dominates.

## MEASURED 2026-08-06 — a standalone kernel win does NOT transfer proportionally

The attention Stage-1 port was measured end to end on the real model, flag ON vs OFF, 3 paired reps
per batch, separate kernel dirs, shipped fusion config on both arms:

| bs | OFF | ON | e2e speedup | fraction of the standalone reduction retained |
|---|---|---|---|---|
| 1 | 765.1 ms | 724.4 ms | **1.056x** | 12.6% |
| 8 | 2069.9 ms | 2035.2 ms | **1.017x** | 3.9% |
| 16 | 3286.4 ms | 3267.7 ms | **1.006x** | 1.3% |

9/9 paired runs favoured ON, no crossovers, tokens byte-identical — the win is REAL but SMALL. The
standalone kernel was **1.75x faster**; end to end that became 1.056x at bs1 and was nearly washed
out at bs16.

WHY, and it is structural: Qwen3.5 has 40 layers but only **10 are full-attention** — the other 30
are GDN. The change removes no scheduler records or barriers and touches no GDN/MoE/GEMM/fusion
work, and at larger batch more attention is overlapped so other work dominates the path.

THIS RECONCILES WITH THE FLOOR MODEL RATHER THAN CONTRADICTING IT, which is the useful part:
attention is ~615.6us of the 4130.7us bs1 critical path (15%), so a 43% reduction predicts ~6.4% and
we measured 5.3%. The model was right; MY EXPECTATION was inflated by the standalone headline. The
counterfactual table above says "zeroing attention ENTIRELY moves bs1 from 1.179 to 1.003" — a 15%
move — and I repeatedly read that as though attention were worth far more than 15%.

CONSEQUENCES:
1. **Stage 2 (TMA) stopped.** Its payoff would be a fraction of an already-diluted 5%, and it
   currently spills 44/52 B default / 72/48 B fast-math against a 0/0 control — violating the hard
   zero-spill gate whose cost is ~9-10% of the step.
2. **Every future kernel win must be measured END TO END before it is believed.** The standalone
   ratio is a proxy; the paired e2e A/B with byte-identical tokens is the truth. This is now the
   default gate for integrations.
3. **Rank levers by their share of the CRITICAL PATH, not by their standalone ratio.** dense and W13
   are on the hot path at every batch size (zeroing dense: 1.179->1.016 at bs1, 1.127->0.950 at
   bs16); attention is not, at bs16 (1.097x even when free).

## 2026-08-06 — THE SECOND E2E MEASUREMENT, AND IT REVERSES SIGN AT bs16

MoE W13/W2 at ferret v024 was integrated (flattening preserved, all gates green: bit-exact both
lanes, full-TU register/spill, racecheck, 15/15 token-identical arms, AC-3 STABLE) and measured
end to end against the shipped kernel:

| bs | shipped | v024 | change | paired wins |
|---|---|---|---|---|
| 1 | 741.79 ms | 725.66 ms | +2.18% | 3/3 |
| 8 | 1991.09 ms | 1969.12 ms | +1.10% | 3/3 |
| 16 | 3070.84 ms | 3097.69 ms | **-0.87%** | **0/6** |

Isolated at production geometry the kernels improved 15.97 / 16.05 / 17.55%. Transferred:
13.6% / 6.9% / **-5.0%**. At bs16 a 17.55% kernel win became a whole-step REGRESSION that lost all
six paired runs.

**This is the top-ranked lever in the counterfactual table** (zeroing W13 gives 0.851 at bs16) and
integrating a large, correct, gate-clean improvement to it made bs16 SLOWER. Combined with the
attention result (1.75x standalone -> 1.056/1.017/1.006 e2e), that is two independent measurements
saying standalone kernel optimization has reached diminishing returns on this model, and at bs16 has
gone negative.

WHAT IT IS NOT: not registers (the full-TU gate passed), not correctness (byte-identical tokens,
AC-3 STABLE), not the flattening (preserved and verified). Codex also independently confirmed the
residency point — "Ferret's standalone CTA-residency heuristic is not meaningful when MPK runs one
persistent worker block per SM".

THE OPEN QUESTION, and it is now the most valuable one on the project: WHY does a faster kernel make
the whole step slower at bs16? Leading hypothesis worth testing — the standalone harness measures a
kernel without the megakernel's CROSS-WORKER MEMORY CONTENTION. w13_bs16 is DRAM-request-bound at
4.72 TB/s with MMA ~0 by the loop's own measurement; a body tuned for single-CTA request efficiency
may issue a pattern that is worse when 136 workers run it concurrently. A second candidate: the
static schedule (runtime.cc prelaunches every task and rewrites deps to EVENT_EMPTY), so changing a
stage's duration reshuffles which task is the straggler.

CONSEQUENCE: the MoE loop was RETIRED (ws3, final v025/1.013) — improving a kernel whose integration
regresses the target is waste. Do not start new standalone kernel loops until the bs16 reversal is
explained. Rank future work by MEASURED e2e transfer, not by standalone ratio or by the
counterfactual table, which both now have a demonstrated failure mode.

## 2026-08-06 — PART-1 DISCRIMINATING MEASUREMENT: the reversal decomposes into TWO stacked losses

The alignment job measured the MoE v024 flag ON vs OFF **inside the megakernel** with MPK's own
profiler at full production geometry (136 workers + 48 schedulers = 184 blocks, bs16, window
[2,350], anchor_qc PASS, tokens byte-identical between arms). Data:
`/var/tmp/alignment/part1/{off,on}/cp136.json`.

| level | OFF | ON | delta |
|---|---|---|---|
| standalone harness (fm lane, prod geometry) | — | — | **−17.55%** (claimed) |
| in-MPK per-task: w13 | 18.764 us | 17.375 us | **−7.40%** |
| in-MPK per-task: w2 | 12.535 us | 11.008 us | **−12.18%** |
| in-MPK w13+w2 path time | 1252.0 us | 1135.3 us | **−9.32%** |
| critical path (levelmax) | 4454.6 us | 4323.5 us | −2.94% |
| step, profiled | 8949.3 us | 8928.5 us | −0.23% |
| step, UNPROFILED (earlier A/B) | 3070.84 ms/96tok | 3097.69 ms | **+0.87% SLOWER** |

**Layer 1 — standalone→in-MPK conversion ≈ 0.53.** Half the isolated win is eaten by the execution
regime itself (leading mechanism: 136 heterogeneous workers sharing L2/DRAM; the codex job is
correlating nsys+counters at 20k iterations to pin it). The harness number was NOT wrong about the
kernel — the kernel IS faster in-MPK — it was wrong about how much.

**Layer 2 — in-MPK per-task→step ≈ 0.** A real −9.3% on 28% of the cp moved cp only −2.9% and the
real step 0 to negative. The surviving win lands off the binding constraint (straggler reshuffle
under the static prelaunched schedule / work-bound coupling at bs16).

CONSEQUENCES (supersedes "do not start new loops" above — user directive 2026-08-06):
1. **The ferret gating metric must become IN-MPK PER-TASK DURATION** (profiler npz → per-type mean
   at production geometry), with standalone KERNEL_RESULT demoted to inner-loop exploration signal.
   A tag is integration-ready only when the in-MPK per-task number confirms.
2. Loops RELAUNCH only after the aligned test mode is verified (codex alignment job Parts 2–4 +
   checklist), per the user's explicit order: align first, then continue the kernel agent.
3. Fresh bs16 cp composition for target ranking (ON arm): LINEAR_SM100 1035us (23.9% of cp),
   MOE w13 695 + w2 440 + topk_softmax 573 + mul_sum 185, LINEAR_FP8 377, ATTN 349, GDN 318.
   TOPK_SOFTMAX (573us, never optimized) and LINEAR_SM100 are the fresh candidates; but any new
   loop must chase the *in-MPK* number, and step-level transfer still passes through layer 2.

## 2026-08-06 — COMBINED TREE: BOTH features beat the sum of parts at EVERY batch size

Four-arm counterbalanced measurement on `mirage-combined-v024` (base ee300d5e + attention Stage-1 +
MoE v024, each behind its own flag; /var/tmp/combined/REPORT.md):

| bs | ATTN alone | MOE alone | BOTH | interaction | BOTH wins |
|---|---|---|---|---|---|
| 1 | +5.91% | +1.06% | **+8.78%** | +10.6 ms | 4/4 |
| 8 | +1.81% | +1.17% | **+3.61%** | +11.0 ms | 4/4 |
| 16 | +0.58% | **−0.86%** (0/4) | **+0.31%** | **+18.5 ms** | 4/4 |

Interaction positive in ALL 12 blocks — the bs16 MoE sign reversal is RESCUED by co-presence with
the attention change (consistent 17.9–18.9ms at bs16). Tokens byte-identical across all 48 arm
outputs. LESSON: features that individually dilute or reverse can compose positively — measure the
COMBINATION, never sum the A/Bs (this is the second direction the composition error runs).

**Zero-spill gate FAILS on the combined TU**: with attention ON (alone or BOTH), fast lane reads
255 regs / 160 B stack / **96 B spill stores + 96 B spill loads** vs OFF's 255/112/0/0. MoE adds
nothing (equals OFF). Attention on the original attn-s1 base fit clean — the ee300d5e context
(fusion default-ON etc.) plus v024 collide. e2e wins WITH the spill, so removing it should add.
→ ws6 re-targeted to `attention-sm100-regfit` (fit at ≤112B stack/0 spill, latency floor 0.98,
TU probe = acceptance instrument).

## 2026-08-06 — LAYER-2 MECHANISM NAMED: faster tasks can PACK WORSE (W13)

From the alignment report's exact-schedule reconstruction (bs16, 136w): W13 per-task t_live
improved 7.4% but the W13 REALIZED STAGE SPAN got **5.6% WORSE** — self-concurrency fell
104.6 → 91.8 live workers and perfect-pack ratio degraded 1.30 → 1.48. The OFF arm is WORK-bound
at bs16; the ON arm flips to **CP-bound** with W13 the dominant straggler (removing W13 gives the
largest exact-CP reduction in every inspected iteration). So layer 2 is not mysterious scheduling
noise: shorter tasks under the static prelaunched schedule bunch worse, and the stage's makespan
is set by packing, not by the sum of task times. CONSEQUENCE for kernel loops: a per-task win
must either keep (or raise) the stage's achievable concurrency, or it converts to ~nothing;
task-count/granularity changes that IMPROVE packing may be worth more than body speed. The MPK-side
fix direction (finer W13 task decomposition or dynamic work stealing) is a scheduler/graph task,
not a kernel task.
