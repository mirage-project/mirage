# E1 — CTA/SM residency and the ferret MoE harness's overstated kernel win

Date: 2026-08-06 (box time). Box: catalyst-fleet1.cs.cmu.edu (8x B200, 148 SMs each).
Worktree: `/var/tmp/e1_residency/tree` = ferret workspace3 @ tag **v024** (`git describe --tags` = v024),
plus the measurement-only patch `kernel.cu.e1.patch` (166-line diff; adds `E1_GRID_OVERRIDE`, an
identically-disciplined golden-timed pass, and a grid-stride-golden bit gate; stock behavior when the
env var is unset is unchanged).
Build: current aligned convention (`build.sh` copied from workspace3, mtime Aug 6 08:02):
`/usr/local/cuda-12.8/bin/nvcc -gencode arch=compute_100a,code=sm_100a -O3 -lineinfo -std=c++20
--expt-relaxed-constexpr -Xptxas -v -lcuda -lcudart [-use_fast_math]`. All numbers below are the
**fast-math lane** (`./kernel`), the scored one.
Main sweep GPU: physical **GPU 3** (pinned via `CUDA_VISIBLE_DEVICES=3`), checked free before and
after every rep. Never GPU 0.

## 1. The delta table (bs16, the headline configs)

ws3 harness, wall-span medians (the harness's own scored metric: device %globaltimer
`max(block exit) − min(block entry)` per launch, L2 read-flush between iterations, 20 warmup +
100 timed pre-enqueued iterations, median; discipline unmodified). Median of 3 reps per regime;
per-rep spread <= 0.6% everywhere. delta% = (cand − golden)/golden.

| regime | grid.x (w13 / w2) | w13 golden us | w13 cand us | w13 delta% | w2 golden us | w2 cand us | w2 delta% |
|---|---|---:|---:|---:|---:|---:|---:|
| (a) stock | 824 / 1648 | 99.136 | 41.520 | **−58.12%** | 56.288 | 22.496 | **−60.03%** |
| (b) 296 | 296 / 296 | 99.200 | 45.568 | −54.07% | 56.320 | 29.824 | −47.05% |
| (c) 148 | 148 / 148 | 99.152 | 67.584 | −31.84% | 56.352 | 49.792 | −11.64% |
| (d) 136 | 136 / 136 | 99.168 | 77.072 | **−22.28%** | 56.320 | 53.232 | **−5.48%** |

Summed (w13+w2) time delta: stock −58.81%, g296 −51.52%, g148 −24.52%, g136 **−16.20%**
(anchors: claimed standalone −17.55%, in-MPK −9.32%).

bs8 (bonus configs, same runs):

| regime | grid.x (w13 / w2) | w13 golden us | w13 cand us | w13 delta% | w2 golden us | w2 cand us | w2 delta% |
|---|---|---:|---:|---:|---:|---:|---:|
| (a) stock | 448 / 912 | 98.112 | 30.496 | −68.92% | 55.520 | 14.320 | −74.21% |
| (b) 296 | 296 | 98.400 | 29.168 | −70.36% | 55.536 | 19.872 | −64.22% |
| (c) 148 | 148 | 98.208 | 44.064 | −55.13% | 55.568 | 29.520 | −46.88% |
| (d) 136 | 136 | 98.304 | 44.512 | −54.72% | 55.552 | 29.632 | −46.66% |

Correctness at every regime and rep (12/12 runs, all 10 configs): the harness's built-in
candidate-vs-serial-golden bit-exact gate **PASS**, and the added grid-stride-golden vs
serial-golden gate **PASS** (`golden_grid_exact=1`) — the "expert-parallel is bit-safe" claim
holds at 824/1648, 296, 148, and 136 for both bodies.

## 2. Setup validation — the required stock reproduction, and what it actually found

The brief required regime (a) to reproduce "the ws3 harness's" claimed W13 −19.88% / W2 −13.55%.
**Stock does not reproduce those numbers against the worktree's frozen `golden::` — it gives
−58.12% / −60.03% — and the discrepancy is a provenance fact, not an instrument fault:**

1. The ws3 harness at v024 never times its `golden::` at all (golden runs once at `grid=(1,1,1)`
   for the bit gate only; only the candidate is timed). There is no stock configuration of this
   harness that ever printed −19.88%. The golden-timed pass in this experiment is new
   (identical discipline, same grid as the candidate).
2. The claimed numbers' actual source is the integration report of record:
   `/var/tmp/moe_v024/gates/standalone_fastmath.log` (Aug 6 03:54), produced by
   `mirage-moe-v024/demo/qwen3_5/accept/opt/m4i7/scripts/bitexact_v024.cu --benchmark`:
   `STANDALONE family=w13 bs=16 nact=101 geometry=split2 v012_us=100.814407 v024_us=80.772797
   reduction_pct=19.879709` and `family=w2 ... v012_us=58.647198 v024_us=50.700802
   reduction_pct=13.549489`. That instrument compares the **shipped v012 PATH1-flat body**
   (v012 + the integrator's work-item flattening — not ws3's pre-flattening `golden::`) against
   the integrated v024 dispatcher, at **grid.x = 128** (`MPK_GRID_X`, = MPK's emitted
   `min(num_experts, mbt*topk)`), **dynamic smem = 205,824 B** (`MAX_DYNAMIC_SHARED_MEMORY_SIZE`,
   which alone forces the occupancy limit to 1 CTA/SM), split-2 N-slices, cudaEvent over 40
   back-to-back iterations (no L2 flush between iterations), 11 interleaved samples, median.
3. **The claimed −19.88% was therefore already a one-block-per-SM measurement.** It is not the
   oversubscribed-harness number the brief assumed. The ws3 harness's own stock regime (3-4
   resident CTAs/SM, table above) overstates far more: −58%/−60% against the v012-era body.
4. Fresh reproduction of the record instrument (rebuilt from the committed tree, nvcc 12.8,
   fast-math; GPU 5, 3 gated reps; GPU 7, 1 rep each at c++17 and c++20):
   - w13 bs16: v012 ~100.3, v024 ~79.5 us -> **−20.5 to −20.8%** — reproduces the recorded
     −19.88% within ~0.9 pt.
   - w2 bs16: v012 ~56.1-57.4, v024 ~45.7-46.4 us -> **−18.6 to −19.0%**, NOT the recorded
     −13.55%. The record log (03:54) predates the final header commit (7ba0d62d, 05:44); the
     committed w2 v024 arm is ~9% faster than the one in the record (45.7-46.4 vs 50.70 us).
     c++17 vs c++20 changes nothing (−18.564 vs −18.560), ruling out the compile-standard as the
     cause. The W2 claim anchor is soft on the current tree.
5. Consistency anchors for this experiment's instrument: candidate stock TFLOPS reproduce the
   loop's record (w13_bs16 12.925 vs the a073-session control 13.066, −1.1%; w2_bs16 11.933 vs
   12.018, −0.7%); candidate stock absolute (41.52 us) matches the alignment job's ncu figure for
   the v024 entry (43.264 us, profiler-inclusive); the worktree golden at bs16 (99.14 us) lands
   1.7% from the record instrument's shipped-flat v012 arm (100.81 us) — two different v012-era
   baselines agreeing at bs16 because both end up with ~1 working block/SM there (103 expert-CTAs
   vs 128 workers).

Per the brief's stop rule, no conclusions are drawn beyond the required arithmetic; the regime
sweep itself ran as specified and is reported in full.

## 3. Do the (c)/(d) deltas approach the in-MPK numbers (W13 −7.40%, W2 −12.18%)?

- **W13: no.** g148 −31.84%, g136 −22.28%. The 1-block/SM regimes approach the *claimed
  standalone* anchor (−19.88%, itself a 1-block/SM measurement) — not the in-MPK −7.40%.
- **W2: it crosses them.** g148 −11.64% lands essentially on the in-MPK −12.18%; g136 −5.48%
  undershoots both the in-MPK number and the claimed −13.55%.
- Summed: g136 −16.20% vs claimed −17.55% (close) vs in-MPK −9.32% (not approached).

One structural note the cross-review needs: with `E1_GRID_OVERRIDE` the host still computes the
data-dependent PATH from `total_work` vs SM count, so at 296/148/136 the candidate keeps running
its **PATH0** (legacy 16 B cp.async) body over a looped work list. Production 1-wave dispatch
(and the record instrument's v024 arm) selects the bulk-fetch path family instead. So regimes
(c)/(d) measure "the harness's PATH0 candidate at 1 block/SM", which is the residency question as
posed, but is not byte-identical to what production executes at 1 block/SM.

## 4. The required fraction arithmetic

Using the brief's formula `(claimed − regime_delta) / (claimed − inMPK)` with claimed W13 19.880,
W2 13.549, in-MPK W13 7.402, W2 12.181:

- W13 @ grid136: (19.880 − 22.281)/(19.880 − 7.402) = −2.401/12.478 = **−0.19** -> the 1-block/SM
  harness point lands 2.4 pts past the claimed number, on the far side from in-MPK; by this
  formula the residency regime explains **~0% (slightly negative)** of the claimed->in-MPK W13 gap.
  (@ grid148: −0.96.)
- W2 @ grid136: (13.549 − 5.483)/(13.549 − 12.181) = 8.066/1.369 = **+5.90** -> degenerate
  (>1): the 1-block/SM point undershoots both anchors. (@ grid148: +1.39, also >1.)
- The formula's premise — that the claimed deltas were oversubscribed-regime numbers — does not
  hold (Section 2); the numbers above are reported for completeness.

Anchored instead at the ws3 harness's actual stock regime (what the harness itself rewards),
fraction = (stock_delta − regime_delta)/(stock_delta − inMPK_delta):

- **W13: stock −58.12% -> g136 −22.28% covers 35.84 of the 50.72-pt stock->in-MPK gap = 70.7%**
  (g148: 51.8%; g296: 8.0%). The residual −22.28% -> −7.40% (29.3% of the gap) is what remains at
  matched 1-block/SM width.
- **W2: stock −60.03% -> g136 −5.48% covers 54.55 of the 47.85-pt gap = 114.0%** (g148: 101.1% —
  an almost exact landing on the in-MPK −12.18%; g296: 27.1%). The 1-block/SM correction
  overshoots the in-MPK value at g136.
- Summed: g136 covers 86.1% of the stock->in-MPK gap (g148: 69.3%).

Equivalent absolute statement: forcing the candidate from stock to 136 CTAs slows it
w13_bs16 41.52 -> 77.07 us (+85.6%) and w2_bs16 22.50 -> 53.23 us (+136.6%), while the golden is
regime-flat (99.14-99.20 us, 0.07% spread; 56.29-56.35 us, 0.11% spread — expert-granular, <=103
working CTAs at every regime, so its residency never changes).

## 5. Occupancy math (what binds, per ptxas + driver)

256 threads/CTA; B200: 65,536 regs/SM, 232,448 B max dyn smem/block (opt-in), ~233,472 B/SM
shared budget; `cudaOccupancyMaxActiveBlocksPerMultiprocessor` values printed per config agree
with the hand math in every case.

| entry (fast-math lane) | regs | stack/spill | dyn smem (bs16) | regs limit | smem limit | occupancy limit |
|---|---:|---|---:|---:|---:|---:|
| `cand_moe_kernel_timed<w13>` | **75** | 0 / 0 | 58,400 (v023 host pad) | floor(65,536/19,200)=3 | floor(233,472/58,400)=3 | **3** (both) |
| `cand_moe_kernel_timed<w13>` bs8 | 75 | 0 / 0 | 42,756 | 3 | 5 | **3** (registers) |
| `cand_moe_kernel_timed<w2>` | **64** | 0 / 0 | 42,756 | floor(65,536/16,384)=4 | 5 | **4** (registers) |
| `golden_moe_kernel_timed<w13>` | 59 | 0 / 0 | 41,620 | 4 | 5 | **4** (registers) |
| `golden_moe_kernel_timed<w2>` | 60 | 0 / 0 | 41,620 | 4 | 5 | **4** (registers) |

- Achieved residency = min(limit, placement): stock w13 **3**/SM (824 CTAs, 5.6 waves), stock w2
  **4**/SM (1648 CTAs, 11.1 waves); g296 -> 2/SM; g148/g136 -> 1/SM (breadth-first placement;
  g136 leaves 12 SMs idle). The brief's "4-5 resident CTAs/SM" is the smem-only bound of the
  41.6 KB PATH0 layout; with the current-convention register counts (75) and the v023 58,400 B
  host pad, the candidate's stock residency is actually 3 (w13) and 4 (w2).
- Golden's placement never exceeds 1 working CTA/SM at any regime (nact = 103 <= 136/148 at bs16;
  56/57 at bs8): its occupancy limit (4) is never the binder, which is why its time is
  regime-flat. Extra no-op CTAs at large grids cost <0.1% (99.136 @ 824 vs 99.168 @ 136).
- Candidate registers under the v024-era convention (in-tree `k_v024f`, CUDA 13.2/c++17-era
  build) were 80 (w13) — same 3 CTA/SM limit, so the regime classification is
  convention-independent. Current-convention entries have 0 spill and 0 stack, matching the
  expected "~64-75 regs, 0 spill".

## 6. cudaEvent cross-checks

Recorded per config alongside the wall spans (logs); event medians sit a constant ~+5.3-6.3 us
above the wall spans at every regime (the box's known launch/event floor), e.g. stock w13_bs16
41.52 wall / 47.10 event; g136 golden 99.17 wall / 104.45 event. No regime-dependent anomaly.

## 7. GPU hygiene

- Main sweep: GPU 3, 12 runs (3 reps x 4 regimes), compute-app list checked empty before and
  after every rep — all clean (`run_regime.sh` gates; the runner aborts on any co-tenant).
- After the sweep completed, a vLLM EngineCore (~170 GB) appeared on GPU 3. Two record-instrument
  runs made during that window (`logs/record_repro_rep{1,2}.log`) are **void** (co-tenant
  present) and were redone: GPU 5, 3 gated reps, all clean; plus GPU 7 (c++17 and c++20 builds),
  clean. GPU 5 acquired its own tenant after its reps completed; final flag-check runs moved to
  GPU 7.
- Cross-GPU anchor: ws3 harness stock + g136 re-run once on GPU 7 — every golden/cand median
  within 0.4% of the GPU 3 values (stock w13 −58.07% vs −58.12%; g136 w13 −22.36% vs −22.28%;
  w2 −60.07/−5.40 vs −60.03/−5.48), so the GPU 3 sweep and the GPU 5/7 record-instrument numbers
  are comparable.
- SM clock/temp/power snapshots per rep in `logs/*_rep*.gpu`.

## 8. Exact commands

```
# worktree (v024) + build
git -C /home/muhengl/mpk-qwen35/ferret/workspace3 worktree add /var/tmp/e1_residency/tree v024
cp /home/muhengl/mpk-qwen35/ferret/workspace3/build.sh /var/tmp/e1_residency/tree/
cp /home/muhengl/mpk-qwen35/ferret/workspace3/data/{*.bin,reference.json,meta.json} /var/tmp/e1_residency/tree/data/
# apply kernel.cu.e1.patch (E1_GRID_OVERRIDE + golden-timed pass), then:
cd /var/tmp/e1_residency/tree && ./build.sh          # both lanes; ./kernel = fast-math (scored)

# the sweep (GPU 3; /var/tmp/e1_residency/run_regime.sh wraps these with free-GPU gates)
CUDA_VISIBLE_DEVICES=3 ./kernel                          # (a) stock, x3
CUDA_VISIBLE_DEVICES=3 E1_GRID_OVERRIDE=296 ./kernel     # (b) x3
CUDA_VISIBLE_DEVICES=3 E1_GRID_OVERRIDE=148 ./kernel     # (c) x3
CUDA_VISIBLE_DEVICES=3 E1_GRID_OVERRIDE=136 ./kernel     # (d) x3
python3 /var/tmp/e1_residency/aggregate.py               # medians -> aggregate.json

# record-instrument reproduction (claim provenance)
/usr/local/cuda-12.8/bin/nvcc -O3 -std=c++17 -gencode=arch=compute_100a,code=sm_100a \
  --expt-relaxed-constexpr -DMIRAGE_GRACE_BLACKWELL -DMPK_TARGET_CC=100 -DMODE_OFFLINE \
  -DMIRAGE_BACKEND_USE_CUDA -DMIRAGE_ENABLE_MOE_FP8_BLOCKSCALE_V024 -use_fast_math \
  -I$D/include/mirage/persistent_kernel -I$D/include/mirage/persistent_kernel/tasks -I$D/include \
  -o /var/tmp/e1_residency/bitexact_v024_fm $D/demo/qwen3_5/accept/opt/m4i7/scripts/bitexact_v024.cu
CUDA_VISIBLE_DEVICES=5 /var/tmp/e1_residency/bitexact_v024_fm --benchmark   # x3 (plus c++20 variant on GPU 7)
```
($D = /home/muhengl/mpk-qwen35/mirage-moe-v024; nothing under the user's trees was modified —
the worktree, /var/tmp/e1_residency, and logs are the only writes.)

## 9. Raw artifacts

- Sweep logs: `/var/tmp/e1_residency/logs/{stock,g296,g148,g136}_rep{1,2,3}.log` (+ `.gpu` snapshots)
- Cross-GPU anchors: `logs/xgpu7_stock.log`, `logs/xgpu7_g136.log`
- Record-instrument runs: `logs/record_gpu5_rep{1,2,3}.log`, `logs/record_gpu7_cpp{17,20}.log`;
  void GPU 3 runs kept for the record: `logs/record_repro_rep{1,2}.log` (co-tenant present)
- Builds: `logs/build.log` (ws3 both lanes, full `-Xptxas -v`), `logs/record_build*.log`
- Aggregates: `/var/tmp/e1_residency/aggregate.json` (per-rep values), `aggregate.py`
- The measurement patch: `/var/tmp/e1_residency/kernel.cu.e1.patch`
- Claim provenance (pre-existing, read-only): `/var/tmp/moe_v024/gates/standalone_fastmath.log`,
  `/var/tmp/moe_v024/REPORT.md` ("Standalone-to-E2E transfer"), instrument source
  `mirage-moe-v024/demo/qwen3_5/accept/opt/m4i7/scripts/bitexact_v024.cu` (MPK_GRID_X=128,
  smem=205,824 B, split-2, 11x40 interleaved cudaEvent)
- In-MPK anchors quoted from `/var/tmp/alignment/REPORT.md` Part 1 (w13 18.764->17.375 us =
  −7.402%; w2 12.535->11.008 us = −12.181%; summed −9.316%).
