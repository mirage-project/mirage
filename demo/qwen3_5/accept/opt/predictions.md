# M3-I1 predictions (written BEFORE any measurement)

Recorded so predicted-vs-observed is falsifiable. Static inputs used: the compiled task graphs
`m2i9/kernel_bs{1,16}/task_graph_rank0.json` (task-type histograms), the M2-I9 timings
artifacts, and `persistent_kernel.cuh` profiler placement. No profiler run had been taken yet.

Static facts already established (not predictions):

* task-graph size / iteration: **41 048 tasks at bs1**, **59 348 at bs16**.
* MoE `241`/`242` = 10 240 each **at both batch sizes** — MoE tasks are per-expert-tile and
  batch-independent (256 tasks/layer × 40 layers). With top-8-of-256 routing, most of them
  cannot have rows to process at small batch.
* GDN `237` (recurrent) 960 → 15 360 and `234` (conv) 240 → 3 840 from bs1 → bs16: exactly
  ×16, i.e. **per-request** tasks.
* profiler `PROFILER_EVENT_START` is emitted *after* the worker's dependency-wait loop, so
  dependency stall time appears as a **gap** on the worker track, never inside a task event.
* 128 worker blocks, 80 schedulers (20 scheduler blocks × 4 warps) on this 148-SM B200.

Predictions:

* **P1 — the step is not compute-bound.** Per-worker busy time (Σ task µs / 128) will be
  < 25 % of the measured step at bs1. Mechanism: 320 tasks per worker per 15.1 ms step = a
  47 µs slot per task, while these tasks are ≤ a few µs of work each.
* **P2 — all-worker-idle "dead" time ≥ 20 % of the bs1 step.** Mechanism: 40 sequential layers
  each with several event-triggered sync points; each event round-trip (worker → sched queue →
  scheduler → worker queue) costs single-digit µs and cannot overlap at bs1.
* **P3 — GDN recurrent (237) is the largest single compute task-type total at bs1**, since 30
  of 40 layers are GDN and the recurrent state read/write is 128.8 MB/token/step (M2-I11).
* **P4 — the bs8→bs16 knee (19.8 → 42.7 ms) is driven by per-request task-count growth, not by
  the MoE**: `237`+`234` grow ×16 with batch while MoE stays flat, so the bs16 step should be
  dominated by GDN task volume + the resulting queue pressure.
* **P5 — `prepare_next_batch` < 5 % of the step at bs1, and larger (but still < 15 %) at bs16.**
* **P6 — dead MoE tasks are visible but small in wall terms**: ~19 800 of 20 480 MoE tasks at
  bs1 have no rows; at ~1 µs each over 128 workers that is ~155 µs/step ≈ 1 % — real, but not
  the headline. The headline lever will be the serialization/latency structure, not task cost.
* **P7 — closure**: Σ(per-iteration trace durations) will reconcile with the CUDA-event wall
  time to within 3 %.
* **P8 — profiling overhead** (profiled vs unprofiled step time, same kernel geometry) < 10 %.

---

## Predicted vs observed (written after the sweep)

| # | prediction | observed | verdict |
|---|---|---|---|
| P1 | per-worker busy < 25 % of the bs1 step | 5195 / 15264 µs = **34.0 %** | REFUTED (direction right, magnitude wrong) |
| P2 | all-worker-idle ≥ 20 % of the bs1 step | 501 µs = **3.3 %**, and flat at every bs | REFUTED badly |
| P3 | GDN recurrent is the largest compute task type at bs1 | 5th (37.8 ms); MoE w13 is 7× larger (266 ms) | REFUTED at bs1, CONFIRMED at bs16 (39.7 % of task time) |
| P4 | the bs8→bs16 knee comes from per-request task growth | task growth is real (GDN ×16) but the knee in the M2 wave numbers is the **admission schedule**: bs16 has 0 clean decode iterations | REFUTED as a cause |
| P5 | `prepare_next_batch` < 5 % (bs1) / < 15 % (bs16) | **0.03 % / 0.16 %** | confirmed, but over-estimated by ~100× |
| P6 | dead MoE tasks visible but ≈1 % | **0.3 %**; the headline is elsewhere | CONFIRMED |
| P7 | trace/wall closure within 3 % | **0.44–0.94 %** | CONFIRMED |
| P8 | profiling overhead < 10 % | **2.85–3.59 %** | CONFIRMED |

**What the prediction set missed entirely**, and why it matters more than what it got right:

1. **`max_num_batched_tokens=16` padding is the dominant structural cost.** Quantize, both dense
   linears and the MoE grid are all sized by `mbt`, not by live tokens, so their cost is flat from
   bs1 to bs16 and bs1 pays 16× (dense/quantize) or ~7× (MoE expert activation) for work it throws
   away. Predicting "the MoE tasks are per-expert and mostly no-ops" was the right *shape* of
   suspicion aimed at the wrong mechanism: the dead tasks are cheap, the wrongly-live ones are not.
2. **`quantize_fp8` is the single largest wall-time consumer at bs ≤ 4** (4.5 ms of a 15.3 ms
   step). Nothing in the static task-graph histogram suggested it — 3840 tasks looked minor next
   to 20 480 MoE tasks. Only the measured per-task duration exposed it.
3. **The bs16 wave never reaches a decode steady state.** The static analysis assumed prefill then
   decode; the real schedule interleaves them for 108 of 203 iterations. This was only caught by
   replaying `prepare_next_batch` after the observed iteration count (203) refused to match the
   analytic one (36 + 107 = 143), i.e. by refusing to accept a 42 % modelling error.

Method lesson: the two predictions that came from *reading the code* (P5, P7, P8) held; the three
that came from *counting tasks in the static graph* (P1, P3, P4) all failed, because task count is
uncorrelated with task cost here (0.6 µs to 60 µs across the same task type).
