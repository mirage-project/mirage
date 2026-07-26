# P9 — batch-8 scheduler-knee attribution

Per v1-architecture.md S14 P9: reproduce the recorded batch-scaling knee
(4.40/4.41/4.44/**7.49** ms/token at r=1/2/4/8, commit `92603ca`), then
attribute it via the MPK profiler (inter-iteration gap vs. summed task time).
Owner: M2-I11.

## Step 1 — reproduction: the knee is GONE, robustly

Ran the doc's exact, unmodified command (`tests/ci-tests/run_batch_perf.py
--max-num-batched-requests {1,2,4,8} --ignore-eos`, `venv-mpk`,
`HF_HOME=/raid/catalyst/models`, GPU 7 verified idle before every run):

| r | this run (ms/tok) | recorded @ 92603ca (ms/tok) |
|---|---|---|
| 1 | 4.434 | 4.40 |
| 2 | 4.430 | 4.41 |
| 4 | 4.468 | 4.44 |
| 8 | 4.507 | **7.49** |

No knee. Batch 8 costs ~1.6% more than batch 1, not +69%. Before trusting a
surprising non-result, this was checked three ways, not just re-run once:

1. **Repeatability** — r=8 re-run 3 more times, plus an r=1 control, plus an
   r=8 run against a brand-new compile-cache directory (rules out kernel-cache
   staleness). All 5 independent r=8 measurements: 4.507, 4.513, 4.515, 4.526,
   4.531 — **0.53% spread**. Not noise.
2. **Code-diff review** — every commit touching
   `include/mirage/persistent_kernel/persistent_kernel.cuh` between `92603ca`
   and current HEAD (`2c87a75`) was read in full: `9af4349` (NVSHMEM_NO_DEVICE_LIB
   removal), `c004c0b` (test-mode consolidation), `52b4e7a` (unrelated
   whitespace in this file), `e3cdbbea` (EAGLE3/`MPK_SPEC_DECODE`). Every
   changed line is gated behind `USE_NVSHMEM`, `MPK_TEST_MODE`, or
   `MPK_SPEC_DECODE` — none defined for this probe's plain
   `MODE_OFFLINE`/`use_cutlass_kernel=True` build. `tests/ci-tests/run_batch_perf.py`
   itself has never been touched since the commit that introduced it. The
   exercised code path is a functional no-op across all four commits.
3. **GPU exclusivity** — `nvidia-smi` checked clean (0% util, <10MiB) before
   every run; no contending processes.

**Conclusion:** the knee's disappearance isn't explained by any change on the
path this probe exercises, and it isn't measurement noise. The leading
hypothesis — not independently proven, see "what wasn't done" below — is that
the original 92603ca figures reflect a one-off measurement artifact (e.g. GPU
contention at record time) rather than a deterministic property of the code.
**This is a genuine reversal of the doc's premise**, not a confirmation of
either of its anticipated outcomes (scheduler-side vs. task-side); it belongs
in memory, per the WRAP-UP trigger for "P9 overturns the hypothesis."

*What wasn't done, and why:* checking out `92603ca` itself and rebuilding to
see whether it reproduces 7.49 ms/token on this exact hardware would
distinguish "historical measurement artifact" from "something not visible in
the .cuh diff" — that needs a fresh MPK build (not just a megakernel
recompile) and was judged out of this probe's bounded scope; flagged for the
coordinator rather than attempted unilaterally.

## Step 2 — profiler attribution: blocked by a real infrastructure bug

Built `probes/runtime/p9_profile_capture.py` (adapted from
`run_batch_perf.py`, `profiler_tensor` attached per `persistent_kernel.py`'s
plumbing) to capture r=4 and r=8 with MPK's own instrumentation on, then
attribute the (now-absent) jump via inter-iteration gap vs. summed task time,
anchored on `TASK_SCHD_PREPARE_BATCH` (raw numeric id **204**, cross-checked
directly against `runtime_header.h`'s `enum TaskType` — not the name map,
which mpk-gaps.md already flags stale, and which this investigation
independently reconfirmed is stale in at least one more spot:
`TASK_SM100_TASK_END` is 298 in `profiler_persistent.py` but 299 in the
canonical header).

Three real bugs surfaced while building this, in order of how they were hit:

1. **`export_to_perfetto_trace` KeyError.** `mpk.__call__()` unconditionally
   runs `export_to_perfetto_trace()` before `export_to_csv()` whenever
   `profiler_tensor` is set, with no `try/except` around either. The
   perfetto exporter's `tid_map` is pre-populated only for
   `block_idx in range(header.num_blocks)`, and the header is written by
   whichever kernel's "block 0" gets there first — empirically the
   `num_schedulers=80` scheduler launch — while worker blocks
   (`num_workers=128`, a *separate* kernel launch) index up to 127.
   `KeyError: (80, 0)` etc. This also blocks the independent, otherwise-fine
   `export_to_csv()` call from ever running. **Workaround** (in our own
   script only): monkey-patch both export functions to no-ops before calling
   `mpk()`, then save the raw device buffer ourselves.
2. **`export_to_csv` buffer aliasing.** Root cause of (1): workers and
   schedulers are separate kernel launches, each computing
   `profiler_write_ptr`/`stride` from its own local `blockIdx`/`gridDim`
   (`profiler.h`'s `PROFILER_INIT`). Worker-block-*b* and scheduler-block-*b*
   therefore write their first event to the same shared-buffer offset
   (`1+b`), colliding whenever their write-cycle counts later coincide too.
   In practice minor here (2 rows out of ~22k dropped by a tolerant
   re-implementation of the pairing logic, `p9_decode_tolerant.py`, built
   once neutralizing the crash-prone exports revealed this).
3. **The actual blocker: `MPK_ENABLE_PROFILING` silently truncates
   `MODE_OFFLINE` runs to ~2 steps.** `persistent_kernel.cuh:271-278`
   (introduced by `c004c0b`, PR #712 "test: improve test mode interface"):
   ```c
   #if defined(MPK_ENABLE_PROFILING) || defined(MPK_TEST_MODE)
         if (true)
   #else
         if ((step + step_advance + 1 >= config.max_seq_length) || (EOS check))
   #endif
         { // Request is done -- free its pages, drop it from request_ids
   ```
   This marks *every* request permanently finished after its first processed
   step whenever `MPK_ENABLE_PROFILING` is defined — not just under
   `MPK_TEST_MODE`, which is presumably what was actually intended (the PR's
   own commit message only discusses test mode). Confirmed directly:
   `p9_summary_r{4,8}.json` both show `"sequence_length": 2,
   "total_time_ms" ~4.6-4.7` — the profiled runs terminated almost
   immediately regardless of `--max-seq-length 128`, independent of batch
   size. There is no way to decouple "I want profiling instrumentation" from
   "I want this always-finalize shortcut" from the Python calling side —
   `MPK_ENABLE_PROFILING` is a single flag `persistent_kernel.py` adds
   automatically whenever `profiler_tensor is not None`, with no override
   hook. A source fix (presumably narrowing the `#if` to `MPK_TEST_MODE`
   alone) is needed before any OFFLINE-mode multi-iteration profiler-based
   analysis is possible.

**None of these are in M2-I11's owned paths**
(`workspace/demo/qwen3_5/accept/probes/runtime/`, `~/mpk-qwen35/probes/`), so
none were fixed here. They're flagged for a profiler-maintenance issue — a
natural fit for M2-I10, which already owns the stale `profiler_persistent.py`
name-map fix — **before M3 relies on "the MPK profiler CSV"** for
scheduler-path optimization, per mpk-gaps.md risk #4's stated mitigation.
That plan is currently blocked for any `MODE_OFFLINE` multi-iteration case.

**Practical impact on this probe: low.** Step 1 already establishes, with 5
independent trials, that there is no significant r=4-to-r=8 per-token jump on
the current codebase (observed 0.050 ms vs. the historical 3.05 ms) — so
step 2's fine-grained gap-vs-task-sum breakdown has no target phenomenon left
to explain. `attribution` in the findings JSON is honestly reported as
blocked, not forced into "scheduler" or "task" from unusable 2-step data.

## Result

`knee_reproduced: false`. `attribution: blocked` (infrastructure bug, not a
data ambiguity). §3.3's keep-lifecycle-out ruling is **not falsified** by
this (no evidence either way on attribution), but the risk-#4 framing itself
("scheduler-section work is the top M3 item") needs re-checking against a
current, non-knee baseline before M3 prioritizes it — and the profiler bug
above must land first regardless, since M3's own plan depends on it working.

## Artifacts

- `probes/runtime/p9_profile_capture.py` — profiler-enabled capture script
  (also at `~/mpk-qwen35/probes/` on B200).
- `probes/runtime/p9_decode_tolerant.py` — tolerant raw-buffer decoder
  (works around bugs 1-2 above; not needed once bug 3 is fixed and a real
  multi-iteration trace exists, but reusable for any future profiler capture).
- `probes/runtime/p9_attribution_analysis.py` — CPU-only gap/task-sum
  clustering (anchored on raw numeric id 204; unit-tested against a synthetic
  trace incl. a genuine 32-bit timestamp-wraparound case before being run
  against real data — see its module docstring).
- `probes/runtime/p9_finalize_findings.py` — deterministic findings
  derivation from the saved logs (no new measurement).
- `probes/runtime/p9_findings.json` — the machine-checkable verdict.
- `probes/runtime/p9_step1.log`, `p9_repeat.log`, `p9_step2.log` — full raw
  logs for every run referenced above.
