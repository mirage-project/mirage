# M3-I2a — the in-process wave reset, root-caused

M2-I9 recorded `HAZARD-WAVE-RESET`: at bs=4 the megakernel wedged on the second
in-process wave, and the suspected cause was `init_request_func` leaving some
task-graph queue state un-reset. This is what re-testing that hypothesis found.

## Verdict

**There is no in-process reset defect.** The wedge is a lost *SM residency*
precondition, which on a shared GPU looks exactly like a wave-boundary bug.

## Ledger

| # | Experiment | Result |
|---|---|---|
| 1 | Same wave twice in one process, bs=4 | pass, byte-identical |
| 2 | The exact failing wave pair (wave 0 → wave 1), bs=4 | pass |
| 3 | All three distinct waves, bs=4 | pass |
| 4 | **M2-I9's own compiled kernel + its exact command line**, 3 waves | pass |
| 5 | 62 in-process launches (30×bs4 + 20×bs8 + 12×bs16), geometry changing every launch | pass, byte-identical |
| 6 | **Positive control**: same run, GEMM co-tenant lands on the GPU mid-sequence | **wedged at wave 5, `step=[0,0,0,0]`** |
| 7 | Static read of the reset path | already complete (below) |

Experiment 4 is what makes the negative results evidence rather than absence of
evidence: the compiled `.so`, task graph, prompts and `max_seq_length` were all
the ones that wedged in M2-I9, and only the GPU changed.

The reset path was already complete: `prepare_kernel` re-zeroes every worker and
scheduler queue tail, `sched_queue_next_free_event_id`, and all `num_events`
event counters, then re-seeds `EVENT_END_OF_TASK_GRAPH`; `init_kernel` re-zeroes
`step`, `request_ids`, `qo_indptr`, `paged_kv_indptr`, the page queue and
`next_request_id`; and the runtime holds no persistent `__device__` state, so
nothing survives a launch boundary except queue *contents*, which the tail
pointers make unreachable.

## The mechanism

Workers and schedulers are spin-waiting peers — a worker blocks until a
scheduler enqueues its task, a scheduler blocks until a worker triggers its
event — and neither ever yields its SM. The whole grid must therefore be
co-resident. The launch config makes that a whole-GPU claim:
`get_configurations_from_gpu` picks 128 workers (one SM each: 205 KB of shared
memory and `__launch_bounds__(..., 1)`) and `4 * (sm_count - workers)` = 80
schedulers packed 4-per-SM into the remaining 20. On a 148-SM B200 that is every
SM, so one block of any other process breaks it.

The resulting deadlock is **self-sustaining**: in experiment 6 the co-tenant
exited and the megakernel still spun at 100% forever, because MPK's own resident
blocks never yield the SMs the missing blocks need. That is why a co-tenant
present for a few seconds burns a GPU for hours, and why the wedged context then
wedges every job that lands on that GPU afterwards.

## The fix

`launch_func` runs a residency probe before every launch: the same two grids,
the same per-block resources, on the same streams, with every block confirming
it saw the whole grid arrive while it was itself resident. A grid that had to
run in two batches can never pass. Failure raises a `RuntimeError` naming the
number of non-resident blocks instead of deadlocking; three retries absorb a
blip; `MPK_SKIP_RESIDENCY_CHECK=1` opts out.

The probe also runs at init — once to warm up, once for the verdict. The first
kernel launch in a process pays ~250 ms to load this (very large) module, which
both hid inside the CI's single timed launch (+21.9% ms/token) and made a
cold probe report every worker block as non-resident. Warm, a probe costs ~1 ms.

**Honest limit:** the probe detects *sustained* contention deterministically and
*intermittent* contention only probabilistically — a bursty co-tenant can pass
the probe and take the SMs in the gap before the real launch. The harness
progress watchdog remains the backstop.

## Regression test

`two_wave_repro.py` runs N waves in one process and asserts both that no wave
wedges and that wave *k* is byte-identical to wave 0:

    python two_wave_repro.py --batch-size 4 --waves 4 --mode cycle \
        --kernel-dir <dir> --reuse-kernel --max-seq-length 132 --out <json>

## Consequence for the AC-3 protocol

The one-wave-per-process workaround is retired; `mpk_engine_run.py` runs every
wave of every batch size in one process again. What it needs is an exclusive
GPU, which is now checked rather than assumed.

## Results on the shipped tree

Compiled fresh from this tree, exclusive B200, all five batch sizes in-process
multi-wave (10/5/3/2/1 waves at bs 1/2/4/8/16):

| Check | Result |
|---|---|
| Per-(prompt, bs) byte-diff vs committed `results/dumps_final/` | 50/50 identical (`m3i2a/bytediff.json`) |
| AC-3 report vs committed `results/run_report_all_bs.json` | no differences; same single non-exact case (p06-poem at all 5 bs, the documented adjudicated reference-side tie) |
| Two-wave regression test | 4 cycling waves pass; 3 repeated waves byte-identical |
| Backward compat, wave-per-process (`--prompt-ids`) | tokens identical to committed `dumps_final/bs4_w1.json` |
| Qwen3-8B CI | tokens identical to committed `results/ci_final.json` (but see the CI reproducibility note below) |
| Init-time residency warnings during the sweep | 0 |

Perf, Qwen3-8B CI ms/token, same binary with `MPK_SKIP_RESIDENCY_CHECK=1` as the
control arm, 4 interleaved reps each (`m3i2a/ci_ab_residency_probe.json`):

| Build | probe on (median) | control (median) | delta |
|---|---|---|---|
| probe at launch only | 4.727 | 3.879 | +21.9% |
| probe warmed at init (shipped) | 3.925 | 3.872 | +1.4% |

+1.4% sits inside the control arm's own spread (3.867-3.942), and the control
median reproduces the committed M2 baseline of 3.871 exactly.

## Fail-closed policy, and what the Qwen3-8B CI actually shows

A probe that cannot RUN must never look like a probe that PASSED — otherwise a
malfunctioning probe silently re-admits the deadlock it exists to prevent. Every
CUDA API the probe uses (`cudaMalloc`, `cudaFuncSetAttribute`, `cudaGetDevice`,
`cudaDeviceGetAttribute`, `cudaMemsetAsync`, both kernel launches via
`cudaGetLastError`, both `cudaStreamSynchronize`, `cudaMemcpy`) therefore fails
closed, returning `MPK_RESIDENCY_PROBE_ERROR` and naming the API plus its
`cudaGetErrorString`. That is terminal inside the retry loop — an API error is
never retried into a success. `MPK_SKIP_RESIDENCY_CHECK=1` is the only fail-open
path and logs a warning when used. Negative tests live in
`two_wave_repro.py --selftest-residency`: a forced `cudaMalloc` failure must
raise, and the env bypass must warn and proceed. Both pass.

**The Qwen3-8B CI decode is not run-to-run reproducible, and that predates this
work.** Runs of one fixed binary, residency probe on vs bypassed, interleaved on
one GPU, produced three distinct `generate_length` values (256 / 258 / 278), and
the single first-100-token mismatch against `results/ci_final.json` landed in
the **probe-bypassed** arm — so the check cannot be its cause. M2-I9's own logs
show the same spread (261 / 258 / 256), and the recurring first-divergence index
(71) matches the difference between the committed `ci_mpk_output.json` and
`ci_final.json`, which points at a genuine near-tie whose winner follows
reduction order. The qwen3.5 AC-3 gate — the acceptance gate that matters —
stayed byte-identical (10/10 at bs=4) in the same validation run. Treat the
Qwen3-8B CI as a smoke test, not a bit-exact gate, until that tie is chased
down.
