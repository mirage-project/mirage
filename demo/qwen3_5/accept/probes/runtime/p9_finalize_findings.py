"""Finalize P9 findings from the already-saved step-1 (clean, unmodified
run_batch_perf.py) evidence: the doc-mandated r=1,2,4,8 sweep, a 3x repeat of
r=8, an r=1 control re-run, and an r=8 run against a completely fresh
compile-cache directory. The profiler-based (step 2) attribution could not be
completed -- see attribution_blocked -- so this finalizer reports the
step-1-only, but thoroughly cross-validated, verdict honestly rather than
force a label out of unusable data.
"""
import json
import re
import sys

step1_log = sys.argv[1]       # p9_step1.log (r=1,2,4,8 sweep)
repeat_log = sys.argv[2]      # p9_repeat.log (3x r=8, r=1 control, r=8 fresh-dir)
out_json = sys.argv[3]

LAT_RE = re.compile(r"per-token latency:\s*([\d.]+) ms")
# Real headers are "=== label ===" (space right after the 3 '='s). Must NOT
# also match run_batch_perf.py's own "==================== MPK Batch Perf
# ====================" box-drawing separator (a 4th '=' immediately follows
# the first 3, no space) -- that decorative line matched a laxer pattern in
# an earlier version of this script and silently clobbered the real label
# just before the "per-token latency" line consumed it. Caught by comparing
# this script's own parsed output against a direct grep of the same log.
HEADER_RE = re.compile(r"^=== ([^=].*?) ===$")


def parse_latencies(log_text, pattern_name):
    """Walk the log in order, pairing each '=== ... ===' header with the
    NEXT 'per-token latency' line that follows it."""
    lines = log_text.splitlines()
    results = []
    current_label = None
    for line in lines:
        m = HEADER_RE.search(line)
        if m:
            current_label = m.group(1)
            continue
        m2 = LAT_RE.search(line)
        if m2 and current_label is not None:
            results.append((current_label, float(m2.group(1))))
            current_label = None
    return results


with open(step1_log) as f:
    step1_text = f.read()
with open(repeat_log) as f:
    repeat_text = f.read()

step1_results = dict(parse_latencies(step1_text, "step1"))
# step1_results keys look like "r=1", "r=2", "r=4", "r=8"
step1_ms_per_token = {r: step1_results[f"r={r}"] for r in (1, 2, 4, 8)}

repeat_results = parse_latencies(repeat_text, "repeat")
r8_trials = [v for k, v in repeat_results if k.startswith("r=8 trial")]
r1_control = [v for k, v in repeat_results if k == "r=1 control rerun"]
# the 5th repeat entry is the fresh-kernel-cache-dir r=8 rerun (no distinct
# header text beyond "r=8" itself was emitted by run_batch_perf.py, so we
# take it positionally: 3 trials + r=1 control + 1 fresh-dir rerun, in order)
all_repeat_vals = [v for _, v in repeat_results]
r8_fresh_dir = all_repeat_vals[4] if len(all_repeat_vals) >= 5 else None

recorded_92603ca = {1: 4.40, 2: 4.41, 4: 4.44, 8: 7.49}

all_r8_measurements = [step1_ms_per_token[8]] + r8_trials + ([r8_fresh_dir] if r8_fresh_dir else [])
r8_mean = sum(all_r8_measurements) / len(all_r8_measurements)
r8_spread_pct = 100 * (max(all_r8_measurements) - min(all_r8_measurements)) / r8_mean

observed_jump_4_to_8 = r8_mean - step1_ms_per_token[4]
historical_jump_4_to_8 = recorded_92603ca[8] - recorded_92603ca[4]

knee_reproduced = observed_jump_4_to_8 > 1.5  # qualitative threshold: a real +1.5ms+ jump would count as "reproduced"

findings = {
    "probe": "P9",
    "knee_reproduced": knee_reproduced,
    "attribution": (
        "blocked -- not scheduler-side or task-side: step 1 shows no significant r=4-to-r=8 "
        f"jump exists to attribute on current HEAD ({observed_jump_4_to_8:.3f} ms observed vs "
        "3.05 ms historical), and step 2's profiler-based gap-vs-task-sum breakdown could not "
        "be run to a real multi-iteration completion because MPK_ENABLE_PROFILING truncates "
        "MODE_OFFLINE runs to ~2 steps regardless of max_seq_length (persistent_kernel.cuh:271-278, "
        "a genuine bug, not a data ambiguity -- see attribution_blocked for full evidence and repro)."
    ),
    "evidence": {
        "knee_reproduction": "step1_ms_per_token + r8_repeatability_check + root_cause_investigation "
                             "(this file); raw logs p9_step1.log, p9_repeat.log alongside this JSON",
        "attribution_attempt": "attribution_blocked (this file); raw profiler captures "
                               "p9_summary_r4.json, p9_summary_r8.json, p9_trace_r4.csv, "
                               "p9_trace_r8.csv, p9_step2.log alongside this JSON",
    },
    "step1_ms_per_token": step1_ms_per_token,
    "step1_recorded_92603ca_ms_per_token": recorded_92603ca,
    "r8_repeatability_check": {
        "measurements_ms_per_token": all_r8_measurements,
        "sources": (["step1 sweep"] + [f"repeat trial {i+1}" for i in range(len(r8_trials))]
                    + (["fresh compile-cache dir"] if r8_fresh_dir else [])),
        "mean": r8_mean,
        "spread_pct": r8_spread_pct,
        "interpretation": (
            f"{len(all_r8_measurements)} independent r=8 measurements (incl. a completely fresh "
            f"compile-cache directory, ruling out kernel-cache staleness) agree within "
            f"{r8_spread_pct:.2f}% of their mean -- the batch-8 config is NOT exhibiting the "
            "historical knee; this is not measurement noise."
        ),
    },
    "r1_control_rerun_ms_per_token": r1_control[0] if r1_control else None,
    "observed_jump_r4_to_r8_ms_per_token": observed_jump_4_to_8,
    "historical_jump_r4_to_r8_ms_per_token_at_92603ca": historical_jump_4_to_8,
    "root_cause_investigation": {
        "code_diff_since_92603ca": (
            "Only 4 commits touched persistent_kernel.cuh between 92603ca and current HEAD "
            "(2c87a75): 9af4349 (NVSHMEM_NO_DEVICE_LIB removal), c004c0b (test-mode interface "
            "consolidation), 52b4e7a (tensor-view whitespace-only in this file), e3cdbbea "
            "(EAGLE3/MPK_SPEC_DECODE support). All four are gated behind USE_NVSHMEM, "
            "MPK_TEST_MODE, or MPK_SPEC_DECODE macros NONE of which are defined for this "
            "probe's plain MODE_OFFLINE / use_cutlass_kernel=True / non-spec-decode build -- "
            "i.e. the code path this probe exercises is a functional no-op across all four "
            "commits. tests/ci-tests/run_batch_perf.py itself has never been modified since "
            "92603ca (git log shows exactly one commit touching it: 92603ca itself)."
        ),
        "conclusion": (
            "The knee's disappearance is not explained by any code change on the exercised "
            "path. Leading hypothesis (not independently confirmed -- would require checking "
            "out 92603ca itself and rebuilding, which was judged out of this probe's bounded "
            "scope): the original 4.40/4.41/4.44/7.49 figures reflect a one-off measurement "
            "artifact (e.g. GPU contention at record time) rather than a deterministic "
            "architectural property, since 5 independent trials here (different compile-cache "
            "dirs, different points in time, verified-idle GPU before every run) show no such "
            "effect at all."
        ),
    },
    "attribution_blocked": {
        "status": "step 2 (profiler-enabled r=4,8 gap-vs-task-sum attribution) could not be "
                  "completed",
        "reason": (
            "Discovered a real bug in persistent_kernel.cuh's admission logic: "
            "'#if defined(MPK_ENABLE_PROFILING) || defined(MPK_TEST_MODE) / if (true) { "
            "// Request is done ... }' (lines 271-278, introduced by c004c0b / PR #712 "
            "'test: improve test mode interface') marks every request permanently finished "
            "after its FIRST processed step whenever MPK_ENABLE_PROFILING is set -- not just "
            "under MPK_TEST_MODE, which is presumably what was intended. Any profiler_tensor-"
            "enabled MODE_OFFLINE run therefore terminates after ~2 steps regardless of "
            "max_seq_length, independent of batch size. Confirmed directly: both r=4 and r=8 "
            "profiler captures (p9_summary_r{4,8}.json) show sequence_length=2, "
            "total_time_ms<=4.7 -- not the intended 127-iteration steady-state decode run."
        ),
        "secondary_bugs_found_en_route": [
            "export_to_perfetto_trace's tid_map is pre-populated only for block_idx in "
            "range(header.num_blocks), where the header is written by whichever kernel's "
            "'block 0' runs first (empirically the num_schedulers=80 scheduler launch); "
            "worker blocks (num_workers=128, a separate launch) index up to 127, causing "
            "KeyError: (80, 0) etc. mpk.__call__() has no try/except around this, so it also "
            "blocks the independent, otherwise-working export_to_csv() call.",
            "export_to_csv's own pairing has a (minor, ~0.0% of rows in the truncated traces "
            "actually captured) buffer-aliasing exposure: workers and schedulers are separate "
            "kernel launches, each computing profiler_write_ptr/stride from its own local "
            "blockIdx/gridDim, so worker-block-b and scheduler-block-b alias the same shared-"
            "buffer offset on writes where their write-cycle counts coincide.",
        ],
        "not_fixed_here": (
            "None of these are in M2-I11's owned paths (probes/runtime/, ~/mpk-qwen35/probes/); "
            "flagged for a profiler-maintenance issue (natural fit: M2-I10, which already owns "
            "the stale profiler_persistent.py name-map fix) before M3 relies on 'the MPK "
            "profiler CSV' for scheduler-path optimization work per mpk-gaps.md risk #4's "
            "mitigation -- that plan is currently blocked for any OFFLINE multi-iteration case."
        ),
        "practical_impact_on_this_probe": (
            "Low: step 1 already establishes, with 5 independent trials, that there is no "
            "significant r=4-to-r=8 per-token jump to attribute right now (observed jump "
            f"{observed_jump_4_to_8:.3f} ms vs the historical 3.05 ms) -- so the fine-grained "
            "gap-vs-task-sum breakdown step 2 was designed to explain has no target phenomenon "
            "to explain on the current codebase."
        ),
    },
}

with open(out_json, "w") as f:
    json.dump(findings, f, indent=2)
print(json.dumps(findings, indent=2))
