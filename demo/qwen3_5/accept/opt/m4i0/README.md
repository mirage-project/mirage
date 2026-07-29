# M4-I0 — the fingerprint-scored cold-run AC-3 stability gate

The deliverable is `harness/gate_ac3_stable.sh` (+ `harness/gate_ac3_stable.py`), destined for
M4's `final.sh`. This file reports what it does and what running it 100 times measured.

**Headline.** At the AC-3 geometry, a single COLD rep's KV/GDN trajectory diverges on
**4.2% of reps** (4/96 scored, 95% upper bound 9.3%) and the tokens follow it **half the
time** (2/96 = 2.1%, upper bound 6.4%). So a one-rep AC-3 token gate is not a coin flip,
but it does produce a wrong answer at a rate around 2% — and two of the four divergences
had a token md5 **identical to the committed baseline**, i.e. a token-scored gate would have
called them clean. Three independent devices reached the gate's PASS condition — 3 cold reps
per batch size, mutually bit-identical in state and byte-identical to `results/dumps_final`
per case — on 3 reps each; one device needed 5 reps at bs2 after quarantining two
divergences; one device (GPU0) could not, and produced a reproducible wrong trajectory.

## The gate

    CUDA_VISIBLE_DEVICES=<one idle device> bash harness/gate_ac3_stable.sh --out DIR [--reps 3]

Per batch size in {1,2,4,8,16}: launch `--reps` independent reps, each a **separate process**
with its **own freshly compiled kernel** (the COLD class — campaign 2's prone class), each
snapshotting the KV/GDN wave-boundary fingerprint next to its token dump. PASS requires both:

- **(a)** every rep's token ids byte-identical, per case, to `results/dumps_final/bs<N>.json`
  — AC-3 itself, no tolerance, unrelaxed; and
- **(b)** `--reps` reps per bs whose fingerprints are identical to each other, key for key.

A rep whose fingerprint deviates from the per-bs consensus is **quarantined** — kept in the
record, replaced by an extra rep, and counted in the reported divergence rate. Verdicts:
`STABLE` (0), `FAIL` (1, a token mismatch), `UNSTABLE` (2, could not reach `--reps`
consistent reps inside the budget), integrity error (3). The report
(`gate_ac3_stable.json`) carries per-rep fingerprint signatures, devices, per-case token
verdicts, quarantine counts and the reps needed per bs.

Detector: `bitfp` over `k_cache`/`v_cache` keeping `(layer, page*page_size+offset)` and over
`conv_state`/`recurrent_state` keeping `(layer, slot)` — a behaviour-identical copy of
`opt/m3i11/scripts/e2_fingerprint.py`, the detector campaign 2 calibrated. Reasons it is
required rather than nice-to-have are in `docs/qwen35/bench-protocol.md`, "Determinism
protocol v2".

Scorer unit tests (no GPU): `scripts/test_gate_scorer.py` — 7 synthetic cases covering clean,
quarantine-and-replace, token mismatch, under-quorum, run error, lost rep and truncated
artifacts.

## What 100 cold reps measured

2026-07-29, `fa24a421` + the two gate files, isolated clone with its own freshly built
extension, 6 windows over 5 physical devices. Every rep records the device from its **own**
CUDA context UUID, not from the candidate list.

| window | GPU | bs | verdict | launched | scored | accepted | quarantined | errors | token mismatch | fp div rate |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| gateA | 6 | all 5 | **STABLE** | 16 | 15 | 15 | 0 | 1 | 0 | 0.0% |
| gateC | 1 | all 5 | **STABLE** | 17 | 17 | 15 | 2 | 0 | 0 | 11.8% |
| gateD | 6 | all 5 | **STABLE** | 15 | 15 | 15 | 0 | 0 | 0 | 0.0% |
| gateE | 7 | all 5 | **STABLE** | 15 | 15 | 15 | 0 | 0 | 0 | 0.0% |
| gateB | 0 | 1,2 | **FAIL** | 8 | 6 | 4 | 2 | 2 | 2 | 33.3% |
| cens5 | 5 | 4 only | census | 29 | 28 | 28 | 0 | 1 | 0 | 0.0% |
| **pooled** | | | | **100** | **96** | **92** | **4** | **4** | **2** | **4.17%** |

- fingerprint divergence **4/96 = 4.17%** (Clopper-Pearson 95% upper bound **9.28%**)
- token divergence **2/96 = 2.08%** (95% upper bound **6.41%**)
- per device: GPU0 **2/6**, GPU1 **2/17**, GPU5 **0/28**, GPU6 **0/30**, GPU7 **0/15**
- reps needed to reach the verdict: **3** at 18 of the 21 (window, bs) pairs that reached one;
  **4** at gateA bs16 (a run error) and gateB bs1; **5** at gateC bs2 (two quarantines).
- 9 of 96 reps started on a device that was not at its quietest observed level, and are
  flagged `device_not_clean_at_start` in the reports.

`cens5` is a **census**, not a gate run: it was invoked with `--reps 30 --max-extra 0` to
force exactly 30 reps at bs4, so its `UNSTABLE` verdict only means 28 < 30 reps survived
(one ENOSPC error, one lost rep) — it had **zero** divergences.

**The per-bs consensus fingerprint is identical across every window and every device**:
bs1 `f66643d43adada64`, bs2 `0449b5655fa57c9b`, bs4 `68e7a9fb004338df`,
bs8 `1e305b6d61e9e263`, bs16 `c91b76a10b2430eb`. The accepted state is therefore a
device-independent invariant of the code, not an artifact of majority voting inside one
window — see the recommendation below.

The AC-3 verdict profile of an accepted rep tree is **identical to the committed baseline**
(`results/ac3_verdict_profile.txt`): the same 5 pre-existing `p06-poem @ pos 60` exact-tie
waiver records, `margin=0.0`, `ref_top1=31000` vs `engine=40581`, at every batch size.

## The four divergence events

| event | GPU | bs | waves touched | KV entries | tokens |
|---|---|---|---|---|---|
| gateC bs2_r1 | 1 | 2 | w4 | 272/20480 (1.3%) | **clean, md5 = baseline** |
| gateC bs2_r2 | 1 | 2 | w2 | 470/20480 (2.3%) | **clean, md5 = baseline** |
| gateB bs1_r3 | 0 | 1 | w2, w4 | 336 and 988/20480 | `p07-format` @ 37, 27/64 differ |
| gateB bs2_r4 | 0 | 2 | w2, w3 | 766 and 252/20480 | `p07-format` @ 37, 27/64 differ |

Two findings in that table beyond the rate.

**The gateC pair is the case the gate exists for.** Both reps' GDN conv/recurrent state was
18-55% perturbed and 1.3-2.3% of KV entries differed, and both emitted a token dump whose md5
**is the baseline md5**. Scored by tokens they are indistinguishable from a clean run.
Campaign 2's central claim reproduces at the AC-3 geometry.

**The GPU0 pair is reproducible, not random.** Two events, at two different batch sizes, both
diverged on the same prompt at the same position with the same count (`p07-format`, position
37, 27 of 64 tokens) and the same dump md5 `3d1018c081314363f550803fe6e8f636`. A random race
would not land twice on the same trajectory. Whatever state GPU0 was in produced a
*repeatable* wrong answer at ~1 rep in 3, which is a much better handle for M4-I0's
root-cause line than campaign 2's one-off events — the paired-census discriminator in
`.pm/issues/M4/M4-I0.md` can now be run against a signature it can check for, not just a rate.

## Sub-argmax: refuted as a general rule, at both geometries

Campaign 2's fix arm (6 state-divergent, 2 token-divergent) suggested divergence is usually
sub-argmax. At the AC-3 geometry it is **50/50**: 2 of 4 divergences reached the token ids.
The split is not random across devices — GPU1's two events stayed sub-argmax and GPU0's two
both flipped tokens, and GPU0's were the severe ones (every conv/rec entry perturbed in
gateB bs1_r3 versus 18% in gateC bs2_r1). Severity, not geometry, decides whether a
divergence surfaces.

So the honest statement is **not** "the token result stays 50/50 when trajectories diverge".
It is: *a divergence severe enough to perturb the whole GDN state moves the tokens; a mild one
usually does not; both classes occur at the AC-3 geometry, roughly equally often.* Any gate
that assumes divergence is invisible at the token level is wrong about half the time.

**Positive control**, same window, same clone, campaign 2's census geometry (`e4_full.py`,
bs4, msl 1280, 1024 new tokens, cold per rep) — this rules out "the box was simply clean
tonight" as the explanation for the null windows:

- GPU7: **1/13 usable reps (7.7%)** divergent — 19 reps attempted, 6 died in the disk-full
  window. That rep perturbed 39% of KV entries and 100% of GDN state
  in wave 1, and flipped 4 of 10 prompts' tokens, with first divergent token positions
  **636, 645, 648, 731** (`results/poscontrol_gpu7_1024tok.txt`).
- GPU2: **0/11 usable** (15 attempted, 4 died the same way; the arm was stopped when a
  foreign 25 GB job joined GPU2).
- pooled **1/24 usable = 4.2%**, against campaign 2's 10-16% on its prone devices.

Two notes on that. The 1024-token rate tonight (4.2%) and the AC-3-geometry rate (4.2%) came
out the same, so on this evidence the *rate* is a property of the device state rather than of
the decode length. But the *first divergent token position* in the 1024-token event was
636-731 — far past AC-3's 64 decoded positions — which is why a shorter workload sees the
same trajectory divergence surface in the tokens less often, not less frequently overall.

## What M4's gate can honestly assert

With `--reps 3` on a device that passes, the defensible claim is:

> At every batch size in {1,2,4,8,16}, three independent cold-compiled reps produced KV/GDN
> state that is bit-identical across reps and token ids byte-identical to
> `results/dumps_final` per case. Fingerprint divergence was observed on X% of reps in this
> run; every divergent rep was quarantined, re-run, and retained in the report.

What it must **not** assert: that any single cold run of AC-3 is reproducible. It is not, at
~4% per rep for the trajectory and ~2% for the tokens.

**A deterministic exit 0 is not available today**, and the reason is specific: the gate's
condition (a) is over *every* rep, so a device in the bad state fails it — gateB exited 1 with
correct code and a correct baseline. Two honest options for `final.sh`, both of which are
coordinator calls, not this issue's to make:

1. **Keep condition (a) over every rep** (what is implemented). Expected false-FAIL rate
   ≈ 2% per gate run on a prone device, 0% on the three clean devices measured. `final.sh`
   then needs a documented re-run policy for a FAIL whose divergent rep is
   fingerprint-flagged, or M4's gate is red about 1 run in 50.
2. **Scope condition (a) to the accepted reps**, treating a fingerprint-flagged rep as an
   environment event rather than evidence about the code, and reporting its token mismatch in
   the divergence record. This is *not* a token-equality relaxation — the bar stays exact for
   every certified rep and the quarantine criterion is measured upstream of and independently
   of the tokens, so it can never be selected on the token outcome. It is still a change in
   what the gate asserts, and per this issue's constraints it is **reported, not
   implemented**. `git log` for this file will show no such flag.

**Recommended regardless of that choice: pin the expected per-bs fingerprint.** The consensus
signature was identical across 5 devices and 6 windows, so `final.sh` can compare against the
five pinned values above instead of a within-window majority. That removes the one structural
weakness of consensus voting — if the first reps of a window all diverged the same way, the
majority would be wrong — and turns the state check into an absolute assertion.

## Caveats

- **The denominator is a lower bound.** The shared `/raid` pool hit 0 bytes twice mid-campaign
  (it is 28T shared and swung 23G → 0 → 310G within the hour). Two reps — gateA `bs16_r3`,
  cens5 `bs4_r13` — could not create their own directory and left no artifacts at all, so they
  are missing from the 100 rather than counted as errors; they are visible only as gaps in the
  rep numbering. The launch-ledger (`launched.txt`) that makes such a rep report as `LOST` was
  added to the gate in response, and `scripts/test_gate_scorer.py` covers it.
- **gateB is incomplete.** It was stopped at bs4 when a 142 GB foreign job landed on GPU0.
  Its bs1/bs2 results stand; it never ran bs4/bs8/bs16. Starting MPK next to live foreign work
  is a deadlock (SM-residency law), so the gate now refuses the rep and records it instead of
  proceeding after a drain timeout.
- **Script provenance.** The gate was hardened during the campaign, so the windows did not all
  run byte-identical drivers: gateA/gateB/cens5 ran the first version, gateC the drain-gate
  fix, gateD the launch ledger, gateE the committed version. Each window's `run_meta.json`
  records the `sha256` it ran. All changes were to *when a rep is allowed to start* and to
  *report robustness*; the measurement path (`rep`) and the scoring criteria are unchanged
  across all of them, and every window was re-scored with the final scorer for the table
  above. gateE's `gate_sh_sha256` matches the committed `gate_ac3_stable.sh`. As a final
  check the exact committed bytes of both files were deployed and run (`verify`, GPU1,
  bs16, 2 reps): **STABLE**, ledger written, `state_sig=c91b76a10b2430eb` matching the
  pinned bs16 consensus, and the `run_meta.json` sha256 pair equal to commit `b0cd73f0`.
- The 4 events are few. The rate's confidence interval is wide (95% upper bounds 9.3% state,
  6.4% token) and per-device rates rest on 6-30 reps each.
- Cap policy: these runs used no `--per-request-token-cap`, matching how `dumps_final` was
  produced. M3-I7 measured `auto` bit-transparent at bs4/8/16; the flag is a passthrough.

## Files

- `harness/gate_ac3_stable.sh`, `harness/gate_ac3_stable.py` — the deliverable.
- `results/{gateA,gateB,gateC,gateD,gateE,cens5}.json` + `*_run_meta.json` — per-window
  reports (per-rep fingerprints, devices, token verdicts) and their provenance.
- `results/pooled.json` — the pooled table, per-bs consensus signatures, the 4 events.
- `results/poscontrol_gpu{2,7}_1024tok.txt` — the 1024-token positive control.
- `results/ac3_verdict_profile.txt` — the AC-3 harness verdict profile of an accepted tree.
- `scripts/` — the B200 drivers (`setup_m4i0.sh`, `run_gate_m4i0.sh`,
  `run_poscontrol_m4i0.sh`), the analyses (`pool_m4i0.py`, `pc_analyze.py`) and the scorer
  unit tests (`test_gate_scorer.py`).
- Raw reps (dumps, `fp_*.npz`, logs): `~/mpk-qwen35/m4i0/` on catalyst-B200.
