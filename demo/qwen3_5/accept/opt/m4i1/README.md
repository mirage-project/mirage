# M4-I1 — the final gate, and what its first real run says

The gate itself is `demo/qwen3_5/accept/final.sh` + `final/` (design record:
`final/README.md`). This directory is the evidence from building it: the first end-to-end run of
every stage against the live B200, on 2026-07-29, at gate sha `c1d09a96`.

## Headline

**AC-3 PASSES. AC-4 and AC-5 FAIL, by 2.8× and 2.3× respectively.** That is the honest current
state — the decode levers are still in flight — and it is what the gate is supposed to say today.

```
RESULT bs=1 AC3=PASS AC4=FAIL mpk_decode_tok_s=102.4 vllm_decode_tok_s=289.0 decode_ratio=0.354
             AC5=FAIL mpk_e2e_s=10.279 vllm_e2e_s=3.559 e2e_ratio=2.888 e2e_ratio_max=1.25
```

The run was **bs1 only, and therefore NON-BINDING** (`--non-binding`, exit 2): it exists to prove
the machinery, not to produce an AC-6 verdict. See "Why this run is non-binding" below.

## What each stage did

| stage | result |
|---|---|
| integrity | **PASS**. Binding variant on a clean checkout: `run/integrity_binding_clean_tree.json`. Reads the pinned contract out of the pinned gate file, re-verifies the prompt digest (`5acabf20…`), the reference identity (`Qwen/Qwen3.5-35B-A3B-FP8` @ `9d1823d2`, 10 × 64 tokens, top-k present at every position), the exactness baseline, the pinned vLLM table, the admission policy from its authority, a clean tree, and the sha256 of all 19 tool files it executes. |
| AC-3 | **PASS**. Pre-flight probe accepted GPU 5 (3 cold reps at bs1, fingerprint divergence 0.0). Sweep: 3/3 cold reps accepted, all at state signature `f66643d43adada64` — the value M4-I0 pinned as the bs1 consensus across 5 devices. 30/30 cases byte-identical to `results/dumps_final`. Min agreement 0.9375, max perplexity ratio 1.000. |
| AC-4/AC-5 | **FAIL**, with a valid measurement on both sides (see the table below). |
| report | Emits `report.json`, `summary.txt` and one grep-able `RESULT` line per batch size. |

## The AC-4/AC-5 numbers, and why they are trustworthy

| quantity | value |
|---|---|
| MPK full arm, per rep (ms) | 10274.2 / 10281.3 / 10279.1 — full range **0.07 %** |
| MPK prefill arm, per rep (ms) | 310.4 / 310.2 / 311.2 — full range **0.33 %** |
| MPK decode (prefill-subtracted slope) | **102.42 tok/s** |
| MPK e2e | **10.279 s** |
| fresh vLLM decode, per rep (tok/s) | 289.15 / 289.01 / 288.88 → median **289.01** |
| fresh vLLM e2e, per rep (s) | 3.557 / 3.559 / 3.559 → median **3.559** |
| pinned vLLM cross-check | 285.51 tok/s / 3.602 s (`baselines/vllm-0.25.1-20260725`) |
| drift statistic (decode) | merged n=6, IQR/median **1.19 %** (bound 5 %), max boot deviation **0.61 %** (bound 3 %) → **valid** |
| drift statistic (e2e) | merged n=6, IQR/median 1.18 %, boot deviation 0.61 % → valid |
| identity cross-check | vLLM 0.25.1, same revision, 256/1024, `language_model_only=off` → **match** |
| device | GPU 3 from the process's own CUDA UUID `51c31609…`, foreign floor 131 MiB, 6/6 cells clean |
| compiled admission cap | `None` at bs1, which is what `admission_policy.py` resolves (`CAP_MIN_BATCH_SIZE=4`) |

Two independent confirmations fall out of this:

- **MPK's bs1 number reproduces M3-I7's** — 102.42 today against 102.2 in the milestone gate
  (0.2 % apart), on a different device, in a different window, through a different driver.
- **The fresh comparator agrees with the four-day-old pinned table** to 1.2 %, well inside the
  protocol's own two-boot statistic. The drift rule passes on real data rather than only on
  fixtures, which is the thing that could not be checked any other way.

The gap to close is unchanged and large: 0.354× of vLLM's decode throughput at bs1, and 2.89× its
e2e latency against a 1.25× bound.

## Why this run is non-binding

The gate was invoked with `--batch-sizes 1`. Its own integrity stage **refused that** on the first
attempt — the invocation disagreed with the pinned batch-size set, which is the anti-weakening
property the stage exists for. A bounded spot-check now has to declare itself with
`--non-binding`: deviations become notes, the report is stamped `binding: false`, and the exit is
forced to 2, so such a run can never read as an AC-6 result. A criterion that failed still reads
FAIL — which is why the AC-4/AC-5 numbers above stand as a real current-state result.

## Five defects this run found

Every one was found by running the thing, and every one is fixed and committed.

1. **`harness/gate_ac3_stable.py` could not produce a single rep at HEAD.** `cmd_rep` raised
   `NameError('cap')`: the admission-policy landing (`348a601a`) moved cap resolution into
   `admission_policy.py` and deleted the local `cap`, but the meta dict still referred to it. The
   rep died after the cold compile and the run, just before writing meta, so the stability gate
   scored zero reps. Regression test: `harness/tests/test_gate_rep_meta.py`.
2. **`_assertion` then crashed on the wreckage.** With no scored rep the divergence rate is `None`,
   and formatting it with `%`-precision raised `TypeError` — the report destroyed itself while
   trying to explain the failure. Latent since M4-I0; only reachable once defect 1 zeroed the
   scored count.
3. **The fresh vLLM collector died with `FileNotFoundError: 'ninja'`.** vLLM's inductor path shells
   out to `ninja`, a pip console script in `venv-vllm/bin`; calling the venv interpreter by
   absolute path does not put that directory on PATH, and non-interactive ssh + tmux gets no
   profile. All three collectors now prepend their interpreter's own `bin`.
4. **AC-3's perplexity stage cannot run in the MPK venv.** The MPK venvs carry transformers 4.57.1,
   which has no Qwen3.5 support at all; `venv-vllm` carries 5.14.1, the version the reference and
   the vLLM baseline were captured with. The collector takes `--hf-python` and defaults to it.
5. **`remote_setup.sh` wiped a tree that already held the right commit.** It compared the requested
   sha to `git rev-parse HEAD` as strings, so a short sha never matched. It now resolves both sides
   to object ids, checks the clone's own object db, and clones into `.new` + swaps only after a
   successful checkout.

Two more, in the gate's own code: the perf audit keyed `meta/*.json` on `"tag"` and died on the
non-cell files in that directory (losing the device floor and every per-cell `gpu_before` — the
inputs the dirty-rep rule needs, i.e. the failure most likely to go unnoticed); and the
integrity-failure early exit used `${NONBINDING:+…}`, which expands for the string `"0"`, so every
integrity failure was stamped NON-BINDING.

## Two calibration findings

- **The goal's literal repetition bound rejects the HF reference itself.** Worst 4-gram repetition
  per reference continuation: `3,1,3,2,2,3,4,3,3,3`. `p07-format` ("numbered list, each with one
  distinguishing fact") emits the markdown list-item 4-gram `(198,262,348,256)` four times, so
  "no n-gram n≥4 repeated >3×" fails the ground truth on 1 of 10 prompts. The bar is
  `max(3, the reference's own count)` — see `final/README.md`.
- **Two committed M3-I7 arms sit above the 5 % dispersion bound** (bs1 prefill 6.41 %, bs16 full
  6.22 %). Irrelevant while AC-4 fails, but a PASS needs a valid measurement, so those arms need
  more reps (the protocol's §6 escalation) before a win can be certified at those batch sizes. In
  this run's own fresh window both bs1 arms came in at 0.07 % and 0.33 %, so this looks like a
  property of that older window rather than of the geometry.

## Files

```
run/summary.txt                       the human report, verbatim
run/report.json                       the machine-readable gate report
run/integrity.json                    integrity for THIS (non-binding) invocation
run/integrity_binding_clean_tree.json integrity, BINDING, on a clean checkout — PASS
run/ac3/ac3_score.json                per-rep, per-prompt, per-position AC-3 verdicts
run/ac3/coherence_inputs.json         decoded text + real HF perplexities (10 ref + 10 engine)
run/ac3/sweep/gate_ac3_stable.json    the cold-rep fingerprint report (3/3 accepted)
run/ac3/preflight/                    the pre-flight device probe that accepted GPU 5
run/perf/perf_score.json              AC-4/AC-5 with the drift block and every per-rep value
run/perf/mpk/{full,pre}/timings_*.json  the six MPK cells
run/perf/mpk/{audit.json,meta,logs}   device identity, foreign floor, per-cell gpu_before
run/perf/vllm_fresh/                  the fresh comparator's own artifacts + its log
```

Raw per-rep fingerprints (`fp_*.npz`), token dumps and kernel directories stay on the box under
`~/mpk-qwen35/final-gate/run-m4i1-proof/`; they are large and reproducible from the committed
drivers.
