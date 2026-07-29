# M4-I1 — the mechanical final gate

`.pm/accept.sh` (pinned) execs `demo/qwen3_5/accept/final.sh`. That script is what AC-6 means by
"the workspace harness at a fixed path": it enforces AC-3, AC-4 and AC-5 mechanically and exits 0
only when they hold.

```bash
bash .pm/accept.sh                        # the real gate, from the agent repo root
bash demo/qwen3_5/accept/final.sh --self-test    # GPU-free: every scorer + fixtures
bash demo/qwen3_5/accept/final.sh --rescore DIR  # re-score a run dir; always exits 2
```

Exit codes, which is the part `accept.sh` and `verify.py` depend on:

| code | meaning |
|---|---|
| 0 | every criterion in scope PASSED |
| 1 | a criterion FAILED (named, with numbers) or an integrity violation |
| 3 | NOT-APPLICABLE: a prerequisite genuinely could not run **and nothing failed** |
| 2 | usage error, or a deliberately non-binding invocation (`--rescore`) |

Precedence is fail-first: `FAIL` outranks `NOT_EVALUABLE`, so no criterion can be turned from red
into "not applicable" by breaking its own measurement.

## Stages and artifacts

| stage | does | writes |
|---|---|---|
| `integrity` | re-reads the pinned contract from `.pm/accept.sh`, verifies the prompt digest, the reference artifact, the exactness baseline, the pinned vLLM table, the clean tree, and digests every tool it will execute | `integrity.json` |
| `ac3` | pre-flight device probe → cold sweep (`harness/gate_ac3_stable.sh`) → HF text+perplexity (`hf_score.py`) → `score_ac3.py` | `ac3/{sweep,coherence_inputs.json,ac3_score.json}` |
| `perf` | MPK full+pre arms (`collect_perf.sh`) + a fresh vLLM sweep (`collect_vllm.sh`) → `score_perf.py` | `perf/{mpk,vllm_fresh,perf_score.json}` |
| `report` | assembles everything, prints per-rep numbers and the one claim the run is entitled to make, decides the exit code | `report.json`, `summary.txt` |

The GPU work runs on the B200 (`--run-mode remote`, the default): `remote_setup.sh` makes an
isolated clone **at the gate's own commit sha** with its **own freshly built extension** — a stale
editable `.so` fakes a total failure at every batch size (`.memory/main/b200-env.md`,
STALE-EXTENSION TRAP), and a gate that measured that would report a false FAIL. Collect stages run
under `tmux` and are polled; `nohup` through a one-shot ssh does not reliably survive here
(bench-protocol.md §9). `--run-mode local` runs in place when the gate is invoked on the GPU box.

## AC-3, as re-pinned 2026-07-29

Scored per (rep, batch size, prompt) by `score_ac3.py` + the pure primitives in `ac3_criteria.py`:

- **(a) coherence** — no n-gram `n>=4` repeated more than the bar; no more non-language characters
  than the pinned reference continuation for that prompt; HF-reference perplexity within 1.5× of
  the reference continuation's own on the same prompt.
- **(b) floor** — ≥90 % of the 64 positions match the reference top-1, and every differing position
  is accounted for.
- **(c) diagnostic** — bit-exactness against `results/dumps_final` measured and reported every run;
  degradations listed, and marked unexplained unless a mechanism entry covers them.

Three implementation decisions worth arguing with:

**Every rep is scored, not just the fingerprint-consensus ones.** The M4 gate policy
(bench-protocol.md) argued the strict form because scoping the assertion to reps that survive
quarantine is indistinguishable from hiding a bug. Under byte-identity that strictness cost a ~2 %
false-FAIL rate. Under the re-pinned AC-3 it costs nothing: the engine's 2.08 %/cold-rep token
divergence moves a few positions, which leaves agreement far above 90 % and coherence untouched. So
the strict form stays, and the quarantine machinery is used only for the reported divergence rate.

**The greedy cascade is an accounted class, gated on its root.** The committed state diverges from
the reference at `p06-poem` positions 60–63 at every batch size: position 60 is an exact reference
tie (margin 0.0, the case M2/M3 adjudicated) and 61–63 are its cascade — once the engine emits a
different token, greedy decode conditions on a prefix the reference never had, so the reference's
top-1 there is not a prediction about this sequence (`docs/qwen35/mpk-gaps.md` §6.5, the same
argument `harness/ac3_runner.py` already encodes as `POST_DIVERGENCE`). A position may therefore be
classified `post_divergence_cascade` **only if the sequence's FIRST divergence is itself accounted
for**. An unexplained first divergence excuses nothing downstream, and the 90 % floor is computed on
raw matches and is not waivable by any class — which is what actually protects against corruption,
since every real defect this project found (router row cap, argmax tie order, quantize redundancy,
the fp8 scale floor) diverged early and massively.

**The repetition bar carries a reference term, because the literal bound rejects the reference.**
Measured on the committed reference (10 prompts × 64 tokens), the HF reference's own worst 4-gram
repetition is `3,1,3,2,2,3,4,3,3,3` — the 4-gram `(198,262,348,256)` is a markdown list-item prefix
and `p07-format` ("numbered list, each with one distinguishing fact") legitimately emits it 4 times.
The goal's literal "repeated >3× fails" therefore fails the HF reference itself on 1 of 10 prompts.
The bar is `max(3, the reference continuation's own worst count)`: the engine may never be *more*
repetitive than the pinned reference on the same prompt, and never worse than the goal's absolute 3.
This is the same reference-derived construction AC-3(a) already uses for perplexity and byte soup,
the reference is immutable, and `tests/test_ac3_criteria.py` pins the finding so the decision stays
data-driven.

Engine-side logit margins (AC-3(b)'s "≤3 bf16 ULPs" route) are **not** collected by the default
path: `expose_logits` changes the compiled graph, and the gate must score the shipping
configuration. They are accepted as an optional input (`final/engine_margins.json`) produced by a
separate documented probe (`opt/m3i9b/probe_chunk_numerics.py`). Their absence waives nothing.
`mechanisms.json` ships empty — an entry must name one exact position (prompt, position, explicit
batch-size list, exact reference and engine ids, a written mechanism), and every applied entry is
printed in the report.

## AC-4 comparator: fresh vLLM is primary, the pinned table is the cross-check

**Decision: a fresh `bench_vllm.py --mode sweep` at the binding config, captured in the same window
as the MPK runs, is the binding comparator. The committed
`baselines/vllm-0.25.1-20260725/` table is the cross-check.**

Why that way round:

1. AC-4 says "measured on the same B200 with no other process on the GPU". The pinned table was
   captured on 2026-07-25 with GPUs 0/1/4 fully loaded by other tenants (bench-protocol.md §6's
   escalation note is about exactly that) — conditions this gate forbids for its own runs. Only a
   same-window capture can satisfy AC-4's own sentence.
2. The ratio is what the criterion is about. A fresh MPK number over a four-day-old vLLM number puts
   box drift (clocks, thermals, driver) into the ratio; a same-window pair cancels it.
3. It cannot be steered. `bench_vllm.py` refuses any deviation from the pinned contract
   (`enforce_binding_contract`), the `language_model_only` value is read out of the pinned baseline
   rather than chosen, and `score_perf.py` re-checks the fresh run's fp8/fairness assertions,
   model identity, workload and rep counts. Making the pinned table primary would instead let a
   stale favourable number stand even if vLLM got faster on the current box.

**Drift rule.** The fresh capture is treated as one more *boot* of the same protocol, and the
protocol's own two-boot statistic decides whether the two agree: merge all boots' reps and require
merged IQR/median ≤ 5 %, every boot median within 3 % of the merged median, ≥6 reps total. Those
bounds are not typed in here — `score_perf.py` imports them from `bench_vllm.py`
(`BINDING_MAX_DISPERSION_PCT`, `BINDING_BOOT_MEDIAN_AGREE_PCT`), so the gate can never run on a
looser bound than the benchmark tool. If the statistic fails, the gate emits
`COMPARATOR DRIFT (decode|e2e)` with **both** medians and **FAILS** (exit 1). There is no code path
that picks the more favourable number. The remedy is a human decision: re-capture and re-pin the
baseline, or root-cause the box. A different vLLM version, revision, workload or
`language_model_only` value is a separate, harder failure (`BASELINE IDENTITY DRIFT`) — that is a
different comparator, not a noisier measurement of the same one.

**Validity asymmetry.** A criterion can only be PASSED on a measurement that satisfies the
protocol's dispersion bound; if the point estimate passes but validity is missing the verdict is
NOT_EVALUABLE. A failing point estimate FAILS regardless — a losing candidate needs no measurement
defence. So degrading the measurement can never turn a FAIL into a PASS.

**Dispersion on the MPK side.** The protocol pins the 5 % bound for the vLLM sweep; this gate
applies the same bound to the MPK arms because both sides are being compared on the same quantity,
with the protocol's own escalation statistic once an arm is big enough to use it: full-range/median
for a single-run set, IQR/median once an arm has ≥6 reps (range is monotone in rep count — protocol
§6, corrected 2026-07-25). The documented remedy for an arm over the bound is therefore to *add*
reps, never to drop one. Two arms in the committed M3-I7 evidence sit above it (bs1 `pre` 6.41 %,
bs16 `full` 6.22 %) — irrelevant while AC-4 is failing, but they must be brought under the bound
before a win can be certified at those batch sizes.

MPK's decode number is the prefill-subtracted slope `bs*(D_full − D_pre)/(wall_full − wall_pre)`
(bench-protocol.md, "Decode-throughput measurement (M3-I7, binding)"), and `score_perf.py` asserts
the geometry the slope depends on rather than assuming it: one wave per run, `max_seq_length`
1280/259, `mbt` 16, `batch_size` distinct prompts in the wave, and — per rep, out of the run's
own artifact — the COMPILED admission cap equal to what `accept/admission_policy.py` resolves for
that batch size. The cap is a compile-time define, so a cell compiled off-policy measured a
different binary and is excluded with its reason rather than averaged in; the same goes for a
multi-wave run. The policy itself is never restated here: `score_perf.py`, `collect_perf.sh` and
`integrity.py` all derive it from that module (which currently caps `auto` from bs4 up), so the
divergence bug it was created to kill cannot come back through the gate.

One consequence worth stating: with the policy default, the AC-3 cold sweep now runs CAPPED at
bs4/bs8/bs16, while `results/dumps_final` was captured uncapped. M3-I7 measured the cap
bit-transparent at exactly those batch sizes (10/10 byte-identical, `gates/bytediff_{cap,bs4,bs8}.json`),
so the AC-3(c) exactness diagnostic remains a valid comparison — and if that ever stops being true,
the diagnostic reports it as a degradation instead of hiding it.

## Self-test

`final.sh --self-test` needs no GPU and no model. It runs 44 tests across the four scorers plus the
reused machinery (`opt/m4i0/scripts/test_gate_scorer.py`, `harness/tests/*`). The fixtures are
committed artifacts wherever real data exists:

- `score_perf.py` is driven by **M3-I7's own per-rep timings** (`opt/m3i7/raw_meta/perf/…`)
  assembled at the current cap policy against the committed vLLM table. It must reproduce M3-I7's
  published decode numbers (102.2 / 200.4 / 395.5 / 746.9 / 1342.3 tok/s) and must FAIL AC-4 — the
  true current state.
- `score_ac3.py` is driven by the real `results/dumps_final` tokens, the real reference (real top-k
  margins, real decoded text) and a real `opt/m4i0/results/gateA.json` gate report. Only the
  perplexity numbers are injected, because they need the model.
- Deliberate must-fail scenarios: unexplained first divergence, below-floor agreement, degenerate
  repetition, perplexity > 1.5×, byte soup, wrong generated length, comparator drift, identity
  drift, winning-but-invalid, multi-wave contamination, cap-policy mismatch, batch sizes outside the
  pinned set.

## Files

| file | role |
|---|---|
| `../final.sh` | the gate: argument contract, stage sequencing, remote deploy, exit codes |
| `integrity.py` | stage 0; re-reads the pinned contract and refuses on any violation |
| `ac3_criteria.py` | pure AC-3 primitives (bf16 ULP, repetition, byte soup, perplexity, position classification) |
| `score_ac3.py` | AC-3 verdict over a cold-rep tree |
| `hf_score.py` | the only GPU part of AC-3: decoded text + teacher-forced perplexity under the pinned HF model |
| `score_perf.py` | AC-4/AC-5 verdict, the drift rule, the geometry assertions |
| `report.py` | machine-readable report + human summary + exit code |
| `collect_ac3.sh` | pre-flight probe, cold sweep via `harness/gate_ac3_stable.sh`, HF stage |
| `collect_perf.sh` | MPK full+pre arms, per-arm kernels, drain gate, device audit |
| `collect_vllm.sh` | the fresh vLLM comparator sweep |
| `remote_setup.sh` | isolated clone at the gate's sha + fresh extension build |
| `mechanisms.json` | the AC-3(b) mechanism registry (ships empty) |
| `tests/` | GPU-free unit + fixture tests |

## What the pinned comment block says vs what this gate enforces

The pinned gate's comment block describes AC-3 as "greedy token ids of mpk == HF transformers
reference on all 10 prompts x 64 tokens". That wording predates the user's 2026-07-29 re-pin and is
now **historical prose**: `.pm/goal.md` is authoritative, and this gate implements the re-pinned
AC-3 (coherence + a >=90 % agreement floor + no silent degradation). Byte-identity is still measured
and reported every run, as AC-3(c) requires -- it is simply no longer the pass condition. Nothing
here needs the pinned gate changed; it is tier-2 pinned and the flag contract it passes is satisfied
verbatim.

One consequence to be aware of: with `final.sh` now present, the pinned gate no longer takes its
"NOT-APPLICABLE (exit 3), final harness absent" branch. It runs the real gate, which means a
multi-hour B200 campaign (5 batch sizes x 3 cold AC-3 reps x 2 perf arms + a fresh vLLM sweep).

## What remains before the gate can be run for a verdict

1. **The decode levers.** AC-4 fails today at 0.354x of vLLM at bs1 and AC-5 at 2.89x against the
   1.25x bound (`opt/m4i1/README.md`). Nothing in the gate changes that; it only reports it.
2. **A full-scope run.** The first end-to-end run covered bs1 only, deliberately (non-binding). The
   binding run needs all five batch sizes: 15 cold AC-3 reps, 30 MPK perf cells and a 5-size fresh
   vLLM sweep. At the observed per-cell costs that is several hours of exclusive-GPU time, and
   `/raid` was at 99-100 % use during this work, so the kernel scratch needs headroom checked first.
3. **A clean tree at invocation.** Integrity requires it, and the shared clone usually has another
   agent's untracked work in it. Run the gate from a dedicated checkout of the commit under test.
4. **The commit under test must be fetchable on the box** -- `remote_setup.sh` clones by sha from the
   box's mirror or from the GitHub remote. An unpushed local commit has to be shipped to the deploy
   path directly (that is how this run was done, and `remote_setup.sh` now says so when the checkout
   fails).
5. **Two dispersion arms.** bs1 prefill and bs16 full sit above the 5 % bound in the committed M3-I7
   evidence; a PASS needs a valid measurement, so those need the protocol's section 6 escalation
   (more reps) if they reproduce. In this run's own window the bs1 arms measured 0.07 % / 0.33 %.
