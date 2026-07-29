# AC-3 correctness harness

Compares an engine's greedy-decode token ids against the committed HF `transformers`
reference (`../reference/reference_outputs.json`) per `.pm/goal.md` AC-3. Gate: **all token
ids equal, no tolerance.** The tie-flip clause in AC-3's second sentence is a documented,
human-adjudicated exception to an already-failed gate — this harness collects the evidence a
human needs to write that adjudication into the run report; it never grants the waiver itself
(see `ac3_types.WaiverRequest`).

MPK does not exist yet at the time this harness was built (M2-I1). It is engine-agnostic by
design — see "Plugging in an engine" below — and is stub-tested against the committed vLLM
smoke artifact plus synthetic fixtures (`tests/`).

## Quick start

```bash
# Real gate, once something can dump token ids per batch size:
./run_ac3.sh --engine-dump-dir /path/to/dumps --output-json run_report.json

# Partial smoke against the committed vLLM artifact (NOT an AC-3 verdict - one prompt, bs=1):
./run_ac3.sh --vllm-smoke ../reference/vllm_smoke/vllm_smoke_result.json

# Tests (stdlib unittest, zero extra dependencies, CPU-only):
python3 tests/test_tie_classifier.py -v
python3 tests/test_stub_e2e.py -v
```

Exit codes: `0` pass, `1` fail (real mismatch — read the waiver-request evidence before
assuming it's a bug), `2` usage/integrity error (bad args, malformed reference, wrong prompt
count), `3` not applicable as the real gate (no engine data given, or an intentional
`--allow-partial`/`--vllm-smoke` smoke run).

## The gate checks full-sequence equality, including length

AC-3 is exact-sequence equality, not "every reference position happened to match": a 65-token
engine sequence whose first 64 ids match a 64-token reference must still fail — the 65th token
means the engine did not stop where the reference stopped. `evaluate_prompt_at_bs` (in
`ac3_runner.py`) enforces this: any engine token past `pref.num_generated` gets its own
`ENGINE_TOO_LONG` position record (`ref_top1_id=None`, `match=False`), symmetric with
`ENGINE_TOO_SHORT` for a truncated sequence, and hard-fails the gate at the first such
position. **A too-long sequence almost always means the run configuration violated the
exactly-N-new-tokens protocol (e.g. `max_new_tokens` set wrong, or EOS handling disabled) —
it is a run-config failure, not a token-level mismatch, and is never eligible for
`CANDIDATE_TIE_FLIP`** (the position has no reference id to tie-flip against in the first
place; `classify_position` is never even called for it — see `tests/test_stub_e2e.py::EngineTooLongTest`).

## Plugging in an engine

Implement `engine_adapter.EngineAdapter.run(requests, batch_size) -> {prompt_id:
EngineSequence}` directly for an in-process integration, or dump one JSON file per batch size
(`bs<N>.json`, a dict keyed by prompt id — see `engine_adapter.py`'s module docstring for the
exact shape) and point `--engine-dump-dir` at the directory holding them. Both `token_ids` and
`output_ids` are accepted as the per-prompt key, matching the two shapes already in this repo
(`demo/deepseek_v3/demo.py`'s dump and `reference_outputs.json`/the vLLM smoke artifact,
respectively) — nothing needs renaming just to be read here.

Running the same 10 prompts against the ONE fixed reference at every batch size in `{1, 2, 4,
8, 16}` is what covers padding/ordering/state-reset: if batching corrupts per-request state,
some batch size will diverge from the reference where bs=1 did not, and the ordinary
position-level compare surfaces it directly — no separate mechanism is needed
(`tests/test_stub_e2e.py::FullSweepPerfectEngineTest::test_bs_dependent_regression_is_caught_against_the_single_reference`
demonstrates this).

## Resolved gap: the committed reference now carries top-k/margin data (M2-I3 addendum)

`docs/qwen35/mpk-gaps.md` §6.5 and `docs/qwen35/v1-architecture.md` §12 ask this harness to
emit, for every position, the reference's top-2 ids and the margin `logit[top1] -
logit[top2]`. **Originally the committed `reference_outputs.json` could not supply this** —
`generate_reference.py` called `torch.max` on each step's logits and kept only the top-1 value
(`top1_logit_per_step`); the rest of the logits vector was discarded before the file was
written. This harness never fabricated a margin from that single float:
`reference_loader.py` checked defensively for optional `top2_id_per_step` /
`top2_logit_per_step` keys and reported `None` / `"available": false` when absent — true for
every position of every prompt as originally committed.

**As of the 2026-07-25 M2-I3 addendum regeneration, the gap is closed.**
`generate_reference.py` now persists real per-step top-`k` (`k=4` by default, `--topk-logits`
to change it; see `../reference/README.md` "Schema addendum") and `reference_outputs.json` was
regenerated with `input_ids`/`output_ids` verified programmatically byte-identical to the prior
committed artifact for all 10 prompts (identity requirement — token ids never changed, only
metadata/logit fields were added; see `../reference/README.md`'s provenance table for the full
regeneration record, including a real tie-breaking bug the regeneration hit and fixed along the
way). `reference_loader.py`/`ac3_types.py` needed NO changes — they already consumed
`top2_id_per_step`/`top2_logit_per_step` defensively, exactly as designed. Confirmed against the
real artifact: `margin_evidence_summary()` now reports `available: true` for all 640 positions
(10 prompts × 64 tokens), margins spanning `[0.0, 18.875]`, mean `9.47` (the `0.0` minimum
reflects 2 real observed exact top-1/top-2 logit ties among the 640 positions — e.g.
`p06-poem` step 56 — each correctly represented as two DISTINCT ids sharing an equal logit, not
a bug; `generate_reference.py` handles this tie case explicitly, including a caught-and-fixed
defect where an earlier version silently duplicated the id instead — see
`../reference/README.md`'s provenance table). A real mismatch
against today's reference can therefore classify as `candidate_tie_flip` when the evidence
supports it, not just `insufficient_evidence` — `tests/test_stub_e2e.py::RealVllmSmokeStubTest`
now asserts `margin_evidence["available"]` is `True` against the real artifact (previously
asserted `False`, documenting the gap this section used to describe);
`tests/test_stub_e2e.py::SyntheticFixtureTest` continues to exercise the full tie/bug
classification via its own hand-authored synthetic reference, independent of the real data.

## Cold-run stability gate (M4-I0)

`run_ac3.py` scores ONE engine dump. It cannot tell a clean run from a run whose arithmetic
diverged mid-flight and happened not to cross an argmax margin — and on this box that happens.
`gate_ac3_stable.sh` wraps the sweep to close that hole:

    CUDA_VISIBLE_DEVICES=<one idle device> bash gate_ac3_stable.sh --out DIR [--reps 3]

It runs `--reps` independent COLD-compiled reps per batch size, captures the KV/GDN
wave-boundary fingerprint per rep next to the token dump, and passes only when every rep's
tokens are byte-identical to `../results/dumps_final` per case AND `--reps` reps agree with
each other fingerprint-for-fingerprint. Fingerprint-divergent reps are quarantined, re-run,
and reported as a rate. Exit 0 STABLE / 1 FAIL (tokens) / 2 UNSTABLE / 3 integrity.

Measured over 100 cold reps on five B200s: 4.2% of reps diverge in state, 2.1% in tokens, and
two of the four observed divergences emitted a token md5 *identical to the baseline* — i.e.
invisible to `run_ac3.py` alone. Numbers, the four events, and what M4's `final.sh` can
honestly assert: `../opt/m4i0/README.md`. Protocol basis:
`docs/qwen35/bench-protocol.md`, "Determinism protocol v2". Scorer unit tests (no GPU):
`../opt/m4i0/scripts/test_gate_scorer.py`.

## Files

| File | Role |
|---|---|
| `ac3_types.py` | Shared dataclasses/enum: `ReferenceStep`, `EngineSequence`, `PositionRecord`, `WaiverRequest`, `GateReport`, `TieVerdict`. |
| `reference_loader.py` | Loads `reference_outputs.json`; computes the margin-evidence summary archived with every run. |
| `engine_adapter.py` | The `EngineAdapter` interface + `JSONDumpAdapter` (file-based, the real integration path today) + `StaticMappingAdapter` / `load_vllm_smoke` (in-memory, used by tests and the `--vllm-smoke` CLI mode). |
| `tie_classifier.py` | `classify_position(...)` — pure, no I/O, independently unit-tested. Implements the mpk-gaps.md §6.5 evidence test. |
| `ac3_runner.py` | Orchestrates the sweep, computes first-divergence per prompt, assembles waiver requests and the `GateReport`. |
| `run_ac3.py` / `run_ac3.sh` | CLI entry point. |
| `gate_ac3_stable.sh` / `gate_ac3_stable.py` | The cold-run stability gate: N independent cold reps per bs, KV/GDN fingerprint scoring, quarantine-and-re-run, machine-readable verdict (M4-I0). |
| `fixtures/synthetic_reference.json` + `synthetic_engine_dump.json` | Hand-authored synthetic scenarios covering every classifier branch (wide-margin bug, top-2-but-wide bug, tie candidate with/without confirming engine logits, a refuted tie, insufficient evidence, and a post-divergence case). |
| `tests/test_tie_classifier.py` | Required unit test for the argmax-tie classifier. |
| `tests/test_stub_e2e.py` | Stub-tests against the real vLLM smoke artifact + the synthetic fixtures + batch-size/missing-prompt semantics. |

## Provisional constant

`tie_classifier.DEFAULT_TIE_MARGIN_THRESHOLD = 0.5` is a placeholder — the real FP8
MPK-vs-HF margin noise floor is an empirical M2/M3 question this harness predates. Override
with `--tie-margin-threshold` once a real distribution exists; never treat it as authoritative.
