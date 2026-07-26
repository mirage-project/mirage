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

## Known gap: the committed reference has no top-2/margin data

`docs/qwen35/mpk-gaps.md` §6.5 and `docs/qwen35/v1-architecture.md` §12 ask this harness to
emit, for every position, the reference's top-2 ids and the margin `logit[top1] -
logit[top2]`. **The committed `reference_outputs.json` cannot supply this today** —
`generate_reference.py` calls `torch.max` on each step's logits and keeps only the top-1 value
(`top1_logit_per_step`); the rest of the logits vector is discarded before the file is written.

This harness does not fabricate a margin from that single float. `reference_loader.py` checks
defensively for optional `top2_id_per_step` / `top2_logit_per_step` keys and reports `None` /
`"available": false` (with the reason above) when they're absent, which is the case for every
position of every prompt in the artifact as committed. A real mismatch against today's
reference therefore classifies as `insufficient_evidence`, never `candidate_tie_flip` — there
is not enough information yet to ever substantiate a tie-flip waiver. A future reference
regeneration that persists a real top-2 (or top-k) per step would need no changes here beyond
populating those two optional keys; `ac3_types.ReferenceStep` and the classifier already
consume them when present (see `tests/test_stub_e2e.py::SyntheticFixtureTest`, which exercises
the full tie/bug classification using a synthetic reference that does carry them).

## Files

| File | Role |
|---|---|
| `ac3_types.py` | Shared dataclasses/enum: `ReferenceStep`, `EngineSequence`, `PositionRecord`, `WaiverRequest`, `GateReport`, `TieVerdict`. |
| `reference_loader.py` | Loads `reference_outputs.json`; computes the margin-evidence summary archived with every run. |
| `engine_adapter.py` | The `EngineAdapter` interface + `JSONDumpAdapter` (file-based, the real integration path today) + `StaticMappingAdapter` / `load_vllm_smoke` (in-memory, used by tests and the `--vllm-smoke` CLI mode). |
| `tie_classifier.py` | `classify_position(...)` — pure, no I/O, independently unit-tested. Implements the mpk-gaps.md §6.5 evidence test. |
| `ac3_runner.py` | Orchestrates the sweep, computes first-divergence per prompt, assembles waiver requests and the `GateReport`. |
| `run_ac3.py` / `run_ac3.sh` | CLI entry point. |
| `fixtures/synthetic_reference.json` + `synthetic_engine_dump.json` | Hand-authored synthetic scenarios covering every classifier branch (wide-margin bug, top-2-but-wide bug, tie candidate with/without confirming engine logits, a refuted tie, insufficient evidence, and a post-divergence case). |
| `tests/test_tie_classifier.py` | Required unit test for the argmax-tie classifier. |
| `tests/test_stub_e2e.py` | Stub-tests against the real vLLM smoke artifact + the synthetic fixtures + batch-size/missing-prompt semantics. |

## Provisional constant

`tie_classifier.DEFAULT_TIE_MARGIN_THRESHOLD = 0.5` is a placeholder — the real FP8
MPK-vs-HF margin noise floor is an empirical M2/M3 question this harness predates. Override
with `--tie-margin-threshold` once a real distribution exists; never treat it as authoritative.
