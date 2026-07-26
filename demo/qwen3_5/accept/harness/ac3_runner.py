"""Orchestrates the AC-3 gate: for every batch size, ask the engine adapter to run all pinned
prompts, compare each prompt's generated ids against the ONE committed reference for that
prompt id, and assemble the full per-position instrumentation + any waiver-request evidence.

Gate: `mpk_token_ids == reference_token_ids`, no tolerance, for every position of every prompt
at every batch size [GOAL AC-3]. Running the SAME reference against every batch size is what
covers "padding/order/state-reset" per the issue contract — if batching corrupts state, some
bs > 1 diverges from the bs-independent reference where bs=1 did not; no separate mechanism is
needed to detect that, the position-level compare surfaces it directly.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from ac3_types import (
    EngineSequence,
    GateReport,
    PositionRecord,
    PromptReference,
    PromptRequest,
    PromptRunResult,
    TieVerdict,
    WaiverRequest,
)
from engine_adapter import EngineAdapter
from reference_loader import margin_evidence_summary
from tie_classifier import DEFAULT_TIE_MARGIN_THRESHOLD, classify_position


def evaluate_prompt_at_bs(
    pref: PromptReference,
    engine_seq: Optional[EngineSequence],
    batch_size: int,
    tie_margin_threshold: float = DEFAULT_TIE_MARGIN_THRESHOLD,
) -> PromptRunResult:
    """Compare one prompt's engine sequence against its reference at one batch size.

    `engine_seq=None` (the engine produced nothing for this prompt at this batch size) is
    treated as an empty sequence rather than special-cased: every position then comes back
    `ENGINE_TOO_SHORT` through the same code path a genuinely-truncated sequence would hit,
    which fails the gate loudly instead of silently skipping the prompt.
    """
    if engine_seq is None:
        engine_seq = EngineSequence(token_ids=[])

    records: List[PositionRecord] = []
    first_divergence: Optional[int] = None

    for step in pref.steps:
        i = step.position
        engine_id = engine_seq.token_ids[i] if i < len(engine_seq.token_ids) else None
        engine_logit_top1 = engine_seq.logit_at(i, step.top1_id)
        engine_logit_top2 = engine_seq.logit_at(i, step.top2_id)

        verdict = classify_position(
            ref_top1_id=step.top1_id,
            ref_top2_id=step.top2_id,
            margin=step.margin,
            engine_argmax_id=engine_id,
            engine_logit_at_ref_top1=engine_logit_top1,
            engine_logit_at_ref_top2=engine_logit_top2,
            tie_margin_threshold=tie_margin_threshold,
        )
        match = None if engine_id is None else (engine_id == step.top1_id)
        is_first = False
        if verdict != TieVerdict.MATCH and first_divergence is None:
            first_divergence = i
            is_first = True

        records.append(
            PositionRecord(
                prompt_id=pref.prompt_id,
                batch_size=batch_size,
                position=i,
                ref_top1_id=step.top1_id,
                ref_top1_logit=step.top1_logit,
                ref_top2_id=step.top2_id,
                ref_top2_logit=step.top2_logit,
                margin=step.margin,
                engine_argmax_id=engine_id,
                engine_logit_at_ref_top1=engine_logit_top1,
                engine_logit_at_ref_top2=engine_logit_top2,
                match=match,
                verdict=verdict.value,
                is_first_divergence=is_first,
            )
        )

    # Everything after the first divergence is fallout from it, not independent evidence
    # [MG §6.5]: greedy decode fed a wrong token back in, so the engine's conditioning context
    # differs from the reference's from here on. That rationale is specifically about a WRONG
    # TOKEN being produced and fed back in — it doesn't apply to ENGINE_TOO_SHORT (no token at
    # all). A truncated/missing tail is independently true at every such position for the same
    # simple reason ("no more data"), not a cascading contamination effect, so those keep their
    # own verdict instead of being relabeled.
    if first_divergence is not None:
        for rec in records[first_divergence + 1 :]:
            if rec.verdict != TieVerdict.ENGINE_TOO_SHORT.value:
                rec.verdict = TieVerdict.POST_DIVERGENCE.value

    passed = first_divergence is None
    waiver: Optional[WaiverRequest] = None
    if not passed:
        fd_rec = records[first_divergence]
        waiver = WaiverRequest(
            prompt_id=pref.prompt_id,
            batch_size=batch_size,
            first_divergent_position=first_divergence,
            evidence=fd_rec,
            classifier_verdict=fd_rec.verdict,
        )

    return PromptRunResult(
        prompt_id=pref.prompt_id,
        batch_size=batch_size,
        positions=records,
        passed=passed,
        first_divergent_position=first_divergence,
        waiver_request=waiver,
    )


def run_ac3(
    adapter: EngineAdapter,
    references: Dict[str, PromptReference],
    batch_sizes: List[int],
    prompt_ids: Optional[List[str]] = None,
    tie_margin_threshold: float = DEFAULT_TIE_MARGIN_THRESHOLD,
    allow_partial: bool = False,
) -> GateReport:
    """Run the full AC-3 sweep: every prompt id in `prompt_ids` at every size in
    `batch_sizes`. With `allow_partial=False` (the real AC-3 gate), a prompt missing from the
    engine's output at some batch size is evaluated as a hard failure (see
    `evaluate_prompt_at_bs`). With `allow_partial=True` (smoke-testing an engine that only
    covers some prompts/sizes, e.g. the vLLM smoke artifact), missing combinations are skipped
    rather than manufactured into failures, and the result is explicitly NOT reported as an
    AC-3 verdict.
    """
    prompt_ids = list(prompt_ids) if prompt_ids is not None else sorted(references.keys())
    missing_from_reference = [pid for pid in prompt_ids if pid not in references]
    if missing_from_reference:
        raise KeyError(f"prompt ids not found in the loaded reference: {missing_from_reference}")

    prompt_results: List[PromptRunResult] = []
    waiver_requests: List[WaiverRequest] = []
    notes: List[str] = []

    for bs in batch_sizes:
        requests = [PromptRequest(pid, references[pid].input_ids) for pid in prompt_ids]
        engine_out = adapter.run(requests, bs)

        missing_here = [pid for pid in prompt_ids if pid not in engine_out]
        if missing_here:
            verb = "skipping (partial mode)" if allow_partial else "scoring as hard failures"
            notes.append(
                f"bs={bs}: engine produced no output for {len(missing_here)}/{len(prompt_ids)} "
                f"prompt(s), {verb}: {missing_here}"
            )

        for pid in prompt_ids:
            eng_seq = engine_out.get(pid)
            if eng_seq is None and allow_partial:
                continue
            result = evaluate_prompt_at_bs(references[pid], eng_seq, bs, tie_margin_threshold)
            prompt_results.append(result)
            if result.waiver_request is not None:
                waiver_requests.append(result.waiver_request)

    if allow_partial:
        status = "partial_smoke_only"
        overall_pass = bool(prompt_results) and all(r.passed for r in prompt_results)
        notes.append(
            "PARTIAL RUN (--allow-partial): NOT an AC-3 verdict. Only the (prompt, batch_size) "
            "pairs actually present in the engine output were scored; anything absent was "
            "skipped, not counted as pass or fail."
        )
        if not prompt_results:
            notes.append("Nothing was scored — engine output was empty for every request.")
    else:
        overall_pass = bool(prompt_results) and all(r.passed for r in prompt_results)
        status = "pass" if overall_pass else "fail"

    scored_prompt_ids = sorted({r.prompt_id for r in prompt_results}) or prompt_ids
    return GateReport(
        overall_pass=overall_pass,
        status=status,
        batch_sizes=list(batch_sizes),
        prompt_ids=prompt_ids,
        tie_margin_threshold=tie_margin_threshold,
        prompt_results=prompt_results,
        waiver_requests=waiver_requests,
        margin_evidence=margin_evidence_summary(
            {pid: references[pid] for pid in scored_prompt_ids if pid in references}
        ),
        notes=notes,
    )
