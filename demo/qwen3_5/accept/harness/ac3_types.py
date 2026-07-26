"""Shared data types for the AC-3 correctness harness.

See `.pm/goal.md` AC-3, `docs/qwen35/mpk-gaps.md` §6.5, and `docs/qwen35/v1-architecture.md`
§12 for the contract these types encode. Kept dependency-free (stdlib only, no torch) so the
harness runs on any CPU box against any engine's token-id dump.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class TieVerdict(str, Enum):
    """Per-position classification produced by `tie_classifier.classify_position`.

    This is a CLASSIFICATION, not a waiver. Only a human, writing into the run report per
    goal.md AC-3's second sentence, grants an actual tie-flip exception. See `WaiverRequest`
    below.
    """

    MATCH = "match"
    ENGINE_TOO_SHORT = "engine_sequence_too_short"
    # Symmetric with ENGINE_TOO_SHORT: the engine produced a token PAST the reference's
    # length. AC-3 requires exact full-sequence equality, including length, so this is a hard
    # failure, not a pass-with-extra-data. Assigned by the orchestrator, like ENGINE_TOO_SHORT
    # — never by classify_position, which only ever sees positions the reference has.
    ENGINE_TOO_LONG = "engine_sequence_too_long"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    IMPLEMENTATION_BUG = "implementation_bug"
    CANDIDATE_TIE_FLIP = "candidate_tie_flip"
    # Assigned by the orchestrator (never by the pure classifier) to any position after the
    # first divergence in its prompt: greedy decode feeds its own output back in, so these are
    # a different conditioning sequence, not independent evidence [MG §6.5].
    POST_DIVERGENCE = "post_divergence_not_independent"


@dataclass
class PromptRequest:
    """What the harness hands an engine adapter: a prompt id + its exact reference input_ids
    (never re-tokenized — reused verbatim from reference_outputs.json [VA §12 item 1])."""

    prompt_id: str
    input_ids: List[int]


@dataclass
class ReferenceStep:
    """One decode position of the committed HF reference.

    `top2_id` / `top2_logit` are `None` unless the source JSON actually carries them.
    **As of this writing, the committed `reference_outputs.json` only stores
    `top1_logit_per_step` — `generate_reference.py` takes `torch.max` and discards the rest of
    the logits vector — so these read `None` for every position of every prompt today.** This
    type stays forward-compatible with a future reference-generator upgrade that persists a
    real top-2 (or top-k); it never fabricates one from the single float that is available.
    """

    position: int
    top1_id: int
    top1_logit: Optional[float]
    top2_id: Optional[int] = None
    top2_logit: Optional[float] = None

    @property
    def margin(self) -> Optional[float]:
        """logit[top1] - logit[top2], or None if either side is unavailable."""
        if self.top1_logit is None or self.top2_logit is None:
            return None
        return self.top1_logit - self.top2_logit


@dataclass
class PromptReference:
    """The committed reference for one prompt id."""

    prompt_id: str
    input_ids: List[int]
    output_ids: List[int]
    num_generated: int
    hit_eos: bool
    eos_step: Optional[int]
    steps: List[ReferenceStep]


@dataclass
class EngineSequence:
    """What an engine adapter returns for one prompt at one batch size.

    `topk_logits`, when present, is a list (one dict per generated step) mapping token_id ->
    logit for whatever top-k the engine reported (e.g. MPK's top-16 dump per VA §12 item 3).
    Absent entirely for engines that only expose token ids (e.g. the vLLM smoke artifact).
    """

    token_ids: List[int]
    topk_logits: Optional[List[Dict[int, float]]] = None

    def logit_at(self, step: int, token_id: Optional[int]) -> Optional[float]:
        if token_id is None or self.topk_logits is None:
            return None
        if step >= len(self.topk_logits):
            return None
        return self.topk_logits[step].get(token_id)


@dataclass
class PositionRecord:
    """Full per-position instrumentation, emitted on EVERY run, passing or failing
    [GOAL AC-3; MG §6.5 items 1-3; VA §12 item 2]."""

    prompt_id: str
    batch_size: int
    position: int
    # None only for an ENGINE_TOO_LONG position: the engine produced a token past the end of
    # the reference, so there is no reference id to report.
    ref_top1_id: Optional[int]
    ref_top1_logit: Optional[float]
    ref_top2_id: Optional[int]
    ref_top2_logit: Optional[float]
    margin: Optional[float]
    engine_argmax_id: Optional[int]
    engine_logit_at_ref_top1: Optional[float]
    engine_logit_at_ref_top2: Optional[float]
    match: Optional[bool]
    verdict: str
    is_first_divergence: bool = False


@dataclass
class WaiverRequest:
    """Evidence package for a human to adjudicate — the harness COLLECTS evidence and emits
    this record; it never sets `auto_waived=True` itself and nothing downstream may either.
    A real waiver is a human writing the mechanism into the run report per goal.md AC-3.
    """

    prompt_id: str
    batch_size: int
    first_divergent_position: int
    evidence: PositionRecord
    classifier_verdict: str
    auto_waived: bool = field(default=False)
    needs_human_adjudication: bool = field(default=True)
    notes: str = (
        "Evidence only. A tie-flip waiver requires a human to document the per-position "
        "logit mechanism in the run report per goal.md AC-3; this record is not a waiver."
    )

    def __post_init__(self) -> None:
        # Structurally prevent this from ever becoming a silent auto-waive, even if a caller
        # tries to construct one with auto_waived=True.
        if self.auto_waived:
            raise ValueError("WaiverRequest.auto_waived must never be True — see docstring.")


@dataclass
class PromptRunResult:
    prompt_id: str
    batch_size: int
    positions: List[PositionRecord]
    passed: bool
    first_divergent_position: Optional[int]
    waiver_request: Optional[WaiverRequest]


@dataclass
class GateReport:
    """Top-level run report: written to disk on every invocation."""

    overall_pass: bool
    status: str  # "pass" | "fail" | "partial_smoke_only"
    batch_sizes: List[int]
    prompt_ids: List[str]
    tie_margin_threshold: float
    prompt_results: List[PromptRunResult]
    waiver_requests: List[WaiverRequest]
    margin_evidence: Dict[str, object]
    notes: List[str] = field(default_factory=list)
