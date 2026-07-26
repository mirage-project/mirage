"""The argmax-tie classifier: a pure function, no I/O, no sequence state.

Implements the evidence test from `docs/qwen35/mpk-gaps.md` §6.5, quoted here because the
classifier's branches map directly onto it:

    "a mismatch is a candidate tie-flip when the reference margin at that position is within
    the run's observed FP8 noise floor AND MPK's ranking of exactly those two ids is inverted
    while everything upstream matches — and even then the waiver is a written entry in the run
    report naming the position, the margin, and the mechanism, not a flag the harness sets."

This function only ever answers "does this ONE position's evidence look like a tie-flip
candidate, an implementation bug, or is there not enough evidence to say" — it does NOT decide
"everything upstream matches" (that requires sequence context: see `ac3_runner`, which
downgrades any post-first-divergence position to `TieVerdict.POST_DIVERGENCE` regardless of
what this function returns, since those positions are a different conditioning sequence, not
independent evidence). It also never sets a waiver; see `ac3_types.WaiverRequest`.
"""
from __future__ import annotations

from typing import Optional

from ac3_types import TieVerdict

# PROVISIONAL placeholder. The real FP8 noise floor for MPK-vs-HF margins is an empirical
# M2/M3 question (no measurement exists yet — this harness predates MPK). Override via
# `--tie-margin-threshold`; never treat this constant as an authoritative gate value.
DEFAULT_TIE_MARGIN_THRESHOLD = 0.5


def classify_position(
    ref_top1_id: int,
    ref_top2_id: Optional[int],
    margin: Optional[float],
    engine_argmax_id: Optional[int],
    engine_logit_at_ref_top1: Optional[float] = None,
    engine_logit_at_ref_top2: Optional[float] = None,
    tie_margin_threshold: float = DEFAULT_TIE_MARGIN_THRESHOLD,
) -> TieVerdict:
    """Classify one decode position given only that position's evidence.

    Args:
        ref_top1_id: the reference's greedy pick (== reference output_ids[position]).
        ref_top2_id: the reference's runner-up id, or None if unavailable.
        margin: ref logit[top1] - logit[top2], or None if unavailable.
        engine_argmax_id: the engine-under-test's pick at this position, or None if the
            engine's sequence didn't reach this position at all (a distinct failure mode from
            a wrong token).
        engine_logit_at_ref_top1 / engine_logit_at_ref_top2: the engine's own logit values at
            the reference's top-1/top-2 ids, when the engine reports enough of its logit
            distribution to look them up (e.g. a top-16 dump). None if unavailable or if the
            id fell outside whatever the engine reported.
        tie_margin_threshold: the noise-floor cutoff (see module docstring caveat).
    """
    if engine_argmax_id is None:
        return TieVerdict.ENGINE_TOO_SHORT

    if engine_argmax_id == ref_top1_id:
        return TieVerdict.MATCH

    # From here on the position is a mismatch: classify how strong the tie-flip evidence is,
    # never whether to waive it.
    if ref_top2_id is None or margin is None:
        return TieVerdict.INSUFFICIENT_EVIDENCE

    if engine_argmax_id != ref_top2_id:
        # The engine picked some third id, not the reference's runner-up — not a simple
        # top-1/top-2 inversion, regardless of margin.
        return TieVerdict.IMPLEMENTATION_BUG

    if margin > tie_margin_threshold:
        # Exactly the top-2 id, but the reference's margin over it was wide — not a tie.
        return TieVerdict.IMPLEMENTATION_BUG

    if engine_logit_at_ref_top1 is not None and engine_logit_at_ref_top2 is not None:
        # When the engine's own logits are available, require a genuine inversion (its logit
        # at the ref's top-2 id actually outranks its logit at the ref's top-1 id) rather than
        # trusting argmax equality alone.
        if not (engine_logit_at_ref_top2 > engine_logit_at_ref_top1):
            return TieVerdict.IMPLEMENTATION_BUG

    return TieVerdict.CANDIDATE_TIE_FLIP
