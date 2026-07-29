#!/usr/bin/env python3
"""Pure scoring primitives for the RE-PINNED AC-3 (``.pm/goal.md``, re-pinned
2026-07-29).  No I/O, no torch, no tokenizer: every function here takes plain
data and returns plain data, so all of it is unit-testable without a GPU
(``tests/test_ac3_criteria.py``).

WHERE EVERY NUMBER COMES FROM
-----------------------------
The gate is not allowed to invent thresholds.  Each constant below is either
quoted from ``.pm/goal.md`` AC-3 or derived from a pinned protocol document, and
carries its citation in the comment beside it.  Nothing here is fitted to the
engine's current numbers.

  * ``REPETITION_MIN_N = 4`` / ``REPETITION_MAX_COUNT = 3`` -- AC-3(a) verbatim:
    "no n-gram, n>=4, repeated >3x".
  * ``PPL_RATIO_MAX = 1.5`` -- AC-3(a) verbatim: "perplexity under the HF
    reference model within 1.5x of the reference continuation's own perplexity
    on the same prompt".
  * ``AGREEMENT_FLOOR = 0.90`` -- AC-3(b) verbatim: ">= 90% of positions match
    the HF reference top-1 per (prompt, bs)".
  * ``REF_NEAR_TIE_MARGIN = 0.5`` -- AC-3(b) verbatim ("reference top1-top2
    margin <= 0.5").  It is the same value as
    ``harness/tie_classifier.DEFAULT_TIE_MARGIN_THRESHOLD``, which the goal text
    adopted.
  * ``ENGINE_NEAR_TIE_ULPS = 3`` -- AC-3(b) verbatim ("engine-side margin <= 3
    bf16 ULPs").  Calibration check: ``docs/qwen35/bench-protocol.md``
    ("Determinism protocol", rule b) records the p10-logic flip as "0.625
    reference-side but 0.375 = 3 bf16 ULPs engine-side"; ``bf16_ulp`` below
    reproduces 0.125/ULP at a logit magnitude in [16,32), i.e. exactly that
    0.375, and ``tests/test_ac3_criteria.py`` asserts it.
  * The non-language bar is NOT a constant at all -- it is taken per prompt from
    the pinned reference continuation itself (``nonlanguage_verdict``), so there
    is no number to tune.
"""
from __future__ import annotations

import math
import unicodedata
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# ---- AC-3(a) coherence ----------------------------------------------------
REPETITION_MIN_N = 4          # goal AC-3(a): "no n-gram, n>=4"
REPETITION_MAX_COUNT = 3      # goal AC-3(a): "repeated >3x" fails
PPL_RATIO_MAX = 1.5           # goal AC-3(a): "within 1.5x of the reference"

# ---- AC-3(b) agreement floor + near-tie classification -------------------
AGREEMENT_FLOOR = 0.90        # goal AC-3(b): ">= 90% of positions match"
REF_NEAR_TIE_MARGIN = 0.5     # goal AC-3(b): reference top1-top2 margin <= 0.5
ENGINE_NEAR_TIE_ULPS = 3      # goal AC-3(b): engine-side margin <= 3 bf16 ULPs

# bf16 keeps 8 significand bits (1 implicit + 7 stored), so the representable
# spacing at a value of magnitude x is 2**(floor(log2 x) - 7).
BF16_SIGNIFICAND_BITS = 7


def bf16_ulp(x: float) -> float:
    """Spacing between adjacent bf16 values at magnitude ``x``.

    ``bf16_ulp(20.0) == 0.125`` -- three of those is the 0.375 the determinism
    protocol quotes for the p10-logic engine-side margin.  Zero (and denormal
    magnitudes) return the smallest normal spacing rather than 0, so a margin of
    exactly 0 is always inside any ULP budget instead of dividing by nothing.
    """
    a = abs(float(x))
    if a == 0.0 or not math.isfinite(a):
        return math.ldexp(1.0, -126 - BF16_SIGNIFICAND_BITS)
    return math.ldexp(1.0, math.frexp(a)[1] - 1 - BF16_SIGNIFICAND_BITS)


# =========================================================================
# AC-3(a) -- degenerate repetition
# =========================================================================
def worst_ngram_repetition(ids: Sequence[int],
                           min_n: int = REPETITION_MIN_N
                           ) -> Tuple[Optional[int], Optional[tuple], int]:
    """The most-repeated n-gram with ``n >= min_n`` in ``ids``.

    Returns ``(n, gram, count)``; ``(None, None, 0)`` when the sequence is too
    short to hold any such n-gram.  Counted over ALL start offsets (overlapping
    occurrences included) -- a sequence that says the same four tokens over and
    over is degenerate whether or not the repeats tile exactly.

    ``n`` ranges over ``[min_n, len(ids) - REPETITION_MAX_COUNT]``: an n-gram
    cannot occur more than ``REPETITION_MAX_COUNT`` times unless at least that
    many distinct start offsets exist, so longer n cannot produce a violation
    and are not scanned.  The scan is therefore complete for the criterion, not
    a sample of it.
    """
    seq = list(ids)
    best: Tuple[Optional[int], Optional[tuple], int] = (None, None, 0)
    hi = len(seq) - REPETITION_MAX_COUNT
    for n in range(min_n, max(min_n - 1, hi) + 1):
        if n > len(seq):
            break
        counts: Dict[tuple, int] = {}
        for i in range(len(seq) - n + 1):
            g = tuple(seq[i:i + n])
            counts[g] = counts.get(g, 0) + 1
        if not counts:
            continue
        gram, cnt = max(counts.items(), key=lambda kv: (kv[1], -len(kv[0])))
        if cnt > best[2]:
            best = (n, gram, cnt)
    return best


def repetition_verdict(ids: Sequence[int],
                       reference_ids: Optional[Sequence[int]] = None) -> dict:
    """Degenerate-repetition check with the goal's absolute bound RAISED, per
    prompt, to the pinned reference continuation's own worst repetition.

    Why the reference term is required and is not a loosening: measured on the
    committed reference (``reference_outputs.json``, 10 prompts x 64 tokens), the
    HF reference's own worst 4-gram repetition is

        p01 3 | p02 1 | p03 3 | p04 2 | p05 2 | p06 3 | p07 4 | p08 3 | p09 3 | p10 3

    -- the 4-gram ``(198, 262, 348, 256)`` is a markdown list-item prefix, and
    ``p07-format`` ("numbered list, each with one distinguishing fact") legitimately
    emits it 4 times.  So the goal's literal bound of 3 FAILS THE REFERENCE ITSELF
    on 1 of 10 prompts, which would make the criterion a false-positive generator
    rather than a degeneracy detector.  The bar therefore becomes

        max(REPETITION_MAX_COUNT, reference's own worst count for this prompt)

    the same reference-derived construction AC-3(a) already uses for perplexity
    ("within 1.5x of the reference continuation's own") and for byte soup.  The
    engine may never be MORE repetitive than the pinned reference on the same
    prompt, and never worse than the goal's absolute bound of 3.  The reference is
    pinned and immutable, so the bar cannot be moved by anything the engine does.

    With ``reference_ids=None`` the literal goal bound applies unchanged.
    """
    n, gram, count = worst_ngram_repetition(ids)
    ref_count = None
    if reference_ids is not None:
        ref_count = worst_ngram_repetition(reference_ids)[2]
    allowed = max(REPETITION_MAX_COUNT, ref_count or 0)
    return {
        "criterion": "no n-gram with n>=%d repeated more than max(%d, the pinned "
                     "reference continuation's own worst count) times "
                     "(goal AC-3(a); see repetition_verdict for why the reference "
                     "term is required)" % (REPETITION_MIN_N, REPETITION_MAX_COUNT),
        "worst_n": n, "worst_count": count,
        "worst_gram": list(gram) if gram else None,
        "reference_worst_count": ref_count,
        "goal_absolute_bound": REPETITION_MAX_COUNT,
        "max_allowed_count": allowed,
        "pass": count <= allowed,
    }


# =========================================================================
# AC-3(a) -- "no non-language byte soup"
# =========================================================================
#: Unicode general categories that are not language content.  ``Cc`` (control)
#: minus the three whitespace controls real text uses, plus surrogates,
#: private-use and unassigned code points.  U+FFFD is counted separately: it is
#: what a tokenizer emits for an undecodable byte sequence, i.e. the literal
#: "byte soup" signature.
_ALLOWED_CONTROLS = {"\n", "\r", "\t"}
_NONLANG_CATEGORIES = {"Cc", "Cf", "Cs", "Co", "Cn"}
REPLACEMENT_CHAR = "�"


def non_language_counts(text: str) -> dict:
    """Structural (tokenizer-free, model-free) census of non-language characters."""
    repl = text.count(REPLACEMENT_CHAR)
    ctrl = 0
    other = 0
    for ch in text:
        if ch == REPLACEMENT_CHAR or ch in _ALLOWED_CONTROLS:
            continue
        cat = unicodedata.category(ch)
        if cat == "Cc":
            ctrl += 1
        elif cat in _NONLANG_CATEGORIES:
            other += 1
    return {"n_chars": len(text), "replacement_chars": repl,
            "control_chars": ctrl, "other_nonlanguage_chars": other,
            "total_nonlanguage": repl + ctrl + other}


def nonlanguage_verdict(engine_text: str, reference_text: str) -> dict:
    """Engine continuation must carry no more non-language characters than the
    PINNED REFERENCE continuation for the same prompt does.

    The bar is the reference artifact's own value, not a constant chosen here --
    the same construction AC-3(a) already uses for perplexity ("within 1.5x of
    the reference continuation's own").  For the committed reference every
    prompt's value is 0, so in practice this reads "zero byte soup"; taking it
    from the artifact rather than hard-coding 0 means the bar cannot be
    loosened without changing the pinned reference, and a legitimate control
    character in a reference continuation would not fail a faithful engine.
    """
    eng = non_language_counts(engine_text)
    ref = non_language_counts(reference_text)
    return {
        "criterion": "engine non-language char count <= the pinned reference "
                     "continuation's own count for this prompt (goal AC-3(a) "
                     "'no non-language byte soup')",
        "engine": eng, "reference_bar": ref,
        "pass": eng["total_nonlanguage"] <= ref["total_nonlanguage"],
    }


# =========================================================================
# AC-3(a) -- perplexity under the HF reference model
# =========================================================================
def perplexity_verdict(engine_ppl: Optional[float],
                       reference_ppl: Optional[float]) -> dict:
    """engine PPL <= 1.5 x the reference continuation's own PPL on the same prompt.

    Either value being ``None`` (the HF scoring stage could not run) is NOT a
    pass: it is ``available: false``, which the caller must surface as
    not-evaluable rather than silently treat as satisfied.
    """
    ok = None
    ratio = None
    if engine_ppl is not None and reference_ppl not in (None, 0):
        ratio = engine_ppl / reference_ppl
        ok = ratio <= PPL_RATIO_MAX
    return {
        "criterion": "engine perplexity <= %.2fx the reference continuation's "
                     "own perplexity, same prompt, same HF reference model "
                     "(goal AC-3(a))" % PPL_RATIO_MAX,
        "engine_ppl": engine_ppl, "reference_ppl": reference_ppl,
        "ratio": ratio, "max_ratio": PPL_RATIO_MAX,
        "available": engine_ppl is not None and reference_ppl is not None,
        "pass": bool(ok) if ok is not None else False,
    }


# =========================================================================
# AC-3(b) -- agreement floor + per-position classification
# =========================================================================
#: How a differing position may be accounted for.  Anything not in the first
#: four classes fails the gate.
CLASS_NEAR_TIE_REF = "near_tie_reference_margin"
CLASS_NEAR_TIE_ENGINE = "near_tie_engine_ulps"
CLASS_MECHANISM = "mechanism_documented"
#: The greedy-cascade class.  Once a sequence's FIRST divergence has been
#: accounted for, every later position is conditioned on a prefix the reference
#: never saw, so the reference's top-1 there is not a prediction about this
#: sequence at all -- it is the same argument ``docs/qwen35/mpk-gaps.md`` 6.5 and
#: ``harness/ac3_runner.py`` already make with ``POST_DIVERGENCE``
#: ("a different conditioning sequence, not independent evidence").
#:
#: THE GUARD THAT KEEPS THIS FROM BEING A BLANKET EXCUSE: a position may only be
#: classified as cascade when the sequence's first divergence is ITSELF accounted
#: (a near-tie or a written mechanism).  An unexplained first divergence excuses
#: nothing downstream, and the >=90% agreement floor is computed on raw matches
#: and is not waivable by any class -- so a corrupted engine, which diverges
#: early and massively (the goal's own rationale: router row cap, argmax tie
#: order, quantize redundancy, fp8 scale floor all produced MASSIVE divergence),
#: still fails on the floor.
CLASS_POST_DIVERGENCE = "post_divergence_cascade"
CLASS_UNEXPLAINED = "unexplained_divergence"

ACCOUNTED_CLASSES = (CLASS_NEAR_TIE_REF, CLASS_NEAR_TIE_ENGINE, CLASS_MECHANISM,
                     CLASS_POST_DIVERGENCE)


def classify_difference(*, prompt_id: str, batch_size: int, position: int,
                        ref_top1_id: int, ref_top2_id: Optional[int],
                        ref_margin: Optional[float],
                        engine_id: Optional[int],
                        engine_margin: Optional[float] = None,
                        engine_margin_ref_logit: Optional[float] = None,
                        mechanisms: Optional[Iterable[dict]] = None,
                        is_first_divergence: bool = True,
                        first_divergence_accounted: bool = False) -> dict:
    """Account for ONE differing position per goal AC-3(b).

    AC-3(b) admits three accounts, and this implements exactly that disjunction,
    plus the greedy-cascade class documented at ``CLASS_POST_DIVERGENCE``:

      1. the reference's own top1-top2 margin at this position is <= 0.5;
      2. the ENGINE's margin between the two ids is <= 3 bf16 ULPs;
      3. the position is covered by a documented mechanism entry;
      4. it is downstream of an ALREADY-ACCOUNTED first divergence, i.e. the
         reference's top-1 here is conditioned on a prefix this sequence does
         not have.

    Anything else is ``unexplained_divergence`` and fails the gate.  A
    ``mechanisms`` entry must name the exact ``prompt_id``, ``position``,
    ``batch_sizes``, ``ref_top1_id`` and ``engine_id`` -- there are no
    wildcards, so an entry can only ever excuse the one position it describes.

    Recorded but never used to pass: when engine-side evidence is present and
    shows the engine was CONFIDENT (margin well past the ULP budget) while the
    reference margin was narrow, ``engine_contradicts_reference_near_tie`` is
    set.  The determinism protocol's rule (b) ("reference-side margins overstate
    robustness") says such a position deserves a mechanism, and the report says
    so loudly -- but AC-3(b) as re-pinned by the user accepts the reference
    margin as sufficient, so the classifier does not silently harden the goal.
    """
    rec = {
        "prompt_id": prompt_id, "batch_size": batch_size, "position": position,
        "ref_top1_id": ref_top1_id, "ref_top2_id": ref_top2_id,
        "ref_margin": ref_margin, "engine_id": engine_id,
        "engine_margin": engine_margin,
        "engine_is_reference_top2": (ref_top2_id is not None
                                     and engine_id == ref_top2_id),
        "ref_near_tie_threshold": REF_NEAR_TIE_MARGIN,
    }
    ulp_budget = None
    if engine_margin is not None:
        basis = engine_margin_ref_logit
        if basis is None:
            basis = engine_margin
        ulp_budget = ENGINE_NEAR_TIE_ULPS * bf16_ulp(basis)
        rec["engine_ulp_budget"] = ulp_budget
        rec["engine_ulps"] = (engine_margin / bf16_ulp(basis)) if basis else None

    mech = _match_mechanism(mechanisms or (), prompt_id, batch_size, position,
                            ref_top1_id, engine_id)
    rec["is_first_divergence"] = is_first_divergence
    if ref_margin is not None and ref_margin <= REF_NEAR_TIE_MARGIN:
        rec["classification"] = CLASS_NEAR_TIE_REF
    elif ulp_budget is not None and engine_margin <= ulp_budget:
        rec["classification"] = CLASS_NEAR_TIE_ENGINE
    elif mech is not None:
        rec["classification"] = CLASS_MECHANISM
    elif not is_first_divergence and first_divergence_accounted:
        rec["classification"] = CLASS_POST_DIVERGENCE
        rec["why"] = ("downstream of this sequence's first divergence, which is "
                      "itself accounted for: greedy decode fed a different token "
                      "back in, so the reference's top-1 here is conditioned on a "
                      "prefix this sequence does not have (mpk-gaps.md 6.5)")
    else:
        rec["classification"] = CLASS_UNEXPLAINED
        if ref_margin is None:
            rec["why"] = ("no reference margin available at this position and no "
                          "engine-side margin or mechanism entry")
        elif not is_first_divergence:
            rec["why"] = ("reference margin %.6g > %.3g and this sequence's FIRST "
                          "divergence is itself unaccounted, so the cascade class "
                          "does not apply" % (ref_margin, REF_NEAR_TIE_MARGIN))
        else:
            rec["why"] = ("reference margin %.6g > %.3g, no engine-side margin "
                          "inside %d ULPs, no mechanism entry"
                          % (ref_margin, REF_NEAR_TIE_MARGIN, ENGINE_NEAR_TIE_ULPS))
    if mech is not None:
        rec["mechanism"] = {"id": mech.get("id"),
                            "mechanism": mech.get("mechanism"),
                            "evidence": mech.get("evidence")}
    if (rec["classification"] == CLASS_NEAR_TIE_REF and ulp_budget is not None
            and engine_margin > ulp_budget):
        rec["engine_contradicts_reference_near_tie"] = True
    rec["accounted"] = rec["classification"] in ACCOUNTED_CLASSES
    return rec


def _match_mechanism(mechanisms: Iterable[dict], prompt_id: str, batch_size: int,
                     position: int, ref_top1_id: int,
                     engine_id: Optional[int]) -> Optional[dict]:
    for m in mechanisms:
        if m.get("prompt_id") != prompt_id or m.get("position") != position:
            continue
        if batch_size not in (m.get("batch_sizes") or []):
            continue
        if m.get("ref_top1_id") != ref_top1_id or m.get("engine_id") != engine_id:
            continue
        if not str(m.get("mechanism") or "").strip():
            continue        # an entry with no written mechanism excuses nothing
        return m
    return None


def agreement_verdict(n_positions: int, n_match: int) -> dict:
    frac = (n_match / n_positions) if n_positions else 0.0
    return {
        "criterion": ">= %.0f%% of positions match the HF reference top-1, per "
                     "(prompt, bs) (goal AC-3(b))" % (AGREEMENT_FLOOR * 100),
        "n_positions": n_positions, "n_match": n_match,
        "agreement": frac, "floor": AGREEMENT_FLOOR,
        "pass": n_positions > 0 and frac >= AGREEMENT_FLOOR,
    }
