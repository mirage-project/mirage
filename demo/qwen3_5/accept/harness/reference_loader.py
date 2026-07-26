"""Loads the committed HF `transformers` reference (`reference_outputs.json`) — the harness's
ground truth per `.pm/goal.md` AC-3. Read-only: never writes into `accept/reference/`.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

from ac3_types import PromptReference, ReferenceStep

# Optional, forward-compatible keys a future reference-generator upgrade might add. Checked
# defensively; absent in the reference_outputs.json committed as of this harness (see
# ac3_types.ReferenceStep docstring) — so top2_id/top2_logit come back None for every position
# against the current artifact. Never fabricated from top1_logit_per_step alone.
_OPTIONAL_TOP2_ID_KEYS = ("top2_id_per_step",)
_OPTIONAL_TOP2_LOGIT_KEYS = ("top2_logit_per_step",)


def _first_present(d: dict, keys) -> Optional[list]:
    for k in keys:
        if k in d:
            return d[k]
    return None


def load_reference(path: Path) -> Dict[str, PromptReference]:
    """Parse `reference_outputs.json` into `{prompt_id: PromptReference}`.

    Raises FileNotFoundError / json.JSONDecodeError / KeyError with the default messages if
    the file is missing or malformed — this loader does not guess at a malformed schema.
    """
    with open(path, "r") as f:
        raw = json.load(f)

    results = raw["results"]
    out: Dict[str, PromptReference] = {}
    for pid, r in results.items():
        output_ids = r["output_ids"]
        top1_logits = r["top1_logit_per_step"]
        num_generated = r["num_generated"]
        if len(output_ids) != num_generated or len(top1_logits) != num_generated:
            raise ValueError(
                f"{path}: {pid}: num_generated={num_generated} but "
                f"len(output_ids)={len(output_ids)}, len(top1_logit_per_step)={len(top1_logits)}"
            )

        top2_ids = _first_present(r, _OPTIONAL_TOP2_ID_KEYS)
        top2_logits = _first_present(r, _OPTIONAL_TOP2_LOGIT_KEYS)

        steps = []
        for i in range(num_generated):
            steps.append(
                ReferenceStep(
                    position=i,
                    top1_id=output_ids[i],
                    top1_logit=top1_logits[i],
                    top2_id=(top2_ids[i] if top2_ids is not None else None),
                    top2_logit=(top2_logits[i] if top2_logits is not None else None),
                )
            )

        out[pid] = PromptReference(
            prompt_id=pid,
            input_ids=r["input_ids"],
            output_ids=output_ids,
            num_generated=num_generated,
            hit_eos=r["hit_eos"],
            eos_step=r["eos_step"],
            steps=steps,
        )
    return out


def margin_evidence_summary(references: Dict[str, PromptReference]) -> dict:
    """Whether the loaded reference actually carries top-2/margin data, and a distribution
    over whatever margins ARE available. Archived with every run report per MG §6.5 ("a
    tie-flip claim is only credible against margins measured while passing").
    """
    margins = []
    total_positions = 0
    for pref in references.values():
        for step in pref.steps:
            total_positions += 1
            if step.margin is not None:
                margins.append(step.margin)

    if not margins:
        return {
            "margin_data_available_positions": 0,
            "total_positions": total_positions,
            "available": False,
            "reason": (
                "reference_outputs.json stores only top1_logit_per_step (see "
                "generate_reference.py); no top-2 id/logit is committed, so no margin can be "
                "computed for any position. This is a known gap, not a bug in this loader — "
                "see the harness README."
            ),
        }
    return {
        "margin_data_available_positions": len(margins),
        "total_positions": total_positions,
        "available": True,
        "min": min(margins),
        "max": max(margins),
        "mean": sum(margins) / len(margins),
    }
