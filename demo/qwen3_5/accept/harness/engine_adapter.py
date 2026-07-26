"""The engine-adapter interface: token-ids-in, so MPK, vLLM, or any engine can plug into the
AC-3 harness without the harness knowing anything about how that engine runs.

`EngineAdapter` is the contract a live, in-process integration would implement directly (e.g.
a future MPK Python binding). `JSONDumpAdapter` is the concrete adapter this issue ships and
tests against today: point it at one pre-generated JSON file per batch size (however that JSON
got produced — offline script, notebook, CI artifact) and it satisfies the same interface. This
is the real integration path until an engine is wired in-process; it is not a mock.

Dump file shape (per batch size), a JSON object keyed by prompt id:
    {"p01-history": {"token_ids": [...]}, "p02-math": {"token_ids": [...]}, ...}
`output_ids` is also accepted as a key (matches `reference_outputs.json` / the vLLM smoke
artifact's own field name) so a dump doesn't need renaming just to be read by this harness.
An optional per-step top-k logit dump is accepted as `"topk_logits": [{"<id>": logit, ...}, ...]`
(JSON object keys are strings; converted to int on load) — matches VA §12 item 3's "MPK top-16
logit dump" when an engine can supply it. Absent entirely if the engine only reports token ids
(e.g. the vLLM smoke artifact).
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List

from ac3_types import EngineSequence, PromptRequest


class EngineAdapter(ABC):
    """Token-ids-in engine contract. `requests` carries the exact reference `input_ids`
    (never re-tokenized) for every prompt the harness is asking about at this batch size; the
    adapter owns how it groups/pads/schedules them into `batch_size`-sized batches. The
    harness only requires that every `prompt_id` in `requests` comes back in the result."""

    @abstractmethod
    def run(self, requests: List[PromptRequest], batch_size: int) -> Dict[str, EngineSequence]:
        raise NotImplementedError


def _extract_token_ids(entry: dict, prompt_id: str, source: str) -> List[int]:
    if "token_ids" in entry:
        return entry["token_ids"]
    if "output_ids" in entry:
        return entry["output_ids"]
    raise KeyError(
        f"{source}: entry for {prompt_id!r} has neither 'token_ids' nor 'output_ids' — "
        f"refusing to guess. Keys present: {sorted(entry.keys())}"
    )


def _extract_topk_logits(entry: dict):
    raw = entry.get("topk_logits")
    if raw is None:
        return None
    # JSON object keys are always strings; convert back to int token ids.
    return [{int(k): v for k, v in step.items()} for step in raw]


class JSONDumpAdapter(EngineAdapter):
    """Reads one JSON file per batch size, each a dict keyed by prompt id (see module
    docstring for the exact shape). `dump_paths` maps batch_size -> path."""

    def __init__(self, dump_paths: Dict[int, Path]):
        self._dump_paths = dump_paths

    def run(self, requests: List[PromptRequest], batch_size: int) -> Dict[str, EngineSequence]:
        if batch_size not in self._dump_paths:
            raise KeyError(
                f"JSONDumpAdapter has no dump configured for batch_size={batch_size} "
                f"(configured: {sorted(self._dump_paths)})"
            )
        path = self._dump_paths[batch_size]
        with open(path, "r") as f:
            raw = json.load(f)

        out: Dict[str, EngineSequence] = {}
        for req in requests:
            if req.prompt_id not in raw:
                continue  # caller (ac3_runner) reports missing prompts explicitly; no guessing
            entry = raw[req.prompt_id]
            out[req.prompt_id] = EngineSequence(
                token_ids=_extract_token_ids(entry, req.prompt_id, str(path)),
                topk_logits=_extract_topk_logits(entry),
            )
        return out


class StaticMappingAdapter(EngineAdapter):
    """Wraps an in-memory `{batch_size: {prompt_id: EngineSequence}}` mapping. Used to plug
    already-materialized results (e.g. the vLLM smoke artifact via `load_vllm_smoke`, or
    hand-built fixtures in tests) into the same `EngineAdapter` contract without a JSON
    round-trip. Returns only the requested prompt ids that are actually present — missing ones
    are surfaced by the caller (`ac3_runner`), never guessed at here.
    """

    def __init__(self, mapping: Dict[int, Dict[str, EngineSequence]]):
        self._mapping = mapping

    def run(self, requests: List[PromptRequest], batch_size: int) -> Dict[str, EngineSequence]:
        by_prompt = self._mapping.get(batch_size, {})
        wanted = {r.prompt_id for r in requests}
        return {pid: seq for pid, seq in by_prompt.items() if pid in wanted}


def load_vllm_smoke(path: Path) -> Dict[str, EngineSequence]:
    """Reshapes the committed `vllm_smoke_result.json` (one prompt, top-level object, key
    `output_ids`, no logits) into the `{prompt_id: EngineSequence}` shape the harness expects.
    Real fields only — nothing here is fabricated or padded to look like more data than the
    one-prompt smoke artifact actually contains.
    """
    with open(path, "r") as f:
        raw = json.load(f)
    prompt_id = raw["prompt_id"]
    return {prompt_id: EngineSequence(token_ids=raw["output_ids"], topk_logits=None)}
