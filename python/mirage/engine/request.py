from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


class RequestStatus(Enum):
    WAITING = auto()   # in the waiting queue, not yet admitted to a GPU slot
    PREFILL = auto()   # admitted; GPU is processing prompt tokens
    DECODE  = auto()   # prompt done; GPU is auto-regressively decoding
    DONE    = auto()   # finished (EOS / length / stop token)


@dataclass
class SamplingParams:
    """Per-request generation constraints. None of these fields are written to
    any kernel-consumed meta_tensor; they are evaluated on the Python side in
    RequestMetadataManager.collect_outputs()."""
    max_new_tokens:  int       = 256
    eos_token_id:    int       = -1    # -1 means "no EOS check"
    stop_token_ids:  list[int] = field(default_factory=list)


class RequestMetadata:
    """Represents one inference request throughout its lifetime.

    Layout of fields by ownership:
      - prompt_token_ids, sampling_params : set by caller, never mutated
      - status, output_token_ids, finish_reason : managed by RequestMetadataManager
      - slot_id, token_row : assigned on admit(), cleared on complete()
    """

    _global_counter: int = 0

    def __init__(
        self,
        prompt_token_ids: list[int],
        sampling_params: Optional[SamplingParams] = None,
    ) -> None:
        # Stable identity across the whole lifetime of the object
        self.request_id: int = RequestMetadata._global_counter
        RequestMetadata._global_counter += 1

        self.prompt_token_ids: list[int] = prompt_token_ids
        self.sampling_params: SamplingParams = (
            sampling_params if sampling_params is not None else SamplingParams()
        )

        # Lifecycle state
        self.status: RequestStatus = RequestStatus.WAITING
        self.output_token_ids: list[int] = []
        self.finish_reason: Optional[str] = None

        # Assigned by RequestMetadataManager.admit(), cleared by complete()
        # slot_id  : index into the active[] list  (== request_ids[] slot on GPU)
        # token_row: row index in meta_tensors["tokens"] (== GPU request_id)
        self.slot_id:   Optional[int] = None
        self.token_row: Optional[int] = None

        # Prefix cache hit data — set by admit() when a KV prefix is reused.
        # prefix_len  : number of tokens whose KV data is already in the cache.
        # prefix_pages: GPU page indices for those KV pages.
        # Requires kernel patch P2 to actually skip recomputing those tokens.
        self.prefix_len:   int       = 0
        self.prefix_pages: list[int] = []

    @property
    def prompt_len(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def num_output_tokens(self) -> int:
        return len(self.output_token_ids)

    def __repr__(self) -> str:
        return (
            f"RequestMetadata(id={self.request_id}, "
            f"status={self.status.name}, "
            f"prompt_len={self.prompt_len}, "
            f"output_len={self.num_output_tokens})"
        )
