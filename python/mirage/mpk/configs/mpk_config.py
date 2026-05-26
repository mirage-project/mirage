"""MPKConfig — top-level aggregator of all PersistentKernel sub-configs.

Holds five sub-configs and a single ``validate()`` that catches the
common user errors (TP not dividing head counts, batched-requests
exceeding the seq-length cap, KV pool too small for the max sequence)
*before* PK construction so the failure mode is a clear Python
exception rather than a kernel crash.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .hf import HFConfig
from .kv_cache import KVCacheConfig
from .parallel import ParallelConfig
from .runtime import RuntimeConfig
from .spec_decode import SpecDecodeConfig


@dataclass
class MPKConfig:
    hf: HFConfig
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    kv_cache: KVCacheConfig = field(default_factory=KVCacheConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    spec_decode: Optional[SpecDecodeConfig] = None

    def validate(self) -> None:
        """Catch the common user errors before PK construction."""
        hf = self.hf
        pc = self.parallel
        kvc = self.kv_cache
        rt = self.runtime

        # TP divides head counts.
        num_q = getattr(hf, "num_attention_heads", None)
        num_kv = getattr(hf, "num_key_value_heads", None)
        if num_q is not None and num_q % pc.tp_size != 0:
            raise ValueError(
                f"MPKConfig.validate: num_attention_heads ({num_q}) is not "
                f"divisible by tp_size ({pc.tp_size}).",
            )
        if num_kv is not None and num_kv % pc.tp_size != 0:
            raise ValueError(
                f"MPKConfig.validate: num_key_value_heads ({num_kv}) is not "
                f"divisible by tp_size ({pc.tp_size}).",
            )

        # Batching fits within the seq-length cap.
        if rt.max_num_batched_requests > rt.max_seq_length:
            raise ValueError(
                f"MPKConfig.validate: max_num_batched_requests "
                f"({rt.max_num_batched_requests}) must be ≤ max_seq_length "
                f"({rt.max_seq_length}).",
            )

        # KV pool covers the longest possible sequence.
        pool_slots = kvc.max_num_pages * kvc.page_size
        if pool_slots < rt.max_seq_length:
            raise ValueError(
                f"MPKConfig.validate: KV pool ({kvc.max_num_pages} pages "
                f"× {kvc.page_size}/page = {pool_slots} slots) is "
                f"smaller than max_seq_length ({rt.max_seq_length}).",
            )
