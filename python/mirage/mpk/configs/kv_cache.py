"""KVCacheConfig — paged KV cache shape knobs.

User-facing fields only: ``max_num_pages``, ``page_size``, and the cache
``dtype``. The full pool tensor shape
``(num_hidden_layers, max_num_pages, page_size, num_kv_heads_per_rank,
head_dim)`` is computed by ``PersistentKernel.build_from_config`` from
HFConfig + ParallelConfig + KVCacheConfig; users never spell it out.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class KVCacheConfig:
    max_num_pages: int = 16
    page_size: int = 4096
    dtype: torch.dtype = torch.bfloat16

    @property
    def total_slots(self) -> int:
        return self.max_num_pages * self.page_size
