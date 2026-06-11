"""RuntimeConfig — PK scheduler + batching + misc runtime knobs.

Everything that's neither model architecture, parallel topology, KV
cache shape, nor speculative decoding lives here. ``num_workers`` /
``num_local_schedulers`` are optional — when ``None``, the factory
auto-derives them from the local GPU via
``mirage.get_configurations_from_gpu(0)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class RuntimeConfig:
    # PK runtime
    mode: str = "offline"
    max_seq_length: int = 512
    max_num_batched_requests: int = 1
    max_num_batched_tokens: int = 8
    test_mode: bool = False
    # eos_token_id semantics:
    #   None: auto-fill from HF config at build_from_config time
    #     -1: never stop on EOS (ignore_eos)
    #   else: explicit integer
    eos_token_id: Optional[int] = None
    trace_name: str = ""
    use_cutlass_kernel: bool = True
    output_dir: Optional[str] = None

    # Schedulers (auto-derived from GPU if not specified)
    num_workers: Optional[int] = None
    num_local_schedulers: Optional[int] = None
    num_remote_schedulers: int = 0

    # Profiling
    profiler_tensor: Optional[torch.Tensor] = None
