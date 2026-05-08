"""TP + EP primitives for the DeepSeek V3 reference.

Aligned with vLLM's distributed sharding for DeepSeek V3:

- **Tensor parallel (TP)**: covers attention, dense MLP, MoE shared
  experts, and the LM head input dim. `tp_size` ranks total.
  - ColumnParallelLinear: weight `[output_size_per_partition, input_size]`.
    Forward computes `x @ W^T` locally, returns sharded output.
  - RowParallelLinear: weight `[output_size, input_size_per_partition]`.
    Forward computes `x @ W^T` locally, then `all_reduce` → replicated
    output across the TP group.
- **Expert parallel (EP)**: routed experts only. `ep_size` groups,
  each holding `n_routed_experts / ep_size` experts. Within each EP
  group, experts are further TP-sharded by `routed_tp_size = tp_size /
  ep_size`.
- All ranks share a single `world` process group of size `tp_size`.
  AllReduce on this group covers both TP combine and the cross-EP
  expert combine — this is mathematically equivalent to a separate
  EP-group all-to-all + TP all-reduce when each token's expert
  contributions are partitioned exclusively across ranks (which they
  are, by construction).

Layout for tp_size=4, ep_size=2:

    rank 0: ep_group=0  routed_tp_rank=0   experts[ 0..127]/2 partial
    rank 1: ep_group=0  routed_tp_rank=1   experts[ 0..127]/2 partial
    rank 2: ep_group=1  routed_tp_rank=0   experts[128..255]/2 partial
    rank 3: ep_group=1  routed_tp_rank=1   experts[128..255]/2 partial

Citations from `vllm/`:
- `tensor_parallel.py` — ColumnParallelLinear / RowParallelLinear semantics
- `model_executor/models/deepseek_v2.py:235-398` — MoE forward + dispatch
- `model_executor/layers/fused_moe/fused_moe.py:select_experts` —
  the topk + correction-bias routing the reference must reproduce
"""

from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ParallelConfig:
    """Per-rank view of the parallelism topology.

    For single-process / no-distributed runs, just `ParallelConfig()`
    (defaults: tp_size=1, ep_size=1, rank=0).

    For 4-GPU TP=4 EP=2:
        ParallelConfig(tp_size=4, ep_size=2, rank=<this rank>)

    Constraints (asserted in `__post_init__`):
        tp_size >= 1
        ep_size in {1, ..., tp_size}
        tp_size % ep_size == 0
        0 <= rank < tp_size
    """

    tp_size: int = 1
    ep_size: int = 1
    rank: int = 0

    def __post_init__(self) -> None:
        if self.tp_size < 1:
            raise ValueError(f"tp_size must be >= 1, got {self.tp_size}")
        if self.ep_size < 1 or self.ep_size > self.tp_size:
            raise ValueError(
                f"ep_size must be in [1, tp_size={self.tp_size}], got {self.ep_size}"
            )
        if self.tp_size % self.ep_size != 0:
            raise ValueError(
                f"tp_size={self.tp_size} must be divisible by "
                f"ep_size={self.ep_size}"
            )
        if not (0 <= self.rank < self.tp_size):
            raise ValueError(
                f"rank must be in [0, tp_size={self.tp_size}), got {self.rank}"
            )

    @property
    def routed_tp_size(self) -> int:
        """TP factor within each EP group (for routed-expert linears)."""
        return self.tp_size // self.ep_size

    @property
    def ep_rank(self) -> int:
        """Which EP group this rank belongs to."""
        return self.rank // self.routed_tp_size

    @property
    def routed_tp_rank(self) -> int:
        """Within this rank's EP group, the TP index."""
        return self.rank % self.routed_tp_size

    def num_local_routed_experts(self, n_total: int) -> int:
        return n_total // self.ep_size

    def first_local_routed_expert(self, n_total: int) -> int:
        return self.ep_rank * self.num_local_routed_experts(n_total)

    def is_single_process(self) -> bool:
        return self.tp_size == 1


def init_distributed_if_needed(pcfg: ParallelConfig, device: str) -> None:
    """Initialise NCCL process group if tp_size > 1 and not already done."""
    if pcfg.is_single_process():
        return
    if dist.is_initialized():
        # Verify the existing group matches.
        ws = dist.get_world_size()
        if ws != pcfg.tp_size:
            raise RuntimeError(
                f"torch.distributed already initialised with world_size={ws}, "
                f"but ParallelConfig requested tp_size={pcfg.tp_size}"
            )
        return
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
    else:
        raise RuntimeError(
            "tp_size > 1 but no torch.distributed env. Launch via "
            "`torchrun --nproc_per_node=<tp_size>` or set RANK / WORLD_SIZE / "
            "MASTER_ADDR / MASTER_PORT manually."
        )


def all_reduce_tp(x: torch.Tensor, pcfg: ParallelConfig) -> torch.Tensor:
    """All-reduce a partial output across the full TP world."""
    if pcfg.is_single_process():
        return x
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    return x


# =============================================================================
# Parallel Linear primitives
# =============================================================================
class ColumnParallelLinear(nn.Module):
    """Output dim split across `tp_size` ranks; no all-reduce.

    Stores `[out_features // tp_size, in_features]`. Forward returns
    `[..., out_features // tp_size]` — the consumer must either:
      (a) feed this sharded output to a RowParallelLinear (which
          all-reduces), OR
      (b) call `all_gather_last_dim` to materialize the full output.

    Used for: q_b_proj (split by head), kv_b_proj (split by head),
    gate_up_proj (split by intermediate dim), shared expert gate_up_proj.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        pcfg: ParallelConfig,
        bias: bool = False,
    ):
        super().__init__()
        if out_features % pcfg.tp_size != 0:
            raise ValueError(
                f"ColumnParallel out_features={out_features} must be "
                f"divisible by tp_size={pcfg.tp_size}"
            )
        self.in_features = in_features
        self.out_features = out_features
        self.out_features_per_partition = out_features // pcfg.tp_size
        self.pcfg = pcfg
        self.weight = nn.Parameter(
            torch.empty(self.out_features_per_partition, in_features)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features_per_partition))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class RowParallelLinear(nn.Module):
    """Input dim split across `tp_size` ranks; all-reduce after matmul.

    Stores `[out_features, in_features // tp_size]`. Forward expects
    sharded input `[..., in_features // tp_size]` and computes the
    partial matmul + AllReduce → full `[..., out_features]`.

    Used for: o_proj (combines per-head attn outputs), down_proj
    (combines per-partition MLP intermediate), shared expert down_proj.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        pcfg: ParallelConfig,
        bias: bool = False,
        all_reduce_after: bool = True,
    ):
        super().__init__()
        if in_features % pcfg.tp_size != 0:
            raise ValueError(
                f"RowParallel in_features={in_features} must be divisible "
                f"by tp_size={pcfg.tp_size}"
            )
        self.in_features = in_features
        self.out_features = out_features
        self.in_features_per_partition = in_features // pcfg.tp_size
        self.pcfg = pcfg
        self.all_reduce_after = all_reduce_after
        self.weight = nn.Parameter(
            torch.empty(out_features, self.in_features_per_partition)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight, None)  # bias added after AllReduce
        if self.all_reduce_after:
            out = all_reduce_tp(out, self.pcfg)
        if self.bias is not None:
            out = out + self.bias
        return out


# =============================================================================
# Routed-expert TP-sharded linear (within an EP group)
# =============================================================================
class RoutedExpertColumnParallel(nn.Module):
    """Like ColumnParallelLinear but split factor is `routed_tp_size`,
    not `tp_size`. Used for routed experts inside an EP group."""

    def __init__(
        self, in_features: int, out_features: int, pcfg: ParallelConfig
    ):
        super().__init__()
        rtp = pcfg.routed_tp_size
        if out_features % rtp != 0:
            raise ValueError(
                f"RoutedExpert ColumnParallel out_features={out_features} "
                f"must be divisible by routed_tp_size={rtp}"
            )
        self.weight = nn.Parameter(
            torch.empty(out_features // rtp, in_features)
        )
        self.pcfg = pcfg

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight)


class RoutedExpertRowParallel(nn.Module):
    """Like RowParallelLinear but split factor is `routed_tp_size`.

    All-reduce is deferred — the MoE forward does ONE global AllReduce
    across the full TP world after summing all local experts' partial
    outputs (which is mathematically equivalent to per-expert local
    AllReduce within the EP group + cross-EP combine).
    """

    def __init__(
        self, in_features: int, out_features: int, pcfg: ParallelConfig
    ):
        super().__init__()
        rtp = pcfg.routed_tp_size
        if in_features % rtp != 0:
            raise ValueError(
                f"RoutedExpert RowParallel in_features={in_features} "
                f"must be divisible by routed_tp_size={rtp}"
            )
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features // rtp)
        )
        self.pcfg = pcfg

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # NO all-reduce here; deferred to MoE.forward.
        return F.linear(x, self.weight, None)
