"""TP-aware Linear variants — vLLM-style.

Four classes mirroring vLLM's ``Linear`` family:

* :class:`ColumnParallelLinear`   — output dim sharded across ``tp_size``.
* :class:`RowParallelLinear`      — input  dim sharded across ``tp_size``;
  caller is responsible for a follow-up ``AllReduce``.
* :class:`MergedColumnParallelLinear` — output dim is the concatenation of
  several logical projections (e.g. ``gate_proj`` + ``up_proj``); each
  logical projection is sharded along the output dim and the per-shard
  data is loaded via ``shard_id ∈ {0, 1, ...}``.
* :class:`QKVParallelLinear`      — fused q/k/v output with GQA-aware
  per-head sharding; per-shard data loaded via ``shard_id ∈ {"q","k","v"}``.

Each class allocates its ``nn.Parameter`` at the **sharded** shape in
``__init__`` (vLLM convention — saves ``tp_size``× GPU memory). The
parameter has a ``weight_loader`` callback attached that knows how to
narrow a full unsharded view (typically a safetensors mmap view) into the
correct rank slice. ``MPKModule.load_weights`` calls these callbacks
automatically when it routes a weight by HF state_dict key path.

All TP-aware leaves MUST be constructed inside ``with pk.compile_scope():``
— they read ``current_pk().parallel_config`` to determine ``tp_size``
and ``tp_rank``. Outside compile scope, ``current_pk()`` raises a clear
``RuntimeError``.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn

from ... import context as _ctx
from .linear import Linear
from .linear_with_residual import LinearWithResidual


__all__ = [
    "ColumnParallelLinear",
    "RowParallelLinear",
    "RowParallelLinearWithResidual",
    "MergedColumnParallelLinear",
    "QKVParallelLinear",
]


# ---------------------------------------------------------------------------
# ColumnParallelLinear
# ---------------------------------------------------------------------------


class ColumnParallelLinear(Linear):
    """Linear with the output dim sharded across ``tp_size`` ranks.

    Weight shape on each rank: ``(out_features // tp_size, in_features)``.
    ``forward()`` returns this rank's slice of shape
    ``(*, out_features // tp_size)`` — the caller is responsible for
    handling cross-rank semantics (typically a downstream ``RowParallel``
    + ``AllReduce`` consumes the slices).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        *,
        prefix: str = "",
    ) -> None:
        pc = _ctx.current_pk().parallel_config
        if out_features % pc.tp_size != 0:
            raise ValueError(
                f"ColumnParallelLinear: out_features ({out_features}) must "
                f"be divisible by tp_size ({pc.tp_size})."
            )
        # Allocate at sharded shape by feeding the smaller out_features to
        # the parent Linear. ``forward()``, ``auto_grid_dim``, and
        # ``compile()`` then operate on the local slab without further
        # change.
        super().__init__(
            in_features, out_features // pc.tp_size, bias=bias, prefix=prefix
        )
        self.out_features_full = out_features
        self._tp_size = pc.tp_size
        self._tp_rank = pc.tp_rank
        self.weight.weight_loader = self._weight_loader

    def _weight_loader(
        self, param: nn.Parameter, loaded_weight: torch.Tensor,
    ) -> None:
        """Narrow the full ``(out_features_full, in_features)`` source along
        dim 0 to this rank's ``(out_features_full // tp_size, in_features)``
        slice and ``copy_`` into ``param.data``.
        """
        shard_size = param.shape[0]
        start = self._tp_rank * shard_size
        param.data.copy_(loaded_weight.narrow(0, start, shard_size))


# ---------------------------------------------------------------------------
# RowParallelLinear / RowParallelLinearWithResidual
# ---------------------------------------------------------------------------


class RowParallelLinear(Linear):
    """Linear with the input (reduction) dim sharded across ``tp_size`` ranks.

    Weight shape on each rank: ``(out_features, in_features // tp_size)``.
    ``forward()`` returns this rank's PARTIAL contribution
    ``F.linear(x_slice, self.weight)`` of full shape ``(*, out_features)``;
    the caller is responsible for the ``AllReduce`` that sums across ranks.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        *,
        prefix: str = "",
    ) -> None:
        pc = _ctx.current_pk().parallel_config
        if in_features % pc.tp_size != 0:
            raise ValueError(
                f"RowParallelLinear: in_features ({in_features}) must be "
                f"divisible by tp_size ({pc.tp_size})."
            )
        super().__init__(
            in_features // pc.tp_size, out_features, bias=bias, prefix=prefix
        )
        self.in_features_full = in_features
        self._tp_size = pc.tp_size
        self._tp_rank = pc.tp_rank
        self.weight.weight_loader = self._weight_loader

    def _weight_loader(
        self, param: nn.Parameter, loaded_weight: torch.Tensor,
    ) -> None:
        shard_size = param.shape[1]
        start = self._tp_rank * shard_size
        param.data.copy_(loaded_weight.narrow(1, start, shard_size))


class RowParallelLinearWithResidual(LinearWithResidual):
    """Like :class:`RowParallelLinear` but with a fused residual add.

    The underlying ``linear_with_residual`` kernel respects an
    ``enable_residual`` task param that the base class drives to 0 on
    non-root ranks — so the residual is added exactly once across the
    shard regardless of how many ranks contribute.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        prefix: str = "",
    ) -> None:
        pc = _ctx.current_pk().parallel_config
        if in_features % pc.tp_size != 0:
            raise ValueError(
                f"RowParallelLinearWithResidual: in_features ({in_features}) "
                f"must be divisible by tp_size ({pc.tp_size})."
            )
        super().__init__(
            in_features // pc.tp_size, out_features, prefix=prefix
        )
        self.in_features_full = in_features
        self._tp_size = pc.tp_size
        self._tp_rank = pc.tp_rank
        self.weight.weight_loader = self._weight_loader

    def _weight_loader(
        self, param: nn.Parameter, loaded_weight: torch.Tensor,
    ) -> None:
        shard_size = param.shape[1]
        start = self._tp_rank * shard_size
        param.data.copy_(loaded_weight.narrow(1, start, shard_size))


# ---------------------------------------------------------------------------
# MergedColumnParallelLinear
# ---------------------------------------------------------------------------


class MergedColumnParallelLinear(ColumnParallelLinear):
    """ColumnParallel for a fused output that concatenates several logical
    projections (e.g. ``gate_proj`` + ``up_proj``).

    ``output_sizes`` lists the full (un-sharded) output size of each
    logical projection. The local weight is
    ``(sum(output_sizes) // tp_size, in_features)``, laid out as the
    per-rank slice of each logical projection concatenated in order:
    ``[gate_shard | up_shard]`` for typical MLP fusion.

    The ``weight_loader`` takes ``shard_id`` (integer index into
    ``output_sizes``) and writes the corresponding loaded weight into the
    matching sub-range of the local fused buffer.
    """

    def __init__(
        self,
        in_features: int,
        output_sizes: List[int],
        bias: bool = False,
        *,
        prefix: str = "",
    ) -> None:
        if not output_sizes:
            raise ValueError("MergedColumnParallelLinear: output_sizes is empty")
        out_features = sum(output_sizes)
        super().__init__(
            in_features, out_features, bias=bias, prefix=prefix
        )
        self.output_sizes = list(output_sizes)
        # Local (per-rank) sizes — each logical projection is sharded.
        self.local_output_sizes = [s // self._tp_size for s in output_sizes]
        # Offsets into the local fused buffer.
        offsets = [0]
        for s in self.local_output_sizes:
            offsets.append(offsets[-1] + s)
        self.local_offsets = offsets
        # Override the loader to require a shard_id (overrides the
        # base ColumnParallel's no-shard-id loader).
        self.weight.weight_loader = self._merged_weight_loader

    def _merged_weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        shard_id: int = 0,
    ) -> None:
        if not (0 <= shard_id < len(self.output_sizes)):
            raise ValueError(
                f"MergedColumnParallelLinear: shard_id ({shard_id}) out of "
                f"range [0, {len(self.output_sizes)})"
            )
        full_size = self.output_sizes[shard_id]
        local_size = self.local_output_sizes[shard_id]
        start_full = self._tp_rank * local_size
        narrow_full = loaded_weight.narrow(0, start_full, local_size)
        # Write into the local fused buffer at this shard_id's slot.
        local_off = self.local_offsets[shard_id]
        param.data.narrow(0, local_off, local_size).copy_(narrow_full)


# ---------------------------------------------------------------------------
# QKVParallelLinear
# ---------------------------------------------------------------------------


class QKVParallelLinear(ColumnParallelLinear):
    """ColumnParallel for a fused QKV output with GQA support.

    Constructor takes ``head_dim``, ``total_num_heads`` (Q heads), and
    ``total_num_kv_heads`` (K/V heads, ``= num_heads`` for non-GQA). The
    local weight is
    ``((num_local_q + 2*num_local_kv) * head_dim, in_features)`` where
    ``num_local_q = total_num_heads // tp_size`` and
    ``num_local_kv = total_num_kv_heads // tp_size``.

    Layout of the local fused output:
    ``[ q[0..num_local_q*head_dim) | k[..) | v[..) ]``.

    ``weight_loader(param, loaded_weight, shard_id)`` with
    ``shard_id ∈ {"q","k","v"}`` writes the rank-narrowed source into the
    matching sub-range of the local fused buffer.
    """

    def __init__(
        self,
        in_features: int,
        head_dim: int,
        total_num_heads: int,
        total_num_kv_heads: int,
        bias: bool = False,
        *,
        prefix: str = "",
    ) -> None:
        pc = _ctx.current_pk().parallel_config
        if total_num_heads % pc.tp_size != 0:
            raise ValueError(
                f"QKVParallelLinear: total_num_heads ({total_num_heads}) "
                f"must be divisible by tp_size ({pc.tp_size})."
            )
        if total_num_kv_heads % pc.tp_size != 0:
            raise ValueError(
                f"QKVParallelLinear: total_num_kv_heads ({total_num_kv_heads}) "
                f"must be divisible by tp_size ({pc.tp_size})."
            )
        out_features = (total_num_heads + 2 * total_num_kv_heads) * head_dim
        super().__init__(
            in_features, out_features, bias=bias, prefix=prefix
        )
        self.head_dim = head_dim
        self.total_num_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads
        self.num_local_q_heads = total_num_heads // self._tp_size
        self.num_local_kv_heads = total_num_kv_heads // self._tp_size
        self.q_size_local = self.num_local_q_heads * head_dim
        self.kv_size_local = self.num_local_kv_heads * head_dim
        self.weight.weight_loader = self._qkv_weight_loader

    def _qkv_weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        shard_id: str = "q",
    ) -> None:
        if shard_id == "q":
            local_off = 0
            local_size = self.q_size_local
            full_per_rank = self.q_size_local
        elif shard_id == "k":
            local_off = self.q_size_local
            local_size = self.kv_size_local
            full_per_rank = self.kv_size_local
        elif shard_id == "v":
            local_off = self.q_size_local + self.kv_size_local
            local_size = self.kv_size_local
            full_per_rank = self.kv_size_local
        else:
            raise ValueError(
                f"QKVParallelLinear: shard_id must be one of "
                f"{{'q','k','v'}}, got {shard_id!r}"
            )
        start_full = self._tp_rank * full_per_rank
        narrow_full = loaded_weight.narrow(0, start_full, full_per_rank)
        param.data.narrow(0, local_off, local_size).copy_(narrow_full)
