"""Stochastic token sampling via Gumbel-Max.

Backed by ``tasks/common/sampling.cuh`` (``sampling_from_logits_kernel``)
dispatched as task ``sampling_sm100``. Implements
``argmax(logits + Gumbel(0,1))``, equivalent to sampling from
softmax(logits). The kernel uses a stateless PRNG keyed by
``(seed, batch_idx, vocab_idx)``. Logits are bf16; **output is int32**
token ids — callers that index into an embedding table must cast to
int64.
"""
from __future__ import annotations

from typing import Any, Optional

import torch

import mirage as mi

from ._base import BlockDim, GridDim, MPKModule


__all__ = ["SamplingSM100"]


class SamplingSM100(MPKModule):
    """Gumbel-Max stochastic sampling over the vocab axis.

    Input ``(B, V)`` bf16, output ``(B, 1)`` int32. ``seed`` is baked
    into the task as a kernel parameter.
    """

    def __init__(
        self,
        *,
        seed: int = 42,
        prefix: str = "",
    ) -> None:
        super().__init__(prefix=prefix)
        self.seed = seed

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """``argmax(logits + Gumbel(0,1))`` returned as int32.

        Uses ``torch``'s default RNG, NOT the kernel's stateless hash —
        tests should compare via ``argmax`` of plain logits or replicate
        the same seed scheme for bit-equivalence.
        """
        u = torch.rand_like(logits.float()).clamp_min(1e-12)
        gumbel = -torch.log(-torch.log(u))
        tokens = (logits.float() + gumbel).argmax(dim=-1, keepdim=True)
        return tokens.to(torch.int32)

    def auto_grid_dim(self, logits: Any) -> GridDim:
        """``(batch_size, 1, 1)`` — one CTA per row."""
        return (logits.dim(0), 1, 1)

    def default_block_dim(self) -> BlockDim:
        return (128, 1, 1)

    def compile(
        self,
        logits: Any,
        *,
        output: Optional[Any] = None,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ) -> Any:
        """Register a ``sampling_sm100`` task — Gumbel-Max stochastic sampling.

        Tensor contract:
          logits: (B, V) bf16, per-row logits.
          output: (B, 1) **int32** (NOT int64 — load-bearing), sampled token id.

        Notes: kernel uses a stateless PRNG keyed by ``(seed, batch_idx,
        vocab_idx)``; ``seed`` is baked into the task as a kernel parameter.
        Callers indexing into an embedding table must cast int32 → int64.
        """
        import torch as _torch
        from .. import context as _ctx

        pk = _ctx.current_pk()

        if grid_dim is None:
            grid_dim = self.auto_grid_dim(logits)
        if block_dim is None:
            block_dim = self.default_block_dim()

        B = logits.dim(0)
        if output is None:
            out_dt = pk.new_tensor(
                dims=(B, 1),
                dtype=mi.int32,
                name=f"{self.prefix}sampled_tokens",
            )
        elif isinstance(output, _torch.Tensor):
            out_dt = pk.attach_input(
                output, name=f"{self.prefix}sampled_tokens"
            )
        else:
            out_dt = output

        from ...core import CyTBGraph
        from ...kernel import TBGraph

        assert logits.num_dims == 2
        assert out_dt.num_dims == 2
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(logits, (0, -1, -1), -1, True)
        tb_graph.new_input(out_dt, (0, -1, -1), -1, True)
        pk.kn_graph.customized([logits, out_dt], tb_graph)
        pk.kn_graph.register_task(tb_graph, "sampling_sm100", [self.seed])
        return out_dt
