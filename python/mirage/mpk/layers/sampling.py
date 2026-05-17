"""Stochastic token sampling via Gumbel-Max.

Wraps :meth:`PersistentKernel.sampling_sm100_layer` — task
``sampling_sm100``. Implements ``argmax(logits + Gumbel(0, 1))``, which
is equivalent to sampling from the softmax distribution.

The PRNG seed is baked into the task as a kernel parameter (the kernel
uses a deterministic stateless PRNG keyed by ``(seed, batch_idx,
vocab_idx)``).

Forward reference
-----------------

``forward()`` adds standard Gumbel noise to the logits and takes the
argmax over the vocab axis. The result depends on the seed; tests
should use ``argmax`` (no noise) when comparing eager-vs-compiled, or
pass the same seed and use a fixed RNG.
"""
from __future__ import annotations

from typing import Any, Optional

import torch

import mirage as mi

from ._base import BlockDim, GridDim, MPKModule


__all__ = ["SamplingSM100"]


class SamplingSM100(MPKModule):
    """Gumbel-Max stochastic sampling over the vocab axis.

    Args:
        seed: Deterministic PRNG seed baked into the kernel task. The
            kernel uses a stateless hash keyed on
            ``(seed, batch_idx, vocab_idx)``.
        prefix: Reserved. No parameters live here.
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
        """``argmax(logits + Gumbel(0, 1))``.

        The reference uses ``torch``'s default RNG (NOT the kernel's
        stateless hash). For bit-equivalent comparison the test driver
        should either compare via the same seed scheme or compare
        ``argmax`` outputs (top-1 of plain logits).

        Args:
            logits: ``(batch_size, vocab_size)`` float (any dtype).

        Returns:
            ``(batch_size, 1)`` int32 token ids. Callers that need int64
            (e.g. to index into an embedding table) should cast at the
            use site: ``tokens64 = tokens.to(torch.int64)``.
        """
        # Gumbel(0, 1) = -log(-log(U)).
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
        """Register a ``sampling_sm100`` task.

        Args:
            logits: ``(B, V)`` float DTensor.
            output: ``None`` allocates a ``(B, 1)`` int32 DTensor (the
                underlying ``kernel::sampling_from_logits_kernel<..., int>``
                writes int32 token ids; the previous int64 declaration
                left the upper 32 bits stale). ``torch.Tensor`` or
                ``DTensor`` route as the other catalog modules do.
                Downstream consumers that index into embedding tables
                should cast: ``tokens64 = output.to(torch.int64)``.
            grid_dim / block_dim: explicit overrides.

        Returns:
            ``output``.
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

        # Inlined task registration (was pk.sampling_sm100_layer).
        from ...core import CyTBGraph
        from ...kernel import TBGraph

        assert logits.num_dims == 2  # (batch_size, vocab_size)
        assert out_dt.num_dims == 2  # (batch_size, 1)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(logits, (0, -1, -1), -1, True)
        tb_graph.new_input(out_dt, (0, -1, -1), -1, True)
        pk.kn_graph.customized([logits, out_dt], tb_graph)
        pk.kn_graph.register_task(tb_graph, "sampling_sm100", [self.seed])
        return out_dt
