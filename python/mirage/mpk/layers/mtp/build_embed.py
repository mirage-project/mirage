"""MTP embedding-input builder — populate per-iteration token buffer.

Catalog wrapper around
:meth:`PersistentKernel.mtp_build_embed_input_layer` (task
``mtp_build_embed_input``). Builds the ``mtp_input_tokens`` buffer that
MTP feeds into its embedding lookup each iteration.

vLLM-aligned semantics (see ``vllm/v1/spec_decode/eagle.py`` L666-669):

* positions ``[0..mbt-2]`` of ``mtp_input_tokens`` read from the
  shifted ground-truth prompt tokens
  (``runtime_config.tokens[step[0] + i + 1]``) — i.e. teacher-forced
  inputs during prefill / early decode.
* position ``mbt - 1`` reads from ``output_tokens[mbt - 1]`` — the
  current iteration's main-model argmax token.

Runtime-metadata dependence
---------------------------

The kernel reads ``runtime_config.tokens`` (the rolling tokens buffer)
and ``runtime_config.step`` directly. They are MPK runtime meta-tensors
and are NOT exposed as DTensor inputs to this layer. Consequently, a
plain PyTorch reference of :meth:`forward` is not feasible without
recreating the runtime — :meth:`forward` raises ``NotImplementedError``.
"""
from __future__ import annotations

from typing import Optional, Tuple

from .._base import BlockDim, GridDim, MPKModule
from ...context import current_pk

from ....core import DTensor


__all__ = ["MTPBuildEmbedInput"]


class MTPBuildEmbedInput(MPKModule):
    """Populate ``mtp_input_tokens`` per MTP iteration.

    Wraps :meth:`PersistentKernel.mtp_build_embed_input_layer`.

    Args:
        batch_size:  First dim of ``output_tokens`` / ``mtp_input_tokens``.
            Baked into the task params.
        max_seq_len: Per-request stride into the runtime ``tokens``
            buffer. Baked into params.
        prefix:      vLLM/HF state_dict prefix.
    """

    def __init__(
        self,
        batch_size: int,
        max_seq_len: int,
        *,
        prefix: str = "",
    ) -> None:
        super().__init__(prefix=prefix)
        if batch_size <= 0:
            raise ValueError(
                f"MTPBuildEmbedInput batch_size must be positive; got {batch_size}"
            )
        if max_seq_len <= 0:
            raise ValueError(
                f"MTPBuildEmbedInput max_seq_len must be positive; got {max_seq_len}"
            )
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len

    # ------------------------------------------------------------------
    # PyTorch reference — depends on runtime meta-tensors.
    # ------------------------------------------------------------------
    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "MTPBuildEmbedInput.forward() has no plain-PyTorch reference: "
            "the kernel reads runtime_config.tokens and "
            "runtime_config.step (MPK runtime meta-tensors) to fill "
            "positions [0..mbt-2]. Use test-mode for end-to-end "
            "validation."
        )

    # ------------------------------------------------------------------
    # Grid heuristic.
    # ------------------------------------------------------------------
    def auto_grid_dim(self, *_: DTensor) -> GridDim:
        """Default grid ``(1, 1, 1)``.

        The kernel iterates over the ``mbt`` positions internally per
        request. One CTA is sufficient; matches the DeepSeek V3 builder
        caller.
        """
        return (1, 1, 1)

    # ------------------------------------------------------------------
    # MPK compile path.
    # ------------------------------------------------------------------
    def compile(
        self,
        output_tokens: DTensor,
        mtp_input_tokens: DTensor,
        *,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ) -> DTensor:
        """Register one ``mtp_build_embed_input`` task on the active PK.

        Args:
            output_tokens:    ``(mbt, 1)`` int64 — main-model argmax.
            mtp_input_tokens: ``(mbt, 1)`` int64 — MTP embed input
                              buffer the kernel writes.
            grid_dim:         Override; ``None`` -> :meth:`auto_grid_dim`.
            block_dim:        Override; ``None`` -> :meth:`default_block_dim`.

        Returns:
            ``mtp_input_tokens`` (consumed for its side effect).
        """
        pk = current_pk()

        if grid_dim is None:
            grid_dim = self.auto_grid_dim(output_tokens, mtp_input_tokens)
        if block_dim is None:
            block_dim = self.default_block_dim()

        # Inlined task registration (formerly pk.mtp_build_embed_input_layer).
        from ....core import CyTBGraph
        from ....kernel import TBGraph

        params = [self.batch_size, self.max_seq_len]
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(output_tokens, (-1, -1, -1), -1, True)
        tb_graph.new_input(mtp_input_tokens, (-1, -1, -1), -1, True)
        pk.kn_graph.customized([output_tokens, mtp_input_tokens], tb_graph)
        pk.kn_graph.register_task(tb_graph, "mtp_build_embed_input", params)
        return mtp_input_tokens
