"""MTP per-iteration embedding-input builder.

Wraps task ``mtp_build_embed_input`` (``mtp_build_embed_input_kernel``)
in ``include/mirage/persistent_kernel/tasks/speculative_decoding/mtp_token_ops.cuh``.
For positions ``[0..mbt-2]`` reads the shifted prompt tokens from
``runtime_config.tokens[step+i+1]``; position ``mbt-1`` reads the
current main-model argmax. Result is the ``mtp_input_tokens`` buffer
fed to MTP's embedding lookup.
"""
from __future__ import annotations

from typing import Optional

from .._base import BlockDim, GridDim, MPKModule
from ...context import current_pk

from ....core import DTensor


__all__ = ["MTPBuildEmbedInput"]


class MTPBuildEmbedInput(MPKModule):
    """Populate ``mtp_input_tokens: (mbt, 1) int64`` per MTP iteration.

    Wraps task ``mtp_build_embed_input``. Params:
    ``[batch_size, max_seq_len]``. Kernel also reads ``qo_indptr`` /
    ``request_ids`` and clamps to ``qo_len in [1, 8]``.
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

    def forward(self, *args, **kwargs):
        """No plain-PyTorch reference: the kernel reads
        ``runtime_config.tokens`` and ``runtime_config.step``. Use
        test-mode for end-to-end validation."""
        raise NotImplementedError(
            "MTPBuildEmbedInput.forward(): runtime meta-tensors required."
        )

    def auto_grid_dim(self, *_: DTensor) -> GridDim:
        """``(1, 1, 1)`` — one CTA strides ``qo_len`` positions."""
        return (1, 1, 1)

    def compile(
        self,
        output_tokens: DTensor,
        mtp_input_tokens: DTensor,
        *,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ) -> DTensor:
        """Register one ``mtp_build_embed_input`` task.

        Tensor contract:
          output_tokens:    (batch_size, 1) int64, dense. Main argmax
                            for the current MTP iteration.
          mtp_input_tokens: (batch_size, 1) int64, dense. Output; the
                            token IDs to embed for MTP's next step.
        Params: ``[batch_size, max_seq_len]``.

        Notes: kernel reads ``runtime_config.tokens`` (full sequence
        buffer), ``runtime_config.step``, ``qo_indptr_buffer`` and
        ``request_ids``; clamped to ``qo_len in [1, 8]``.
        """
        pk = current_pk()

        if grid_dim is None:
            grid_dim = self.auto_grid_dim(output_tokens, mtp_input_tokens)
        if block_dim is None:
            block_dim = self.default_block_dim()

        from ....core import CyTBGraph
        from ....kernel import TBGraph

        params = [self.batch_size, self.max_seq_len]
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(output_tokens, (-1, -1, -1), -1, True)
        tb_graph.new_input(mtp_input_tokens, (-1, -1, -1), -1, True)
        pk.kn_graph.customized([output_tokens, mtp_input_tokens], tb_graph)
        pk.kn_graph.register_task(tb_graph, "mtp_build_embed_input", params)
        return mtp_input_tokens
