"""MTP verify-phase token-buffer setup.

Wraps task ``mtp_prepare_verify`` (``mtp_prepare_verify_input_kernel``)
in ``include/mirage/persistent_kernel/tasks/speculative_decoding/mtp_token_ops.cuh``.
Lays out ``[main_token, draft_0..draft_{K-1}]`` into
``tokens_buffer[req, step+1 : step+K+2]`` so the target model's next
forward pass sees ``num_draft_tokens + 1`` candidate tokens. The kernel
also writes ``num_new_tokens[req] = NUM_DRAFT + 1`` (clamped).
"""
from __future__ import annotations

from typing import Optional

from .._base import BlockDim, GridDim, MPKModule
from ...context import current_pk

from ....core import DTensor


__all__ = ["MTPPrepareVerify"]


class MTPPrepareVerify(MPKModule):
    """Write main + draft tokens into the rolling token buffer.

    Wraps task ``mtp_prepare_verify``. Params
    ``[num_draft_tokens, max_seq_len]``; also reads ``qo_indptr`` /
    ``request_ids`` and skips chunk-prefill slots (``qo_len > 8``).
    """

    def __init__(
        self,
        num_draft_tokens: int,
        max_seq_len: int,
        *,
        prefix: str = "",
    ) -> None:
        super().__init__(prefix=prefix)
        if num_draft_tokens <= 0:
            raise ValueError(
                f"MTPPrepareVerify num_draft_tokens must be positive; "
                f"got {num_draft_tokens}"
            )
        if max_seq_len <= 0:
            raise ValueError(
                f"MTPPrepareVerify max_seq_len must be positive; "
                f"got {max_seq_len}"
            )
        self.num_draft_tokens = num_draft_tokens
        self.max_seq_len = max_seq_len

    def forward(self, *args, **kwargs):
        """No plain-PyTorch reference: the kernel reads MPK meta-tensors
        (``step``, ``qo_indptr``, ``request_ids``) and writes
        ``num_new_tokens``. Use test-mode for end-to-end validation."""
        raise NotImplementedError(
            "MTPPrepareVerify.forward(): runtime meta-tensors required."
        )

    def auto_grid_dim(self, *_: DTensor) -> GridDim:
        """``(1, 1, 1)`` — one CTA writes ``num_draft_tokens + 1`` slots."""
        return (1, 1, 1)

    def compile(
        self,
        main_token: DTensor,
        draft_tokens: DTensor,
        tokens_buffer: DTensor,
        step: DTensor,
        num_new_tokens: DTensor,
        *,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ) -> DTensor:
        """Register one ``mtp_prepare_verify`` task.

        Tensor contract:
          main_token:     (batch_size, 1) int64, dense. Main argmax.
          draft_tokens:   (batch_size, num_draft_tokens) int64, dense.
          tokens_buffer:  (num_requests, max_seq_len) int64, dense.
                          In/out: writes ``[req, step+1 : step+K+2]``.
          step:           (num_requests,) int32. Per-request decode step.
          num_new_tokens: (num_requests,) int32. Output;
                          set to ``NUM_DRAFT + 1`` (clamped).
        Params: ``[num_draft_tokens, max_seq_len]``.

        Notes: kernel reads ``runtime_config.qo_indptr_buffer`` /
        ``request_ids`` and skips chunk-prefill (qo_len>8 or <1).
        """
        pk = current_pk()

        if grid_dim is None:
            grid_dim = self.auto_grid_dim(
                main_token, draft_tokens, tokens_buffer, step, num_new_tokens
            )
        if block_dim is None:
            block_dim = self.default_block_dim()

        from ....core import CyTBGraph
        from ....kernel import TBGraph

        params = [self.num_draft_tokens, self.max_seq_len]
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(main_token, (-1, -1, -1), -1, True)
        tb_graph.new_input(draft_tokens, (-1, -1, -1), -1, True)
        tb_graph.new_input(tokens_buffer, (-1, -1, -1), -1, True)
        tb_graph.new_input(step, (-1, -1, -1), -1, True)
        tb_graph.new_input(num_new_tokens, (-1, -1, -1), -1, True)
        pk.kn_graph.customized(
            [main_token, draft_tokens, tokens_buffer, step, num_new_tokens],
            tb_graph,
        )
        pk.kn_graph.register_task(tb_graph, "mtp_prepare_verify", params)
        return tokens_buffer
