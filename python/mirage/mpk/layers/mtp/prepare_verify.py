"""MTP verify-phase setup — write candidate tokens into the rolling buffer.

Catalog wrapper around :meth:`PersistentKernel.mtp_prepare_verify_layer`
(task ``mtp_prepare_verify``). Runs once per MTP iteration, AFTER the
main-model argmax has produced ``main_token`` and the MTP draft loop has
accumulated ``draft_tokens``, but BEFORE the target-model verify pass.

What the kernel does
--------------------

Conceptually::

    pos = step[0] + num_new_tokens[0]    # current write head into tokens_buffer
    tokens_buffer[pos]                   = main_token[mbt - 1]
    tokens_buffer[pos+1 : pos+1+num_draft_tokens] = draft_tokens

so the next target-model step finds ``num_draft_tokens + 1`` candidate
tokens laid out in the same per-request token buffer that the rest of
the kernel reads via ``runtime_config.tokens``.

Runtime-metadata dependence
---------------------------

Both ``step`` and ``num_new_tokens`` are MPK runtime meta-tensors (the
per-request decode position and per-iteration new-token count). The
kernel reads them directly to compute the write offset into the rolling
``tokens_buffer``. A standalone PyTorch reference would need the entire
meta-tensor stack — so :meth:`forward` raises ``NotImplementedError``.
"""
from __future__ import annotations

from typing import Optional, Tuple

from .._base import BlockDim, GridDim, MPKModule
from ...context import current_pk

from ....core import DTensor


__all__ = ["MTPPrepareVerify"]


class MTPPrepareVerify(MPKModule):
    """Write main-model token + draft tokens into the rolling token buffer.

    Wraps :meth:`PersistentKernel.mtp_prepare_verify_layer`.

    Args:
        num_draft_tokens: Number of draft tokens MTP emits per
            iteration. Baked into the task params.
        max_seq_len:      Max per-request sequence length — the stride
            into ``tokens_buffer`` per request. Baked into params.
        prefix:           vLLM/HF state_dict prefix; no weights here, so
            this is only used as a debug uniquifier.
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

    # ------------------------------------------------------------------
    # PyTorch reference — depends on runtime meta-tensors; not feasible.
    # ------------------------------------------------------------------
    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "MTPPrepareVerify.forward() has no plain-PyTorch reference: "
            "the kernel reads MPK runtime meta-tensors (step, "
            "num_new_tokens) to compute the write offset into the "
            "rolling tokens buffer. Use the test-mode driver to "
            "validate end-to-end."
        )

    # ------------------------------------------------------------------
    # Grid heuristic.
    # ------------------------------------------------------------------
    def auto_grid_dim(self, *_: DTensor) -> GridDim:
        """Default grid ``(1, 1, 1)``.

        The kernel walks ``num_draft_tokens + 1`` writes serially per
        request; a single CTA is sufficient. Matches the DeepSeek V3
        builder caller.
        """
        return (1, 1, 1)

    # ------------------------------------------------------------------
    # MPK compile path.
    # ------------------------------------------------------------------
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
        """Register one ``mtp_prepare_verify`` task on the active PK.

        Args:
            main_token:     ``(mbt, 1)`` int64 — main-model argmax.
            draft_tokens:   ``(batch, num_draft_tokens)`` int64 — MTP
                            draft outputs.
            tokens_buffer:  ``(batch, max_seq_len)`` int64 — rolling
                            token buffer the kernel writes into.
            step:           Runtime meta-tensor: per-request decode
                            position.
            num_new_tokens: Runtime meta-tensor: per-iteration count of
                            new tokens.
            grid_dim:       Override; ``None`` -> :meth:`auto_grid_dim`.
            block_dim:      Override; ``None`` -> :meth:`default_block_dim`.

        Returns:
            ``tokens_buffer`` (consumed for its side effect).
        """
        pk = current_pk()

        if grid_dim is None:
            grid_dim = self.auto_grid_dim(
                main_token, draft_tokens, tokens_buffer, step, num_new_tokens
            )
        if block_dim is None:
            block_dim = self.default_block_dim()

        # Inlined task registration (formerly pk.mtp_prepare_verify_layer).
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
