"""N-gram lookup for prompt-lookup speculative decode.

One class :class:`FindNgram` with a ``scope`` kwarg picking between two
pk methods:

* ``scope="partial"`` -> :meth:`PersistentKernel.find_ngram_partial_layer`
  -> task ``find_ngram_partial``. Scans the request's existing tokens
  for the most recent matching n-gram and emits a per-task partial
  result tensor of shape ``(batch, num_tasks)``. ``grid.y`` is the
  task fan-out.
* ``scope="global"``  -> :meth:`PersistentKernel.find_ngram_global_layer`
  -> task ``find_ngram_global``. Reduces all partial results into a
  single ``(batch, spec_length + 1)`` speculative-token vector — the
  draft tokens for the next iteration.

This is the prompt-lookup spec-decode handler (no MTP / no draft model
— just look for a repeated prefix in the request's own tokens). It is
NOT a DeepSeek V3 path; it lives next to the MTP helpers because it is
part of the same speculative-decode infrastructure.

Runtime-metadata dependence
---------------------------

Both kernels read MPK's rolling ``tokens`` and ``step`` meta-tensors to
know where to scan from. A plain PyTorch reference cannot model the
scan without those — :meth:`forward` therefore raises
``NotImplementedError``.
"""
from __future__ import annotations

from typing import Literal, Optional, Tuple

from .._base import BlockDim, GridDim, MPKModule
from ...context import current_pk

from ....core import DTensor


__all__ = ["FindNgram"]


NgramScope = Literal["partial", "global"]


class FindNgram(MPKModule):
    """N-gram lookup over the request's own tokens.

    Args:
        ngram_size:  Length of the n-gram pattern to search for. Baked
            into the task params. (Default 3 mirrors the pk method.)
        spec_length: For ``scope="global"`` only — number of draft
            tokens to emit after the matched suffix (so the global
            output is ``(batch, spec_length + 1)``).
        scope:       ``"partial"`` (per-task scan) or ``"global"``
            (reduce + emit draft tokens).
        prefix:      vLLM/HF state_dict prefix.
    """

    def __init__(
        self,
        ngram_size: int = 3,
        spec_length: int = 5,
        *,
        scope: NgramScope = "partial",
        prefix: str = "",
    ) -> None:
        super().__init__(prefix=prefix)
        if scope not in ("partial", "global"):
            raise ValueError(
                f"FindNgram scope must be 'partial' or 'global'; got {scope!r}"
            )
        if ngram_size <= 0:
            raise ValueError(
                f"FindNgram ngram_size must be positive; got {ngram_size}"
            )
        if scope == "global" and spec_length <= 0:
            raise ValueError(
                f"FindNgram(scope='global') spec_length must be positive; "
                f"got {spec_length}"
            )
        self.ngram_size = ngram_size
        self.spec_length = spec_length
        self.scope = scope

    # ------------------------------------------------------------------
    # PyTorch reference — runtime-driven.
    # ------------------------------------------------------------------
    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            f"FindNgram(scope={self.scope!r}).forward() has no plain-PyTorch "
            "reference: the kernel reads MPK runtime meta-tensors "
            "(tokens, step) to scan the request's history. Use the "
            "test-mode driver for end-to-end validation."
        )

    # ------------------------------------------------------------------
    # Grid heuristic.
    # ------------------------------------------------------------------
    def auto_grid_dim(self, *_: DTensor) -> GridDim:
        """Per-scope default grid.

        * ``partial``: ``(1, 1, 1)`` — caller typically passes a wider
          grid (with ``grid.y = num_tasks``) explicitly. The pk method
          does not enforce a grid; the kernel uses ``grid.y`` for task
          fan-out. We default to ``(1, 1, 1)`` and let callers override.
        * ``global``:  ``(1, 1, 1)`` — matches the ``prompt_lookup_spec_handler``
          caller in pk (``grid_dim=(1, 1, 1)``).
        """
        return (1, 1, 1)

    # ------------------------------------------------------------------
    # MPK compile path — dispatch on ``scope``.
    # ------------------------------------------------------------------
    def compile(
        self,
        *,
        input: Optional[DTensor] = None,
        partial_results: Optional[DTensor] = None,
        tokens: Optional[DTensor] = None,
        output: DTensor,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ) -> DTensor:
        """Register one find_ngram task on the active PK.

        Required-arg matrix:

        * ``scope="partial"``: ``input`` (the per-request tokens
          ``(batch, seq_len)``) and ``output``
          (``(batch, num_tasks)``) — both required.
        * ``scope="global"``: ``partial_results`` (output of the
          partial scan), ``tokens`` (request tokens, ``(batch, vocab)``
          per the pk method's contract), and ``output``
          (``(batch, spec_length + 1)`` int64 — the draft tokens for
          the next iteration).

        Returns:
            ``output`` (consumed for its side effect).
        """
        pk = current_pk()

        if grid_dim is None:
            grid_dim = self.auto_grid_dim()
        if block_dim is None:
            block_dim = self.default_block_dim()

        # Inlined task registration (formerly pk.find_ngram_*_layer).
        from ....core import CyTBGraph
        from ....kernel import TBGraph

        if self.scope == "partial":
            if input is None:
                raise ValueError(
                    "FindNgram(scope='partial').compile requires "
                    "input= (the per-request tokens DTensor)."
                )
            if partial_results is not None or tokens is not None:
                raise ValueError(
                    "FindNgram(scope='partial') consumes only "
                    "(input, output); pass scope='global' if you also "
                    "want a partial_results + tokens reduction."
                )
            assert input.num_dims == 2  # (batch_size, seq_len)
            assert output.num_dims == 2  # (batch_size, num_tasks)
            tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
            tb_graph.new_input(input, (-1, -1, -1), -1, True)
            tb_graph.new_input(output, (1, -1, -1), -1, True)
            pk.kn_graph.customized([input, output], tb_graph)
            pk.kn_graph.register_task(
                tb_graph, "find_ngram_partial", [self.ngram_size]
            )
        else:  # "global"
            if partial_results is None or tokens is None:
                raise ValueError(
                    "FindNgram(scope='global').compile requires "
                    "partial_results= and tokens= DTensors."
                )
            if input is not None:
                raise ValueError(
                    "FindNgram(scope='global') does not consume input=; "
                    "pass partial_results= and tokens= instead."
                )
            assert partial_results.num_dims == 2  # (batch_size, num_tasks)
            assert tokens.num_dims == 2  # (batch_size, vocab_size)
            assert output.num_dims == 2  # (batch_size, 1)
            tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
            tb_graph.new_input(partial_results, (-1, -1, -1), -1, True)
            tb_graph.new_input(tokens, (-1, -1, -1), -1, True)
            tb_graph.new_input(output, (-1, -1, -1), -1, True)
            pk.kn_graph.customized(
                [partial_results, tokens, output], tb_graph
            )
            pk.kn_graph.register_task(
                tb_graph, "find_ngram_global", [self.ngram_size, self.spec_length]
            )

        return output
