"""N-gram lookup for prompt-lookup speculative decoding.

Two wrappers around tasks ``find_ngram_partial`` /
``find_ngram_global`` in
``include/mirage/persistent_kernel/tasks/speculative_decoding/prompt_lookup.cuh``.
:class:`FindNgramPartial` fans the scan out across ``blockIdx.x``;
:class:`FindNgramGlobal` reduces partial results and emits draft
tokens reading ``runtime_config.tokens + step``.
"""
from __future__ import annotations

from typing import Optional

from .._base import BlockDim, GridDim, MPKModule
from ...context import current_pk

from ....core import DTensor


__all__ = ["FindNgramPartial", "FindNgramGlobal"]


class _FindNgramBase(MPKModule):
    """Shared ctor / forward skeleton for the two find-ngram variants.

    Both kernels read MPK runtime meta-tensors (``tokens`` / ``step``) to
    scan the request's history, so no plain-PyTorch reference is
    feasible.
    """

    def __init__(
        self,
        ngram_size: int = 3,
        *,
        prefix: str = "",
    ) -> None:
        super().__init__(prefix=prefix)
        if ngram_size <= 0:
            raise ValueError(
                f"{type(self).__name__} ngram_size must be positive; "
                f"got {ngram_size}"
            )
        self.ngram_size = ngram_size

    def forward(self, *args, **kwargs):
        """No plain-PyTorch reference: kernel reads runtime ``tokens`` /
        ``step``. Use test-mode for end-to-end validation."""
        raise NotImplementedError(
            f"{type(self).__name__}.forward(): runtime meta-tensors required."
        )


class FindNgramPartial(_FindNgramBase):
    """Per-task n-gram match scan over the request's own tokens.

    Wraps task ``find_ngram_partial``. The kernel uses ``blockIdx.x`` as
    ``task_id`` to fan out the scan over the input sequence and writes
    one ``int64`` per task (the first match index, or ``LLONG_MAX``).
    """

    def auto_grid_dim(self, *_: DTensor) -> GridDim:
        """``(1, 1, 1)`` default; callers typically override with
        ``grid.x = num_partial_tasks`` to fan out the scan."""
        return (1, 1, 1)

    def compile(
        self,
        input: DTensor,
        output: DTensor,
        *,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ) -> DTensor:
        """Register one ``find_ngram_partial`` task.

        Tensor contract:
          input:  (batch_size, seq_len) int64, dense. Token history.
                  Kernel scans ``[0, input_token_num - NGRAM)``.
          output: (batch_size, num_partial_tasks) int64, dense. Per-task
                  match position (``LLONG_MAX`` if no match); ``grid.x =
                  num_partial_tasks`` fans the scan across CTAs.
        Params: ``[ngram_size]``; ``num_partial_tasks`` inferred from
        ``output.dim[1]`` at registrar.

        Notes: kernel reads ``runtime_config.step[0] + 1`` as the
        scanned length; ``blockIdx.x`` is the task_id (fan-out).
        """
        pk = current_pk()

        if grid_dim is None:
            grid_dim = self.auto_grid_dim(input, output)
        if block_dim is None:
            block_dim = self.default_block_dim()

        from ....core import CyTBGraph
        from ....kernel import TBGraph

        assert input.num_dims == 2
        assert output.num_dims == 2
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (-1, -1, -1), -1, True)
        tb_graph.new_input(output, (1, -1, -1), -1, True)
        pk.kn_graph.customized([input, output], tb_graph)
        pk.kn_graph.register_task(
            tb_graph, "find_ngram_partial", [self.ngram_size]
        )
        return output


class FindNgramGlobal(_FindNgramBase):
    """Reduce partial results and emit ``spec_length + 1`` draft tokens.

    Wraps task ``find_ngram_global``. Reads ``tokens[step]`` plus
    ``tokens[match_idx + ngram_size + i]`` for the next
    ``spec_length`` positions; out-of-range slots emit ``-1``.
    """

    def __init__(
        self,
        ngram_size: int = 3,
        spec_length: int = 5,
        *,
        prefix: str = "",
    ) -> None:
        super().__init__(ngram_size=ngram_size, prefix=prefix)
        if spec_length <= 0:
            raise ValueError(
                f"FindNgramGlobal spec_length must be positive; "
                f"got {spec_length}"
            )
        self.spec_length = spec_length

    def auto_grid_dim(self, *_: DTensor) -> GridDim:
        """``(1, 1, 1)`` — single-CTA reduce + emit."""
        return (1, 1, 1)

    def compile(
        self,
        partial_results: DTensor,
        tokens: DTensor,
        output: DTensor,
        *,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ) -> DTensor:
        """Register one ``find_ngram_global`` task.

        Tensor contract:
          partial_results: (batch_size, num_partial_tasks) int64, dense.
                           Output of :class:`FindNgramPartial`.
          tokens:          (batch_size, vocab_or_seq_len) int64, dense.
                           Token history; kernel indexes by step+offset.
          output:          (batch_size, spec_length + 1) int64, dense.
                           Slot 0 = ``tokens[step]``; slots 1.. are the
                           next ``spec_length`` n-gram successors
                           (``-1`` when OOB or no match).
        Params: ``[ngram_size, spec_length]``; ``num_partial_tasks``
        inferred from ``partial_results.dim[1]``.

        Notes: kernel reads ``runtime_config.step[0]``.
        """
        pk = current_pk()

        if grid_dim is None:
            grid_dim = self.auto_grid_dim(partial_results, tokens, output)
        if block_dim is None:
            block_dim = self.default_block_dim()

        from ....core import CyTBGraph
        from ....kernel import TBGraph

        assert partial_results.num_dims == 2
        assert tokens.num_dims == 2
        assert output.num_dims == 2
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
