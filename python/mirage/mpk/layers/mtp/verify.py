"""MTP speculative-decode verify + accept-commit primitives.

Two classes:

* :class:`MTPVerify` — one front for three verify kernels selected by a
  ``mode`` kwarg::

      mode="probabilistic" -> mtp_verify_probabilistic_layer
                              -> task ``mtp_verify_probabilistic``
      mode="strict"        -> mtp_verify_strict_layer
                              -> task ``mtp_verify_strict``
      mode="target_greedy" -> target_verify_greedy_layer
                              -> task ``target_verify_greedy``

  ``probabilistic`` accepts a draft token when
  ``P_target / P_draft > u`` (a per-step uniform sample) — the standard
  speculative-decode acceptance rule. ``strict`` accepts only when the
  draft and target tokens match exactly. ``target_greedy`` is the
  non-MTP prompt-lookup verify path (no draft probabilities involved).

* :class:`MTPAcceptCommit` -> :meth:`PersistentKernel.mtp_accept_commit_layer`
  -> task ``mtp_accept_commit``. Runs AFTER the verify kernel and
  commits the accepted prefix into the request's final output, updates
  the request's decode position, and writes ``num_new_tokens`` for the
  next iteration.

Runtime-metadata dependence
---------------------------

All four kernels read at least one MPK runtime meta-tensor
(``current_position`` / ``new_position`` / ``num_new_tokens`` /
per-request ``step``). The reference :meth:`forward` paths therefore
raise ``NotImplementedError`` — they're not implementable without
replicating the runtime.
"""
from __future__ import annotations

from typing import Literal, Optional, Tuple

from .._base import BlockDim, GridDim, MPKModule
from ...context import current_pk

from ....core import DTensor


__all__ = ["MTPVerify", "MTPAcceptCommit"]


VerifyMode = Literal["probabilistic", "strict", "target_greedy"]


class MTPVerify(MPKModule):
    """MTP speculative-decode verifier — three modes share one front.

    Args:
        num_draft_tokens: Number of draft tokens MTP emits per
            iteration. Baked into the task params for the two MTP modes
            (the ``target_greedy`` task does not need it but accepting it
            uniformly keeps the API symmetric).
        mode: One of ``"probabilistic"``, ``"strict"``, ``"target_greedy"``.
        prefix: vLLM/HF state_dict prefix.

    Choice of mode
    --------------
    The mode is fixed at construction time so the module is a faithful
    1-to-1 wrapper for one underlying pk method. Callers that want both
    a probabilistic and a strict verifier in the same compile scope
    instantiate two :class:`MTPVerify` objects with different ``mode``.
    """

    def __init__(
        self,
        num_draft_tokens: int,
        *,
        mode: VerifyMode = "probabilistic",
        prefix: str = "",
    ) -> None:
        super().__init__(prefix=prefix)
        if mode not in ("probabilistic", "strict", "target_greedy"):
            raise ValueError(
                f"MTPVerify mode must be one of "
                f"'probabilistic'/'strict'/'target_greedy'; got {mode!r}"
            )
        if num_draft_tokens <= 0:
            raise ValueError(
                f"MTPVerify num_draft_tokens must be positive; "
                f"got {num_draft_tokens}"
            )
        self.num_draft_tokens = num_draft_tokens
        self.mode = mode

    # ------------------------------------------------------------------
    # PyTorch reference — runtime-driven, no plain reference feasible.
    # ------------------------------------------------------------------
    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            f"MTPVerify(mode={self.mode!r}).forward() has no plain-PyTorch "
            "reference: the kernel reads MPK runtime meta-tensors "
            "(accepted_count, output_tokens, draft probs/tokens already "
            "scattered into per-position buffers via prob_scatter / "
            "mtp_float_scatter / mtp_token_scatter). Use the test-mode "
            "driver to validate end-to-end."
        )

    # ------------------------------------------------------------------
    # Grid heuristic.
    # ------------------------------------------------------------------
    def auto_grid_dim(self, *_: DTensor) -> GridDim:
        """Default grid ``(1, 1, 1)`` for all three modes.

        Each verify kernel iterates over ``num_draft_tokens`` and over
        ``batch`` internally; a single CTA suffices, and this matches
        every DeepSeek V3 builder call site.
        """
        return (1, 1, 1)

    # ------------------------------------------------------------------
    # MPK compile path — dispatch on ``mode``.
    # ------------------------------------------------------------------
    def compile(
        self,
        draft_token_ids: DTensor,
        target_token_ids: DTensor,
        accepted_count: DTensor,
        output_tokens: DTensor,
        *,
        target_probs: Optional[DTensor] = None,
        draft_probs: Optional[DTensor] = None,
        seed: Optional[DTensor] = None,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ) -> Tuple[DTensor, DTensor]:
        """Register one verify task on the active PK.

        Required-arg matrix:

        * ``mode="probabilistic"``: ``target_probs``, ``draft_probs``,
          ``seed`` are required (kernel computes
          ``P_target / P_draft > u``).
        * ``mode="strict"``: only the four positional args are needed;
          ``target_probs``/``draft_probs``/``seed`` MUST be ``None``.
        * ``mode="target_greedy"``: ``draft_token_ids`` and
          ``target_token_ids`` are renamed in the pk layer to
          ``spec_tokens`` and ``target_tokens`` but the contract is the
          same (`(batch, vocab_size)` int64 each); ``accepted_count``
          plays the role of the pk method's ``output`` (single-element
          ``(1, 1)`` int64). ``output_tokens`` is unused by the
          ``target_greedy`` task — pass any DTensor (callers typically
          re-use ``accepted_count`` or a dummy DTensor).

        Returns:
            ``(accepted_count, output_tokens)`` — the same DTensors the
            caller passed in (consumed for their side effects). Returned
            for convenience so callers can thread the dependency in
            downstream :class:`MTPAcceptCommit` calls.
        """
        pk = current_pk()

        if grid_dim is None:
            grid_dim = self.auto_grid_dim(
                draft_token_ids, target_token_ids, accepted_count, output_tokens
            )
        if block_dim is None:
            block_dim = self.default_block_dim()

        # Inlined task registration (formerly pk.mtp_verify_*_layer /
        # pk.target_verify_greedy_layer).
        from ....core import CyTBGraph
        from ....kernel import TBGraph

        if self.mode == "probabilistic":
            if target_probs is None or draft_probs is None or seed is None:
                raise ValueError(
                    "MTPVerify(mode='probabilistic').compile requires "
                    "target_probs, draft_probs, and seed DTensors."
                )
            params = [self.num_draft_tokens]
            tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
            tb_graph.new_input(draft_token_ids, (-1, -1, -1), -1, True)
            tb_graph.new_input(target_token_ids, (-1, -1, -1), -1, True)
            tb_graph.new_input(target_probs, (-1, -1, -1), -1, True)
            tb_graph.new_input(draft_probs, (-1, -1, -1), -1, True)
            tb_graph.new_input(seed, (-1, -1, -1), -1, True)
            tb_graph.new_input(accepted_count, (-1, -1, -1), -1, True)
            tb_graph.new_input(output_tokens, (-1, -1, -1), -1, True)
            pk.kn_graph.customized(
                [draft_token_ids, target_token_ids, target_probs, draft_probs,
                 seed, accepted_count, output_tokens], tb_graph,
            )
            pk.kn_graph.register_task(
                tb_graph, "mtp_verify_probabilistic", params
            )
        elif self.mode == "strict":
            if target_probs is not None or draft_probs is not None or seed is not None:
                raise ValueError(
                    "MTPVerify(mode='strict') does not use "
                    "target_probs/draft_probs/seed — pass None for all "
                    "three (strict verify compares token IDs only)."
                )
            params = [self.num_draft_tokens]
            tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
            tb_graph.new_input(draft_token_ids, (-1, -1, -1), -1, True)
            tb_graph.new_input(target_token_ids, (-1, -1, -1), -1, True)
            tb_graph.new_input(accepted_count, (-1, -1, -1), -1, True)
            tb_graph.new_input(output_tokens, (-1, -1, -1), -1, True)
            pk.kn_graph.customized(
                [draft_token_ids, target_token_ids, accepted_count, output_tokens],
                tb_graph,
            )
            pk.kn_graph.register_task(tb_graph, "mtp_verify_strict", params)
        else:  # "target_greedy"
            if target_probs is not None or draft_probs is not None or seed is not None:
                raise ValueError(
                    "MTPVerify(mode='target_greedy') does not use "
                    "target_probs/draft_probs/seed — pass None for all "
                    "three."
                )
            # target_verify_greedy_layer takes (spec_tokens, target_tokens)
            # tuple input and a single output DTensor. We re-use the
            # ``accepted_count`` arg as the output ((1, 1) int64) per the
            # builder caller pattern.
            spec_tokens = draft_token_ids
            target_tokens = target_token_ids
            assert spec_tokens.num_dims == 2  # (batch_size, vocab_size)
            assert target_tokens.num_dims == 2  # (batch_size, vocab_size)
            assert accepted_count.num_dims == 2  # (batch_size, 1)
            tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
            tb_graph.new_input(spec_tokens, (-1, -1, -1), -1, True)
            tb_graph.new_input(target_tokens, (-1, -1, -1), -1, True)
            tb_graph.new_input(accepted_count, (-1, -1, -1), -1, True)
            pk.kn_graph.customized(
                [spec_tokens, target_tokens, accepted_count], tb_graph,
            )
            pk.kn_graph.register_task(tb_graph, "target_verify_greedy")

        return accepted_count, output_tokens


class MTPAcceptCommit(MPKModule):
    """Commit accepted prefix into final output and update meta-tensors.

    Wraps :meth:`PersistentKernel.mtp_accept_commit_layer` (task
    ``mtp_accept_commit``).

    Args:
        num_draft_tokens: Number of draft tokens MTP emits per
            iteration. Baked into the task params.
        prefix:           vLLM/HF state_dict prefix.

    Forward
    -------
    :meth:`forward` raises ``NotImplementedError`` — the kernel mutates
    runtime meta-tensors (``current_position``, ``new_position``,
    ``num_new_tokens``) which a plain PyTorch reference cannot model
    faithfully without recreating the runtime.
    """

    def __init__(
        self,
        num_draft_tokens: int,
        *,
        prefix: str = "",
    ) -> None:
        super().__init__(prefix=prefix)
        if num_draft_tokens <= 0:
            raise ValueError(
                f"MTPAcceptCommit num_draft_tokens must be positive; "
                f"got {num_draft_tokens}"
            )
        self.num_draft_tokens = num_draft_tokens

    # ------------------------------------------------------------------
    # PyTorch reference — runtime-driven.
    # ------------------------------------------------------------------
    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "MTPAcceptCommit.forward() has no plain-PyTorch reference: "
            "the kernel mutates MPK runtime meta-tensors "
            "(current_position, new_position, num_new_tokens). Use "
            "test-mode for end-to-end validation."
        )

    # ------------------------------------------------------------------
    # Grid heuristic.
    # ------------------------------------------------------------------
    def auto_grid_dim(self, *_: DTensor) -> GridDim:
        """Default grid ``(1, 1, 1)``."""
        return (1, 1, 1)

    # ------------------------------------------------------------------
    # MPK compile path.
    # ------------------------------------------------------------------
    def compile(
        self,
        accepted_count: DTensor,
        output_tokens: DTensor,
        current_position: DTensor,
        new_position: DTensor,
        final_output: DTensor,
        num_new_tokens: DTensor,
        *,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ) -> DTensor:
        """Register one ``mtp_accept_commit`` task on the active PK.

        Args:
            accepted_count:   ``(batch, 1)`` int64 from the verifier.
            output_tokens:    ``(batch, num_draft_tokens+1)`` int64
                              from the verifier.
            current_position: Runtime meta-tensor: pre-commit per-request
                              position.
            new_position:     Runtime meta-tensor: post-commit position
                              (kernel writes here).
            final_output:     Per-request final output token buffer the
                              kernel appends into.
            num_new_tokens:   Runtime meta-tensor: per-iteration new-token
                              count (kernel writes here).
            grid_dim:         Override; ``None`` -> :meth:`auto_grid_dim`.
            block_dim:        Override; ``None`` -> :meth:`default_block_dim`.

        Returns:
            ``final_output`` (consumed for its side effect).
        """
        pk = current_pk()

        if grid_dim is None:
            grid_dim = self.auto_grid_dim(
                accepted_count, output_tokens, current_position,
                new_position, final_output, num_new_tokens,
            )
        if block_dim is None:
            block_dim = self.default_block_dim()

        # Inlined task registration (formerly pk.mtp_accept_commit_layer).
        from ....core import CyTBGraph
        from ....kernel import TBGraph

        params = [self.num_draft_tokens]
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(accepted_count, (-1, -1, -1), -1, True)
        tb_graph.new_input(output_tokens, (-1, -1, -1), -1, True)
        tb_graph.new_input(current_position, (-1, -1, -1), -1, True)
        tb_graph.new_input(new_position, (-1, -1, -1), -1, True)
        tb_graph.new_input(final_output, (-1, -1, -1), -1, True)
        tb_graph.new_input(num_new_tokens, (-1, -1, -1), -1, True)
        pk.kn_graph.customized(
            [accepted_count, output_tokens, current_position,
             new_position, final_output, num_new_tokens], tb_graph,
        )
        pk.kn_graph.register_task(tb_graph, "mtp_accept_commit", params)
        return final_output
