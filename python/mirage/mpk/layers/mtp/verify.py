"""MTP speculative-decode verify + accept-commit.

Wrappers around tasks in
``include/mirage/persistent_kernel/tasks/speculative_decoding/{target_verify_mtp,target_verify}.cuh``:
``MTPVerifyProbabilistic`` -> ``mtp_verify_probabilistic``,
``MTPVerifyStrict`` -> ``mtp_verify_strict``,
``MTPVerifyTargetGreedy`` -> ``target_verify_greedy``,
``MTPAcceptCommit`` -> ``mtp_accept_commit``.
"""
from __future__ import annotations

from typing import Optional, Tuple

from .._base import BlockDim, GridDim, MPKModule
from ...context import current_pk

from ....core import DTensor


__all__ = [
    "MTPVerifyProbabilistic",
    "MTPVerifyStrict",
    "MTPVerifyTargetGreedy",
    "MTPAcceptCommit",
]


class _MTPVerifyBase(MPKModule):
    """Shared ctor / forward skeleton for the MTP verify variants.

    All three kernels emit ``accepted_count + bonus`` plus accepted
    token IDs and consume runtime meta-tensors, so no plain-PyTorch
    reference is feasible.
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
                f"{type(self).__name__} num_draft_tokens must be positive; "
                f"got {num_draft_tokens}"
            )
        self.num_draft_tokens = num_draft_tokens

    def forward(self, *args, **kwargs):
        """No plain-PyTorch reference: kernel reads/writes MPK runtime
        meta-tensors. Use test-mode for end-to-end validation."""
        raise NotImplementedError(
            f"{type(self).__name__}.forward(): runtime meta-tensors required."
        )

    def auto_grid_dim(self, *_: DTensor) -> GridDim:
        """``(1, 1, 1)`` — verify kernels iterate ``num_draft_tokens`` and
        ``batch`` sequentially (single-thread acceptance loop)."""
        return (1, 1, 1)


class MTPVerifyProbabilistic(_MTPVerifyBase):
    """Probabilistic acceptance: ``P_target > u * P_draft``, ``u ~ U(0,1)``.

    Wraps ``mtp_verify_probabilistic``. Requires fp32 ``target_probs`` /
    ``draft_probs`` (length ``num_draft_tokens``) and a uint64 ``seed``
    DTensor; falls back to greedy match when ``p_draft == 0``.
    """

    def compile(
        self,
        draft_token_ids: DTensor,
        target_token_ids: DTensor,
        target_probs: DTensor,
        draft_probs: DTensor,
        seed: DTensor,
        accepted_count: DTensor,
        output_tokens: DTensor,
        *,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ) -> Tuple[DTensor, DTensor]:
        """Register one ``mtp_verify_probabilistic`` task.

        Tensor contract:
          draft_token_ids:  (B, num_draft_tokens) int64, dense.
          target_token_ids: (B, num_draft_tokens + 1) int64, dense.
          target_probs:     (B, num_draft_tokens) float32, dense.
          draft_probs:      (B, num_draft_tokens) float32, dense.
          seed:             (B,) uint64. RNG seed per batch.
          accepted_count:   (B, 1) int32, dense. Output; ``accepted+1``.
          output_tokens:    (B, num_draft_tokens + 1) int64. Output;
                            accepted prefix + bonus token at reject pos.
        Params: ``[num_draft_tokens]``.

        Notes: no runtime meta deps; ``p_draft == 0`` falls back to
        greedy match. Returns ``(accepted_count, output_tokens)``.
        """
        pk = current_pk()

        if grid_dim is None:
            grid_dim = self.auto_grid_dim(
                draft_token_ids, target_token_ids, accepted_count, output_tokens
            )
        if block_dim is None:
            block_dim = self.default_block_dim()

        from ....core import CyTBGraph
        from ....kernel import TBGraph

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
        return accepted_count, output_tokens


class MTPVerifyStrict(_MTPVerifyBase):
    """Strict acceptance: accept up to the first ``draft != target`` mismatch.

    Wraps task ``mtp_verify_strict``. No probabilities or RNG; compares
    token IDs only and emits ``accepted_count + 1`` (the +1 is the
    bonus token at the rejected position).
    """

    def compile(
        self,
        draft_token_ids: DTensor,
        target_token_ids: DTensor,
        accepted_count: DTensor,
        output_tokens: DTensor,
        *,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ) -> Tuple[DTensor, DTensor]:
        """Register one ``mtp_verify_strict`` task.

        Tensor contract:
          draft_token_ids:  (B, num_draft_tokens) int64, dense.
          target_token_ids: (B, num_draft_tokens + 1) int64, dense.
          accepted_count:   (B, 1) int32, dense. Output; ``accepted+1``.
          output_tokens:    (B, num_draft_tokens + 1) int64, dense.
                            Output; copies ``target_token_ids`` up to
                            ``accepted_count`` (incl. bonus).
        Params: ``[num_draft_tokens]``.

        Notes: no probabilities/RNG; no runtime meta deps. Returns
        ``(accepted_count, output_tokens)``.
        """
        pk = current_pk()

        if grid_dim is None:
            grid_dim = self.auto_grid_dim(
                draft_token_ids, target_token_ids, accepted_count, output_tokens
            )
        if block_dim is None:
            block_dim = self.default_block_dim()

        from ....core import CyTBGraph
        from ....kernel import TBGraph

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
        return accepted_count, output_tokens


class MTPVerifyTargetGreedy(_MTPVerifyBase):
    """Prompt-lookup greedy verify (no draft probs, no MTP).

    Wraps ``target_verify_greedy``. Writes accepted length to
    ``runtime_config.new_token_nums`` and tokens to ``runtime_config.tokens
    + step + 1`` directly; ``accepted_count`` is a placeholder DTensor.
    """

    def compile(
        self,
        spec_tokens: DTensor,
        target_tokens: DTensor,
        accepted_count: DTensor,
        *,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ) -> DTensor:
        """Register one ``target_verify_greedy`` task.

        Tensor contract:
          spec_tokens:    (B, num_spec_tokens + 1) int64, dense.
                          Slot 0 is the original token; 1.. are drafts.
          target_tokens:  (B, num_spec_tokens) int64, dense. Target
                          argmax for each draft position.
          accepted_count: (B, 1) int32, dense. Placeholder DTensor; the
                          kernel ignores it and writes acceptance to
                          ``runtime_config.new_token_nums`` directly.
        Params: none; registrar infers ``num_spec_tokens =
        spec_tokens.dim[1] - 1``.

        Notes: writes committed tokens to ``runtime_config.tokens +
        step + 1`` in place; ``runtime_config.step`` is read.
        """
        pk = current_pk()

        if grid_dim is None:
            grid_dim = self.auto_grid_dim(
                spec_tokens, target_tokens, accepted_count
            )
        if block_dim is None:
            block_dim = self.default_block_dim()

        from ....core import CyTBGraph
        from ....kernel import TBGraph

        assert spec_tokens.num_dims == 2
        assert target_tokens.num_dims == 2
        assert accepted_count.num_dims == 2
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(spec_tokens, (-1, -1, -1), -1, True)
        tb_graph.new_input(target_tokens, (-1, -1, -1), -1, True)
        tb_graph.new_input(accepted_count, (-1, -1, -1), -1, True)
        pk.kn_graph.customized(
            [spec_tokens, target_tokens, accepted_count], tb_graph,
        )
        pk.kn_graph.register_task(tb_graph, "target_verify_greedy")
        return accepted_count


class MTPAcceptCommit(MPKModule):
    """Commit verified prefix + update runtime decode meta-tensors.

    Wraps task ``mtp_accept_commit`` (target_verify_mtp.cuh). Writes
    ``new_position = current_position + count``, ``num_new_tokens =
    count``, and copies accepted + bonus tokens into ``final_output``.
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

    def forward(self, *args, **kwargs):
        """No plain-PyTorch reference: kernel mutates runtime meta-tensors
        (``current_position`` / ``new_position`` / ``num_new_tokens``).
        Use test-mode for end-to-end validation."""
        raise NotImplementedError(
            "MTPAcceptCommit.forward(): runtime meta-tensors required."
        )

    def auto_grid_dim(self, *_: DTensor) -> GridDim:
        """``(1, 1, 1)`` — one CTA copies ``count`` tokens (single-digit)."""
        return (1, 1, 1)

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
        """Register one ``mtp_accept_commit`` task.

        Tensor contract:
          accepted_count:   (B, 1) int32, dense. From verify kernel
                            (already includes the bonus token).
          output_tokens:    (B, num_draft_tokens + 1) int64, dense.
                            Verified tokens (accepted prefix + bonus).
          current_position: (B, 1) int32, dense. Current decode pos.
          new_position:     (B, 1) int32, dense. Output;
                            ``current_position + accepted_count``.
          final_output:     (B, num_draft_tokens + 1) int64, dense.
                            Output; copies first ``count`` tokens.
          num_new_tokens:   (B, 1) int32, dense. Output; ``count``.
        Params: ``[num_draft_tokens]``. No runtime meta deps.
        """
        pk = current_pk()

        if grid_dim is None:
            grid_dim = self.auto_grid_dim(
                accepted_count, output_tokens, current_position,
                new_position, final_output, num_new_tokens,
            )
        if block_dim is None:
            block_dim = self.default_block_dim()

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
