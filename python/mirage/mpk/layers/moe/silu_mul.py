"""MoE SiLU-Mul activation — accepts both 2-D and 3-D layouts.

Wraps :meth:`PersistentKernel.moe_silu_mul_layer` (task ``moe_silu_mul``).
The underlying kernel accepts two layouts:

* **3-D** (legacy / OLD MoE path):
  ``input  (batch_size, num_experts_per_tok, 2 * intermediate_size)``,
  ``output (batch_size, num_experts_per_tok, intermediate_size)``.
  Used by qwen3 MoE (post-W13 → pre-W2) and DeepSeek V3's OLD per-expert
  W13/W2 pipeline. The first row half is the gate, the second half is
  the up.

* **2-D** (NEW MoE path, PR-674 group GEMM):
  ``input  (m_total, 2 * intermediate_size)``,
  ``output (m_total, intermediate_size)`` (where ``m_total = E_local *
  bm_padding`` is the post-permute row count). Internally the codegen
  treats this as ``(m_total, 1, 2*intermediate_size)`` —
  ``num_experts_per_tok = 1`` — so the same kernel can serve both
  layouts.

Halved gate/up layout: like the dense :class:`SiluMul`, per-task the
input is ``[gate | up]`` over the trailing axis. Per-task slab width is
``2 * intermediate_size / grid.x`` (3-D) or analogous (2-D).

Parallelism
-----------

* 3-D layout: ``grid_dim = (batch_size, num_experts_per_tok, 1)``,
  i.e. one CTA per (token, slot). Block 128.
* 2-D layout: ``grid_dim = (m_total_or_capped, 1, 1)``. The DeepSeek V3
  NEW path uses ``min(pk.num_workers, m_total)`` for grid.x.

Forward (PyTorch reference) implements ``SiLU(gate) * up`` in fp32 (to
match the kernel) and returns the result in the input dtype. Accepts
the same 2-D or 3-D layout the kernel does.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from .._base import BlockDim, GridDim, MPKModule

from ....core import DTensor


__all__ = ["MoESiluMul"]


class MoESiluMul(MPKModule):
    """SiLU-Mul activation for the MoE pipeline (2-D or 3-D layout).

    Args:
        intermediate_size: Per-side gate/up width. Trailing input dim
            is ``2 * intermediate_size``; trailing output dim is
            ``intermediate_size``.
        prefix: HF state_dict prefix (no weights live on this module —
            used only for output DTensor names).
    """

    def __init__(
        self,
        intermediate_size: int,
        *,
        prefix: str = "",
    ) -> None:
        super().__init__(prefix=prefix)
        self.intermediate_size = intermediate_size

    # ------------------------------------------------------------------
    # PyTorch reference
    # ------------------------------------------------------------------
    def forward(self, gateup: torch.Tensor) -> torch.Tensor:
        """``SiLU(gate) * up`` over the trailing axis.

        Accepts 2-D ``(M, 2*intermediate)`` or 3-D
        ``(B, K, 2*intermediate)``. Computes in fp32, returns in the
        input dtype.
        """
        if gateup.dim() not in (2, 3):
            raise ValueError(
                "MoESiluMul.forward expects 2-D or 3-D input; "
                f"got {gateup.dim()}-D"
            )
        if gateup.size(-1) != 2 * self.intermediate_size:
            raise ValueError(
                "MoESiluMul: input trailing dim must equal "
                f"2*intermediate_size={2*self.intermediate_size}; "
                f"got {gateup.size(-1)}"
            )
        gate = gateup[..., : self.intermediate_size]
        up = gateup[..., self.intermediate_size :]
        return (F.silu(gate.float()) * up.float()).to(gateup.dtype)

    # ------------------------------------------------------------------
    # Grid heuristic
    # ------------------------------------------------------------------
    def auto_grid_dim(self, input_dt: DTensor) -> GridDim:
        """Default grid: ``(B, K, 1)`` for 3-D inputs, ``(M_capped, 1, 1)`` for 2-D.

        The legacy demos use ``(batch_size, num_experts_per_tok, 1)`` for
        the 3-D layout (qwen3 + DeepSeek V3 OLD MoE), and
        ``(min(pk.num_workers, m_total), 1, 1)`` for the 2-D NEW-MoE
        layout. We mirror both here.
        """
        from ... import context as _ctx

        pk = _ctx.current_pk()
        if input_dt.num_dims == 3:
            return (input_dt.dim(0), input_dt.dim(1), 1)
        m_total = input_dt.dim(0)
        gx = max(1, min(int(getattr(pk, "num_workers", 1)), m_total))
        return (gx, 1, 1)

    def default_block_dim(self) -> BlockDim:
        return (128, 1, 1)

    # ------------------------------------------------------------------
    # Compile
    # ------------------------------------------------------------------
    def compile(
        self,
        gateup: DTensor,
        output: DTensor,
        *,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ) -> DTensor:
        """Register a ``moe_silu_mul`` task.

        Args:
            gateup: 2-D ``(M, 2*intermediate_size)`` or 3-D
                ``(B, K, 2*intermediate_size)`` input.
            output: Caller-allocated output of matching rank and shape
                with trailing dim ``intermediate_size``.
            grid_dim, block_dim: see :class:`MPKModule`.
        """
        from ... import context as _ctx

        pk = _ctx.current_pk()

        if gateup.num_dims not in (2, 3):
            raise ValueError(
                f"MoESiluMul expects 2-D or 3-D gateup; "
                f"got num_dims={gateup.num_dims}"
            )
        if gateup.num_dims != output.num_dims:
            raise ValueError(
                "MoESiluMul: gateup and output must have matching rank "
                f"(gateup.num_dims={gateup.num_dims}, output.num_dims={output.num_dims})"
            )
        if gateup.dim(gateup.num_dims - 1) != 2 * self.intermediate_size:
            raise ValueError(
                "MoESiluMul: gateup trailing dim must equal "
                f"2*intermediate_size={2*self.intermediate_size}; "
                f"got {gateup.dim(gateup.num_dims - 1)}"
            )
        if output.dim(output.num_dims - 1) != self.intermediate_size:
            raise ValueError(
                "MoESiluMul: output trailing dim must equal "
                f"intermediate_size={self.intermediate_size}; "
                f"got {output.dim(output.num_dims - 1)}"
            )

        if grid_dim is None:
            grid_dim = self.auto_grid_dim(gateup)
        if block_dim is None:
            block_dim = self.default_block_dim()

        # Inlined task registration (formerly pk.moe_silu_mul_layer). The
        # underlying kernel accepts both 2-D (NEW MoE path) and 3-D (OLD
        # MoE path) layouts; the input_map differs per rank.
        from ....core import CyTBGraph
        from ....kernel import TBGraph

        assert gateup.num_dims in (2, 3)
        assert output.num_dims == gateup.num_dims
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        if gateup.num_dims == 3:
            tb_graph.new_input(gateup, (0, 1, -1), -1, True)
            tb_graph.new_input(output, (0, 1, -1), -1, True)
        else:
            tb_graph.new_input(gateup, (0, -1, -1), -1, True)
            tb_graph.new_input(output, (0, -1, -1), -1, True)
        pk.kn_graph.customized([gateup, output], tb_graph)
        pk.kn_graph.register_task(tb_graph, "moe_silu_mul")
        return output
