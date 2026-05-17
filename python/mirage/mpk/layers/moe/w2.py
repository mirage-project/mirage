"""MoE W2 (down projection) expert layer — bf16 and FP8 variants.

W2 is the post-activation down projection in the per-expert MLP::

    out[t, slot, :] = silu_mul_out[t, slot, :] @ W2[expert].T

The input is 3-D ``(batch_size, num_experts_per_tok, intermediate_size)``
(produced by :class:`MoESiluMul` from the W13 output) and the output is
``(batch_size, num_experts_per_tok, hidden_size)``. The downstream
``moe_mul_sum_add`` task combines this with the top-k weights and adds
the residual to produce a single ``(batch_size, hidden_size)`` slab.

Variants — same dispatch logic as :class:`MoEW13`:

* :meth:`PersistentKernel.moe_w2_linear_layer` (bf16,
  task ``moe_w2_linear_sm100`` / ``moe_w2_linear_sm90``).
* :meth:`PersistentKernel.moe_w2_fp8_layer` (fp8,
  task ``moe_w2_fp8_sm100``, Blackwell only).

FP8 layout (variant ``dtype="fp8"``)
-----------------------------------

* ``input_fp8``   : ``(batch_size, num_experts_per_tok, intermediate_size)``
  E4M3.
* ``input_scale`` : ``(batch_size, num_experts_per_tok,
  intermediate_size // 128)`` float32.
* ``weight_fp8``  : ``(num_experts, hidden_size, intermediate_size)``
  E4M3.
* ``weight_scale``: ``(num_experts, hidden_size,
  intermediate_size // 128)`` float32 — per-row, per-128-group scales.
  Note the legacy builder repeat_interleaves a (E, hidden, intermediate
  // 128) raw HF scale into shape ``(E, hidden, intermediate)`` for
  some kernels, but ``moe_w2_fp8_sm100`` consumes the **un-expanded**
  ``intermediate // 128`` layout — that's the convention we mirror.
* ``output``      : ``(batch_size, num_experts_per_tok, hidden_size)``
  bf16.

UE8M0 packing: ``scale_ue8m0`` is recorded for API symmetry but the
``moe_w2_fp8_sm100`` kernel always consumes plain fp32 scales (UE8M0
applies to the permute / group_gemm path, not the per-expert W2).

Routing-tensor contract
-----------------------

Same as W13: ``moe_routing_indices`` is ``(num_experts, batch_size)``
int32 expert-major, ``moe_mask`` is ``(num_experts + 1,)`` int32.

Parallelism — see :class:`MoEW13`. ``_moe_fp8_m_split`` preferred=14
for W2 (per the builder), grid.x = expert split (preferred_groups=10
for W2).

Forward (PyTorch reference) follows the same pattern as W13: dequant
inputs/weights, do expert-wise matmul. UE8M0 is not honored.
"""
from __future__ import annotations

from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn

from .._base import BlockDim, GridDim, MPKModule
from .w13 import _expert_for

from ....core import DTensor


__all__ = ["MoEW2"]


W2Dtype = Literal["bf16", "fp8"]


class MoEW2(MPKModule):
    """Per-expert down projection (W2).

    Args:
        num_experts: Total experts in ``self.weight`` (dim 0).
        num_experts_per_tok: Top-k width (input/output dim 1).
        hidden_size: Output trailing dim. Weight dim 1.
        intermediate_size: Reduction (K) axis. Input dim 2; weight dim 2.
        dtype: ``"bf16"`` or ``"fp8"``.
        scale_ue8m0: Informational (W2 kernel always reads plain fp32
            scales).
        prefix: HF state_dict prefix. ``self.weight`` -> ``{prefix}weight``;
            ``self.weight_scale`` (fp8 only) -> ``{prefix}weight_scale``.

    Attributes:
        weight: ``(num_experts, hidden_size, intermediate_size)``
            (bf16 or float8_e4m3fn).
        weight_scale: (fp8 only) ``(num_experts, hidden_size,
            intermediate_size // 128)`` float32.
    """

    def __init__(
        self,
        num_experts: int,
        num_experts_per_tok: int,
        hidden_size: int,
        intermediate_size: int,
        *,
        dtype: W2Dtype = "bf16",
        scale_ue8m0: bool = False,
        prefix: str = "",
    ) -> None:
        super().__init__(prefix=prefix)
        if dtype not in ("bf16", "fp8"):
            raise ValueError(
                f"MoEW2 dtype must be 'bf16' or 'fp8'; got {dtype!r}"
            )
        if dtype == "fp8" and intermediate_size % 128 != 0:
            raise ValueError(
                f"MoEW2(fp8) requires intermediate_size % 128 == 0; "
                f"got {intermediate_size}"
            )
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.dtype = dtype
        self.scale_ue8m0 = scale_ue8m0

        if dtype == "bf16":
            self.weight = nn.Parameter(
                torch.empty(
                    num_experts, hidden_size, intermediate_size,
                    dtype=torch.bfloat16,
                )
            )
            self.register_parameter("weight_scale", None)
        else:
            self.weight = nn.Parameter(
                torch.empty(
                    num_experts, hidden_size, intermediate_size,
                    dtype=torch.float8_e4m3fn,
                ),
                requires_grad=False,
            )
            self.weight_scale = nn.Parameter(
                torch.empty(
                    num_experts, hidden_size, intermediate_size // 128,
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )

    # ------------------------------------------------------------------
    # PyTorch reference
    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        routing_indices: torch.Tensor,
        *,
        x_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Per-expert down projection.

        For ``dtype="fp8"`` the caller must pass ``x_scale`` (per-128-
        group fp32 scales over the trailing axis). Returns
        ``(batch_size, num_experts_per_tok, hidden_size)`` bf16.
        """
        if x.dim() != 3 or x.size(2) != self.intermediate_size:
            raise ValueError(
                f"x must have shape (B, K, {self.intermediate_size}); "
                f"got {tuple(x.shape)}"
            )
        if x.size(1) != self.num_experts_per_tok:
            raise ValueError(
                f"x.size(1) must equal num_experts_per_tok="
                f"{self.num_experts_per_tok}; got {x.size(1)}"
            )
        batch_size = x.size(0)
        K = self.num_experts_per_tok
        expert_for = _expert_for(routing_indices, K)

        if self.dtype == "bf16":
            if x_scale is not None:
                raise ValueError("bf16 W2.forward does not accept x_scale")
            x_f32 = x.float()
            w_f32 = self.weight.float()
        else:
            if x_scale is None:
                raise ValueError(
                    "fp8 W2.forward requires x_scale of shape "
                    f"(B, K, {self.intermediate_size // 128})"
                )
            x_f32 = x.float()
            x_f32 = x_f32 * x_scale.float().repeat_interleave(128, dim=2)
            w_f32 = self.weight.float()
            w_scale_full = self.weight_scale.float().repeat_interleave(128, dim=2)
            w_f32 = w_f32 * w_scale_full

        out = torch.zeros(
            batch_size, K, self.hidden_size, dtype=torch.float32, device=x.device
        )
        for b in range(batch_size):
            for k in range(K):
                e = int(expert_for[b, k].item())
                if e < 0:
                    continue
                out[b, k] = x_f32[b, k] @ w_f32[e].t()
        return out.to(torch.bfloat16)

    # ------------------------------------------------------------------
    # Grid heuristic
    # ------------------------------------------------------------------
    def auto_grid_dim(self, *_: DTensor) -> GridDim:
        """Mirror ``_moe_expert_grid_x``/``_moe_fp8_m_split`` defaults for W2.

        Builder uses ``preferred_groups=10`` for grid.x and
        ``preferred=14`` for grid.y. The exact divisor walks are done
        here against ``num_experts`` and ``hidden_size`` respectively.
        Additional constraint: each per-CTA N-slab MUST be a multiple of
        ``MMA_M=128`` (otherwise the sm100 kernel writes a partial slab
        with wrong column stride). This is hit by small-shape unit tests
        (e.g. hidden=256 walks to gy=8 → slab=32 < 128). Production
        shapes (hidden=7168 with gy=14 → slab=512) hit gy=14 naturally
        and are unaffected.
        """
        gx = max(1, min(self.num_experts, 10))
        while self.num_experts % gx != 0 and gx > 1:
            gx -= 1
        gy = min(14, self.hidden_size)
        # gy must divide hidden_size AND leave a per-CTA slab >= 128.
        while gy > 1 and (
            self.hidden_size % gy != 0 or self.hidden_size // gy < 128
        ):
            gy -= 1
        return (gx, gy, 1)

    def default_block_dim(self) -> BlockDim:
        return (128, 1, 1)

    # ------------------------------------------------------------------
    # Compile
    # ------------------------------------------------------------------
    def compile(
        self,
        x: DTensor,
        routing_indices: DTensor,
        mask: DTensor,
        output: DTensor,
        *,
        x_scale: Optional[DTensor] = None,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ) -> DTensor:
        """Register the chosen ``moe_w2_*`` task on the current PK."""
        from ... import context as _ctx

        pk = _ctx.current_pk()

        if grid_dim is None:
            grid_dim = self.auto_grid_dim(x)
        if block_dim is None:
            block_dim = self.default_block_dim()

        # Inlined task registration (formerly pk.moe_w2_*_layer).
        from ....core import CyTBGraph
        from ....kernel import TBGraph

        if self.dtype == "bf16":
            if x_scale is not None:
                raise ValueError("MoEW2(bf16).compile: x_scale must be None")
            weight_dt = pk.attach_input(
                self.weight, name=f"{self.prefix}weight"
            )
            assert x.num_dims == 3  # (batch_size, num_expert_per_tok, intermediate_size)
            assert weight_dt.num_dims == 3  # (num_experts, hidden_size, intermediate_size)
            assert routing_indices.num_dims == 2  # (num_experts_per_tok, batch_size)
            assert mask.num_dims == 1  # (num_experts + 1)
            assert output.num_dims == 3  # (batch_size, num_expert_per_tok, hidden_size)
            tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
            tb_graph.new_input(x, (-1, -1, -1), 2, True)
            tb_graph.new_input(weight_dt, (-1, 1, -1), 2, True)
            tb_graph.new_input(routing_indices, (-1, -1, -1), -1, True)
            tb_graph.new_input(mask, (-1, -1, -1), -1, True)
            tb_graph.new_input(output, (-1, 2, -1), -1, True)
            pk.kn_graph.customized(
                [x, weight_dt, routing_indices, mask, output], tb_graph
            )
            if pk.target_cc == 100:
                pk.kn_graph.register_task(tb_graph, "moe_w2_linear_sm100")
            elif pk.target_cc == 90:
                pk.kn_graph.register_task(tb_graph, "moe_w2_linear_sm90")
            else:
                assert False
        else:
            if x_scale is None:
                raise ValueError(
                    "MoEW2(fp8).compile: x_scale is required "
                    f"(shape (B, K, {self.intermediate_size // 128}) float32)"
                )
            weight_dt = pk.attach_input(
                self.weight, name=f"{self.prefix}weight"
            )
            weight_scale_dt = pk.attach_input(
                self.weight_scale, name=f"{self.prefix}weight_scale"
            )
            assert x.num_dims == 3
            assert x_scale.num_dims == 3
            assert weight_dt.num_dims == 3
            assert weight_scale_dt.num_dims == 3
            assert routing_indices.num_dims == 2
            assert mask.num_dims == 1
            assert output.num_dims == 3
            tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
            # store_in_dmem=True for all inputs to work around a TBGraph
            # segfault with 3D tensors when store_in_dmem=False.
            tb_graph.new_input(x, (-1, -1, -1), -1, True)
            tb_graph.new_input(x_scale, (-1, -1, -1), -1, True)
            tb_graph.new_input(weight_dt, (-1, 1, -1), -1, True)
            tb_graph.new_input(weight_scale_dt, (-1, 1, -1), -1, True)
            tb_graph.new_input(routing_indices, (-1, -1, -1), -1, True)
            tb_graph.new_input(mask, (-1, -1, -1), -1, True)
            tb_graph.new_input(output, (-1, 2, -1), -1, True)
            pk.kn_graph.customized(
                [x, x_scale, weight_dt, weight_scale_dt,
                 routing_indices, mask, output], tb_graph,
            )
            assert pk.target_cc == 100, "FP8 group GEMM requires SM100 (Blackwell)"
            pk.kn_graph.register_task(tb_graph, "moe_w2_fp8_sm100")
        return output
