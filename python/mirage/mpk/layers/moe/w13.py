"""MoE W13 (fused gate+up) expert projection — bf16 and FP8 variants.

This catalog module wraps two pk methods:

* :meth:`PersistentKernel.moe_w13_linear_layer`
  (task ``moe_w13_linear_sm100`` on Blackwell, ``moe_w13_linear_sm90``
  on Hopper). bf16 weights, bf16 activations.
* :meth:`PersistentKernel.moe_w13_fp8_layer`
  (task ``moe_w13_fp8_sm100``, Blackwell only). FP8 E4M3 weights AND
  activations with per-128-group fp32 scales.

W13 computes the fused gate+up projection per active expert::

    out[t, slot, :] = x[t, :] @ W[expert_for(t, slot)].T

Where ``x`` is the layer input ``(batch_size, hidden_size)`` and
``W[e]`` is the expert weight ``(2 * intermediate_size, hidden_size)``.
``slot`` indexes the top-k slot (0..num_experts_per_tok-1). The
routing tensors (``moe_routing_indices`` + ``moe_mask``) tell each task
which (token, slot) pairs land in which expert.

The output is laid out as ``(batch_size, num_experts_per_tok,
2 * intermediate_size)`` so the downstream silu_mul kernel can read
``[gate | up]`` per row.

FP8 layout (variant ``dtype="fp8"``)
-----------------------------------

* ``input_fp8``   : ``(batch_size, hidden_size)`` E4M3.
* ``input_scale`` : ``(batch_size, hidden_size // 128)`` float32.
  Each scale covers 128 contiguous input elements along the
  ``hidden_size`` axis. UE8M0-packed input scales are NOT used by the
  ``moe_w13_fp8_sm100`` kernel — it consumes plain fp32 scales. The
  ``scale_ue8m0`` kwarg is therefore informational here (the FP8 W13
  kernel ignores UE8M0); we keep it for symmetry with W2 and the
  permute path that DO honor it.
* ``weight_fp8``  : ``(num_experts, 2 * intermediate_size, hidden_size)``
  E4M3.
* ``weight_scale``: ``(num_experts, 2 * intermediate_size,
  hidden_size // 128)`` float32 — per-row, per-128-group scales.
* ``output``      : ``(batch_size, num_experts_per_tok,
  2 * intermediate_size)`` bf16.

The scale factor (``128``) is fixed in the kernel.

Routing-tensor contract (both variants)
---------------------------------------

* ``moe_routing_indices`` : ``(num_experts, batch_size)`` int32 —
  expert-major. Slot index (1-indexed) of token ``t`` for expert
  ``e``, or 0 if not routed.
* ``moe_mask``            : ``(num_experts + 1,)`` int32 — prefix
  count of routed (token, slot) pairs per expert. The kernel uses
  this to skip inactive experts.

Parallelism
-----------

The kernel partitions along two axes:

* ``grid.x`` → expert groups (each CTA processes a contiguous range
  of experts via ``task_metadata.expert_offset``). MMA-M=128 is
  hard-wired in the kernel; ``grid.x`` should divide
  ``num_experts``.
* ``grid.y`` → M-split (the ``2 * intermediate_size`` axis). Each
  CTA owns a per-expert row slab of width ``2*intermediate_size /
  grid.y``. ``grid.y`` should divide ``2*intermediate_size`` and is
  the same M-split flag the legacy demo exposes via
  ``MPK_MOE_W13_M_SPLIT``.

Block dim is 128 on Blackwell (per builder convention). On Hopper the
bf16 path uses the same 128-thread kernel.

Forward (PyTorch reference)
---------------------------

For ``dtype="bf16"``, ``forward()`` does a plain expert-wise matmul:
for each token ``t`` and slot ``k``, look up the expert from
``routing_indices``, then do ``x[t] @ W[e].T``.

For ``dtype="fp8"``, ``forward()`` first dequantizes
``input_fp8 * input_scale`` and ``weight_fp8 * weight_scale`` to fp32
(per the per-128-group layout) and runs the same expert-wise matmul.
This is the test-mode oracle — it's faithful to the kernel's
fp32-accumulator semantics and matches what DeepSeek V3's eager
reference produces. UE8M0 packing is NOT honored here (W13 doesn't
use UE8M0).
"""
from __future__ import annotations

from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn

from .._base import BlockDim, GridDim, MPKModule

from ....core import DTensor


__all__ = ["MoEW13"]


W13Dtype = Literal["bf16", "fp8"]


def _expert_for(routing_indices: torch.Tensor, num_experts_per_tok: int) -> torch.Tensor:
    """Return (batch, k) int32 of the expert id assigned to each (token, slot).

    ``routing_indices`` is (E, B) with 1-indexed slot id (0 if not
    routed). We invert it into a (B, K) expert-id grid.
    """
    E, B = routing_indices.shape
    out = torch.full((B, num_experts_per_tok), -1, dtype=torch.long,
                     device=routing_indices.device)
    for e in range(E):
        slots = routing_indices[e]  # (B,) — 1-indexed slot, 0 = unrouted
        nz = slots.nonzero(as_tuple=False).squeeze(-1)
        if nz.numel() == 0:
            continue
        out[nz, slots[nz].long() - 1] = e
    return out


class MoEW13(MPKModule):
    """Fused gate+up projection across the routed experts.

    Args:
        num_experts: Total experts in the weight tensor (its dim 0).
        num_experts_per_tok: Top-k width (output dim 1).
        hidden_size: Reduction (K) axis. Input dim 1; weight dim 2.
        intermediate_size: Per-side gate/up width. Weight dim 1 is
            ``2 * intermediate_size``; output trailing dim is the same.
        dtype: ``"bf16"`` (qwen3) or ``"fp8"`` (DeepSeek V3).
        scale_ue8m0: (fp8 only) Informational — the
            ``moe_w13_fp8_sm100`` kernel always consumes plain fp32
            scales, so this flag is recorded but not forwarded. We
            keep it for API symmetry with W2 / permute / quantize.
        prefix: HF state_dict prefix. ``self.weight`` is loaded from
            ``state_dict[f"{prefix}weight"]``; for fp8 the scales are
            loaded from ``state_dict[f"{prefix}weight_scale"]``.

    Attributes:
        weight: ``nn.Parameter`` of shape ``(num_experts,
            2 * intermediate_size, hidden_size)``. dtype bf16 for the
            bf16 variant; ``torch.float8_e4m3fn`` for fp8.
        weight_scale: (fp8 only) ``nn.Parameter`` of shape
            ``(num_experts, 2 * intermediate_size, hidden_size // 128)``,
            dtype float32.
    """

    def __init__(
        self,
        num_experts: int,
        num_experts_per_tok: int,
        hidden_size: int,
        intermediate_size: int,
        *,
        dtype: W13Dtype = "bf16",
        scale_ue8m0: bool = False,
        prefix: str = "",
    ) -> None:
        super().__init__(prefix=prefix)
        if dtype not in ("bf16", "fp8"):
            raise ValueError(
                f"MoEW13 dtype must be 'bf16' or 'fp8'; got {dtype!r}"
            )
        if hidden_size % 128 != 0 and dtype == "fp8":
            raise ValueError(
                f"MoEW13(fp8) requires hidden_size % 128 == 0; got {hidden_size}"
            )
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.dtype = dtype
        self.scale_ue8m0 = scale_ue8m0

        w_out = 2 * intermediate_size
        if dtype == "bf16":
            self.weight = nn.Parameter(
                torch.empty(
                    num_experts, w_out, hidden_size, dtype=torch.bfloat16
                )
            )
            self.register_parameter("weight_scale", None)
        else:
            self.weight = nn.Parameter(
                torch.empty(
                    num_experts, w_out, hidden_size,
                    dtype=torch.float8_e4m3fn,
                ),
                requires_grad=False,
            )
            self.weight_scale = nn.Parameter(
                torch.empty(
                    num_experts, w_out, hidden_size // 128,
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
        """Expert-wise fused gate+up projection.

        For ``dtype="fp8"`` the caller passes ``x`` and ``x_scale``
        separately (E4M3 + per-128-group fp32 scales). For ``"bf16"``
        the caller just passes ``x`` (and ``x_scale`` must be None).

        Returns ``(batch_size, num_experts_per_tok, 2*intermediate_size)``
        in bf16 (matches the kernel output).
        """
        if x.dim() != 2 or x.size(1) != self.hidden_size:
            raise ValueError(
                f"x must have shape (B, {self.hidden_size}); got {tuple(x.shape)}"
            )
        batch_size = x.size(0)
        expert_for = _expert_for(routing_indices, self.num_experts_per_tok)

        if self.dtype == "bf16":
            if x_scale is not None:
                raise ValueError("bf16 W13.forward does not accept x_scale")
            x_f32 = x.float()
            w_f32 = self.weight.float()
        else:
            if x_scale is None:
                raise ValueError(
                    "fp8 W13.forward requires x_scale "
                    f"of shape (B, {self.hidden_size // 128})"
                )
            # Dequant input: x_f32[b, c] = x_fp8[b, c] * x_scale[b, c//128].
            x_f32 = x.float()
            x_f32 = x_f32 * x_scale.float().repeat_interleave(128, dim=1)
            # Dequant weight: w_f32[e, r, c] = w_fp8[e, r, c]
            #                                   * w_scale[e, r, c//128].
            w_f32 = self.weight.float()
            w_scale_full = self.weight_scale.float().repeat_interleave(128, dim=2)
            w_f32 = w_f32 * w_scale_full

        K = self.num_experts_per_tok
        out_dim = 2 * self.intermediate_size
        out = torch.zeros(
            batch_size, K, out_dim, dtype=torch.float32, device=x.device
        )
        for b in range(batch_size):
            for k in range(K):
                e = int(expert_for[b, k].item())
                if e < 0:
                    continue
                out[b, k] = x_f32[b] @ w_f32[e].t()
        return out.to(torch.bfloat16)

    # ------------------------------------------------------------------
    # Grid heuristic
    # ------------------------------------------------------------------
    def auto_grid_dim(self, *_: DTensor) -> GridDim:
        """Pick a default ``(grid.x, grid.y, 1)`` mirroring the legacy demos.

        * ``grid.x``: an expert-axis split. Heuristic = ``min(num_experts, 8)``
          rounded down to a divisor of ``num_experts``. Matches the
          ``_moe_expert_grid_x`` heuristic in deepseek_v3/builder.py
          (preferred_groups=8 for W13).
        * ``grid.y``: an M-split on the ``2*intermediate_size`` axis.
          Heuristic mirrors ``_moe_fp8_m_split`` with preferred=16 (the
          builder's default for W13): the largest divisor of
          ``2*intermediate_size`` that is <= 16.
        """
        # grid.x — divide num_experts into <=8 groups.
        gx = max(1, min(self.num_experts, 8))
        while self.num_experts % gx != 0 and gx > 1:
            gx -= 1
        # grid.y — divisor of (2 * intermediate_size) up to 16.
        out_dim = 2 * self.intermediate_size
        gy = min(16, out_dim)
        while out_dim % gy != 0 and gy > 1:
            gy -= 1
        return (gx, gy, 1)

    def default_block_dim(self) -> BlockDim:
        """W13 kernels (both bf16 and fp8) use 128 threads."""
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
        """Register the appropriate ``moe_w13_*`` task on the current PK.

        Args:
            x: Input DTensor. bf16 ``(B, hidden)`` for the bf16
                variant; FP8 E4M3 ``(B, hidden)`` for the fp8 variant.
            routing_indices: ``(num_experts, B)`` int32 expert-major
                routing tensor (produced by :class:`MoETopkRouting`).
            mask: ``(num_experts+1,)`` int32 prefix counts.
            output: Caller-allocated output DTensor of shape
                ``(B, num_experts_per_tok, 2*intermediate_size)``, bf16.
            x_scale: Required when ``dtype="fp8"``. Shape
                ``(B, hidden_size // 128)`` float32. Must be ``None``
                for the bf16 variant.
            grid_dim, block_dim: See :class:`MPKModule`.

        Returns:
            The provided ``output`` DTensor.
        """
        from ... import context as _ctx

        pk = _ctx.current_pk()

        if grid_dim is None:
            grid_dim = self.auto_grid_dim(x)
        if block_dim is None:
            block_dim = self.default_block_dim()

        # Inlined task registration (formerly pk.moe_w13_*_layer).
        from ....core import CyTBGraph
        from ....kernel import TBGraph

        if self.dtype == "bf16":
            if x_scale is not None:
                raise ValueError("MoEW13(bf16).compile: x_scale must be None")
            weight_dt = pk.attach_input(
                self.weight, name=f"{self.prefix}weight"
            )
            assert x.num_dims == 2  # (batch_size, hidden_size / world_size)
            assert weight_dt.num_dims == 3  # (num_experts, 2*intermediate_size, hidden_size)
            assert routing_indices.num_dims == 2  # (num_experts_per_tok, batch_size)
            assert mask.num_dims == 1  # (num_experts + 1)
            assert output.num_dims == 3  # (batch_size, num_expert_per_tok, 2*intermediate_size)
            tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
            tb_graph.new_input(x, (-1, -1, -1), 1, True)
            tb_graph.new_input(weight_dt, (-1, 1, -1), 2, True)
            tb_graph.new_input(routing_indices, (-1, -1, -1), -1, True)
            tb_graph.new_input(mask, (-1, -1, -1), -1, True)
            tb_graph.new_input(output, (-1, 2, -1), -1, True)
            pk.kn_graph.customized(
                [x, weight_dt, routing_indices, mask, output], tb_graph
            )
            if pk.target_cc == 100:
                pk.kn_graph.register_task(tb_graph, "moe_w13_linear_sm100")
            elif pk.target_cc == 90:
                pk.kn_graph.register_task(tb_graph, "moe_w13_linear_sm90")
            else:
                assert False
        else:
            if x_scale is None:
                raise ValueError(
                    "MoEW13(fp8).compile: x_scale is required "
                    f"(shape (B, {self.hidden_size // 128}) float32)"
                )
            weight_dt = pk.attach_input(
                self.weight, name=f"{self.prefix}weight"
            )
            weight_scale_dt = pk.attach_input(
                self.weight_scale, name=f"{self.prefix}weight_scale"
            )
            assert x.num_dims == 2
            assert x_scale.num_dims == 2
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
            pk.kn_graph.register_task(tb_graph, "moe_w13_fp8_sm100")
        return output
