"""MoE W2 (down projection) — bf16 and FP8 subclasses.

Wraps the per-expert down-projection tasks in
``include/mirage/persistent_kernel/tasks/{blackwell,hopper}``:

* :class:`MoEW2BF16` -> ``moe_w2_linear_sm100`` / ``moe_w2_linear_sm90``.
* :class:`MoEW2FP8`  -> ``moe_w2_fp8_sm100`` (Blackwell only).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .._base import BlockDim, GridDim, MPKModule
from .w13 import _expert_for

from ....core import DTensor


__all__ = ["MoEW2BF16", "MoEW2FP8", "MoEW2"]


def _w2_auto_grid(num_experts: int, hidden_size: int) -> GridDim:
    """Per-expert grid.x (<=10 divisor) + per-tile grid.y on hidden (<=14 divisor).

    Product targets up to ~140 CTAs — within the typical num_workers
    budget (192-216 on Blackwell). The walks mirror
    ``_moe_expert_grid_x(preferred=10)`` and ``_moe_fp8_m_split(preferred=14)``
    from the DeepSeek V3 builder. Extra constraint: each per-CTA N-slab
    (``hidden_size // gy``) MUST be a multiple of MMA_M=128 or the sm100
    kernel writes a partial slab with wrong column stride; small-shape
    unit tests hit this (e.g. hidden=256 walks down to gy<=2).
    """
    gx = max(1, min(num_experts, 10))
    while num_experts % gx != 0 and gx > 1:
        gx -= 1
    gy = min(14, hidden_size)
    while gy > 1 and (hidden_size % gy != 0 or hidden_size // gy < 128):
        gy -= 1
    return (gx, gy, 1)


class _MoEW2Base(MPKModule):
    """Shared base for MoEW2BF16 / MoEW2FP8.

    Per-expert weights are stacked along dim 0: ``self.weight`` is
    ``(num_local_experts, hidden, intermediate)``. Input is 3-D ``(B, K,
    intermediate)`` (produced by silu_mul on the W13 output), output is
    3-D ``(B, K, hidden)``.
    """

    def __init__(
        self,
        num_experts: int,
        num_experts_per_tok: int,
        hidden_size: int,
        intermediate_size: int,
        *,
        ep_size: int = 1,
        ep_rank: int = 0,
        prefix: str = "",
    ) -> None:
        super().__init__(prefix=prefix)
        if num_experts % ep_size != 0:
            raise ValueError(
                f"MoEW2: num_experts ({num_experts}) % ep_size ({ep_size}) != 0"
            )
        if not (0 <= ep_rank < ep_size):
            raise ValueError(
                f"MoEW2: ep_rank ({ep_rank}) must be in [0, ep_size={ep_size})"
            )
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.ep_size = ep_size
        self.ep_rank = ep_rank
        self.num_local_experts = num_experts // ep_size
        self.local_expert_start = ep_rank * self.num_local_experts

    def auto_grid_dim(self, *_: DTensor) -> GridDim:
        """Expert split (grid.x<=10) + hidden tile split (grid.y<=14)."""
        return _w2_auto_grid(self.num_local_experts, self.hidden_size)

    def default_block_dim(self) -> BlockDim:
        return (128, 1, 1)


class MoEW2BF16(_MoEW2Base):
    """bf16 per-expert down projection.

    Weight: ``(num_local_experts, hidden, intermediate)`` bf16, stacked along
    dim 0. Under EP, only the local expert slice is allocated.
    Computes ``out[t, k] = x[t, k] @ W[e].T`` for each
    (token, slot) the routing tensor assigns to expert ``e``.
    """

    def __init__(
        self,
        num_experts: int,
        num_experts_per_tok: int,
        hidden_size: int,
        intermediate_size: int,
        *,
        ep_size: int = 1,
        ep_rank: int = 0,
        prefix: str = "",
    ) -> None:
        super().__init__(
            num_experts, num_experts_per_tok, hidden_size, intermediate_size,
            ep_size=ep_size, ep_rank=ep_rank, prefix=prefix,
        )
        self.weight = nn.Parameter(
            torch.empty(
                self.num_local_experts, hidden_size, intermediate_size,
                dtype=torch.bfloat16,
            )
        )

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor,
                      *, expert_id: int) -> bool:
        """Write one expert's down-proj slab into the local w2 slot.

        Returns False (writes nothing) if expert_id is not local to this rank.
        loaded_weight is (hidden, intermediate).
        Invoked directly by the owning model's load_weights with
        (param, loaded_weight, expert_id[, slot]); it is NOT attached to the
        parameter and is not reached via the default resolve_weight path
        (which cannot supply expert_id/slot).
        """
        local = expert_id - self.local_expert_start
        if not (0 <= local < self.num_local_experts):
            return False
        param.data[local].copy_(loaded_weight)
        return True

    def forward(self, x: torch.Tensor, routing_indices: torch.Tensor) -> torch.Tensor:
        """Expert-wise ``x[t, k] @ W[e].T`` in fp32, cast to bf16."""
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
        x_f32 = x.float()
        w_f32 = self.weight.float()
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

    def compile(
        self,
        x: DTensor,
        routing_indices: DTensor,
        mask: DTensor,
        output: DTensor,
        *,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ) -> DTensor:
        """Register ``moe_w2_linear_sm100`` (Blackwell) or ``moe_w2_linear_sm90`` (Hopper).

        Tensor contract:
          x: (B, num_experts_per_tok, intermediate_size) bf16 (3-D, output of silu_mul).
          weight (``{prefix}weight``): (num_local_experts, hidden_size, intermediate_size) bf16, stacked dim 0.
          routing_indices: (num_experts, B) int32, EXPERT-MAJOR (slot+1 or 0).
          mask: (num_experts + 1,) int32 prefix counts.
          output: (B, num_experts_per_tok, hidden_size) bf16, 3-D.

        Notes: grid.x splits experts, grid.y splits hidden (per-CTA N-slab must be a
        multiple of MMA_M=128 on sm100).
        """
        from ... import context as _ctx
        from ....core import CyTBGraph
        from ....kernel import TBGraph

        pk = _ctx.current_pk()
        grid_dim = grid_dim or self.auto_grid_dim(x)
        block_dim = block_dim or self.default_block_dim()

        weight_dt = pk.attach_input(self.weight, name=f"{self.prefix}weight")
        assert x.num_dims == 3                  # (B, K, inter)
        assert weight_dt.num_dims == 3          # (E, hidden, inter)
        assert routing_indices.num_dims == 2    # (E, B) expert-major
        assert mask.num_dims == 1               # (E + 1,)
        assert output.num_dims == 3             # (B, K, hidden)
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
            raise AssertionError(f"unsupported target_cc={pk.target_cc} for MoEW2BF16")
        return output


class MoEW2FP8(_MoEW2Base):
    """FP8 E4M3 per-expert down projection (Blackwell).

    Scale layout: ``self.weight_scale`` is per-row, per-128-group fp32 —
    shape ``(num_experts, hidden, intermediate//128)``. The kernel
    consumes the un-expanded ``intermediate//128`` layout (NOT the
    repeat_interleaved-to-intermediate layout some legacy code uses for
    other kernels). ``scale_ue8m0`` is kept for API symmetry but the
    ``moe_w2_fp8_sm100`` kernel always reads plain fp32 scales.
    """

    def __init__(
        self,
        num_experts: int,
        num_experts_per_tok: int,
        hidden_size: int,
        intermediate_size: int,
        *,
        scale_ue8m0: bool = False,
        prefix: str = "",
    ) -> None:
        super().__init__(
            num_experts, num_experts_per_tok, hidden_size, intermediate_size,
            prefix=prefix,
        )
        if intermediate_size % 128 != 0:
            raise ValueError(
                f"MoEW2FP8 requires intermediate_size % 128 == 0; "
                f"got {intermediate_size}"
            )
        self.scale_ue8m0 = scale_ue8m0
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

    def forward(
        self,
        x: torch.Tensor,
        routing_indices: torch.Tensor,
        *,
        x_scale: torch.Tensor,
    ) -> torch.Tensor:
        """Dequant E4M3*fp32 scales (per-128-group on trailing axis) then matmul."""
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
        if x_scale is None:
            raise ValueError(
                f"MoEW2FP8.forward: x_scale required, shape "
                f"(B, K, {self.intermediate_size // 128}) float32"
            )
        batch_size = x.size(0)
        K = self.num_experts_per_tok
        expert_for = _expert_for(routing_indices, K)

        x_f32 = x.float() * x_scale.float().repeat_interleave(128, dim=2)
        w_f32 = self.weight.float() * self.weight_scale.float().repeat_interleave(128, dim=2)

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
        """Register ``moe_w2_fp8_sm100`` (Blackwell only); x_scale is required.

        Tensor contract:
          x: (B, num_experts_per_tok, intermediate_size) fp8_e4m3 (viewed as uint8 in kernel).
          x_scale: (B, num_experts_per_tok, intermediate_size // 128) fp32, plain fp32 per-128-K block.
          weight (``{prefix}weight``): (num_experts, hidden_size, intermediate_size) fp8_e4m3.
          weight_scale (``{prefix}weight_scale``): (num_experts, hidden_size, intermediate_size//128) fp32
            — plain fp32 per-row per-128-K block (NOT UE8M0, despite the flag).
          routing_indices: (num_experts, B) int32, EXPERT-MAJOR.
          mask: (num_experts + 1,) int32 prefix counts.
          output: (B, num_experts_per_tok, hidden_size) bf16.

        Notes: requires ``intermediate_size % 128 == 0`` and SM100; ``scale_ue8m0`` is ignored.
        """
        from ... import context as _ctx
        from ....core import CyTBGraph
        from ....kernel import TBGraph

        pk = _ctx.current_pk()
        if x_scale is None:
            raise ValueError(
                f"MoEW2FP8.compile: x_scale is required "
                f"(shape (B, K, {self.intermediate_size // 128}) float32)"
            )
        grid_dim = grid_dim or self.auto_grid_dim(x)
        block_dim = block_dim or self.default_block_dim()
        assert pk.target_cc == 100, "FP8 group GEMM requires SM100 (Blackwell)"

        weight_dt = pk.attach_input(self.weight, name=f"{self.prefix}weight")
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
        # store_in_dmem=True for all inputs to work around a TBGraph segfault
        # with 3D tensors when store_in_dmem=False.
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
        pk.kn_graph.register_task(tb_graph, "moe_w2_fp8_sm100")
        return output


def MoEW2(
    num_experts: int,
    num_experts_per_tok: int,
    hidden_size: int,
    intermediate_size: int,
    *,
    dtype: str = "bf16",
    ep_size: int = 1,
    ep_rank: int = 0,
    scale_ue8m0: bool = False,
    prefix: str = "",
):
    """Back-compat factory: dispatches to :class:`MoEW2BF16` / :class:`MoEW2FP8`.

    Tensor contract (delegated to the chosen subclass ``compile``):
      x: (B, num_experts_per_tok, intermediate_size) bf16 OR fp8_e4m3
        (with required x_scale (B, K, intermediate_size//128) fp32).
      weight: (num_experts, hidden_size, intermediate_size) bf16 or fp8_e4m3.
      weight_scale (fp8 only): (num_experts, hidden_size, intermediate_size//128) fp32 (plain, not UE8M0).
      routing_indices: (num_experts, B) int32 EXPERT-MAJOR; mask: (num_experts+1,) int32.
      output: (B, num_experts_per_tok, hidden_size) bf16.

    Notes: ``dtype='bf16'`` runs on SM90/100; ``dtype='fp8'`` requires SM100.
    ``ep_size``/``ep_rank`` are forwarded to :class:`MoEW2BF16` only (FP8 is not EP-aware yet).
    """
    if dtype == "bf16":
        return MoEW2BF16(
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            ep_size=ep_size,
            ep_rank=ep_rank,
            prefix=prefix,
        )
    if dtype == "fp8":
        return MoEW2FP8(
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            scale_ue8m0=scale_ue8m0,
            prefix=prefix,
        )
    raise ValueError(f"MoEW2 dtype must be 'bf16' or 'fp8'; got {dtype!r}")
