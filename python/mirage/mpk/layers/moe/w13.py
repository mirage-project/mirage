"""MoE W13 (fused gate+up) — bf16 and FP8 subclasses.

Wraps the gate+up grouped-GEMM tasks in
``include/mirage/persistent_kernel/tasks/{blackwell,hopper}``:

* :class:`MoEW13BF16` -> ``moe_w13_linear_sm100`` (Blackwell) /
  ``moe_w13_linear_sm90`` (Hopper).
* :class:`MoEW13FP8`  -> ``moe_w13_fp8_sm100`` (Blackwell only).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .._base import BlockDim, GridDim, MPKModule

from ....core import DTensor


__all__ = ["MoEW13BF16", "MoEW13FP8", "MoEW13"]


def _expert_for(routing_indices: torch.Tensor, num_experts_per_tok: int) -> torch.Tensor:
    """Convert (E, B) expert-major slot grid into (B, K) expert-id grid.

    ``routing_indices[e, t]`` is the 1-indexed slot in [1, K] of token
    ``t`` for expert ``e`` (0 if not routed). We invert it into a (B, K)
    grid indexed by token-then-slot. This is the (E, B) vs (B, K)
    confusion mentioned in MPK comments; the kernel reads (E, B),
    the eager reference here wants (B, K).
    """
    E, B = routing_indices.shape
    out = torch.full(
        (B, num_experts_per_tok), -1, dtype=torch.long, device=routing_indices.device
    )
    for e in range(E):
        slots = routing_indices[e]
        nz = slots.nonzero(as_tuple=False).squeeze(-1)
        if nz.numel() == 0:
            continue
        out[nz, slots[nz].long() - 1] = e
    return out


def _w13_auto_grid(num_experts: int, intermediate_size: int) -> GridDim:
    """Per-expert grid.x (<=8 divisor) + per-tile grid.y on 2*inter (<=16 divisor).

    Product targets up to ~128 CTAs per task instance — well under the
    typical num_workers budget (192-216 on Blackwell). The exact split
    mirrors the legacy ``_moe_expert_grid_x(preferred=8)`` /
    ``_moe_fp8_m_split(preferred=16)`` walks used by the DeepSeek V3
    builder. MMA-M=128 is hard-wired in the kernel; grid.x must divide
    ``num_experts`` so each CTA handles a contiguous expert range via
    ``task_metadata.expert_offset``.
    """
    gx = max(1, min(num_experts, 8))
    while num_experts % gx != 0 and gx > 1:
        gx -= 1
    out_dim = 2 * intermediate_size
    gy = min(16, out_dim)
    while out_dim % gy != 0 and gy > 1:
        gy -= 1
    return (gx, gy, 1)


class _MoEW13Base(MPKModule):
    """Shared base for MoEW13BF16 / MoEW13FP8.

    Owns ``(num_experts, num_experts_per_tok, hidden_size,
    intermediate_size)``. Per-expert weights are stacked along dim 0:
    ``self.weight`` is ``(num_experts, 2*intermediate, hidden)``; this is
    the layout the W13 kernel expects (gate and up are concatenated
    along the row axis, with gate as the first half).
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
                f"MoEW13: num_experts ({num_experts}) % ep_size ({ep_size}) != 0"
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
        """Expert split (grid.x<=8) + 2*intermediate tile split (grid.y<=16)."""
        return _w13_auto_grid(self.num_local_experts, self.intermediate_size)

    def default_block_dim(self) -> BlockDim:
        """W13 kernels (both bf16 and fp8) use 128 threads."""
        return (128, 1, 1)


class MoEW13BF16(_MoEW13Base):
    """bf16 fused gate+up projection per expert.

    Weight: ``(num_local_experts, 2*intermediate, hidden)`` bf16, stacked
    along dim 0. Under EP, only the local expert slice is allocated.
    Computes ``out[t, k] = x[t] @ W[e].T`` for each
    (token, slot) the routing tensor assigns to expert ``e``. Output is
    bf16 ``(B, num_experts_per_tok, 2*intermediate)`` with gate in the
    first half of the trailing axis and up in the second.
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
                self.num_local_experts, 2 * intermediate_size, hidden_size,
                dtype=torch.bfloat16,
            )
        )

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor,
                      *, expert_id: int, slot: str) -> bool:
        """Write one expert's gate|up slab into the local w13 slot.

        Returns False (writes nothing) if expert_id is not local to this rank.
        slot is 'gate' or 'up'; loaded_weight is (intermediate, hidden).
        """
        local = expert_id - self.local_expert_start
        if not (0 <= local < self.num_local_experts):
            return False
        inter = self.intermediate_size
        row0 = 0 if slot == "gate" else inter
        param.data[local, row0:row0 + inter].copy_(loaded_weight)
        return True

    def forward(self, x: torch.Tensor, routing_indices: torch.Tensor) -> torch.Tensor:
        """Expert-wise ``x @ W[e].T`` in fp32, cast to bf16."""
        if x.dim() != 2 or x.size(1) != self.hidden_size:
            raise ValueError(
                f"x must have shape (B, {self.hidden_size}); got {tuple(x.shape)}"
            )
        batch_size = x.size(0)
        K = self.num_experts_per_tok
        expert_for = _expert_for(routing_indices, K)
        x_f32 = x.float()
        w_f32 = self.weight.float()
        out_dim = 2 * self.intermediate_size
        out = torch.zeros(batch_size, K, out_dim, dtype=torch.float32, device=x.device)
        for b in range(batch_size):
            for k in range(K):
                e = int(expert_for[b, k].item())
                if e < 0:
                    continue
                out[b, k] = x_f32[b] @ w_f32[e].t()
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
        """Register ``moe_w13_linear_sm100`` (Blackwell) or ``moe_w13_linear_sm90`` (Hopper).

        Tensor contract:
          x: (B, hidden_size) bf16, the per-token activation.
          weight (self.weight, ``{prefix}weight``): (num_experts, 2*intermediate_size, hidden_size) bf16,
            row layout = [gate(intermediate) | up(intermediate)] along dim 1.
          routing_indices: (num_experts, B) int32, EXPERT-MAJOR (slot+1 or 0).
          mask: (num_experts + 1,) int32 prefix counts (kernel skips inactive experts).
          output: (B, num_experts_per_tok, 2*intermediate_size) bf16, 3-D.

        Notes: grid.x splits experts (expert_offset via task_metadata), grid.y splits the
        2*intermediate axis; MMA-M=128 hard-wired — grid.x must divide num_experts.
        """
        from ... import context as _ctx
        from ....core import CyTBGraph
        from ....kernel import TBGraph

        pk = _ctx.current_pk()
        grid_dim = grid_dim or self.auto_grid_dim(x)
        block_dim = block_dim or self.default_block_dim()

        weight_dt = pk.attach_input(self.weight, name=f"{self.prefix}weight")
        assert x.num_dims == 2                  # (B, hidden)
        assert weight_dt.num_dims == 3          # (E, 2*inter, hidden)
        assert routing_indices.num_dims == 2    # (E, B) expert-major
        assert mask.num_dims == 1               # (E + 1,)
        assert output.num_dims == 3             # (B, K, 2*inter)
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
            raise AssertionError(f"unsupported target_cc={pk.target_cc} for MoEW13BF16")
        return output


class MoEW13FP8(_MoEW13Base):
    """FP8 E4M3 fused gate+up projection per expert (Blackwell).

    Quirky scale layout: ``self.weight_scale`` is per-row, per-128-group
    fp32 — shape ``(num_experts, 2*intermediate, hidden//128)``. The
    ``moe_w13_fp8_sm100`` kernel always consumes *plain fp32* scales;
    it does NOT understand UE8M0 packing (that is only for the permute
    path). ``scale_ue8m0`` is kept for API symmetry but only logged.
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
        if hidden_size % 128 != 0:
            raise ValueError(
                f"MoEW13FP8 requires hidden_size % 128 == 0; got {hidden_size}"
            )
        self.scale_ue8m0 = scale_ue8m0
        self.weight = nn.Parameter(
            torch.empty(
                num_experts, 2 * intermediate_size, hidden_size,
                dtype=torch.float8_e4m3fn,
            ),
            requires_grad=False,
        )
        self.weight_scale = nn.Parameter(
            torch.empty(
                num_experts, 2 * intermediate_size, hidden_size // 128,
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
        """Dequant E4M3*fp32 scales then run expert-wise matmul (fp32 accum)."""
        if x.dim() != 2 or x.size(1) != self.hidden_size:
            raise ValueError(
                f"x must have shape (B, {self.hidden_size}); got {tuple(x.shape)}"
            )
        if x_scale is None:
            raise ValueError(
                f"MoEW13FP8.forward: x_scale required, shape "
                f"(B, {self.hidden_size // 128}) float32"
            )
        batch_size = x.size(0)
        K = self.num_experts_per_tok
        expert_for = _expert_for(routing_indices, K)

        x_f32 = x.float() * x_scale.float().repeat_interleave(128, dim=1)
        w_f32 = self.weight.float() * self.weight_scale.float().repeat_interleave(128, dim=2)

        out_dim = 2 * self.intermediate_size
        out = torch.zeros(batch_size, K, out_dim, dtype=torch.float32, device=x.device)
        for b in range(batch_size):
            for k in range(K):
                e = int(expert_for[b, k].item())
                if e < 0:
                    continue
                out[b, k] = x_f32[b] @ w_f32[e].t()
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
        """Register ``moe_w13_fp8_sm100`` (Blackwell only); x_scale is required.

        Tensor contract:
          x: (B, hidden_size) fp8_e4m3 (pre-quantized; viewed as uint8 in kernel).
          x_scale: (B, hidden_size // 128) fp32, per-128-element block scale (plain fp32, NOT UE8M0).
          weight (``{prefix}weight``): (num_experts, 2*intermediate_size, hidden_size) fp8_e4m3 (uint8).
          weight_scale (``{prefix}weight_scale``): (num_experts, 2*intermediate_size, hidden_size//128) fp32
            — plain fp32 per-row per-128-K block (the kernel does NOT accept UE8M0 here).
          routing_indices: (num_experts, B) int32, EXPERT-MAJOR.
          mask: (num_experts + 1,) int32 prefix counts.
          output: (B, num_experts_per_tok, 2*intermediate_size) bf16, [gate|up] along trailing axis.

        Notes: requires ``hidden_size % 128 == 0``; ``scale_ue8m0`` flag is ignored by this kernel.
        """
        from ... import context as _ctx
        from ....core import CyTBGraph
        from ....kernel import TBGraph

        pk = _ctx.current_pk()
        if x_scale is None:
            raise ValueError(
                f"MoEW13FP8.compile: x_scale is required "
                f"(shape (B, {self.hidden_size // 128}) float32)"
            )
        grid_dim = grid_dim or self.auto_grid_dim(x)
        block_dim = block_dim or self.default_block_dim()
        assert pk.target_cc == 100, "FP8 group GEMM requires SM100 (Blackwell)"

        weight_dt = pk.attach_input(self.weight, name=f"{self.prefix}weight")
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
        pk.kn_graph.register_task(tb_graph, "moe_w13_fp8_sm100")
        return output


def MoEW13(
    num_experts: int,
    num_experts_per_tok: int,
    hidden_size: int,
    intermediate_size: int,
    *,
    dtype: str = "bf16",
    scale_ue8m0: bool = False,
    prefix: str = "",
):
    """Back-compat factory: dispatches to :class:`MoEW13BF16` / :class:`MoEW13FP8`.

    Tensor contract (delegated to the chosen subclass ``compile``):
      x: (B, hidden_size) bf16 OR fp8_e4m3 (with required x_scale (B, hidden_size//128) fp32).
      weight: (num_experts, 2*intermediate_size, hidden_size) bf16 or fp8_e4m3.
      weight_scale (fp8 only): (num_experts, 2*intermediate_size, hidden_size//128) fp32 (plain, not UE8M0).
      routing_indices: (num_experts, B) int32 EXPERT-MAJOR; mask: (num_experts+1,) int32.
      output: (B, num_experts_per_tok, 2*intermediate_size) bf16, [gate|up] trailing.

    Notes: ``dtype='bf16'`` runs on SM90/100; ``dtype='fp8'`` requires SM100.
    """
    if dtype == "bf16":
        return MoEW13BF16(
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            prefix=prefix,
        )
    if dtype == "fp8":
        return MoEW13FP8(
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            scale_ue8m0=scale_ue8m0,
            prefix=prefix,
        )
    raise ValueError(f"MoEW13 dtype must be 'bf16' or 'fp8'; got {dtype!r}")
