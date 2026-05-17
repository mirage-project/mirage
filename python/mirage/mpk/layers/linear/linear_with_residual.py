"""Fused ``F.linear(x, W) + residual`` as an MPK catalog module.

Backed by :meth:`PersistentKernel.linear_with_residual_layer`
(``python/mirage/mpk/persistent_kernel.py`` ~line 2966) and the per-arch
CUDA kernels under
``include/mirage/persistent_kernel/tasks/{ampere,hopper,blackwell}/``.

Why a separate module instead of ``Linear(x) + add(residual)``
--------------------------------------------------------------

This is **the** fused-output primitive qwen3 uses for both
``o_proj`` (attention output projection) and ``down_proj`` (MLP output
projection). Fusing the elementwise add into the GEMM epilogue saves a
trip through global memory for the residual stream and is what the
existing demo path emits — see ``demo/qwen3/demo.py`` at the two
``mpk.linear_with_residual_layer(...)`` call sites. Keeping it as a
first-class catalog module preserves bit-for-bit equivalence with the
existing demo while letting model authors write the natural PyTorch
expression ``F.linear(x, W) + residual`` in ``forward``.

Tensor contract
---------------
* ``x``        : 2-D ``bfloat16`` DTensor, shape ``(batch_size, in_features)``.
* ``residual`` : 2-D ``bfloat16`` DTensor, shape ``(batch_size, out_features)``
  — **must already match the output shape** of the linear (i.e., the
  residual stream sits at the projection's output width, not its input
  width). For ``o_proj`` this is ``hidden_size``; for ``down_proj`` this
  is also ``hidden_size``. The kernel reads it row-major and adds it
  into the GEMM accumulator before writing.
* ``output``   : 2-D ``bfloat16`` DTensor, shape ``(batch_size, out_features)``.
* ``weight``   : 2-D ``bfloat16`` DTensor, shape ``(out_features, in_features)``
  — standard PyTorch ``nn.Linear`` layout (output features outer). The
  kernel does ``x @ weight.T + residual`` (i.e., consumes the weight as
  ``[out, in]`` and contracts on the inner axis).

The four ``num_dims == 2`` asserts live in
``persistent_kernel.py:linear_with_residual_layer`` (~line 2976).

Multi-GPU residual masking
--------------------------
When ``world_size > 1`` and this rank is not rank 0, the layer disables
the residual add (``enable_residual = 0``, passed as task param 0 in
``persistent_kernel.py:2989``). This is so the residual stream is added
**exactly once** across a TP shard, on rank 0, and the other ranks'
linear outputs only carry the matmul contribution. The PyTorch reference
``forward()`` here is the single-GPU case (always add) — the masking is
a property of the distributed runtime, not the algebra of the op.

Parallelism axis & alignment
----------------------------
The kernel tiles the output (``out_features``) axis across the grid:
each task owns one ``OUTPUT_ATOM_SIZE``-wide column slab of the output
(and of the residual).

* Hopper / Ampere kernels (``linear_swapAB_with_residual_hopper``,
  ``linear_with_residual``): per-task output slab is **64**, so
  ``out_features`` must be divisible by 64 and a natural grid choice is
  ``(out_features // 64, 1, 1)``. The qwen3 demo picks exactly this
  (``grid_dim=(hidden_size // 64, 1, 1)``).
* Blackwell SM100 kernel (``linear_with_residual_sm100``): per-task
  output slab is **128** (see ``OUTPUT_ATOM_SIZE = 128`` in
  ``linear_sm100_mpk.cuh:311``), so ``out_features`` must be divisible
  by 128 and the natural grid is ``(out_features // 128, 1, 1)``.

Both choices saturate the parallelism dimension; if it exceeds
``current_pk().num_workers`` we cap at ``num_workers`` (the task
descriptors are still emitted; workers process them serially).

Accumulate dtype
----------------
The matmul accumulator is fp32 inside each kernel (MMA fp32 accumulate
on Hopper/Blackwell, software fp32 on Ampere), then the
``+ residual`` is performed in fp32 inside the epilogue, and the final
write is bf16. Reference computation in ``forward()`` follows the
PyTorch default (``F.linear`` accumulates in fp32 internally for bf16
inputs), so the algebraic match is exact modulo a single bf16 rounding
step at the store; ``atol=rtol=1.0`` is the convention used in the
existing fused-MLP test (``test_qwen3_mlp_testmode.test_gateup_silu_down``).
"""
from __future__ import annotations

from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .._base import MPKModule


__all__ = ["LinearWithResidual"]


GridDim = Tuple[int, int, int]
BlockDim = Tuple[int, int, int]


class LinearWithResidual(MPKModule):
    """Fused ``(x @ weight.T) + residual`` — qwen3 ``o_proj`` / ``down_proj``.

    Owns the projection weight as an ``nn.Parameter`` so that
    ``state_dict`` / ``load_state_dict`` round-trip with HF checkpoints
    using the standard ``f"{prefix}weight"`` key. Bias is not exposed
    because the qwen3 callers do not use it (and neither does the
    underlying ``pk.linear_with_residual_layer``).

    Args:
        in_features:  Inner dimension of the weight (contraction axis).
            For ``o_proj`` this is ``num_heads * head_dim``; for
            ``down_proj`` this is ``intermediate_size``.
        out_features: Outer dimension of the weight; also the width of
            the residual and of the produced output. Must be divisible
            by the kernel's output-tile size (64 on Hopper/Ampere, 128
            on Blackwell SM100).
        prefix:       state_dict / tensor-name prefix. The parameter is
            registered as ``f"{prefix}weight"``.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        prefix: str = "",
    ) -> None:
        super().__init__(prefix=prefix)
        self.in_features = in_features
        self.out_features = out_features
        # PyTorch convention: weight is (out_features, in_features). The
        # MPK kernel consumes it in the same layout (the swapAB Hopper
        # kernel transposes internally; the Ampere/Blackwell kernels
        # operate on the [out, in] layout directly).
        self.weight = nn.Parameter(torch.empty(out_features, in_features))

    # ------------------------------------------------------------------
    # PyTorch reference path
    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> torch.Tensor:
        """Eager reference: ``F.linear(x, W) + residual``.

        Args:
            x:        ``(*, in_features)`` tensor (typically 2-D).
            residual: ``(*, out_features)`` tensor; same leading shape
                      as the output of the linear.
        Returns:
            Tensor with the same shape as ``residual``.
        """
        return F.linear(x, self.weight) + residual

    # ------------------------------------------------------------------
    # Grid heuristic
    # ------------------------------------------------------------------
    def auto_grid_dim(self, x_dt=None, residual_dt=None) -> GridDim:
        """Pick ``grid.x`` = ``out_features // OUTPUT_ATOM_SIZE``, capped at workers.

        Mirrors the qwen3 demo choice
        ``grid_dim=(hidden_size // 64, 1, 1)`` on Hopper. On Blackwell
        SM100 the kernel's atom is 128, so we use ``// 128``.

        ``x_dt`` / ``residual_dt`` are accepted for signature consistency
        with the base class but the heuristic only depends on
        ``self.out_features`` (the parallelism axis) and the active PK's
        ``target_cc`` and ``num_workers``.
        """
        from ... import context as _ctx
        pk = _ctx.current_pk()
        # OUTPUT_ATOM_SIZE: 128 on SM100, 64 on Hopper/Ampere. See module
        # docstring "Parallelism axis & alignment".
        atom = 128 if pk.target_cc >= 100 else 64
        if self.out_features % atom != 0:
            raise ValueError(
                f"LinearWithResidual: out_features={self.out_features} is "
                f"not divisible by the kernel output atom size {atom} "
                f"(target_cc={pk.target_cc}). Pad out_features or pick a "
                "different layer."
            )
        gx = self.out_features // atom
        # Cap at num_workers — workers will serialize extra task descriptors,
        # but we don't want to emit more than the pool can hold.
        gx = max(1, min(gx, int(pk.num_workers)))
        return (gx, 1, 1)

    # ------------------------------------------------------------------
    # MPK compile path
    # ------------------------------------------------------------------
    def compile(
        self,
        x,
        residual,
        *,
        output: Optional[Any] = None,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ):
        """Register a ``linear_with_residual`` task into the active PK.

        Args:
            x:        Input DTensor, shape ``(batch_size, in_features)``.
            residual: Residual DTensor, shape ``(batch_size, out_features)``.
                Must already be a DTensor — there is no ``torch.Tensor``
                shortcut here because the residual normally comes from a
                prior MPK op (a Linear, RMSNorm, etc.).
            output:   Optional output destination:
                * ``None`` (production): allocate a fresh DTensor via
                  ``pk.new_tensor`` matching the residual's shape.
                * ``torch.Tensor`` (test mode): bind a host buffer via
                  ``pk.attach_input`` so the driver can inspect the
                  result after ``pk()`` returns.
                * ``DTensor``: use as-is (advanced; caller owns its
                  registration).
            grid_dim: Optional explicit grid override. ``None`` falls
                      back to :meth:`auto_grid_dim`.
            block_dim: Optional explicit block override. ``None`` falls
                      back to :meth:`default_block_dim` (128 on Ampere,
                      256 on Hopper/Blackwell).

        Returns:
            The output DTensor.
        """
        from ... import context as _ctx
        pk = _ctx.current_pk()

        # Attach the parameter weight as a graph input. nn.Parameter is a
        # torch.Tensor subclass so attach_input accepts it directly; the
        # module retains a reference via self.weight so the underlying
        # storage outlives the PK's lifetime.
        w_dt = pk.attach_input(self.weight, name=f"{self.prefix}weight")

        # Resolve the output DTensor.
        out_features = self.out_features
        batch_size = residual.dim(0)
        if output is None:
            # Production: allocate a fresh CUDA tensor sized to the residual.
            out_dt = pk.new_tensor(
                dims=(batch_size, out_features),
                dtype=residual.dtype,
                name=f"{self.prefix}linear_with_residual_out",
            )
        elif isinstance(output, torch.Tensor):
            # Test-readback: bind a host buffer as a kernel tensor.
            out_dt = pk.attach_input(
                output,
                name=f"{self.prefix}linear_with_residual_out",
            )
        else:
            # Assume DTensor or DTensor-like.
            out_dt = output

        if grid_dim is None:
            grid_dim = self.auto_grid_dim(x, residual)
        if block_dim is None:
            block_dim = self.default_block_dim()

        # Inlined task registration (the body that used to live on
        # ``PersistentKernel.linear_with_residual_layer``). Each catalog
        # module owns its own task wiring so adding a new layer doesn't
        # require editing ``persistent_kernel.py``.
        from ....core import CyTBGraph
        from ....kernel import TBGraph

        assert x.num_dims == 2  # (batch_size, hidden_size / world_size)
        assert w_dt.num_dims == 2  # (hidden_size, hidden_size / world_size)
        assert residual.num_dims == 2  # (batch_size, hidden_size)
        assert out_dt.num_dims == 2  # (batch_size, hidden_size)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(x, (-1, -1, -1), 1, True)
        tb_graph.new_input(w_dt, (0, -1, -1), 1, True)
        tb_graph.new_input(residual, (1, -1, -1), -1, True)
        tb_graph.new_input(out_dt, (1, -1, -1), -1, True)
        pk.kn_graph.customized([x, w_dt, residual, out_dt], tb_graph)

        # On non-root TP ranks the residual must NOT be added — the
        # allreduce-merge step adds it exactly once on rank 0. Task
        # param[0] is ``enable_residual``.
        enable_residual = 1
        if pk.world_size > 1 and pk.mpi_rank != 0:
            enable_residual = 0
        params = [enable_residual]

        if 100 <= pk.target_cc < 120:
            pk.kn_graph.register_task(
                tb_graph, "linear_with_residual_sm100", params
            )
        elif 90 <= pk.target_cc < 100:
            # Hopper: the legacy code had a branch on per-task output
            # width that always picked the swapAB variant, so we use the
            # one variant unconditionally.
            pk.kn_graph.register_task(
                tb_graph, "linear_swapAB_with_residual_hopper", params
            )
        elif 80 <= pk.target_cc < 90:
            pk.kn_graph.register_task(tb_graph, "linear_with_residual")
        else:
            raise RuntimeError(
                f"LinearWithResidual.compile: unsupported compute "
                f"capability {pk.target_cc}. Supported: SM80-89 (Ampere), "
                f"SM90 (Hopper), SM100-119 (Blackwell)."
            )
        return out_dt
