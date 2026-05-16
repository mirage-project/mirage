"""SiLU-Mul activation modules for the MPK layer catalog.

This module exposes two catalog modules:

* :class:`SiluMul` -- a pure activation that consumes a fused
  ``(B, 2*intermediate_size)`` gate-up tensor and produces
  ``(B, intermediate_size)`` via ``SiLU(gate) * up``. Backed by
  :meth:`PersistentKernel.silu_mul_layer` and the kernels in
  ``include/mirage/persistent_kernel/tasks/{ampere,hopper}/silu_mul*.cuh``.
  In particular, the Ampere kernel is::

      d_input = static_cast<T const *>(input_ptr);
      d_mul   = static_cast<T const *>(input_ptr) + OUTPUT_SIZE;
      for (i = ...)
          batch_idx = i / OUTPUT_SIZE;
          offset    = i % OUTPUT_SIZE;
          float gate = float(d_input[batch_idx * I_STRIDE + offset]);
          T     up   = d_mul[batch_idx * I_STRIDE + offset];
          d_output[batch_idx * O_STRIDE + offset] =
              T(gate / (1.0f + expf(-gate))) * up;

  i.e. per-task the layout is **halved (concatenated)**: the first
  ``OUTPUT_SIZE`` columns of the input row are the *gate* values, the
  next ``OUTPUT_SIZE`` columns are the *up* values (``OUTPUT_SIZE`` is
  the per-task output width = ``intermediate_size / grid.x``).

* :class:`SiluMulLinearWithResidual` -- a fused activation + linear
  down-projection + residual-add. Backed by
  :meth:`PersistentKernel.silu_mul_linear_with_residual_layer` and
  ``include/mirage/persistent_kernel/tasks/ampere/silu_mul_linear.cuh``
  (``kernel::silu_mul_linear_task_impl``). The kernel internally
  performs ``residual + (SiLU(gate) * up) @ w_down.T`` and uses the
  same halved layout for ``input``: per-task,
  ``d_input[:, :REDUCTION_SIZE]`` is gate and
  ``d_input[:, REDUCTION_SIZE:]`` is up (where ``REDUCTION_SIZE`` is the
  per-task K dimension = ``intermediate_size``; the down-proj weight is
  ``(hidden_size, intermediate_size)``).

Gate / up layout (read carefully)
---------------------------------
The kernel reads ``up`` from ``d_input + OUTPUT_SIZE`` *per task*. With
``grid_dim=(1, 1, 1)`` (or for the linear-with-residual variant which
does not slice the input on dim 1), this means the WHOLE input is
``[gate | up]`` (halved -- first ``intermediate_size`` columns = gate,
last ``intermediate_size`` columns = up). This is the layout
``forward()`` documents and uses.

For ``SiluMul`` with ``grid.x > 1``, the partitioning on dim 1 makes
the per-task view halved (gate slab followed by up slab), but the
whole-tensor layout becomes *interleaved by slab pairs*::

    column [t*W .. t*W + W/2)     -> gate slab t
    column [t*W + W/2 .. (t+1)*W) -> up   slab t       (W = 2*intermediate_size/grid.x)

The qwen3 MLP pipeline produces exactly this whole-tensor layout by
calling :meth:`PersistentKernel.shuffle_tensors` on the gate/up weight
rows upstream with ``num_groups=grid.x`` (see
``tests/runtime_python/test_mode/test_qwen3_mlp_testmode.py``,
``test_gateup_silu`` and ``test_gateup_silu_down``). In that case the
linear-layer output already has the per-task slabs in halved order, and
``SiluMul`` works without additional reshuffling.

For ``SiluMulLinearWithResidual``, the underlying linear is *not*
tiled on the K axis (it does its own K tiling internally), so the
input is always plain halved at the whole-tensor level -- no upstream
shuffle is needed.

Both modules' ``forward()`` consume a plain halved input
``[gate | up]``. To validate the kernel path against a shuffled
upstream, compose the shuffle in the test (the canonical test below
does so).

Parallelism axis
----------------
* :class:`SiluMul` partitions on the output / intermediate axis
  (``new_input(input, (1, -1, -1))`` -- dim 1 mapped to grid.x). The
  default heuristic picks ``grid.x = intermediate_size // 64`` (matching
  the qwen3 demo's ``num_tasks_gatedup // 2`` choice for
  ``intermediate_size = 2048 -> grid.x = 32``), capped at the worker
  pool. The kernel additionally loops the per-task ``BATCH_SIZE`` /
  ``OUTPUT_SIZE`` tile sequentially across the threads in the block, so
  ``BATCH_SIZE`` may be arbitrary (no alignment).

* :class:`SiluMulLinearWithResidual` partitions on the *output* dim
  (``residual``/``output`` dim 1 mapped to grid.x; the
  ``intermediate_size`` reduction is tiled internally with
  ``TILE_SIZE=128``). The default heuristic is
  ``grid.x = hidden_size // 64`` (matching qwen3's ``linear_with_residual``
  grid choice).

Alignment constraints
---------------------
* ``SiluMul``:
    * ``intermediate_size`` must be divisible by ``grid.x``.
    * Implicitly, ``2 * intermediate_size`` (the input width) must
      match what the linear-layer feeding it produced.
* ``SiluMulLinearWithResidual``:
    * ``intermediate_size`` (the K axis) must be divisible by
      ``TILE_SIZE = 128`` (kernel-internal pipeline).
    * ``hidden_size`` must be divisible by ``grid.x`` and the per-task
      ``OUTPUT_ATOM_SIZE`` (``min(hidden_size_per_task, 128)``).
    * The down-projection weight has shape
      ``(hidden_size, intermediate_size)`` -- i.e. the same convention
      PyTorch uses for ``nn.Linear(intermediate_size, hidden_size)``.

Dtype
-----
Both kernels are hard-coded to ``bfloat16`` at task-register time
(``register_silu_mul_task`` and
``register_silu_mul_linear_with_residual_task`` in
``src/kernel/task_register.cc`` both emit ``<bfloat16, ...>`` template
parameters unconditionally). Passing any other dtype is undefined.
"""
from __future__ import annotations

from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ....core import DTensor
from .._base import BlockDim, GridDim, MPKModule


__all__ = ["SiluMul", "SiluMulLinearWithResidual"]


def _split_gate_up_halved(gateup: torch.Tensor, intermediate_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split a fused ``(B, 2*intermediate)`` tensor into ``(gate, up)``.

    Uses the halved convention -- ``gateup[:, :intermediate]`` is gate,
    ``gateup[:, intermediate:]`` is up. This matches what the kernel
    reads per task (``d_mul = d_input + OUTPUT_SIZE``). See the module
    docstring for the relationship to upstream ``shuffle_tensors``.
    """
    if gateup.dim() != 2:
        raise ValueError(
            f"gateup must be 2-D (B, 2*intermediate_size); got shape {tuple(gateup.shape)}"
        )
    if gateup.size(1) != 2 * intermediate_size:
        raise ValueError(
            f"gateup.size(1) must equal 2*intermediate_size={2 * intermediate_size}; "
            f"got {gateup.size(1)}"
        )
    gate = gateup[:, :intermediate_size]
    up = gateup[:, intermediate_size:]
    return gate, up


class SiluMul(MPKModule):
    """``SiLU(gate) * up`` for a fused gate-up tensor.

    Args (``__init__``):
        intermediate_size: The per-side feature width, i.e. half of the
            fused input's trailing dim. The output's trailing dim is
            ``intermediate_size``; the input's is ``2 * intermediate_size``.
        prefix: vLLM/HF state_dict prefix. No weights live on this
            module, so ``prefix`` is used only as a uniquifier for the
            output DTensor name (``f"{prefix}silu_mul_out"``).

    Tensor contract:
        * input ``gateup``  : ``(B, 2 * intermediate_size)``, bfloat16,
          row-major. Layout: ``[gate | up]`` at the whole-tensor level
          when called with ``grid_dim=(1, 1, 1)``; *interleaved by slab
          pairs* (per-task halved) when ``grid.x > 1`` and the upstream
          linear's weights have been shuffled accordingly. See the
          module docstring.
        * output            : ``(B, intermediate_size)``, bfloat16,
          row-major.
    """

    def __init__(self, intermediate_size: int, *, prefix: str = "") -> None:
        super().__init__(prefix=prefix)
        self.intermediate_size = intermediate_size

    # ------------------------------------------------------------------
    # PyTorch reference
    # ------------------------------------------------------------------
    def forward(self, gateup: torch.Tensor) -> torch.Tensor:
        """Reference: ``F.silu(gate) * up`` on the halved layout.

        ``gateup[:, :intermediate_size]`` is treated as gate,
        ``gateup[:, intermediate_size:]`` as up. Run in fp32 internally
        for numerical stability, then cast back to the input dtype --
        matches the kernel which computes the SiLU in fp32 (``float
        input_val = float(d_input[...]); ... expf(-input_val) ...``).
        """
        gate, up = _split_gate_up_halved(gateup, self.intermediate_size)
        return (F.silu(gate.float()) * up.float()).to(gateup.dtype)

    # ------------------------------------------------------------------
    # Grid-dim heuristic
    # ------------------------------------------------------------------
    def auto_grid_dim(self, gateup_dt) -> GridDim:
        """Pick ``grid.x`` to match the qwen3 default: ``intermediate_size // 64``.

        The kernel partitions on the output (dim-1) axis. Each task owns
        ``intermediate_size / grid.x`` output columns and ``BATCH_SIZE``
        rows. The qwen3 demo uses ``num_tasks_gatedup // 2`` where
        ``num_tasks_gatedup = fused_outdim // 96`` or ``// 64`` -- which
        works out to ``intermediate_size // 64`` for the typical
        ``intermediate_size=2048`` case. We pick the same here, capped
        at the worker pool, and rounded down to a divisor of
        ``intermediate_size``.
        """
        from ... import context as _ctx

        pk = _ctx.current_pk()
        cap = max(1, int(getattr(pk, "num_workers", 1)))
        # Preferred -- mirrors qwen3 demo wiring.
        preferred = max(1, self.intermediate_size // 64)
        target = min(preferred, cap)
        # Walk down to the largest divisor of intermediate_size that
        # is <= target. ``grid.x`` must divide intermediate_size so each
        # task gets an integer number of output columns.
        gx = 1
        for d in range(1, target + 1):
            if self.intermediate_size % d == 0:
                gx = d
        return (gx, 1, 1)

    # ------------------------------------------------------------------
    # MPK compile
    # ------------------------------------------------------------------
    def compile(
        self,
        gateup: DTensor,
        *,
        output: Optional[Any] = None,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ) -> DTensor:
        """Register a ``silu_mul`` task on the active PersistentKernel.

        Args:
            gateup: DTensor of shape ``(B, 2 * intermediate_size)``.
            output: How to resolve the output DTensor.

                * ``None`` (default, production path): allocate a fresh
                  DTensor via ``pk.new_tensor`` of shape
                  ``(B, intermediate_size)``.
                * ``torch.Tensor``: attach via ``pk.attach_input`` so
                  the test driver can read the result.
                * ``DTensor``: use as-is (advanced; caller owns the
                  registration).
            grid_dim: Optional explicit grid override. ``None`` falls
                back to :meth:`auto_grid_dim`.
            block_dim: Optional explicit block override. ``None`` falls
                back to :meth:`default_block_dim`.

        Returns:
            The output DTensor (shape ``(B, intermediate_size)``,
            bfloat16).
        """
        from ... import context as _ctx

        pk = _ctx.current_pk()

        if gateup.num_dims != 2:
            raise ValueError(
                f"SiluMul expects a 2-D gateup DTensor; got num_dims={gateup.num_dims}"
            )
        if gateup.dim(1) != 2 * self.intermediate_size:
            raise ValueError(
                "SiluMul: gateup.dim(1) must equal 2*intermediate_size="
                f"{2 * self.intermediate_size}; got {gateup.dim(1)}"
            )

        if grid_dim is None:
            grid_dim = self.auto_grid_dim(gateup)
        if block_dim is None:
            block_dim = self.default_block_dim()

        batch_size = gateup.dim(0)
        if output is None:
            out_dt = pk.new_tensor(
                dims=(batch_size, self.intermediate_size),
                dtype=gateup.dtype,
                name=f"{self.prefix}silu_mul_out",
            )
        elif isinstance(output, torch.Tensor):
            out_dt = pk.attach_input(
                output, name=f"{self.prefix}silu_mul_out"
            )
        else:
            out_dt = output

        pk.silu_mul_layer(
            input=gateup,
            output=out_dt,
            grid_dim=grid_dim,
            block_dim=block_dim,
        )
        return out_dt


class SiluMulLinearWithResidual(MPKModule):
    """Fused ``F.linear(SiLU(gate) * up, w_down) + residual``.

    Args (``__init__``):
        intermediate_size: K axis of the down-projection -- equals the
            per-side width of the fused gate-up tensor. The fused input
            has trailing dim ``2 * intermediate_size``.
        hidden_size: Output / residual feature width. The down-proj
            weight is ``(hidden_size, intermediate_size)`` (PyTorch
            convention).
        prefix: vLLM/HF state_dict prefix. ``self.weight`` is loaded
            from ``state_dict[f"{prefix}weight"]``.

    Tensor contract:
        * input ``gateup``  : ``(B, 2 * intermediate_size)``, bfloat16.
          Halved layout ``[gate | up]`` -- the kernel reads
          ``d_mul = d_input + REDUCTION_SIZE`` with
          ``REDUCTION_SIZE = intermediate_size``. No upstream
          ``shuffle_tensors`` is required (the linear K-axis is not
          partitioned across CTAs).
        * input ``residual``: ``(B, hidden_size)``, bfloat16.
        * weight            : ``(hidden_size, intermediate_size)``,
          bfloat16. Stored as a stock ``nn.Parameter``.
        * output            : ``(B, hidden_size)``, bfloat16.
    """

    def __init__(
        self,
        intermediate_size: int,
        hidden_size: int,
        *,
        prefix: str = "",
    ) -> None:
        raise RuntimeError(
            "layers.SiluMulLinearWithResidual (wraps "
            "pk.silu_mul_linear_with_residual_layer) is broken in "
            "Mirage: the generated kernel call has an int-vs-void* "
            "argument-type mismatch, same root cause as RMSNormLinear "
            "(see src/kernel/task_register.cc — the fix belongs in the "
            "Mirage compiler). For now, compose `SiluMul` followed by "
            "`LinearWithResidual` instead — qwen3's current demo also "
            "takes that unfused path."
        )
        super().__init__(prefix=prefix)
        self.intermediate_size = intermediate_size
        self.hidden_size = hidden_size
        # down_proj weight: F.linear(silu_mul_out, weight) ->
        # (B, hidden_size) = (B, intermediate) @ weight.T.
        self.weight = nn.Parameter(
            torch.empty(hidden_size, intermediate_size)
        )

    # ------------------------------------------------------------------
    # PyTorch reference
    # ------------------------------------------------------------------
    def forward(
        self,
        gateup: torch.Tensor,
        residual: torch.Tensor,
    ) -> torch.Tensor:
        """Reference: ``F.linear(SiLU(gate) * up, weight) + residual``.

        Splits ``gateup`` in halved layout (see module docstring),
        computes SiLU-mul in fp32 to match the kernel's promotion, runs
        the linear in fp32, then casts the result back to the input
        dtype.
        """
        if residual.dim() != 2 or residual.size(1) != self.hidden_size:
            raise ValueError(
                f"residual must have shape (B, {self.hidden_size}); "
                f"got {tuple(residual.shape)}"
            )

        gate, up = _split_gate_up_halved(gateup, self.intermediate_size)
        silu_out = F.silu(gate.float()) * up.float()
        out = F.linear(silu_out, self.weight.float())
        return (out + residual.float()).to(gateup.dtype)

    # ------------------------------------------------------------------
    # Grid-dim heuristic
    # ------------------------------------------------------------------
    def auto_grid_dim(self, gateup_dt, residual_dt) -> GridDim:
        """Pick ``grid.x = hidden_size // 64``, capped at the worker pool.

        Mirrors the qwen3 demo's ``linear_with_residual`` grid
        (``hidden_size // 64``). The kernel partitions on the output /
        residual dim-1 axis. ``grid.x`` must divide ``hidden_size``.
        """
        from ... import context as _ctx

        pk = _ctx.current_pk()
        cap = max(1, int(getattr(pk, "num_workers", 1)))
        preferred = max(1, self.hidden_size // 64)
        target = min(preferred, cap)
        gx = 1
        for d in range(1, target + 1):
            if self.hidden_size % d == 0:
                gx = d
        return (gx, 1, 1)

    # ------------------------------------------------------------------
    # MPK compile
    # ------------------------------------------------------------------
    def compile(
        self,
        gateup: DTensor,
        residual: DTensor,
        *,
        output: Optional[Any] = None,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ) -> DTensor:
        """Register a ``silu_mul_linear_with_residual`` task.

        Args:
            gateup: DTensor of shape ``(B, 2 * intermediate_size)``,
                halved layout (gate then up).
            residual: DTensor of shape ``(B, hidden_size)`` -- added to
                the down-projection output.
            output: Output resolution -- see :meth:`SiluMul.compile`.
            grid_dim: Optional explicit grid override.
            block_dim: Optional explicit block override.

        Returns:
            The output DTensor (shape ``(B, hidden_size)``, bfloat16).
        """
        from ... import context as _ctx

        pk = _ctx.current_pk()

        if gateup.num_dims != 2:
            raise ValueError(
                f"SiluMulLinearWithResidual expects 2-D gateup; got num_dims={gateup.num_dims}"
            )
        if gateup.dim(1) != 2 * self.intermediate_size:
            raise ValueError(
                "SiluMulLinearWithResidual: gateup.dim(1) must equal "
                f"2*intermediate_size={2 * self.intermediate_size}; got {gateup.dim(1)}"
            )
        if residual.num_dims != 2:
            raise ValueError(
                f"residual must be 2-D; got num_dims={residual.num_dims}"
            )
        if residual.dim(1) != self.hidden_size:
            raise ValueError(
                f"residual.dim(1) must equal hidden_size={self.hidden_size}; "
                f"got {residual.dim(1)}"
            )
        if residual.dim(0) != gateup.dim(0):
            raise ValueError(
                f"residual.dim(0)={residual.dim(0)} must match "
                f"gateup.dim(0)={gateup.dim(0)}"
            )

        if grid_dim is None:
            grid_dim = self.auto_grid_dim(gateup, residual)
        if block_dim is None:
            block_dim = self.default_block_dim()

        # Attach the down-projection weight as a graph input. The
        # ``nn.Parameter`` is held by ``self.weight`` for the lifetime
        # of the module, so the underlying CUDA pointer stays valid.
        weight_dt = pk.attach_input(
            self.weight, name=f"{self.prefix}weight"
        )

        batch_size = gateup.dim(0)
        if output is None:
            out_dt = pk.new_tensor(
                dims=(batch_size, self.hidden_size),
                dtype=gateup.dtype,
                name=f"{self.prefix}silu_mul_linear_res_out",
            )
        elif isinstance(output, torch.Tensor):
            out_dt = pk.attach_input(
                output, name=f"{self.prefix}silu_mul_linear_res_out"
            )
        else:
            out_dt = output

        pk.silu_mul_linear_with_residual_layer(
            input=gateup,
            weight=weight_dt,
            residual=residual,
            output=out_dt,
            grid_dim=grid_dim,
            block_dim=block_dim,
        )
        return out_dt
