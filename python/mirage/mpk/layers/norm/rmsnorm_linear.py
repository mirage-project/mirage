"""Fused RMSNorm + Linear catalog module.

Catalog counterpart to :meth:`PersistentKernel.rmsnorm_linear_layer`.
This is a single MPK task that streams each row of the input through an
RMSNorm reduction and immediately consumes the normalised activations as
the GEMM ``A``-operand, writing the projected output in one pass. The
GEMM ``B``-operand is the weight of the projection, stored
``(out_features, in_features)`` in the standard ``nn.Linear`` layout.

The motivating user is qwen3's QKV projection where, per layer, we want
to fold

    ``F.linear(input_layernorm(x), torch.cat([q_proj, k_proj, v_proj]))``

into one task instead of an ``RMSNorm`` task followed by a ``Linear``
task (two tasks, two trips through global memory, two scheduler
dispatches). Per the plan's 1:1 design rule (decision #2: "no internal
dispatch", "no auto-fusion compiler pass" — see plan's "Out of Scope"
list) the fused module is a *separate* class from ``RMSNorm + Linear``
composed: the model author opts into fusion explicitly by picking
``RMSNormLinear`` rather than chaining the two leaves.

Tensor contract
---------------

* Input  ``x``             : 2-D ``bfloat16`` ``DTensor`` of shape
                             ``(batch_size, hidden_size)``. Row-major.
                             Each row is one token; the kernel normalises
                             over the trailing ``hidden_size`` axis.
* Weight ``weight_norm``   : 1-D ``bfloat16`` ``nn.Parameter`` of shape
                             ``(hidden_size,)``. The RMSNorm scale —
                             same role as ``RMSNorm.weight``.
* Weight ``weight_linear`` : 2-D ``bfloat16`` ``nn.Parameter`` of shape
                             ``(out_features, hidden_size)`` (PyTorch
                             ``nn.Linear`` layout). For the qwen3 QKV
                             use-case this is typically a concatenated
                             ``[q_proj | k_proj | v_proj]`` along dim 0,
                             interleaved per the qwen3 builder's
                             ``shuffle_tensors`` call.
* Output                   : 2-D ``bfloat16`` ``DTensor`` of shape
                             ``(batch_size, out_features)``.

Epsilon (READ THIS)
-------------------

The MPK kernel hard-codes ``eps = 1e-6f`` in
``src/kernel/task_register.cc:226`` (``register_rmsnorm_linear_task``):

    code.e("    1e-6f,");

The ``eps`` ``__init__`` argument is therefore *PyTorch-only* — it
exists so :meth:`forward` reproduces whatever epsilon the HF config
specifies for the reference. If you instantiate
``RMSNormLinear(eps=1e-5)`` and call :meth:`compile`, the compiled
kernel still uses ``1e-6``. We default ``eps=1e-6`` to match the
compiled path. Mismatched eps between ``forward()`` and ``compile()``
is the same silent-correctness pitfall the sibling :class:`RMSNorm`
flags; the fix is the same: plumb ``eps`` through ``register_task``
params in a follow-up PR.

Alignment / shape constraints
-----------------------------

The kernel template is

    ``kernel::norm_linear_task_impl<bfloat16, BATCH_SIZE, OUTPUT_SIZE,
                                   REDUCTION_SIZE, O_STRIDE>(...)``

with the constants instantiated per task from the registered DTensor
shapes (see ``register_rmsnorm_linear_task`` in
``src/kernel/task_register.cc:215-221``). Constraints from
``include/mirage/persistent_kernel/tasks/ampere/norm_linear.cuh``:

* ``TILE_SIZE = 128`` and ``static_assert(REDUCTION_SIZE % TILE_SIZE ==
  0)`` — i.e. ``hidden_size`` must be a multiple of 128.
* ``OUTPUT_ATOM_SIZE`` is auto-derived as the smaller of 128 and a
  power-of-two upper-bounded by available shared memory, then
  ``NUM_OUTPUT_ATOMS = OUTPUT_SIZE / OUTPUT_ATOM_SIZE``. The kernel
  handles a non-zero ``LAST_OUTPUT_ATOM_SIZE`` correctly, but the
  qwen3-grade tiling assumes ``OUTPUT_SIZE`` divides cleanly into 96-
  or 64-element per-task slabs (see the parallelism note).
* CHUNK_SIZE = 16/sizeof(T) = 8 bf16 elements per cp.async chunk;
  ``TILE_SIZE / CHUNK_SIZE = 16`` is hard-wired.

For the per-task output slab (``out_features / grid.x``), the kernel
expects 64 or 96 elements per task, matching :func:`Linear`'s
``grid_for_linear`` heuristic. The :meth:`auto_grid_dim` here delegates
to the same helper that ``demo/qwen3/demo.py`` and
``python/mirage/mpk/models/qwen3/builder.py`` already use
(:func:`mirage.mpk.models.utils.grid_for_rmsnorm_linear_layer`) so the
catalog wrapper matches legacy behaviour 1:1.

Architecture support
--------------------

Currently **Ampere-only**. The .cuh files only exist as
``tasks/ampere/norm_linear.cuh`` and ``tasks/ampere/norm_linear_new.cuh``
— there is no ``hopper/`` or ``blackwell/`` variant of this task. The
qwen3 builder reflects this: the demo's QKV path uses the unfused
``rmsnorm_layer + linear_layer`` pair on Hopper/Blackwell and only
the fused path on Ampere is exercised. The graph layer in
``persistent_kernel.py:756`` registers the task name unconditionally as
``"rmsnorm_linear"`` (no ``_hopper`` switch), so calling
:meth:`compile` on a Hopper/Blackwell PK will produce a compile-time
error from ``register_rmsnorm_linear_task``. Users on Hopper/Blackwell
should chain :class:`RMSNorm` + :class:`Linear` instead until the
fused .cuh is ported.

Parallelism axis
----------------

One task per output-feature tile, identical to :class:`Linear`:
``grid.x`` slices the ``out_features`` dimension; each task computes a
``(batch_size, out_features // grid.x)`` slab. ``grid.y`` and
``grid.z`` are unused. The TBGraph wiring in
``persistent_kernel.py:rmsnorm_linear_layer`` partitions
``weight_linear`` on dim 0 and the output on dim 1 (``new_input(output,
(1, -1, -1), -1, True)``); the input and ``weight_norm`` are broadcast
to every task because the rmsnorm reduction is per-row and the
per-task GEMM consumes the same full row for its slab of outputs.

Why this differs from ``RMSNorm + Linear`` composed
---------------------------------------------------

The composed pair would:

1. Run an RMSNorm task: read ``x`` from gmem, write normalised
   activations to gmem.
2. Run a Linear task: re-read the normalised activations from gmem,
   GEMM with the projection weight, write the projection to gmem.

The fused task keeps the normalised activations resident in shared
memory across the GEMM, halving the gmem reads of the activations and
folding two scheduler dispatches into one. The price is that
``OUTPUT_ATOM_SIZE`` budget competes with the SMEM space the rmsnorm
buffers consume (see the ``MAX_OUTPUT_ATOM_SIZE`` calculation in
``norm_linear.cuh:45``), which is why the kernel caps the per-task
output width at 128 even when GMEM bandwidth would allow more.
"""
from __future__ import annotations

from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .._base import MPKModule
from ...context import current_pk

# DTensor is the public Cython class used throughout the codebase. The
# four dots reach from .../layers/norm/rmsnorm_linear.py up to
# .../mpk/__init__.py and across into mirage.core.
from ....core import DTensor


__all__ = ["RMSNormLinear"]


GridDim = Tuple[int, int, int]
BlockDim = Tuple[int, int, int]


def _grid_x_for_out_features(out_features: int) -> int:
    """Replicate ``grid_for_rmsnorm_linear_layer`` from the qwen3 helper.

    Mirrors :func:`mirage.mpk.models.utils.grid_for_rmsnorm_linear_layer`
    (and the same-named helper in ``demo/qwen3/demo.py``). We inline the
    logic rather than import to avoid a cross-package dependency on
    ``models.utils`` from the layer catalog (the catalog must be
    importable in isolation for tests). Keep this in sync if the helper
    changes upstream.

    * ``out_features / 96 > 400`` -> use ``// 256`` (workaround for the
      output-too-big regression the qwen3 helper documents).
    * else ``out_features % 96 == 0`` -> ``// 96``
    * else ``out_features % 64 == 0`` -> ``// 64``
    * else raise — caller must pass ``grid_dim`` explicitly.
    """
    if out_features / 96 > 400:
        if out_features % 256 != 0:
            raise ValueError(
                f"RMSNormLinear.auto_grid_dim: out_features={out_features} "
                "is in the 'too big' regime (>96*400) and is not divisible "
                "by 256. Pass grid_dim explicitly to compile()."
            )
        return out_features // 256
    if out_features % 96 == 0:
        return out_features // 96
    if out_features % 64 == 0:
        return out_features // 64
    raise ValueError(
        f"RMSNormLinear.auto_grid_dim: out_features={out_features} is not "
        "divisible by 96 or 64. Pass grid_dim explicitly to compile() if "
        "you need a different tile width."
    )


class RMSNormLinear(MPKModule):
    """Fused RMSNorm + Linear: ``y = F.linear(RMSNorm(x), weight_linear)``.

    Used for qwen3's fused input-layernorm + QKV projection, where
    ``weight_linear`` is typically a concatenated
    ``[q_proj | k_proj | v_proj]`` along dim 0 (sometimes interleaved
    by kv-head via ``shuffle_tensors``; see ``builder.py:281-295``).
    The single fused task replaces a separate ``RMSNorm`` task followed
    by a ``Linear`` task.

    Args:
        hidden_size: Number of features per token (the RMSNorm reduction
            axis and the linear's ``in_features``). Must be a multiple
            of 128 (TILE_SIZE in ``norm_linear.cuh:59``).
        out_features: Output feature dim of the linear projection. For
            :meth:`auto_grid_dim` to work, must be divisible by 96 or 64
            (or by 256 when in the >96*400 regime). Pass ``grid_dim``
            explicitly to :meth:`compile` to bypass.
        eps: RMSNorm variance epsilon used by the PyTorch reference.
            **The compiled MPK path ignores this value and uses
            ``1e-6`` hard-coded in ``src/kernel/task_register.cc:226``**
            — see module docstring. Defaults to ``1e-6`` so the two
            paths agree out of the box; mismatched ``eps`` is a silent
            correctness bug today.
        prefix: vLLM-style state_dict key prefix. The two parameters
            are attached to MPK with names ``f"{prefix}weight_norm"``
            and ``f"{prefix}weight_linear"``.

    Attributes:
        weight_norm (``nn.Parameter``): shape ``(hidden_size,)``,
            initialised to ones (matches ``Qwen3RMSNorm``).
        weight_linear (``nn.Parameter``): shape
            ``(out_features, hidden_size)``, ``torch.empty``-initialised
            so callers must ``load_state_dict`` (or ``.data.copy_``)
            before use.
        hidden_size, out_features, eps: cached for the grid heuristic
            and the PyTorch reference.
    """

    def __init__(
        self,
        hidden_size: int,
        out_features: int,
        eps: float = 1e-6,
        *,
        prefix: str = "",
    ) -> None:
        raise RuntimeError(
            "layers.RMSNormLinear (wraps pk.rmsnorm_linear_layer) is "
            "broken in Mirage: the generated call passes "
            "`runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS]` "
            "(an int) where the kernel template expects a void*, causing "
            "nvcc to fail with 'operand types are incompatible (\"int\" "
            "and \"void *\")' (see src/kernel/task_register.cc:225 — the "
            "fix belongs in the Mirage compiler, not here). For now, "
            "compose `RMSNorm` followed by `Linear` instead — qwen3's "
            "current demo also takes that unfused path."
        )
        super().__init__(prefix=prefix)
        self.hidden_size = hidden_size
        self.out_features = out_features
        self.eps = eps
        # Standard initialisation: norm weight = ones (matches
        # Qwen3RMSNorm / LlamaRMSNorm); linear weight = empty so
        # load_state_dict can overwrite without wasted init work. We use
        # bf16 because the kernel is bf16-only; the caller's
        # .to(device, dtype) will keep the dtype.
        self.weight_norm = nn.Parameter(
            torch.ones(hidden_size, dtype=torch.bfloat16)
        )
        self.weight_linear = nn.Parameter(
            torch.empty(out_features, hidden_size, dtype=torch.bfloat16)
        )

    # ------------------------------------------------------------------
    # PyTorch reference
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Faithful PyTorch reference: RMSNorm(x) then F.linear.

        Matches the standard transformers RMSNorm convention — variance
        accumulated in fp32, cast back to the input dtype before the
        scale — exactly as :class:`RMSNorm`.forward does, followed by a
        plain ``F.linear`` against ``weight_linear``. No bias term (the
        underlying kernel has none).
        """
        input_dtype = x.dtype
        variance = x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
        x_normed = x.to(torch.float32) * torch.rsqrt(variance + self.eps)
        x_normed = (x_normed.to(input_dtype) * self.weight_norm).to(input_dtype)
        return F.linear(x_normed, self.weight_linear)

    # ------------------------------------------------------------------
    # Grid-dim heuristic
    # ------------------------------------------------------------------
    def auto_grid_dim(self, x_dt: Any) -> GridDim:
        """Pick one task per output-feature tile, capped at ``num_workers``.

        Tile width follows the qwen3 helper
        :func:`mirage.mpk.models.utils.grid_for_rmsnorm_linear_layer`
        (replicated in :func:`_grid_x_for_out_features` above). The
        resulting ``grid.x`` is capped at ``current_pk().num_workers``
        so a single wave through the persistent runtime drains the
        task queue. ``x_dt`` is unused but accepted so subclasses with
        batch-size-dependent grids can branch on the input shape.
        """
        pk = current_pk()
        gx = _grid_x_for_out_features(self.out_features)
        gx = max(1, min(gx, int(pk.num_workers)))
        return (gx, 1, 1)

    # ------------------------------------------------------------------
    # MPK compile path
    # ------------------------------------------------------------------
    def compile(
        self,
        x: DTensor,
        *,
        output: Optional[Union[torch.Tensor, DTensor]] = None,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ) -> DTensor:
        """Register one fused ``rmsnorm_linear`` task for the current PK.

        Args:
            x:         2-D bf16 ``DTensor`` of shape
                       ``(batch_size, hidden_size)``.
            output:    Output buffer routing (same convention as
                       :class:`Linear`):

                       * ``None`` (default, production): allocate a
                         fresh DTensor of shape
                         ``(batch_size, out_features)`` via
                         ``pk.new_tensor``.
                       * ``torch.Tensor``: attach as a kernel input via
                         ``pk.attach_input`` so the host can read it
                         back after ``pk()`` returns (the canonical
                         test-readback path).
                       * ``DTensor``: caller-provided buffer, used
                         as-is.
            grid_dim:  Explicit grid override. ``None`` falls back to
                       :meth:`auto_grid_dim`.
            block_dim: Explicit block override. ``None`` falls back to
                       :meth:`default_block_dim` (128 on Ampere; on
                       Hopper/Blackwell the default is 256 but the
                       kernel itself does not currently support those
                       architectures — see the docstring).

        Returns:
            The output ``DTensor``.

        Raises:
            RuntimeError: if called outside ``pk.compile_scope()``
                (raised by :func:`current_pk`).
            ValueError: if ``x.num_dims != 2`` or ``output`` has an
                unsupported type.
        """
        pk = current_pk()

        if x.num_dims != 2:
            raise ValueError(
                f"RMSNormLinear.compile expects a 2-D input DTensor; "
                f"got num_dims={x.num_dims}"
            )

        # Attach the two learnable parameters. nn.Parameter is a
        # torch.Tensor subclass; ``self.weight_norm`` / ``weight_linear``
        # keep strong refs so the live-pointer GC pitfall flagged in the
        # plan's pre-implementation risks is moot.
        wn_dt = pk.attach_input(
            self.weight_norm, name=f"{self.prefix}weight_norm"
        )
        wl_dt = pk.attach_input(
            self.weight_linear, name=f"{self.prefix}weight_linear"
        )

        # Resolve the output DTensor.
        batch_size = x.dim(0)
        if output is None:
            out_dt = pk.new_tensor(
                dims=(batch_size, self.out_features),
                dtype=x.dtype,
                name=f"{self.prefix}rmsnorm_linear_out",
            )
        elif isinstance(output, torch.Tensor):
            # Test-readback path: bind a host torch tensor so the test
            # driver can inspect the result after pk() returns.
            out_dt = pk.attach_input(
                output, name=f"{self.prefix}rmsnorm_linear_out"
            )
        elif isinstance(output, DTensor):
            out_dt = output
        else:
            raise TypeError(
                "RMSNormLinear.compile output must be None, a torch.Tensor, "
                f"or a DTensor; got {type(output).__name__}"
            )

        # Resolve grid / block.
        if grid_dim is None:
            grid_dim = self.auto_grid_dim(x)
        if block_dim is None:
            block_dim = self.default_block_dim()

        pk.rmsnorm_linear_layer(
            input=x,
            weight_norm=wn_dt,
            weight_linear=wl_dt,
            output=out_dt,
            grid_dim=grid_dim,
            block_dim=block_dim,
        )
        return out_dt
