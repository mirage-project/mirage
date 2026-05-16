"""Dense bf16 linear projection (no bias, no residual) for the MPK catalog.

Backed by :meth:`PersistentKernel.linear_layer`. The underlying CUDA tasks
are architecture-specific:

* ``include/mirage/persistent_kernel/tasks/ampere/linear.cuh``        (CC 80–89, task ``"linear"``)
* ``include/mirage/persistent_kernel/tasks/hopper/linear_swapAB_hopper.cuh`` (CC 90, task ``"linear_swapAB_hopper"``)
* ``include/mirage/persistent_kernel/tasks/blackwell/linear_sm100_mpk.cuh``  (CC 100, task ``"linear_sm100"``)

The python wrapper in ``persistent_kernel.py`` (line ~2935) picks the
task name from ``self.target_cc``; we delegate to it unchanged.

Tensor contract
---------------
* ``x``     : 2-D ``DTensor``, ``dtype=bfloat16``, shape
              ``(batch_size, in_features)``. Row-major, contiguous.
* ``weight``: 2-D ``nn.Parameter``, ``dtype=bfloat16``, shape
              ``(out_features, in_features)`` — the PyTorch standard
              layout for ``nn.Linear.weight``. This is also what
              ``pk.linear_layer`` expects: ``weight.dim(0) == out_features``
              and ``weight.dim(1) == in_features``. The kernel reads the
              weight as ``A``-operand (transposed via swapAB on Hopper and
              by layout on Blackwell), so no host-side transpose is needed.
* ``output``: 2-D ``DTensor`` (or ``torch.Tensor`` / ``None`` in
              ``compile()``), ``dtype=bfloat16``, shape
              ``(batch_size, out_features)``.

The PyTorch reference is the plain ``F.linear(x, weight)`` — i.e.
``x @ weight.T`` — with **no bias term** (the catalog's
``LinearWithResidual`` covers fused-residual; bias-only is not a kernel
the MPK runtime currently provides). The ``bias`` constructor flag is
accepted for ``nn.Linear`` API parity but defaults to ``False``;
passing ``True`` raises ``NotImplementedError``.

Parallelism axis
----------------
One task per output-feature tile. ``grid.x`` slices the
``out_features`` dimension; each task computes a
``(batch_size, out_features // grid.x)`` slab of the output. ``grid.y``
and ``grid.z`` are unused. The kernel's per-tile output width is set by
``output_size / grid.x``; the demos use the same heuristic that the
test fixtures encode as ``grid_for_linear`` in
``tests/runtime_python/test_mode/test_qwen3_mlp_testmode.py``:

* if ``out_features % 96 == 0`` → ``grid.x = out_features // 96``
* elif ``out_features % 64 == 0`` → ``grid.x = out_features // 64``

These two tile widths cover every dense linear in Qwen3 (gate/up at
``out=11008``, down at ``out=4096``, qkv/o projections at multiples of
``head_dim=128``). The pattern keeps each task's per-CTA output width
inside the ``OUTPUT_ATOM_SIZE=64`` window the Hopper/Ampere kernels
assume; on Blackwell the kernel's ``MMA_M`` is 128 internally, but
``out_features // grid.x`` is allowed to be 64 or 96 (the kernel pads
its internal tile to 128 — see ``linear_sm100_mpk.cuh``).

Alignment requirements
----------------------
``out_features`` MUST be divisible by 96 or 64; if neither, the auto
heuristic raises. Callers that need a different shape must pass
``grid_dim`` explicitly.

``in_features`` (reduction dim) must be divisible by the kernel's
``TILE_SIZE``: 128 for Ampere, 64 for Hopper/Blackwell. Qwen3's hidden
sizes (4096, 8192) satisfy both. The MPK runtime does not check this
itself — a misaligned ``in_features`` will produce silent wrong
results, not a compile error.

``batch_size`` (number of input rows) constraints come from the
``swapAB`` variants: Hopper asserts ``BATCH_SIZE <= 16`` in
``linear_swapAB_hopper.cuh``. For larger batches the calling code is
expected to chunk along batch dim; this is the same constraint that
the existing ``pk.linear_layer`` callers in ``demo/qwen3/demo.py``
work around. We do **not** insert that chunking here — it stays the
caller's responsibility, matching legacy behavior.

Dtype constraints
-----------------
Strictly bf16. The kernel is templated on ``cute::bfloat16_t`` and the
code-gen path in ``src/kernel/task_register.cc:1737`` hard-codes the
type. fp16 / fp32 / fp8 paths exist as separate catalog entries
(``linear_fp8_*``); do **not** route them through this module.
"""
from __future__ import annotations

from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .._base import MPKModule


__all__ = ["Linear"]


GridDim = Tuple[int, int, int]
BlockDim = Tuple[int, int, int]


def _grid_x_for_out_features(out_features: int) -> int:
    """Replicate ``grid_for_linear`` from the canonical qwen3 test fixture.

    Mirrors ``tests/runtime_python/test_mode/test_qwen3_mlp_testmode.py:25-33``
    so the catalog wrapper picks the same tile width the legacy demo
    picks. Anything finer would shrink the per-task output column count
    below the kernel's expected 64- or 96-element window.
    """
    if out_features % 96 == 0:
        return out_features // 96
    elif out_features % 64 == 0:
        return out_features // 64
    raise ValueError(
        f"Linear.auto_grid_dim: out_features={out_features} is not divisible "
        "by 96 or 64. Pass grid_dim explicitly to compile() if you need a "
        "different tile width."
    )


class Linear(MPKModule):
    """Plain bf16 dense linear projection: ``out = x @ weight.T``.

    No bias is supported by the underlying ``pk.linear_layer`` kernel.
    The constructor accepts a ``bias=`` kwarg for ``nn.Linear`` API
    parity, but only ``bias=False`` is implemented; passing ``bias=True``
    raises ``NotImplementedError`` at construction time. If your HF
    checkpoint contains a bias term, fold it into a separate add
    (``layers.add``) at load time or use a fused module.

    Args:
        in_features:  Reduction dim (matches the trailing dim of ``x``).
                      Must be divisible by 128 on Ampere, 64 on
                      Hopper/Blackwell.
        out_features: Output feature dim. Must be divisible by 96 or 64
                      for the auto-grid heuristic; pass ``grid_dim``
                      explicitly to compile() to bypass.
        bias:         Reserved for ``nn.Linear`` API parity. Must be
                      ``False``.
        prefix:       Tensor-name / state_dict-key prefix. The weight is
                      attached to MPK with name ``f"{prefix}weight"``.

    Attributes:
        weight: ``nn.Parameter`` of shape ``(out_features, in_features)``,
                ``dtype=torch.bfloat16``.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        *,
        prefix: str = "",
    ) -> None:
        super().__init__(prefix=prefix)
        if bias:
            # pk.linear_layer has no bias path. Document the workaround
            # in the error so the caller has a clear next step.
            raise NotImplementedError(
                "Linear(bias=True) is not supported by pk.linear_layer. "
                "Fold the bias into the weight at load time, or add it as "
                "a separate layers.add() call in your compile() body."
            )
        self.in_features = in_features
        self.out_features = out_features
        # Standard PyTorch nn.Linear weight layout: (out_features, in_features).
        # We default to bf16 because that's the only dtype the kernel
        # supports; callers should still be able to .to(dtype) for
        # consistency with the rest of the model.
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=torch.bfloat16)
        )

    # ------------------------------------------------------------------
    # PyTorch reference path
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Plain dense projection: ``F.linear(x, self.weight)``.

        Equivalent to ``x @ self.weight.T``. No bias term.
        """
        return F.linear(x, self.weight)

    # ------------------------------------------------------------------
    # Grid-dim heuristic
    # ------------------------------------------------------------------
    def auto_grid_dim(self, x_dt: Any) -> GridDim:
        """Pick one task per output tile, capped at ``num_workers``.

        Tile width is 96 if ``out_features % 96 == 0`` else 64
        (matches the legacy qwen3 selection in
        ``test_qwen3_mlp_testmode.py:25-33``). The resulting
        ``grid.x`` is then capped at ``current_pk().num_workers`` so a
        single wave through the persistent runtime is enough to clear
        all tasks. ``x_dt`` is unused but is passed so subclasses (e.g.
        a future split-K variant) can branch on input shape.
        """
        from ... import context as _ctx

        pk = _ctx.current_pk()
        gx = _grid_x_for_out_features(self.out_features)
        gx = max(1, min(gx, int(pk.num_workers)))
        return (gx, 1, 1)

    # ------------------------------------------------------------------
    # MPK compile path
    # ------------------------------------------------------------------
    def compile(
        self,
        x: Any,
        *,
        output: Optional[Any] = None,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ) -> Any:
        """Register a ``linear`` task computing ``x @ weight.T``.

        Args:
            x:         Input ``DTensor``, 2-D bf16 row-major with shape
                       ``(batch_size, in_features)``.
            output:    Destination for the result.
                       * ``None`` (production path): allocate a new
                         ``DTensor`` of shape
                         ``(batch_size, out_features)`` via
                         ``pk.new_tensor``.
                       * ``torch.Tensor``: attach as a kernel input
                         (test-readback path) so the host can inspect
                         the result after ``pk()`` returns.
                       * ``DTensor``: use the caller-provided buffer
                         directly.
            grid_dim:  Explicit grid override. ``None`` falls back to
                       :meth:`auto_grid_dim`.
            block_dim: Explicit block override. ``None`` falls back to
                       :meth:`default_block_dim` (128 on Ampere, 256 on
                       Hopper/Blackwell).

        Returns:
            The output ``DTensor`` (whichever path produced it).

        Raises:
            RuntimeError: if called outside a ``pk.compile_scope()``
                block (raised by :func:`current_pk`).
        """
        from ... import context as _ctx

        pk = _ctx.current_pk()

        if grid_dim is None:
            grid_dim = self.auto_grid_dim(x)
        if block_dim is None:
            block_dim = self.default_block_dim()

        # Attach the parameter as a graph input. nn.Parameter is a
        # torch.Tensor subclass; pk.attach_input handles it transparently
        # and tracks a reference (so the live-pointer GC pitfall flagged
        # in the plan's pre-implementation sanity checks is moot — the
        # nn.Module also keeps the Parameter alive via self.weight).
        w_dt = pk.attach_input(self.weight, name=f"{self.prefix}weight")

        # Resolve the output DTensor.
        batch_size = x.dim(0)
        if output is None:
            # Production path: allocate a fresh CUDA tensor.
            out_dt = pk.new_tensor(
                dims=(batch_size, self.out_features),
                dtype=x.dtype,
                name=f"{self.prefix}linear_out",
            )
        elif isinstance(output, torch.Tensor):
            # Test-readback path: bind a host torch tensor so the test
            # driver can inspect the result after pk() returns.
            out_dt = pk.attach_input(output, name=f"{self.prefix}linear_out")
        else:
            # DTensor (or DTensor-like) — caller owns the registration.
            out_dt = output

        pk.linear_layer(
            input=x,
            weight=w_dt,
            output=out_dt,
            grid_dim=grid_dim,
            block_dim=block_dim,
        )
        return out_dt
