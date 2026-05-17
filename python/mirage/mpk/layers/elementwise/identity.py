"""Identity / no-op layer for MPK.

Backed by ``PersistentKernel.identity_layer`` and the CUDA kernel
``include/mirage/persistent_kernel/tasks/ampere/identity.cuh``
(``kernel::identity_task_impl``).

What the kernel actually does
-----------------------------
``identity_task_impl`` is a **bfloat16 elementwise copy** from
``input_ptr`` to ``output_ptr``. Every thread in the block walks a
``[OUTER_DIM_SIZE x OUTPUT_SIZE]`` tile and writes
``d_output[i] = d_input[i]``. It is not a view / not a no-op at the
memory level — a new buffer is produced.

Tensor contract
---------------
* ``input``  : ``DTensor`` with ``num_dims in {2, 3}``, ``dtype=bfloat16``,
  row-major layout (the C++ task_register asserts ``DmemRowMajor``).
* ``output`` : same ``num_dims``, **same shape**, same dtype as ``input``.
  ``input.dim(i) == output.dim(i)`` for every ``i`` — strictly enforced
  by ``identity_layer``.
* Parallelism: the kernel partitions the **last** dim across ``grid.x``;
  ``grid.x`` must divide the inner (last) dimension. Each thread block
  copies one ``[outer_dim, inner_dim // grid.x]`` slab. Grid.y / grid.z
  are unused. The kernel is blockIdx-agnostic at the runtime level —
  task dispatch is handled by the MPK scheduler.

Why this layer exists
---------------------
In production code (``qwen3``), identity is **not used at all**. It
shows up in DeepSeek V3's MLA prefill path as a "phantom bridge" to
legalize the MPK task graph:

  When a single task is simultaneously a *fork-producer* (multiple
  downstream consumers) **and** a *join-producer* (one of its consumers
  is itself a join-consumer with other producers), MPK's
  ``FullTaskDesc`` cannot fire two distinct ``trigger_event``s — the
  scheduler rejects this as ``annotated_graph.cc`` "case 3". Inserting
  an identity copy between the offending producer and the join-consumer
  turns the offending edge into ``producer -> identity -> join``: the
  identity has exactly one producer and one consumer, so it is neither
  fork nor join, and the original producer is no longer a join-producer.

  See ``python/mirage/mpk/models/deepseek_v3/builder.py`` (~line 1900)
  for the canonical comment block.

In short: Identity is a **graph-shape primitive**, not a numeric
primitive. The compute is real (it copies bytes), but the *purpose* is
ordering, not transformation.

``dependent_tensor`` kwarg
--------------------------
``pk.identity_layer`` accepts a ``dependent_tensor=`` keyword. As of
this writing the parameter is declared but **not consumed** by the
body (no ``new_input`` is registered for it, and no params are passed
to ``register_task``). It exists as a forward-looking hook for an
explicit dependency edge that does not feed into the kernel's actual
data. We expose it on this module's ``compile()`` as a passthrough so
that once the runtime side wires the dep edge in, callers picking up
this module do not need a signature change. If you need ordering
**today**, do not rely on this kwarg — instead use the phantom-bridge
pattern: insert ``Identity(...).compile(x, output=x_bridged)`` between
the producer and the join-consumer, and route the consumer to read
``x_bridged`` instead of ``x``. The data dependency through
``x_bridged`` is what enforces the ordering.

Examples
--------

>>> bridge = layers.Identity(prefix="layer3.kpe_bridge.")
>>> kpe_bridged_dt = bridge.compile(kpe_dt)             # auto grid_dim
>>> # ...consumer reads kpe_bridged_dt instead of kpe_dt
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import torch

from .._base import MPKModule


__all__ = ["Identity"]


class Identity(MPKModule):
    """Element-wise identity / memory copy.

    ``forward(x)`` returns ``x.clone()`` (matching the kernel's
    semantics, which materializes a fresh output buffer rather than
    aliasing the input).

    ``compile(x, *, dependent=None, output=None, grid_dim=None,
    block_dim=None)`` registers an ``identity`` task that copies ``x``
    into a freshly-allocated output DTensor (or into the
    caller-provided ``output``).

    Args (``__init__``):
        prefix: state_dict / tensor-name prefix; identity has no
                weights, so prefix is used only as a uniquifier for the
                output DTensor name (``f"{prefix}identity_out"``).
    """

    def __init__(self, *, prefix: str = "") -> None:
        super().__init__(prefix=prefix)

    # ------------------------------------------------------------------
    # PyTorch reference path
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a copy of ``x``.

        We clone (rather than return ``x`` directly) so that
        ``forward()`` matches the kernel: the kernel produces a new
        buffer, and any caller that mutates the output in place would
        observe different behavior between the reference and the
        compiled path if we aliased.
        """
        return x.clone()

    # ------------------------------------------------------------------
    # Grid-dim heuristic
    # ------------------------------------------------------------------
    def auto_grid_dim(self, x) -> Tuple[int, int, int]:
        """Pick ``grid.x`` so it divides the inner dim, capped at ``num_workers``.

        The kernel partitions the last dim across ``grid.x``; ``grid.x``
        must therefore divide ``inner_dim``. The copy is memory-bound;
        more blocks help up to the point where each block has enough
        work to amortize launch overhead. We pick the largest divisor of
        ``inner_dim`` that is <= ``num_workers``.
        """
        from ... import context as _ctx

        pk = _ctx.current_pk()
        # ``x`` may be a DTensor or a torch.Tensor depending on call
        # site (auto_grid_dim is sometimes called from compile()).
        if hasattr(x, "num_dims"):
            inner = x.dim(x.num_dims - 1)
        else:
            inner = int(x.shape[-1])
        cap = max(1, int(pk.num_workers))
        # Largest divisor of inner that is <= cap.
        gx = 1
        for d in range(1, min(inner, cap) + 1):
            if inner % d == 0:
                gx = d
        return (gx, 1, 1)

    # ------------------------------------------------------------------
    # MPK compile path
    # ------------------------------------------------------------------
    def compile(
        self,
        x,
        *,
        dependent: Optional[Any] = None,
        output: Optional[Any] = None,
        grid_dim: Optional[Tuple[int, int, int]] = None,
        block_dim: Optional[Tuple[int, int, int]] = None,
    ):
        """Register an ``identity`` task copying ``x`` to a new DTensor.

        Args:
            x:          The input ``DTensor``. Must be 2D or 3D, bf16,
                        row-major.
            dependent:  Optional ``DTensor`` whose producer should be
                        sequenced before this copy. Forwarded to
                        ``pk.identity_layer(dependent_tensor=...)``.
                        Note: the kwarg is currently a forward-looking
                        hook in ``persistent_kernel.py`` and is not yet
                        wired into the task graph; if you need true
                        ordering today, see the module docstring.
            output:     Optional pre-existing destination.
                        - ``None``: allocate a new ``DTensor`` via
                          ``pk.new_tensor`` matching ``x``'s shape.
                        - ``torch.Tensor``: attach to the kernel
                          (test-readback path) via ``pk.attach_input``.
                        - DTensor (anything else):                   use as-is.
            grid_dim:   Explicit grid override. If ``None``, falls back
                        to :meth:`auto_grid_dim`. Note: ``grid_dim[0]``
                        must divide ``x``'s inner dim.
            block_dim:  Explicit block override. If ``None``, falls back
                        to :meth:`default_block_dim` (128 on Ampere,
                        256 on Hopper/Blackwell).

        Returns:
            The output ``DTensor``.
        """
        from ... import context as _ctx

        pk = _ctx.current_pk()

        if grid_dim is None:
            grid_dim = self.auto_grid_dim(x)
        if block_dim is None:
            block_dim = self.default_block_dim()

        # Resolve the output DTensor.
        if output is None:
            # Allocate via pk.new_tensor.
            shape = tuple(x.dim(i) for i in range(x.num_dims))
            out_dt = pk.new_tensor(
                dims=shape,
                dtype=x.dtype,
                name=f"{self.prefix}identity_out",
            )
        elif isinstance(output, torch.Tensor):
            # Test-readback path: bind a host torch tensor as a kernel
            # tensor so the test driver can inspect the result.
            out_dt = pk.attach_input(output, name=f"{self.prefix}identity_out")
        else:
            # Assume DTensor (or DTensor-like; we don't import the type
            # here to avoid coupling).
            out_dt = output

        # Inlined task registration (the body that used to live on
        # ``PersistentKernel.identity_layer``). Each catalog module owns
        # its own task wiring so adding a new layer doesn't require
        # editing ``persistent_kernel.py``.
        #
        # The ``dependent`` kwarg is accepted for API parity with the old
        # pk method but is not consumed by the task body (the original
        # method also did not wire it into the graph — see persistent_
        # kernel.py:identity_layer). It is reserved for a future
        # explicit dependency edge.
        from ....core import CyTBGraph
        from ....kernel import TBGraph

        assert x.num_dims == out_dt.num_dims
        last_dim = 0
        for i in range(x.num_dims):
            assert x.dim(i) == out_dt.dim(i)
            last_dim = i
        assert last_dim == 1 or last_dim == 2
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(x, (last_dim, -1, -1), 1, True)
        tb_graph.new_input(out_dt, (last_dim, -1, -1), 1, True)
        pk.kn_graph.customized([x, out_dt], tb_graph)
        pk.kn_graph.register_task(tb_graph, "identity")
        return out_dt
