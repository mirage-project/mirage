"""Root-Mean-Square LayerNorm (RMSNorm) catalog module.

This is the catalog counterpart to :meth:`PersistentKernel.rmsnorm_layer`. The
module owns a single learnable scale parameter ``weight`` of shape
``(hidden_size,)`` and implements the standard "T5/LLaMA-style" RMSNorm::

    variance = x.to(float32).pow(2).mean(-1, keepdim=True)
    x_normed = x * rsqrt(variance + eps)
    return (x_normed * weight).to(x.dtype)

Tensor contract
---------------

* Input  ``x``  : 2-D ``bfloat16`` device tensor with shape
  ``(batch_size, hidden_size)`` (row-major). Each row is one token.
* Weight        : 1-D ``bfloat16`` device tensor with shape
  ``(hidden_size,)``. Stored as an ``nn.Parameter`` so ``state_dict()``
  works the standard way.
* Output        : same shape and dtype as ``x``. Either auto-allocated
  (``pk.new_tensor``) or supplied by the caller (``torch.Tensor`` for
  test-mode readback, or an existing ``DTensor``).

Accumulation precision
----------------------

The reduction is done in ``float32`` inside the .cuh kernel (see
``include/mirage/persistent_kernel/tasks/ampere/rmsnorm.cuh`` lines
107-138 and the matching ``hopper/rmsnorm_hopper.cuh``). Outputs are
cast back to the input dtype, matching the PyTorch ``forward()``.

Epsilon caveat (READ THIS)
--------------------------

The ``eps`` constructor argument is currently a **PyTorch-only**
parameter. The MPK kernel is code-generated with a hard-coded
``1e-6f`` epsilon in ``src/kernel/task_register.cc``
(``register_rmsnorm_task`` line 182, ``register_rmsnorm_hopper_task``
line 1252). If you instantiate ``RMSNorm(eps=1e-5)`` and call
``compile()``, the compiled kernel will still use ``1e-6``. We store
``self.eps`` so the PyTorch reference matches whatever HF config the
user loaded (e.g. Qwen3 ``rms_norm_eps=1e-6``, which happens to agree),
but be aware that mismatched eps between ``forward()`` and the compiled
path is a silent correctness bug today. A follow-up PR can plumb
``eps`` through ``register_task(... [process_dim, in_off, out_off,
eps_as_bits])`` once the .cuh changes to a runtime param.

Sliced-norm offsets (DeepSeek QKV-a, etc.)
------------------------------------------

The kernel supports operating on a column slice of a wider input/output
buffer via ``IN_OFFSET`` / ``OUT_OFFSET`` template params:

* ``process_dim``     — number of contiguous columns to normalize. Defaults
  to ``hidden_size`` (i.e. the full row).
* ``in_offset_elems`` — starting column of the read slice in the wider
  input row.
* ``out_offset_elems`` — starting column of the write slice in the wider
  output row.

This is exercised by DeepSeek V3's ``qkv_a_out`` (where the fused
``q_a_layernorm`` covers ``[0:1536)`` and ``kv_a_layernorm`` covers
``[1536:2048)`` of a 2176-wide row, sharing the same backing buffer).
Qwen3 does not use this; the defaults preserve legacy contiguous
behaviour. See the FuseTensor comment in the .cuh files for the
runtime row-pointer pre-offset detail.

Parallelism axis
----------------

The kernel processes **one token per task** (the .cu register code
asserts ``batch_size == 1`` per TBGraph). The natural grid is therefore
``(batch_size, 1, 1)`` — one task per row — capped at the worker pool.
``rmsnorm_layer`` builds the TBGraph with
``new_input(input, (0, -1, -1), 1, True)``, partitioning on the row axis.

Architecture variant selection
------------------------------

``pk.rmsnorm_layer`` switches between two backends at compile time
based on ``pk.target_cc``:

* ``target_cc < 90``  (Ampere) -> task name ``"rmsnorm"``           -> ``ampere/rmsnorm.cuh``
* ``target_cc >= 90`` (Hopper / Blackwell) -> task name ``"rmsnorm_hopper"`` -> ``hopper/rmsnorm_hopper.cuh``

Both kernels are bfloat16-only (the code-gen hard-wires
``kernel::rms_norm{,_hopper}_impl<bfloat16, ...>``).

Hidden-size alignment
---------------------

From the .cuh: ``HIDDEN_DIM % NUM_THREADS == 0`` and
``HIDDEN_DIM * sizeof(T) / NUM_THREADS`` must be a multiple of 4
(``BYTES_PER_THREAD % 4 == 0``) so the kernel can issue at least a
32-bit cp.async. For ``bfloat16`` (2 bytes) this means
``HIDDEN_DIM >= 2 * NUM_THREADS`` and ``HIDDEN_DIM`` divisible by
``NUM_THREADS``. With ``NUM_THREADS=256`` (Hopper/Blackwell) the
smallest legal ``hidden_size`` is 512; with ``NUM_THREADS=128``
(Ampere), 256. See the ``HIDDEN_DIM * sizeof(dtype) / NUM_THREADS >= 4``
comment in ``tests/runtime_python/test_mode/test_rmsnorm_testmode.py``.
"""
from __future__ import annotations

from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn

from .._base import MPKModule
from ...context import current_pk

# DTensor is the public Cython class used everywhere in the codebase.
from ....core import DTensor


GridDim = Tuple[int, int, int]
BlockDim = Tuple[int, int, int]


class RMSNorm(MPKModule):
    """Token-wise RMSNorm with a learnable per-channel scale.

    Args:
        hidden_size: Number of features per token (the inner dim that is
            normalized over). Must be divisible by the worker block-dim
            (128 on Ampere, 256 on Hopper/Blackwell) and large enough to
            satisfy ``hidden_size * sizeof(bfloat16) / NUM_THREADS >= 4``.
        eps: Variance epsilon used by the PyTorch reference. **The
            compiled MPK path ignores this value and uses ``1e-6``
            hard-coded in ``src/kernel/task_register.cc``** — see module
            docstring. Stored on the instance for documentation / future
            plumbing only.
        prefix: HF state_dict key prefix (vLLM convention). Setting
            ``prefix="model.layers.3.input_layernorm."`` makes the
            weight load from
            ``state_dict["model.layers.3.input_layernorm.weight"]``.

    Attributes:
        weight (``nn.Parameter``): shape ``(hidden_size,)``, dtype set
            by the model's ``.to(...)`` (typically ``bfloat16``).
        eps (``float``): epsilon, PyTorch-side only (see caveat above).
        hidden_size (``int``): cached for the auto-grid heuristic.
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        *,
        prefix: str = "",
    ) -> None:
        super().__init__(prefix=prefix)
        self.hidden_size = hidden_size
        self.eps = eps
        # Standard initialization: ones, matching Qwen3RMSNorm / LlamaRMSNorm.
        # Real weights overwrite this via load_state_dict.
        self.weight = nn.Parameter(torch.ones(hidden_size))

    # ------------------------------------------------------------------
    # PyTorch reference
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Faithful RMSNorm in PyTorch.

        Matches the standard transformers / HF implementation:
        accumulate the squared-mean in float32, divide by rsqrt, then
        cast back to the input dtype before applying the scale. This is
        exactly the reference used by ``test_rmsnorm_testmode.py``.
        """
        input_dtype = x.dtype
        variance = x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
        x_normed = x.to(torch.float32) * torch.rsqrt(variance + self.eps)
        # Cast to weight dtype before the scale, then back to the input
        # dtype. (Equivalently: ``(x_normed * weight).to(input_dtype)``.)
        return (x_normed.to(input_dtype) * self.weight).to(input_dtype)

    # ------------------------------------------------------------------
    # MPK grid heuristic
    # ------------------------------------------------------------------
    def auto_grid_dim(self, x_dt: DTensor) -> GridDim:
        """Default grid: one task per token, capped at the worker pool.

        ``rmsnorm_layer`` partitions on dim 0 (the batch / token axis)
        and asserts one token per task in code-gen. The natural choice
        is therefore ``(batch_size, 1, 1)``; we cap at
        ``current_pk().num_workers`` so we never overcommit the queue.

        Existing callers (``demo/qwen3/demo.py`` line 514) pick
        ``(pk.max_num_batched_tokens, 1, 1)``, which is equivalent up
        to the cap when batch_size matches the runtime max.
        """
        pk = current_pk()
        batch_size = x_dt.dim(0)
        return (max(1, min(batch_size, pk.num_workers)), 1, 1)

    # ------------------------------------------------------------------
    # MPK task registration
    # ------------------------------------------------------------------
    def compile(
        self,
        x: DTensor,
        *,
        process_dim: Optional[int] = None,
        in_offset_elems: int = 0,
        out_offset_elems: int = 0,
        output: Optional[Union[torch.Tensor, DTensor]] = None,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
        name: Optional[str] = None,
    ) -> DTensor:
        """Register one ``rmsnorm`` task for the current PK.

        Args:
            x: 2-D bfloat16 DTensor, ``(batch_size, hidden_size)``.
            process_dim: Number of columns to normalize. ``None``
                (default) means the full ``hidden_size`` of the output
                row — i.e. legacy contiguous RMSNorm. Set to a smaller
                value together with ``in_offset_elems`` /
                ``out_offset_elems`` to operate on a column slice of a
                wider buffer (DeepSeek-style fused QKV-a).
            in_offset_elems: Starting column in the input row. Forwarded
                to the kernel's ``IN_OFFSET`` template parameter.
            out_offset_elems: Starting column in the output row.
                Forwarded to ``OUT_OFFSET``.
            output: Output buffer routing, identical to ``add()``:

                * ``None`` (default, production) — allocate a fresh
                  DTensor via ``pk.new_tensor`` with the same shape and
                  dtype as ``x``.
                * ``torch.Tensor`` — attach via ``pk.attach_input`` so
                  the test driver can read back from it after ``pk()``
                  returns (canonical test path, see
                  ``test_rmsnorm_testmode.py``).
                * ``DTensor`` — use directly (advanced; caller owns
                  registration, e.g. an in-place write into a wider
                  fused buffer for the offset variant).
            grid_dim: Explicit override; ``None`` -> ``auto_grid_dim``.
            block_dim: Explicit override; ``None`` -> arch default
                (128 on Ampere, 256 on Hopper/Blackwell).
            name: Optional name for the auto-allocated output buffer.
                Only used when ``output is None``. Must be unique within
                the PK tensor registry.

        Returns:
            The output DTensor.

        Raises:
            RuntimeError: when called outside ``pk.compile_scope()``
                (raised by :func:`current_pk`).
            ValueError: when ``x.num_dims != 2`` or ``output`` has an
                unsupported type.
        """
        pk = current_pk()

        if x.num_dims != 2:
            raise ValueError(
                f"RMSNorm.compile expects a 2-D input DTensor; "
                f"got num_dims={x.num_dims}"
            )

        # Resolve output DTensor.
        prefix = self.prefix or "rmsnorm"
        if output is None:
            out_name = name if name is not None else f"{prefix}out"
            out_dt = pk.new_tensor(
                dims=(x.dim(0), x.dim(1)),
                dtype=x.dtype,
                name=out_name,
            )
        elif isinstance(output, torch.Tensor):
            out_name = name if name is not None else f"{prefix}out"
            out_dt = pk.attach_input(output, name=out_name)
        elif isinstance(output, DTensor):
            out_dt = output
        else:
            raise TypeError(
                "RMSNorm.compile output must be None, a torch.Tensor, "
                f"or a DTensor; got {type(output).__name__}"
            )

        # Attach the learnable scale. nn.Parameter is a torch.Tensor
        # subclass, so attach_input accepts it; we keep a strong ref via
        # ``self.weight`` so the buffer is not GC'd while PK holds the
        # raw pointer.
        w_dt = pk.attach_input(self.weight.data, name=f"{prefix}weight")

        # Resolve grid / block.
        if grid_dim is None:
            grid_dim = self.auto_grid_dim(x)
        if block_dim is None:
            block_dim = self.default_block_dim()

        # Inlined task registration (the body that used to live on
        # ``PersistentKernel.rmsnorm_layer``). Each catalog module owns
        # its own task wiring so adding a new layer doesn't require
        # editing ``persistent_kernel.py``.
        from ....core import CyTBGraph
        from ....kernel import TBGraph

        assert x.num_dims == 2
        assert out_dt.num_dims == 2
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(x, (0, -1, -1), 1, True)
        tb_graph.new_input(w_dt, (-1, -1, -1), 0, True)
        tb_graph.new_input(out_dt, (0, -1, -1), 1, True)
        pk.kn_graph.customized([x, w_dt, out_dt], tb_graph)

        task_name = "rmsnorm_hopper" if pk.target_cc >= 90 else "rmsnorm"
        if (process_dim is None and in_offset_elems == 0
                and out_offset_elems == 0):
            pk.kn_graph.register_task(tb_graph, task_name)
        else:
            if process_dim is None:
                process_dim = out_dt.dim(1)
            pk.kn_graph.register_task(
                tb_graph,
                task_name,
                [process_dim, in_offset_elems, out_offset_elems],
            )
        return out_dt
