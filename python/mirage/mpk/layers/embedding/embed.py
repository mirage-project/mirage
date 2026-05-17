"""Token embedding layer (``layers.Embed``).

Catalog wrapper around :meth:`PersistentKernel.embed_layer`. One MPK task
per ``compile()`` call; the underlying ``.cuh`` kernel loops over the
whole batch in a single threadblock (grid is ``(1, 1, 1)``), so this
layer is bandwidth-bound on the embedding-table read.

Tensor contract
---------------
- ``self.weight`` — ``nn.Parameter`` of shape
  ``(num_embeddings, embedding_dim)``, dtype ``bfloat16`` (the kernel
  hard-codes ``T = bfloat16`` in
  ``include/mirage/persistent_kernel/tasks/{ampere,hopper}/embedding*.cuh``).
- ``input`` (``forward``) / ``input_dt`` (``compile``) — token IDs.

  * If ``input_source == 1`` (the qwen3 default) the kernel reads
    ``task_desc->input_ptrs[0]``, i.e. the DTensor passed as ``input``.
    Shape ``(max_num_batched_tokens,)`` (1D), dtype ``int64`` (the
    kernel reinterprets the buffer as ``int64_t *``). Typically this
    DTensor wraps ``pk.meta_tensors["input_tokens"]``.
  * If ``input_source == 0`` the kernel ignores ``input_ptrs[0]`` and
    reads a single token from ``runtime_config.tokens +
    runtime_config.step[0]`` — i.e. the next token from the
    persistent-runtime token buffer. ``input`` is still required to
    register the task graph edge, but its data is unused by the kernel.
- ``output`` — shape ``(batch_size, embedding_dim)`` (2D), dtype
  ``bfloat16``. ``batch_size`` here is the kernel-template constant
  ``BATCH_SIZE`` baked at register time from ``output.dim[0]``, i.e.
  the static ``max_num_batched_tokens``. ``OUTPUT_DIM_SIZE`` is
  ``embedding_dim`` and ``output_stride`` is the row stride of the
  output DTensor.

Parallelism
-----------
The kernel uses a single CTA (grid ``(1, 1, 1)``) and parallelises the
``BATCH_SIZE * OUTPUT_DIM_SIZE`` element copy across ``blockDim.x``
threads (and on Hopper, restricts itself to ``CONSUMER_NUM_THREADS``
within a 256-thread block). There is no productive way to grow the
grid for this op — :meth:`auto_grid_dim` therefore always returns
``(1, 1, 1)`` to match the demo wiring in
``demo/qwen3/demo.py`` and ``python/mirage/mpk/models/qwen3/builder.py``.

``input_source`` semantics
--------------------------
Mirror of the kernel-side switch in
``src/kernel/task_register.cc:register_embedding_task``:

- ``input_source = 0`` ("all_tokens"): kernel reads
  ``runtime_config.tokens + runtime_config.step[0]``. Use when the
  embedding consumes the persistent runtime's rolling token buffer at
  the current decoding step — i.e. classic single-token decode.
- ``input_source = 1`` ("input_token"): kernel reads
  ``task_desc->input_ptrs[0]``, i.e. the DTensor you pass as ``input``.
  Use when the embedding consumes an explicit token tensor (the qwen3
  builder path, ``mpk.meta_tensors["input_tokens"]``, and the
  recommended default for test-mode runs).
"""

from typing import TYPE_CHECKING, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .._base import BlockDim, GridDim, MPKModule

if TYPE_CHECKING:
    # DTensor lives in the compiled Cython core; importing it eagerly
    # would force the .so to load even when this module is imported in
    # a pure-PyTorch context (no MPK installed). Only the type hint
    # needs the symbol, so guard it.
    from ....core import DTensor


class Embed(MPKModule):
    """Embedding lookup ``y[i] = weight[input[i]]``.

    Args:
        num_embeddings: Vocab size (rows of the embedding table).
        embedding_dim: Hidden size (columns of the embedding table).
        prefix: vLLM/HF state_dict prefix. Combined with the trailing
            ``"weight"`` key gives the unique tensor name attached to
            the MPK graph (e.g. ``prefix="model.embed_tokens."`` yields
            ``model.embed_tokens.weight`` — matching HF's Qwen3 layout).
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        prefix: str = "",
    ) -> None:
        super().__init__(prefix=prefix)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        # Weight stays a stock nn.Parameter so state_dict()/load_state_dict()
        # / state_dict-by-prefix work the standard way. MPK reads the raw
        # CUDA pointer via pk.attach_input(...) inside compile().
        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim)
        )

    # ------------------------------------------------------------------
    # PyTorch reference path
    # ------------------------------------------------------------------
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Reference: stock ``torch.nn.functional.embedding``."""
        return F.embedding(input, self.weight)

    # ------------------------------------------------------------------
    # Grid heuristic
    # ------------------------------------------------------------------
    def auto_grid_dim(self, input_dt: "DTensor") -> GridDim:
        """Embedding is a single-CTA op.

        The ``.cuh`` loops over the whole ``BATCH_SIZE * OUTPUT_DIM_SIZE``
        copy inside one threadblock — there is no per-CTA tiling on the
        output. ``grid_dim=(1, 1, 1)`` matches both the demo and the
        registered task variant. The caller may still override via the
        ``grid_dim`` kwarg on ``compile()``.
        """
        return (1, 1, 1)

    # ------------------------------------------------------------------
    # MPK compile path
    # ------------------------------------------------------------------
    def compile(
        self,
        input_dt: "DTensor",
        *,
        input_source: int = 0,
        output: Optional[Any] = None,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ) -> "DTensor":
        """Register the embedding task on the active PersistentKernel.

        Args:
            input_dt: "DTensor" for the token IDs. Even when
                ``input_source == 0`` (kernel reads
                ``runtime_config.tokens``), an ``input_dt`` is required
                to wire the task graph — its data is then unused.
            input_source: ``0`` to source tokens from the persistent
                runtime's ``runtime_config.tokens`` buffer at the
                current step; ``1`` to source tokens from
                ``input_dt`` (i.e. ``task_desc->input_ptrs[0]``).
                See module docstring for full semantics. Default ``0``
                matches the ``pk.embed_layer`` Python signature.
            output: Where to put the output DTensor:

                * ``None`` (default): allocate a fresh DTensor via
                  ``pk.new_tensor`` of shape
                  ``(input_dt.dim(0), embedding_dim)``.
                * ``torch.Tensor``: attach via ``pk.attach_input`` so
                  the test path can read the result. Useful in
                  test-mode tests.
                * ``DTensor``: reuse the caller's existing DTensor
                  (composite ``compile()`` paths where the buffer is
                  pre-allocated upstream).
            grid_dim: Override for the grid. Defaults to
                :meth:`auto_grid_dim` (which is always ``(1, 1, 1)``).
            block_dim: Override for the block. Defaults to
                :meth:`default_block_dim`.

        Returns:
            The output DTensor (shape ``(batch_size, embedding_dim)``,
            ``bfloat16``).
        """
        from ... import context as _ctx

        pk = _ctx.current_pk()
        grid_dim = grid_dim if grid_dim is not None else self.auto_grid_dim(input_dt)
        block_dim = block_dim if block_dim is not None else self.default_block_dim()

        # Attach the embedding table to the graph. ``self.weight`` is an
        # ``nn.Parameter`` (a torch.Tensor subclass) — ``attach_input``
        # accepts it and we hold a strong ref via ``self.weight`` so it
        # is never GC'd out from under the kernel.
        weight_dt = pk.attach_input(
            torch_tensor=self.weight,
            name=f"{self.prefix}weight",
        )

        # Resolve output DTensor per the three-way contract above.
        if output is None:
            batch_size = input_dt.dim(0)
            out_dt = pk.new_tensor(
                dims=(batch_size, self.embedding_dim),
                dtype=weight_dt.dtype,
                name=f"{self.prefix}out",
                io_category="cuda_tensor",
            )
        elif isinstance(output, torch.Tensor):
            out_dt = pk.attach_input(
                torch_tensor=output,
                name=f"{self.prefix}out",
            )
        else:
            # Assume it's already a DTensor — leave it alone.
            out_dt = output

        # Inlined task registration (the body that used to live on
        # ``PersistentKernel.embed_layer``). Each catalog module owns its
        # own task wiring so adding a new layer doesn't require editing
        # ``persistent_kernel.py``.
        from ....core import CyTBGraph
        from ....kernel import TBGraph

        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input_dt, (-1, 1, -1), -1, True)
        tb_graph.new_input(weight_dt, (1, -1, -1), -1, True)
        tb_graph.new_input(out_dt, (1, 0, -1), -1, True)
        pk.kn_graph.customized([input_dt, weight_dt, out_dt], tb_graph)
        # The legacy pk method used a ternary that picked
        # "embedding" on both branches; collapsed here for clarity.
        pk.kn_graph.register_task(tb_graph, "embedding", [input_source])
        return out_dt
