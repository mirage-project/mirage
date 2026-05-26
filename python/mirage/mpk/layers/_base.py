"""Base class for the MPK layer catalog.

Every catalog module inherits from :class:`MPKModule` (which is itself a
``torch.nn.Module``). The class defines the contract Phase-2 subagents
implement, the model-author audience reads, and Phase-3 composite
modules orchestrate.

The contract has three parts:

1. ``forward(self, ...)`` — a faithful PyTorch reference. The module
   owns its weights as ``nn.Parameter`` so ``state_dict()`` and
   ``load_state_dict()`` work the standard way. ``forward()`` runs in
   eager PyTorch with no MPK installed and no compile scope; it is the
   correctness oracle for the compiled path.

2. ``compile(self, ..., *, grid_dim=None, block_dim=None)`` — registers
   one or more MPK tasks into the active :class:`PersistentKernel` (read
   via :func:`mirage.mpk.context.current_pk` inside the body). Returns
   the output ``DTensor`` (or tuple of DTensors, mirroring the
   ``forward()`` signature). ``grid_dim`` and ``block_dim`` are
   keyword-only overrides; when omitted the leaf falls back to
   :meth:`auto_grid_dim` and :meth:`default_block_dim`.

3. ``auto_grid_dim(self, input_dt)`` — per-layer heuristic that returns
   a ``(x, y, z)`` tuple with product at most
   ``current_pk().num_workers``, respecting any kernel-specific
   alignment constraints (e.g., MMA-M=128 for MoE GEMMs). Each leaf
   overrides this; the base class only declares the signature.

``prefix`` is the vLLM/transformers convention for state_dict key
resolution. Setting ``prefix="model.layers.3.self_attn."`` makes the
module's weight load from ``state_dict["model.layers.3.self_attn.q_proj.weight"]``
without any custom loader plumbing.
"""

from typing import Tuple

import torch
import torch.nn as nn


GridDim = Tuple[int, int, int]
BlockDim = Tuple[int, int, int]


class MPKModule(nn.Module):
    """Base class for every layer in ``mirage.mpk.layers``.

    Subclasses MUST override:
        * ``forward`` — the PyTorch reference.
        * ``compile`` — the MPK task-registration path.
        * ``auto_grid_dim`` — per-layer grid heuristic.
    """

    def __init__(self, *, prefix: str = "") -> None:
        super().__init__()
        # ``prefix`` is a string path used both as the HF state_dict-key
        # prefix and as a unique name component for tensors attached to
        # MPK (see ``pk.attach_input(weight, name=f"{prefix}weight")``).
        # The empty-string default lets unit tests instantiate modules
        # standalone without picking a prefix.
        self.prefix = prefix

    def compile(self, *args, **kwargs):
        raise NotImplementedError(
            f"{type(self).__name__}.compile() is not implemented. "
            "Every MPKModule subclass must implement compile()."
        )

    def auto_grid_dim(self, *args, **kwargs) -> GridDim:
        raise NotImplementedError(
            f"{type(self).__name__}.auto_grid_dim() is not implemented. "
            "Either implement it on the subclass or pass grid_dim "
            "explicitly to compile()."
        )

    def default_block_dim(self) -> BlockDim:
        """Architecture-conditioned default block dim.

        128 threads per worker on Ampere (CC<90), 256 on Hopper/Blackwell —
        matches the values in ``include/mirage/persistent_kernel/`` headers.
        Subclasses with kernel-specific requirements (e.g., one warp per
        lane) override this.
        """
        from .. import context as _ctx
        pk = _ctx.current_pk()
        return (128, 1, 1) if pk.target_cc < 90 else (256, 1, 1)
