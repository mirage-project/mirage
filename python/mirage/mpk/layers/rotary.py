"""Rotary position embedding tables for RoPE (LLaMA / Qwen3 family).

This module owns the precomputed ``cos`` and ``sin`` tables that the
MPK attention task consumes as ``cos_pos_embed`` / ``sin_pos_embed``
(see :meth:`PersistentKernel.attention_layer`, around line 749 in
``python/mirage/mpk/persistent_kernel.py``). It exists for three
reasons:

1. **State ownership.** The tables are non-trainable persistent state
   tied to the model's architecture (``head_dim``,
   ``max_position_embeddings``, ``base``). They are not learned, so
   they must NOT be ``nn.Parameter`` — that would put them in
   ``state_dict()`` and break HF safetensor loading (Qwen3 weights do
   not contain rotary-table entries). They are also not transient, so
   they should not be recomputed on every call. ``nn.Buffer`` (i.e.
   ``register_buffer`` with ``persistent=False``) is the
   PyTorch-standard slot for exactly this case.

2. **PyTorch reference.** ``forward(positions)`` returns
   ``(cos[positions], sin[positions])`` exactly the way
   ``Qwen3RotaryEmbedding.forward`` in
   ``demo/qwen3/models/modeling_qwen3.py`` does — so the new catalog
   model's ``forward()`` path can call it as a drop-in replacement.

3. **DTensor exposure.** ``compile()`` registers the buffers with the
   active :class:`PersistentKernel` via ``pk.attach_input(...)`` and
   returns the two DTensors. There is **no MPK task backing
   RotaryEmbedding** — the rotation is performed inside the attention
   kernel itself; this module's sole compile-time job is to hand the
   precomputed tables to that kernel.

Tensor contract
---------------

``cos`` and ``sin`` are stored as ``(max_position_embeddings, head_dim)``
2-D ``bfloat16`` tensors. The last-dim layout is the
``torch.cat((freqs, freqs), dim=-1)`` convention used by the HF
``LlamaRotaryEmbedding`` / ``Qwen3RotaryEmbedding`` implementations:
the first half of ``head_dim`` and the second half hold the same
``cos(theta_i)`` / ``sin(theta_i)`` values, so the kernel's
``rotate_half`` step is purely on the data axis. ``attention_layer``
asserts ``cos.num_dims == 2`` and ``cos.dim(1) == head_dim`` (see
``persistent_kernel.py:781-784``); both are honoured here.

Wiring
------

The module is typically owned by ``Attention`` (or by the top-level
``Qwen3Model``) and its ``compile()`` is called once during the
parent's ``compile()`` to obtain DTensors for the cos/sin tables.
Those DTensors are then passed straight into
``pk.attention_layer(cos_pos_embed=..., sin_pos_embed=...)``.
Positions are not passed to the kernel — the attention task indexes
the table internally using its per-task ``request_id`` /
meta-tensor step.

Example
-------

.. code-block:: python

    class Attention(MPKModule):
        def __init__(self, config, *, prefix=""):
            super().__init__(prefix=prefix)
            self.rotary = layers.RotaryEmbedding(
                head_dim=config.head_dim,
                max_position_embeddings=config.max_position_embeddings,
                base=config.rope_theta,
                prefix=f"{prefix}rotary_emb.",
            )
            # ... q_proj / k_proj / v_proj / o_proj here ...

        def compile(self, x):
            cos_dt, sin_dt = self.rotary.compile()
            # ... build qkv, then ...
            pk = current_pk()
            pk.attention_layer(
                input=...,
                cos_pos_embed=cos_dt,
                sin_pos_embed=sin_dt,
                ...,
            )

Design rationale: buffer, not parameter
---------------------------------------

``nn.Parameter`` would surface ``rotary.cos`` and ``rotary.sin`` in
``state_dict()``, causing ``model.load_state_dict(safetensors)`` to
emit "unexpected key" errors against HF checkpoints (which only ship
weight matrices, never RoPE tables). It would also make the tables
look trainable to optimizers. ``register_buffer(..., persistent=...)``
solves both: the tables move with ``.to(device, dtype)`` and survive
through ``.cuda()``, but they stay out of the optimizer and out of
the state-dict (when ``persistent=False``).
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch

from ._base import MPKModule  # rotary.py sits directly in layers/
# The DTensor symbol comes from the Cython core (one extra level up
# than the subpackage layers, which is why this is 3 dots not 4).
from ...core import DTensor


class RotaryEmbedding(MPKModule):
    """Precomputed cos/sin tables for RoPE.

    Args:
        head_dim: Per-head channel dimension. Must match the
            ``head_dim`` of the attention layer that will consume the
            tables (``persistent_kernel.attention_layer`` asserts
            ``cos.dim(1) == head_dim``). Must be even — the HF
            ``rotate_half`` convention splits the channels into two
            equal halves.
        max_position_embeddings: Number of distinct positions to
            precompute. The HF Qwen3 demo uses 4096 (see
            ``demo/qwen3/demo.py:349`` where the table is sliced to
            ``[:4096, :]``). Pick the max sequence length you expect
            the model to see at inference time.
        base: RoPE frequency base. Qwen3-8B uses
            ``config.rope_theta == 1_000_000.0`` per its config; LLaMA-2
            uses ``10000.0``. Default here is ``10000.0`` to match the
            HF default; the caller should pass ``config.rope_theta``
            explicitly for Qwen3.
        prefix: HF/vLLM-style state_dict key prefix. Empty by default.
            Because ``cos`` / ``sin`` are non-persistent buffers, this
            prefix is currently only used to name the DTensors attached
            to MPK (so two RotaryEmbedding instances in one PK don't
            collide). It is kept for API consistency with the rest of
            the catalog.

    Attributes:
        cos (``torch.Tensor`` / non-persistent buffer): shape
            ``(max_position_embeddings, head_dim)``, ``bfloat16``.
        sin (``torch.Tensor`` / non-persistent buffer): shape
            ``(max_position_embeddings, head_dim)``, ``bfloat16``.
        head_dim (``int``)
        max_position_embeddings (``int``)
        base (``float``)
    """

    def __init__(
        self,
        head_dim: int,
        max_position_embeddings: int,
        base: float = 10000.0,
        *,
        prefix: str = "",
    ) -> None:
        super().__init__(prefix=prefix)
        if head_dim % 2 != 0:
            raise ValueError(
                f"RotaryEmbedding requires an even head_dim (the "
                f"rotate_half convention splits the channel axis into "
                f"two halves); got head_dim={head_dim}."
            )
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = float(base)

        cos, sin = self._precompute_freqs(
            head_dim=head_dim,
            max_pos=max_position_embeddings,
            base=self.base,
        )
        # ``persistent=False`` keeps the tables out of ``state_dict()``
        # so they do not collide with HF safetensor keys (which never
        # contain RoPE tables). The buffers still move with .to(device,
        # dtype) and survive .cuda(), which is all we need.
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    # ------------------------------------------------------------------
    # Precomputation
    # ------------------------------------------------------------------
    @staticmethod
    def _precompute_freqs(
        head_dim: int,
        max_pos: int,
        base: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Standard HF / LLaMA RoPE precomputation in float32 → bfloat16.

        Mirrors ``Qwen3RotaryEmbedding.forward`` in
        ``demo/qwen3/models/modeling_qwen3.py`` (lines ~60-117) with
        the "default" rope_type and ``attention_scaling == 1.0``
        (Qwen3-8B has no ``rope_scaling``, so the default path is
        exact).

        Returns:
            cos, sin — each ``(max_pos, head_dim)``, dtype ``bfloat16``,
            ready to be moved to CUDA via ``.to(device)`` by the caller
            (the buffers themselves stay on CPU at construction time
            and migrate with ``module.to(device)``).
        """
        # inv_freq[i] = 1 / base ** (2i / head_dim), i in [0, head_dim/2)
        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, head_dim, 2, dtype=torch.float32)
                / head_dim
            )
        )
        # positions: (max_pos,) float32
        positions = torch.arange(max_pos, dtype=torch.float32)
        # freqs: (max_pos, head_dim / 2) — outer product positions ⊗ inv_freq
        freqs = torch.einsum("p,d->pd", positions, inv_freq)
        # emb: (max_pos, head_dim) — HF rotate_half convention: the two
        # halves repeat the same theta sequence so that
        # ``cos.unsqueeze(...) * x + sin.unsqueeze(...) * rotate_half(x)``
        # yields the rotated pair (x_2i, x_{2i+head_dim/2}).
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(torch.bfloat16)
        sin = emb.sin().to(torch.bfloat16)
        return cos, sin

    # ------------------------------------------------------------------
    # PyTorch reference path
    # ------------------------------------------------------------------
    def forward(
        self, positions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Look up cos/sin for ``positions``.

        Args:
            positions: integer tensor of arbitrary shape. Each entry
                must satisfy ``0 <= p < max_position_embeddings``.

        Returns:
            ``(cos, sin)``, each with shape ``positions.shape +
            (head_dim,)`` and dtype ``bfloat16``. Lives on the same
            device as the buffers.

        Note:
            This matches the table-lookup style used by the HF
            ``apply_rotary_pos_emb`` reference (where ``cos`` and
            ``sin`` are pre-sliced for the positions of interest before
            being broadcast against q/k). The MPK attention kernel
            performs an equivalent index internally; ``forward`` here
            is purely for the PyTorch-side correctness oracle.
        """
        if not torch.is_tensor(positions):
            raise TypeError(
                "RotaryEmbedding.forward expects a torch.Tensor of "
                f"position indices; got {type(positions).__name__}."
            )
        # Long-indexing into the buffers preserves the buffer dtype
        # (bfloat16) and adds a trailing head_dim dimension.
        return self.cos[positions], self.sin[positions]

    # ------------------------------------------------------------------
    # Auto grid — not applicable
    # ------------------------------------------------------------------
    def auto_grid_dim(self, *args, **kwargs):
        """RotaryEmbedding emits no MPK task — no grid to autotune.

        The actual RoPE math runs *inside* the attention kernel; this
        module only owns the precomputed tables. Callers should never
        ask this module for a grid_dim. Raising rather than returning
        a no-op makes the contract explicit.
        """
        raise NotImplementedError(
            "RotaryEmbedding does not emit an MPK task and therefore "
            "has no grid_dim. The RoPE rotation is performed inside "
            "the attention kernel that consumes the (cos, sin) "
            "DTensors returned by RotaryEmbedding.compile()."
        )

    # ------------------------------------------------------------------
    # MPK compile path
    # ------------------------------------------------------------------
    def compile(self) -> Tuple[DTensor, DTensor]:
        """Attach cos/sin buffers to the active PK and return DTensors.

        Called once by the owning module (``Attention``, or
        ``Qwen3Model`` if the tables are shared across layers) during
        ``model.compile()``. The returned DTensors should be threaded
        straight into ``pk.attention_layer(cos_pos_embed=...,
        sin_pos_embed=...)``.

        Returns:
            ``(cos_dt, sin_dt)`` — both 2-D ``bfloat16`` DTensors of
            shape ``(max_position_embeddings, head_dim)``. Matches the
            ``attention_layer`` precondition
            ``cos_pos_embed.num_dims == 2 and
            cos_pos_embed.dim(1) == head_dim``.
        """
        # Local import so that test files which use ``layers`` without
        # ever entering a compile scope can still import the module.
        from ..context import current_pk

        pk = current_pk()
        # ``attach_input`` works on any torch.Tensor (Parameter or
        # plain). We keep strong references to the buffers via
        # ``self.cos`` / ``self.sin`` so the runtime's pointer GC
        # (``persistent_kernel.py:_torch_tensor_refs``) never sees a
        # dangling buffer.
        cos_name = f"{self.prefix}cos" if self.prefix else "rotary_cos"
        sin_name = f"{self.prefix}sin" if self.prefix else "rotary_sin"
        cos_dt = pk.attach_input(self.cos, name=cos_name)
        sin_dt = pk.attach_input(self.sin, name=sin_name)
        return cos_dt, sin_dt
