"""Precomputed RoPE cos/sin tables.

No MPK task — the actual RoPE rotation runs inside the attention
kernel. This module owns the precomputed ``cos`` and ``sin`` tables as
``nn.Buffer`` (``register_buffer(..., persistent=False)``) so they
move with ``.to(device, dtype)``, stay out of ``state_dict()`` (HF
checkpoints don't ship RoPE tables), and are not seen as trainable.
``compile()`` returns ``(cos_dt, sin_dt)`` for the attention kernel to
consume as ``cos_pos_embed`` / ``sin_pos_embed``.
"""
from __future__ import annotations

from typing import Tuple

import torch

from ._base import MPKModule
from ...core import DTensor


class RotaryEmbedding(MPKModule):
    """Precomputed cos/sin tables for RoPE.

    ``cos`` and ``sin``: ``(max_position_embeddings, head_dim)`` bf16
    non-persistent buffers, using the HF
    ``torch.cat((freqs, freqs), dim=-1)`` ``rotate_half`` convention.
    ``head_dim`` must be even.
    """

    def __init__(
        self,
        head_dim: int,
        max_position_embeddings: int,
        base: float = 10000.0,
        *,
        prefix: str = "",
    ) -> None:
        """Precompute the RoPE cos/sin tables and register them as buffers.

        Tensor contract (set by ``__init__``; no task is emitted):
          cos: (max_position_embeddings, head_dim) bf16 non-persistent buffer.
          sin: (max_position_embeddings, head_dim) bf16 non-persistent buffer.

        Notes: uses the HF/LLaMA ``torch.cat((freqs, freqs), dim=-1)``
        ``rotate_half`` convention — each (cos, sin) value is
        ``repeat_interleave``-doubled across the head_dim axis (NOT
        ``repeat_interleave(2)``; the duplicate halves go to ``[..., :D/2]``
        and ``[..., D/2:]``). ``head_dim`` must be even.
        """
        super().__init__(prefix=prefix)
        if head_dim % 2 != 0:
            raise ValueError(
                f"RotaryEmbedding requires an even head_dim; got head_dim={head_dim}."
            )
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = float(base)

        cos, sin = self._precompute_freqs(
            head_dim=head_dim,
            max_pos=max_position_embeddings,
            base=self.base,
        )
        # persistent=False so the tables don't collide with HF state_dict
        # keys yet still migrate with .to(device, dtype).
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @staticmethod
    def _precompute_freqs(
        head_dim: int,
        max_pos: int,
        base: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Standard HF/LLaMA RoPE precomputation in fp32, cast to bf16."""
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        positions = torch.arange(max_pos, dtype=torch.float32)
        freqs = torch.einsum("p,d->pd", positions, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(torch.bfloat16)
        sin = emb.sin().to(torch.bfloat16)
        return cos, sin

    def forward(
        self, positions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Lookup ``(cos[positions], sin[positions])`` (bf16, same device)."""
        if not torch.is_tensor(positions):
            raise TypeError(
                "RotaryEmbedding.forward expects a torch.Tensor of "
                f"position indices; got {type(positions).__name__}."
            )
        return self.cos[positions], self.sin[positions]

    def auto_grid_dim(self, *args, **kwargs):
        """Not applicable — RotaryEmbedding emits no MPK task."""
        raise NotImplementedError(
            "RotaryEmbedding does not emit an MPK task; the RoPE rotation "
            "is performed inside the attention kernel that consumes "
            "(cos, sin) DTensors returned by RotaryEmbedding.compile()."
        )

    def compile(self) -> Tuple[DTensor, DTensor]:
        """Attach precomputed cos/sin buffers to the active PK (no task emitted).

        Tensor contract:
          cos_dt: (max_position_embeddings, head_dim) bf16, RoPE table.
          sin_dt: (max_position_embeddings, head_dim) bf16, RoPE table.

        Notes: returns the two DTensors threaded into
        ``pk.attention_layer(cos_pos_embed=..., sin_pos_embed=...)``. The
        actual RoPE rotation lives inside the attention kernel.
        """
        from ..context import current_pk

        pk = current_pk()
        cos_name = f"{self.prefix}cos" if self.prefix else "rotary_cos"
        sin_name = f"{self.prefix}sin" if self.prefix else "rotary_sin"
        cos_dt = pk.attach_input(self.cos, name=cos_name)
        sin_dt = pk.attach_input(self.sin, name=sin_name)
        return cos_dt, sin_dt
