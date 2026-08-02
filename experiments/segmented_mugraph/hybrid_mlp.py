"""Drop-in replacement for ``Qwen3MLP`` backed by segmented muGraph regions.

Only the MLP is replaced.  Embeddings, attention, RoPE, KV-cache management,
sampling and every unsupported shape stay on the ordinary PyTorch / Hugging
Face path, and the transformer residual add stays exactly where the original
model applies it -- Region B is the *down projection alone*, matching
``Qwen3MLP.forward``:

    down_proj(act_fn(gate_proj(x)) * up_proj(x))

Nothing here constructs a ``PersistentKernel`` or generates a task graph.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Set

import torch
import torch.nn as nn

from .runner import SegmentedMuGraphRunner


class HybridQwen3MLP(nn.Module):
    """Routes supported token counts through muGraph regions, else to PyTorch.

    Weights stay ordinary runtime inputs, so all structurally identical layers
    share one compiled graph per shape bucket.
    """

    def __init__(
        self,
        original: nn.Module,
        runner: SegmentedMuGraphRunner,
        allowed_tokens: Set[int],
        stats: Dict[str, int],
        layer_idx: int = -1,
    ):
        super().__init__()
        self.original = original
        self.runner = runner
        self.allowed_tokens = allowed_tokens
        self.stats = stats
        self.layer_idx = layer_idx
        self.hidden_size = original.hidden_size
        self.intermediate_size = original.intermediate_size

    @property
    def w_gate(self) -> torch.Tensor:
        return self.original.gate_proj.weight

    @property
    def w_up(self) -> torch.Tensor:
        return self.original.up_proj.weight

    @property
    def w_down(self) -> torch.Tensor:
        return self.original.down_proj.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = x.numel() // self.hidden_size
        if tokens not in self.allowed_tokens:
            # Prefill, or any unseen / unsupported shape.
            self.stats["fallback_calls"] += 1
            return self.original(x)

        flat = x.reshape(tokens, self.hidden_size)
        if not flat.is_contiguous():
            flat = flat.contiguous()

        # Region A and Region B back to back, no synchronization in between.
        mid = self.runner.region_a(flat, self.w_gate, self.w_up, keep_padding=True)
        out = self.runner.region_b(mid, self.w_down, logical_tokens=tokens)
        self.stats["mugraph_calls"] += 1

        # The region output aliases a buffer reused by the next layer, so hand
        # the model its own copy.  At decode this is a few KB.
        return out.clone().reshape(*x.shape[:-1], self.hidden_size)


def patch_qwen3_mlps(
    model: nn.Module,
    runner: SegmentedMuGraphRunner,
    allowed_tokens: Iterable[int] = (1,),
    stats: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """Replace every decoder-layer MLP with :class:`HybridQwen3MLP`.

    Returns a handle carrying the shared *stats* counters and the list of
    patched layer indices.
    """
    stats = {"mugraph_calls": 0, "fallback_calls": 0} if stats is None else stats
    allowed = set(int(t) for t in allowed_tokens)
    patched: List[int] = []
    for idx, layer in enumerate(model.model.layers):
        layer.mlp = HybridQwen3MLP(layer.mlp, runner, allowed, stats, idx)
        patched.append(idx)
    return {"stats": stats, "patched_layers": patched, "allowed_tokens": sorted(allowed)}


def precompile_buckets(
    model: nn.Module,
    runner: SegmentedMuGraphRunner,
    buckets: Iterable[int],
    dtype: torch.dtype = torch.bfloat16,
) -> None:
    """Compile Region A/B for each fixed token bucket ahead of the first token.

    All Qwen3 dense layers share one hidden/intermediate pair, so this compiles
    exactly two graphs per bucket regardless of layer count.
    """
    cfg = model.config
    for tokens in buckets:
        runner.precompile_mlp(
            tokens=int(tokens),
            hidden_size=cfg.hidden_size,
            intermediate_size=cfg.intermediate_size,
            torch_dtype=dtype,
            with_residual=False,  # residual stays in the decoder layer
        )
