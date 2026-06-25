"""The MPK layer catalog.

Each module in this package is a ``torch.nn.Module`` wrapping one MPK
kernel task. Both ``forward()`` (PyTorch reference) and ``compile()``
(MPK task registration) are implemented; the model author composes
them like normal PyTorch.

``MPKModule`` (the base class) lives in :mod:`._base`. Concrete layers
live in topic subpackages (``linear/``, ``norm/``, ``attention/``, etc.)
and are re-exported here for ``from mirage.mpk import layers`` users.

This catalog is pruned to the Qwen3 dependency closure. DeepSeek-V3 /
MoE / MLA / MTP / FP8 layers were removed alongside the DSV3 demo.
"""

from ._base import MPKModule

# Standalone utility modules
from .allreduce import AllReduce

# Embedding
from .embedding.embed import Embed

# Normalization
from .norm.rmsnorm import RMSNorm

# Linear / GEMM
from .linear.linear import Linear
from .linear.linear_with_residual import LinearWithResidual
from .linear.parallel_linear import (
    ColumnParallelLinear,
    RowParallelLinear,
    RowParallelLinearWithResidual,
    MergedColumnParallelLinear,
    QKVParallelLinear,
)
from .linear.splitk_linear import SplitKLinear

# Argmax (split-reduce decode sampling)
from .argmax.argmax_partial import ArgmaxPartial
from .argmax.argmax_reduce import ArgmaxReduce

# Positional / rotary
from .rotary import RotaryEmbedding

# Attention (plain decode + paged prefill/decode)
from .attention.attention import Attention
from .attention.paged_attention import PagedAttention

__all__ = [
    "MPKModule",
    # standalone utility modules
    "AllReduce",
    # embedding
    "Embed",
    # norm
    "RMSNorm",
    # linear
    "Linear",
    "LinearWithResidual",
    "ColumnParallelLinear",
    "RowParallelLinear",
    "RowParallelLinearWithResidual",
    "MergedColumnParallelLinear",
    "QKVParallelLinear",
    "SplitKLinear",
    # argmax
    "ArgmaxPartial",
    "ArgmaxReduce",
    # rotary
    "RotaryEmbedding",
    # attention
    "Attention",
    "PagedAttention",
]
