"""The MPK layer catalog.

Each module in this package is a ``torch.nn.Module`` wrapping one MPK
kernel task. Both ``forward()`` (PyTorch reference) and ``compile()``
(MPK task registration) are implemented; the model author composes
them like normal PyTorch.

``MPKModule`` (the base class) lives in :mod:`._base`. Concrete layers
live in topic subpackages (``linear/``, ``norm/``, ``attention/``, etc.)
and are re-exported here for ``from mirage.mpk import layers`` users.
"""

from ._base import MPKModule

# Elementwise
from .elementwise.identity import Identity
from .elementwise import add

# Embedding
from .embedding.embed import Embed

# Normalization
from .norm.rmsnorm import RMSNorm
from .norm.rmsnorm_linear import RMSNormLinear

# Linear / GEMM
from .linear.linear import Linear
from .linear.linear_with_residual import LinearWithResidual

# Activation
from .activation.silu_mul import SiluMul, SiluMulLinearWithResidual

# Argmax (single-shot + split-reduce; MLA / nvshmem variants in follow-ups)
from .argmax.argmax import Argmax
from .argmax.argmax_partial import ArgmaxPartial
from .argmax.argmax_reduce import ArgmaxReduce

# Positional / rotary
from .rotary import RotaryEmbedding

# Attention (plain decode + paged prefill/decode; MLA / split-KV in follow-ups)
from .attention.attention import Attention
from .attention.paged_attention import PagedAttention

__all__ = [
    "MPKModule",
    # elementwise
    "Identity",
    "add",
    # embedding
    "Embed",
    # norm
    "RMSNorm",
    "RMSNormLinear",
    # linear
    "Linear",
    "LinearWithResidual",
    # activation
    "SiluMul",
    "SiluMulLinearWithResidual",
    # argmax
    "Argmax",
    "ArgmaxPartial",
    "ArgmaxReduce",
    # rotary
    "RotaryEmbedding",
    # attention
    "Attention",
    "PagedAttention",
]
