"""Dense GEMM-style layers."""

from .linear import Linear
from .linear_with_residual import LinearWithResidual

__all__ = ["Linear", "LinearWithResidual"]
