"""MoE (Mixture-of-Experts) catalog layers.

One catalog module per kernel task. Variant-carrying classes have been
split into single-purpose subclasses; the old umbrella names are kept
as back-compat factories so existing model code still works.

* :class:`MoETopkSoftmaxRouting` -> ``moe_topk_softmax_sm100``
* :class:`MoETopkSigmoidRouting` -> ``moe_topk_sigmoid_sm100`` (DeepSeek V3)
* :class:`MoEW13BF16` / :class:`MoEW13FP8` -> ``moe_w13_linear_sm{100,90}`` / ``moe_w13_fp8_sm100``
* :class:`MoEW2BF16`  / :class:`MoEW2FP8`  -> ``moe_w2_linear_sm{100,90}``  / ``moe_w2_fp8_sm100``
* :class:`MoESiluMul`     -> ``moe_silu_mul``
* :class:`MoeMulSumAdd`   -> ``moe_mul_sum_add_sm100``
* :class:`MoEPermute` / :class:`MoEUnpermute` -> ``moe_permute_sm100`` / ``moe_unpermute_sm100``
"""

from .routing import (
    MoETopkSoftmaxRouting,
    MoETopkSigmoidRouting,
    MoETopkRouting,  # back-compat factory
)
from .w13 import MoEW13BF16, MoEW13FP8, MoEW13  # MoEW13 = back-compat factory
from .w2 import MoEW2BF16, MoEW2FP8, MoEW2      # MoEW2  = back-compat factory
from .silu_mul import MoESiluMul
from .mul_sum_add import MoeMulSumAdd
from .permute import MoEPermute, MoEUnpermute

__all__ = [
    # Routing
    "MoETopkSoftmaxRouting",
    "MoETopkSigmoidRouting",
    "MoETopkRouting",
    # W13
    "MoEW13BF16",
    "MoEW13FP8",
    "MoEW13",
    # W2
    "MoEW2BF16",
    "MoEW2FP8",
    "MoEW2",
    # Single-purpose
    "MoESiluMul",
    "MoeMulSumAdd",
    "MoEPermute",
    "MoEUnpermute",
]
