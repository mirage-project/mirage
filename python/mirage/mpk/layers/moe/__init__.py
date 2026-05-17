"""MoE (Mixture-of-Experts) catalog layers.

Re-exports the catalog modules wrapping :class:`PersistentKernel`'s
MoE pk methods. Used by qwen3 MoE (softmax + bf16 experts) and
DeepSeek V3 (sigmoid + FP8 experts + NEW MoE permute path).

Variant maps:

* :class:`MoETopkRouting` -- ``variant in {"softmax", "sigmoid"}`` ->
  tasks ``moe_topk_softmax_sm100`` / ``moe_topk_sigmoid_sm100``.
  DeepSeek V3 always uses ``sigmoid``.
* :class:`MoEW13`         -- ``dtype in {"bf16", "fp8"}`` ->
  tasks ``moe_w13_linear_sm{100,90}`` / ``moe_w13_fp8_sm100``.
* :class:`MoEW2`          -- ``dtype in {"bf16", "fp8"}`` ->
  tasks ``moe_w2_linear_sm{100,90}`` / ``moe_w2_fp8_sm100``.
* :class:`MoESiluMul`     -- standalone, accepts 2-D or 3-D input,
  task ``moe_silu_mul``.
* :class:`MoeMulSumAdd`   -- standalone, task ``moe_mul_sum_add_sm100``.
* :class:`MoEPermute` / :class:`MoEUnpermute` -- standalone, tasks
  ``moe_permute_sm100`` / ``moe_unpermute_sm100`` (NEW MoE path).
"""

from .routing import MoETopkRouting
from .w13 import MoEW13
from .w2 import MoEW2
from .silu_mul import MoESiluMul
from .mul_sum_add import MoeMulSumAdd
from .permute import MoEPermute, MoEUnpermute

__all__ = [
    "MoETopkRouting",
    "MoEW13",
    "MoEW2",
    "MoESiluMul",
    "MoeMulSumAdd",
    "MoEPermute",
    "MoEUnpermute",
]
