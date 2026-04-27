"""Pure-PyTorch reference for block-wise BF16 -> FP8 quantization.

Mirrors what the SM100 ``quantize_fp8_sm100`` task computes. Two scale layouts
are supported:

  * ``layout="row_major"``  -> packed UE8M0 scales of shape
    ``(outer_dim, packed_k)`` with row-major strides; matches
    :func:`allocate_packed_ue8m0_scale`.
  * ``layout="deepgemm_col_major"`` -> packed UE8M0 scales of shape
    ``(outer_dim, packed_k)`` but with column-major strides
    (physical layout ``[packed_k, aligned_outer]``); matches
    :func:`allocate_packed_ue8m0_scale_deepgemm_style` and is the layout the
    MPK ``quantize_fp8_layer`` writes to.
"""

import os
import sys

import torch

# The block-wise quantize helpers live in the shared common dir alongside
# the kernel layout utilities. Reuse them rather than duplicate the math.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
COMMON_DIR = os.path.abspath(os.path.join(THIS_DIR, "../common"))
if COMMON_DIR not in sys.path:
    sys.path.insert(0, COMMON_DIR)

from sm100_fp8_scale_layout import (  # noqa: E402  (import after sys.path tweak)
    BLOCK_K,
    quantize_to_fp8_deepgemm_style,
    quantize_to_fp8_packed_ue8m0,
)


def quantize_fp8_ref(
    input: torch.Tensor,
    scale_ue8m0: bool = True,
    layout: str = "row_major",
):
    """Reference block-wise BF16->FP8 quantization.

    Args:
        input: ``(outer_dim, reduction_size)`` ``bfloat16`` tensor.
        scale_ue8m0: must be ``True``. ``False`` (raw float32 scale, MoE path)
            is not handled by this reference yet -- the MPK MoE path uses a
            different layer name and a different reference would belong here
            if/when needed.
        layout: ``"row_major"`` (kernel-test default; matches
            ``allocate_packed_ue8m0_scale``) or ``"deepgemm_col_major"``
            (the layout the MPK ``quantize_fp8_layer`` produces, with
            physical shape ``[packed_k, aligned_outer]``).

    Returns:
        ``(output_fp8, output_scale)`` -- ``output_fp8`` is
        ``float8_e4m3fn`` of the same shape as ``input``; ``output_scale``
        is ``uint32`` with packed UE8M0 bytes (4 logical scales per
        ``uint32``).
    """
    if not scale_ue8m0:
        raise NotImplementedError(
            "quantize_fp8_ref currently only implements the UE8M0 path; "
            "scale_ue8m0=False is unused by the test_mode layer."
        )
    assert input.dtype == torch.bfloat16, "expected bf16 input"

    if layout == "row_major":
        return quantize_to_fp8_packed_ue8m0(input)
    if layout == "deepgemm_col_major":
        return quantize_to_fp8_deepgemm_style(input)
    raise ValueError(f"unsupported layout: {layout}")


__all__ = ["quantize_fp8_ref", "BLOCK_K"]
