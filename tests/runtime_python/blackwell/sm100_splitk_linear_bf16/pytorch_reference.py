"""PyTorch reference for `splitk_linear_layer` (BF16, sm100)."""
import torch


def splitk_linear_ref(input_bf16: torch.Tensor,
                      weight_bf16: torch.Tensor,
                      pre_output: torch.Tensor | None = None,
                      accumulate: bool = False) -> torch.Tensor:
    """Mirror the BF16 splitk_linear_sm100 kernel's behavior.

    The kernel computes ``out += input @ weight.T`` via TMA reduce-add. With
    ``accumulate=True``, ``pre_output`` carries forward (the kernel adds onto
    it). With ``accumulate=False`` the layer prepends a tensor_init that
    zeroes the output first, so the result is a pure sum.

    Args:
      input_bf16:  [batch, K]
      weight_bf16: [N, K]
      pre_output:  [batch, N] — the buffer's value when the kernel starts.
                   Required when ``accumulate=True``; ignored otherwise.
      accumulate:  whether the kernel accumulates onto pre_output.
    """
    matmul = (input_bf16.float() @ weight_bf16.float().t()).to(input_bf16.dtype)
    if accumulate:
        assert pre_output is not None
        return (pre_output.float() + matmul.float()).to(input_bf16.dtype)
    return matmul
