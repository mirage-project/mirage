"""PyTorch reference implementations for the sm100_linear MPK kernel tests.

Two ops:
  - linear_ref(input, weight)               -> input @ weight.T
  - linear_with_residual_ref(input, w, res) -> input @ weight.T + residual

Inputs are bf16 contiguous CUDA tensors. Internally we cast to f32 for the
matmul and cast the final result back to bf16 to keep numerical noise small
when comparing against MPK kernels under rtol/atol = 1e-2.
"""

import torch


def linear_ref(input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Compute `input @ weight.T` in fp32 internally, returned in input's dtype.

    input:  [batch, in_dim]
    weight: [out_dim, in_dim]
    return: [batch, out_dim]
    """
    out_dtype = input.dtype
    return (input.float() @ weight.float().T).to(out_dtype)


def linear_with_residual_ref(
    input: torch.Tensor, weight: torch.Tensor, residual: torch.Tensor
) -> torch.Tensor:
    """Compute `input @ weight.T + residual` in fp32 internally, returned in input's dtype.

    input:    [batch, in_dim]
    weight:   [out_dim, in_dim]
    residual: [batch, out_dim]
    return:   [batch, out_dim]
    """
    out_dtype = input.dtype
    return (input.float() @ weight.float().T + residual.float()).to(out_dtype)
