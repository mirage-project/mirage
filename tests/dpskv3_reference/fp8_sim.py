"""FP8 GEMM simulation that mirrors MPK's quantize-then-matmul path.

The default reference uses BF16 weights (after lossless FP8 → BF16 dequant on
load) and BF16 activations end-to-end. MPK instead does the matmul in FP8:

  1. Activation A_bf16 is quantized to FP8 with a per-row, 128-wide group
     scale: s_a = absmax(A_group) / 448, A_fp8 = round(A / s_a) clamp [-448,
     448] then bitcast to e4m3.
  2. Weight is already stored as FP8 with a per-block 128×128 scale s_w
     (= weight_scale_inv).
  3. Output O_bf16 = (A_fp8 * s_a) @ (W_fp8 * s_w)^T (block-scaled MMA).

Token alignment vs the BF16 reference diverges at iter 0 because of the
activation-side FP8 quantization noise. To get an FP8-faithful reference we
need to replicate steps 1-3 above per linear, then we can do FP8-vs-FP8
correctness diff against MPK and isolate kernel bugs from quantization noise.

This module provides `fp8_simulated_linear`: PyTorch-only implementation of
the same quantize → block-scaled-matmul → dequant sequence. The hardware
FP8 round-to-even is approximated by PyTorch's native
`bfloat16 -> float8_e4m3fn` cast, which uses the standard satfin rounding
matching what MPK's `per_token_group_quantize_fp8.cuh` produces.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


FP8_MAX = 448.0  # max representable magnitude for float8_e4m3fn
GROUP_SIZE = 128  # MPK's per-row activation group + per-block weight scale tile


def _quantize_activation_fp8(
    a_bf16: torch.Tensor, group_size: int = GROUP_SIZE
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row, group-wise activation quantization matching MPK.

    Input  a_bf16:  [..., K], any leading shape collapses to M rows
    Returns:
      a_fp8:  [..., K] float8_e4m3fn
      scale:  [M, K // group_size] float32 (one scale per group of `group_size`
              consecutive K elements within each row)
    """
    if a_bf16.shape[-1] % group_size != 0:
        raise ValueError(
            f"activation last dim {a_bf16.shape[-1]} not divisible by "
            f"group_size {group_size}"
        )
    leading = a_bf16.shape[:-1]
    K = a_bf16.shape[-1]
    M = int(torch.tensor(leading).prod()) if len(leading) > 0 else 1
    a_flat = a_bf16.reshape(M, K).float()
    num_groups = K // group_size
    a_groups = a_flat.view(M, num_groups, group_size)
    absmax = a_groups.abs().amax(dim=-1, keepdim=False).clamp(min=1e-10)
    scale = absmax / FP8_MAX  # [M, num_groups]
    a_normalized = a_groups / scale.unsqueeze(-1)  # in [-FP8_MAX, FP8_MAX]
    a_fp8 = a_normalized.view(M, K).to(torch.float8_e4m3fn)
    return a_fp8.view(*leading, K), scale


def _dequant_block_weight(
    w_fp8: torch.Tensor, w_scale: torch.Tensor, block: int = GROUP_SIZE
) -> torch.Tensor:
    """Reconstruct the BF16 weight from FP8 + per-(`block`×`block`) scale.

    w_fp8:   [N, K] float8_e4m3fn
    w_scale: [ceil(N/block), ceil(K/block)] float32  (DeepSeek layout)
    Returns w_bf16 [N, K] bfloat16.

    DeepSeek's checkpoint allows N or K that's not a multiple of `block`
    (e.g. `kv_a_proj_with_mqa` has N=576 = 4.5 blocks). The scale tensor
    still has `ceil(N/block)` rows; the trailing partial block re-uses the
    same scale. We replicate the scale to a (Nb*block, Kb*block) grid,
    crop to (N, K), then multiply.

    This matches the math MPK's `fp8_gemm_dense` kernels do per tile (the
    fused tcgen05.mma reads w_fp8 and w_scale and dequantizes on the fly).
    """
    N, K = w_fp8.shape
    Nb = (N + block - 1) // block
    Kb = (K + block - 1) // block
    if w_scale.shape != (Nb, Kb):
        raise ValueError(
            f"weight scale shape {tuple(w_scale.shape)} mismatch with "
            f"expected (ceil({N}/{block})={Nb}, ceil({K}/{block})={Kb})"
        )
    w_f32 = w_fp8.to(torch.float32)
    # Broadcast the (Nb, Kb) scale to (Nb*block, Kb*block) then crop.
    scale_full = w_scale.repeat_interleave(block, dim=0).repeat_interleave(
        block, dim=1
    )
    scale_full = scale_full[:N, :K]
    return (w_f32 * scale_full).to(torch.bfloat16)


def fp8_simulated_linear(
    a_bf16: torch.Tensor,
    w_fp8: torch.Tensor,
    w_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    group_size: int = GROUP_SIZE,
) -> torch.Tensor:
    """Drop-in for `F.linear` that simulates MPK's FP8 block-scaled GEMM.

    a_bf16:  [..., K]
    w_fp8:   [N, K] float8_e4m3fn
    w_scale: [ceil(N/group_size), K // group_size] float32
    bias:    [N] (optional)

    Returns [..., N] bf16.

    Math (matching `fp8_gemm_dense_smallm_sm100`'s
    `tcgen05.mma.kind::f8f6f4` semantics exactly):

        A_fp8[i, k]  = round_to_e4m3( A_bf16[i, k] / a_scale[i, k//128] )
        W_fp8[j, k]  =                W_fp8[j, k]                 (from ckpt)
        out[i, j]    = sum_k ( A_fp8[i, k] * a_scale[i, k//128]
                              * W_fp8[j, k] * w_scale[j//128, k//128] )
                         accumulated in FP32, cast to BF16 at the end.

    The accumulation happens in FP32, NOT bf16. A previous version of this
    function dequant'd `A_fp8` to bf16 first; that extra cast threw away
    the precision we just paid for, making the simulation strictly less
    faithful than MPK's hardware path. This version keeps the dequant'd
    activation in FP32 so the matmul matches MPK bit-for-bit on the
    accumulator side. The only remaining numerical gap vs MPK is the
    FP8 rounding-mode used by `bfloat16 -> float8_e4m3fn` cast (saturating
    round-to-nearest-even per PyTorch), which is the same scheme as
    MPK's `per_token_group_quantize_fp8.cuh`.
    """
    a_fp8, a_scale = _quantize_activation_fp8(a_bf16, group_size=group_size)
    leading = a_bf16.shape[:-1]
    K = a_bf16.shape[-1]
    M = int(torch.tensor(leading).prod()) if len(leading) > 0 else 1
    num_groups = K // group_size
    # FP8 -> FP32 is exact (no precision loss). Apply the per-row group
    # scale in FP32; the result is the same value MPK's MMA sees on the
    # A side per inner-product step.
    a_f32 = (
        a_fp8.reshape(M, num_groups, group_size).to(torch.float32)
        * a_scale.unsqueeze(-1)
    ).view(M, K)
    # Dequant the weight to FP32 as well so the matmul carries no extra
    # rounding from the activation side. _dequant_block_weight casts to
    # bf16 at the end; instead we re-run the math in FP32 here.
    N, _ = w_fp8.shape
    Nb = (N + group_size - 1) // group_size
    Kb = K // group_size
    scale_full = w_scale.repeat_interleave(group_size, dim=0).repeat_interleave(
        group_size, dim=1
    )[:N, :K]
    w_f32 = w_fp8.to(torch.float32) * scale_full
    # FP32 matmul -> BF16 output (matches MPK's tcgen05 accum -> bf16 store).
    out_f32 = a_f32 @ w_f32.t()
    if bias is not None:
        out_f32 = out_f32 + bias.float()
    return out_f32.to(torch.bfloat16).reshape(*leading, N)


class FP8SimulatedLinear(nn.Module):
    """An `nn.Module` wrapper exposing the same API as `nn.Linear` but
    routing forward through `fp8_simulated_linear`.

    Holds FP8 weight + per-block weight scale as buffers (not parameters —
    they are not trained; the FP8 values come from the loaded checkpoint).
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        if in_features % GROUP_SIZE != 0 or out_features % GROUP_SIZE != 0:
            raise ValueError(
                f"FP8SimulatedLinear requires shapes divisible by "
                f"GROUP_SIZE={GROUP_SIZE}; got ({out_features}, {in_features})"
            )
        # Empty FP8 weight + scale; the loader will populate these.
        self.register_buffer(
            "weight_fp8",
            torch.empty(out_features, in_features, dtype=torch.float8_e4m3fn),
        )
        self.register_buffer(
            "weight_scale_inv",
            torch.empty(
                out_features // GROUP_SIZE,
                in_features // GROUP_SIZE,
                dtype=torch.float32,
            ),
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return fp8_simulated_linear(
            x, self.weight_fp8, self.weight_scale_inv, self.bias
        )

    def load_from_fp8(self, weight_fp8: torch.Tensor, weight_scale: torch.Tensor) -> None:
        """Copy a checkpoint-style (FP8 weight, FP32 scale) pair into this
        module's buffers, with dtype + shape checks."""
        if weight_fp8.dtype != torch.float8_e4m3fn:
            raise TypeError(
                f"expected float8_e4m3fn weight, got {weight_fp8.dtype}"
            )
        if weight_fp8.shape != (self.out_features, self.in_features):
            raise ValueError(
                f"shape mismatch: weight {tuple(weight_fp8.shape)} vs "
                f"({self.out_features}, {self.in_features})"
            )
        Nb = self.out_features // GROUP_SIZE
        Kb = self.in_features // GROUP_SIZE
        if weight_scale.shape != (Nb, Kb):
            raise ValueError(
                f"scale shape {tuple(weight_scale.shape)} vs ({Nb}, {Kb})"
            )
        self.weight_fp8.copy_(weight_fp8.to(self.weight_fp8.device))
        self.weight_scale_inv.copy_(weight_scale.float().to(self.weight_scale_inv.device))
