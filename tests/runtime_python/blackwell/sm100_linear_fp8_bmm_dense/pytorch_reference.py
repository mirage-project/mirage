"""Canonical PyTorch references for the per-head dense FP8 BMM family
(sm100_linear_fp8_bmm_dense).

Two in-scope layers share this file:
  - linear_fp8_bmm_dense_sm100        (bf16 output)
  - linear_fp8_bmm_dense_fp8out_sm100 (D3: FP8 + float32-scale fused output)

Both compute the same per-head GEMM:
    out[m, h, :] = sum_k  A_dq[m, h, k] * B_dq[h, :, k]
with
    A  [M, H, K]  FP8 + float32 1x128-group activation scale sa [M, H, K/128]
    B  [H, N, K]  FP8 + float32 128x128-block weight scale     sb [H, N/128, K/128]
The fp8out flavor additionally fuses the downstream per-token-group float32
quantize that feeds the o_proj dense GEMM: the per-head N=128 output IS exactly
one 128-K-group of the o_proj input row [M, H*128], so the per-head row max is
that group's scale.
"""
import torch

FP8_MAX = 448.0


def quantize_act_1x128(a_bf16):
    """A [M,H,K] -> FP8 [M,H,K] + float32 1x128-group scale sa [M,H,K/128].
    Matches quantize_fp8_layer(scale_ue8m0=False): per (row, 128-K-group) max."""
    M, H, K = a_bf16.shape
    nk = K // 128
    a_f32 = a_bf16.float()
    blk = a_f32.reshape(M, H, nk, 128)
    amax = blk.abs().amax(dim=-1).clamp(min=1e-10)            # [M,H,nk]
    scale = amax / FP8_MAX                                    # 1x128 group scale
    q = (blk / scale.unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX)
    a_fp8 = q.reshape(M, H, K).to(torch.float8_e4m3fn)
    return a_fp8, scale.contiguous()                          # sa [M,H,nk]


def quantize_wt_128x128(b_f32):
    """B [H,N,K] -> FP8 [H,N,K] + float32 128x128-block scale sb [H,N/128,K/128].
    For N=128 -> [H,1,nk]."""
    H, N, K = b_f32.shape
    bN, nk = N // 128, K // 128
    blk = b_f32.reshape(H, bN, 128, nk, 128)
    amax = blk.abs().amax(dim=(2, 4)).clamp(min=1e-12)        # [H,bN,nk]
    scale = amax / FP8_MAX
    q = (blk / scale.unsqueeze(2).unsqueeze(4)).clamp(-FP8_MAX, FP8_MAX)
    b_fp8 = q.reshape(H, N, K).to(torch.float8_e4m3fn)
    return b_fp8, scale.contiguous()                          # sb [H,bN,nk]


def bmm_dense_f32(a_fp8, sa, b_fp8, sb):
    """Per-head C[m,h,n] = sum_k A_dq[m,h,k] * B_dq[h,n,k], FP32 (no bf16 cast).

    This is the float32 accumulator the kernel epilogue quantizes from, so use
    it (not the bf16-rounded form) as the source for the fp8out reference."""
    M, H, K = a_fp8.shape
    N = b_fp8.shape[1]
    nk = K // 128
    a_dq = (a_fp8.float().reshape(M, H, nk, 128) *
            sa.unsqueeze(-1)).reshape(M, H, K)                # [M,H,K]
    b_dq = (b_fp8.float().reshape(H, N // 128, 128, nk, 128) *
            sb.unsqueeze(2).unsqueeze(4)).reshape(H, N, K)    # [H,N,K]
    return torch.einsum("mhk,hnk->mhn", a_dq, b_dq)           # [M,H,N] f32


def reference_bf16(a_fp8, sa, b_fp8, sb):
    """bf16-output reference (linear_fp8_bmm_dense_sm100)."""
    return bmm_dense_f32(a_fp8, sa, b_fp8, sb).to(torch.bfloat16)


def reference_fp8out(a_fp8, sa, b_fp8, sb):
    """D3 fp8out reference: BMM (FP32) then per-head float32-scale quantize.

    Output layout mirrors the o_proj input the fused epilogue writes:
      out_fp8   [M, H*N] row-major FP8   (per head: out_f32[:, h, :] / scale[:, h])
      out_scale [M, H]   float32         (per head: rowmax(out_f32[:, h, :]) / 448)
    N is the per-head width (128). Each per-head output row is exactly one
    128-K-group of the o_proj input, so there is one scale per (m, h).
    """
    out_f32 = bmm_dense_f32(a_fp8, sa, b_fp8, sb)             # [M,H,N]
    M, H, N = out_f32.shape
    assert N == 128, "fp8out reference assumes per-head N == 128 (one group)"
    rowmax = out_f32.abs().amax(dim=-1).clamp(min=1e-30)      # [M,H]
    scale = rowmax / FP8_MAX                                  # [M,H] float32
    q = (out_f32 / scale.unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX)
    out_fp8 = q.reshape(M, H * N).to(torch.float8_e4m3fn)     # [M, H*N]
    return out_fp8, scale.contiguous()                       # [M,H*N], [M,H]


def cosine_sim(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return (torch.dot(a, b) / (a.norm() * b.norm() + 1e-12)).item()
