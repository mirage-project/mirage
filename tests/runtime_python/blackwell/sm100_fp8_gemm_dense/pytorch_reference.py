"""PyTorch references for the dense FP8 block-scaled GEMM family.

Canonical home for the quantizers + reference math shared by every test in
`sm100_fp8_gemm_dense/`. The dense FP8 scale layout is plain float32:
    sa: float32 [M, K/128]    row-major   (1x128 group activation scale)
    sb: float32 [N/128, K/128] row-major  (128x128 block weight scale)

Kernels covered:
  * fp8_gemm_dense_{smallm,mediumm}_sm100        -> bf16 out
  * fp8_gemm_dense_{smallm,mediumm}_fp8out_sm100 -> fp8 out + UE8M0 scale
  * fp8_gemm_dense_decode_splitk_sm100           -> bf16 out (split-K accum)

The fp8out scale is flat uint32 `[M, N/128]` row-major; each entry's low 8
bits hold the UE8M0 exponent byte of the per-128-N-group max
(`encode_ue8m0(local_max / 448)`), upper 24 bits zero. See
`fp8_gemm_dense_qout_sm100_common.cuh` lines 350-404.
"""
import os
import sys

import torch

# Reuse the shared UE8M0 encode/decode helper rather than re-deriving it.
_COMMON_DIR = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../common"))
if _COMMON_DIR not in sys.path:
    sys.path.insert(0, _COMMON_DIR)
from sm100_fp8_scale_layout import (  # noqa: E402
    FP8_MAX,
    encode_ue8m0,
    decode_ue8m0,
)


# ---------------------------------------------------------------------------
# Dense f32-block quantizers (canonical; previously in _build_helper.py)
# ---------------------------------------------------------------------------

def quantize_a_f32scale(a_bf16: torch.Tensor):
    """Quantize A [M, K] -> FP8 e4m3 + float32 scale [M, K/128].

    Each scale covers a 1x128 group (per-row, per-128-columns chunk):
        sa[m, ki] = abs_max(A[m, ki*128:(ki+1)*128]) / FP8_MAX
    """
    M, K = a_bf16.shape
    assert K % 128 == 0, "K must be a multiple of 128"
    nk = K // 128

    a_fp8 = torch.empty_like(a_bf16, dtype=torch.float8_e4m3fn)
    sa = torch.zeros((M, nk), dtype=torch.float32, device=a_bf16.device)

    a_f32 = a_bf16.float()
    for m in range(M):
        for ki in range(nk):
            block = a_f32[m, ki * 128:(ki + 1) * 128]
            abs_max = block.abs().max().item()
            scale = abs_max / FP8_MAX if abs_max > 0 else 1.0
            sa[m, ki] = scale
            a_fp8[m, ki * 128:(ki + 1) * 128] = (
                (block / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
            )
    return a_fp8, sa


def quantize_b_f32scale(b_bf16: torch.Tensor):
    """Quantize B [N, K] -> FP8 e4m3 + float32 scale [N/128, K/128].

    Each scale covers a 128x128 block:
        sb[bi, ki] = abs_max(B[bi*128:(bi+1)*128, ki*128:(ki+1)*128]) / FP8_MAX
    """
    N, K = b_bf16.shape
    assert K % 128 == 0 and N % 128 == 0, "N and K must be multiples of 128"
    nb = N // 128
    nk = K // 128

    b_fp8 = torch.empty_like(b_bf16, dtype=torch.float8_e4m3fn)
    sb = torch.zeros((nb, nk), dtype=torch.float32, device=b_bf16.device)

    b_f32 = b_bf16.float()
    for bi in range(nb):
        for ki in range(nk):
            block = b_f32[bi * 128:(bi + 1) * 128, ki * 128:(ki + 1) * 128]
            abs_max = block.abs().max().item()
            scale = abs_max / FP8_MAX if abs_max > 0 else 1.0
            sb[bi, ki] = scale
            b_fp8[bi * 128:(bi + 1) * 128, ki * 128:(ki + 1) * 128] = (
                (block / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
            )
    return b_fp8, sb


# ---------------------------------------------------------------------------
# Reference GEMM (dequant A, B; matmul in f32; cast bf16)
# ---------------------------------------------------------------------------

def dequant_a_f32scale(a_fp8: torch.Tensor, sa: torch.Tensor) -> torch.Tensor:
    M, K = a_fp8.shape
    nk = K // 128
    a_f32 = a_fp8.float()
    a_dq = torch.empty(M, K, dtype=torch.float32, device=a_fp8.device)
    for m in range(M):
        for ki in range(nk):
            a_dq[m, ki * 128:(ki + 1) * 128] = (
                a_f32[m, ki * 128:(ki + 1) * 128] * sa[m, ki])
    return a_dq


def dequant_b_f32scale(b_fp8: torch.Tensor, sb: torch.Tensor) -> torch.Tensor:
    N, K = b_fp8.shape
    nk = K // 128
    nb = N // 128
    b_f32 = b_fp8.float()
    b_dq = torch.empty(N, K, dtype=torch.float32, device=b_fp8.device)
    for bi in range(nb):
        for ki in range(nk):
            b_dq[bi * 128:(bi + 1) * 128, ki * 128:(ki + 1) * 128] = (
                b_f32[bi * 128:(bi + 1) * 128,
                      ki * 128:(ki + 1) * 128] * sb[bi, ki])
    return b_dq


def reference_gemm(a_fp8, sa, b_fp8, sb):
    """Dequant A, B then compute C = A @ B.T in f32, return bf16."""
    a_dq = dequant_a_f32scale(a_fp8, sa)
    b_dq = dequant_b_f32scale(b_fp8, sb)
    return torch.matmul(a_dq, b_dq.t()).to(torch.bfloat16)


def reference_gemm_f32(a_fp8, sa, b_fp8, sb):
    """Same as reference_gemm but returns the un-rounded f32 result.

    Used by the fp8out re-quantize reference so the per-group max isn't first
    perturbed by a bf16 round-trip — matches the kernel which re-quantizes the
    f32 accumulator in registers.
    """
    a_dq = dequant_a_f32scale(a_fp8, sa)
    b_dq = dequant_b_f32scale(b_fp8, sb)
    return torch.matmul(a_dq, b_dq.t())


# ---------------------------------------------------------------------------
# fp8out re-quantize reference
# ---------------------------------------------------------------------------

def requantize_fp8out_ref(c_f32: torch.Tensor):
    """Re-quantize a [M, N] f32 GEMM result to the fp8out kernel's output.

    Mirrors `fp8_gemm_dense_qout_sm100_common.cuh` epilogue:
      * group = 128 consecutive N columns (one 128-N tile per consumer thread)
      * local_max  = max(|c|) over the 128-group, floored at 1e-30
      * y_scale    = local_max / 448
      * scale_byte = encode_ue8m0(y_scale)              (UE8M0 exponent)
      * inv_scale  = 2^(127 - scale_byte)
      * fp8[n]     = clamp(c[n] * inv_scale, -448, 448) -> e4m3

    Returns
    -------
    c_fp8   : [M, N] float8_e4m3fn
    c_scale : [M, N/128] uint32   (low 8 bits = scale_byte, flat row-major)
    """
    M, N = c_f32.shape
    assert N % 128 == 0, "fp8out requires N divisible by 128"
    ngroups = N // 128
    device = c_f32.device

    c_fp8 = torch.empty((M, N), dtype=torch.float8_e4m3fn, device=device)
    c_scale = torch.zeros((M, ngroups), dtype=torch.uint32, device=device)

    for m in range(M):
        for g in range(ngroups):
            block = c_f32[m, g * 128:(g + 1) * 128]
            local_max = max(block.abs().max().item(), 1e-30)
            y_scale = local_max / FP8_MAX
            scale_byte = encode_ue8m0(y_scale)
            inv_scale = 2.0 ** (127.0 - float(scale_byte))
            q = (block * inv_scale).clamp(-FP8_MAX, FP8_MAX)
            c_fp8[m, g * 128:(g + 1) * 128] = q.to(torch.float8_e4m3fn)
            c_scale[m, g] = scale_byte
    return c_fp8, c_scale


def dequant_fp8out(c_fp8: torch.Tensor, c_scale: torch.Tensor) -> torch.Tensor:
    """Dequant the fp8out (fp8 + flat-uint32-UE8M0-scale) pair back to f32.

    scale of group g of row m = 2^(scale_byte - 127), scale_byte = low 8 bits
    of c_scale[m, g]. Returns [M, N] f32.
    """
    M, N = c_fp8.shape
    ngroups = c_scale.shape[1]
    assert ngroups == N // 128
    c_f = c_fp8.float()
    out = torch.empty((M, N), dtype=torch.float32, device=c_fp8.device)
    for m in range(M):
        for g in range(ngroups):
            scale_byte = int(c_scale[m, g].item()) & 0xFF
            scale = decode_ue8m0(scale_byte)
            out[m, g * 128:(g + 1) * 128] = (
                c_f[m, g * 128:(g + 1) * 128] * scale)
    return out


# ---------------------------------------------------------------------------
# decode_splitk reference (split-K only changes accumulation order)
# ---------------------------------------------------------------------------

def reference_gemm_splitk(a_fp8, sa, b_fp8, sb, split_k: int):
    """Reference for fp8_gemm_dense_decode_splitk.

    Mathematically identical to `reference_gemm` (split-K partitions the K
    axis across CTAs and reduce-adds bf16 partials). We replicate the bf16
    partial accumulation so the reference carries the same intermediate
    rounding the kernel does: each of `split_k` K-slices is accumulated in
    f32 then the partials are summed in bf16 (red.global.add.bf16x2).
    """
    M, K = a_fp8.shape
    N = b_fp8.shape[0]
    assert K % (128 * split_k) == 0
    nk = K // 128
    nk_per_slice = nk // split_k

    a_dq = dequant_a_f32scale(a_fp8, sa)
    b_dq = dequant_b_f32scale(b_fp8, sb)

    acc = torch.zeros((M, N), dtype=torch.float32, device=a_fp8.device)
    for s in range(split_k):
        k0 = s * nk_per_slice * 128
        k1 = (s + 1) * nk_per_slice * 128
        partial = torch.matmul(a_dq[:, k0:k1], b_dq[:, k0:k1].t())
        # bf16 reduce-add: round each partial to bf16, accumulate as bf16.
        acc = (acc + partial.to(torch.bfloat16).float())
    return acc.to(torch.bfloat16)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a_f = a.float().flatten()
    b_f = b.float().flatten()
    # Guard (added 2026-06-16): a NaN/inf in either tensor poisons dot()+norm()
    # to nan, which reads as a mysterious "cos=nan" and gets mistaken for a
    # harness bug (cost a KDA GEMV several rounds). Return -2.0 (an impossible
    # cosine) so the caller sees a clearly-invalid, non-passing sentinel it can
    # diagnose — not a silent nan. The caller should classify the output for
    # nan/inf BEFORE calling this (see the per-task test) for a named cause.
    if not (torch.isfinite(a_f).all() and torch.isfinite(b_f).all()):
        return -2.0
    return (torch.dot(a_f, b_f) / (a_f.norm() * b_f.norm() + 1e-12)).item()


def rel_mean(out: torch.Tensor, ref: torch.Tensor) -> float:
    """Mean |out-ref| / (mean |ref| + eps) — magnitude-robust rel error."""
    o = out.float()
    r = ref.float()
    return (o - r).abs().mean().item() / (r.abs().mean().item() + 1e-12)
