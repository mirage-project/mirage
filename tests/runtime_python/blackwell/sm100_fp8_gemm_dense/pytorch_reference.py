"""PyTorch references for the dense FP8 block-scaled GEMM family.

Canonical home for the quantizers + reference math shared by every test in
`sm100_fp8_gemm_dense/`. The dense FP8 scale layout is plain float32:
    sa: float32 [M, K/128]    row-major   (1x128 group activation scale)
    sb: float32 [N/128, K/128] row-major  (128x128 block weight scale)

Kernels covered:
  * fp8_gemm_dense_{smallm,mediumm}_sm100        -> bf16 out

"""
import os
import sys

import torch

# Reuse the shared FP8_MAX constant from the UE8M0 scale-layout helper.
_COMMON_DIR = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../common"))
if _COMMON_DIR not in sys.path:
    sys.path.insert(0, _COMMON_DIR)
from sm100_fp8_scale_layout import FP8_MAX  # noqa: E402


# ---------------------------------------------------------------------------
# Dense f32-block quantizers (canonical home).
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
