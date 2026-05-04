"""Standalone tests for the new mHC kernels (K2, K4, K5, K3 reuse)
and the end-to-end hc_pre / hc_post pipeline."""
import os
import sys

import pytest
import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROFILE_DIR = os.path.join(THIS_DIR, "profile")
if PROFILE_DIR not in sys.path:
    sys.path.insert(0, PROFILE_DIR)

from utils import (
    hc_post_reference,
    hc_pre_reference,
    k1_reference,
    k2_reference,
    k4_reference,
    k5_reference,
    sinkhorn_knopp_torch,
)

import runtime_kernel_blackwell_mhc as rt

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)


# -----------------------------------------------------------------------------
# K1 (rmsnorm half): per-token RMSNorm with implicit unit weight
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("num_tokens,hidden", [
    (1, 256), (16, 1024), (1024, 4096), (32, 16384),
])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_k1_rmsnorm(num_tokens, hidden, dtype):
    gen = torch.Generator(device="cuda").manual_seed(50 + num_tokens + hidden)
    x = torch.randn(num_tokens, hidden, device="cuda", dtype=dtype, generator=gen)
    y = torch.empty_like(x)
    eps = 1e-6
    rt.mHC_rmsnorm(x, y, eps=eps)

    rsqrt = torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps)
    ref = (x.float() * rsqrt).to(dtype)
    rtol = 1e-2 if dtype == torch.bfloat16 else 1e-5
    atol = 1e-2 if dtype == torch.bfloat16 else 1e-6
    torch.testing.assert_close(y, ref, rtol=rtol, atol=atol)


# -----------------------------------------------------------------------------
# K2: affine + split + activation
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("num_tokens,n", [(1, 4), (8, 4), (32, 4), (5, 2), (3, 8)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_k2_affine_split_activation(num_tokens, n, dtype):
    gen = torch.Generator(device="cuda").manual_seed(100 + num_tokens + n)
    mix_hc = n * n + 2 * n
    mixes = torch.randn(num_tokens, mix_hc, device="cuda", dtype=dtype, generator=gen)
    scale = torch.randn(3, device="cuda", dtype=torch.float32, generator=gen)
    base = torch.randn(mix_hc, device="cuda", dtype=torch.float32, generator=gen)

    h_pre = torch.empty(num_tokens, n, device="cuda", dtype=torch.float32)
    h_post = torch.empty(num_tokens, n, device="cuda", dtype=torch.float32)
    h_res_logits = torch.empty(num_tokens, n * n, device="cuda", dtype=torch.float32)

    rt.mHC_affine_split_activation(mixes, scale, base, h_pre, h_post, h_res_logits, n)

    ref_pre, ref_post, ref_res = k2_reference(mixes, scale, base, n)
    rtol = 1e-2 if dtype == torch.bfloat16 else 1e-5
    atol = 1e-2 if dtype == torch.bfloat16 else 1e-6
    torch.testing.assert_close(h_pre, ref_pre, rtol=rtol, atol=atol)
    torch.testing.assert_close(h_post, ref_post, rtol=rtol, atol=atol)
    torch.testing.assert_close(h_res_logits, ref_res, rtol=rtol, atol=atol)


# -----------------------------------------------------------------------------
# K3: sinkhorn (existing kernel; smoke test in this wrapper)
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("num_tokens,n", [(1, 4), (16, 4), (32, 8)])
def test_k3_sinkhorn(num_tokens, n):
    gen = torch.Generator(device="cuda").manual_seed(200 + num_tokens + n)
    res_logits = torch.randn(num_tokens, n, n, device="cuda", dtype=torch.float32, generator=gen)
    out = torch.empty_like(res_logits)
    rt.sinkhorn_sm100(res_logits, out, repeat=20, eps=1e-9, token_block_size=1)
    ref = sinkhorn_knopp_torch(res_logits, repeat=20, eps=1e-9)
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-5)


# -----------------------------------------------------------------------------
# K4: weighted sum across n streams (mul_sum_add with zero residual)
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("num_tokens,n,c", [
    (1, 4, 128),
    (4, 4, 1024),
    (8, 4, 4096),
    (3, 2, 1024),
    (2, 8, 128),
])
def test_k4_mul_sum_add_zero_residual(num_tokens, n, c):
    gen = torch.Generator(device="cuda").manual_seed(300 + num_tokens + n + c)
    x = torch.randn(num_tokens, n, c, device="cuda", dtype=torch.bfloat16, generator=gen)
    h_pre = torch.rand(num_tokens, n, device="cuda", dtype=torch.float32, generator=gen)
    residual = torch.zeros(num_tokens, c, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(num_tokens, c, device="cuda", dtype=torch.bfloat16)

    rt.mul_sum_add_sm100(x, h_pre, residual, out, n)

    ref = k4_reference(h_pre, x)
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)


# -----------------------------------------------------------------------------
# K5: residual mix + post outer product
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("num_tokens,n,c", [
    (1, 4, 128),
    (4, 4, 1024),
    (8, 4, 4096),
    (3, 2, 1024),
    (2, 8, 128),
])
def test_k5_mul_sum_add_with_outer(num_tokens, n, c):
    gen = torch.Generator(device="cuda").manual_seed(400 + num_tokens + n + c)
    residual = torch.randn(num_tokens, n, c, device="cuda", dtype=torch.bfloat16, generator=gen)
    x = torch.randn(num_tokens, c, device="cuda", dtype=torch.bfloat16, generator=gen)
    comb = torch.rand(num_tokens, n, n, device="cuda", dtype=torch.float32, generator=gen)
    comb = comb / comb.sum(-1, keepdim=True)  # row-stochastic, similar to sinkhorn output
    post = torch.rand(num_tokens, n, device="cuda", dtype=torch.float32, generator=gen)
    out = torch.empty(num_tokens, n, c, device="cuda", dtype=torch.bfloat16)

    rt.mHC_mul_sum_add_with_outer(residual, x, comb, post, out, n)

    ref = k5_reference(residual, x, comb, post)
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)


# -----------------------------------------------------------------------------
# End-to-end pipeline: hc_pre then hc_post, vs PyTorch reference
# -----------------------------------------------------------------------------

def _run_hc_pre_with_kernels(x, hc_fn, hc_scale, hc_base, n,
                             sinkhorn_iters, hc_eps, norm_eps):
    """Run hc_pre using the wrapper kernels for K2/K3/K4 and PyTorch for K1
    (rmsnorm-with-ones + linear; those have their own existing kernels)."""
    b, s, n_chk, C = x.shape
    assert n_chk == n
    bs = b * s
    nC = n * C

    # K1 rmsnorm half (mHC kernel) + matmul (torch bf16 for now).
    x_flat_fp32 = x.reshape(bs, nC).float().contiguous()
    x_norm_bf16 = torch.empty(bs, nC, device=x.device, dtype=torch.bfloat16)
    rt.mHC_rmsnorm(x_flat_fp32, x_norm_bf16, eps=norm_eps)
    hc_fn_bf16 = hc_fn.to(torch.bfloat16)
    mixes = (x_norm_bf16.float() @ hc_fn_bf16.float().T).to(torch.bfloat16)

    # K2
    mix_hc = n * n + 2 * n
    h_pre = torch.empty(bs, n, device=x.device, dtype=torch.float32)
    h_post = torch.empty(bs, n, device=x.device, dtype=torch.float32)
    h_res_logits = torch.empty(bs, n * n, device=x.device, dtype=torch.float32)
    rt.mHC_affine_split_activation(mixes, hc_scale, hc_base, h_pre, h_post,
                                     h_res_logits, n)

    # K3
    res_mat = h_res_logits.reshape(bs, n, n).contiguous()
    comb = torch.empty_like(res_mat)
    rt.sinkhorn_sm100(res_mat, comb, repeat=sinkhorn_iters, eps=hc_eps,
                      token_block_size=1)

    # K4: F_pre = sum_i h_pre[i] * x[i,:]
    x_bs = x.reshape(bs, n, C).to(torch.bfloat16).contiguous()
    zero_res = torch.zeros(bs, C, device=x.device, dtype=torch.bfloat16)
    f_pre = torch.empty(bs, C, device=x.device, dtype=torch.bfloat16)
    rt.mul_sum_add_sm100(x_bs, h_pre, zero_res, f_pre, n)

    return (
        f_pre.reshape(b, s, C),
        h_post.reshape(b, s, n),
        comb.reshape(b, s, n, n),
    )


def _run_hc_post_with_kernel(x, residual, post, comb, n):
    b, s, C = x.shape
    bs = b * s
    out = torch.empty(bs, n, C, device=x.device, dtype=torch.bfloat16)
    rt.mHC_mul_sum_add_with_outer(
        residual.reshape(bs, n, C).contiguous(),
        x.reshape(bs, C).contiguous(),
        comb.reshape(bs, n, n).contiguous(),
        post.reshape(bs, n).contiguous(),
        out,
        n,
    )
    return out.reshape(b, s, n, C)


@pytest.mark.parametrize("b,s,n,C", [
    (1, 1, 4, 128),
    (1, 4, 4, 1024),
    (2, 4, 4, 4096),
])
def test_hc_pre_pipeline(b, s, n, C):
    gen = torch.Generator(device="cuda").manual_seed(500 + b + s + n + C)
    nC = n * C
    mix_hc = n * n + 2 * n

    x = torch.randn(b, s, n, C, device="cuda", dtype=torch.bfloat16, generator=gen)
    hc_fn = torch.randn(mix_hc, nC, device="cuda", dtype=torch.float32, generator=gen) * 0.02
    hc_scale = torch.randn(3, device="cuda", dtype=torch.float32, generator=gen)
    hc_base = torch.randn(mix_hc, device="cuda", dtype=torch.float32, generator=gen) * 0.1

    sinkhorn_iters = 20
    hc_eps = 1e-9
    norm_eps = 1e-6

    f_pre, post, comb = _run_hc_pre_with_kernels(
        x, hc_fn, hc_scale, hc_base, n, sinkhorn_iters, hc_eps, norm_eps,
    )
    f_pre_ref, post_ref, comb_ref = hc_pre_reference(
        x, hc_fn, hc_scale, hc_base, n, sinkhorn_iters, hc_eps, norm_eps,
    )

    # post and comb depend only on K2+K3 outputs (fp32) -> tight tolerances
    torch.testing.assert_close(post, post_ref, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(comb, comb_ref, rtol=2e-2, atol=2e-2)
    # f_pre goes through bf16 GEMV + bf16 reduction -> looser
    torch.testing.assert_close(f_pre, f_pre_ref, rtol=5e-2, atol=5e-2)


@pytest.mark.parametrize("b,s,n,C", [
    (1, 1, 4, 128),
    (1, 4, 4, 1024),
    (2, 4, 4, 4096),
])
def test_hc_post_pipeline(b, s, n, C):
    gen = torch.Generator(device="cuda").manual_seed(600 + b + s + n + C)
    bs = b * s
    x = torch.randn(b, s, C, device="cuda", dtype=torch.bfloat16, generator=gen)
    residual = torch.randn(b, s, n, C, device="cuda", dtype=torch.bfloat16, generator=gen)
    post = torch.rand(b, s, n, device="cuda", dtype=torch.float32, generator=gen)
    comb = torch.rand(b, s, n, n, device="cuda", dtype=torch.float32, generator=gen)
    comb = comb / comb.sum(-1, keepdim=True)

    out = _run_hc_post_with_kernel(x, residual, post, comb, n)
    ref = hc_post_reference(x, residual, post, comb)

    torch.testing.assert_close(out, ref, rtol=3e-2, atol=3e-2)


@pytest.mark.parametrize("b,s,n,C", [(1, 4, 4, 1024)])
def test_full_block_roundtrip(b, s, n, C):
    """hc_pre then hc_post composed (with a stand-in identity attn/ffn)
    should match the equivalent PyTorch composition."""
    gen = torch.Generator(device="cuda").manual_seed(700 + b + s + n + C)
    nC = n * C
    mix_hc = n * n + 2 * n

    x = torch.randn(b, s, n, C, device="cuda", dtype=torch.bfloat16, generator=gen)
    hc_fn = torch.randn(mix_hc, nC, device="cuda", dtype=torch.float32, generator=gen) * 0.02
    hc_scale = torch.randn(3, device="cuda", dtype=torch.float32, generator=gen)
    hc_base = torch.randn(mix_hc, device="cuda", dtype=torch.float32, generator=gen) * 0.1

    f_pre, post, comb = _run_hc_pre_with_kernels(
        x, hc_fn, hc_scale, hc_base, n, 20, 1e-9, 1e-6
    )
    # stand-in attn/ffn: identity (operate on f_pre directly)
    out = _run_hc_post_with_kernel(f_pre, x, post, comb, n)

    f_pre_ref, post_ref, comb_ref = hc_pre_reference(
        x, hc_fn, hc_scale, hc_base, n, 20, 1e-9, 1e-6
    )
    out_ref = hc_post_reference(f_pre_ref, x, post_ref, comb_ref)
    torch.testing.assert_close(out, out_ref, rtol=5e-2, atol=5e-2)
