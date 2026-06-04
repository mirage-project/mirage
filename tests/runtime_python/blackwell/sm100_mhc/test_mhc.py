"""Standalone tests for the new mHC kernels (K2, K4, K5, K3 reuse)
and the end-to-end hc_pre / hc_post pipeline."""
import os
import sys

import pytest
import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
VLLM_DIR = os.path.join(THIS_DIR, "vllm")
if VLLM_DIR not in sys.path:
    sys.path.insert(0, VLLM_DIR)

from utils import (
    hc_post_reference,
    hc_pre_reference,
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
# K3: sinkhorn (existing kernel; smoke test in this wrapper)
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("num_tokens", [1, 16, 1024])
def test_k3_sinkhorn(num_tokens):
    gen = torch.Generator(device="cuda").manual_seed(200 + num_tokens)
    res_logits = torch.randn(num_tokens, 4, 4, device="cuda", dtype=torch.float32, generator=gen)
    out = torch.empty_like(res_logits)
    rt.sinkhorn_sm100(res_logits, out, repeat=20, eps=1e-9)
    ref = sinkhorn_knopp_torch(res_logits, repeat=20, eps=1e-9)
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-5)




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
def test_mhc_post(num_tokens, n, c):
    gen = torch.Generator(device="cuda").manual_seed(400 + num_tokens + n + c)
    residual = torch.randn(num_tokens, n, c, device="cuda", dtype=torch.bfloat16, generator=gen)
    x = torch.randn(num_tokens, c, device="cuda", dtype=torch.bfloat16, generator=gen)
    comb = torch.rand(num_tokens, n, n, device="cuda", dtype=torch.float32, generator=gen)
    comb = comb / comb.sum(-1, keepdim=True)  # row-stochastic, similar to sinkhorn output
    post = torch.rand(num_tokens, n, device="cuda", dtype=torch.float32, generator=gen)
    out = torch.empty(num_tokens, n, c, device="cuda", dtype=torch.bfloat16)

    rt.mhc_post(residual, x, comb, post, out, n)

    ref = k5_reference(residual, x, comb, post)
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)



# -----------------------------------------------------------------------------
# End-to-end hc_post pipeline (uses mhc_post)
# -----------------------------------------------------------------------------

def _run_hc_post_with_kernel(x, residual, post, comb, n):
    b, s, C = x.shape
    bs = b * s
    out = torch.empty(bs, n, C, device=x.device, dtype=torch.bfloat16)
    rt.mhc_post(
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
