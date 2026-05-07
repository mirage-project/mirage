import os
import sys

import pytest
import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROFILE_DIR = os.path.join(THIS_DIR, "profile")
if PROFILE_DIR not in sys.path:
    sys.path.insert(0, PROFILE_DIR)

from utils import sinkhorn_knopp_torch

import runtime_kernel_blackwell_sinkhorn as runtime_kernel_blackwell

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)


@pytest.mark.parametrize(
    "num_tokens,repeat",
    [(1, 20), (7, 20), (1024, 20), (4096, 10)],
)
def test_sinkhorn_matches_torch(num_tokens, repeat):
    generator = torch.Generator(device="cuda").manual_seed(1000 + num_tokens)
    comb_res_mix = torch.randn(
        (num_tokens, 4, 4),
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    )
    comb_res_mix_out = torch.empty_like(comb_res_mix)

    runtime_kernel_blackwell.sinkhorn_sm100(
        comb_res_mix, comb_res_mix_out, repeat=repeat, eps=1e-9,
    )
    ref = sinkhorn_knopp_torch(comb_res_mix, repeat=repeat, eps=1e-9)

    torch.testing.assert_close(comb_res_mix_out, ref, rtol=1e-4, atol=1e-5)


def test_sinkhorn_is_doubly_stochastic_after_projection():
    comb_res_mix = torch.randn((64, 4, 4), device="cuda", dtype=torch.float32)
    comb_res_mix_out = torch.empty_like(comb_res_mix)

    runtime_kernel_blackwell.sinkhorn_sm100(
        comb_res_mix, comb_res_mix_out, repeat=20, eps=1e-9,
    )

    ones = torch.ones((64, 4), device="cuda", dtype=torch.float32)
    torch.testing.assert_close(
        comb_res_mix_out.sum(dim=-1), ones, rtol=1e-4, atol=1e-4
    )
    torch.testing.assert_close(
        comb_res_mix_out.sum(dim=-2), ones, rtol=1e-5, atol=1e-5
    )


def test_sinkhorn_rejects_non_4x4():
    comb_res_mix = torch.randn((2, 3, 3), device="cuda", dtype=torch.float32)
    comb_res_mix_out = torch.empty_like(comb_res_mix)

    with pytest.raises(RuntimeError, match="must be 4x4"):
        runtime_kernel_blackwell.sinkhorn_sm100(
            comb_res_mix, comb_res_mix_out, repeat=20, eps=1e-9,
        )
