"""PA kernel correctness: dflash_norm_rope_sm100 vs PyTorch reference.

Run after: python setup.py build_ext --inplace
Run: CUDA_VISIBLE_DEVICES=2 python test_norm_rope.py
"""
import os
import sys

import torch
import runtime_kernel_dflash as rk

sys.path.insert(0, os.path.dirname(__file__))
from pytorch_reference import dflash_norm_rope, EPS  # noqa: E402

device, dtype = "cuda", torch.bfloat16
D = 128


def run(N, NH):
    torch.manual_seed(1)
    x = torch.randn(N, NH, D, dtype=dtype, device=device)
    w = torch.randn(D, dtype=dtype, device=device)
    # NeoX duplicated cos/sin (first half == second half)
    ang = torch.randn(N, D // 2, device=device)
    cos = torch.cat([ang.cos(), ang.cos()], dim=-1).to(dtype)
    sin = torch.cat([ang.sin(), ang.sin()], dim=-1).to(dtype)
    o = torch.zeros(N, NH, D, dtype=dtype, device=device)

    rk.dflash_norm_rope(x, w, cos, sin, o, EPS)
    ref = dflash_norm_rope(x, w, cos, sin, EPS)
    err = (o.float() - ref.float()).abs().max().item()
    rel = err / max(ref.float().abs().max().item(), 1e-6)
    ok = rel < 0.02
    print(f"N={N} NH={NH}: maxerr {err:.4f} relmax {rel:.4f} {'OK' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    res = [run(8, 64), run(24, 8), run(2108, 8), run(1, 64)]
    print("ALL PASSED" if all(res) else "SOME FAILED")
    assert all(res)
