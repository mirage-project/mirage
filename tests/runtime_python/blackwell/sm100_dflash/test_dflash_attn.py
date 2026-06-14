"""PA kernel correctness: dflash_attention_sm100 core vs PyTorch reference.

Random pre-normed/roped q,k,v. Tests non-causal full attention and the
sliding-window mask. Run after: python setup.py build_ext --inplace
Run: CUDA_VISIBLE_DEVICES=2 python test_dflash_attn.py
"""
import os
import sys

import torch
import runtime_kernel_dflash as rk

sys.path.insert(0, os.path.dirname(__file__))
from pytorch_reference import dflash_attention_core  # noqa: E402

device, dtype = "cuda", torch.bfloat16
NQ, NKV, D = 64, 8, 128


def run_case(B, ctx_len, sliding_window, seed=0):
    torch.manual_seed(seed)
    T = ctx_len + B
    q = torch.randn(B, NQ, D, dtype=dtype, device=device)
    k = torch.randn(T, NKV, D, dtype=dtype, device=device)
    v = torch.randn(T, NKV, D, dtype=dtype, device=device)
    o = torch.zeros(B, NQ, D, dtype=dtype, device=device)

    ck = k[:ctx_len].contiguous(); cv = v[:ctx_len].contiguous()
    bk = k[ctx_len:].contiguous(); bv = v[ctx_len:].contiguous()
    rk.dflash_attn(q, ck, cv, bk, bv, o, sliding_window)
    ref = dflash_attention_core(q, k, v, sliding_window, NQ, NKV, D)

    err = (o.float() - ref.float()).abs().max().item()
    refmax = ref.float().abs().max().item()
    rel = err / max(refmax, 1e-6)
    ok = rel < 0.02
    print(f"B={B} ctx_len={ctx_len} sw={sliding_window}: maxerr {err:.4f} "
          f"relmax {rel:.4f} {'OK' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    results = []
    results.append(run_case(8, 16, 0))         # full non-causal, no window
    results.append(run_case(8, 16, 2048))      # window larger than T -> no masking
    results.append(run_case(8, 64, 0))
    results.append(run_case(8, 2100, 0))       # longer ctx, full
    results.append(run_case(8, 2100, 2048))    # window actually masks
    results.append(run_case(1, 16, 0))
    print("ALL PASSED" if all(results) else "SOME FAILED")
    assert all(results)
