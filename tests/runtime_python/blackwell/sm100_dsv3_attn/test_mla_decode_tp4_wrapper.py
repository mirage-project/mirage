"""
Kernel-wrapper correctness test for the DeepSeek-V3 TP=4 MLA decode kernel
(mla_mtp_decode_tp4_sm100.cuh).

Calls the device task functions directly (via runtime_kernel_mla_decode_tp4)
with kv_len passed as an explicit int, so we reach a genuine DECODE state
(kv_len=256, q_len=1) that test_mode cannot produce.

The kernel output `attn_out` (final, reduced) is compared against the verified
direct full-attention reference in pytorch_reference.mla_decode_full_ref.

Build:
    cd tests/runtime_python/blackwell/sm100_dsv3_attn
    python setup_mla_decode_tp4.py build_ext --inplace

Run:
    python test_mla_decode_tp4_wrapper.py [--kv-len 256] [--batch 1] [--q-len 1]
"""

import argparse
import os

import torch

torch.set_printoptions(sci_mode=False)

import runtime_kernel_mla_decode_tp4 as ext
from pytorch_reference import (
    NUM_HEADS,
    D_K,
    D_V,
    deepseek_softmax_scale,
    mla_decode_full_ref,
)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.reshape(-1).float()
    b = b.reshape(-1).float()
    denom = a.norm() * b.norm()
    if denom.item() == 0.0:
        return 0.0
    return (torch.dot(a, b) / denom).item()


def run_case(batch_size: int, q_len: int, kv_len: int, seed: int = 42) -> bool:
    print(f"\n{'='*64}")
    print(f"DeepSeek-V3 TP=4 MLA decode  (kernel-wrapper)")
    print(f"  B={batch_size}, q_len={q_len}, kv_len={kv_len}, "
          f"NUM_HEADS={NUM_HEADS}, D_K={D_K}, D_V={D_V}")
    print(f"{'='*64}")

    device = "cuda"
    torch.manual_seed(seed)

    # Q: [B * q_len * NUM_HEADS, D_K] bf16  (row h = head h, head-major)
    # KV:[B * kv_len, D_K]            bf16  (V = first D_V of each row)
    # Scaled down like the standalone harness so bf16 scores stay well-behaved.
    q = (torch.randn(batch_size * q_len * NUM_HEADS, D_K,
                     device=device, dtype=torch.bfloat16) * 0.1)
    kv = (torch.randn(batch_size * kv_len, D_K,
                      device=device, dtype=torch.bfloat16) * 0.1)

    # attn_out: [B * q_len, NUM_HEADS * D_V] bf16 head-major
    attn_out = torch.zeros(batch_size * q_len, NUM_HEADS * D_V,
                           device=device, dtype=torch.bfloat16)

    ss = deepseek_softmax_scale()

    ext.mla_decode_tp4_test(
        q, kv, attn_out, batch_size, q_len, kv_len, ss
    )

    out = attn_out.reshape(batch_size, q_len, NUM_HEADS, D_V)

    ref = mla_decode_full_ref(q, kv, batch_size, q_len, kv_len)  # [B,q,H,D_V] bf16

    diff = (out.float() - ref.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    cos = cosine(out, ref)

    print(f"  cos:           {cos:.6f}")
    print(f"  max abs diff:  {max_diff:.6f}")
    print(f"  mean abs diff: {mean_diff:.6f}")

    # Sanity: output must not be all-zero (would falsely pass cos via 0/0 guard).
    out_norm = out.float().norm().item()
    ref_norm = ref.float().norm().item()
    print(f"  ||out||={out_norm:.4f}  ||ref||={ref_norm:.4f}")

    ok = (cos > 0.99) and (out_norm > 1e-3)
    try:
        torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)
        tol_ok = True
    except AssertionError as e:
        tol_ok = False
        print(f"  [assert_close rtol/atol 2e-2 failed]\n{e}")

    verdict = "PASS" if (ok and tol_ok) else "FAIL"
    print(f"  ==> {verdict} (cos>0.99: {cos > 0.99}, tol2e-2: {tol_ok})")
    return ok and tol_ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--q-len", type=int, default=1)
    parser.add_argument("--kv-len", type=int, default=256)
    args = parser.parse_args()

    passed = run_case(args.batch, args.q_len, args.kv_len)
    raise SystemExit(0 if passed else 1)
