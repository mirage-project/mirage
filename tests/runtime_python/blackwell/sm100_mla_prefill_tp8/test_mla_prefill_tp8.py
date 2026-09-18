"""Correctness + perf for mla_prefill_tp8 device function (MPK-adapted kernel).

Compares the MPK-adapted kernel (wrapped by a thin __global__ that forwards
blockIdx) against a PyTorch reference MLA prefill computation.
"""

import math
import os
import sys
import time

import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(THIS_DIR, "build", "lib.linux-x86_64-cpython-312"))

import runtime_kernel_mla_prefill_tp8 as ext  # noqa: E402


# ─ Shape constants (matches standalone kernel) ─────────────────────────
D_QK_NOPE = 128
D_QK_ROPE = 64
D_QK = 192
D_V = 128


def reference_mla_prefill(q_nope, q_pe, k, v, sm_scale):
    """MLA prefill reference in fp32 with causal mask.

    q_nope: [B, S, H, 128]
    q_pe:   [B, S, H, 64]
    k:      [B, S, 192]   (nope+rope concat)
    v:      [B, S, 128]
    returns o: [B, S, H, 128]
    """
    B, S, H, _ = q_nope.shape
    q = torch.cat([q_nope, q_pe], dim=-1).float()  # [B,S,H,192]
    kf = k.float().unsqueeze(2).expand(B, S, H, D_QK)  # [B,S,H,192]
    vf = v.float().unsqueeze(2).expand(B, S, H, D_V)   # [B,S,H,128]

    # scores[b,i,h,j] = sum_d q[b,i,h,d] * k[b,j,d]
    scores = torch.einsum("bihd,bjhd->bhij", q, kf) * sm_scale  # [B,H,S,S]
    mask = torch.triu(
        torch.ones(S, S, device=q.device, dtype=torch.bool), diagonal=1
    )
    scores.masked_fill_(mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    out = torch.einsum("bhij,bjhd->bihd", probs, vf)  # [B,S,H,128]
    return out.to(q_nope.dtype)


def run_case(B: int, S: int, H: int, atol: float = 3e-2):
    sm_scale = 1.0 / math.sqrt(D_QK)

    torch.manual_seed(0)
    device = "cuda"
    q_nope = torch.randn(B, S, H, D_QK_NOPE, dtype=torch.bfloat16, device=device) * 0.2
    q_pe = torch.randn(B, S, H, D_QK_ROPE, dtype=torch.bfloat16, device=device) * 0.2
    k = torch.randn(B, S, D_QK, dtype=torch.bfloat16, device=device) * 0.2
    v = torch.randn(B, S, D_V, dtype=torch.bfloat16, device=device) * 0.2
    o = torch.zeros(B, S, H, D_V, dtype=torch.bfloat16, device=device)

    ext.mla_prefill_tp8_test(q_nope, q_pe, k, v, o, sm_scale)
    o_ref = reference_mla_prefill(q_nope, q_pe, k, v, sm_scale)

    err = (o.float() - o_ref.float()).abs()
    max_err = err.max().item()
    mean_err = err.mean().item()
    status = "OK" if max_err < atol else "FAIL"
    print(
        f"B={B} S={S} H={H} max_abs_err={max_err:.5f} "
        f"mean_abs_err={mean_err:.5f} [{status}]"
    )
    return max_err < atol


def bench(B: int, S: int, H: int, n_iters: int = 50, warmup: int = 10):
    sm_scale = 1.0 / math.sqrt(D_QK)
    device = "cuda"
    q_nope = torch.randn(B, S, H, D_QK_NOPE, dtype=torch.bfloat16, device=device) * 0.2
    q_pe = torch.randn(B, S, H, D_QK_ROPE, dtype=torch.bfloat16, device=device) * 0.2
    k = torch.randn(B, S, D_QK, dtype=torch.bfloat16, device=device) * 0.2
    v = torch.randn(B, S, D_V, dtype=torch.bfloat16, device=device) * 0.2
    o = torch.zeros(B, S, H, D_V, dtype=torch.bfloat16, device=device)

    for _ in range(warmup):
        ext.mla_prefill_tp8_test(q_nope, q_pe, k, v, o, sm_scale)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iters):
        ext.mla_prefill_tp8_test(q_nope, q_pe, k, v, o, sm_scale)
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / n_iters
    flops = 2.0 * B * H * S * S * (D_QK + D_V)
    tf = flops / (ms / 1000.0) / 1e12
    print(f"B={B} S={S} H={H} per-iter={ms*1000:.1f} us  {tf:.2f} TFLOPS")


if __name__ == "__main__":
    # Correctness on small shapes where the reference is tractable
    ok = True
    for S in [128, 512, 1024]:
        ok &= run_case(B=1, S=S, H=16)
    if not ok:
        sys.exit(1)
    print()
    # Perf sweep
    for S in [1024, 2048, 4096]:
        bench(B=1, S=S, H=16)
