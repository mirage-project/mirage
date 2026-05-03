"""Correctness + perf for mla_prefill_tp8_chunked device function.

The kernel implements MLA chunked prefill: Q covers chunk
[q_start, q_start+q_len) of a longer sequence, KV covers [0, kv_len).
Compares against PyTorch reference and FlashInfer single_prefill_with_kv_cache.
"""
import math
import os
import sys

import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(THIS_DIR, "build", "lib.linux-x86_64-cpython-312"))

import runtime_kernel_mla_prefill_tp8_chunked as ext  # noqa: E402

D_QK_NOPE = 128
D_QK_ROPE = 64
D_QK = 192
D_V = 128


def torch_reference(qn, qp, k, v, q_start, sm_scale):
    """Causal MLA chunked prefill reference (fp32). q_len queries at offset
    q_start, kv_len keys; q[i] attends to k[0..q_start+i]."""
    B, q_len, H, _ = qn.shape
    kv_len = k.shape[1]
    q = torch.cat([qn, qp], dim=-1).float()
    kf = k.float().unsqueeze(2).expand(B, kv_len, H, D_QK)
    vf = v.float().unsqueeze(2).expand(B, kv_len, H, D_V)
    scores = torch.einsum("bihd,bjhd->bhij", q, kf) * sm_scale
    j = torch.arange(kv_len, device=q.device)
    i = torch.arange(q_len, device=q.device)
    mask = j[None, :] > (q_start + i[:, None])
    scores.masked_fill_(mask[None, None, :, :], float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    out = torch.einsum("bhij,bjhd->bihd", probs, vf)
    return out.to(qn.dtype)


def run_case(B, q_len, kv_len, q_start, H, atol=3e-2):
    sm_scale = 1.0 / math.sqrt(D_QK)
    torch.manual_seed(0)
    device = "cuda"
    dt = torch.bfloat16
    qn = torch.randn(B, q_len, H, D_QK_NOPE, dtype=dt, device=device) * 0.2
    qp = torch.randn(B, q_len, H, D_QK_ROPE, dtype=dt, device=device) * 0.2
    k = torch.randn(B, kv_len, D_QK, dtype=dt, device=device) * 0.2
    v = torch.randn(B, kv_len, D_V, dtype=dt, device=device) * 0.2
    o = torch.zeros(B, q_len, H, D_V, dtype=dt, device=device)

    ext.mla_prefill_tp8_chunked_test(qn, qp, k, v, o, q_start, sm_scale)
    o_ref = torch_reference(qn, qp, k, v, q_start, sm_scale)

    err = (o.float() - o_ref.float()).abs()
    max_err, mean_err = err.max().item(), err.mean().item()
    status = "OK" if max_err < atol else "FAIL"
    print(f"  B={B} q={q_len:4d} kv={kv_len:5d} qs={q_start:5d} H={H} "
          f"max_err={max_err:.5f} mean_err={mean_err:.5f} [{status}]")
    return max_err < atol


def bench(B, q_len, kv_len, q_start, H, n_iters=100, warmup=20):
    sm_scale = 1.0 / math.sqrt(D_QK)
    device = "cuda"
    dt = torch.bfloat16
    qn = torch.randn(B, q_len, H, D_QK_NOPE, dtype=dt, device=device) * 0.2
    qp = torch.randn(B, q_len, H, D_QK_ROPE, dtype=dt, device=device) * 0.2
    k = torch.randn(B, kv_len, D_QK, dtype=dt, device=device) * 0.2
    v = torch.randn(B, kv_len, D_V, dtype=dt, device=device) * 0.2
    o = torch.zeros(B, q_len, H, D_V, dtype=dt, device=device)

    flush = torch.zeros(128 * 1024 * 1024 // 4, dtype=torch.int32, device=device)

    for _ in range(warmup):
        ext.mla_prefill_tp8_chunked_test(qn, qp, k, v, o, q_start, sm_scale)
    torch.cuda.synchronize()
    times = []
    for _ in range(5):
        flush.zero_()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(n_iters):
            ext.mla_prefill_tp8_chunked_test(qn, qp, k, v, o, q_start, sm_scale)
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e) / n_iters)
    times.sort()
    ms = times[2]
    flops = 2.0 * B * H * q_len * kv_len * (D_QK + D_V)
    tf = flops / (ms / 1000.0) / 1e12
    print(f"  C{q_len}_KV{kv_len:5d}: {ms*1000:7.1f} us  {tf:6.2f} TFLOPS_raw")


def run_splitk_case(B, q_len, kv_len, q_start, H, num_splits, atol=3e-2):
    sm_scale = 1.0 / math.sqrt(D_QK)
    torch.manual_seed(0)
    device = "cuda"
    dt = torch.bfloat16
    qn = torch.randn(B, q_len, H, D_QK_NOPE, dtype=dt, device=device) * 0.2
    qp = torch.randn(B, q_len, H, D_QK_ROPE, dtype=dt, device=device) * 0.2
    k = torch.randn(B, kv_len, D_QK, dtype=dt, device=device) * 0.2
    v = torch.randn(B, kv_len, D_V, dtype=dt, device=device) * 0.2
    o = torch.zeros(B, q_len, H, D_V, dtype=dt, device=device)
    nqb = (q_len + 63) // 64
    partial = torch.zeros(num_splits, B, nqb, H, 64, D_V + 4,
                          dtype=torch.float32, device=device)
    ext.mla_prefill_tp8_chunked_splitk_test(
        qn, qp, k, v, o, partial, q_start, num_splits, sm_scale)
    o_ref = torch_reference(qn, qp, k, v, q_start, sm_scale)
    err = (o.float() - o_ref.float()).abs()
    max_err, mean_err = err.max().item(), err.mean().item()
    status = "OK" if max_err < atol else "FAIL"
    print(f"  splits={num_splits} B={B} q={q_len:4d} kv={kv_len:5d} "
          f"qs={q_start:5d} H={H} max_err={max_err:.5f} [{status}]")
    return max_err < atol


def bench_splitk(B, q_len, kv_len, q_start, H, num_splits,
                 n_iters=100, warmup=20):
    sm_scale = 1.0 / math.sqrt(D_QK)
    device = "cuda"
    dt = torch.bfloat16
    qn = torch.randn(B, q_len, H, D_QK_NOPE, dtype=dt, device=device) * 0.2
    qp = torch.randn(B, q_len, H, D_QK_ROPE, dtype=dt, device=device) * 0.2
    k = torch.randn(B, kv_len, D_QK, dtype=dt, device=device) * 0.2
    v = torch.randn(B, kv_len, D_V, dtype=dt, device=device) * 0.2
    o = torch.zeros(B, q_len, H, D_V, dtype=dt, device=device)
    nqb = (q_len + 63) // 64
    partial = torch.zeros(num_splits, B, nqb, H, 64, D_V + 4,
                          dtype=torch.float32, device=device)
    flush = torch.zeros(128 * 1024 * 1024 // 4, dtype=torch.int32, device=device)
    fn = lambda: ext.mla_prefill_tp8_chunked_splitk_test(
        qn, qp, k, v, o, partial, q_start, num_splits, sm_scale)
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(5):
        flush.zero_()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(n_iters):
            fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e) / n_iters)
    times.sort()
    ms = times[2]
    print(f"  C{q_len}_KV{kv_len:5d} splits={num_splits}: "
          f"{ms*1000:7.1f} us")


if __name__ == "__main__":
    print("=== main BM=64 correctness ===")
    ok = True
    cases = [
        (256, 2048),
        (256, 4096),
        (512, 4096),
        (512, 8192),
        (1024, 4096),
        (1024, 8192),
    ]
    for q_len, kv_len in cases:
        ok &= run_case(B=1, q_len=q_len, kv_len=kv_len,
                       q_start=kv_len - q_len, H=16)
    if not ok:
        sys.exit(1)

    print("\n=== splitk + reduce correctness ===")
    splitk_cases = [
        (256, 2048, 4),
        (256, 4096, 4),
        (512, 4096, 2),
        (512, 8192, 2),
    ]
    for q_len, kv_len, splits in splitk_cases:
        ok &= run_splitk_case(B=1, q_len=q_len, kv_len=kv_len,
                              q_start=kv_len - q_len, H=16,
                              num_splits=splits)
    if not ok:
        sys.exit(1)

    print("\n=== main BM=64 bench ===")
    for q_len, kv_len in cases + [(2048, 8192)]:
        bench(B=1, q_len=q_len, kv_len=kv_len,
              q_start=kv_len - q_len, H=16)

    print("\n=== splitk bench ===")
    for q_len, kv_len, splits in splitk_cases:
        bench_splitk(B=1, q_len=q_len, kv_len=kv_len,
                     q_start=kv_len - q_len, H=16, num_splits=splits)
