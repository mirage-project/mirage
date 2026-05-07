"""Benchmark: MPK MLA chunked prefill (per-head) vs FlashInfer FA2 ragged.

Same workload (chunked prefill: q_len queries at offset q_start, kv_len keys),
per-head DeepSeek MLA dims: H=16 heads × D_QK=192 × D_V=128.
"""
import math
import os
import sys

import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(THIS_DIR, "build", "lib.linux-x86_64-cpython-312"))

import runtime_kernel_mla_prefill_tp8_chunked as ext  # noqa: E402
from flashinfer.prefill import BatchPrefillWithRaggedKVCacheWrapper

D_QK_NOPE = 128
D_QK_ROPE = 64
D_QK = 192
D_V = 128


def bench_main(qn, qp, kn, kr, v, o, q_start, sm_scale, n_iters=100, warmup=20):
    flush = torch.zeros(128 * 1024 * 1024 // 4, dtype=torch.int32, device=qn.device)
    fn = lambda: ext.mla_prefill_tp8_chunked_test(
        qn, qp, kn, kr, v, o, q_start, sm_scale)
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
    return times[2]


def bench_splitk(qn, qp, kn, kr, v, o, partial, q_start, num_splits, sm_scale,
                 n_iters=100, warmup=20):
    flush = torch.zeros(128 * 1024 * 1024 // 4, dtype=torch.int32, device=qn.device)
    fn = lambda: ext.mla_prefill_tp8_chunked_splitk_test(
        qn, qp, kn, kr, v, o, partial, q_start, num_splits, sm_scale)
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
    return times[2]


def bench_flashinfer(B, q_len, kv_len, q_start, H, sm_scale,
                     n_iters=100, warmup=20):
    device = "cuda"
    dt = torch.bfloat16
    # Concatenated MLA layout: per token we have D_QK=192 (nope+rope on the
    # query side; nope+rope on the key side — K_rope shared across heads but
    # FlashInfer treats each head as independent, so we replicate K_rope).
    q = torch.randn(B * q_len, H, D_QK, dtype=dt, device=device) * 0.2
    k = torch.randn(B * kv_len, H, D_QK, dtype=dt, device=device) * 0.2
    v = torch.randn(B * kv_len, H, D_V, dtype=dt, device=device) * 0.2

    qo_indptr = torch.tensor([0, q_len * B], dtype=torch.int32, device=device)
    kv_indptr = torch.tensor([0, kv_len * B], dtype=torch.int32, device=device)
    workspace = torch.zeros(256 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = BatchPrefillWithRaggedKVCacheWrapper(workspace, kv_layout="NHD")

    # Custom mask: q[i] attends to k[0..q_start+i].
    if q_start == kv_len - q_len:
        causal = True
        custom_mask = None
    else:
        # General chunked mask
        i = torch.arange(q_len, device=device)
        j = torch.arange(kv_len, device=device)
        mask2d = (j[None, :] <= (q_start + i[:, None]))
        custom_mask = mask2d.flatten().to(torch.bool)
        causal = False

    wrapper.plan(
        qo_indptr=qo_indptr,
        kv_indptr=kv_indptr,
        num_qo_heads=H,
        num_kv_heads=H,
        head_dim_qk=D_QK,
        head_dim_vo=D_V,
        causal=causal,
        custom_mask=custom_mask,
        sm_scale=sm_scale,
        q_data_type=dt,
        kv_data_type=dt,
    )

    flush = torch.zeros(128 * 1024 * 1024 // 4, dtype=torch.int32, device=device)
    for _ in range(warmup):
        wrapper.run(q, k, v)
    torch.cuda.synchronize()
    times = []
    for _ in range(5):
        flush.zero_()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(n_iters):
            wrapper.run(q, k, v)
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e) / n_iters)
    times.sort()
    return times[2]


def main():
    H = 16
    B = 1
    sm_scale = 1.0 / math.sqrt(D_QK)
    cases = [
        (256, 2048),
        (256, 4096),
        (512, 4096),
        (512, 8192),
        (1024, 4096),
        (1024, 8192),
        (2048, 8192),
    ]

    print(f"{'q_len':>6} {'kv_len':>6} | {'main us':>8} {'splitk':>10}  "
          f"{'best us':>8} {'FI us':>8}  speedup  {'best TF':>8} {'FI TF':>8}")
    print("-" * 88)
    for q_len, kv_len in cases:
        q_start = kv_len - q_len  # last chunk -> causal in FI
        device = "cuda"
        dt = torch.bfloat16
        torch.manual_seed(0)
        qn = torch.randn(B, q_len, H, D_QK_NOPE, dtype=dt, device=device) * 0.2
        qp = torch.randn(B, q_len, H, D_QK_ROPE, dtype=dt, device=device) * 0.2
        kn = torch.randn(B, kv_len, H, D_QK_NOPE, dtype=dt, device=device) * 0.2
        kr = torch.randn(B, kv_len, 1, D_QK_ROPE, dtype=dt, device=device) * 0.2
        v = torch.randn(B, kv_len, H, D_V, dtype=dt, device=device) * 0.2
        o = torch.zeros(B, q_len, H, D_V, dtype=dt, device=device)

        ms_main = bench_main(qn, qp, kn, kr, v, o, q_start, sm_scale)

        # Try splitk for small chunks (where the main grid under-fills the GPU).
        # Pick num_splits so total grid blocks ~ #SMs (148).
        nqb = (q_len + 63) // 64
        target = 148
        cur_grid = H * nqb * B
        best_split_ms = None
        best_split_n = 0
        for ns in [2, 4, 8]:
            if cur_grid * ns > target * 2 and cur_grid >= target:
                continue
            partial = torch.zeros(ns, B, nqb, H, 64, D_V + 4,
                                  dtype=torch.float32, device=device)
            ms = bench_splitk(qn, qp, kn, kr, v, o, partial, q_start, ns, sm_scale)
            if best_split_ms is None or ms < best_split_ms:
                best_split_ms = ms
                best_split_n = ns

        ms_fi = bench_flashinfer(B, q_len, kv_len, q_start, H, sm_scale)

        ms_best = min(ms_main, best_split_ms) if best_split_ms else ms_main
        flops = 2.0 * B * H * q_len * kv_len * (D_QK + D_V)
        tf_best = flops / (ms_best / 1000.0) / 1e12
        tf_fi = flops / (ms_fi / 1000.0) / 1e12
        sp = ms_fi / ms_best
        sk_str = (f"{best_split_ms*1000:.1f}/x{best_split_n}"
                  if best_split_ms else "    -")
        print(f"{q_len:>6} {kv_len:>6} | {ms_main*1000:>7.1f} {sk_str:>10}  "
              f"{ms_best*1000:>7.1f} {ms_fi*1000:>7.1f}  "
              f"{sp:>5.2f}x  {tf_best:>7.1f} {tf_fi:>7.1f}")


if __name__ == "__main__":
    main()
