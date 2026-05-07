"""MPK end-to-end smoke + bench for fp8_gemm_dense_mediumm_layer.

Same as test_mpk_smoke.py but uses the M=512..2048 sweet-spot kernel.
"""
import math
import os
import sys

import torch
import mirage
from mirage.core import bfloat16, float32, float8_e4m3
from mirage.mpk.persistent_kernel import PersistentKernel


def torch_reference(A_fp8, B_fp8, sa, sb, M, N, K):
    BLK = 128
    A = A_fp8.float()
    B = B_fp8.float()
    out = torch.zeros(M, N, dtype=torch.float32, device=A.device)
    nk = K // BLK
    for ki in range(nk):
        a_blk = A[:, ki * BLK:(ki + 1) * BLK]
        b_blk = B[:, ki * BLK:(ki + 1) * BLK]
        partial = a_blk @ b_blk.T
        sa_col = sa[:, ki:ki + 1]
        sb_row = sb[:, ki].repeat_interleave(BLK)
        out += partial * sa_col * sb_row[None, :]
    return out.to(torch.bfloat16)


def main():
    M = int(os.environ.get("M", 1024))
    K = int(os.environ.get("K", 512))
    N = int(os.environ.get("N", 4096))
    print(f"shape M={M} K={K} N={N}")

    device = "cuda"
    torch.manual_seed(0)

    A_bf16 = (torch.randn(M, K, dtype=torch.bfloat16, device=device) * 0.5)
    B_bf16 = (torch.randn(N, K, dtype=torch.bfloat16, device=device) * 0.5)
    A_fp8 = A_bf16.to(torch.float8_e4m3fn)
    B_fp8 = B_bf16.to(torch.float8_e4m3fn)
    sa = (0.5 + torch.rand(M, K // 128, dtype=torch.float32, device=device) * 0.5).contiguous()
    sb = (0.5 + torch.rand(N // 128, K // 128, dtype=torch.float32, device=device) * 0.5).contiguous()
    C = torch.zeros(M, N, dtype=torch.bfloat16, device=device)

    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params.update(
        test_mode=True,
        num_workers=num_workers,
        num_local_schedulers=num_schedulers,
        max_num_batched_tokens=max(M, 1),
        max_num_batched_requests=1,
        max_seq_length=1,
    )
    pk = PersistentKernel(**params)

    A_dt = pk.attach_input(A_fp8, name="A_fp8")
    B_dt = pk.attach_input(B_fp8, name="B_fp8")
    sa_dt = pk.attach_input(sa, name="sa")
    sb_dt = pk.attach_input(sb, name="sb")
    C_dt = pk.attach_input(C, name="C_bf16")

    pk.fp8_gemm_dense_mediumm_layer(
        input_fp8=A_dt,
        weight_fp8=B_dt,
        input_scale=sa_dt,
        weight_scale=sb_dt,
        output=C_dt,
        num_workers=num_workers,
    )

    folder = os.path.dirname(os.path.abspath(__file__))
    print("compiling...", flush=True)
    pk.compile(output_dir=folder)
    print("running...", flush=True)
    pk.run_test_mode()
    torch.cuda.synchronize()

    C_ref = torch_reference(A_fp8, B_fp8, sa, sb, M, N, K)
    err = (C.float() - C_ref.float()).abs()
    max_err = err.max().item()
    mean_err = err.mean().item()
    status = "OK" if (mean_err < 1e-2 and max_err < 0.5) else "FAIL"
    print(f"M={M} K={K} N={N} max_err={max_err:.4f} mean_err={mean_err:.4f} "
          f"[{status}]")
    if not (mean_err < 1e-2 and max_err < 0.5):
        sys.exit(1)

    n_iters = int(os.environ.get("BENCH_ITERS", 20))
    if n_iters > 0:
        for _ in range(20):
            pk.run_test_mode()
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(n_iters):
            pk.run_test_mode()
        e.record()
        torch.cuda.synchronize()
        ms = s.elapsed_time(e) / n_iters
        flops = 2.0 * M * N * K
        tf = flops / (ms / 1000.0) / 1e12
        print(f"MPK bench: {ms*1000:7.1f} us  {tf:6.1f} TFLOPS")


if __name__ == "__main__":
    main()
