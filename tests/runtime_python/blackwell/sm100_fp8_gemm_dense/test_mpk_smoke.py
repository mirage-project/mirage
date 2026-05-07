"""MPK end-to-end smoke for fp8_gemm_dense_layer.

Allocates A (fp8), B (fp8), sa/sb (fp32 scales), C (bf16) at kv_b_proj
shape (M, K=512, N=4096), runs as a single MPK task in test_mode, compares
to torch fp32 reference.
"""
import math
import os
import sys

import torch
import mirage
from mirage.core import bfloat16, float32, float8_e4m3
from mirage.mpk.persistent_kernel import PersistentKernel


def torch_reference(A_fp8, B_fp8, sa, sb, M, N, K):
    """Block-scaled FP8 GEMM reference in fp32."""
    BLK = 128
    A = A_fp8.float()  # [M, K]
    B = B_fp8.float()  # [N, K]
    out = torch.zeros(M, N, dtype=torch.float32, device=A.device)
    nk = K // BLK
    for ki in range(nk):
        a_blk = A[:, ki * BLK:(ki + 1) * BLK]              # [M, 128]
        b_blk = B[:, ki * BLK:(ki + 1) * BLK]              # [N, 128]
        partial = a_blk @ b_blk.T                           # [M, N]
        # sa[m, ki] * sb[n//128, ki]
        sa_col = sa[:, ki:ki + 1]                          # [M, 1]
        sb_row = sb[:, ki].repeat_interleave(BLK)          # [N]
        out += partial * sa_col * sb_row[None, :]
    return out.to(torch.bfloat16)


def main():
    M = int(os.environ.get("M", 512))
    K = 512
    N = 4096
    print(f"shape M={M} K={K} N={N}")

    device = "cuda"
    torch.manual_seed(0)

    # Build raw fp8 / scale tensors. Avoid 0x7F / 0xFF (FP8 e4m3 NaN encodings)
    # by quantizing real bf16 randoms instead of dumping random bytes.
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
        max_num_batched_tokens=M,
        max_num_batched_requests=1,
        max_seq_length=1,
    )
    pk = PersistentKernel(**params)

    A_dt = pk.attach_input(A_fp8, name="A_fp8")
    B_dt = pk.attach_input(B_fp8, name="B_fp8")
    sa_dt = pk.attach_input(sa, name="sa")
    sb_dt = pk.attach_input(sb, name="sb")
    C_dt = pk.attach_input(C, name="C_bf16")

    pk.fp8_gemm_dense_smallm_layer(
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
    # FP8 e4m3 rounding gives O(0.05) absolute error on random inputs.
    status = "OK" if (mean_err < 1e-2 and max_err < 0.5) else "FAIL"
    print(f"M={M} K={K} N={N} max_err={max_err:.4f} mean_err={mean_err:.4f} "
          f"[{status}]")
    if not (mean_err < 1e-2 and max_err < 0.5):
        sys.exit(1)

    # Microbench: time pk.run_test_mode() (one full task graph pass = one GEMM).
    n_iters = int(os.environ.get("BENCH_ITERS", 50))
    if n_iters > 0:
        # Warmup
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
