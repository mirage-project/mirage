"""Test: fp8_gemm_dense_smallm_sm100_task_impl<128, 3> via PersistentKernel test_mode.

Shape under investigation: kv_b_v projection in DSv3 chunked prefill.
Tests M=128 (=BM, baseline) and M=64 (< BM, suspected bug case).

Scale layout (per fp8_gemm_dense_sm100_common.cuh):
    sa: float32 [M, K/128]   row-major  (1x128 group activation scale)
    sb: float32 [N/128, K/128] row-major (128x128 block weight scale)

The test uses PersistentKernel test_mode which goes through the full MPK
compile pipeline (proper TMA descriptor creation, megakernel codegen, nvcc).
This is the same path used in the DeepSeek V3 demo.

Run:
    python tests/runtime_python/blackwell/sm100_fp8_gemm_dense/test_fp8_gemm_dense_smallm_pk_testmode.py
"""

import os
import sys
import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
COMMON_DIR = os.path.abspath(os.path.join(THIS_DIR, "../common"))
if COMMON_DIR not in sys.path:
    sys.path.insert(0, COMMON_DIR)

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel

FP8_MAX = 448.0


# ---------------------------------------------------------------------------
# Quantize helpers for the fp8_gemm_dense scale layout (plain float32)
# ---------------------------------------------------------------------------

def quantize_a_f32scale(a_bf16: torch.Tensor):
    """Quantize A [M, K] to FP8 e4m3 + float32 scale [M, K/128].
    Each scale covers a 1x128 group (per-row, per-128-columns chunk).
    sa[m, ki] = abs_max(A[m, ki*128:(ki+1)*128]) / FP8_MAX
    """
    M, K = a_bf16.shape
    assert K % 128 == 0
    nk = K // 128

    a_fp8 = torch.empty_like(a_bf16, dtype=torch.float8_e4m3fn)
    sa = torch.zeros((M, nk), dtype=torch.float32, device=a_bf16.device)

    a_f32 = a_bf16.float()
    for m in range(M):
        for ki in range(nk):
            block = a_f32[m, ki * 128:(ki + 1) * 128]
            abs_max = block.abs().max().item()
            scale = abs_max / FP8_MAX if abs_max > 0 else 1.0
            sa[m, ki] = scale
            a_fp8[m, ki * 128:(ki + 1) * 128] = (
                (block / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
            )

    return a_fp8, sa


def quantize_b_f32scale(b_bf16: torch.Tensor):
    """Quantize B [N, K] to FP8 e4m3 + float32 scale [N/128, K/128].
    Each scale covers a 128x128 block.
    sb[bi, ki] = abs_max(B[bi*128:(bi+1)*128, ki*128:(ki+1)*128]) / FP8_MAX
    """
    N, K = b_bf16.shape
    assert K % 128 == 0 and N % 128 == 0
    nb = N // 128
    nk = K // 128

    b_fp8 = torch.empty_like(b_bf16, dtype=torch.float8_e4m3fn)
    sb = torch.zeros((nb, nk), dtype=torch.float32, device=b_bf16.device)

    b_f32 = b_bf16.float()
    for bi in range(nb):
        for ki in range(nk):
            block = b_f32[bi * 128:(bi + 1) * 128, ki * 128:(ki + 1) * 128]
            abs_max = block.abs().max().item()
            scale = abs_max / FP8_MAX if abs_max > 0 else 1.0
            sb[bi, ki] = scale
            b_fp8[bi * 128:(bi + 1) * 128, ki * 128:(ki + 1) * 128] = (
                (block / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
            )

    return b_fp8, sb


def reference_gemm(a_fp8, sa, b_fp8, sb):
    """Dequant A, B then compute C = A @ B.T in f32, return bf16."""
    M, K = a_fp8.shape
    N = b_fp8.shape[0]
    nk = K // 128

    a_dq = torch.empty(M, K, dtype=torch.float32, device=a_fp8.device)
    for m in range(M):
        for ki in range(nk):
            a_dq[m, ki * 128:(ki + 1) * 128] = (
                a_fp8[m, ki * 128:(ki + 1) * 128].float() * sa[m, ki]
            )

    nb = N // 128
    b_dq = torch.empty(N, K, dtype=torch.float32, device=b_fp8.device)
    for bi in range(nb):
        for ki in range(nk):
            b_dq[bi * 128:(bi + 1) * 128, ki * 128:(ki + 1) * 128] = (
                b_fp8[bi * 128:(bi + 1) * 128,
                      ki * 128:(ki + 1) * 128].float() * sb[bi, ki]
            )

    return torch.matmul(a_dq, b_dq.t()).to(torch.bfloat16)


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a_f = a.float().flatten()
    b_f = b.float().flatten()
    return (torch.dot(a_f, b_f) / (a_f.norm() * b_f.norm() + 1e-12)).item()


# ---------------------------------------------------------------------------
# PersistentKernel test_mode runner
# ---------------------------------------------------------------------------

def run_pk_testmode(M: int, N: int, K: int, seed: int = 42):
    """Run fp8_gemm_dense_smallm via PersistentKernel test_mode.

    The PK framework handles:
    - TMA descriptor creation (correct parameters from tma.cuh)
    - Megakernel codegen + nvcc compilation
    - Scheduler dispatch
    - Single-CTA execution
    """
    label = f"M={M}, N={N}, K={K}"
    print(f"\n{'='*70}")
    print(f"PK test_mode: {label}")
    print(f"{'='*70}")

    device = "cuda"
    torch.manual_seed(seed)
    g = torch.Generator(device=device).manual_seed(seed)

    # The PK allocates A tensor with shape [M, K] = [max_num_batched_tokens, K].
    # M is passed as a compile-time constant to the kernel.
    # Here M IS max_num_batched_tokens (no padding needed for test).
    a_bf16 = torch.randn((M, K), device=device, dtype=torch.bfloat16, generator=g)
    b_bf16 = torch.randn((N, K), device=device, dtype=torch.bfloat16, generator=g)

    a_fp8, sa = quantize_a_f32scale(a_bf16)
    b_fp8, sb = quantize_b_f32scale(b_bf16)

    print(f"  a_fp8: {tuple(a_fp8.shape)}, sa: {tuple(sa.shape)}")
    print(f"  b_fp8: {tuple(b_fp8.shape)}, sb: {tuple(sb.shape)}")

    ref = reference_gemm(a_fp8, sa, b_fp8, sb)
    print(f"  ref[0, :4]: {ref[0, :4].tolist()}")

    output = torch.zeros((M, N), device=device, dtype=torch.bfloat16)

    # Build PersistentKernel in test mode.
    # max_num_batched_tokens=M tells the kernel the A-tensor row count.
    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["num_workers"] = num_workers
    params["num_local_schedulers"] = num_schedulers
    params["mpi_rank"] = 0
    params["world_size"] = 1
    params["max_num_batched_tokens"] = M
    params["max_num_batched_requests"] = M
    pk = PersistentKernel(**params)

    a_dt  = pk.attach_input(a_fp8,  name="a_fp8")
    b_dt  = pk.attach_input(b_fp8,  name="b_fp8")
    sa_dt = pk.attach_input(sa,     name="sa")
    sb_dt = pk.attach_input(sb,     name="sb")
    out_dt = pk.attach_input(output, name="output")

    # fp8_gemm_dense_smallm_layer(input_fp8, weight_fp8, input_scale, weight_scale, output, num_workers)
    # num_workers=1 for a single-CTA test.
    pk.fp8_gemm_dense_smallm_layer(
        input_fp8=a_dt,
        weight_fp8=b_dt,
        input_scale=sa_dt,
        weight_scale=sb_dt,
        output=out_dt,
        num_workers=1,
    )

    compile_dir = os.path.join(THIS_DIR, f"pk_compile_{M}_{N}_{K}")
    os.makedirs(compile_dir, exist_ok=True)

    print("  Compiling...")
    pk.compile(output_dir=compile_dir)
    print("  Running...")
    pk()
    torch.cuda.synchronize()

    print(f"  out[0, :4]: {output[0, :4].tolist()}")

    zero_rows = (output.float().abs().sum(dim=1) == 0).nonzero(as_tuple=True)[0]
    if zero_rows.numel() > 0:
        print(f"  WARNING: {zero_rows.numel()} zero rows: {zero_rows.tolist()[:20]}")
    else:
        print(f"  No zero rows.")

    max_diff = (output.float() - ref.float()).abs().max().item()
    cos = cosine_sim(output, ref)
    print(f"  Max abs diff: {max_diff:.6f}")
    print(f"  Cosine sim:   {cos:.6f}")

    passed = cos > 0.99
    print(f"  Result: {'PASSED' if passed else 'FAILED'} (cos={cos:.4f})")

    pk.finalize()
    return passed, cos, max_diff


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    results = {}

    # Case 1: M=128 = BM — baseline, should always pass
    passed, cos, md = run_pk_testmode(M=128, N=4096, K=512)
    results["M=128"] = (passed, cos, md)

    # Case 2: M=64 < BM — suspected bug case (DSv3 kv_b_v at prompt_len=64)
    passed, cos, md = run_pk_testmode(M=64, N=4096, K=512)
    results["M=64"] = (passed, cos, md)

    # Case 3: M=32 — extreme small M for further boundary verification
    passed, cos, md = run_pk_testmode(M=32, N=4096, K=512)
    results["M=32"] = (passed, cos, md)

    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}")
    all_passed = True
    for label, (p, cos, md) in results.items():
        status = "PASS" if p else "FAIL"
        print(f"  {label}: {status}  cos={cos:.4f}  max_abs_diff={md:.4f}")
        all_passed = all_passed and p

    if all_passed:
        print("\nAll cases PASSED — kernel is correct for M<BM.")
    else:
        print("\nSome cases FAILED — M<BM case exposes kernel incorrectness.")
        m128_ok = results["M=128"][0]
        m64_fail = not results["M=64"][0]
        if m128_ok and m64_fail:
            print("  Confirmed: M=128 passes but M=64 fails => bug is M<BM specific.")
        elif not m128_ok:
            print("  M=128 also fails => likely a test harness issue, not M<BM specific.")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
