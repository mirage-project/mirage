"""Test: dsv3_router_gemm_sm100 via PersistentKernel test_mode.

Validates the CUDA-core router-gate GEMV ported from TRT-LLM/vLLM
(dsv3_router_gemm). Skinny GEMM out[M, N] = act[M, K] @ gate_w[N, K]^T with
N = NUM_EXPERTS (256), K = HIDDEN (7168), M = decode tokens (<=16). The kernel
is crash-free by construction (no tensor core / tcgen05 / TMA / cross-CTA
reduce) — the intended replacement for the ~12.6us split-K swapAB router gate.

Goes through the full MPK compile pipeline (task registration with the (2 in,
1 out) tuple, megakernel codegen, nvcc) — the same path the DeepSeek V3 demo
uses. Compares vs a bf16 torch reference (cos > 0.99) and reports the per-task
wall-span (kernel latency) for an A/B vs the current router.

Run:
  CUDA_VISIBLE_DEVICES=<free-gpu> python tests/runtime_python/blackwell/sm100_dsv3_router_gemm/test_dsv3_router_gemm_testmode.py
"""

import os
import sys

import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

import mirage  # noqa: E402
from mirage.mpk.persistent_kernel import PersistentKernel  # noqa: E402


def cosine_sim(a, b):
    a = a.float().flatten()
    b = b.float().flatten()
    return (torch.dot(a, b) / (a.norm() * b.norm() + 1e-12)).item()


def run_case(M, K=7168, N=256, n_per_cta=2, seed=0):
    grid_x = N // n_per_cta
    label = f"M={M} K={K} N={N} grid_x={grid_x} (N_PER_CTA={n_per_cta})"
    print(f"\n{'=' * 70}\nPK test_mode dsv3_router_gemm: {label}\n{'=' * 70}")

    device = "cuda"
    g = torch.Generator(device=device).manual_seed(seed)
    # router weight is small magnitude (gate.weight); keep act/weight bf16.
    act = (torch.randn(M, K, dtype=torch.bfloat16, device=device, generator=g)
           * 0.1).contiguous()
    weight = (torch.randn(N, K, dtype=torch.bfloat16, device=device, generator=g)
              / (K ** 0.5)).contiguous()
    out = torch.zeros(M, N, dtype=torch.bfloat16, device=device)

    ref = (act.float() @ weight.float().T).to(torch.bfloat16)

    nw, nsched = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params.update(test_mode=True, num_workers=nw, num_local_schedulers=nsched,
                  mpi_rank=0, world_size=1,
                  max_num_batched_tokens=M, max_num_batched_requests=M,
                  trace_name=f"router_gemv_{M}_{N}",
                  profiler_tensor=torch.zeros(3000 * 128, dtype=torch.uint64,
                                              device=device))
    pk = PersistentKernel(**params)
    a_dt = pk.attach_input(act, name="act")
    w_dt = pk.attach_input(weight, name="weight")
    o_dt = pk.attach_input(out, name="out")
    pk.dsv3_router_gemm_sm100_layer(
        input=a_dt, weight=w_dt, output=o_dt,
        grid_dim=(grid_x, 1, 1), block_dim=(128, 1, 1))

    cdir = os.path.join("/tmp", f"pk_router_gemv_{M}_{K}_{N}")
    os.makedirs(cdir, exist_ok=True)
    pk.compile(output_dir=cdir)
    pk()
    torch.cuda.synchronize()

    cos = cosine_sim(out, ref)
    md = (out.float() - ref.float()).abs().max().item()
    zero_rows = (out.float().abs().sum(dim=1) == 0).sum().item()
    print(f"  out[0,:6]={out.float()[0,:6].tolist()}")
    print(f"  ref[0,:6]={ref.float()[0,:6].tolist()}")
    print(f"  cosine={cos:.6f}  max_abs_diff={md:.4f}  zero_rows={zero_rows}")
    passed = cos > 0.99 and zero_rows == 0
    print(f"  Result: {'PASSED' if passed else 'FAILED'}")
    pk.finalize()
    return passed, cos


def main():
    results = {}
    for n_per_cta in (1, 2):
        for M in (1, 2, 4):
            p, c = run_case(M=M, n_per_cta=n_per_cta, seed=M * 10 + n_per_cta)
            results[f"M={M},EPC={n_per_cta}"] = (p, c)
    print(f"\n{'=' * 70}\nSummary\n{'=' * 70}")
    ok = True
    for k, (p, c) in results.items():
        print(f"  {k}: {'PASS' if p else 'FAIL'}  cos={c:.4f}")
        ok = ok and p
    print("\nAll PASSED" if ok else "\nSome FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
