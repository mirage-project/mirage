"""Perf reproducer for `fp8_gemm_dense_mediumm_sm100_task_impl<128, 3>`.

Goal
----
This test reproduces the exact (M, N, K) shapes the MediumM dense FP8 kernel
sees inside the DeepSeek-V3 megakernel at TP=4 EP=2 mbt=128 — the production
optimization target. It exists so that anyone re-tuning the kernel can verify
their change against the *real* per-shape wallclock the megakernel observes,
not just an isolated micro-benchmark.

We drive the kernel via the public `PersistentKernel` test_mode path so the
test goes through:

  * full task-graph build (`fp8_gemm_dense_mediumm_layer` → register_task →
    annotated_graph),
  * megakernel codegen + nvcc compile,
  * scheduler dispatch with one persistent worker per output tile,
  * the SAME TMA descriptor encoding the production demo uses.

A standalone CUTLASS-style micro-benchmark would NOT measure the scheduler
fan-out / per-task launch cost, which is part of what the MediumM owner needs
to optimize against. This test does.

Shapes covered (extracted from the codegen of a real DSv3 19L TP=4 decode
run — see the table in the test body):

    Tag           N         K         μs (env-OFF baseline, mbt=128)
    q_a       2176     7168     ~28-32 μs    (single instance / layer, 80 CTAs)
    O_proj    7168    16384     ~65-70 μs    (single instance / layer, 80 CTAs)
    q_b       18432    1536     ~14-17 μs
    kv_b_k     4096      512     ~ 3-4  μs
    shared_e   9216    7168     ~30-35 μs    (MoE shared expert)

Each shape runs num_workers = 80 (matches the production default after B26).

Numerical correctness is verified against a bf16-dequant matmul reference.
Per-call wallclock is measured via `torch.cuda.Event`. We run a warm-up
followed by N=10 timed iters and report min/median/mean.

How to run
----------
This test is self-contained — it builds and runs the kernel through the
mirage Python package on the GPU available to the current process. Invoke
the test as a script:

    python tests/runtime_python/blackwell/sm100_fp8_gemm_dense/test_fp8_gemm_dense_mediumm_perf.py

Optional environment knobs:
    MEDIUMM_PERF_SHAPES=q_a,O_proj   # only run a comma-list of tags (default = all)
    MEDIUMM_PERF_ITERS=20             # timed iters per shape (default = 10)
    MEDIUMM_PERF_NUM_WORKERS=80       # persistent workers used (default = 80)
    MEDIUMM_PERF_NO_REF=1             # skip the slow bf16 cosine check
                                       (only collect wallclock)

The test exits 0 if every shape passes the correctness check (cos > 0.99
unless `MEDIUMM_PERF_NO_REF=1`). Wallclocks are printed regardless of
correctness so a kernel author can iterate on perf while breaking
correctness temporarily.
"""

import os
import sys
import time

import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
COMMON_DIR = os.path.abspath(os.path.join(THIS_DIR, "../common"))
if COMMON_DIR not in sys.path:
    sys.path.insert(0, COMMON_DIR)

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel


FP8_MAX = 448.0

# ---------------------------------------------------------------------------
# Shape catalogue — pulled from the DSv3 19L TP=4 megakernel codegen.
# Each entry: tag, N, K. M is always 128 (= mbt).
# ---------------------------------------------------------------------------
SHAPES = [
    # tag,        N,      K
    ("q_a",       2176,   7168),   # qkv_a projection — 28-32 μs straggler
    ("O_proj",    7168,   16384),  # attention output proj — 65-70 μs straggler
    ("q_b",       18432,  1536),   # q_b decode (32 heads * 576)
    ("kv_b_k",    4096,   512),    # kv_b at prefill prompt tile
    ("shared_e",  9216,   7168),   # MoE shared expert (gate + up)
]


# ---------------------------------------------------------------------------
# Quantize helpers — match the f32-scale layout that mediumm consumes:
#   sa: float32 [M, K/128]      row-major (per-row, per-128-col group)
#   sb: float32 [N/128, K/128]  row-major (per 128x128 block)
# This is the SAME layout produced by `quantize_fp8_f32scale_sm100` in the
# real megakernel and read by `fp8_gemm_dense_mediumm_sm100`.
# ---------------------------------------------------------------------------
def quantize_a_f32scale(a_bf16: torch.Tensor):
    M, K = a_bf16.shape
    assert K % 128 == 0
    nk = K // 128
    a_fp8 = torch.empty_like(a_bf16, dtype=torch.float8_e4m3fn)
    sa = torch.empty((M, nk), dtype=torch.float32, device=a_bf16.device)
    a_view = a_bf16.float().view(M, nk, 128)
    abs_max = a_view.abs().amax(dim=-1)
    scale = (abs_max / FP8_MAX).clamp_min(1e-12)
    sa.copy_(scale)
    quantized = (a_view / scale.unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX).view(M, K)
    a_fp8.copy_(quantized.to(torch.float8_e4m3fn))
    return a_fp8, sa


def quantize_b_f32scale(b_bf16: torch.Tensor):
    N, K = b_bf16.shape
    assert N % 128 == 0 and K % 128 == 0
    nb = N // 128
    nk = K // 128
    b_fp8 = torch.empty_like(b_bf16, dtype=torch.float8_e4m3fn)
    sb = torch.empty((nb, nk), dtype=torch.float32, device=b_bf16.device)
    b_view = b_bf16.float().view(nb, 128, nk, 128)
    abs_max = b_view.abs().amax(dim=(1, 3))
    scale = (abs_max / FP8_MAX).clamp_min(1e-12)
    sb.copy_(scale)
    quantized = (b_view / scale.view(nb, 1, nk, 1)).clamp(-FP8_MAX, FP8_MAX)
    b_fp8.copy_(quantized.view(N, K).to(torch.float8_e4m3fn))
    return b_fp8, sb


def reference_gemm(a_fp8, sa, b_fp8, sb):
    """Dequant A, B then compute C = A @ B^T in fp32, return bf16."""
    M, K = a_fp8.shape
    N = b_fp8.shape[0]
    a_dq = (
        a_fp8.float().view(M, K // 128, 128)
        * sa.unsqueeze(-1)
    ).view(M, K)
    b_dq = (
        b_fp8.float().view(N // 128, 128, K // 128, 128)
        * sb.view(N // 128, 1, K // 128, 1)
    ).view(N, K)
    return (a_dq @ b_dq.t()).to(torch.bfloat16)


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.float().flatten()
    bf = b.float().flatten()
    return (torch.dot(af, bf) / (af.norm() * bf.norm() + 1e-12)).item()


# ---------------------------------------------------------------------------
# Per-shape PK test_mode driver
# ---------------------------------------------------------------------------
def run_shape(tag: str, M: int, N: int, K: int,
              num_workers: int, timed_iters: int, do_ref: bool, seed: int = 42):
    print(f"\n{'='*78}")
    print(f"[{tag}]  M={M}  N={N}  K={K}  num_workers={num_workers}")
    print(f"{'='*78}")

    device = "cuda"
    torch.manual_seed(seed)
    g = torch.Generator(device=device).manual_seed(seed)

    a_bf16 = torch.randn((M, K), device=device, dtype=torch.bfloat16, generator=g)
    b_bf16 = torch.randn((N, K), device=device, dtype=torch.bfloat16, generator=g)
    a_fp8, sa = quantize_a_f32scale(a_bf16)
    b_fp8, sb = quantize_b_f32scale(b_bf16)

    output = torch.zeros((M, N), device=device, dtype=torch.bfloat16)

    nw, ns = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["num_workers"] = nw
    params["num_local_schedulers"] = ns
    params["mpi_rank"] = 0
    params["world_size"] = 1
    params["max_num_batched_tokens"] = M
    params["max_num_batched_requests"] = M
    pk = PersistentKernel(**params)

    a_dt = pk.attach_input(a_fp8, name="a_fp8")
    b_dt = pk.attach_input(b_fp8, name="b_fp8")
    sa_dt = pk.attach_input(sa, name="sa")
    sb_dt = pk.attach_input(sb, name="sb")
    out_dt = pk.attach_input(output, name="output")

    pk.fp8_gemm_dense_mediumm_layer(
        input_fp8=a_dt, weight_fp8=b_dt,
        input_scale=sa_dt, weight_scale=sb_dt,
        output=out_dt,
        num_workers=num_workers,
    )

    compile_dir = os.path.join(THIS_DIR, f"pk_mediumm_perf_{tag}_{M}_{N}_{K}")
    os.makedirs(compile_dir, exist_ok=True)

    print(f"  compiling megakernel into {compile_dir} ...")
    pk.compile(output_dir=compile_dir)

    # Warm-up: 3 iters to get the megakernel pages resident + warm caches.
    for _ in range(3):
        pk()
    torch.cuda.synchronize()

    # Timing: CUDA events around each pk() call (megakernel launch + run).
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(timed_iters)]
    ends   = [torch.cuda.Event(enable_timing=True) for _ in range(timed_iters)]
    for i in range(timed_iters):
        starts[i].record()
        pk()
        ends[i].record()
    torch.cuda.synchronize()
    times_ms = [s.elapsed_time(e) for s, e in zip(starts, ends)]
    times_us = sorted(t * 1000.0 for t in times_ms)
    n = len(times_us)
    p50 = times_us[n // 2]
    p_min = times_us[0]
    p_max = times_us[-1]
    p_mean = sum(times_us) / n
    print(f"  wallclock per pk() call (μs): "
          f"min={p_min:.2f}  median={p50:.2f}  mean={p_mean:.2f}  max={p_max:.2f}  n={n}")
    # FLOPS for sanity check (B200 SM100 peak FP8: 4.5 PFLOPS/GPU at sm100a).
    flops = 2.0 * M * N * K
    tflops = flops / (p50 / 1e6) / 1e12
    print(f"  achieved compute: {tflops:.1f} TFLOPS @ p50")

    if do_ref:
        print(f"  computing bf16-dequant reference ...")
        ref = reference_gemm(a_fp8, sa, b_fp8, sb)
        cos = cosine_sim(output, ref)
        max_abs = (output.float() - ref.float()).abs().max().item()
        passed = cos > 0.99
        print(f"  correctness: cos={cos:.4f}  max_abs={max_abs:.4f}  "
              f"{'PASS' if passed else 'FAIL'}")
    else:
        passed = True
        cos = float("nan")

    pk.finalize()
    return {
        "tag": tag, "M": M, "N": N, "K": K,
        "min_us": p_min, "median_us": p50, "mean_us": p_mean, "max_us": p_max,
        "tflops_p50": tflops, "cos": cos, "passed": passed,
    }


def main():
    only = set(s for s in os.environ.get("MEDIUMM_PERF_SHAPES", "").split(",") if s)
    iters = int(os.environ.get("MEDIUMM_PERF_ITERS", "10"))
    num_workers = int(os.environ.get("MEDIUMM_PERF_NUM_WORKERS", "80"))
    do_ref = os.environ.get("MEDIUMM_PERF_NO_REF") != "1"

    todo = [s for s in SHAPES if not only or s[0] in only]
    if not todo:
        print(f"No shape matched MEDIUMM_PERF_SHAPES={os.environ.get('MEDIUMM_PERF_SHAPES')!r}",
              file=sys.stderr)
        return 2

    M = 128  # = mbt; all DSv3 mediumm callsites use BM=128.
    results = []
    for tag, N, K in todo:
        results.append(run_shape(tag, M, N, K, num_workers, iters, do_ref))

    print(f"\n{'='*78}")
    print(f"SUMMARY  (M=128, num_workers={num_workers}, timed iters={iters})")
    print(f"{'='*78}")
    print(f"{'tag':10s} {'N':>6} {'K':>6}   min_μs  med_μs  mean_μs  max_μs  "
          f"TFLOPS  cos")
    for r in results:
        print(f"{r['tag']:10s} {r['N']:>6d} {r['K']:>6d}   "
              f"{r['min_us']:6.2f}  {r['median_us']:6.2f}  "
              f"{r['mean_us']:6.2f}  {r['max_us']:6.2f}  "
              f"{r['tflops_p50']:6.1f}  {r['cos']:.4f}")
    all_ok = all(r["passed"] for r in results)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
