"""DSV3 fine-N dense FP8 block-scaled GEMM (bf16 out) via PersistentKernel
test_mode.

Validates the new `pk.fp8_gemm_dense_finen_layer` ->
TASK_FP8_GEMM_DENSE_FINEN_SM100 (308). The finen kernel is the mediumm device
body re-tiled to BN=16 (NS default 6, NE=4 baked in the finen fn). Its host-side
delta vs the dense GEMM is solely the B (weight) TMA descriptor box height (=16
instead of 128); A keeps box=128. It handles all M (correct at M=1 decode AND
M>1 prefill), so this exercises both.

Scale layout (per fp8_gemm_dense_sm100_common.cuh, plain float32):
    sa: float32 [M, K/128]    row-major  (1x128 group activation scale)
    sb: float32 [N/128, K/128] row-major (128x128 block weight scale)

Shapes respect the builder.py finen gate (weight.dim(0)=N <= 2304, N%16==0,
K%512==0):
    qkv_a : N=2176, K=7168   (down-proj to lora ranks; the ferret v003 target)
    kv_b  : N=2048, K=512    (small-K)
    small : N=2304, K=1024   (N at the gate's upper bound; BN=16 tile sweep)

Run:
    CUDA_VISIBLE_DEVICES=<free gpu> \
      python tests/runtime_python/blackwell/sm100_fp8_gemm_dense/\
test_fp8_gemm_dense_finen_pk_testmode.py
"""

import os
import sys

import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

import mirage  # noqa: E402
from mirage.mpk.persistent_kernel import PersistentKernel  # noqa: E402
from pytorch_reference import (  # noqa: E402
    quantize_a_f32scale,
    quantize_b_f32scale,
    reference_gemm,
    cosine_sim,
    rel_mean,
)


def run_case(bs: int, N: int, K: int, label: str, seed: int = 42):
    tag = f"bs={bs} N={N} K={K} [{label}]"
    print(f"\n{'='*78}\n{tag}\n{'='*78}", flush=True)

    device = "cuda"
    g = torch.Generator(device=device).manual_seed(seed)
    a_bf16 = torch.randn((bs, K), device=device, dtype=torch.bfloat16,
                         generator=g)
    b_bf16 = torch.randn((N, K), device=device, dtype=torch.bfloat16,
                         generator=g)

    a_fp8, sa = quantize_a_f32scale(a_bf16)
    b_fp8, sb = quantize_b_f32scale(b_bf16)
    ref = reference_gemm(a_fp8, sa, b_fp8, sb)

    output = torch.zeros((bs, N), device=device, dtype=torch.bfloat16)

    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["num_workers"] = num_workers
    params["num_local_schedulers"] = num_schedulers
    params["mpi_rank"] = 0
    params["world_size"] = 1
    params["max_num_batched_tokens"] = bs
    params["max_num_batched_requests"] = bs
    # mediumm-class kernel -> msl > 512 path is irrelevant here (finen is its
    # own task name), but keep a representative value.
    params["max_seq_length"] = 4096
    pk = PersistentKernel(**params)

    a_dt = pk.attach_input(a_fp8, name="a_fp8")
    b_dt = pk.attach_input(b_fp8, name="b_fp8")
    sa_dt = pk.attach_input(sa, name="sa")
    sb_dt = pk.attach_input(sb, name="sb")
    out_dt = pk.attach_input(output, name="output")

    pk.fp8_gemm_dense_finen_layer(
        input_fp8=a_dt, weight_fp8=b_dt,
        input_scale=sa_dt, weight_scale=sb_dt,
        output=out_dt, num_workers=num_workers,
    )

    compile_dir = os.path.join(
        THIS_DIR, f".pk_compile_finen_{label}_{bs}")
    os.makedirs(compile_dir, exist_ok=True)
    pk.compile(output_dir=compile_dir)
    pk()
    torch.cuda.synchronize()

    zero_rows = (output.float().abs().sum(dim=1) == 0).nonzero(
        as_tuple=True)[0]
    cos = cosine_sim(output, ref)
    rel = rel_mean(output, ref)
    max_diff = (output.float() - ref.float()).abs().max().item()
    # Stricter than the decision-log 0.99 floor: this is a bit-for-bit re-tile
    # of the mediumm body, so require cos >= 0.999 (per the integration spec).
    passed = cos >= 0.999 and zero_rows.numel() == 0
    print(f"  cos={cos:.5f} rel={rel*100:.3f}% max_abs_diff={max_diff:.4f} "
          f"zero_rows={zero_rows.numel()} -> "
          f"{'PASS' if passed else 'FAIL'}", flush=True)

    pk.finalize()
    return passed, cos, rel, tag


def main():
    results = []
    # M=1 (decode, the lever's target) AND M>1 (prefill/ingest) for each shape,
    # since finen handles all M with no dual-dispatch.
    shapes = [
        (2176, 7168, "qkv_a"),
        (2048, 512, "kv_b"),
        (2304, 1024, "small_Nmax"),
    ]
    smoke = os.environ.get("MPK_SMOKE") == "1"
    bs_list = [1] if smoke else [1, 4, 8]
    if smoke:
        shapes = shapes[:1]
    for N, K, label in shapes:
        for bs in bs_list:
            results.append(run_case(bs, N=N, K=K, label=label))
    return _summary(results)


def _summary(results):
    print(f"\n{'='*78}\nSummary\n{'='*78}", flush=True)
    all_passed = True
    for passed, cos, rel, tag in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {status}  cos={cos:.4f} rel={rel*100:.3f}%  {tag}",
              flush=True)
        all_passed = all_passed and passed

    print(f"\n{'ALL PASS' if all_passed else 'SOME FAILED'} "
          f"({sum(r[0] for r in results)}/{len(results)})", flush=True)
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
