"""
Test: NVFP4 swapAB linear via PersistentKernel test_mode.

Validates the new TASK_LINEAR_NVFP4_SM100 MPK task end-to-end through the full
pipeline (Python layer -> task registration -> code generation -> nvcc ->
runtime dispatch + runtime-built fp4/bf16 TMA descriptors).

The kernel math is already validated by the standalone NVFP4 test extension; the
same pre-quantized inputs are fed to both the standalone kernel (the oracle) and
the MPK task, and the two outputs are compared. This isolates the MPK wiring /
TMA-descriptor correctness from the kernel arithmetic.

Run:
    python tests/runtime_python/test_mode/test_nvfp4_linear_testmode.py
"""

import os
import sys

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel

# Import the standalone NVFP4 extension + its quant utilities.
_NVFP4_DIR = os.path.join(
    os.path.dirname(__file__), "..", "blackwell", "sm100_linear_nvfp4"
)
sys.path.insert(0, _NVFP4_DIR)
sys.path.insert(0, os.path.join(_NVFP4_DIR, "profile"))
import runtime_kernel_blackwell_linear_nvfp4 as nvfp4_ext  # noqa: E402
from nvfp4_util import (  # noqa: E402
    make_random_nvfp4_tensors,
    interleave_sf_tensor,
)


def make_pk(batch_size, **extra):
    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params.update(
        test_mode=True,
        num_workers=num_workers,
        num_local_schedulers=num_schedulers,
        mpi_rank=0,
        world_size=1,
        max_num_batched_tokens=batch_size,
        max_num_batched_requests=batch_size,
    )
    params.update(extra)
    return PersistentKernel(**params)


def test_linear_nvfp4(batch_size=64, output_size=256, reduction_size=512):
    """Pre-quantized NVFP4 swapAB GEMM: output[B, N] = x_fp4 @ w_fp4^T."""
    device = "cuda"
    torch.manual_seed(0)
    print(f"\n{'='*70}\nTest: NVFP4 swapAB linear "
          f"(B={batch_size}, N={output_size}, K={reduction_size})")

    # Random pre-quantized fp4 weight + bf16 activation. The activation SF must
    # use the layout for the path the dispatcher will pick: swapAB per-tile for
    # small batch (mma_n>0), interleaved for the 1d2d path (mma_n=0). The MPK
    # linear_nvfp4_layer selects the same path by batch size, so the oracle (also
    # shape-dispatched) and the MPK task see matching SF layouts.
    x_bf16 = torch.randn(batch_size, reduction_size, dtype=torch.bfloat16,
                         device=device) * 0.3
    _, w_packed, _, w_sf = make_random_nvfp4_tensors(
        batch_size, output_size, reduction_size, device=device
    )
    w_sf_interleaved = interleave_sf_tensor(w_sf)
    mma_n = nvfp4_ext.swapab_mma_n(x_bf16, w_packed) if batch_size < 128 else 0
    x_q = nvfp4_ext.quantize_nvfp4_sm100(x_bf16, mma_n)
    x_packed, x_sf = x_q[0], x_q[1]

    # Oracle: the validated standalone kernel on the identical inputs.
    ref = torch.empty(batch_size, output_size, dtype=torch.bfloat16, device=device)
    nvfp4_ext.linear_nvfp4_sm100_no_quantization(
        x_packed, x_sf, w_packed, w_sf_interleaved, None, ref
    )
    torch.cuda.synchronize()

    output = torch.zeros(batch_size, output_size, dtype=torch.bfloat16, device=device)
    pk = make_pk(batch_size)
    t_x = pk.attach_input(x_packed, name="x_fp4")
    t_w = pk.attach_input(w_packed, name="w_fp4")
    t_xsf = pk.attach_input(x_sf, name="x_sf")
    t_wsf = pk.attach_input(w_sf_interleaved, name="w_sf")
    t_out = pk.attach_input(output, name="output")

    pk.linear_nvfp4_layer(
        input_fp4=t_x, weight_fp4=t_w,
        input_scale=t_xsf, weight_scale=t_wsf, output=t_out,
        grid_dim=(1, 1, 1), block_dim=(256, 1, 1),
    )

    print("Compiling...")
    pk.compile(output_dir=os.path.dirname(__file__))
    print("Running...")
    pk.run_test_mode()
    torch.cuda.synchronize()

    print(f"MPK output[0, :8]: {output[0, :8]}")
    print(f"Oracle   [0, :8]: {ref[0, :8]}")
    max_abs = (output.float() - ref.float()).abs().max().item()
    max_rel = max_abs / max(ref.float().abs().max().item(), 1e-6)
    print(f"Max abs diff: {max_abs:.4f}, Max rel err: {max_rel:.4f}")
    passed = max_rel < 1e-2  # same kernel -> should match to bf16 rounding
    print(f"{'PASSED' if passed else 'FAILED'}: NVFP4 swapAB linear")
    pk.finalize()
    return passed


if __name__ == "__main__":
    # batch<128 -> swapAB path; batch>=128 -> 1d2d 1SM path. Both compared
    # against the standalone kernel oracle on identical inputs.
    ok = True
    ok &= test_linear_nvfp4(batch_size=64, output_size=256, reduction_size=512)
    ok &= test_linear_nvfp4(batch_size=128, output_size=256, reduction_size=512)
    print(f"\n{'='*70}")
    if not ok:
        sys.exit(1)
    print("NVFP4 linear tests PASSED!")
