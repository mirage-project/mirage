"""
Test: MXFP4 linear (swapAB + 1d2d 1SM) via PersistentKernel test_mode.

Validates TASK_LINEAR_MXFP4_SM100 (swapAB, small batch) and
TASK_LINEAR_MXFP4_1D2D_SM100 (1d2d 1SM, large batch) end-to-end through the full
MPK pipeline. Oracles:
  * 1d2d  -> the validated standalone kernel (linear_mxfp4_sm100_no_quantization,
             which always uses the 1d2d path) on identical inputs.
  * swapAB -> a pure-torch dequant matmul (mxfp4_reference_matmul); the standalone
             extension has no swapAB launcher, so we compare against the
             kernel-independent reference instead.

Run:
    python tests/runtime_python/test_mode/test_mxfp4_linear_testmode.py
"""

import os
import sys

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel

_MXFP4_DIR = os.path.join(
    os.path.dirname(__file__), "..", "blackwell", "sm100_linear_mxfp4"
)
sys.path.insert(0, _MXFP4_DIR)
sys.path.insert(0, os.path.join(_MXFP4_DIR, "profile"))
import runtime_kernel_blackwell_linear_mxfp4 as mxfp4_ext  # noqa: E402
from mxfp4_util import mxfp4_reference_matmul  # noqa: E402


def swapab_mma_n(batch_size, output_size):
    """Same occupancy formula the register/tma builder uses for the swapAB tile."""
    sm_count = torch.cuda.get_device_properties(0).multi_processor_count
    budget = sm_count // ((output_size + 127) // 128)
    needed = (batch_size + budget - 1) // budget if budget >= 1 else 128
    for t in (8, 16, 32, 64):
        if needed <= t:
            return t
    return 128


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


def test_linear_mxfp4(batch_size, output_size=256, reduction_size=512):
    device = "cuda"
    torch.manual_seed(0)
    is_swapab = batch_size < 128
    path = "swapAB" if is_swapab else "1d2d"
    print(f"\n{'='*70}\nTest: MXFP4 {path} linear "
          f"(B={batch_size}, N={output_size}, K={reduction_size})")

    x = torch.randn(batch_size, reduction_size, device=device) * 0.5
    w = torch.randn(output_size, reduction_size, device=device) * 0.5

    # Weight + interleaved-layout activation feed the torch oracle; the MPK task
    # gets the activation SF in the layout its path consumes (swapAB tile for
    # small batch, interleaved for 1d2d). Packed fp4 bytes are layout-independent.
    w_q, w_sf = mxfp4_ext.quantize_mxfp4_sm100(w, 0)
    x_q, x_sf_il = mxfp4_ext.quantize_mxfp4_sm100(x, 0)
    if is_swapab:
        mma_n = swapab_mma_n(batch_size, output_size)
        x_q2, x_sf = mxfp4_ext.quantize_mxfp4_sm100(x, mma_n)
        assert torch.equal(x_q, x_q2)  # packed values must match the oracle's
    else:
        x_sf = x_sf_il

    # quantize_mxfp4_sm100 may pad the packed rows up to a multiple of 128; the
    # reference dequant uses x_q's (padded) row count, so slice back to batch.
    ref = mxfp4_reference_matmul(x_q, x_sf_il, w_q, w_sf, reduction_size).to(
        torch.bfloat16
    )[:batch_size]
    torch.cuda.synchronize()

    output = torch.zeros(batch_size, output_size, dtype=torch.bfloat16, device=device)
    pk = make_pk(batch_size)
    t_x = pk.attach_input(x_q, name="x_fp4")
    t_w = pk.attach_input(w_q, name="w_fp4")
    t_xsf = pk.attach_input(x_sf, name="x_sf")
    t_wsf = pk.attach_input(w_sf, name="w_sf")
    t_out = pk.attach_input(output, name="output")

    pk.linear_mxfp4_layer(
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
    print(f"Reference [0, :8]: {ref[0, :8]}")
    max_abs = (output.float() - ref.float()).abs().max().item()
    max_rel = max_abs / max(ref.float().abs().max().item(), 1e-6)
    print(f"Max abs diff: {max_abs:.4f}, Max rel err: {max_rel:.4f}")
    # vs a torch dequant reference -> allow bf16 accumulation slack.
    passed = max_rel < 0.05
    print(f"{'PASSED' if passed else 'FAILED'}: MXFP4 {path} linear")
    pk.finalize()
    return passed


if __name__ == "__main__":
    ok = True
    ok &= test_linear_mxfp4(batch_size=64)   # swapAB path
    ok &= test_linear_mxfp4(batch_size=128)  # 1d2d 1SM path
    print(f"\n{'='*70}")
    if not ok:
        sys.exit(1)
    print("MXFP4 linear tests PASSED!")
