"""Test mode wrapper around ``PersistentKernel.quantize_fp8_layer``.

Builds a one-task graph that runs the SM100 block-wise BF16 -> FP8 quantizer
end-to-end through the MPK compile + run pipeline, then compares both the
packed UE8M0 scale tensor and the FP8 output against the pure-PyTorch
reference (which is what the kernel test already validates).

The output scale tensor uses the same layout the MPK builder allocates for
real models (column-major ``[packed_k, aligned_batch]``, matching what the
SM100 kernel writes), so we can compare it bit-for-bit against
``quantize_fp8_ref(..., layout="deepgemm_col_major")``.
"""

import os
import sys

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
COMMON_DIR = os.path.abspath(os.path.join(THIS_DIR, "../common"))
if COMMON_DIR not in sys.path:
    sys.path.insert(0, COMMON_DIR)

from sm100_fp8_scale_layout import (  # noqa: E402
    BLOCK_K,
    allocate_packed_ue8m0_scale_deepgemm_style,
)
from pytorch_reference import quantize_fp8_ref  # noqa: E402


def test_quantize_fp8_testmode():
    device = "cuda"
    torch.manual_seed(42)

    batch_size = 8
    hidden_dim = 4096
    assert hidden_dim % BLOCK_K == 0

    # Inputs / outputs (real torch storage; attached to the persistent kernel).
    x = torch.randn(batch_size, hidden_dim, dtype=torch.bfloat16, device=device)
    out_fp8 = torch.zeros(
        batch_size, hidden_dim, dtype=torch.float8_e4m3fn, device=device
    )
    # Scale tensor: packed UE8M0 uint32, column-major
    # [packed_k, aligned_batch] physical layout. ``allocate_*_deepgemm_style``
    # gives us a torch view with shape (batch, packed_k) and the right strides.
    out_scale = allocate_packed_ue8m0_scale_deepgemm_style(
        batch_size, hidden_dim, device
    )
    out_scale.zero_()

    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["num_workers"] = num_workers
    params["num_local_schedulers"] = num_schedulers
    params["mpi_rank"] = 0
    params["world_size"] = 1
    params["max_num_batched_tokens"] = batch_size
    params["max_num_batched_requests"] = batch_size
    pk = PersistentKernel(**params)

    assert pk.target_cc == 100, (
        "quantize_fp8_layer requires SM100 (Blackwell); "
        f"current target_cc={pk.target_cc}"
    )

    x_dt = pk.attach_input(x, name="x")
    out_fp8_dt = pk.attach_input(out_fp8, name="out_fp8")
    out_scale_dt = pk.attach_input(out_scale, name="out_scale")

    block_dim = (128, 1, 1)  # matches the MPK builder for this layer
    pk.quantize_fp8_layer(
        input=x_dt,
        output_fp8=out_fp8_dt,
        output_scale=out_scale_dt,
        grid_dim=(batch_size, 1, 1),
        block_dim=block_dim,
        scale_ue8m0=True,
    )

    print("Compiling test kernel...")
    pk.compile(output_dir=THIS_DIR)

    print("Running test kernel...")
    pk.run_test_mode()
    torch.cuda.synchronize()

    # Reference uses the same column-major UE8M0 packed scale layout the
    # kernel writes, so we can compare scales exactly.
    ref_fp8, ref_scale = quantize_fp8_ref(
        x, scale_ue8m0=True, layout="deepgemm_col_major"
    )

    assert out_scale.shape == ref_scale.shape, (
        f"scale shape mismatch: got {out_scale.shape} vs ref {ref_scale.shape}"
    )
    assert out_scale.stride() == ref_scale.stride(), (
        f"scale stride mismatch: got {out_scale.stride()} vs ref {ref_scale.stride()}"
    )
    torch.testing.assert_close(out_scale, ref_scale, rtol=0, atol=0)
    print("Scale tensor matches reference exactly.")

    # FP8 outputs: compare in float (matches the kernel-test pattern).
    torch.testing.assert_close(
        out_fp8.float(),
        ref_fp8.float(),
        rtol=1e-1,
        atol=16.0,
    )
    max_diff = (out_fp8.float() - ref_fp8.float()).abs().max().item()
    print(f"FP8 max abs diff (in float): {max_diff}")

    print("PASSED: quantize_fp8 test_mode produces correct output")
    pk.finalize()


if __name__ == "__main__":
    test_quantize_fp8_testmode()
