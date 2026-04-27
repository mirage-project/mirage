"""Test: linear_fp8_with_residual_layer (SM100) via PersistentKernel test_mode.

Builds a single-task FP8 linear-with-residual layer through the full MPK
compilation pipeline, runs it once, and validates against
``linear_fp8_with_residual_ref`` from ``pytorch_reference.py``.

Run:
    python tests/runtime_python/blackwell/sm100_linear_fp8/test_linear_fp8_with_residual_testmode.py
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
    aligned_scale_outer_dim,
    packed_scale_k_for_reduction_size,
    quantize_to_fp8_deepgemm_style,
)
from pytorch_reference import linear_fp8_with_residual_ref  # noqa: E402

# Reuse the local helpers from the no-residual testmode file so we don't
# duplicate scale-layout logic.
sys.path.insert(0, THIS_DIR)
from test_linear_fp8_testmode import (  # noqa: E402
    _input_scale_for_mpk,
    _input_scale_dequant_view,
)


def test_linear_fp8_with_residual_testmode():
    device = "cuda"
    torch.manual_seed(123)

    batch_size = 8
    output_size = 128
    reduction_size = 768

    print(f"\n{'='*70}")
    print("Test: linear_fp8_with_residual_sm100 via PersistentKernel test_mode")
    print(f"  batch_size={batch_size} output_size={output_size} "
          f"reduction_size={reduction_size}")
    print(f"{'='*70}")

    g = torch.Generator(device=device).manual_seed(2025)
    x_bf16 = torch.randn(
        (batch_size, reduction_size), device=device, dtype=torch.bfloat16,
        generator=g,
    )
    w_bf16 = torch.randn(
        (output_size, reduction_size), device=device, dtype=torch.bfloat16,
        generator=g,
    )
    residual = torch.randn(
        (batch_size, output_size), device=device, dtype=torch.bfloat16,
        generator=g,
    )

    x_fp8, x_scale_packed = _input_scale_for_mpk(x_bf16)
    w_fp8, w_scale_strided = quantize_to_fp8_deepgemm_style(w_bf16)

    output = torch.zeros(
        (batch_size, output_size), device=device, dtype=torch.bfloat16
    )

    x_scale_ref_view = _input_scale_dequant_view(x_scale_packed, batch_size)
    ref = linear_fp8_with_residual_ref(
        x_fp8, x_scale_ref_view, w_fp8, w_scale_strided, residual
    )

    # Build PersistentKernel in test mode.
    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["num_workers"] = num_workers
    params["num_local_schedulers"] = num_schedulers
    params["mpi_rank"] = 0
    params["world_size"] = 1
    params["max_num_batched_tokens"] = batch_size
    params["max_num_batched_requests"] = batch_size
    # ``linear_fp8_with_residual_layer`` is used in the deepseek_v3 builder
    # alongside layers that consume ``qo_indptr_buffer``; provide a minimal
    # one to be safe in case any downstream metadata wiring requires it.
    qo_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    qo_indptr[batch_size] = batch_size
    params["meta_tensors"] = {"qo_indptr_buffer": qo_indptr}
    pk = PersistentKernel(**params)

    x_dt = pk.attach_input(x_fp8, name="input_fp8")
    xs_dt = pk.attach_input(x_scale_packed, name="input_scale")
    w_dt = pk.attach_input(w_fp8, name="weight_fp8")
    ws_dt = pk.attach_input(w_scale_strided, name="weight_scale")
    res_dt = pk.attach_input(residual, name="residual")
    out_dt = pk.attach_input(output, name="output")

    block_dim = (256, 1, 1) if pk.target_cc >= 90 else (128, 1, 1)
    pk.linear_fp8_with_residual_layer(
        input_fp8=x_dt,
        input_scale=xs_dt,
        weight_fp8=w_dt,
        weight_scale=ws_dt,
        residual=res_dt,
        output=out_dt,
        grid_dim=(1, 1, 1),
        block_dim=block_dim,
    )

    print("Compiling test kernel...")
    pk.compile(output_dir=THIS_DIR)
    print("Running test kernel...")
    pk.run_test_mode()
    torch.cuda.synchronize()

    print(f"\noutput[0, :8]:    {output[0, :8]}")
    print(f"reference[0, :8]: {ref[0, :8]}")
    max_diff = (output.float() - ref.float()).abs().max().item()
    print(f"Max abs diff:     {max_diff:.6f}")

    torch.testing.assert_close(output, ref, rtol=1e-2, atol=1e-2)
    print("\nPASSED: linear_fp8_with_residual_sm100 test_mode produces correct output")
    pk.finalize()


if __name__ == "__main__":
    test_linear_fp8_with_residual_testmode()
