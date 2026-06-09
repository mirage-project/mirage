"""Test: linear_fp8_layer (SM100) via PersistentKernel test_mode.

Builds a single-task FP8 linear layer through the full MPK compilation
pipeline, runs it once, and validates against ``linear_fp8_ref`` from
``pytorch_reference.py``.

Run:
    python tests/runtime_python/blackwell/sm100_linear_fp8/test_linear_fp8_testmode.py
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
    SCALE_PACK_SIZE,
    aligned_scale_outer_dim,
    packed_scale_k_for_reduction_size,
    quantize_to_fp8_deepgemm_style,
)
from pytorch_reference import linear_fp8_ref  # noqa: E402


def _input_scale_for_mpk(input_bf16):
    """Quantize ``input_bf16`` to FP8 + scales in the layout expected by
    the MPK linear_fp8_sm100 task.

    The MPK runtime TMA descriptor for SFA reads the input-scale DTensor
    with logical shape ``(packed_k, aligned_batch)`` row-major contiguous
    (see ``deepseek_v3/builder.py``).  ``quantize_to_fp8_deepgemm_style``
    produces a strided tensor with logical shape ``(batch, packed_k)`` and
    strides ``(1, aligned_batch)``; its underlying storage already has the
    desired ``(packed_k, aligned_batch)`` row-major contiguous layout, so
    we copy it through a fresh contiguous tensor.
    """
    batch, reduction = input_bf16.shape
    packed_k = packed_scale_k_for_reduction_size(reduction)
    aligned_batch = aligned_scale_outer_dim(batch)

    x_fp8, x_scale_strided = quantize_to_fp8_deepgemm_style(input_bf16)

    # Build a row-major contiguous (packed_k, aligned_batch) scale tensor.
    scale_packed = torch.zeros(
        (packed_k, aligned_batch), dtype=torch.uint32, device=input_bf16.device
    )
    # x_scale_strided[b, k] is the packed scale for batch row b, packed-k k.
    # We need scale_packed[k, b] = x_scale_strided[b, k].
    scale_packed[:, :batch] = x_scale_strided.t().contiguous()
    return x_fp8, scale_packed


def _input_scale_dequant_view(scale_packed, batch):
    """Reinterpret a ``(packed_k, aligned_batch)`` row-major packed scale
    tensor as a ``(batch, packed_k)`` strided tensor compatible with the
    common ``dequant_from_packed_ue8m0`` helper (which detects either
    row_major or deepgemm_col_major layout)."""
    packed_k = scale_packed.shape[0]
    aligned_batch = scale_packed.shape[1]
    # Underlying storage is contiguous (packed_k, aligned_batch) row-major.
    # That is exactly the deepgemm column-major layout for logical shape
    # (batch, packed_k) with strides (1, aligned_batch).
    return torch.as_strided(
        scale_packed, size=(batch, packed_k), stride=(1, aligned_batch)
    )


def test_linear_fp8_testmode():
    device = "cuda"
    torch.manual_seed(42)

    batch_size = 8
    output_size = 128
    reduction_size = 768

    print(f"\n{'='*70}")
    print("Test: linear_fp8_sm100 via PersistentKernel test_mode")
    print(f"  batch_size={batch_size} output_size={output_size} "
          f"reduction_size={reduction_size}")
    print(f"{'='*70}")

    g = torch.Generator(device=device).manual_seed(1234)
    x_bf16 = torch.randn(
        (batch_size, reduction_size), device=device, dtype=torch.bfloat16,
        generator=g,
    )
    w_bf16 = torch.randn(
        (output_size, reduction_size), device=device, dtype=torch.bfloat16,
        generator=g,
    )

    # Quantize inputs/weights into the layouts the MPK FP8 linear task expects.
    x_fp8, x_scale_packed = _input_scale_for_mpk(x_bf16)
    # weight_scale: deepgemm column-major (logical (M, packed_k), stride (1, aligned_M))
    w_fp8, w_scale_strided = quantize_to_fp8_deepgemm_style(w_bf16)

    output = torch.zeros(
        (batch_size, output_size), device=device, dtype=torch.bfloat16
    )

    # Reference uses the standard (batch, packed_k) view of the scale tensors.
    x_scale_ref_view = _input_scale_dequant_view(x_scale_packed, batch_size)
    ref = linear_fp8_ref(x_fp8, x_scale_ref_view, w_fp8, w_scale_strided)

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
    pk = PersistentKernel(**params)

    x_dt = pk.attach_input(x_fp8, name="input_fp8")
    xs_dt = pk.attach_input(x_scale_packed, name="input_scale")
    w_dt = pk.attach_input(w_fp8, name="weight_fp8")
    ws_dt = pk.attach_input(w_scale_strided, name="weight_scale")
    out_dt = pk.attach_input(output, name="output")

    block_dim = (256, 1, 1) if pk.target_cc >= 90 else (128, 1, 1)
    pk.linear_fp8_layer(
        input_fp8=x_dt,
        input_scale=xs_dt,
        weight_fp8=w_dt,
        weight_scale=ws_dt,
        output=out_dt,
        grid_dim=(1, 1, 1),
        block_dim=block_dim,
    )

    print("Compiling test kernel...")
    pk.compile(output_dir=THIS_DIR)
    print("Running test kernel...")
    pk()
    torch.cuda.synchronize()

    print(f"\noutput[0, :8]:    {output[0, :8]}")
    print(f"reference[0, :8]: {ref[0, :8]}")
    max_diff = (output.float() - ref.float()).abs().max().item()
    print(f"Max abs diff:     {max_diff:.6f}")

    torch.testing.assert_close(output, ref, rtol=1e-2, atol=1e-2)
    print("\nPASSED: linear_fp8_sm100 test_mode produces correct output")
    pk.finalize()


if __name__ == "__main__":
    test_linear_fp8_testmode()
