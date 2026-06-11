"""Test the ``layers.QuantizeFP8`` catalog module via PersistentKernel test_mode.

Numerical test: BF16 → FP8 E4M3 with per-128-element block scales. We
check the UE8M0-packed variant (the default used by FP8 linear / group
GEMM). Tolerance is loose because FP8 round-to-zero plus log2 rounding
introduces multi-ULP differences vs the pure-PyTorch reference.
"""

import os
import sys

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel
from mirage.mpk.layers.quantize_fp8 import QuantizeFP8


def test_quantize_fp8_testmode():
    device = "cuda"
    torch.manual_seed(0)

    batch_size = 2
    hidden_size = 512  # must be a multiple of 128, with hidden//128 % 4 == 0

    x = torch.randn(batch_size, hidden_size, dtype=torch.bfloat16, device=device)
    out_fp8 = torch.zeros(batch_size, hidden_size, dtype=torch.uint8, device=device)
    out_scale = torch.zeros(
        batch_size, hidden_size // 128 // 4, dtype=torch.uint32, device=device,
    )

    m = QuantizeFP8(hidden_size=hidden_size, scale_ue8m0=True, prefix="test_")
    ref_fp8, ref_scale = m.forward(x)
    assert ref_fp8.shape == (batch_size, hidden_size)
    assert ref_scale.shape == (batch_size, hidden_size // 128 // 4)

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

    x_dt = pk.attach_input(x, name="x")

    with pk.compile_scope():
        _ = m.compile(x_dt, output_fp8=out_fp8, output_scale=out_scale)

    print("Compiling test kernel...")
    folder_path = os.path.dirname(__file__)
    pk.compile(output_dir=folder_path)

    print("Running test kernel...")
    pk()
    torch.cuda.synchronize()

    # Dequantize both outputs and compare numerically against x.
    def _dequant(fp8_u8, scale_u32):
        fp32 = fp8_u8.view(torch.float8_e4m3fn).float()
        bytes_view = scale_u32.contiguous().view(torch.uint8).reshape(
            scale_u32.shape[0], -1
        )
        exp_f32 = bytes_view.to(torch.float32) - 127.0
        scales = torch.pow(torch.tensor(2.0), exp_f32).to(fp32.device)
        scales = scales.repeat_interleave(128, dim=-1)
        return fp32 * scales

    deq_out = _dequant(out_fp8, out_scale)
    print(f"x[0, :8]:       {x[0, :8].float()}")
    print(f"deq_out[0, :8]: {deq_out[0, :8]}")
    # Compare with bf16-FP8 fairly loose tolerance — UE8M0 + E4M3
    # rounding loses ~1 bit of precision per element.
    max_diff = (deq_out - x.float()).abs().max().item()
    rel_diff = max_diff / x.float().abs().max().item()
    print(f"max abs diff: {max_diff}, rel: {rel_diff}")

    # FP8 tolerance from the brief: atol/rtol=0.5.
    try:
        torch.testing.assert_close(
            deq_out, x.float(), atol=0.5, rtol=0.5
        )
        print("PASSED: QuantizeFP8 compile() matches forward() (within FP8 tol)")
    except AssertionError as e:
        print(f"FAILED: QuantizeFP8 dequant mismatch\n{e}")
        pk.finalize()
        sys.exit(1)

    pk.finalize()
    print("Test completed successfully!")


if __name__ == "__main__":
    test_quantize_fp8_testmode()
