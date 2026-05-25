"""Basic correctness test for the MXFP4 1d2d kernels (1SM and 2SM)."""

import _runtime_path  # noqa: F401
import torch
import runtime_kernel_blackwell_linear_mxfp4 as runtime_kernel_blackwell
from mxfp4_util import mxfp4_reference_matmul

torch.manual_seed(0)
torch.set_printoptions(sci_mode=False)

# Modest shape: large enough to exercise multi-stage pipeline, small enough that
# the CPU/dequant reference fits comfortably in memory.
REDUCTION_SIZE = 1024
OUTPUT_SIZE = 256
BATCH_SIZE = 256

RTOL, ATOL = 5e-2, 5e-2


def _quantize(t):
    return runtime_kernel_blackwell.quantize_mxfp4_sm100(t, 0)


def _run(x_q, x_sf, w_q, w_sf, residual, use_2sm):
    return runtime_kernel_blackwell.linear_mxfp4_sm100_no_quantization(
        x_q, x_sf, w_q, w_sf, residual, use_2sm
    )


def _make_random_input():
    return torch.randn(BATCH_SIZE, REDUCTION_SIZE, device="cuda",
                       dtype=torch.float32) * 0.5


def _make_random_weight():
    return torch.randn(OUTPUT_SIZE, REDUCTION_SIZE, device="cuda",
                       dtype=torch.float32) * 0.5


def main():
    for use_2sm, label in [(False, "1SM"), (True, "2SM")]:
        for use_residual in (False, True):
            x_fp32 = _make_random_input()
            w_fp32 = _make_random_weight()
            x_q, x_sf = _quantize(x_fp32)
            w_q, w_sf = _quantize(w_fp32)

            residual = (
                torch.randn(BATCH_SIZE, OUTPUT_SIZE, device="cuda",
                            dtype=torch.bfloat16)
                if use_residual else None
            )

            ref = mxfp4_reference_matmul(x_q[:BATCH_SIZE], x_sf, w_q, w_sf,
                                         REDUCTION_SIZE, residual=residual)
            out = _run(x_q, x_sf, w_q, w_sf, residual, use_2sm)
            torch.testing.assert_close(out, ref, rtol=RTOL, atol=ATOL)
            tag = f"{label}{' + residual' if use_residual else ''}"
            print(f"PASS  {tag:<18} (B={BATCH_SIZE} N={OUTPUT_SIZE} K={REDUCTION_SIZE})")

    print("\nAll MXFP4 1d2d basic tests passed.")


if __name__ == "__main__":
    main()
