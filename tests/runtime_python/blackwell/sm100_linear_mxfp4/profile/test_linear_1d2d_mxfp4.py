import _runtime_path  # noqa: F401
import torch
import runtime_kernel_blackwell_linear_mxfp4 as runtime_kernel_blackwell
from mxfp4_util import deinterleave_sf_tensor, mxfp4_scaled_mm

torch.set_printoptions(sci_mode=False, profile="full")

REDUCTION_SIZE = 768
OUTPUT_SIZE = 128
BATCH_SIZE = 1024 * 4
SMALL_M_VALUES = [1, 2, 4, 8, 16, 32, 64, 96, 128]
RTOL, ATOL = 1e-2, 1e-2


def check_close(output: torch.Tensor, ref: torch.Tensor, label: str) -> None:
    torch.testing.assert_close(output, ref.to(output.device), rtol=RTOL, atol=ATOL)
    print(f"{label} passed!")


def select_mma_n(m: int) -> int:
    if m <= 8:
        return 8
    if m <= 16:
        return 16
    if m <= 32:
        return 32
    if m <= 64:
        return 64
    return 128


def quantize_interleaved(x_fp32: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    q, sf = runtime_kernel_blackwell.quantize_mxfp4_sm100(x_fp32, 0)
    return q, sf


def run_no_quant(x_q, x_sf, w_q, w_sf_il, batch_size: int, residual=None):
    output = torch.empty(batch_size, OUTPUT_SIZE, device="cuda", dtype=torch.float32)
    runtime_kernel_blackwell.linear_mxfp4_sm100_no_quantization(
        x_q, x_sf, w_q, w_sf_il, residual, output
    )
    return output


if __name__ == "__main__":
    torch.manual_seed(0)
    print(f"\n=== BATCH={BATCH_SIZE} N={OUTPUT_SIZE} K={REDUCTION_SIZE} ===\n")

    for label, use_residual in [
        ("Test 1: random, no residual", False),
        ("Test 2: random, residual", True),
    ]:
        x_fp32 = torch.randn(BATCH_SIZE, REDUCTION_SIZE, device="cuda", dtype=torch.float32)
        w_fp32 = torch.randn(OUTPUT_SIZE, REDUCTION_SIZE, device="cuda", dtype=torch.float32)
        x_q, x_sf_il = quantize_interleaved(x_fp32)
        w_q, w_sf_il = quantize_interleaved(w_fp32)
        x_sf = deinterleave_sf_tensor(x_sf_il, BATCH_SIZE)
        w_sf = deinterleave_sf_tensor(w_sf_il, OUTPUT_SIZE)
        residual = (
            torch.randn(BATCH_SIZE, OUTPUT_SIZE, device="cuda", dtype=torch.float32)
            if use_residual
            else None
        )

        ref = mxfp4_scaled_mm(w_q[:OUTPUT_SIZE], w_sf, x_q[:BATCH_SIZE], x_sf, residual=residual)
        output = run_no_quant(x_q, x_sf_il, w_q[:OUTPUT_SIZE], w_sf_il, BATCH_SIZE, residual)
        check_close(output, ref, label)

    x_fp32 = torch.randn(BATCH_SIZE, REDUCTION_SIZE, device="cuda", dtype=torch.float32)
    w_fp32 = torch.randn(OUTPUT_SIZE, REDUCTION_SIZE, device="cuda", dtype=torch.float32)
    x_q, x_sf_il = quantize_interleaved(x_fp32)
    w_q, w_sf_il = quantize_interleaved(w_fp32)

    explicit = run_no_quant(x_q, x_sf_il, w_q[:OUTPUT_SIZE], w_sf_il, BATCH_SIZE, None)
    auto = torch.empty_like(explicit)
    runtime_kernel_blackwell.linear_mxfp4_sm100(
        x_fp32, w_q[:OUTPUT_SIZE], w_sf_il, None, auto
    )
    torch.testing.assert_close(auto, explicit, rtol=0.0, atol=0.0)
    print("Test 3: auto-quantize matches explicit quantize passed!")

    for m in SMALL_M_VALUES:
        x_fp32 = torch.randn(m, REDUCTION_SIZE, device="cuda", dtype=torch.float32)
        w_fp32 = torch.randn(OUTPUT_SIZE, REDUCTION_SIZE, device="cuda", dtype=torch.float32)
        mma_n = select_mma_n(m)

        x_q_tiled, x_sf_tiled = runtime_kernel_blackwell.quantize_mxfp4_sm100(x_fp32, mma_n)
        _, x_sf_il = runtime_kernel_blackwell.quantize_mxfp4_sm100(x_fp32, 0)
        w_q, w_sf_il = quantize_interleaved(w_fp32)

        x_sf = deinterleave_sf_tensor(x_sf_il, m)
        w_sf = deinterleave_sf_tensor(w_sf_il, OUTPUT_SIZE)

        ref = mxfp4_scaled_mm(
            w_q[:OUTPUT_SIZE],
            w_sf,
            x_q_tiled[:m],
            x_sf,
            residual=None,
        )
        output = run_no_quant(x_q_tiled, x_sf_tiled, w_q[:OUTPUT_SIZE], w_sf_il, m, None)
        check_close(output, ref, f"Test 4 (small batch M={m})")

    print("Test 4: small-batch swapAB path passed!")

    for m in SMALL_M_VALUES:
        x_fp32 = torch.randn(m, REDUCTION_SIZE, device="cuda", dtype=torch.float32)
        w_fp32 = torch.randn(OUTPUT_SIZE, REDUCTION_SIZE, device="cuda", dtype=torch.float32)
        mma_n = select_mma_n(m)

        x_q_tiled, x_sf_tiled = runtime_kernel_blackwell.quantize_mxfp4_sm100(x_fp32, mma_n)
        w_q, w_sf_il = quantize_interleaved(w_fp32)

        explicit = run_no_quant(x_q_tiled, x_sf_tiled, w_q[:OUTPUT_SIZE], w_sf_il, m, None)
        auto = torch.empty_like(explicit)
        runtime_kernel_blackwell.linear_mxfp4_sm100(
            x_fp32, w_q[:OUTPUT_SIZE], w_sf_il, None, auto
        )
        assert torch.equal(auto, explicit), f"Test 5 mismatch at M={m}"

    print("Test 5: small-batch auto-quantize matches explicit quantize passed!")

    print("\nAll tests passed!")
