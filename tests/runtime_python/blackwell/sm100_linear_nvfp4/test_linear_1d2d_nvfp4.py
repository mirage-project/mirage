import torch
import runtime_kernel_blackwell_linear_nvfp4 as runtime_kernel_blackwell
from nvfp4_util import (
    interleave_sf_tensor,
    make_random_nvfp4_tensors,
    make_sequential_nvfp4_tensors,
    nvfp4_scaled_mm,
)

torch.set_printoptions(sci_mode=False, profile="full")

REDUCTION_SIZE = 768
OUTPUT_SIZE = 128
BATCH_SIZE = 1024 * 4

SMALL_M_VALUES = list(range(1, 9)) + [16, 24, 32, 48, 64, 80, 96, 112, 120, 128]
RTOL, ATOL = 1e-2, 1e-2


def select_mma_n(m: int) -> int:
    if m <= 8:   return 8
    if m <= 16:  return 16
    if m <= 32:  return 32
    if m <= 64:  return 64
    return 128


def check_close(output: torch.Tensor, ref: torch.Tensor, label: str) -> None:
    torch.testing.assert_close(output, ref.to(output.device), rtol=RTOL, atol=ATOL)
    print(f"{label} passed!")


def run_1d2d(x, w, x_sf, w_sf, residual=None):
    """Run the 1d2d kernel (M=BATCH_SIZE). SF in interleaved layout."""
    output = torch.empty(BATCH_SIZE, OUTPUT_SIZE, device="cuda", dtype=torch.float32)
    runtime_kernel_blackwell.linear_nvfp4_sm100_no_quantization(
        x, interleave_sf_tensor(x_sf),
        w, interleave_sf_tensor(w_sf),
        residual, output,
    )
    return output


if __name__ == "__main__":
    print(f"\n=== BATCH={BATCH_SIZE} N={OUTPUT_SIZE} K={REDUCTION_SIZE} ===\n")

    # ------------------------------------------------------------------
    # Tests 1–4: 1d2d path (M=4096), sequential and random, with/without residual
    # ------------------------------------------------------------------
    for label, use_sequential, use_residual in [
        ("Test 1: 1d2d sequential, no residual",  True,  False),
        ("Test 2: 1d2d random,     no residual",  False, False),
        ("Test 3: 1d2d sequential, residual",     True,  True),
        ("Test 4: 1d2d random,     residual",     False, True),
    ]:
        make = make_sequential_nvfp4_tensors if use_sequential else make_random_nvfp4_tensors
        x, w, x_sf, w_sf = make(BATCH_SIZE, OUTPUT_SIZE, REDUCTION_SIZE)
        residual = torch.randn(BATCH_SIZE, OUTPUT_SIZE, device="cuda", dtype=torch.float32) if use_residual else None

        ref, _ = nvfp4_scaled_mm(w, w_sf, x, x_sf, REDUCTION_SIZE, residual=residual)
        output = run_1d2d(x, w, x_sf, w_sf, residual)
        check_close(output, ref, label)

    # ------------------------------------------------------------------
    # Test 5: swapAB path (M=1..128), no residual, against nvfp4_scaled_mm
    # Use float32 activations so the GPU quantizer produces consistent packed data
    # and SF in both layouts from the same input.
    # ------------------------------------------------------------------
    for m in SMALL_M_VALUES:
        x_fp32 = torch.randn(m, REDUCTION_SIZE, device="cuda", dtype=torch.float32)
        _, w, _, w_sf = make_random_nvfp4_tensors(OUTPUT_SIZE, OUTPUT_SIZE, REDUCTION_SIZE)
        w_sf_interleaved = interleave_sf_tensor(w_sf)
        mma_n = select_mma_n(m)

        # Per-tile SF for the swapAB kernel
        x_q, x_sf_pertile = runtime_kernel_blackwell.quantize_nvfp4_sm100(x_fp32, mma_n)

        # Interleaved SF for the reference: inverse of interleave_sf_tensor is the same
        # permutation (dims 1 and 3 are swapped in both directions).
        _, x_sf_il_5d = runtime_kernel_blackwell.quantize_nvfp4_sm100(x_fp32, 0)
        padded_m = x_sf_il_5d.shape[0] * 128
        sf_k = x_sf_il_5d.shape[1] * 4
        x_sf_2d = x_sf_il_5d.permute(0, 3, 2, 1, 4).reshape(padded_m, sf_k)[:m]

        ref, _ = nvfp4_scaled_mm(w, w_sf, x_q[:m], x_sf_2d, REDUCTION_SIZE, residual=None)

        output = torch.empty(m, OUTPUT_SIZE, device="cuda", dtype=torch.float32)
        runtime_kernel_blackwell.linear_nvfp4_sm100_no_quantization(
            x_q, x_sf_pertile, w, w_sf_interleaved, None, output
        )
        check_close(output, ref, f"Test 5 (M={m})")

    print("Test 5 passed!")

    # ------------------------------------------------------------------
    # Test 6: auto-quantize entry point matches explicit quantize + no-quant
    # ------------------------------------------------------------------
    for m in SMALL_M_VALUES:
        x = torch.randn(m, REDUCTION_SIZE, device="cuda", dtype=torch.float32)
        _, w, _, w_sf = make_random_nvfp4_tensors(OUTPUT_SIZE, OUTPUT_SIZE, REDUCTION_SIZE)
        w_sf_interleaved = interleave_sf_tensor(w_sf)

        mma_n = select_mma_n(m)
        x_q, x_sf = runtime_kernel_blackwell.quantize_nvfp4_sm100(x, mma_n)

        explicit = torch.empty(m, OUTPUT_SIZE, device="cuda", dtype=torch.float32)
        runtime_kernel_blackwell.linear_nvfp4_sm100_no_quantization(
            x_q, x_sf, w, w_sf_interleaved, None, explicit
        )
        auto = torch.empty_like(explicit)
        runtime_kernel_blackwell.linear_nvfp4_sm100(x, w, w_sf_interleaved, None, auto)

        assert torch.equal(auto, explicit), f"Test 6 mismatch at M={m}"

    print("Test 6 passed!")

    # ------------------------------------------------------------------
    # Performance (1d2d path, M=4096)
    # ------------------------------------------------------------------
    WARMUP, REPS = 10, 100
    x, w, x_sf, w_sf = make_random_nvfp4_tensors(BATCH_SIZE, OUTPUT_SIZE, REDUCTION_SIZE)
    x_sf_il = interleave_sf_tensor(x_sf)
    w_sf_il = interleave_sf_tensor(w_sf)
    output  = torch.empty(BATCH_SIZE, OUTPUT_SIZE, device="cuda", dtype=torch.float32)

    print("\n--- PERFORMANCE ---")

    for _ in range(WARMUP):
        runtime_kernel_blackwell.linear_nvfp4_sm100_no_quantization(x, x_sf_il, w, w_sf_il, None, output)
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(REPS):
        runtime_kernel_blackwell.linear_nvfp4_sm100_no_quantization(x, x_sf_il, w, w_sf_il, None, output)
    end.record()
    torch.cuda.synchronize()
    avg_ms = start.elapsed_time(end) / REPS
    tflops = 2 * BATCH_SIZE * OUTPUT_SIZE * REDUCTION_SIZE / (avg_ms * 1e-3) / 1e12
    print(f"[Custom (NVFP4)]           {avg_ms:.6f} ms  ({tflops:.2f} TFLOP/s)")

    for _ in range(WARMUP):
        nvfp4_scaled_mm(w, w_sf, x, x_sf, REDUCTION_SIZE)
    times = [nvfp4_scaled_mm(w, w_sf, x, x_sf, REDUCTION_SIZE)[1] for _ in range(REPS)]
    avg_ms_ref = sum(times) / REPS
    tflops_ref = 2 * BATCH_SIZE * OUTPUT_SIZE * REDUCTION_SIZE / (avg_ms_ref * 1e-3) / 1e12
    print(f"[torch._scaled_mm (NVFP4)] {avg_ms_ref:.6f} ms  ({tflops_ref:.2f} TFLOP/s)")
