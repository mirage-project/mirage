import torch
import runtime_kernel_blackwell_linear_nvfp4 as runtime_kernel_blackwell
from nvfp4_util import make_sequential_nvfp4_tensors, nvfp4_block_scaled_matmul, make_random_nvfp4_tensors, interleave_sf_tensor, make_unit_scale_factors

torch.set_printoptions(sci_mode=False, profile="full")

# BATCH_SIZE must be divisible by MMA_M = 128
# OUTPUT_SIZE must be divisible by MMA_N = 128
# REDUCTION_SIZE must be divisible by bK = 256
reduction_sizes = [4096]
output_sizes = [4096]
batch_size = 4096
has_residual = False

for reduction_size in reduction_sizes:
    for output_size in output_sizes:
        print(
            f"\n=== Testing batch_size = {batch_size} output_size = {output_size} reduction_size = {reduction_size} has_residual = {has_residual} ==="
        )

        # make_sequential_nvfp4_tensors | make_random_nvfp4_tensors
        x, w, x_sf, w_sf = make_random_nvfp4_tensors(
            batch_size, output_size, reduction_size
        )
        x_sf_interleaved = interleave_sf_tensor(x_sf)
        w_sf_interleaved = interleave_sf_tensor(w_sf)
        
        residual = torch.randn(batch_size, output_size, device="cuda", dtype=torch.float32)
        output = torch.empty(batch_size, output_size, device="cuda", dtype=torch.float32)

        if not has_residual:
            residual = None
        print("Launching reference implementation")
        torch_out = nvfp4_block_scaled_matmul(w, w_sf, x, x_sf, reduction_size, residual=residual)
        print(torch_out[0, :10])
        print("Launching custom implementation")
        runtime_kernel_blackwell.linear_nvfp4_1d2d_sm100(x, x_sf_interleaved, w, w_sf_interleaved, residual, output)
        print(output[0, :10])        
        torch.testing.assert_close(
            output,
            torch_out.to(output.device),
            rtol=1e-2,
            atol=10.0,
        )
        print("Test 1 passed!")
        
        x, w, x_sf, w_sf = make_random_nvfp4_tensors(
            batch_size, output_size, reduction_size
        )
        x_sf_interleaved = interleave_sf_tensor(x_sf)
        w_sf_interleaved = interleave_sf_tensor(w_sf)
        
        residual = torch.randn(batch_size, output_size, device="cuda", dtype=torch.float32)
        output = torch.empty(batch_size, output_size, device="cuda", dtype=torch.float32)

        if not has_residual:
            residual = None
        print("Launching reference implementation")
        torch_out = nvfp4_block_scaled_matmul(w, w_sf, x, x_sf, reduction_size, residual=residual)
        # print(torch_out[0:2])
        print("Launching custom implementation")
        runtime_kernel_blackwell.linear_nvfp4_1d2d_sm100(x, x_sf_interleaved, w, w_sf_interleaved, residual, output)
        # print(output[0:2])
        
        torch.testing.assert_close(
            output,
            torch_out.to(output.device),
            rtol=1e-2,
            atol=10.0,
        )
        print("Test 2 passed!")

        # Warm-up
        for _ in range(3):
            runtime_kernel_blackwell.linear_nvfp4_1d2d_sm100(x, x_sf_interleaved, w, w_sf_interleaved, residual, output)

        torch.cuda.synchronize()
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        repetitions = 10

        starter.record()
        for rep in range(repetitions):
            runtime_kernel_blackwell.linear_nvfp4_1d2d_sm100(x, x_sf_interleaved, w, w_sf_interleaved, residual, output)
        ender.record()
        torch.cuda.synchronize()
        avg_time = starter.elapsed_time(ender) / repetitions
        tflops = 2 * batch_size * output_size * reduction_size / (avg_time * 1e-3) / 1e12
        print(f"[Custom]    Average time over {repetitions} runs: {avg_time:.6f} ms  ({tflops:.2f} TFLOP/s)")

        starter.record()
        for rep in range(repetitions):
            nvfp4_block_scaled_matmul(w, w_sf, x, x_sf, reduction_size, residual=residual)
        ender.record()
        torch.cuda.synchronize()
        avg_time_ref = starter.elapsed_time(ender) / repetitions
        tflops_ref = 2 * batch_size * output_size * reduction_size / (avg_time_ref * 1e-3) / 1e12
        print(f"[Reference] Average time over {repetitions} runs: {avg_time_ref:.6f} ms  ({tflops_ref:.2f} TFLOP/s)")