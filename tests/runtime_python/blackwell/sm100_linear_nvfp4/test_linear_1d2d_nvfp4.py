import torch
import runtime_kernel_blackwell_linear_nvfp4 as runtime_kernel_blackwell
from nvfp4_util import make_sequential_nvfp4_tensors, nvfp4_block_scaled_matmul, make_random_nvfp4_tensors, interleave_sf_tensor, make_unit_scale_factors

torch.set_printoptions(sci_mode=False, profile="full")

# Minimum is 256
# MMA_M = 128, N 128, K 64
# 16B
# 16 SF
# 16 FP4
# 128 x 4b
# 128 x 16b
reduction_sizes = [256*12]
output_sizes = [128*12]
batch_size = 128*12
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
            atol=1e-2,
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
            atol=1e-2,
        )
        print("Test 2 passed!")

        # Warm-up 
        for _ in range(16):
            runtime_kernel_blackwell.linear_nvfp4_1d2d_sm100(x, x_sf_interleaved, w, w_sf_interleaved, residual, output)

        torch.cuda.synchronize()
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        repetitions = 1000
        starter.record()
        for rep in range(repetitions):
            runtime_kernel_blackwell.linear_nvfp4_1d2d_sm100(x, x_sf_interleaved, w, w_sf_interleaved, residual, output)
        ender.record()
        torch.cuda.synchronize()
        total_time = starter.elapsed_time(ender)
        avg_time = total_time / repetitions
        print(f"Average time over {repetitions} runs: {avg_time:.6f} ms")