import torch
import runtime_kernel_blackwell_linear_nvfp4 as runtime_kernel_blackwell
from nvfp4_util import make_unit_scale_factors, nvfp4_block_scaled_matmul

torch.set_printoptions(sci_mode=False, profile="full")

g = torch.Generator(device="cuda").manual_seed(1234)

reduction_sizes = [768]
output_sizes = [128]
batch_size = 1
has_residual = False

for reduction_size in reduction_sizes:
    for output_size in output_sizes:
        print(
            f"\n=== Testing batch_size = {batch_size} output_size = {output_size} reduction_size = {reduction_size} has_residual = {has_residual} ==="
        )

        x = torch.randint(0, 256, (batch_size, reduction_size // 2), device="cuda", dtype=torch.uint8)
        x_sf = make_unit_scale_factors(batch_size, reduction_size // 16, device="cuda")
        w = torch.randint(0, 256, (output_size, reduction_size // 2), device="cuda", dtype=torch.uint8)
        w_sf = make_unit_scale_factors(output_size, reduction_size // 16, device="cuda")
        residual = torch.randn(batch_size, output_size, device="cuda", dtype=torch.float32)
        output = torch.empty(batch_size, output_size, device="cuda", dtype=torch.float32)

        if not has_residual:
            residual = None
        runtime_kernel_blackwell.linear_nvfp4_1d2d_sm100(x, x_sf, w, w_sf, residual, output)
        torch_out = nvfp4_block_scaled_matmul(w, w_sf, x, x_sf, reduction_size, residual=residual)
        
        torch.testing.assert_close(
            output,
            torch_out,
            rtol=1e-2,
            atol=1e-2,
        )
        print("Test passed!")

        # Warm-up 
        for _ in range(16):
            runtime_kernel_blackwell.linear_nvfp4_1d2d_sm100(x, x_sf, w, w_sf, residual, output)

        torch.cuda.synchronize()
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        repetitions = 1000
        starter.record()
        for rep in range(repetitions):
            runtime_kernel_blackwell.linear_nvfp4_1d2d_sm100(x, x_sf, w, w_sf, residual, output)
        ender.record()
        torch.cuda.synchronize()
        total_time = starter.elapsed_time(ender)
        avg_time = total_time / repetitions
        print(f"Average time over {repetitions} runs: {avg_time:.6f} ms")