import torch
import runtime_kernel_blackwell_linear_nvfp4 as runtime_kernel_blackwell
from nvfp4_util import make_sequential_nvfp4_tensors, nvfp4_block_scaled_matmul, nvfp4_scaled_mm, make_random_nvfp4_tensors, interleave_sf_tensor, make_unit_scale_factors

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

        # Test 1: Random nvfp4 tensors
        x, w, x_sf, w_sf = make_sequential_nvfp4_tensors(
            batch_size, output_size, reduction_size
        )
        x_sf_interleaved = interleave_sf_tensor(x_sf)
        w_sf_interleaved = interleave_sf_tensor(w_sf)
        residual = torch.randn(batch_size, output_size, device="cuda", dtype=torch.float32)
        output = torch.empty(batch_size, output_size, device="cuda", dtype=torch.float32)

        if not has_residual:
            residual = None
        torch_out, _ = nvfp4_scaled_mm(w, w_sf, x, x_sf, reduction_size, residual=residual)
        runtime_kernel_blackwell.linear_nvfp4_1d2d_sm100(x, x_sf_interleaved, w, w_sf_interleaved, residual, output)
        torch_out_cuda = torch_out.to(output.device)
        error = (output - torch_out_cuda).abs()
        
        print("\n--- TEST 1 ---")
        print("max error:", error.max())
        print("mean error:", error.mean())
        print("relative max:", (error / torch_out_cuda.abs().clamp_min(1e-5)).max())
        # print(output[0])
        # print(torch_out_cuda[0])
        torch.testing.assert_close(
            output,
            torch_out_cuda,
            rtol=1e-2,
            atol=1e-2,
        )
        print("\nTest 1 passed!")
        
        # Test 2: Random nvfp4 tensors
        x, w, x_sf, w_sf = make_random_nvfp4_tensors(
            batch_size, output_size, reduction_size
        )
        x_sf_interleaved = interleave_sf_tensor(x_sf)
        w_sf_interleaved = interleave_sf_tensor(w_sf)
        residual = torch.randn(batch_size, output_size, device="cuda", dtype=torch.float32)
        output = torch.empty(batch_size, output_size, device="cuda", dtype=torch.float32)

        if not has_residual:
            residual = None
        torch_out, _ = nvfp4_scaled_mm(w, w_sf, x, x_sf, reduction_size, residual=residual)
        runtime_kernel_blackwell.linear_nvfp4_1d2d_sm100(x, x_sf_interleaved, w, w_sf_interleaved, residual, output)
        torch_out_cuda = torch_out.to(output.device)
        error = (output - torch_out_cuda).abs()
        
        print("\n--- TEST 2 ---")
        print("max error:", error.max())
        print("mean error:", error.mean())
        print("relative max:", (error / torch_out_cuda.abs().clamp_min(1e-5)).max())
        torch.testing.assert_close(
            output,
            torch_out_cuda,
            rtol=1e-2,
            atol=1e-2,
        )
        print("\nTest 2 passed!")

        # Test 3: Sequential nvfp4 tensors with residual
        x, w, x_sf, w_sf = make_sequential_nvfp4_tensors(
            batch_size, output_size, reduction_size
        )
        x_sf_interleaved = interleave_sf_tensor(x_sf)
        w_sf_interleaved = interleave_sf_tensor(w_sf)
        residual = torch.randn(batch_size, output_size, device="cuda", dtype=torch.float32)
        output = torch.empty(batch_size, output_size, device="cuda", dtype=torch.float32)

        torch_out, _ = nvfp4_scaled_mm(w, w_sf, x, x_sf, reduction_size, residual=residual)
        runtime_kernel_blackwell.linear_nvfp4_1d2d_sm100(x, x_sf_interleaved, w, w_sf_interleaved, residual, output)
        torch_out_cuda = torch_out.to(output.device)
        error = (output - torch_out_cuda).abs()

        print("\n--- TEST 3 ---")
        print("max error:", error.max())
        print("mean error:", error.mean())
        print("relative max:", (error / torch_out_cuda.abs().clamp_min(1e-5)).max())
        torch.testing.assert_close(
            output,
            torch_out_cuda,
            rtol=1e-2,
            atol=1e-2,
        )
        print("\nTest 3 passed!")

        # Test 4: Random nvfp4 tensors with residual
        x, w, x_sf, w_sf = make_random_nvfp4_tensors(
            batch_size, output_size, reduction_size
        )
        x_sf_interleaved = interleave_sf_tensor(x_sf)
        w_sf_interleaved = interleave_sf_tensor(w_sf)
        residual = torch.randn(batch_size, output_size, device="cuda", dtype=torch.float32)
        output = torch.empty(batch_size, output_size, device="cuda", dtype=torch.float32)

        torch_out, _ = nvfp4_scaled_mm(w, w_sf, x, x_sf, reduction_size, residual=residual)
        runtime_kernel_blackwell.linear_nvfp4_1d2d_sm100(x, x_sf_interleaved, w, w_sf_interleaved, residual, output)
        torch_out_cuda = torch_out.to(output.device)
        error = (output - torch_out_cuda).abs()

        print("\n--- TEST 4 ---")
        print("max error:", error.max())
        print("mean error:", error.mean())
        print("relative max:", (error / torch_out_cuda.abs().clamp_min(1e-5)).max())
        torch.testing.assert_close(
            output,
            torch_out_cuda,
            rtol=1e-2,
            atol=1e-2,
        )
        print("\nTest 4 passed!")

        # Performance Tests
        torch.cuda.synchronize()
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        WARM_UP = 10
        REPETITIONS = 50
        print("\n--- PERFORMANCE ---")        

        # Custom implementation
        for _ in range(WARM_UP):
            runtime_kernel_blackwell.linear_nvfp4_1d2d_sm100(x, x_sf_interleaved, w, w_sf_interleaved, residual, output)
        starter.record()
        for rep in range(REPETITIONS):
            runtime_kernel_blackwell.linear_nvfp4_1d2d_sm100(x, x_sf_interleaved, w, w_sf_interleaved, residual, output)
        ender.record()
        torch.cuda.synchronize()
        avg_time = starter.elapsed_time(ender) / REPETITIONS
        tflops = 2 * batch_size * output_size * reduction_size / (avg_time * 1e-3) / 1e12
        print(f"[Custom (NVFP4)]                        Average time over {REPETITIONS} runs: {avg_time:.6f} ms  ({tflops:.2f} TFLOP/s)")

        # torch._scaled_mm implementation
        for _ in range(WARM_UP):
            nvfp4_scaled_mm(w, w_sf, x, x_sf, reduction_size, residual=residual)
        scaled_mm_times = []
        for rep in range(REPETITIONS):
            _, elapsed = nvfp4_scaled_mm(w, w_sf, x, x_sf, reduction_size, residual=residual)
            scaled_mm_times.append(elapsed)
        avg_time_scaled_mm = sum(scaled_mm_times) / REPETITIONS
        tflops_scaled_mm = 2 * batch_size * output_size * reduction_size / (avg_time_scaled_mm * 1e-3) / 1e12
        print(f"[Reference - torch._scaled_mm (NVFP4)]  Average time over {REPETITIONS} runs: {avg_time_scaled_mm:.6f} ms  ({tflops_scaled_mm:.2f} TFLOP/s)")

        # torch.matmul implementation
        for _ in range(WARM_UP):
            nvfp4_block_scaled_matmul(w, w_sf, x, x_sf, reduction_size, residual=residual)
        ref_times = []
        for rep in range(REPETITIONS):
            _, elapsed = nvfp4_block_scaled_matmul(w, w_sf, x, x_sf, reduction_size, residual=residual)
            ref_times.append(elapsed)
        avg_time_ref = sum(ref_times) / REPETITIONS
        tflops_ref = 2 * batch_size * output_size * reduction_size / (avg_time_ref * 1e-3) / 1e12
        print(f"[Reference - torch.matmul (FP32)]       Average time over {REPETITIONS} runs: {avg_time_ref:.6f} ms  ({tflops_ref:.2f} TFLOP/s)\n")

        