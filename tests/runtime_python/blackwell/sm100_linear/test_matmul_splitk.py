import torch
import runtime_kernel_blackwell

torch.set_printoptions(sci_mode=False, profile="full")
# torch.set_printoptions(sci_mode=False)

g = torch.Generator(device="cuda").manual_seed(1234)

reduction_sizes = [2048]
output_sizes = [128]
batch_size = 1

has_residual = False

for reduction_size in reduction_sizes:
    for output_size in output_sizes:
        print(
            f"\n=== Testing batch_size = {batch_size} output_size = {output_size} reduction_size = {reduction_size} has_residual = {has_residual} ==="
        )

        x = torch.randn((batch_size, reduction_size), device="cuda", dtype=torch.bfloat16)
        w = torch.randn(
            (output_size, reduction_size), device="cuda", dtype=torch.bfloat16
        )
        acc_output = torch.randn(batch_size, output_size, device="cuda", dtype=torch.bfloat16)
        torch_acc_output = acc_output.clone()

        if not has_residual:
            residual = None
        runtime_kernel_blackwell.linear_splitk_sm100(x, w, residual, acc_output) # with residual and swapAB
        torch_out = torch.matmul(x, torch.transpose(w, 0, 1))
        torch_acc_out = torch_out + torch_acc_output
        
        torch.testing.assert_close(
            acc_output,
            torch_acc_out,
            rtol=1e-2,
            atol=1e-2,
        )
        print("Test passed!")

        # Warm-up
        for _ in range(16):
            runtime_kernel_blackwell.linear_splitk_sm100(x, w, residual, acc_output)

        torch.cuda.synchronize()
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
            enable_timing=True
        )
        repetitions = 1000
        starter.record()
        for rep in range(repetitions):
            runtime_kernel_blackwell.linear_splitk_sm100(x, w, residual, acc_output)
        ender.record()
        torch.cuda.synchronize()
        total_time = starter.elapsed_time(ender)
        avg_time = total_time / repetitions
        print(f"Average time over {repetitions} runs: {avg_time:.6f} ms")
