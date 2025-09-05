import torch
import runtime_kernel_hopper

# torch.set_printoptions(sci_mode=False, profile="full")
torch.set_printoptions(sci_mode=False)

g = torch.Generator(device="cuda").manual_seed(1234)

reduction_sizes = [4096]
output_sizes = [16, 32, 64]
batch_size = 64

for reduction_size in reduction_sizes:
    for output_size in output_sizes:
        print(
            f"\n=== Testing output_size = {output_size} reduction_size = {reduction_size} ==="
        )

        x = torch.randn((batch_size, reduction_size), device="cuda", dtype=torch.bfloat16)
        w = torch.randn(
            (output_size, reduction_size), device="cuda", dtype=torch.bfloat16
        )
        residual = torch.randn(batch_size, output_size, device="cuda", dtype=torch.bfloat16)
        output = torch.empty(batch_size, output_size, device="cuda", dtype=torch.bfloat16)

        for i in range(batch_size):
            for j in range(reduction_size):
                x[i, j] = 0.01 * (i * reduction_size + j)

        runtime_kernel_hopper.linear(x, w, residual, output)
        torch_out = torch.matmul(x, torch.transpose(w, 0, 1))
        torch_out = torch_out + residual



        # print("torch_out.shape", torch_out.shape)
        # print(torch_out)
        # print("output.shape", output.shape)
        # print(output)

        print("Ratio (kernel / torch):")
        print(output / torch_out)

        # Warm-up
        # for _ in range(16):
        #     runtime_kernel_hopper.linear(x, w, residual, output)

        # torch.cuda.synchronize()
        # starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        #     enable_timing=True
        # )
        # repetitions = 1000
        # starter.record()
        # for rep in range(repetitions):
        #     runtime_kernel_hopper.linear(x, w, residual, output)
        # ender.record()
        # torch.cuda.synchronize()
        # total_time = starter.elapsed_time(ender)
        # avg_time = total_time / repetitions
        # print(f"Average time over {repetitions} runs: {avg_time:.6f} ms")
