import torch
import runtime_kernel_hopper

torch.set_printoptions(sci_mode=False, profile="full")
# torch.set_printoptions(sci_mode=False)

g = torch.Generator(device="cuda").manual_seed(1234)

# reduction_sizes = [4096, 12288]
# output_sizes = [64, 128, 256]
reduction_sizes = [4096]
output_sizes = [64]
batch_size = 8

for reduction_size in reduction_sizes:
    for output_size in output_sizes:
        print(
            f"\n=== Testing output_size = {output_size} reduction_size = {reduction_size} ==="
        )

        x = torch.randn(
            (batch_size, reduction_size), device="cuda", dtype=torch.bfloat16
        )
        w = torch.randn(
            (output_size, reduction_size), device="cuda", dtype=torch.bfloat16
        )
        residual = torch.randn(
            batch_size, output_size, device="cuda", dtype=torch.bfloat16
        )
        output = torch.empty(
            batch_size, output_size, device="cuda", dtype=torch.bfloat16
        )

        for i in range(batch_size):
            for j in range(reduction_size):
                x[i, j] = 0.1
        for i in range(output_size):
            for j in range(reduction_size):
                w[i, j] = 0.1

        # swapAB version
        # (n, k) * (m, k) -> (n, m)
        # (64, 128) * (8, 128) -> (64, 8)
        # runtime_kernel_hopper.linear_swapAB(x, w, residual, output) # with residual
        runtime_kernel_hopper.linear_swapAB(x, w, None, output)  # without residual

        # normal version
        # runtime_kernel_hopper.linear(x, w, residual, output) # with residual
        # runtime_kernel_hopper.linear(x, w, None, output) # without residual

        torch_out = torch.matmul(x, torch.transpose(w, 0, 1))
        # torch_out = torch_out + residual # with residual

        print("Ratio (kernel / torch):")
        print(output / torch_out)
