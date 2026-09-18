import torch
import runtime_kernel_blackwell

from torch.nn import functional as F

torch.set_printoptions(sci_mode=False, profile="full")
# torch.set_printoptions(sci_mode=False)

g = torch.Generator(device="cuda").manual_seed(1234)

output_sizes = [768]
num_topks = [1]
batch_size = 1

for output_size in output_sizes:
    for num_topk in num_topks:
        print(
            f"\n=== Testing batch_size = {batch_size} output_size = {output_size} num_topk = {num_topk} ==="
        )

        input = torch.randn((batch_size, num_topk, output_size*2), device="cuda", dtype=torch.bfloat16, generator=g)
        output = torch.empty((batch_size, num_topk, output_size), device="cuda", dtype=torch.bfloat16)

        # MPK impl
        runtime_kernel_blackwell.silu_mul(input, output)

        # Reference impl 
        w1_output = F.silu(input[:, :, :output_size].to(torch.float))
        torch_output = w1_output * input[:, :, output_size:].to(torch.float)
        torch_output = torch_output.to(torch.bfloat16)
        
        torch.testing.assert_close(
            output,
            torch_output,
            rtol=1e-2,
            atol=1e-2,
        )
        print("Test passed!")

        # Warm-up
        for _ in range(16):
            runtime_kernel_blackwell.silu_mul(input, output)

        torch.cuda.synchronize()
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
            enable_timing=True
        )
        repetitions = 1000
        starter.record()
        for rep in range(repetitions):
            runtime_kernel_blackwell.silu_mul(input, output)
        ender.record()
        torch.cuda.synchronize()
        total_time = starter.elapsed_time(ender)
        avg_time = total_time / repetitions
        print(f"Average time over {repetitions} runs: {avg_time:.6f} ms")
