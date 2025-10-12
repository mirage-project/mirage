import torch
import runtime_kernel_blackwell

from torch.nn import functional as F

torch.set_printoptions(sci_mode=False, profile="full")
# torch.set_printoptions(sci_mode=False)

g = torch.Generator(device="cuda").manual_seed(1234)

reduction_sizes = [2048]
num_experts = [128]
num_topks = [8]
batch_size = 8

has_residual = False

for reduction_size in reduction_sizes:
    for num_expert in num_experts:
        for num_topk in num_topks:
            print(
                f"\n=== Testing batch_size = {batch_size} num_experts = {num_expert} num_topk = {num_topk} reduction_size = {reduction_size} has_residual = {has_residual} ==="
            )

            x = torch.randn((batch_size, reduction_size), device="cuda", dtype=torch.bfloat16)
            w = torch.randn(
                (num_expert, reduction_size), device="cuda", dtype=torch.bfloat16
            )

            topk_indices = torch.empty(batch_size, num_topk, device="cuda", dtype=torch.int32)
            topk_weights = torch.empty(batch_size, num_topk, device="cuda", dtype=torch.float)

            assert(not has_residual)
            residual = None
            runtime_kernel_blackwell.gate_topk_sm100(x, w, residual, topk_indices, topk_weights)
            
            # ref implementation
            torch_out = torch.matmul(x, torch.transpose(w, 0, 1)).to(torch.float)
            torch_topk_values, torch_topk_indices = torch.topk(torch_out, num_topk, dim=1)
            torch_topk_weights = F.softmax(torch_topk_values, dim=1, dtype=torch.float)
            
            # print(torch_topk_indices, torch_topk_values)
            
            # torch.testing.assert_close(
            #     output,
            #     torch_out,
            #     rtol=1e-2,
            #     atol=1e-2,
            # )
            # print("Test passed!")

            # # Warm-up
            # for _ in range(16):
            #     runtime_kernel_blackwell.gate_topk_sm100(x, w, residual, output)

            # torch.cuda.synchronize()
            # starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
            #     enable_timing=True
            # )
            # repetitions = 1000
            # starter.record()
            # for rep in range(repetitions):
            #     runtime_kernel_blackwell.gate_topk_sm100(x, w, residual, output)
            # ender.record()
            # torch.cuda.synchronize()
            # total_time = starter.elapsed_time(ender)
            # avg_time = total_time / repetitions
            # print(f"Average time over {repetitions} runs: {avg_time:.6f} ms")
