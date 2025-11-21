import torch
import runtime_kernel_blackwell

from torch.nn import functional as F

torch.set_printoptions(sci_mode=False, profile="full")
# torch.set_printoptions(sci_mode=False)

g = torch.Generator(device="cuda").manual_seed(1234)

output_sizes = [256]
batch_size = 8
num_experts = 128
num_topk = 8

for output_size in output_sizes:
    print(
        f"\n=== Testing batch_size = {batch_size} output_size = {output_size} num_experts = {num_experts} num_topk = {num_topk} ==="
    )

    x = torch.randn((batch_size, num_topk, output_size), device="cuda", dtype=torch.bfloat16)
    residual = torch.randn((batch_size, output_size), device="cuda", dtype=torch.bfloat16)
    expert_score = torch.randn((batch_size, num_experts), device="cuda", dtype=torch.bfloat16)
    topk_expert_score, topk_expert_indices = torch.topk(expert_score, num_topk, dim=1)
    torch_topk_weights = F.softmax(topk_expert_score, dim=1, dtype=torch.float)
    output = torch.zeros(batch_size, output_size, device="cuda", dtype=torch.bfloat16)
        
    # mpk impl
    runtime_kernel_blackwell.mul_sum_add_sm100(x, residual, torch_topk_weights, output)
    # reference impl
    torch_out = x.to(torch.float) * torch_topk_weights.unsqueeze(-1)
    torch_out = torch_out.sum(dim=1).to(torch.bfloat16)
    torch_out += residual

    torch.testing.assert_close(
        output,
        torch_out,
        rtol=1e-2,
        atol=1e-2,
    )
    print("Test passed!")

    # Warm-up
    for _ in range(16):
        runtime_kernel_blackwell.mul_sum_add_sm100(x, residual, torch_topk_weights, output)

    torch.cuda.synchronize()
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    repetitions = 1000
    starter.record()
    for rep in range(repetitions):
        runtime_kernel_blackwell.mul_sum_add_sm100(x, residual, torch_topk_weights, output)
    ender.record()
    torch.cuda.synchronize()
    total_time = starter.elapsed_time(ender)
    avg_time = total_time / repetitions
    print(f"Average time over {repetitions} runs: {avg_time:.6f} ms")
