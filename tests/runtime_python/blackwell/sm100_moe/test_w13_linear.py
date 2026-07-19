import torch
import runtime_kernel_blackwell
from pytorch_reference import moe_w13_linear_ref

torch.set_printoptions(sci_mode=False, profile="full")
# torch.set_printoptions(sci_mode=False)

g = torch.Generator(device="cuda").manual_seed(1234)

reduction_sizes = [2048]
output_sizes = [128]
batch_size = 1
num_experts = 128
num_topk = 8
expert_offset = 0
expert_stride = 10

is_w2_linear = False
has_residual = False

for reduction_size in reduction_sizes:
    for output_size in output_sizes:
        print(
            f"\n=== Testing batch_size = {batch_size} output_size = {output_size} reduction_size = {reduction_size} num_experts = {num_experts} num_topk = {num_topk} has_residual = {has_residual} ==="
        )

        x = torch.randn((batch_size, reduction_size), device="cuda", dtype=torch.bfloat16)
        w = torch.randn((num_experts, output_size, reduction_size), device="cuda", dtype=torch.bfloat16)
        expert_score = torch.randn((batch_size, num_experts), device="cuda", dtype=torch.bfloat16)
        topk_expert_score, topk_expert_indices = torch.topk(expert_score, num_topk, dim=1)
        expert_mask = torch.zeros((num_experts), device="cuda", dtype=torch.int32)
        residual = torch.randn(num_experts, batch_size, output_size, device="cuda", dtype=torch.bfloat16)
        output = torch.zeros(batch_size, num_topk, output_size, device="cuda", dtype=torch.bfloat16)
        
        # reference impl
        torch_out, expert_hit = moe_w13_linear_ref(
            x=x,
            w=w,
            topk_expert_indices=topk_expert_indices,
            num_experts=num_experts,
            num_topk=num_topk,
            batch_size=batch_size,
            reduction_size=reduction_size,
            output_size=output_size,
            residual=residual if has_residual else None,
            expert_offset=expert_offset,
            expert_stride=expert_stride,
        )

        # mpk impl
        mpk_routing_indices = torch.zeros((num_experts, batch_size), device="cuda", dtype=torch.int32)
        mpk_expert_mask = torch.zeros((num_experts+1), device="cuda", dtype=torch.int32)
        
        for token_idx in range(batch_size):
            for topk_idx in range(num_topk):
                expert_idx = topk_expert_indices[token_idx, topk_idx]
                mpk_routing_indices[expert_idx, token_idx] = topk_idx + 1

        for i, expert_idx in enumerate(expert_hit):
            mpk_expert_mask[i] = expert_idx
        mpk_expert_mask[num_experts] = len(expert_hit)  # end marker

        print("num_expert activated:", mpk_expert_mask[num_experts].item())

        if not has_residual:
            residual = None
        runtime_kernel_blackwell.moe_w13_linear_sm100(x, w, residual, mpk_routing_indices, mpk_expert_mask, output)
        
        torch.testing.assert_close(
            output,
            torch_out,
            rtol=1e-2,
            atol=1e-2,
        )
        print("Test passed!")

        # Warm-up
        for _ in range(16):
            runtime_kernel_blackwell.moe_w13_linear_sm100(x, w, residual, mpk_routing_indices, mpk_expert_mask, output)

        torch.cuda.synchronize()
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
            enable_timing=True
        )
        repetitions = 1000
        starter.record()
        for rep in range(repetitions):
            runtime_kernel_blackwell.moe_w13_linear_sm100(x, w, residual, mpk_routing_indices, mpk_expert_mask, output)
        ender.record()
        torch.cuda.synchronize()
        total_time = starter.elapsed_time(ender)
        avg_time = total_time / repetitions
        print(f"Average time over {repetitions} runs: {avg_time:.6f} ms")
