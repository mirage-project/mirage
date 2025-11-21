import torch
import runtime_kernel_moe_hopper

torch.set_printoptions(sci_mode=False, profile="full")
# torch.set_printoptions(sci_mode=False)

g = torch.Generator(device="cuda").manual_seed(1234)

reduction_sizes = [768]
output_sizes = [128]
batch_size = 1
num_experts = 128
num_topk = 8
expert_offset = 0
expert_stride = 8

is_w2_linear = True
has_residual = False

for reduction_size in reduction_sizes:
    for output_size in output_sizes:
        print(
            f"\n=== Testing batch_size = {batch_size} output_size = {output_size} reduction_size = {reduction_size} num_experts = {num_experts} num_topk = {num_topk} has_residual = {has_residual} ==="
        )

        x = torch.randn((batch_size, num_topk, reduction_size), device="cuda", dtype=torch.bfloat16)
        w = torch.randn((num_experts, output_size, reduction_size), device="cuda", dtype=torch.bfloat16)
        expert_score = torch.randn((batch_size, num_experts), device="cuda", dtype=torch.bfloat16)
        topk_expert_score, topk_expert_indices = torch.topk(expert_score, num_topk, dim=1)
        residual = torch.randn(num_experts, batch_size, output_size, device="cuda", dtype=torch.bfloat16)
        output = torch.zeros(batch_size, num_topk, output_size, device="cuda", dtype=torch.bfloat16)
        
        # reference impl
        expert_mask = torch.nn.functional.one_hot(topk_expert_indices, num_classes=num_experts).permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        torch_out = torch.zeros((batch_size, num_topk, output_size), device="cuda", dtype=torch.bfloat16)
        for i, expert_idx in enumerate(expert_hit):
            if (i+expert_offset) % expert_stride != 0:
                continue
            # print(f"idx: {i}, expert_idx: {expert_idx.item()}")
            expert_w = w[expert_idx].squeeze(0)
            idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))
            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. 
            current_state = x[None, top_x, idx].reshape(-1, reduction_size)
            # print(f"top_x {top_x} idx {idx}  expert {expert_idx} ")
            current_hidden_states = torch.matmul(current_state, expert_w.T)
            if has_residual:
                expert_residual = residual[expert_idx].squeeze(0)
                current_residual = expert_residual[None, top_x].reshape(-1, output_size)
                current_hidden_states += current_residual
            torch_out[top_x, idx] = current_hidden_states

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
        runtime_kernel_moe_hopper.moe_w2_linear_sm90(x, w, residual, mpk_routing_indices, mpk_expert_mask, output)
        
        torch.testing.assert_close(
            output,
            torch_out,
            rtol=1e-2,
            atol=1e-2,
        )
        print("Test passed!")

        # Warm-up
        for _ in range(16):
            runtime_kernel_moe_hopper.moe_w2_linear_sm90(x, w, residual, mpk_routing_indices, mpk_expert_mask, output)

        torch.cuda.synchronize()
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
            enable_timing=True
        )
        repetitions = 1000
        starter.record()
        for rep in range(repetitions):
            runtime_kernel_blackwell.moe_w2_linear_sm100(x, w, residual, mpk_routing_indices, mpk_expert_mask, output)
        ender.record()
        torch.cuda.synchronize()
        total_time = starter.elapsed_time(ender)
        avg_time = total_time / repetitions
        print(f"Average time over {repetitions} runs: {avg_time:.6f} ms")