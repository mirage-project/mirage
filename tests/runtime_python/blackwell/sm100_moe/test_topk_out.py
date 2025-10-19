import torch
import runtime_kernel_blackwell

from torch.nn import functional as F

torch.set_printoptions(sci_mode=False, profile="full")
# torch.set_printoptions(sci_mode=False)

g = torch.Generator(device="cuda").manual_seed(1234)

num_experts_list = [128]
num_topks = [8]
batch_size = 8

for num_expert in num_experts_list:
    for num_topk in num_topks:
        print(
            f"\n=== Testing batch_size = {batch_size} num_experts = {num_expert} num_topk = {num_topk} ==="
        )

        # Random gating outputs (pre-softmax logits)
        gating_output = torch.randn((batch_size, num_expert), device="cuda", dtype=torch.float)

        topk_indices = torch.empty(batch_size, num_topk, device="cuda", dtype=torch.int32)
        topk_weights = torch.empty(batch_size, num_topk, device="cuda", dtype=torch.float)
        mpk_routing_indices = torch.zeros((num_expert, batch_size), device="cuda", dtype=torch.int32)
        mpk_expert_mask = torch.zeros((num_expert,), device="cuda", dtype=torch.int32)

        # Run fused topk softmax
        runtime_kernel_blackwell.topk_softmax_sm100(gating_output, topk_indices, topk_weights, mpk_routing_indices, mpk_expert_mask)

        # Reference: select topk then softmax over those values
        torch_topk_values, torch_topk_indices = torch.topk(gating_output, num_topk, dim=1)
        torch_topk_weights = F.softmax(torch_topk_values, dim=1, dtype=torch.float)

        torch.testing.assert_close(
            topk_indices,
            torch_topk_indices.to(torch.int32),
            rtol=1e-2,
            atol=1e-2,
        )
        torch.testing.assert_close(
            topk_weights,
            torch_topk_weights,
            rtol=1e-2,
            atol=1e-2,
        )
        print("Test passed!")

        # Warm-up
        for _ in range(16):
            runtime_kernel_blackwell.topk_softmax_sm100(gating_output, topk_indices, topk_weights, mpk_routing_indices, mpk_expert_mask)

        torch.cuda.synchronize()
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
            enable_timing=True
        )
        repetitions = 1000
        starter.record()
        for rep in range(repetitions):
            runtime_kernel_blackwell.topk_softmax_sm100(gating_output, topk_indices, topk_weights, mpk_routing_indices, mpk_expert_mask)
        ender.record()
        torch.cuda.synchronize()
        total_time = starter.elapsed_time(ender)
        avg_time = total_time / repetitions
        print(f"Average time over {repetitions} runs: {avg_time:.6f} ms")
