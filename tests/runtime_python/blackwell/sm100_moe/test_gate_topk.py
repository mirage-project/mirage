import torch
import runtime_kernel_blackwell

from torch.nn import functional as F

torch.set_printoptions(sci_mode=False, profile="full")
# torch.set_printoptions(sci_mode=False)

g = torch.Generator(device="cuda").manual_seed(1234)

num_experts_list = [128]
num_topks = [8]
batch_size = 1

for num_expert in num_experts_list:
    for num_topk in num_topks:
        print(
            f"\n=== Testing batch_size = {batch_size} num_experts = {num_expert} num_topk = {num_topk} ==="
        )

        # Random gating outputs (pre-softmax logits) should be using bfloat16 but the bfloat16 range is a bit small for randn so we test with float here
        gating_output = torch.randn((batch_size, num_expert), device="cuda", dtype=torch.bfloat16, generator=g)

        topk_weights = torch.empty(batch_size, num_topk, device="cuda", dtype=torch.float)
        mpk_routing_indices = torch.zeros((num_expert, batch_size), device="cuda", dtype=torch.int32)
        mpk_active_ids = torch.empty((num_expert+1,), device="cuda", dtype=torch.int32) # store both active ids and num active at the end

        # Preserve a copy of inputs for reference before kernel mutates input
        gating_output_ref = gating_output.clone()

        # Run fused topk softmax
        runtime_kernel_blackwell.topk_softmax_sm100(gating_output, topk_weights, mpk_routing_indices, mpk_active_ids)

        # Reference: select topk then softmax over those values
        gating_output_f = gating_output_ref.to(torch.float)
        norm_gating_output = gating_output_f - gating_output_f.amax(dim=1, keepdim=True)
        torch_softmax = F.softmax(norm_gating_output, dim=1, dtype=torch.float)
        # print("torch_softmax:", torch_softmax)
        torch_topk_values, torch_topk_indices = torch.topk(torch_softmax, num_topk, dim=1)
        # print("torch_topk_indices:", torch_topk_indices)
        # print("mpk_topk_indices:", topk_indices)
        torch_topk_weights = torch_topk_values / torch_topk_values.sum(dim=-1, keepdim=True)
        torch_routing_indices = torch.zeros((num_expert, batch_size), device="cuda", dtype=torch.int32)
        torch_expert_mask = torch.zeros((num_expert,), device="cuda", dtype=torch.int32)

        for token_idx in range(batch_size):
            for topk_idx in range(num_topk):
                expert_idx = torch_topk_indices[token_idx, topk_idx]
                torch_routing_indices[expert_idx, token_idx] = topk_idx + 1
                torch_expert_mask[expert_idx] = 1

        torch.testing.assert_close(
            topk_weights,
            torch_topk_weights,
            rtol=1e-2,
            atol=1e-2,
        )
        torch.testing.assert_close(
            mpk_routing_indices,
            torch_routing_indices,
            rtol=1e-2,
            atol=1e-2,
        )
        # Reconstruct mask from active ids output and print diagnostics
        num_active = int(mpk_active_ids[-1].item())
        print(f"Active experts: {num_active}")
        if num_active > 0:
            print(f"Active expert IDs: {mpk_active_ids[:num_active].cpu().tolist()}")
        recon_mask = torch.zeros((num_expert,), device="cuda", dtype=torch.int32)
        if num_active > 0:
            active_ids = mpk_active_ids[:num_active].to(torch.long)
            recon_mask.index_fill_(0, active_ids, 1)
        torch.testing.assert_close(
            recon_mask,
            torch_expert_mask,
            rtol=1e-2,
            atol=1e-2,
        )
        print("Test passed!")

        # Warm-up
        for _ in range(16):
            runtime_kernel_blackwell.topk_softmax_sm100(gating_output, topk_weights, mpk_routing_indices, mpk_active_ids)

        torch.cuda.synchronize()
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
            enable_timing=True
        )
        repetitions = 1000
        starter.record()
        for rep in range(repetitions):
            runtime_kernel_blackwell.topk_softmax_sm100(gating_output, topk_weights, mpk_routing_indices, mpk_active_ids)
        ender.record()
        torch.cuda.synchronize()
        total_time = starter.elapsed_time(ender)
        avg_time = total_time / repetitions
        print(f"Average time over {repetitions} runs: {avg_time:.6f} ms")
