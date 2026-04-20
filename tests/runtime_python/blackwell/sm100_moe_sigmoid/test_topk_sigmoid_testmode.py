"""
Test: topk_sigmoid_sm100 via PersistentKernel test_mode.

Tests the moe_topk_sigmoid_routing_layer by building a minimal PersistentKernel
in test_mode, compiling it, running it once, and comparing outputs against a
PyTorch reference that replicates the DeepSeek V3 group-aware sigmoid routing.

Validates: topk_weights, routing_indices, and active_expert_ids.

Run:
    python tests/runtime_python/blackwell/sm100_moe_sigmoid/test_topk_sigmoid_testmode.py
"""

import torch
import sys
import os

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel


# ============================================================================
# DeepSeek V3 routing configuration
# ============================================================================
NUM_EXPERTS = 256
NUM_EXPERTS_PER_TOK = 8
NUM_GROUPS = 8
TOPK_GROUP = 4
ROUTED_SCALING_FACTOR = 2.5
EXPERTS_PER_GROUP = NUM_EXPERTS // NUM_GROUPS  # 32


def reference_sigmoid_routing(logits_bf16, bias, batch_size):
    """PyTorch reference: DeepSeek V3 group-aware sigmoid routing."""
    # Step 1: sigmoid
    scores = torch.sigmoid(logits_bf16.float())

    # Step 2: add bias for selection decisions
    biased = scores + bias.unsqueeze(0)

    # Step 3: group top-2, sum -> group scores
    biased_grouped = biased.view(batch_size, NUM_GROUPS, EXPERTS_PER_GROUP)
    top2_per_group, _ = biased_grouped.topk(2, dim=-1)
    group_scores = top2_per_group.sum(dim=-1)

    # Step 4: select top-K groups
    _, top_groups = group_scores.topk(TOPK_GROUP, dim=-1, sorted=False)
    group_mask = torch.zeros(batch_size, NUM_GROUPS, device="cuda")
    group_mask.scatter_(1, top_groups, 1.0)
    expert_mask = (
        group_mask.unsqueeze(-1)
        .expand(-1, -1, EXPERTS_PER_GROUP)
        .reshape(batch_size, NUM_EXPERTS)
    )

    # Step 5: mask non-selected groups, find top-K experts
    biased_masked = biased.clone()
    biased_masked[expert_mask == 0] = -10000.0
    _, topk_indices = biased_masked.topk(NUM_EXPERTS_PER_TOK, dim=-1)

    # Step 6: gather ORIGINAL sigmoid scores (no bias)
    topk_weights = scores.gather(1, topk_indices)

    # Step 7: normalize + scale
    topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)
    topk_weights = topk_weights * ROUTED_SCALING_FACTOR

    # Build routing indices (expert-major, 1-indexed)
    routing_indices = torch.zeros(
        (NUM_EXPERTS, batch_size), device="cuda", dtype=torch.int32
    )
    expert_active = torch.zeros((NUM_EXPERTS,), device="cuda", dtype=torch.int32)
    for tok in range(batch_size):
        for k in range(NUM_EXPERTS_PER_TOK):
            eidx = topk_indices[tok, k]
            routing_indices[eidx, tok] = k + 1
            expert_active[eidx] = 1

    return topk_weights, routing_indices, expert_active


def test_topk_sigmoid_testmode():
    device = "cuda"
    batch_size = 8
    seed = 42

    print(f"\n{'='*60}")
    print(f"Test: topk_sigmoid_sm100 via PersistentKernel test_mode")
    print(f"  batch_size={batch_size}, num_experts={NUM_EXPERTS}")
    print(f"  num_experts_per_tok={NUM_EXPERTS_PER_TOK}")
    print(f"  num_groups={NUM_GROUPS}, topk_group={TOPK_GROUP}")
    print(f"  routed_scaling_factor={ROUTED_SCALING_FACTOR}")
    print(f"{'='*60}")

    g = torch.Generator(device=device).manual_seed(seed)

    # Create input tensors
    gating_output = torch.randn(
        (batch_size, NUM_EXPERTS), device=device, dtype=torch.bfloat16, generator=g
    )
    bias = torch.randn(
        NUM_EXPERTS, device=device, dtype=torch.float32, generator=g
    ) * 0.1

    # Clone input for reference (kernel writes zeros back into input for split-k)
    gating_output_ref = gating_output.clone()

    # Create output tensors
    topk_weights = torch.zeros(
        batch_size, NUM_EXPERTS_PER_TOK, device=device, dtype=torch.float32
    )
    routing_indices = torch.zeros(
        (NUM_EXPERTS, batch_size), device=device, dtype=torch.int32
    )
    active_expert_ids = torch.full(
        (NUM_EXPERTS + 1,), -1, device=device, dtype=torch.int32
    )

    # Build PersistentKernel in test mode
    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["num_workers"] = num_workers
    params["num_local_schedulers"] = num_schedulers
    params["mpi_rank"] = 0
    params["world_size"] = 1
    params["max_num_batched_tokens"] = batch_size
    params["max_num_batched_requests"] = batch_size
    pk = PersistentKernel(**params)

    # Attach all tensors (inputs AND outputs)
    gating_dt = pk.attach_input(gating_output, name="gating_output")
    bias_dt = pk.attach_input(bias, name="bias")
    topk_weights_dt = pk.attach_input(topk_weights, name="topk_weights")
    routing_indices_dt = pk.attach_input(routing_indices, name="routing_indices")
    active_ids_dt = pk.attach_input(active_expert_ids, name="active_expert_ids")

    # Build layer — one task block handles all 8 rows (8 warps, 1 row/warp)
    block_dim = (256, 1, 1)
    pk.moe_topk_sigmoid_routing_layer(
        input=gating_dt,
        bias=bias_dt,
        output=(topk_weights_dt, routing_indices_dt, active_ids_dt),
        grid_dim=(1, 1, 1),
        block_dim=block_dim,
        num_groups=NUM_GROUPS,
        topk_group=TOPK_GROUP,
        routed_scaling_factor=ROUTED_SCALING_FACTOR,
    )

    # Compile
    print("Compiling test kernel...")
    folder_path = os.path.dirname(os.path.abspath(__file__))
    pk.compile(output_dir=folder_path)

    # Run
    print("Running test kernel...")
    pk.run_test_mode()
    torch.cuda.synchronize()

    # Compute reference
    ref_weights, ref_routing, ref_expert_mask = reference_sigmoid_routing(
        gating_output_ref, bias, batch_size
    )

    # Check topk_weights
    print(f"\ntopk_weights[0]:   {topk_weights[0]}")
    print(f"reference[0]:      {ref_weights[0]}")
    weight_diff = (topk_weights - ref_weights).abs().max().item()
    print(f"Max weight diff:   {weight_diff}")

    if weight_diff >= 0.01:
        print(f"FAILED: topk_weights max diff {weight_diff} exceeds tolerance 0.01")
        pk.finalize()
        sys.exit(1)
    print("  topk_weights: PASS")

    # Check routing indices
    routing_match = torch.equal(routing_indices, ref_routing)
    print(f"  routing_indices: {'PASS' if routing_match else 'FAIL'}")
    if not routing_match:
        diff_mask = routing_indices != ref_routing
        num_diffs = diff_mask.sum().item()
        print(f"    {num_diffs} mismatched entries")
        pk.finalize()
        sys.exit(1)

    # Check active expert IDs (set equality)
    num_active = int(active_expert_ids[-1].item())
    if num_active > 0:
        recon_mask = torch.zeros((NUM_EXPERTS,), device=device, dtype=torch.int32)
        active_ids = active_expert_ids[:num_active].to(torch.long)
        recon_mask.index_fill_(0, active_ids, 1)
        active_match = torch.equal(recon_mask, ref_expert_mask)
    else:
        active_match = ref_expert_mask.sum().item() == 0
    print(f"  active_expert_ids: {'PASS' if active_match else 'FAIL'} ({num_active} active)")
    if not active_match:
        pk.finalize()
        sys.exit(1)

    print(f"\nPASSED: topk_sigmoid test_mode produces correct output")

    # Cleanup
    pk.finalize()
    print("Test completed successfully!")


if __name__ == "__main__":
    test_topk_sigmoid_testmode()
