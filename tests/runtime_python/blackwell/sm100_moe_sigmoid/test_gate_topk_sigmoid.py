import torch
import runtime_kernel_moe_sigmoid

from torch.nn import functional as F

torch.set_printoptions(sci_mode=False, profile="full")

# ============================================================================
# DeepSeek V3 configuration (from configuration_deepseek_v3.py)
# ============================================================================
NUM_EXPERTS = 256           # n_routed_experts
NUM_EXPERTS_PER_TOK = 8     # num_experts_per_tok
NUM_GROUPS = 8              # n_group
TOPK_GROUP = 4              # topk_group
ROUTED_SCALING_FACTOR = 2.5 # routed_scaling_factor
EXPERTS_PER_GROUP = NUM_EXPERTS // NUM_GROUPS  # 32

BATCH_SIZES = [1, 2, 4, 8]
SEED = 42


# ============================================================================
# PyTorch reference: DeepSeek V3 group-aware sigmoid routing
# (matches DeepseekV3TopkRouter.forward + get_topk_indices)
# ============================================================================
def reference_sigmoid_routing(logits_bf16, bias, batch_size):
    """Exact replica of DeepseekV3TopkRouter logic."""
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


# ============================================================================
# Correctness tests
# ============================================================================
print("=" * 70)
print("CORRECTNESS TESTS — DeepSeek V3 Sigmoid Routing")
print(f"  n_routed_experts={NUM_EXPERTS}  num_experts_per_tok={NUM_EXPERTS_PER_TOK}")
print(f"  n_group={NUM_GROUPS}  topk_group={TOPK_GROUP}")
print(f"  routed_scaling_factor={ROUTED_SCALING_FACTOR}")
print("=" * 70)

g = torch.Generator(device="cuda").manual_seed(SEED)

for batch_size in BATCH_SIZES:
    print(f"\n--- batch_size = {batch_size} ---")

    gating_output = torch.randn(
        (batch_size, NUM_EXPERTS), device="cuda", dtype=torch.bfloat16, generator=g
    )
    bias = torch.randn(
        NUM_EXPERTS, device="cuda", dtype=torch.float32, generator=g
    ) * 0.1

    topk_weights = torch.empty(
        batch_size, NUM_EXPERTS_PER_TOK, device="cuda", dtype=torch.float
    )
    mpk_routing_indices = torch.zeros(
        (NUM_EXPERTS, batch_size), device="cuda", dtype=torch.int32
    )
    mpk_active_ids = torch.empty(
        (NUM_EXPERTS + 1,), device="cuda", dtype=torch.int32
    )

    gating_output_ref = gating_output.clone()

    # Run kernel
    runtime_kernel_moe_sigmoid.topk_sigmoid_sm100(
        gating_output, bias, topk_weights, mpk_routing_indices,
        mpk_active_ids, ROUTED_SCALING_FACTOR, NUM_GROUPS, TOPK_GROUP,
    )

    # Reference
    ref_weights, ref_routing, ref_expert_mask = reference_sigmoid_routing(
        gating_output_ref, bias, batch_size
    )

    # Check topk_weights
    torch.testing.assert_close(topk_weights, ref_weights, rtol=1e-2, atol=1e-2)
    print("  topk_weights:      PASS")

    # Check routing indices
    torch.testing.assert_close(mpk_routing_indices, ref_routing, rtol=0, atol=0)
    print("  routing_indices:   PASS")

    # Check active expert IDs (set equality)
    num_active = int(mpk_active_ids[-1].item())
    if num_active > 0:
        recon_mask = torch.zeros((NUM_EXPERTS,), device="cuda", dtype=torch.int32)
        active_ids = mpk_active_ids[:num_active].to(torch.long)
        recon_mask.index_fill_(0, active_ids, 1)
        torch.testing.assert_close(recon_mask, ref_expert_mask, rtol=0, atol=0)
    print(f"  active_expert_ids: PASS  ({num_active} active)")

print("\n>>> All correctness tests PASSED <<<\n")


# ============================================================================
# Benchmark: sigmoid vs softmax at same DeepSeek V3 config (256 experts)
# ============================================================================
print("=" * 70)
print("BENCHMARK — topk_sigmoid vs topk_softmax  (256 experts, top-8)")
print("=" * 70)

WARMUP = 50
REPETITIONS = 5000

for batch_size in BATCH_SIZES:
    # -- Allocate shared tensors --
    gating_output = torch.randn(
        (batch_size, NUM_EXPERTS), device="cuda", dtype=torch.bfloat16
    )
    bias = torch.randn(NUM_EXPERTS, device="cuda", dtype=torch.float32) * 0.1
    topk_weights = torch.empty(
        batch_size, NUM_EXPERTS_PER_TOK, device="cuda", dtype=torch.float
    )
    mpk_routing_indices = torch.zeros(
        (NUM_EXPERTS, batch_size), device="cuda", dtype=torch.int32
    )
    mpk_active_ids = torch.empty(
        (NUM_EXPERTS + 1,), device="cuda", dtype=torch.int32
    )

    # -- Benchmark sigmoid --
    for _ in range(WARMUP):
        runtime_kernel_moe_sigmoid.topk_sigmoid_sm100(
            gating_output, bias, topk_weights, mpk_routing_indices,
            mpk_active_ids, ROUTED_SCALING_FACTOR, NUM_GROUPS, TOPK_GROUP,
        )
    torch.cuda.synchronize()

    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    start_evt.record()
    for _ in range(REPETITIONS):
        runtime_kernel_moe_sigmoid.topk_sigmoid_sm100(
            gating_output, bias, topk_weights, mpk_routing_indices,
            mpk_active_ids, ROUTED_SCALING_FACTOR, NUM_GROUPS, TOPK_GROUP,
        )
    end_evt.record()
    torch.cuda.synchronize()
    sigmoid_us = start_evt.elapsed_time(end_evt) / REPETITIONS * 1000  # microseconds

    # -- Benchmark softmax (same expert count / batch / topk) --
    gating_softmax = torch.randn(
        (batch_size, NUM_EXPERTS), device="cuda", dtype=torch.bfloat16
    )

    for _ in range(WARMUP):
        runtime_kernel_moe_sigmoid.topk_softmax_sm100(
            gating_softmax, topk_weights, mpk_routing_indices, mpk_active_ids,
        )
    torch.cuda.synchronize()

    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    start_evt.record()
    for _ in range(REPETITIONS):
        runtime_kernel_moe_sigmoid.topk_softmax_sm100(
            gating_softmax, topk_weights, mpk_routing_indices, mpk_active_ids,
        )
    end_evt.record()
    torch.cuda.synchronize()
    softmax_us = start_evt.elapsed_time(end_evt) / REPETITIONS * 1000

    ratio = sigmoid_us / softmax_us
    print(
        f"  batch_size={batch_size:2d}  |  sigmoid: {sigmoid_us:7.3f} us  "
        f"softmax: {softmax_us:7.3f} us  |  ratio: {ratio:.2f}x"
    )

print()
