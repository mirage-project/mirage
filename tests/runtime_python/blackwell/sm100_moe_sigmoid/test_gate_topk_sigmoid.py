import torch
import runtime_kernel_moe_sigmoid

from torch.nn import functional as F

from pytorch_reference import moe_topk_sigmoid_routing_ref

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
    ref_weights, ref_routing, ref_expert_mask = moe_topk_sigmoid_routing_ref(
        gating_output_ref,
        bias,
        batch_size,
        num_experts=NUM_EXPERTS,
        num_experts_per_tok=NUM_EXPERTS_PER_TOK,
        num_groups=NUM_GROUPS,
        topk_group=TOPK_GROUP,
        routed_scaling_factor=ROUTED_SCALING_FACTOR,
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
