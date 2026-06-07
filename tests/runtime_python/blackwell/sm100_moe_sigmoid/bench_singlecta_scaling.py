"""Measure CURRENT single-CTA topk_sigmoid latency vs num_rows (batch).
Decode shape = batch 1 (num_active_rows=1). Prefill shape = batch 128.
This validates the '~108us full mbt=128 in 1 CTA' premise before any kernel change.
"""
import torch
import runtime_kernel_moe_sigmoid

NUM_EXPERTS = 256
NUM_EXPERTS_PER_TOK = 8
NUM_GROUPS = 8
TOPK_GROUP = 4
ROUTED_SCALING_FACTOR = 2.5

WARMUP = 50
REPS = 2000

print("single-CTA topk_sigmoid scaling (grid=(1,1,1), num_active_rows=batch)")
print(f"{'batch':>6} {'us/call':>10}")
for batch in [1, 8, 16, 32, 64, 128, 256]:
    g = torch.randn((batch, NUM_EXPERTS), device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(NUM_EXPERTS, device="cuda", dtype=torch.float32) * 0.1
    w = torch.empty(batch, NUM_EXPERTS_PER_TOK, device="cuda", dtype=torch.float)
    ri = torch.zeros((NUM_EXPERTS, batch), device="cuda", dtype=torch.int32)
    aid = torch.empty((NUM_EXPERTS + 1,), device="cuda", dtype=torch.int32)

    for _ in range(WARMUP):
        runtime_kernel_moe_sigmoid.topk_sigmoid_sm100(
            g, bias, w, ri, aid, ROUTED_SCALING_FACTOR, NUM_GROUPS, TOPK_GROUP)
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(REPS):
        runtime_kernel_moe_sigmoid.topk_sigmoid_sm100(
            g, bias, w, ri, aid, ROUTED_SCALING_FACTOR, NUM_GROUPS, TOPK_GROUP)
    e.record()
    torch.cuda.synchronize()
    us = s.elapsed_time(e) / REPS * 1000
    print(f"{batch:>6} {us:>10.3f}")
