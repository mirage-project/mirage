import torch
from torch.nn import functional as F
import runtime_kernel as runtime_kernel_ep_moe

torch.set_printoptions(sci_mode=False)

BATCH_SIZE       = 8
NUM_EXPERTS      = 8
TOPK             = 2
WORLD_SIZE       = 1
EXPERTS_PER_RANK = NUM_EXPERTS // WORLD_SIZE

torch.manual_seed(42)
router_logits = torch.randn(BATCH_SIZE, NUM_EXPERTS, device="cuda", dtype=torch.bfloat16)

routing_indices = torch.zeros(BATCH_SIZE, TOPK, device="cuda", dtype=torch.int32)
routing_weights = torch.zeros(BATCH_SIZE, TOPK, device="cuda", dtype=torch.bfloat16)
dispatch_counts = torch.zeros(WORLD_SIZE, device="cuda", dtype=torch.int32)

runtime_kernel_ep_moe.moe_routing(
    router_logits, routing_indices, routing_weights, dispatch_counts,
    EXPERTS_PER_RANK, 0, 0.0)

ref_scores, ref_indices = torch.topk(router_logits.float(), TOPK, dim=1)
ref_weights = F.softmax(ref_scores, dim=1).to(torch.bfloat16)

# Sort indices before comparing (heap sort order may differ from torch.topk when scores tie)
kernel_sorted_idx, kernel_order = routing_indices.sort(dim=1)
kernel_sorted_wts = routing_weights.gather(1, kernel_order)
ref_sorted_idx, ref_order = ref_indices.sort(dim=1)
ref_sorted_wts = ref_weights.gather(1, ref_order)

torch.testing.assert_close(kernel_sorted_idx, ref_sorted_idx.to(torch.int32), rtol=0, atol=0)
torch.testing.assert_close(kernel_sorted_wts, ref_sorted_wts, rtol=1e-2, atol=1e-2)
print("Test passed!")

for _ in range(16):
    runtime_kernel_ep_moe.moe_routing(
        router_logits, routing_indices, routing_weights, dispatch_counts,
        EXPERTS_PER_RANK, 0, 0.0)

torch.cuda.synchronize()
starter = torch.cuda.Event(enable_timing=True)
ender   = torch.cuda.Event(enable_timing=True)
repetitions = 1000
starter.record()
for _ in range(repetitions):
    runtime_kernel_ep_moe.moe_routing(
        router_logits, routing_indices, routing_weights, dispatch_counts,
        EXPERTS_PER_RANK, 0, 0.0)
ender.record()
torch.cuda.synchronize()
print(f"Average time: {starter.elapsed_time(ender) / repetitions:.6f} ms")
