import torch
import runtime_kernel as runtime_kernel_ep_moe

torch.set_printoptions(sci_mode=False)

BATCH_SIZE       = 8
TOPK             = 2
HIDDEN_DIM       = 64
WORLD_SIZE       = 1
NUM_EXPERTS      = 8
EXPERTS_PER_RANK = NUM_EXPERTS // WORLD_SIZE

torch.manual_seed(42)
expert_outputs  = torch.randn(BATCH_SIZE, TOPK, HIDDEN_DIM, device="cuda", dtype=torch.bfloat16)
routing_indices = torch.randint(0, NUM_EXPERTS, (BATCH_SIZE, TOPK), device="cuda", dtype=torch.int32)
routing_weights = torch.softmax(
    torch.randn(BATCH_SIZE, TOPK, device="cuda", dtype=torch.float32), dim=1
).to(torch.bfloat16)
residual = torch.randn(BATCH_SIZE, HIDDEN_DIM, device="cuda", dtype=torch.bfloat16)

output      = torch.zeros(BATCH_SIZE, HIDDEN_DIM, device="cuda", dtype=torch.bfloat16)
recv_counts = torch.tensor([BATCH_SIZE], dtype=torch.int32, device="cuda")
recv_offsets = torch.zeros(WORLD_SIZE, dtype=torch.int32, device="cuda")
sync_flags  = torch.ones(WORLD_SIZE, dtype=torch.int32, device="cuda")

runtime_kernel_ep_moe.moe_combine(
    expert_outputs, routing_indices, routing_weights, residual, output,
    recv_counts, recv_offsets, sync_flags,
    NUM_EXPERTS, EXPERTS_PER_RANK, 0)

ref_out = (
    expert_outputs.float() * routing_weights.float().unsqueeze(-1)
).sum(dim=1).to(torch.bfloat16) + residual

torch.testing.assert_close(output, ref_out, rtol=1e-2, atol=1e-2)
print("Test passed!")

for _ in range(16):
    runtime_kernel_ep_moe.moe_combine(
        expert_outputs, routing_indices, routing_weights, residual, output,
        recv_counts, recv_offsets, sync_flags,
        NUM_EXPERTS, EXPERTS_PER_RANK, 0)

torch.cuda.synchronize()
starter = torch.cuda.Event(enable_timing=True)
ender   = torch.cuda.Event(enable_timing=True)
repetitions = 1000
starter.record()
for _ in range(repetitions):
    runtime_kernel_ep_moe.moe_combine(
        expert_outputs, routing_indices, routing_weights, residual, output,
        recv_counts, recv_offsets, sync_flags,
        NUM_EXPERTS, EXPERTS_PER_RANK, 0)
ender.record()
torch.cuda.synchronize()
print(f"Average time: {starter.elapsed_time(ender) / repetitions:.6f} ms")
