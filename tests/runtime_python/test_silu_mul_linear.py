import numpy as np
import torch
import runtime_kernel


torch.set_printoptions(sci_mode=False)

x = torch.randn((1, 3584), device="cuda", dtype=torch.bfloat16)
m = torch.randn((1, 3584), device="cuda", dtype=torch.bfloat16)
w = torch.randn((3584, 32), device="cuda", dtype=torch.bfloat16)
output = torch.empty(1, 32, device="cuda", dtype=torch.bfloat16)
runtime_kernel.silu_mul_linear(x, m, w, output)
silu = torch.nn.SiLU()
activated = silu(x)
print(output)
print(torch.matmul(torch.mul(activated, m), w))
print(output / torch.matmul(torch.mul(activated, m), w))

# warm up runs
for _ in range(16):
    runtime_kernel.silu_mul_linear(x, m, w, output)

torch.cuda.synchronize()
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
    enable_timing=True
)
repetitions = 100000
timings = np.zeros((repetitions, 1))
starter.record()
for rep in range(repetitions):
    runtime_kernel.silu_mul_linear(x, m, w, output)
ender.record()
torch.cuda.synchronize()
curr_time = starter.elapsed_time(ender)
mean_syn = curr_time / repetitions

print(mean_syn)
