import numpy as np
import torch
import runtime_kernel
import numpy as np


torch.set_printoptions(sci_mode=False)

rms_norm = torch.nn.RMSNorm(3584, device="cuda:0", dtype=torch.bfloat16)


def torch_rms_norm(X, W):
    D = rms_norm(X)
    E = torch.matmul(D, W)
    return E
x = torch.randn((1, 3584), device="cuda", dtype=torch.bfloat16)
w = torch.randn((3584, 64), device="cuda", dtype=torch.bfloat16)
output = torch.empty(1, 64, device="cuda", dtype=torch.bfloat16)
runtime_kernel.norm_linear(x, w, output)
print(output)
print(torch_rms_norm(x, w))
print(output / torch_rms_norm(x, w))

# warm up runs
for _ in range(16):
    runtime_kernel.norm_linear(x, w, output)

torch.cuda.synchronize()
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
    enable_timing=True
)
repetitions = 1000
timings = np.zeros((repetitions, 1))
starter.record()
for rep in range(repetitions):
    runtime_kernel.norm_linear(x, w, output)
ender.record()
torch.cuda.synchronize()
curr_time = starter.elapsed_time(ender)
mean_syn = curr_time / 1000

print(mean_syn)
