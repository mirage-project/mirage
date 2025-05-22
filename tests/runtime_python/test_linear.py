import torch
import runtime_kernel


torch.set_printoptions(sci_mode=False)

x = torch.randn((1, 3584), device='cuda', dtype=torch.bfloat16)
w = torch.randn((3584, 64), device='cuda', dtype=torch.bfloat16)
output = torch.empty(1, 64, device='cuda', dtype=torch.bfloat16)
runtime_kernel.linear(x, w, output)
print(output)
print(torch.matmul(x, w))