import torch
import runtime_kernel


torch.set_printoptions(sci_mode=False)

x = torch.randn((1, 3584), device='cuda', dtype=torch.bfloat16)
m = torch.randn((1, 3584), device='cuda', dtype=torch.bfloat16)
w = torch.randn((3584, 64), device='cuda', dtype=torch.bfloat16)
output = torch.empty(1, 64, device='cuda', dtype=torch.bfloat16)
runtime_kernel.silu_mul_linear(x, m, w, output)
print(output)
silu = torch.nn.SiLU()
activated = silu(x)    
print(torch.matmul(torch.mul(activated,m), w))