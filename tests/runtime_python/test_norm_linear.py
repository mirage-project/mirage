import torch
import runtime_kernel


torch.set_printoptions(sci_mode=False)

rms_norm = torch.nn.RMSNorm(3584, device='cuda:0', dtype=torch.bfloat16)

def torch_rms_norm(X, W):
    D = rms_norm(X)
    E = torch.matmul(D, W)
    return E

x = torch.randn((1, 3584), device='cuda', dtype=torch.bfloat16)
w = torch.randn((3584, 64), device='cuda', dtype=torch.bfloat16)
output = torch.empty(1, 64, device='cuda', dtype=torch.bfloat16)
runtime_kernel.norm_linear(x, w, output)
print(output)
print(torch_rms_norm(x,w))