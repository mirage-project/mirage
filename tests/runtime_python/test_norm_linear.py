import torch
import norm_linear_cuda


rms_norm = torch.nn.RMSNorm(4096, device='cuda:0', dtype=torch.float16)

def torch_rms_norm(X, W):
    D = rms_norm(X)
    E = torch.matmul(D, W)
    return E

x = torch.randn(16, 64, device='cuda', dtype=torch.bfloat16)
w = torch.randn(64, 64, device='cuda', dtype=torch.bfloat16)
output = torch.empty(16, 64, device='cuda', dtype=torch.bfloat16)

norm_linear_cuda.norm_linear(x, w, output)


torch_rms = torch_rms_norm(x, w)
print(output)
print(torch_rms)