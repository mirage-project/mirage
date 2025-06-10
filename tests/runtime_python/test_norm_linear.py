import mirage as mi
import torch
import runtime_kernel

torch.set_printoptions(sci_mode=False)

reduction_size = 4096
output_sizes = [64]


def torch_rms_norm(X, WT, G, eps):
    variance = X.pow(2).mean(-1, keepdim=True)
    X = X * torch.rsqrt(variance + eps)
    X = torch.mul(X, G)
    O = torch.matmul(X, WT)
    return O


for output_size in output_sizes:
    print(f"\n=== Testing output_size = {output_size} ===")

    x = torch.randn((1, reduction_size), device="cuda", dtype=torch.bfloat16)
    norm_w = torch.randn((1, reduction_size), device="cuda", dtype=torch.bfloat16)
    w = torch.randn((output_size, reduction_size),  device="cuda", dtype=torch.bfloat16)
    wt = torch.transpose(w, 0, 1)
    eps = 0.8765
    output = torch.empty(1, output_size, device="cuda", dtype=torch.bfloat16)



    runtime_kernel.norm_linear(x, wt, norm_w, eps, output)
    torch_out = torch_rms_norm(x, wt, norm_w, eps)

    print(output)
    print(torch_out)

    print("Ratio (kernel / torch):")
    print(output / torch_out)
