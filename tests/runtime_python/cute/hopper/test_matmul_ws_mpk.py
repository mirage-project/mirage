import torch
import runtime_kernel_cute_hopper

# torch.set_printoptions(sci_mode=False, profile="full")
torch.set_printoptions(sci_mode=False)

g = torch.Generator(device="cuda").manual_seed(1234)

reduction_size = 4096
batch_size = 8
output_size = 64

x = torch.randn((batch_size, reduction_size), device="cuda", dtype=torch.bfloat16, generator=g)
weight = torch.randn((output_size, reduction_size), device="cuda", dtype=torch.bfloat16, generator=g)

residual = torch.randn(batch_size, output_size, device="cuda", dtype=torch.bfloat16)

output = torch.empty(batch_size, output_size, device="cuda", dtype=torch.bfloat16)

runtime_kernel_cute_hopper.linear_mpk(weight, x, residual, output)

# torch_out = torch.matmul(weight, x)
torch_out = torch.matmul(x, weight.T)
torch_out = torch_out
print(output.shape)
print(torch_out.shape)
print("output from kernel:")
print(output)
print(torch_out)
print(torch.allclose(output, torch_out, atol=1e-2, rtol=1e-2))