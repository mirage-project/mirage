import torch
from runtime_kernel import window_rms_norm

torch.set_printoptions(sci_mode=False)
seed_value = 611
torch.manual_seed(seed_value)

def torch_rms_norm(X, W, eps):
    # TODO(Wenqin): add rope test
    variance = X.pow(2).mean(-1, keepdim=True)
    X = X * torch.rsqrt(variance + eps)
    X = torch.mul(X, W)
    return X

# Define the input tensor and parameters
EPS = 1e-5
batch_size = 1
window_size = 2
head_dim = 128

input_tensor = torch.randn((batch_size, window_size, head_dim), dtype=torch.bfloat16, device="cuda")
torch_output = torch.randn((batch_size, window_size, head_dim), dtype=torch.bfloat16, device="cuda")
kernel_output = torch.randn((batch_size, window_size, head_dim), dtype=torch.bfloat16, device="cuda")
weights = torch.randn((head_dim), dtype=torch.bfloat16, device="cuda")
# input_tensor = torch.ones((batch_size, window_size, head_dim), dtype=torch.bfloat16, device="cuda")
# torch_output = torch.ones((batch_size, window_size, head_dim), dtype=torch.bfloat16, device="cuda")
# kernel_output = torch.ones((batch_size, window_size, head_dim), dtype=torch.bfloat16, device="cuda")
# weights = torch.ones((head_dim), dtype=torch.bfloat16, device="cuda")
# weights /= 2


window_rms_norm(
    input_tensor,
    weights,
    EPS,
    kernel_output,
)

# Print the output tensor
print("Output from window_norm:")
print(kernel_output)
print("Shape of output tensor:", kernel_output.shape)

# Compare the output with the expected output
torch_output = torch_rms_norm(input_tensor, weights, EPS)
print("Output from expected window norm:")
print(torch_output)
print("Shape of expected tensor:", torch_output.shape)

print("===========================")
print("Ratio (kernel / torch):")
print(torch_output / kernel_output)
