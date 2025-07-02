import torch
from runtime_kernel import window_rms_norm

torch.set_printoptions(sci_mode=False)
seed_value = 611
torch.manual_seed(seed_value)
# TODO: Just support bf16 yet, try to support other data type later.
dtype=torch.bfloat16
device="cuda"

def precompute_rope_freqs_cis(
    dim,
    max_position_embeddings=2048,
    base=10000.0,
    dtype=dtype,
    device=device
):
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=dtype, device=device) / dim))
    t = torch.arange(max_position_embeddings, dtype=dtype, device=device)

    freqs = torch.outer(t, inv_freq)
    emb = torch.stack([freqs, freqs], dim=-1).reshape(*freqs.shape[:-1], -1)

    cos = emb.cos().to(dtype)
    sin = emb.sin().to(dtype)

    return cos, sin

def apply_rope_rotation(x, cos, sin, position_ids):
    assert x.size(-2) == position_ids.size(-1), "seq_len of x should be equal to size of position_ids"
    cos_pos = cos[position_ids].to(x.dtype)
    sin_pos = sin[position_ids].to(x.dtype)

    x_rot_real = x[..., :x.shape[-1] // 2]
    x_rot_imag = x[..., x.shape[-1] // 2:]

    x_out_real = x_rot_real * cos_pos[..., :x.shape[-1] // 2] - x_rot_imag * sin_pos[..., :x.shape[-1] // 2]
    x_out_imag = x_rot_real * sin_pos[..., x.shape[-1] // 2:] + x_rot_imag * cos_pos[..., x.shape[-1] // 2:]

    x_rotated = torch.cat((x_out_real, x_out_imag), dim=-1)

    return x_rotated

def torch_rms_norm(X, W, eps, cos=None, sin=None, position_ids=None):
    variance = X.pow(2).mean(-1, keepdim=True)
    X = X * torch.rsqrt(variance + eps)
    X = torch.mul(X, W)
    if cos != None and sin != None and position_ids != None:
        X = apply_rope_rotation(X, cos, sin, position_ids)
    return X

# Define the input tensor and parameters
EPS = 1e-5
batch_size = 1
window_size = 3
head_dim = 128

input_tensor = torch.randn((batch_size, window_size, head_dim), dtype=dtype, device=device)
torch_output = torch.randn((batch_size, window_size, head_dim), dtype=dtype, device=device)
kernel_output = torch.randn((batch_size, window_size, head_dim), dtype=dtype, device=device)
weights = torch.randn((head_dim), dtype=dtype, device=device)

cos, sin = precompute_rope_freqs_cis(head_dim)
position_ids = torch.arange(0, window_size, dtype=torch.int32, device=device).unsqueeze(0)
print("cos: ", cos)


window_rms_norm(
    input_tensor,
    weights,
    EPS,
    kernel_output,
    True,
    cos,
    sin,
)

# Print the output tensor
print("Output from window_norm:")
print(kernel_output)
print("Shape of output tensor:", kernel_output.shape)

# Compare the output with the expected output
torch_output = torch_rms_norm(input_tensor, weights, EPS, cos, sin, position_ids)
print("Output from expected torch window norm:")
print(torch_output)
print("Shape of expected tensor:", torch_output.shape)

print("===========================")
print("Ratio (kernel / torch):")
print(torch_output / kernel_output)