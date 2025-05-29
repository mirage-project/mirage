import torch
from runtime_kernel import norm_linear

torch.set_printoptions(sci_mode=False)
rms_norm = torch.nn.RMSNorm(4096, dtype=torch.bfloat16, device="cuda")


def torch_rms_norm(X, W):
    D = rms_norm(X)
    E = torch.matmul(D, W)
    return E


# Define the input tensor and parameters
input_token = torch.randint(32768, (1, 1), dtype=torch.uint16, device="cuda")
embed_tokens = torch.randn((32768, 4096), dtype=torch.bfloat16, device="cuda")
layer_0_input_layernorm = torch.randn(4096, dtype=torch.bfloat16, device="cuda")
layer_0_q_proj = torch.randn((4096, 4096), dtype=torch.bfloat16, device="cuda")
layer_0_k_proj = torch.randn((4096, 1024), dtype=torch.bfloat16, device="cuda")
layer_0_v_proj = torch.randn((4096, 1024), dtype=torch.bfloat16, device="cuda")

embed_out = torch.empty((1, 4096), dtype=torch.bfloat16, device="cuda")
attn_in = torch.empty((1, 6144), dtype=torch.bfloat16, device="cuda")

embed_out = embed_tokens[input_token[0, 0], :]

q_tiles = [layer_0_q_proj[:, i * 32 : (i + 1) * 32] for i in range(128)]
k_tiles = [layer_0_k_proj[:, i * 32 : (i + 1) * 32] for i in range(32)]
v_tiles = [layer_0_v_proj[:, i * 32 : (i + 1) * 32] for i in range(32)]

layer_0_qkv_proj = torch.cat(
    [
        torch.cat(
            q_tiles[i * 16 : (i + 1) * 16]
            + k_tiles[i * 4 : (i + 1) * 4]
            + v_tiles[i * 4 : (i + 1) * 4],
            dim=1,
        )
        for i in range(8)
    ],
    dim=1,
)

weights = [
    layer_0_qkv_proj[:, i * 32 : (i + 1) * 32].clone().detach().to(torch.bfloat16)
    for i in range(192)
]

attn_in_tiles = [
    torch.zeros((1, 32), dtype=torch.bfloat16, device="cuda") for _ in range(192)
]

for i in range(192):
    norm_linear(
        embed_out,
        weights[i],
        attn_in_tiles[i],
    )
attn_in = torch.cat(attn_in_tiles, dim=1)

# Print the output tensor
print("Output from norm_linear:")
print(attn_in)
print("Shape of output tensor:", attn_in.shape)

# Compare the output with the expected output
expected_outputs = [
    torch_rms_norm(embed_out, layer_0_qkv_proj[:, i * 32 : (i + 1) * 32]).unsqueeze(0)
    for i in range(192)
]
expected_output = torch.cat(expected_outputs, dim=1)

print("Expected output:")
print(expected_output)
print("Shape of expected output tensor:", expected_output.shape)

print("Ratio (kernel / torch):")
print(attn_in / expected_output)
