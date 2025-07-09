# test_multitoken_decoding.py
import torch
import runtime_kernel
import numpy as np

q_heads = 4
k_heads = 1
v_heads = 1
head_dim = 128
num_tokens = 3  # Test with 3 tokens
max_seq_len = 512

device = "cuda"
dtype = torch.bfloat16

# Initialize caches
k_cache_mirage = torch.empty((max_seq_len, head_dim), device=device, dtype=dtype)
v_cache_mirage = torch.empty((max_seq_len, head_dim), device=device, dtype=dtype)

# Test with multiple tokens
seq_len = 10

# Create input for multiple tokens - 5 heads per token (4 Q + 1 K + 1 V)
qkv = torch.randn(num_tokens * (q_heads + k_heads + v_heads), head_dim, device=device, dtype=dtype)

# Expected output shape: (num_tokens * q_heads, head_dim)
mirage_output = torch.empty((num_tokens * q_heads, head_dim), device=device, dtype=dtype)

# Call your multitoken kernel with all required parameters
runtime_kernel.single_batch_multitoken_decoding(
    qkv, k_cache_mirage, v_cache_mirage, mirage_output, seq_len,
    qk_norm=False,  # No QK normalization
    rotary_embed=False,  # No rotary embeddings
    qnorm_weight=None,  # No Q normalization weights
    knorm_weight=None,  # No K normalization weights
    cos=None,  # No cosine for rotary
    sin=None,  # No sine for rotary
    q_eps=0.0,  # Q epsilon for normalization
    k_eps=0.0   # K epsilon for normalization
)

expected_output_shape = (num_tokens * q_heads, head_dim)
print("Output shape:", mirage_output.shape)
print("Expected output shape:", expected_output_shape)
print("First token output:", mirage_output[:q_heads])

# Add comprehensive validation
print(f"Input qkv shape: {qkv.shape}")
print(f"Expected qkv shape: {num_tokens * (q_heads + k_heads + v_heads)} x {head_dim}")
print(f"Number of tokens: {num_tokens}")
print(f"Heads per token: {q_heads + k_heads + v_heads}")

assert mirage_output.shape == expected_output_shape, f"Expected {expected_output_shape}, got {mirage_output.shape}"

# Test with different token counts
for test_tokens in [1, 2, 4]:
    print(f"\nTesting with {test_tokens} tokens:")
    test_qkv = torch.randn(test_tokens * (q_heads + k_heads + v_heads), head_dim, device=device, dtype=dtype)
    test_output = torch.empty((test_tokens * q_heads, head_dim), device=device, dtype=dtype)
    
    try:
        runtime_kernel.single_batch_multitoken_decoding(
            test_qkv, k_cache_mirage, v_cache_mirage, test_output, seq_len,
            qk_norm=False,
            rotary_embed=False,
            qnorm_weight=None,
            knorm_weight=None,
            cos=None,
            sin=None,
            q_eps=0.0,
            k_eps=0.0
        )
        print(f"✓ {test_tokens} tokens: Success")
        print(f"  Input shape: {test_qkv.shape}")
        print(f"  Output shape: {test_output.shape}")
    except Exception as e:
        print(f"✗ {test_tokens} tokens: Failed - {e}")

print("\nAll tests completed!")

