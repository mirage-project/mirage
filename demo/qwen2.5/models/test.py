import triton
import triton.language as tl
import torch
import time

from ropev2 import apply_rotary_pos_emb_triton
from real_flash_rope import apply_rotary

# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def test_rotary_embeddings():
    """Test function to compare the Triton implementation with the PyTorch implementation."""
    import torch
    
    # Setup test tensors
    batch_size, seq_len, num_heads, num_kv_heads, head_dim = 1, 39, 16, 2, 128
    # batch_size, seq_len, num_heads, num_kv_heads, head_dim = 1, 8, 2, 2, 8
    # q = torch.randn(batch_size, seq_len, num_heads, head_dim, device='cuda', dtype=torch.bfloat16)
    # k = torch.randn(batch_size, seq_len, num_kv_heads, head_dim, device='cuda', dtype=torch.bfloat16)
    q = torch.ones(batch_size, seq_len, num_heads, head_dim, device='cuda', dtype=torch.bfloat16)
    k = torch.ones(batch_size, seq_len, num_kv_heads, head_dim, device='cuda', dtype=torch.bfloat16)
    
    # Create position embeddings (match dimensions with Qwen2 implementation)
    inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device='cuda').float() / head_dim))
    # inv_freq = torch.zeros(head_dim//2, device='cuda', dtype=torch.bfloat16)
    #  inv_freq = torch.ones(head_dim//2, device='cuda', dtype=torch.bfloat16) * (math.pi / 2)
    pos = torch.arange(seq_len, device='cuda').float()
    freqs = torch.outer(pos, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().to(dtype=torch.bfloat16)
    sin = emb.sin().to(dtype=torch.bfloat16)
    
    # Copy cos/sin to match batch dimension for broadcasting
    cos = cos.expand(batch_size, seq_len, head_dim)
    sin = sin.expand(batch_size, seq_len, head_dim)
    
    # Use same unsqueeze_dim as in the model
    unsqueeze_dim = 2
    
    # Run PyTorch implementation
    q_pytorch = q.clone()
    k_pytorch = k.clone()
    q_pytorch_flash = torch.empty_like(q)
    q_pytorch_flash.copy_(q)
    k_pytorch_flash = torch.empty_like(k)
    k_pytorch_flash.copy_(k)
    # k_pytorch_flash = k.clone()
    # clear q as zero

    start_time = time.time()
    q_pytorch_out, k_pytorch_out = apply_rotary_pos_emb(q_pytorch, k_pytorch, cos, sin, unsqueeze_dim=unsqueeze_dim)
    end_time = time.time()
    print(f"PyTorch implementation time: {end_time - start_time:.8f} seconds")
    
    # Run our implementation
    triton_start_time = time.time()
    half_cos = cos
    half_sin = sin
    q_triton_out = apply_rotary_pos_emb_triton(q, k, cos, sin, unsqueeze_dim=unsqueeze_dim)
    triton_end_time = time.time()
    print(f"Triton implementation time: {triton_end_time - triton_start_time:.8f} seconds")
    half_cos = cos[0][:, :head_dim//2]
    half_sin = sin[0][:, :head_dim//2]
    
    # Run flash_rope implementation
    flash_rope_start_time = time.time()
    q_flash_rope_out= apply_rotary(q_pytorch_flash, half_cos, half_sin, interleaved=False)
    k_flash_rope_out = apply_rotary(k_pytorch_flash, half_cos, half_sin, interleaved=False)
    flash_rope_end_time = time.time()
    print(f"Flash_rope implementation time: {flash_rope_end_time - flash_rope_start_time:.8f} seconds")
    
    # Check results
    q_diff = torch.abs(q_pytorch_out - q_triton_out)
    # k_diff = torch.abs(k_pytorch_out - k_triton_out)
    q_err = q_diff.max().item()
    # k_err = k_diff.max().item()

    q_diff_flash = torch.abs(q_pytorch_out - q_flash_rope_out)
    k_diff_flash = torch.abs(k_pytorch_out - k_flash_rope_out)
    q_err_flash = q_diff_flash.max().item()
    k_err_flash = k_diff_flash.max().item()

    # print(f"q_pytorch_out: {q_pytorch_out}")
    # print(f"q_triton_out: {q_triton_out}")
    # print(f"k_pytorch_out: {k_pytorch_out}")
    # print(f"k_triton_out: {k_triton_out}")
    
    print(f"Max Q error: {q_err:.8f}")
    # print(f"Max K error: {k_err:.8f}")
    print(f"Average Q error: {q_diff.mean().item():.8f}")
    # print(f"Average K error: {k_diff.mean().item():.8f}")
    # print(f"Test {'passed' if q_err < 1e-3 and k_err < 1e-3 else 'failed'}")

    print(f"Max Q error flash: {q_err_flash:.8f}")
    print(f"Max K error flash: {k_err_flash:.8f}")
    print(f"Average Q error flash: {q_diff_flash.mean().item():.8f}")
    print(f"Average K error flash: {k_diff_flash.mean().item():.8f}")

    # print("q_pytorch_out: ", q_pytorch_out)
    # print("q_triton_out: ", q_triton_out)
    # print("q_flash_rope_out: ", q_flash_rope_out)
    
    
    # return q_err < 1e-3 #and k_err < 1e-3

if __name__ == "__main__":
    test_rotary_embeddings()

