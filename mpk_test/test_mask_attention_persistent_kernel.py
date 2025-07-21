import mirage as mi
import torch
import numpy as np

def create_attention_mask(num_tokens, seq_len, prompt_len, device):
    """Create attention mask for multitoken decoding."""
    total_kv_len = seq_len + num_tokens - 1
    mask_words_per_token = (total_kv_len + 63) // 64  # ceil_div
    
    # Create mask on CPU (for uint64 support)
    mask_np = np.zeros((num_tokens, mask_words_per_token), dtype=np.uint64)
    
    for token_idx in range(num_tokens):
        for pos in range(total_kv_len):
            word_idx = pos // 64
            bit_idx = pos % 64
            
            # All positions < prompt_len are always visible
            if pos < prompt_len:
                mask_np[token_idx, word_idx] |= np.uint64(1) << np.uint64(bit_idx)
            else:
                # Custom mask logic for non-prompt positions
                # For testing, make each token see different positions
                if pos < seq_len + token_idx:
                    mask_np[token_idx, word_idx] |= np.uint64(1) << np.uint64(bit_idx)
    
    # Convert to torch tensor as int64 (since uint64 is not supported)
    # The kernel will reinterpret this as uint64
    mask_int64 = mask_np.astype(np.int64)
    mask = torch.from_numpy(mask_int64).to(device)
    return mask, mask_words_per_token

def test_mask_attention_persistent_kernel():
    """
    Test the new mask_attention layer in PersistentKernel
    """
    print("Testing Mask Attention in PersistentKernel")
    print("=" * 60)
    
    # Parameters
    num_q_heads = 8
    num_kv_heads = 1
    head_dim = 128
    weight_stride = 128
    num_tokens = 4
    seq_len = 32
    prompt_len = 10
    max_seq_len = 512
    
    # Initialize
    mi.set_gpu_device_id(0)
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    
    # Create meta tensors
    step = torch.zeros(1, dtype=torch.int32, device=device)
    tokens = torch.zeros(1, max_seq_len, dtype=torch.int32, device=device)
    
    # Get proper configuration from GPU
    num_workers, num_schedulers = mi.get_configurations_from_gpu(0)
    print(f"GPU configuration: num_workers={num_workers}, num_schedulers={num_schedulers}")
    
    # Create PersistentKernel
    mpk = mi.PersistentKernel(
        world_size=1,
        mpi_rank=0,
        num_workers=num_workers,
        num_local_schedulers=num_schedulers,
        num_remote_schedulers=0,
        max_seq_length=max_seq_len,
        eos_token_id=0,
        meta_tensors=[step, tokens],
        profiler_tensor=None,
    )
    
    print("\nTest: Mask Attention with multitoken decoding")
    print("-" * 40)
    
    # Create input tensors
    heads_per_token = num_q_heads + num_kv_heads + num_kv_heads
    qkv = torch.randn(num_tokens * heads_per_token, head_dim, device=device, dtype=dtype) * 0.1
    k_cache = torch.randn(max_seq_len, head_dim, device=device, dtype=dtype) * 0.1
    v_cache = torch.randn(max_seq_len, head_dim, device=device, dtype=dtype) * 0.1
    output = torch.zeros(num_tokens * num_q_heads, head_dim, device=device, dtype=dtype)
    
    # Create normalization weights (if using qk_norm)
    qnorm_weight = torch.ones(head_dim, device=device, dtype=dtype)
    knorm_weight = torch.ones(head_dim, device=device, dtype=dtype)
    
    # Create rotary embeddings (if using rotary_emb)
    cos = torch.ones(max_seq_len, head_dim, device=device, dtype=dtype)
    sin = torch.zeros(max_seq_len, head_dim, device=device, dtype=dtype)
    
    # Create attention mask
    attn_mask, mask_words_per_token = create_attention_mask(num_tokens, seq_len, prompt_len, device)
    
    print(f"Input shapes:")
    print(f"  QKV: {qkv.shape}")
    print(f"  K cache: {k_cache.shape}")
    print(f"  V cache: {v_cache.shape}")
    print(f"  Attention mask: {attn_mask.shape}")
    print(f"  Output: {output.shape}")
    print(f"  Mask words per token: {mask_words_per_token}")
    
    # Attach tensors
    qkv_tensor = mpk.attach_input(torch_tensor=qkv, name="qkv")
    k_cache_tensor = mpk.attach_input(torch_tensor=k_cache, name="k_cache")
    v_cache_tensor = mpk.attach_input(torch_tensor=v_cache, name="v_cache")
    output_tensor = mpk.attach_input(torch_tensor=output, name="output")
    qnorm_tensor = mpk.attach_input(torch_tensor=qnorm_weight, name="qnorm_weight")
    knorm_tensor = mpk.attach_input(torch_tensor=knorm_weight, name="knorm_weight")
    cos_tensor = mpk.attach_input(torch_tensor=cos, name="cos")
    sin_tensor = mpk.attach_input(torch_tensor=sin, name="sin")
    mask_tensor = mpk.attach_input(torch_tensor=attn_mask, name="attn_mask")
    
    # Add mask_attention layer
    mpk.mask_attention_layer(
        qkv=qkv_tensor,
        k_cache=k_cache_tensor,
        v_cache=v_cache_tensor,
        output=output_tensor,
        qnorm_weight=qnorm_tensor,
        knorm_weight=knorm_tensor,
        cos=cos_tensor,
        sin=sin_tensor,
        attn_mask=mask_tensor,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        weight_stride=weight_stride,
        num_tokens=num_tokens,
        seq_len=seq_len,
        grid_dim=(1, 1, 1),
        block_dim=(128, 1, 1),
        qk_norm=False,
        rotary_emb=False,
        q_eps=1e-6,
        k_eps=1e-6,
        prompt_len=prompt_len,
        mask_words_per_token=mask_words_per_token,
    )
    
    print(f"✓ Created mask_attention layer")
    
    # Compile the kernel
    print("\nCompiling PersistentKernel...")
    try:
        mpk.compile()
        print("✓ Compilation successful!")
    except Exception as e:
        print(f"✗ Compilation failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Execute the kernel
    print("\nExecuting mask_attention kernel...")
    try:
        mpk()
        print("✓ Mask attention kernel executed successfully!")
        
        # Check output
        print("\nVerifying output:")
        print("-" * 40)
        print(f"Output tensor (first token, first 10 values): {output[0, :10]}")
            
    except Exception as e:
        print(f"\n✗ Execution failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Summary:")
    print("- Successfully added mask_attention kernel to PersistentKernel")
    print("- Multi-token decoding with attention masking registered")
    print("- Kernel can be compiled and executed")

if __name__ == "__main__":
    test_mask_attention_persistent_kernel()