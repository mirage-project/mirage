import torch
import triton
import triton.language as tl

@triton.jit
def apply_rope_kernel(
    q_ptr, k_ptr, 
    cos_ptr, sin_ptr, 
    q_out_ptr, k_out_ptr,
    seq_len, head_dim, num_q_heads, num_k_heads,
    q_stride_b, q_stride_s, q_stride_h, q_stride_d,
    k_stride_b, k_stride_s, k_stride_h, k_stride_d,
    cos_stride_b, cos_stride_s, cos_stride_d,
    sin_stride_b, sin_stride_s, sin_stride_d,
    q_out_stride_b, q_out_stride_s, q_out_stride_h, q_out_stride_d,
    k_out_stride_b, k_out_stride_s, k_out_stride_h, k_out_stride_d,
    batch_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(0)
    
    # Calculate indices
    batch_id = pid // (seq_len * (num_q_heads + num_k_heads))
    pid_within_batch = pid % (seq_len * (num_q_heads + num_k_heads))
    seq_id = pid_within_batch // (num_q_heads + num_k_heads)
    head_id = pid_within_batch % (num_q_heads + num_k_heads)
    
    is_query = head_id < num_q_heads
    actual_head_id = head_id if is_query else head_id - num_q_heads
    
    # Set up pointers
    if is_query:
        ptr = q_ptr + batch_id * q_stride_b + seq_id * q_stride_s + actual_head_id * q_stride_h
        out_ptr = q_out_ptr + batch_id * q_out_stride_b + seq_id * q_out_stride_s + actual_head_id * q_out_stride_h
        dim_stride = q_stride_d
        out_dim_stride = q_out_stride_d
    else:
        ptr = k_ptr + batch_id * k_stride_b + seq_id * k_stride_s + actual_head_id * k_stride_h
        out_ptr = k_out_ptr + batch_id * k_out_stride_b + seq_id * k_out_stride_s + actual_head_id * k_out_stride_h
        dim_stride = k_stride_d
        out_dim_stride = k_out_stride_d
    
    cos_ptr_base = cos_ptr + batch_id * cos_stride_b + seq_id * cos_stride_s
    sin_ptr_base = sin_ptr + batch_id * sin_stride_b + seq_id * sin_stride_s
    
    # Load cos and sin values
    offs_d = tl.arange(0, BLOCK_SIZE)
    mask_d = offs_d < head_dim
    
    # Handle half dimension operations
    half_dim = head_dim // 2
    
    # Process the first half
    first_half_mask = mask_d & (offs_d < half_dim)
    first_half_idx = offs_d
    
    # Values from the first half
    x1 = tl.load(ptr + first_half_idx * dim_stride, mask=first_half_mask, other=0.0)
    cos_values = tl.load(cos_ptr_base + first_half_idx * cos_stride_d, mask=first_half_mask, other=0.0)
    sin_values = tl.load(sin_ptr_base + first_half_idx * sin_stride_d, mask=first_half_mask, other=0.0)
    
    # Apply rotary embeddings for first half
    result_first_half = x1 * cos_values
    
    # Store first half results
    tl.store(out_ptr + first_half_idx * out_dim_stride, result_first_half, mask=first_half_mask)
    
    # Process the second half
    second_half_mask = mask_d & (offs_d >= half_dim)
    second_half_idx = offs_d
    second_half_idx_aligned = offs_d - half_dim
    
    # Values from the second half
    x2 = tl.load(ptr + second_half_idx * dim_stride, mask=second_half_mask, other=0.0)
    
    # For the second half, apply rotations
    # We need to get cos and sin values again but for the first half indices
    cos_values_second = tl.load(
        cos_ptr_base + second_half_idx_aligned * cos_stride_d, 
        mask=second_half_mask, 
        other=0.0
    )
    sin_values_second = tl.load(
        sin_ptr_base + second_half_idx_aligned * sin_stride_d, 
        mask=second_half_mask, 
        other=0.0
    )
    
    # Apply rotary embeddings for second half
    result_second_half = x2 * cos_values_second
    
    # Store second half results
    tl.store(out_ptr + second_half_idx * out_dim_stride, result_second_half, mask=second_half_mask)
    
    # Now handle the rotated components
    # Load the first half again for applying to the second half position
    x1_for_rotate = tl.load(ptr + first_half_idx * dim_stride, mask=first_half_mask, other=0.0)
    
    # Load the second half again for applying to the first half position (with negation)
    x2_for_rotate = tl.load(ptr + (second_half_idx_aligned + half_dim) * dim_stride, mask=first_half_mask, other=0.0)
    
    # Apply rotary embeddings with rotated positions and sum with the previous results
    result_rotate_first = tl.load(out_ptr + first_half_idx * out_dim_stride, mask=first_half_mask, other=0.0)
    result_rotate_first += (-x2_for_rotate) * sin_values
    tl.store(out_ptr + first_half_idx * out_dim_stride, result_rotate_first, mask=first_half_mask)
    
    # Similarly for the second half positions
    result_rotate_second = tl.load(out_ptr + (second_half_idx_aligned + half_dim) * out_dim_stride, mask=first_half_mask, other=0.0)
    result_rotate_second += x1_for_rotate * sin_values
    tl.store(out_ptr + (second_half_idx_aligned + half_dim) * out_dim_stride, result_rotate_second, mask=first_half_mask)


def apply_rotary_pos_emb_triton(q, k, cos, sin, unsqueeze_dim=1):
    """
    Apply rotary positional embeddings to q and k tensors using a Triton kernel.
    
    Args:
        q: query tensor of shape [batch_size, seq_len, num_heads, head_dim]
        k: key tensor of shape [batch_size, seq_len, num_kv_heads, head_dim]  
        cos: cosine tensor of shape [batch_size, seq_len, head_dim]
        sin: sine tensor of shape [batch_size, seq_len, head_dim]
        unsqueeze_dim: dimension to unsqueeze cos and sin tensors
    
    Returns:
        q_embed: query tensor with positional embeddings applied
        k_embed: key tensor with positional embeddings applied
    """
    # Unsqueeze cos and sin
    time0 = time.time()
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    
    # Get shapes
    batch_size, seq_len, num_q_heads, head_dim = q.shape
    _, _, num_k_heads, _ = k.shape
    
    # Allocate output tensors
    q_embed = torch.empty_like(q)
    k_embed = torch.empty_like(k)
    
    # Compute strides
    q_stride_b, q_stride_s, q_stride_h, q_stride_d = q.stride()
    k_stride_b, k_stride_s, k_stride_h, k_stride_d = k.stride()
    
    cos_stride = cos.stride()
    cos_stride_b, cos_stride_s, _, cos_stride_d = cos_stride if len(cos_stride) == 4 else (0, cos_stride[0], 0, cos_stride[-1])
    
    sin_stride = sin.stride()
    sin_stride_b, sin_stride_s, _, sin_stride_d = sin_stride if len(sin_stride) == 4 else (0, sin_stride[0], 0, sin_stride[-1])
    
    q_out_stride_b, q_out_stride_s, q_out_stride_h, q_out_stride_d = q_embed.stride()
    k_out_stride_b, k_out_stride_s, k_out_stride_h, k_out_stride_d = k_embed.stride()
    
    # Define grid size
    grid = (batch_size * seq_len * (num_q_heads + num_k_heads),)
    
    # Determine block size based on head dimension (rounded up to nearest multiple of 32)
    BLOCK_SIZE = triton.next_power_of_2(head_dim)
    BLOCK_SIZE = min(BLOCK_SIZE, 1024)  # cap at 1024
    
    # Launch kernel
    time1 = time.time()
    apply_rope_kernel[grid](
        q, k,
        cos, sin,
        q_embed, k_embed,
        seq_len, head_dim, num_q_heads, num_k_heads,
        q_stride_b, q_stride_s, q_stride_h, q_stride_d,
        k_stride_b, k_stride_s, k_stride_h, k_stride_d,
        cos_stride_b, cos_stride_s, cos_stride_d,
        sin_stride_b, sin_stride_s, sin_stride_d,
        q_out_stride_b, q_out_stride_s, q_out_stride_h, q_out_stride_d,
        k_out_stride_b, k_out_stride_s, k_out_stride_h, k_out_stride_d,
        batch_size=batch_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    time2 = time.time()
    print("Before kernel time: ", time1 - time0)
    print(f"Triton Kernel time: {time2 - time1:.8f} seconds")
    return q_embed, k_embed