import triton
import triton.language as tl
import torch
import time

@triton.jit
def rotary_kernel(
    q_ptr, k_ptr, cos_ptr, sin_ptr, out_q_ptr, out_k_ptr,
    batch_size, seq_len, num_heads, num_kv_heads, head_dim, half_dim,
    q_stride_b, q_stride_s, q_stride_h, q_stride_d,
    k_stride_b, k_stride_s, k_stride_h, k_stride_d,
    cos_stride_b, cos_stride_s, cos_stride_d,
    sin_stride_b, sin_stride_s, sin_stride_d,
    out_q_stride_b, out_q_stride_s, out_q_stride_h, out_q_stride_d,
    out_k_stride_b, out_k_stride_s, out_k_stride_h, out_k_stride_d,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_s = tl.program_id(0)  
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)

    if pid_s * BLOCK_M >= seq_len:
        return
    
    q_ptr = q_ptr + pid_b * q_stride_b + pid_h * q_stride_h
    out_q_ptr = out_q_ptr + pid_b * out_q_stride_b + pid_h * out_q_stride_h
    
    # range of sequence
    rm = pid_s * BLOCK_M + tl.arange(0, BLOCK_M)
    s_mask = rm < seq_len
    
    q_base = pid_b * q_stride_b + pid_h * q_stride_h
    out_q_base = pid_b * out_q_stride_b + pid_h * out_q_stride_h
    
    rk_half = tl.arange(0, BLOCK_K // 2)
    rk_half_mask = rk_half < half_dim
    
    q_offset = q_base + (rm[:, None] * q_stride_s + rk_half[None, :] * q_stride_d) # half
    
    cos_offset = rm[:, None] * cos_stride_s + rk_half[None, :] * cos_stride_d
    sin_offset = rm[:, None] * sin_stride_s + rk_half[None, :] * sin_stride_d
    
    cos = tl.load(cos_ptr + cos_offset, mask=s_mask[:, None] & rk_half_mask[None, :], other=1.0)
    sin = tl.load(sin_ptr + sin_offset, mask=s_mask[:, None] & rk_half_mask[None, :], other=0.0)
    
    q_first = tl.load(q_ptr + q_offset, mask=s_mask[:, None] & rk_half_mask[None, :], other=0.0)
    q_second = tl.load(q_ptr + q_offset + half_dim * q_stride_d,
                      mask=s_mask[:, None] & rk_half_mask[None, :], other=0.0)
    
    q_out_first = q_first * cos - q_second * sin
    q_out_second = q_second * cos + q_first * sin
    
    out_q_offset = out_q_base + (rm[:, None] * out_q_stride_s + rk_half[None, :] * out_q_stride_d)
    
    tl.store(out_q_ptr + out_q_offset, q_out_first, 
             mask=s_mask[:, None] & rk_half_mask[None, :])
    tl.store(out_q_ptr + out_q_offset + half_dim * out_q_stride_d, q_out_second, 
             mask=s_mask[:, None] & rk_half_mask[None, :])

    # # Goal: reuse cos and sin
    # if pid_h < num_kv_heads:
    #     k_base = pid_b * k_stride_b + pid_h * k_stride_h
    #     k_offset = k_base + (rm[:, None] * k_stride_s + rk_half[None, :] * k_stride_d)
        
    #     k_first = tl.load(k_ptr + k_offset, mask=s_mask[:, None] & rk_half_mask[None, :], other=0.0)
    #     k_second = tl.load(k_ptr + k_offset + half_dim * k_stride_d, 
    #                     mask=s_mask[:, None] & rk_half_mask[None, :], other=0.0)
    #     k_out_first = k_first * cos - k_second * sin
    #     k_out_second = k_second * cos + k_first * sin
    #     out_k_base = pid_b * out_k_stride_b + pid_h * out_k_stride_h
        
    #     out_k_offset = out_k_base + (rm[:, None] * out_k_stride_s + rk_half[None, :] * out_k_stride_d)
    #     tl.store(out_k_ptr + out_k_offset, k_out_first, 
    #             mask=s_mask[:, None] & rk_half_mask[None, :])
    #     tl.store(out_k_ptr + out_k_offset + half_dim * out_k_stride_d, k_out_second, 
    #             mask=s_mask[:, None] & rk_half_mask[None, :])
        
def apply_rotary_pos_emb_triton(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """
    Triton implementation of apply_rotary_pos_emb with 3D grid.
    
    Args:
        q: Query tensor of shape [batch, seq_len, num_heads, head_dim]
        k: Key tensor of shape [batch, seq_len, num_kv_heads, head_dim]
        cos/sin: Tensors from rotary embedding
        position_ids: Optional tensor of position indices
        unsqueeze_dim: Dimension to unsqueeze the cosine/sine tensors
    
    Returns:
        Tuple of rotated query and key tensors
    """
    # cos = cos.unsqueeze(unsqueeze_dim)
    # sin = sin.unsqueeze(unsqueeze_dim)

    # debug
    # cos = torch.arange(1, cos.numel() + 1, dtype=cos.dtype, device=cos.device).reshape(cos.shape)
    # print("cos shape: ", cos.shape)
    # print("sin shape: ", sin.shape)
    time0 = time.time()
    batch_size, seq_len, num_heads, head_dim = q.shape
    _, _, num_kv_heads, _ = k.shape
    half_dim = head_dim // 2
    
    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)
    
    q_stride = q.stride()
    k_stride = k.stride()
    cos_stride = cos.stride()
    sin_stride = sin.stride()
    q_out_stride = q_out.stride()
    k_out_stride = k_out.stride()
    
    BLOCK_K = min(triton.next_power_of_2(half_dim), 128)
    BLOCK_M = min(8, triton.next_power_of_2(min(seq_len, 32)))
    
    # Dimensions: (sequence blocks, batch, heads)
    BLOCK_K = (
        32
        if head_dim <= 32
        else (64 if head_dim <= 64 else (128 if head_dim <= 128 else 256))
    )
    BLOCK_M = 4 #if interleaved else (8 if head_dim <= 128 else 4)
    # Block num is about seq_len * batch_size * num_heads * head_dim / 1024
    grid = lambda META: (triton.cdiv(seq_len, META["BLOCK_M"]), batch_size, num_heads)  # noqa
    # print("grid: ", (triton.cdiv(seq_len, BLOCK_M), batch_size, num_heads))
    # print("Per block M: ", BLOCK_M)
    # print("Per block K: ", BLOCK_K)
    # q_first_half = torch.empty(1, BLOCK_M, 1, head_dim//2, device='cuda', dtype=torch.bfloat16)
    # q_second_half = torch.empty(1, BLOCK_M, 1, head_dim//2, device='cuda', dtype=torch.bfloat16)
    # q_out_first_debug = torch.empty(1, BLOCK_M, 1, head_dim//2, device='cuda', dtype=torch.bfloat16)
    # q_out_second_debug = torch.empty(1, BLOCK_M, 1, head_dim//2, device='cuda', dtype=torch.bfloat16)
    # cos_load_debug = torch.empty(1, BLOCK_M, 1, head_dim//2, device='cuda', dtype=torch.bfloat16)
    # sin_load_debug = torch.empty(1, BLOCK_M, 1, head_dim//2, device='cuda', dtype=torch.bfloat16)
    # print("q_first_half.shape: ", q_first_half.shape)
    # print("q_second_half.shape: ", q_second_half.shape)
    print("My grid: ", (triton.cdiv(seq_len, BLOCK_M), batch_size, num_heads))
    time1 = time.time()
    with torch.cuda.device(q.device):
        rotary_kernel[grid](
            q, k, cos, sin, q_out, k_out,
            batch_size, seq_len, num_heads, num_kv_heads, head_dim, half_dim,
            q_stride[0], q_stride[1], q_stride[2], q_stride[3],
            k_stride[0], k_stride[1], k_stride[2], k_stride[3],
            cos_stride[0], cos_stride[1], cos_stride[2],
            sin_stride[0], sin_stride[1], sin_stride[2],
            q_out_stride[0], q_out_stride[1], q_out_stride[2], q_out_stride[3],
            k_out_stride[0], k_out_stride[1], k_out_stride[2], k_out_stride[3],
            BLOCK_M, BLOCK_K,
        )
    time2 = time.time()
    # print("q_first_half_loaded: ", q_first_half)
    # print("q_second_half_loaded: ", q_second_half)
    # print("q_out_first_debug_loaded: ", q_out_first_debug)
    # print("q_out_second_debug_loaded: ", q_out_second_debug)
    # print("real cos: ", cos)
    # print("real sin: ", sin)
    # print("cos_load_debug_loaded: ", cos_load_debug)
    # print("sin_load_debug_loaded: ", sin_load_debug)
    print("--------------------------------")
    print(f"Before kernel time: {time1 - time0:.8f} seconds")
    print(f"Triton Kernel time: {time2 - time1:.8f} seconds")
    print("--------------------------------")
    return q_out, k_out
