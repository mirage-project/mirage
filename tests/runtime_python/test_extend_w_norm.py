import torch
import torch.nn.functional as F
import runtime_kernel
import numpy as np

# Fix random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)

torch.set_printoptions(sci_mode=False)#, linewidth=10000)

q_heads = 4
# q_heads = 1
k_heads = 1
v_heads = 1
head_dim = 128
num_total_heads = q_heads + k_heads + v_heads
max_seq_len = 512

device = "cuda"
dtype = torch.bfloat16

# print full tensor
torch.set_printoptions(threshold=float('inf'), sci_mode=False)

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    q_fp32 = q.to(torch.float32)
    k_fp32 = k.to(torch.float32)
    cos_fp32 = cos.to(torch.float32)
    sin_fp32 = sin.to(torch.float32)

    cos_fp32 = cos_fp32.unsqueeze(unsqueeze_dim)
    sin_fp32 = sin_fp32.unsqueeze(unsqueeze_dim)
    q_embed = (q_fp32 * cos_fp32) + (rotate_half(q_fp32) * sin_fp32)
    k_embed = (k_fp32 * cos_fp32) + (rotate_half(k_fp32) * sin_fp32)
    return q_embed.to(torch.bfloat16), k_embed.to(torch.bfloat16)


def rmsnorm(X, W, eps):
    X_fp32 = X.to(torch.float32)
    W_fp32 = W.to(torch.float32)

    variance = X_fp32.pow(2).mean(-1, keepdim=True)
    # print("torch square sum:", variance)
    inv_rms = torch.rsqrt(variance + eps)
    X_normed = X_fp32 * inv_rms
    out = X_normed * W_fp32
    return out.to(torch.bfloat16)


def attention_extend(q_tokens, k_new_tokens, v_new_tokens, k_cache, v_cache, valid_len, extend_num, q_weight, k_weight, eps, cos, sin):
    """
    PyTorch reference implementation for extend operation
    q_tokens: [extend_num+1, q_heads, head_dim] - raw Q tokens (not normalized)
    k_new_tokens: [extend_num+1, head_dim] - raw K tokens (not normalized)
    v_new_tokens: [extend_num+1, head_dim] - raw V tokens (not normalized)
    k_cache: [1, max_seq_len, head_dim] - existing K cache (already normalized)
    v_cache: [1, max_seq_len, head_dim] - existing V cache (already normalized)
    valid_len: current valid length before extend
    extend_num: number of new tokens to extend
    
    Returns: output, q_norm_debug, k_norm_debug - CUDA compatible format with debug tensors
    """
    # Normalize new Q tokens for each token and head
    q_norm_list = []
    for i in range(extend_num + 1):
        q_token_heads = q_tokens[i]  # [q_heads, head_dim]
        q_norm_token = rmsnorm(q_token_heads.unsqueeze(0), q_weight, eps).squeeze(0)  # [q_heads, head_dim]
        q_norm_list.append(q_norm_token)
    q_norm = torch.stack(q_norm_list, dim=0)  # [extend_num+1, q_heads, head_dim]
    
    # Normalize new K tokens
    k_norm = rmsnorm(k_new_tokens, k_weight, eps)
    
    # Apply rotary embedding to Q and K
    cos_extend = cos[valid_len-1:valid_len+extend_num].unsqueeze(1)  # [extend_num+1, 1, head_dim]
    sin_extend = sin[valid_len-1:valid_len+extend_num].unsqueeze(1)  # [extend_num+1, 1, head_dim]
    
    q_rot = []
    k_rot = []
    for i in range(extend_num + 1):
        # For Q: apply RoPE to each head of each token
        q_heads_rot = []
        for h in range(q_norm.shape[1]):  # q_heads
            q_h, k_i = apply_rotary_pos_emb(
                q_norm[i:i+1, h:h+1], k_norm[i:i+1], 
                cos_extend[i], sin_extend[i], unsqueeze_dim=1
            )
            q_heads_rot.append(q_h.squeeze(0))  # [1, head_dim]
        q_token_rot = torch.cat(q_heads_rot, dim=0)  # [q_heads, head_dim]
        q_rot.append(q_token_rot)
        
        # For K: only one head per token
        _, k_i = apply_rotary_pos_emb(
            q_norm[i:i+1, 0:1], k_norm[i:i+1], 
            cos_extend[i], sin_extend[i], unsqueeze_dim=1
        )
        k_rot.append(k_i.squeeze(0))  # [1, head_dim]
    
    q_rot = torch.stack(q_rot, dim=0)  # [extend_num+1, q_heads, head_dim]
    k_rot = torch.cat(k_rot, dim=0)  # [extend_num+1, head_dim]
    
    # Save debug info - reshape to match CUDA format
    q_norm_debug = q_rot.clone()  # [extend_num+1, q_heads, head_dim]
    # change to [(extend_num+1) * q_heads, head_dim]
    q_norm_debug = q_norm_debug.reshape((extend_num + 1) * q_heads, head_dim)
    k_norm_debug = k_rot.clone()  # [extend_num+1, head_dim]
    
    # Update K cache with normalized and rotated K tokens
    for i in range(extend_num + 1):
        k_cache[0, valid_len - 1 + i] = k_rot[i] # [1, head_dim]
    
    # Update V cache with new V tokens (V doesn't need rotary embedding)
    for i in range(extend_num + 1):
        v_cache[0, valid_len - 1 + i] = v_new_tokens[i]
    
    # Expand K and V for attention computation
    total_len = valid_len + extend_num
    print("Pytorch total_len:", total_len)
    k_all = k_cache[:, :total_len, :].expand(q_heads, -1, -1)  # [q_heads, total_len, head_dim]
    v_all = v_cache[:, :total_len, :].expand(q_heads, -1, -1)  # [q_heads, total_len, head_dim]
    
    # Compute attention for each Q token
    outputs = []
    for i in range(extend_num + 1):
        q_i = q_rot[i:i+1].transpose(0, 1)  # [q_heads, 1, head_dim]
        scores = torch.matmul(q_i, k_all.transpose(-2, -1)) / np.sqrt(head_dim) # [q_heads, 1, total_len]
        # print("first scores.shape:", scores.shape)
        
        # Apply causal mask
        token_pos = valid_len - 1 + i
        # mask = torch.arange(total_len, device=scores.device)[None, :] <= token_pos
        # TODO: This is wrong, only to align with problematic cuda implementation
        # mask = torch.arange(total_len, device=scores.device) <= total_len  # [total_len]
        mask = torch.arange(total_len, device=scores.device) <= token_pos
        # print("mask.shape:", mask.shape)
        
        # Expand mask to match scores shape [q_heads, 1, total_len]
        mask_expanded = mask.unsqueeze(0).unsqueeze(0).expand(q_heads, 1, -1)  # [q_heads, 1, total_len]
        scores = scores.masked_fill(~mask_expanded, float("-inf"))
        # print("scores.shape:", scores.shape)
        
        attn = F.softmax(scores, dim=-1) # [q_heads, 1, total_len] 
        # print("attn.shape:", attn.shape)
        # print("v_all.shape:", v_all.shape)
        
        out = torch.matmul(attn, v_all)  # [q_heads, 1, head_dim]
        outputs.append(out.squeeze(1))  # [q_heads, head_dim]
    
    # Convert to CUDA format: [(extend_num+1)*q_heads, head_dim]
    # Where each token's qheads are adjacent
    result = []
    for i in range(extend_num + 1):  # For each token
        for j in range(q_heads):  # For each qhead of this token
            result.append(outputs[i][j])  # [head_dim]
    
    ret = torch.stack(result, dim=0)  # [(extend_num+1)*q_heads, head_dim]
    print("PyTorch output shape:", ret.shape)
    return ret, q_norm_debug, k_norm_debug


def test_extend_correctness():
    """Test single_batch_extend with different extend numbers"""
    # extend_nums = [1, 2, 3]
    # extend_nums = [0]
    # extend_nums = [5]
    extend_nums = [0, 1, 2, 3, 4, 5]
    # test_seq_lens = [10, 50, 100]
    test_seq_lens = [10, 50, 100, 250, 500]
    # test_seq_lens = [10, 100, 250, 500]
    
    k_cache_torch = torch.empty((1, max_seq_len, head_dim), device=device, dtype=dtype)
    v_cache_torch = torch.empty((1, max_seq_len, head_dim), device=device, dtype=dtype)
    k_cache_mirage = torch.empty((max_seq_len, head_dim), device=device, dtype=dtype)
    v_cache_mirage = torch.empty((max_seq_len, head_dim), device=device, dtype=dtype)
    
    all_cos = torch.randn((513, head_dim), device=device, dtype=dtype)
    all_sin = torch.randn((513, head_dim), device=device, dtype=dtype)
    
    for seq_len in test_seq_lens:
        for extend_num in extend_nums:
            print(f"\nTesting seq_len={seq_len}, extend_num={extend_num}")
            
            # Prepare normalization weights first
            eps = 1e-6
            qnorm_weight = torch.randn((1, head_dim), device=device, dtype=dtype)
            knorm_weight = torch.randn((1, head_dim), device=device, dtype=dtype)
            
            # Initialize cache with random NORMALIZED data (simulating existing cache)
            for i in range(seq_len - 1):
                k_data = torch.randn(head_dim, device=device, dtype=dtype)
                v_data = torch.randn(head_dim, device=device, dtype=dtype)
                k_cache_torch[0, i] = k_data
                v_cache_torch[0, i] = v_data
                k_cache_mirage[i] = k_data
                v_cache_mirage[i] = v_data
            
            # Create QKV for extend_num + 1 tokens (last token + new tokens)
            total_extend_heads = q_heads * (extend_num + 1)
            qkv = torch.randn(extend_num + 1, (q_heads + k_heads + v_heads) * head_dim, device=device, dtype=dtype)
            
            # Extract Q, K, V from QKV tensor (these are RAW, not normalized)
            qkv_reshaped = qkv.view(extend_num + 1, q_heads + k_heads + v_heads, head_dim)
            
            # Extract Q tokens for current query (all extend_num + 1 tokens) - RAW
            q_tokens = qkv_reshaped[:, :q_heads, :]  # [extend_num + 1, q_heads, head_dim]
            
            # Extract K, V tokens (only the new tokens) - RAW
            k_new_tokens = qkv_reshaped[:, q_heads:q_heads+k_heads, :].squeeze(1)  # [extend_num + 1, head_dim]
            v_new_tokens = qkv_reshaped[:, q_heads+k_heads:, :].squeeze(1)  # [extend_num + 1, head_dim]
            
            # Debug: Print raw input data to verify consistency
            # print(f"\n=== Raw Input Data (before norm) ===")
            # print(f"Q tokens shape: {q_tokens.shape}")
            # print(f"K tokens shape: {k_new_tokens.shape}")
            # print(f"Q[-1]: {q_tokens[-1,:]}")
            # print(f"K[0,:5]: {k_new_tokens[0,:5]}")
            
            # NOTE: We DO NOT pre-populate the cache with new K, V tokens
            # The extend kernel will handle normalization and cache updates internally
            
            # PyTorch reference implementation
            torch_output, q_norm_debug, k_norm_debug = attention_extend(
                q_tokens.clone(),
                k_new_tokens.clone(),
                v_new_tokens.clone(),
                k_cache_torch.clone(),
                v_cache_torch.clone(),
                seq_len,
                extend_num,
                qnorm_weight,
                knorm_weight,
                eps,
                all_cos,
                all_sin
            )
            # print(torch_output)
            print(torch_output.shape)
            # exit()
            
            # Mirage implementation
            total_q_vec_num = q_heads * (extend_num + 1)
            mirage_output_flat = torch.empty((total_q_vec_num, head_dim), device=device, dtype=dtype)
            
            # Debug tensors for normalization outputs
            q_norm_debug_cuda = torch.empty((q_heads * (extend_num + 1), head_dim), device=device, dtype=dtype)
            k_norm_debug_cuda = torch.empty((extend_num + 1, head_dim), device=device, dtype=dtype)
            
            try:
                runtime_kernel.single_batch_extend(
                    qkv,
                    k_cache_mirage,
                    v_cache_mirage,
                    mirage_output_flat,
                    seq_len,
                    extend_num,
                    True,  # qk_norm
                    True,  # rotary_embed
                    qnorm_weight,
                    knorm_weight,
                    all_cos,
                    all_sin,
                    eps,
                    eps,
                    q_norm_debug_cuda,  # Debug Q norm output
                    k_norm_debug_cuda,  # Debug K norm output
                )
                
                # Both outputs are now in same format: [(extend_num+1)*q_heads, head_dim]
                # print("PyTorch output shape:", torch_output.shape)
                # print("CUDA output shape:", mirage_output_flat.shape)
                
                # Compare normalization results (for debugging)
                # print("\n=== Normalization Comparison ===")
                # print("extend_num used:", extend_num)
                # print("TOTAL_Q_VEC_NUM expected:", q_heads * (extend_num + 1))
                # print("PyTorch Q norm shape:", q_norm_debug.shape)
                # print("PyTorch K norm shape:", k_norm_debug.shape)
                # print("CUDA Q norm shape:", q_norm_debug_cuda.shape)
                # print("CUDA K norm shape:", k_norm_debug_cuda.shape)
                
                # Compare first few elements only
                # print("\n=== Q Norm Comparison (first few elements) ===")
                # print("PyTorch Q norm:", q_norm_debug)
                # print("CUDA Q norm:", q_norm_debug_cuda)
                # for i in range(q_heads * (extend_num), q_heads * (extend_num + 1)):
                #     print(f"PyTorch Q norm[{i}]:", q_norm_debug[i][:6])
                #     print(f"CUDA Q norm[{i}]:", q_norm_debug_cuda[i][:6])
                # print("Q norm ratio:", (q_norm_debug/ q_norm_debug_cuda)[-5:])
                # print("Q norm diff:", (q_norm_debug - q_norm_debug_cuda))
                # exit()
                
                # print("\n=== K Norm Comparison (first few elements) ===")
                # print("PyTorch K norm:", k_norm_debug)
                # print("CUDA K norm:", k_norm_debug_cuda)
                # print("K norm ratio:", (k_norm_debug / k_norm_debug_cuda)[-5:])
                # print("K norm diff:", (k_norm_debug - k_norm_debug_cuda))
                
                
                
                # Note: CUDA norm tensors are currently empty since kernel doesn't fill them yet
                # This is just to test the interface
                # exit()
                
                # Additional debugging: print first few elements to verify format
                # print("\n=== Final Output Comparison ===")
                # for i in range(q_heads * (extend_num + 1)):
                #     max_cur_row = torch.max(torch.abs(mirage_output_flat[i] - torch_output[i]))
                #     print(f"row: {i}, max_cur_row: {max_cur_row}")
                #     if max_cur_row > 0.1:
                #         print(f"PyTorch Q norm[{i}]:", mirage_output_flat[i])
                #         print(f"CUDA Q norm[{i}]:", torch_output[i])
                # print("torch_output[0:8, 0:5]:")
                # print(torch_output[:8, :5])
                # print("mirage_output_flat[0:8, 0:5]:")
                # print(mirage_output_flat[:8, :5])
                
                # Compare results directly
                diff = mirage_output_flat - torch_output
                max_diff = diff.abs().max().item()
                mean_diff = diff.abs().mean().item()
                
                print(f"  Max diff: {max_diff:.6f}")
                print(f"  Mean diff: {mean_diff:.6f}")
                
                if max_diff < 0.1:  # bfloat16 tolerance
                    print("  ✓ Test passed!")
                else:
                    print("  ✗ Test failed!")
                    print(f"  Torch output shape: {torch_output.shape}")
                    print(f"  Mirage output shape: {mirage_output_flat.shape}")
                    print(f"  Sample torch output: {torch_output[0, :100]}")
                    print(f"  Sample mirage output: {mirage_output_flat[0, :100]}")
                    # print(f"  Sample diff: {diff}")
                    print("ratio:", (mirage_output_flat / torch_output)[-5:])
                    print("diff:", (mirage_output_flat - torch_output)[-5:])
                    # 20 22 23 24
                    
            except Exception as e:
                print(f"  ✗ Error: {e}")


if __name__ == "__main__":
    test_extend_correctness() 