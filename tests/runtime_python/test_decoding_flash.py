import torch
import torch.nn.functional as F
import runtime_kernel
import numpy as np
torch.set_printoptions(sci_mode=False)

q_heads = 4
k_heads = 1
v_heads = 1
head_dim = 128
num_total_heads = q_heads + k_heads + v_heads

def attention_decode(q,k_cache, v_cache):
    k = k_cache.expand(q_heads, -1, -1)
    v = v_cache.expand(q_heads, -1, -1)
    scores = torch.matmul(q, k.transpose(-2, -1))
    # mask
    T_kv = k.size(1)
    mask = torch.arange(T_kv, device=scores.device)[None, :] <= (T_kv - 1)
    scores = scores.masked_fill(~mask[None, None, :], float("-inf"))
    
    attn = F.softmax(scores, dim=-1) 
    # print(attn)
    
    out = torch.matmul(attn, v) 
    return out
for i in range(1):
    seq_len = 63
    # k_cache = torch.full(((1, seq_len, head_dim)), 0.1,  device="cuda",  dtype=torch.bfloat16)
    k_cache = torch.randn(1, seq_len, head_dim,  device="cuda",  dtype=torch.bfloat16) * 10.0

    # v_cache = torch.full(((1, seq_len, head_dim)), 0.1, device="cuda",  dtype=torch.bfloat16)
    v_cache = torch.randn(1, seq_len, head_dim, device="cuda",  dtype=torch.bfloat16) * 10.0

    k_cache_t = k_cache.clone()
    v_cache_t = v_cache.clone()

    # q = torch.randn(q_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    # q = torch.full((q_heads, head_dim), 0.3, device="cuda", dtype=torch.bfloat16)
    # k = torch.full((k_heads, head_dim), 0.4, device="cuda", dtype=torch.bfloat16)
    # # k = torch.randn(k_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    # v = torch.full((v_heads, head_dim), 0.8, device="cuda", dtype=torch.bfloat16)

    # q1 = torch.full((4, 2), 0.3, device="cuda", dtype=torch.bfloat16)
    # # q1 = torch.randn(4, 2, device="cuda", dtype=torch.bfloat16)

    # q2 = torch.full((4, 126), 0.3, device="cuda", dtype=torch.bfloat16)

    # k1 = torch.full((1, 4), 0.4, device="cuda", dtype=torch.bfloat16)
    # k2 = torch.full((1, 124), 0.4, device="cuda", dtype=torch.bfloat16)
    # k3 = torch.full((1, 111), 0.8, device="cuda", dtype=torch.bfloat16)


    # v1 = torch.full((1, 1), 0.8, device="cuda", dtype=torch.bfloat16)
    # v2 = torch.full((1, 127), 0.8, device="cuda", dtype=torch.bfloat16)

    # v = torch.cat([v1, v2], dim=1)
    # k = torch.cat([k1, k2], dim=1)
    # q = torch.cat([q1, q2], dim=1)

    # v = torch.randn(v_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    # qkv = torch.cat([q, k, v], dim=0)
    qkv = torch.randn(num_total_heads, head_dim, device="cuda",  dtype=torch.bfloat16) * 10.0
    # qkv = torch.full((num_total_heads, head_dim), 0.6, device="cuda",  dtype=torch.bfloat16)

    q = qkv[:q_heads].unsqueeze(1)
    k = qkv[q_heads:q_heads+1].unsqueeze(1)
    v = qkv[-1:].unsqueeze(1)
    k_cache = torch.cat([k_cache, k], dim=1)
    v_cache = torch.cat([v_cache, v], dim=1)

    torch_output = attention_decode(q, k_cache, v_cache)
    print(torch_output)
    torch_output = torch_output.squeeze(0).squeeze(1)


    mirage_ouput = torch.zeros(q_heads, 128, device="cuda", dtype=torch.bfloat16)
    runtime_kernel.single_batch_decoding(qkv, k_cache_t, v_cache_t, mirage_ouput, seq_len + 1)
    print(mirage_ouput)

    diff = torch_output - mirage_ouput
    abs_diff = diff.abs()
    abs_val = torch_output.abs()
    rel_error = abs_diff / (abs_val + 1e-6)
    print('max rel_error', rel_error.max())
