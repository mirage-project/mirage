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
max_seq_len = 512

device = "cuda"
dtype = torch.bfloat16

def attention_decode(q, k_cache, v_cache, valid_len):
    k = k_cache[:, :valid_len, :].expand(q_heads, -1, -1)
    v = v_cache[:, :valid_len, :].expand(q_heads, -1, -1)
    scores = torch.matmul(q, k.transpose(-2, -1))
    mask = torch.arange(valid_len, device=scores.device)[None, :] <= (valid_len - 1)
    scores = scores.masked_fill(~mask[None, None, :], float("-inf"))
    
    out = torch.matmul(scores, v)
    return out

k_cache_torch = torch.empty((1, max_seq_len, head_dim), device=device, dtype=dtype)
v_cache_torch = torch.empty((1, max_seq_len, head_dim), device=device, dtype=dtype)
k_cache_mirage = torch.empty((max_seq_len, head_dim), device=device, dtype=dtype)
v_cache_mirage =torch.empty((max_seq_len, head_dim), device=device, dtype=dtype)

for i in range(512):
    seq_len = i + 1

    qkv = torch.randn(num_total_heads, head_dim, device=device, dtype=dtype)
    q = qkv[:q_heads].unsqueeze(1)
    k = qkv[q_heads:q_heads+1]
    v = qkv[-1:]

    k_cache_torch[0, seq_len - 1] = k
    v_cache_torch[0, seq_len - 1] = v

    k_cache_mirage[seq_len - 1] = k
    v_cache_mirage[seq_len - 1] = v

    torch_output = attention_decode(q, k_cache_torch, v_cache_torch, seq_len)
    torch_output = torch_output.squeeze(0).squeeze(1)

    # print('k_cache_mirage', v_cache_mirage)

    mirage_output = torch.empty((q_heads, head_dim), device=device, dtype=dtype)
    runtime_kernel.single_batch_gqa(qkv, k_cache_mirage, v_cache_mirage, mirage_output, seq_len)
    torch.cuda.synchronize()
    
    # print('mirage_output', v_cache_mirage)

    diff = torch_output - mirage_output
    print("seq_len:", seq_len, "min:", diff.min().item(), "max:", diff.max().item())