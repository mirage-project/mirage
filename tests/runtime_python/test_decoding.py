import torch
import torch.nn.functional as F
import runtime_kernel
torch.set_printoptions(sci_mode=False)

q_heads = 7
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
    
    print(attn.shape)
    print(v.shape)
    out = torch.matmul(attn, v) 
    return out

k_cache = torch.full(((1, 63, head_dim)), 0.1,  device="cuda",  dtype=torch.bfloat16)
v_cache = torch.full(((1, 63, head_dim)), 0.1, device="cuda",  dtype=torch.bfloat16)
qkv = torch.full((num_total_heads, head_dim), 0.1, device="cuda",  dtype=torch.bfloat16)

q = qkv[:q_heads].unsqueeze(1)
k = qkv[q_heads:q_heads+1].unsqueeze(1)
v = qkv[-1:].unsqueeze(1)
k_cache = torch.cat([k_cache, k], dim=1)
v_cache = torch.cat([v_cache, v], dim=1)

torch_output = attention_decode(q, k_cache, v_cache)
# print(torch_output)
print(torch_output.shape)

# mirage_ouput = torch.empty(7, 128, device="cuda", dtype=torch.bfloat16)
# runtime_kernel.single_batch_decoding(qkv, k_cache, v_cache, 64)
# print(mirage_ouput)


