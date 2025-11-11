import numpy as np
import torch
import torch.nn.functional as F
import runtime_kernel

torch.set_printoptions(sci_mode=False)
torch.set_printoptions(profile="full")
torch.set_printoptions(sci_mode=False)

qo_heads = 4
kv_heads = 1
head_dim = 128
page_size = 64
max_num_pages = 64
prompt_len = 8
max_tokens = 4

device = "cuda"
dtype = torch.bfloat16


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=0):
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
    inv_rms = torch.rsqrt(variance + eps)
    X_normed = X_fp32 * inv_rms
    out = X_normed * W_fp32
    return out.to(torch.bfloat16)


def torch_multitoken_paged_attention(
    q,
    paged_k_cache,
    paged_v_cache,
    paged_kv_indptr_buffer,
    paged_kv_indices_buffer,
    paged_kv_last_page_len_buffer,
    request_id,
    num_tokens,
    qk_norm,
    rope,
    q_norm_weight,
    k_norm_weight,
    cos,
    sin,
    eps=1e-6,
):
    first_page_pos = paged_kv_indptr_buffer[request_id]
    last_page_pos = paged_kv_indptr_buffer[request_id + 1]
    num_pages = last_page_pos - first_page_pos
    page_indices = [
        paged_kv_indices_buffer[i] for i in range(first_page_pos, last_page_pos)
    ]
    last_page_len = paged_kv_last_page_len_buffer[request_id]
    seq_len = (num_pages - 1) * page_size + last_page_len
    k_cache = torch.cat(
        [paged_k_cache[page_idx] for page_idx in page_indices],
        dim=0,
    )
    v_cache = torch.cat(
        [paged_v_cache[page_idx] for page_idx in page_indices],
        dim=0,
    )

    norm_q = torch.zeros_like(q)
    norm_k = torch.zeros_like(k_cache[-num_tokens:, :])
    for i in range(num_tokens):
        if qk_norm:
            norm_q[i, :] = rmsnorm(q[i, :], q_norm_weight, eps)
            norm_k[i, :] = rmsnorm(
                k_cache[seq_len - num_tokens + i, :], k_norm_weight, eps
            )
        else:
            norm_q[i, :] = q[i, :]
            norm_k[i, :] = k_cache[seq_len - num_tokens + i, :]
        
        if rope:
            norm_q[i, :], norm_k[i, :] = apply_rotary_pos_emb(
                norm_q[i, :], norm_k[i, :], cos[seq_len - num_tokens + i, :], sin[seq_len - num_tokens + i, :]
            )
    k_cache[seq_len - num_tokens : seq_len, :] = norm_k
    paged_k_cache[page_indices[-1], last_page_len - num_tokens : last_page_len, :] = (
        norm_k
    )
    norm_q = norm_q.view(num_tokens * qo_heads, head_dim)
    k = k_cache[:seq_len, :]
    v = v_cache[:seq_len, :]
    v = v.view(seq_len * kv_heads, head_dim)
    scores = torch.matmul(norm_q, k.transpose(-2, -1))
    #scores = scores.reshape(num_tokens * qo_heads, (num_tokens + prompt_len)* kv_heads)
    assert scores.shape==(num_tokens * qo_heads, seq_len * kv_heads)
    mask = torch.tril(torch.ones((num_tokens, num_tokens), device=device, dtype=dtype))
    mask = torch.cat((torch.ones((num_tokens, prompt_len), device=device, dtype=dtype), mask), dim=-1)
    mask = mask.repeat_interleave(qo_heads, dim=0).repeat_interleave(kv_heads, dim=1)

    T, Hq, Hk = num_tokens, qo_heads, kv_heads
    S = scores.shape[1] // Hk
    base = S - T                                    # start index of the last-T (new) tokens
    cols = torch.arange(S, device=device)

    # allow all cols < base, and within the last-T block allow up to current t
    mask = (cols[None, :] < base) | (cols[None, :] < (base + torch.arange(1, T+1, device=device)[:, None]))
    mask = mask.repeat_interleave(Hq, 0).repeat_interleave(Hk, 1)

    # print("scores.shape", scores.shape)
    # print("mask.shape", mask.shape)
    # print(mask)

    scores = scores.masked_fill(mask == 0, float("-inf"))
    # print(scores)
    attn = F.softmax(scores / np.sqrt(head_dim), dim=-1)
    output = torch.matmul(attn, v)
    return output


paged_k_cache = torch.randn(
    (max_num_pages, page_size, kv_heads * head_dim),
    device=device,
    dtype=dtype,
)
paged_v_cache = torch.randn(
    (max_num_pages, page_size, kv_heads * head_dim),
    device=device,
    dtype=dtype,
)


# paged_k_cache = torch.full(
#     (max_num_pages, page_size, kv_heads * head_dim), 0.1, 
#     device=device,
#     dtype=dtype,
# )
# paged_v_cache = torch.full(
#     (max_num_pages, page_size, kv_heads * head_dim),0.1, 
#     device=device,
#     dtype=dtype,
# )

paged_kv_indptr_buffer = torch.arange(
    max_num_pages + 1, device=device, dtype=torch.int32
)
qo_indptr_buffer = torch.tensor([0, max_tokens], device=device, dtype=torch.int32)
paged_kv_indptr_buffer = torch.tensor([0, 8], device=device, dtype=torch.int32)

paged_kv_indices_buffer = torch.arange(max_num_pages, device=device, dtype=torch.int32)
# paged_kv_last_page_len_buffer = torch.tensor(
#     [prompt_len + max_tokens], device=device, dtype=torch.int32
# )
paged_kv_last_page_len_buffer = torch.tensor([64], device=device, dtype=torch.int32)


torch_paged_k_cache = paged_k_cache.clone()
torch_paged_v_cache = paged_v_cache.clone()

all_cos = torch.randn((513, head_dim), device=device, dtype=dtype)
all_sin = torch.randn((513, head_dim), device=device, dtype=dtype)

qkv = torch.randn(
    (max_tokens, (qo_heads + 2 * kv_heads) * head_dim), device=device, dtype=dtype
)

# qkv = torch.full(
#     ((max_tokens, (qo_heads + 2 * kv_heads) * head_dim)),0.8, device=device, dtype=dtype
# )



q = qkv[:max_tokens, : qo_heads * head_dim]
q = q.view(max_tokens, qo_heads, head_dim)
k = qkv[:max_tokens, qo_heads * head_dim : qo_heads * head_dim + kv_heads * head_dim]
v = qkv[:max_tokens, qo_heads * head_dim + kv_heads * head_dim :]

# page_idx = paged_kv_indices_buffer[0]

# page_offset = prompt_len
# print("page_idx", page_idx)
# assert prompt_len < page_size, "Assume prompt can fit in a single page for now"

# torch_paged_k_cache[page_idx, page_offset : page_offset + max_tokens] = k
# torch_paged_v_cache[page_idx, page_offset : page_offset + max_tokens] = v

first = int(paged_kv_indptr_buffer[0].item())
last  = int(paged_kv_indptr_buffer[1].item())      # exclusive
last_page_global_idx = last - 1

page_idx = int(paged_kv_indices_buffer[last_page_global_idx].item())
last_page_len = int(paged_kv_last_page_len_buffer[0].item())

# place the T new tokens at the tail of the sequence:
page_offset = last_page_len - max_tokens            # local start inside last page

print("page_idx", page_idx)
print("page_offset", page_offset)

# write K/V
torch_paged_k_cache[page_idx, page_offset:page_offset+max_tokens] = k
torch_paged_v_cache[page_idx, page_offset:page_offset+max_tokens] = v

mirage_qkv = qkv.clone()

q_norm_weight = torch.randn((1, head_dim), device=device, dtype=dtype)
k_norm_weight = torch.randn((1, head_dim), device=device, dtype=dtype)

# torch_cos = all_cos[0 : max_tokens + prompt_len + 1, :]
# torch_sin = all_sin[0 : max_tokens + prompt_len + 1, :]
torch_cos = all_cos
torch_sin = all_sin

eps = 1e-5

mirage_output = torch.empty(max_tokens * qo_heads, head_dim, device=device, dtype=dtype)

runtime_kernel.multitoken_paged_attention(
    mirage_qkv,
    paged_k_cache,
    paged_v_cache,
    mirage_output,
    qo_indptr_buffer,
    paged_kv_indptr_buffer,
    paged_kv_indices_buffer,
    paged_kv_last_page_len_buffer,
    0,
    False,
    False,
    q_norm_weight,
    k_norm_weight,
    all_cos,
    all_sin,
    eps,
    eps
)

torch_out = torch_multitoken_paged_attention(
    q,
    torch_paged_k_cache,
    torch_paged_v_cache,
    paged_kv_indptr_buffer,
    paged_kv_indices_buffer,
    paged_kv_last_page_len_buffer,
    0,
    max_tokens,
    False,
    False,
    q_norm_weight,
    k_norm_weight,
    torch_cos,
    torch_sin,
    eps=eps,
)
print("Ratio (Mirage / Torch):")
print(mirage_output / torch_out)

