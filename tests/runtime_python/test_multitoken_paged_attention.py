import numpy as np
import torch
import torch.nn.functional as F
import runtime_kernel

torch.set_printoptions(sci_mode=False)

qo_heads = 4
kv_heads = 1
head_dim = 128
page_size = 64
max_num_pages = 64
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
        norm_q[i, :] = rmsnorm(q[i, :], q_norm_weight, eps)
        norm_k[i, :] = rmsnorm(
            k_cache[last_page_len - max_tokens + i, :], k_norm_weight, eps
        )
        norm_q[i, :], norm_k[i, :] = apply_rotary_pos_emb(
            norm_q[i, :], norm_k[i, :], cos[i, :], sin[i, :]
        )
    k_cache[last_page_len - max_tokens : last_page_len, :] = norm_k
    paged_k_cache[page_indices[-1], last_page_len - num_tokens : last_page_len, :] = (
        norm_k
    )
    norm_q = norm_q.view(num_tokens * qo_heads, head_dim)
    k = k_cache[:num_tokens, :]
    v = v_cache[:num_tokens, :]
    v = v.view(num_tokens * kv_heads, head_dim)
    scores = torch.matmul(norm_q, k.transpose(-2, -1))
    scores = scores.reshape(num_tokens * qo_heads, num_tokens * kv_heads)
    mask = torch.tril(torch.ones((seq_len, seq_len), device=device, dtype=dtype))
    mask = mask.repeat_interleave(qo_heads, dim=0).repeat_interleave(kv_heads, dim=1)

    scores = scores.masked_fill(mask == 0, float("-inf"))
    attn = F.softmax(scores / np.sqrt(head_dim), dim=-1)
    output = torch.matmul(attn, v)
    return output


paged_k_cache = torch.empty(
    (max_num_pages, page_size, kv_heads * head_dim),
    device=device,
    dtype=dtype,
)
paged_v_cache = torch.empty(
    (max_num_pages, page_size, kv_heads * head_dim),
    device=device,
    dtype=dtype,
)
paged_kv_indptr_buffer = torch.arange(
    max_num_pages + 1, device=device, dtype=torch.int32
)
qo_indptr_buffer = torch.tensor([0, 4], device=device, dtype=torch.int32)
paged_kv_indptr_buffer = torch.tensor([0, 1], device=device, dtype=torch.int32)
paged_kv_indices_buffer = torch.arange(max_num_pages, device=device, dtype=torch.int32)
paged_kv_last_page_len_buffer = torch.tensor(
    [max_tokens], device=device, dtype=torch.int32
)

torch_paged_k_cache = paged_k_cache.clone()
torch_paged_v_cache = paged_v_cache.clone()

all_cos = torch.randn((513, head_dim), device=device, dtype=dtype)
all_sin = torch.randn((513, head_dim), device=device, dtype=dtype)

qkv = torch.randn(
    (max_tokens, (qo_heads + 2 * kv_heads) * head_dim), device=device, dtype=dtype
)

q = qkv[:max_tokens, : qo_heads * head_dim]
q = q.view(max_tokens, qo_heads, head_dim)
k = qkv[:max_tokens, qo_heads * head_dim : qo_heads * head_dim + kv_heads * head_dim]
v = qkv[:max_tokens, qo_heads * head_dim + kv_heads * head_dim :]

page_idx = paged_kv_indices_buffer[0]
page_offset = 0

torch_paged_k_cache[page_idx, page_offset : page_offset + max_tokens] = k
torch_paged_v_cache[page_idx, page_offset : page_offset + max_tokens] = v

mirage_qkv = qkv.clone()

q_norm_weight = torch.randn((1, head_dim), device=device, dtype=dtype)
k_norm_weight = torch.randn((1, head_dim), device=device, dtype=dtype)

torch_cos = all_cos[1 : max_tokens + 1, :]
torch_sin = all_sin[1 : max_tokens + 1, :]

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
    True,
    True,
    q_norm_weight,
    k_norm_weight,
    all_cos,
    all_sin,
    eps,
    eps,
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
    q_norm_weight,
    k_norm_weight,
    torch_cos,
    torch_sin,
    eps=eps,
)
print("Ratio (Mirage / Torch):")
print(mirage_output / torch_out)
