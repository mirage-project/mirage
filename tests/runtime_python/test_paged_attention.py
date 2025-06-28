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

device = "cuda"
dtype = torch.bfloat16


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
    inv_rms = torch.rsqrt(variance + eps)
    X_normed = X_fp32 * inv_rms
    out = X_normed * W_fp32
    return out.to(torch.bfloat16)


def torch_paged_attention(
    q,
    paged_k_cache,
    paged_v_cache,
    paged_kv_indices_buffer,
    seq_len,
    q_norm_weight,
    k_norm_weight,
    cos,
    sin,
    eps,
):
    num_pages = (seq_len + page_size - 1) // page_size
    page_indices = [paged_kv_indices_buffer[i] for i in range(num_pages)]
    k_cache = torch.cat(
        [paged_k_cache[page_idx] for page_idx in page_indices],
        dim=0,
    )
    v_cache = torch.cat(
        [paged_v_cache[page_idx] for page_idx in page_indices],
        dim=0,
    )

    norm_q = rmsnorm(q, q_norm_weight, eps)
    norm_k = rmsnorm(k_cache[seq_len - 1, :], k_norm_weight, eps)
    q_rot, k_rot = apply_rotary_pos_emb(norm_q, norm_k, cos, sin)

    q = q_rot.squeeze(1)
    k_cache[seq_len - 1, :] = k_rot
    paged_k_cache[page_indices[-1], (seq_len - 1) % page_size, :] = k_rot

    k = k_cache[:seq_len, :]
    v = v_cache[:seq_len, :]
    scores = torch.matmul(q, k.transpose(-2, -1))
    mask = torch.arange(seq_len, device=scores.device)[None, :] <= (seq_len - 1)
    scores = scores.masked_fill(~mask[None, None, :], float("-inf"))
    attn = F.softmax(scores / np.sqrt(head_dim), dim=-1)
    out = torch.matmul(attn, v)
    return out


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
paged_kv_indices_buffer = torch.arange(max_num_pages, device=device, dtype=torch.int32)

torch_paged_k_cache = paged_k_cache.clone()
torch_paged_v_cache = paged_v_cache.clone()

all_cos = torch.randn((513, head_dim), device=device, dtype=dtype)
all_sin = torch.randn((513, head_dim), device=device, dtype=dtype)

for seq_len in range(1, 513):
    qkv = torch.randn((qo_heads + 2 * kv_heads, head_dim), device=device, dtype=dtype)

    q = qkv[:qo_heads].unsqueeze(1)
    k = qkv[qo_heads : qo_heads + kv_heads]
    v = qkv[-kv_heads:]

    page_idx = paged_kv_indices_buffer[(seq_len - 1) // page_size]
    page_offset = (seq_len - 1) % page_size

    torch_paged_k_cache[page_idx, page_offset] = k
    torch_paged_v_cache[page_idx, page_offset] = v

    mirage_qkv = qkv.clone()

    q_norm_weight = torch.randn((1, head_dim), device=device, dtype=dtype)
    k_norm_weight = torch.randn((1, head_dim), device=device, dtype=dtype)

    torch_cos = all_cos[seq_len].unsqueeze(0)
    torch_sin = all_sin[seq_len].unsqueeze(0)

    eps = 1e-5

    mirage_output = torch.empty(qo_heads, head_dim, device=device, dtype=dtype)

    runtime_kernel.paged_attention(
        mirage_qkv,
        paged_k_cache,
        paged_v_cache,
        mirage_output,
        paged_kv_indices_buffer,
        seq_len,
        True,
        True,
        q_norm_weight,
        k_norm_weight,
        all_cos,
        all_sin,
        eps,
        eps,
    )

    torch_out = torch_paged_attention(
        q,
        torch_paged_k_cache,
        torch_paged_v_cache,
        paged_kv_indices_buffer,
        seq_len,
        q_norm_weight,
        k_norm_weight,
        torch_cos,
        torch_sin,
        eps,
    )
    ratio = mirage_output / torch_out.squeeze(0).squeeze(1)
    # randomly print ratio
    if torch.rand(1).item() < 0.05:
        print("Ratio (kernel / torch) for seq_len", seq_len, ":")
        print(ratio)
