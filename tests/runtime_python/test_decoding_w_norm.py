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
    # variance = X.pow(2).mean(-1, keepdim=True)
    # X = X * torch.rsqrt(variance + eps)
    # X = torch.mul(X, W)
    # return X
    X_fp32 = X.to(torch.float32)
    W_fp32 = W.to(torch.float32)

    variance = X_fp32.pow(2).mean(-1, keepdim=True)
    inv_rms = torch.rsqrt(variance + eps)
    X_normed = X_fp32 * inv_rms
    out = X_normed * W_fp32
    return out.to(torch.bfloat16)


def attention_decode(q, k_cache, v_cache, valid_len, q_weight, k_weight, eps, cos, sin):

    qnorm = rmsnorm(q, q_weight, eps)
    knorm = rmsnorm(k_cache[:, valid_len - 1, :], k_weight, eps)

    q_rot, k_rot = apply_rotary_pos_emb(qnorm, knorm, cos, sin)
    q = q_rot
    k_cache[:, valid_len - 1, :] = k_rot

    # k_cache[:, valid_len - 1, :] = rmsnorm(k_cache[:, valid_len - 1, :], k_weight, eps)
    k = k_cache[:, :valid_len, :].expand(q_heads, -1, -1)

    v = v_cache[:, :valid_len, :].expand(q_heads, -1, -1)
    scores = torch.matmul(q, k.transpose(-2, -1))
    mask = torch.arange(valid_len, device=scores.device)[None, :] <= (valid_len - 1)
    scores = scores.masked_fill(~mask[None, None, :], float("-inf"))

    attn = F.softmax(scores / np.sqrt(head_dim), dim=-1)
    out = torch.matmul(attn, v)
    return out


k_cache_torch = torch.empty((1, max_seq_len, head_dim), device=device, dtype=dtype)
v_cache_torch = torch.empty((1, max_seq_len, head_dim), device=device, dtype=dtype)
k_cache_mirage = torch.empty((max_seq_len, head_dim), device=device, dtype=dtype)
v_cache_mirage = torch.empty((max_seq_len, head_dim), device=device, dtype=dtype)

all_cos = torch.randn((513, head_dim), device=device, dtype=dtype)
all_sin = torch.randn((513, head_dim), device=device, dtype=dtype)

for i in range(512):
    seq_len = i + 1

    qkv = torch.randn(num_total_heads, head_dim, device=device, dtype=dtype)
    # qkv = torch.full((num_total_heads, head_dim), 0.1, device=device, dtype=dtype)

    q = qkv[:q_heads].unsqueeze(1)
    k = qkv[q_heads : q_heads + 1]
    v = qkv[-1:]

    k_cache_torch[0, seq_len - 1] = k
    v_cache_torch[0, seq_len - 1] = v

    k_cache_mirage[seq_len - 1] = k.clone()
    v_cache_mirage[seq_len - 1] = v.clone()

    qkv_mirage = qkv.clone()

    eps = 1e-5
    qnorm_weight = torch.randn((1, head_dim), device=device, dtype=dtype)
    knorm_weight = torch.randn((1, head_dim), device=device, dtype=dtype)

    cos = all_cos[seq_len].unsqueeze(0).clone()
    sin = all_sin[seq_len].unsqueeze(0).clone()

    torch_output = attention_decode(
        q,
        k_cache_torch,
        v_cache_torch,
        seq_len,
        q_weight=qnorm_weight,
        k_weight=knorm_weight,
        eps=eps,
        cos=cos,
        sin=sin,
    )
    torch_output = torch_output.squeeze(0).squeeze(1)

    mirage_output = torch.empty((q_heads, head_dim), device=device, dtype=dtype)
    # runtime_kernel.single_batch_gqa(qkv_mirage, k_cache_mirage, v_cache_mirage, mirage_output, seq_len, False)
    runtime_kernel.single_batch_decoding(
        qkv_mirage,
        k_cache_mirage,
        v_cache_mirage,
        mirage_output,
        seq_len,
        True,
        True,
        qnorm_weight,
        knorm_weight,
        all_cos,
        all_sin,
        eps,
        eps,
    )
    print("torch_output / mirage_output:")
    print(torch_output / mirage_output)
    diff = mirage_output - torch_output
    print("seq_len res:", seq_len, "min:", diff.min().item(), "max:", diff.max().item())
