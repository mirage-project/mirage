"""PyTorch reference implementations for mHC kernels K1-K5 and the
hc_pre / hc_post block-level operators."""
import torch
import torch.nn.functional as F


def sinkhorn_knopp_torch(x, repeat=20, eps=1e-9):
    """K3 reference: matches the existing sm100_sinkhorn test util."""
    comb = torch.softmax(x.float(), dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(1, repeat):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return comb


def k1_reference(x_flat, hc_fn, eps):
    """K1 reference (reordered formulation):
        x_normalized = x_flat * rsqrt(mean(x_flat**2) + eps)
        mixes = linear(x_normalized, hc_fn)

    x_flat: [num_tokens, nC] fp32
    hc_fn:  [mix_hc, nC] fp32 (or bf16; cast to fp32 for the math)
    Returns mixes [num_tokens, mix_hc] fp32, equal to F.linear(x_flat) * rsqrt.
    """
    x_flat = x_flat.float()
    rsqrt = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + eps)
    return F.linear(x_flat * rsqrt, hc_fn.float())


def k2_reference(mixes, scale, base, n):
    """K2 reference: affine + split + sigmoid/2sigmoid/identity.

    mixes: [num_tokens, n*n + 2n] fp32 or bf16
    scale: [3] fp32
    base:  [mix_hc] fp32
    """
    mix_hc = n * n + 2 * n
    assert mixes.shape[-1] == mix_hc
    mixes_f = mixes.float()
    pre_logits = mixes_f[..., :n] * scale[0] + base[:n]
    post_logits = mixes_f[..., n : 2 * n] * scale[1] + base[n : 2 * n]
    res_logits = mixes_f[..., 2 * n :] * scale[2] + base[2 * n :]
    h_pre = torch.sigmoid(pre_logits)
    h_post = 2.0 * torch.sigmoid(post_logits)
    return h_pre, h_post, res_logits


def k4_reference(h_pre, x):
    """K4 reference: F_pre[c] = sum_i H_pre[i] * x[i, c]

    h_pre: [num_tokens, n] fp32
    x:     [num_tokens, n, c] bf16
    Returns f_pre [num_tokens, c] bf16.
    """
    return torch.einsum("tn,tnc->tc", h_pre, x.float()).to(x.dtype)


def k5_reference(residual, x, comb, post):
    """K5 reference:
        y[k, c] = post[k] * x[c] + sum_i comb[k, i] * residual[i, c]

    residual: [num_tokens, n, c] bf16
    x:        [num_tokens, c] bf16
    comb:     [num_tokens, n, n] fp32
    post:     [num_tokens, n] fp32
    Returns y [num_tokens, n, c] bf16.
    """
    outer = post.unsqueeze(-1) * x.unsqueeze(-2).float()  # [t, n, c]
    mix = torch.einsum("tki,tic->tkc", comb, residual.float())  # [t, n, c]
    return (outer + mix).to(x.dtype)


def hc_pre_reference(x, hc_fn, hc_scale, hc_base, n, sinkhorn_iters, hc_eps,
                     norm_eps):
    """Block-level hc_pre reference matching the user's PyTorch Block impl.

    x: [b, s, n, C] (fp32 or bf16)
    Returns: f_pre [b, s, C], h_post [b, s, n], comb [b, s, n, n]
    """
    b, s, n_chk, C = x.shape
    assert n_chk == n
    dtype = x.dtype
    x_flat = x.flatten(2)  # [b, s, nC]
    bs = b * s
    x_flat_2d = x_flat.reshape(bs, n * C)

    mixes = k1_reference(x_flat_2d, hc_fn, norm_eps)
    h_pre, h_post, res_logits = k2_reference(mixes, hc_scale, hc_base, n)
    res_mat = res_logits.reshape(bs, n, n)
    comb = sinkhorn_knopp_torch(res_mat, repeat=sinkhorn_iters, eps=hc_eps)
    f_pre = k4_reference(h_pre, x.reshape(bs, n, C).to(dtype)).reshape(b, s, C)
    return f_pre, h_post.reshape(b, s, n), comb.reshape(b, s, n, n)


def hc_post_reference(x, residual, post, comb):
    """Block-level hc_post reference.

    x:        [b, s, C] bf16
    residual: [b, s, n, C] bf16
    post:     [b, s, n] fp32
    comb:     [b, s, n, n] fp32
    Returns:  [b, s, n, C] bf16
    """
    b, s, n, C = residual.shape
    return k5_reference(
        residual.reshape(b * s, n, C),
        x.reshape(b * s, C),
        comb.reshape(b * s, n, n),
        post.reshape(b * s, n),
    ).reshape(b, s, n, C)
