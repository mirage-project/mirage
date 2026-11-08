"""Check outer-warp attention merging against FP64 attention.

Build with ``python setup.py build_ext --inplace``, then run this script on
SM100. Uses the existing direct launcher, including cached-prefix decoding
and windows that leave some warps without valid keys.
"""

import torch

import runtime_kernel_paged_attention as rk


def check(seq_len, window, scale, seed):
    torch.manual_seed(seed)
    tokens, heads, dim, page_size = 8, 8, 64, 64
    dtype, device = torch.bfloat16, "cuda"
    page_count = (seq_len + page_size - 1) // page_size
    pages = torch.tensor([5, 2, 7, 1][:page_count], device=device)
    qkv = torch.randn(tokens, (heads + 2) * dim, dtype=dtype, device=device)
    qkv[:, : (heads + 1) * dim] *= scale
    keys = torch.randn(8, page_size, 1, dim, dtype=dtype, device=device)
    values = torch.randn_like(keys)
    keys_before, values_before, qkv_before = keys.clone(), values.clone(), qkv.clone()
    backing = torch.full((tokens * heads * dim + 256,), 1.25,
                         dtype=dtype, device=device)
    out = backing[128:-128].view(tokens, heads * dim)
    qo = torch.tensor([0, tokens], dtype=torch.int32, device=device)
    kv = torch.tensor([0, page_count], dtype=torch.int32, device=device)
    last = torch.tensor([seq_len - (page_count - 1) * page_size],
                        dtype=torch.int32, device=device)
    norm = torch.ones(dim, dtype=dtype, device=device)
    cos = torch.ones(256, dim, dtype=dtype, device=device)
    sin = torch.zeros_like(cos)
    args = (qkv, keys, values, out, qo, kv, pages.to(torch.int32), last,
            norm, norm, cos, sin, window)

    prefix = seq_len - tokens
    new_k = qkv[:, heads * dim : (heads + 1) * dim]
    new_v = qkv[:, (heads + 1) * dim :]
    expected_k, expected_v = keys_before.clone(), values_before.clone()
    for row in range(tokens):
        pos = prefix + row
        expected_k[pages[pos // page_size], pos % page_size, 0] = new_k[row]
        expected_v[pages[pos // page_size], pos % page_size, 0] = new_v[row]
    positions = torch.arange(seq_len, device=device)
    physical = pages[positions // page_size]
    k = expected_k[physical, positions % page_size, 0].double()
    v = expected_v[physical, positions % page_size, 0].double()
    q = qkv[:, : heads * dim].view(tokens, heads, dim).double()
    scores = torch.einsum("thd,sd->ths", q, k) / dim**0.5
    query_pos = torch.arange(prefix, seq_len, device=device)
    keep = positions[None, :] <= query_pos[:, None]
    if window:
        keep &= positions[None, :] > query_pos[:, None] - window
    scores.masked_fill_(~keep[:, None, :], -torch.inf)
    ref = torch.einsum("ths,sd->thd", scores.softmax(-1), v)

    for _ in range(2):
        rk.paged_attention_sm100(*args)
        torch.cuda.synchronize()
        actual = out.view(tokens, heads, dim).double()
        error = actual - ref
        rrms = error.square().mean(-1).sqrt() / ref.square().mean(-1).sqrt().clamp_min(1e-8)
        assert torch.isfinite(actual).all(), (seq_len, window, scale)
        assert error.abs().max() <= 0.05, (seq_len, window, scale, error.abs().max())
        assert rrms.max() <= 0.01, (seq_len, window, scale, rrms.max())
        assert torch.equal(keys, expected_k)
        assert torch.equal(values, expected_v)
        assert torch.equal(qkv, qkv_before)
        assert (backing[:128] == 1.25).all() and (backing[-128:] == 1.25).all()
    return error.abs().max().item(), rrms.max().item()


if __name__ == "__main__":
    assert torch.cuda.get_device_capability() == (10, 0), "SM100 required"
    errors = [check(length, window, scale, 1108 + i)
              for i, length in enumerate((64, 65, 128, 200, 256))
              for window in (0, 32, 96)
              for scale in (0.5, 1.0, 4.0)]
    print(f"Passed {len(errors)} cases, two launches each; "
          f"max abs={max(x[0] for x in errors):.6g}, "
          f"max per-head RRMS={max(x[1] for x in errors):.6g}")
