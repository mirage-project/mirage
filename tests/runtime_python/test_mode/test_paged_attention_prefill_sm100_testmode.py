"""
Correctness test for the SM100 wide-Q paged attention prefill kernel.

Mirrors Qwen3-8B shape (32 Q heads, 8 KV heads, head_dim=128, gqa=4).
With ``max_tokens=32`` the per-task Q tile is 128 packed rows — matches
FlashInfer's P_Q_TILE_SIZE.

QKV tensor layout (matches Qwen3 builder's ``shuffle_tensors`` output):
  per token, interleaved by KV head group:
    [Q[0..gqa-1], K[kv=0], V[kv=0], Q[gqa..2gqa-1], K[kv=1], V[kv=1], ...]
"""

import os
import sys

import numpy as np
import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel


def build_qkv_interleaved(q_clean, k_clean, v_clean):
    """Pack (T, Hq, D), (T, Hkv, D), (T, Hkv, D) into the interleaved
    KV-head-grouped layout the kernel expects.

    Output shape: (T, (Hq + 2*Hkv) * D).
    """
    T, Hq, D = q_clean.shape
    Hkv = k_clean.shape[1]
    assert v_clean.shape == k_clean.shape
    assert Hq % Hkv == 0
    gqa = Hq // Hkv
    qkv = torch.empty(T, Hkv, (gqa + 2) * D, dtype=q_clean.dtype, device=q_clean.device)
    q_grouped = q_clean.view(T, Hkv, gqa, D)
    qkv[:, :, 0 : gqa * D] = q_grouped.reshape(T, Hkv, gqa * D)
    qkv[:, :, gqa * D : (gqa + 1) * D] = k_clean
    qkv[:, :, (gqa + 1) * D : (gqa + 2) * D] = v_clean
    return qkv.reshape(T, Hkv * (gqa + 2) * D)


def torch_attention_ref(q, k_hist, v_hist, k_new, v_new):
    """Causal SDPA. q: (T, Hq, D), k_hist/v_hist: (S_hist, Hkv, D), k_new/v_new: (T, Hkv, D)."""
    T, Hq, D = q.shape
    Hkv = k_hist.shape[1]
    gqa = Hq // Hkv
    S_hist = k_hist.shape[0]
    seq_len = S_hist + T

    k_full = torch.cat([k_hist, k_new], dim=0)  # (seq_len, Hkv, D)
    v_full = torch.cat([v_hist, v_new], dim=0)

    sm_scale = 1.0 / np.sqrt(D)
    out = torch.zeros(T, Hq, D, dtype=torch.float32, device=q.device)
    for h_kv in range(Hkv):
        k_h = k_full[:, h_kv, :].to(torch.float32)
        v_h = v_full[:, h_kv, :].to(torch.float32)
        for g in range(gqa):
            h_q = h_kv * gqa + g
            q_h = q[:, h_q, :].to(torch.float32)
            scores = q_h @ k_h.transpose(0, 1) * sm_scale  # (T, seq_len)
            cols = torch.arange(seq_len, device=q.device)
            abs_pos = torch.arange(S_hist, seq_len, device=q.device)[:, None]
            mask = cols[None, :] <= abs_pos
            scores = scores.masked_fill(~mask, float("-inf"))
            attn = torch.softmax(scores, dim=-1)
            out[:, h_q, :] = attn @ v_h

    return out.to(q.dtype).reshape(T, Hq * D)


def test_paged_attention_prefill_sm100_testmode():
    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.bfloat16

    # Qwen3-8B single-GPU shape
    num_q_heads = 32
    num_kv_heads = 8
    head_dim = 128
    gqa = num_q_heads // num_kv_heads  # 4
    page_size = 64
    max_num_pages = 64
    max_seq_length = 256
    max_num_batched_requests = 1

    max_tokens = 32   # per-task Q tile (32*4 = 128 packed Q rows)
    num_tokens = 32
    max_num_batched_tokens = max_tokens
    history_len = 2 * page_size  # 128
    seq_len = history_len + num_tokens
    num_pages_used = (seq_len + page_size - 1) // page_size
    last_pl = seq_len - (num_pages_used - 1) * page_size

    # ---- meta tensors --------------------------------------------------------
    qo_indptr = torch.tensor([0, num_tokens], device=device, dtype=torch.int32)
    paged_kv_indptr = torch.tensor([0, num_pages_used], device=device, dtype=torch.int32)
    paged_kv_indices = torch.arange(num_pages_used, device=device, dtype=torch.int32)
    paged_kv_last_page_len = torch.tensor([last_pl], device=device, dtype=torch.int32)

    tokens = torch.zeros(1, max_seq_length, dtype=torch.int64, device=device)
    input_tokens = torch.zeros(max_num_batched_tokens, dtype=torch.int64, device=device)
    output_tokens = torch.zeros(max_num_batched_tokens, dtype=torch.int64, device=device)
    new_token_nums = torch.zeros(max_num_batched_requests, dtype=torch.int32, device=device)
    step = torch.zeros(1, dtype=torch.int32, device=device)
    prompt_length = torch.tensor([seq_len], dtype=torch.int32, device=device)

    # ---- clean Q/K/V (no rope, no qk_norm) ----------------------------------
    q_new = torch.randn(num_tokens, num_q_heads, head_dim, dtype=dtype, device=device)
    k_new = torch.randn(num_tokens, num_kv_heads, head_dim, dtype=dtype, device=device)
    v_new = torch.randn(num_tokens, num_kv_heads, head_dim, dtype=dtype, device=device)

    # Cache holds the history (positions 0..history_len-1)
    k_hist = torch.randn(history_len, num_kv_heads, head_dim, dtype=dtype, device=device)
    v_hist = torch.randn(history_len, num_kv_heads, head_dim, dtype=dtype, device=device)

    paged_k = torch.zeros(max_num_pages, page_size, num_kv_heads, head_dim,
                          dtype=dtype, device=device)
    paged_v = torch.zeros(max_num_pages, page_size, num_kv_heads, head_dim,
                          dtype=dtype, device=device)
    # Populate the history pages (page 0 = positions 0..63, page 1 = 64..127)
    paged_k[0] = k_hist[0:page_size]
    paged_k[1] = k_hist[page_size:2 * page_size]
    paged_v[0] = v_hist[0:page_size]
    paged_v[1] = v_hist[page_size:2 * page_size]
    # Page 2 will be filled by the kernel with the new tokens.

    # Build interleaved QKV (matches Qwen3 builder's shuffle_tensors layout)
    qkv = build_qkv_interleaved(q_new, k_new, v_new)
    # Pad to max_num_batched_tokens
    fused_outdim = (num_q_heads + 2 * num_kv_heads) * head_dim
    qkv_padded = torch.zeros(max_num_batched_tokens, fused_outdim, dtype=dtype, device=device)
    qkv_padded[:num_tokens] = qkv
    output = torch.zeros(max_num_batched_tokens, num_q_heads * head_dim,
                         dtype=dtype, device=device)

    # dummy norm / rope tensors (kernel will ignore them since flags=0)
    q_norm = torch.ones(head_dim, dtype=dtype, device=device)
    k_norm = torch.ones(head_dim, dtype=dtype, device=device)
    cos_pos = torch.zeros(max_seq_length, head_dim, dtype=dtype, device=device)
    sin_pos = torch.zeros(max_seq_length, head_dim, dtype=dtype, device=device)

    # ---- build PersistentKernel ---------------------------------------------
    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["num_workers"] = num_workers
    params["num_local_schedulers"] = num_schedulers
    params["mpi_rank"] = 0
    params["world_size"] = 1
    params["max_seq_length"] = max_seq_length
    params["max_num_batched_requests"] = max_num_batched_requests
    params["max_num_batched_tokens"] = max_num_batched_tokens
    params["max_num_pages"] = max_num_pages
    params["page_size"] = page_size
    params["meta_tensors"] = {
        "step": step,
        "tokens": tokens,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "new_token_nums": new_token_nums,
        "prompt_length": prompt_length,
        "qo_indptr_buffer": qo_indptr,
        "paged_kv_indptr_buffer": paged_kv_indptr,
        "paged_kv_indices_buffer": paged_kv_indices,
        "paged_kv_last_page_len_buffer": paged_kv_last_page_len,
    }
    pk = PersistentKernel(**params)

    qkv_dt = pk.attach_input(qkv_padded, name="qkv")
    k_cache_dt = pk.attach_input(paged_k, name="paged_k")
    v_cache_dt = pk.attach_input(paged_v, name="paged_v")
    q_norm_dt = pk.attach_input(q_norm, name="q_norm")
    k_norm_dt = pk.attach_input(k_norm, name="k_norm")
    cos_dt = pk.attach_input(cos_pos, name="cos")
    sin_dt = pk.attach_input(sin_pos, name="sin")
    out_dt = pk.attach_input(output, name="out")

    pk.paged_attention_prefill_layer(
        input=qkv_dt,
        k_cache=k_cache_dt,
        v_cache=v_cache_dt,
        q_norm=q_norm_dt,
        k_norm=k_norm_dt,
        cos_pos_embed=None,
        sin_pos_embed=None,
        output=out_dt,
        grid_dim=(max_num_batched_requests, num_kv_heads, 1),
        block_dim=(128, 1, 1),
        max_tokens=max_tokens,
    )

    folder = os.path.dirname(__file__)
    print("Compiling test kernel...")
    pk.compile(output_dir=folder)
    print("Running...")
    pk()
    torch.cuda.synchronize()

    # ---- compute reference --------------------------------------------------
    ref = torch_attention_ref(q_new, k_hist, v_hist, k_new, v_new)
    actual = output[:num_tokens]
    diff = (actual.to(torch.float32) - ref.to(torch.float32)).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    print(f"max_diff = {max_diff:.4f}, mean_diff = {mean_diff:.6f}")
    per_token_max = diff.max(dim=1).values
    for i in range(num_tokens):
        print(f"  token {i:2d}: max_diff = {per_token_max[i].item():.4f}")

    pk.finalize()

    # bf16 attention accumulates rounding error over the KV iteration; the
    # kernel writes softmaxed scores in bf16 for the V mma. Mean-abs is the
    # meaningful metric.
    if max_diff < 0.3 and mean_diff < 0.02:
        print("PASSED")
    else:
        print("FAILED")
        sys.exit(1)


if __name__ == "__main__":
    test_paged_attention_prefill_sm100_testmode()
