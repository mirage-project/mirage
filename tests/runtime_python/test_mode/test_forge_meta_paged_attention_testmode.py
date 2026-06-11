"""Test mode with forged runtime state: exercise the runtime-state-dependent
paged-attention kernel (multitoken_paged_attention_sm100) over a variety of
ARBITRARY forged paged-KV layouts and compare against a PyTorch reference.

Why this test exists
--------------------
`multitoken_paged_attention` reads its sequence layout from meta tensors in
`runtime_config` (qo_indptr_buffer, paged_kv_indptr_buffer,
paged_kv_indices_buffer, paged_kv_last_page_len_buffer) rather than only from
its input/output tensors. Test mode (MPK_TEST_MODE) runs the task graph exactly
once and skips init_request_resources() + prepare_next_batch(), so the
user-supplied meta tensors reach the kernel VERBATIM — they are not reset or
recomputed. That lets a test forge an arbitrary runtime status (decode at an
arbitrary step, multi-page sequences, scattered physical pages, GQA) and check
the kernel against a reference.

Each case forges a single-token (decode) attention over a `seq_len`-long
sequence laid out across the given physical `page_indices`, with the history
pre-filled into the paged KV cache and the new token's K/V supplied via the
packed QKV input (the kernel appends it at position seq_len-1, then attends
causally over all seq_len positions). rope is made identity (cos=1,sin=0) and
qk-norm disabled, so the reference is a plain scaled-dot-product attention.

Run on a Blackwell (SM100) GPU node via test-on-gpu.
"""

import math
import torch
import mirage
from mirage.mpk.persistent_kernel import PersistentKernel

HEAD_DIM = 128  # the SM100 kernel is specialized for head_dim=128


def run_case(label, num_q_heads, num_kv_heads, page_size, max_num_pages,
             seq_len, page_indices):
    """Forge a single-token decode over `seq_len` positions laid out across
    `page_indices` (physical page ids), run it in test mode, and compare to a
    PyTorch attention reference. Returns max abs error."""
    assert num_kv_heads == 1, "test layout assumes a single KV head"
    device, dtype = "cuda", torch.bfloat16
    torch.manual_seed(1234)

    num_pages = (seq_len + page_size - 1) // page_size
    assert len(page_indices) == num_pages, (
        f"{label}: need {num_pages} page ids for seq_len={seq_len}, "
        f"page_size={page_size}; got {page_indices}")
    assert max(page_indices) < max_num_pages
    last_page_len = seq_len - (num_pages - 1) * page_size
    max_seq_len = ((seq_len + 127) // 128) * 128  # round up for cos/sin sizing

    # Logical K/V for every position 0..seq_len-1. Positions 0..seq_len-2 are
    # "history" pre-filled into the cache; position seq_len-1 is the new token,
    # supplied via the QKV input (the kernel writes it into the cache).
    k_logical = (torch.randn(seq_len, HEAD_DIM, dtype=dtype, device=device) * 0.1)
    v_logical = (torch.randn(seq_len, HEAD_DIM, dtype=dtype, device=device) * 0.1)
    q_act = torch.randn(num_q_heads, HEAD_DIM, dtype=dtype, device=device) * 0.1

    # Packed QKV for the 1 new token: [Q(all heads) | K_new | V_new].
    qkv_dim = (num_q_heads + 2 * num_kv_heads) * HEAD_DIM
    qkv = torch.zeros(1, qkv_dim, dtype=dtype, device=device)
    qkv[0, 0:num_q_heads * HEAD_DIM] = q_act.reshape(-1)
    qkv[0, num_q_heads * HEAD_DIM:(num_q_heads + 1) * HEAD_DIM] = k_logical[seq_len - 1]
    qkv[0, (num_q_heads + 1) * HEAD_DIM:(num_q_heads + 2) * HEAD_DIM] = v_logical[seq_len - 1]

    # Paged KV cache (NHD): [pages, page_size, kv_heads, head_dim]. Fill history
    # positions 0..seq_len-2 at their physical (page, offset).
    k_cache = torch.zeros(max_num_pages, page_size, num_kv_heads, HEAD_DIM, dtype=dtype, device=device)
    v_cache = torch.zeros(max_num_pages, page_size, num_kv_heads, HEAD_DIM, dtype=dtype, device=device)
    for p in range(seq_len - 1):
        phys = page_indices[p // page_size]
        off = p % page_size
        k_cache[phys, off, 0, :] = k_logical[p]
        v_cache[phys, off, 0, :] = v_logical[p]

    output = torch.zeros(1, num_q_heads * HEAD_DIM, dtype=dtype, device=device)
    cos_pos = torch.ones(max_seq_len, HEAD_DIM, dtype=dtype, device=device)   # identity rope
    sin_pos = torch.zeros(max_seq_len, HEAD_DIM, dtype=dtype, device=device)
    q_norm = torch.ones(HEAD_DIM, dtype=dtype, device=device)
    k_norm = torch.ones(HEAD_DIM, dtype=dtype, device=device)

    # Forged runtime state: 1 request, 1 query token, num_pages pages.
    qo_indptr = torch.tensor([0, 1], dtype=torch.int32, device=device)
    paged_kv_indptr = torch.tensor([0, num_pages], dtype=torch.int32, device=device)
    pkv_indices = torch.zeros(max_num_pages, dtype=torch.int32, device=device)
    pkv_indices[:num_pages] = torch.tensor(page_indices, dtype=torch.int32, device=device)
    paged_kv_last_page_len = torch.tensor([last_page_len], dtype=torch.int32, device=device)
    step = torch.tensor([seq_len - 1], dtype=torch.int32, device=device)
    tokens = torch.zeros(1, max_seq_len, dtype=torch.int64, device=device)

    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params.update(
        test_mode=True,
        num_workers=num_workers,
        num_local_schedulers=num_schedulers,
        max_seq_length=max_seq_len,
        max_num_batched_requests=1,
        max_num_batched_tokens=1,
        max_num_pages=max_num_pages,
        page_size=page_size,
        meta_tensors={
            "tokens": tokens, "step": step,
            "qo_indptr_buffer": qo_indptr,
            "paged_kv_indptr_buffer": paged_kv_indptr,
            "paged_kv_indices_buffer": pkv_indices,
            "paged_kv_last_page_len_buffer": paged_kv_last_page_len,
        },
    )
    pk = PersistentKernel(**params)

    qkv_dt = pk.attach_input(qkv.contiguous(), name="qkv")
    k_dt = pk.attach_input(k_cache.contiguous(), name="k_cache")
    v_dt = pk.attach_input(v_cache.contiguous(), name="v_cache")
    qn_dt = pk.attach_input(q_norm.contiguous(), name="q_norm")
    kn_dt = pk.attach_input(k_norm.contiguous(), name="k_norm")
    cos_dt = pk.attach_input(cos_pos.contiguous(), name="cos_pos")
    sin_dt = pk.attach_input(sin_pos.contiguous(), name="sin_pos")
    out_dt = pk.attach_input(output.contiguous(), name="attn_out")

    pk.paged_attention_layer(
        input=qkv_dt, k_cache=k_dt, v_cache=v_dt, q_norm=qn_dt, k_norm=kn_dt,
        cos_pos_embed=cos_dt, sin_pos_embed=sin_dt, output=out_dt,
        grid_dim=(1, num_kv_heads, 1), block_dim=(128, 1, 1),
        enable_qk_norm=False,
    )
    folder = "./output/forge_meta_paged_attn"
    pk.compile(output_dir=folder)
    pk()
    torch.cuda.synchronize()

    # Reference: the single query token (at position seq_len-1) attends causally
    # over all seq_len positions; per query head h (GQA: all heads share the 1 KV head).
    k_all = k_logical.float()                       # (seq_len, D)
    v_all = v_logical.float()
    o_ref = torch.zeros(num_q_heads, HEAD_DIM, device=device)
    for h in range(num_q_heads):
        scores = (q_act[h].float()[None, :] @ k_all.t()) / math.sqrt(HEAD_DIM)  # (1, seq_len)
        o_ref[h] = (torch.softmax(scores, dim=-1) @ v_all).squeeze(0)
    o_ref = o_ref.reshape(-1)

    max_abs = (output[0].float() - o_ref).abs().max().item()
    # forged meta must survive verbatim (no init_request_resources / prepare_next_batch)
    assert qo_indptr.tolist() == [0, 1]
    assert paged_kv_indptr.tolist() == [0, num_pages]
    assert paged_kv_last_page_len.tolist() == [last_page_len]
    print(f"[{label}] heads(q={num_q_heads},kv={num_kv_heads}) seq_len={seq_len} "
          f"pages={page_indices} last_page_len={last_page_len}  max|MPK-ref|={max_abs:.5f}")
    pk.finalize()
    return max_abs


def main():
    tol = 2e-2  # bf16
    cases = [
        # label,                 q_heads, kv_heads, page_size, max_pages, seq_len, page_indices
        ("single_page_decode",         1, 1, 64,  4,   5,  [0]),
        ("multipage_decode",           1, 1, 64,  4, 150,  [0, 1, 2]),   # fills 3 pages, partial last
        ("noncontiguous_pages",        1, 1, 64,  8, 150,  [3, 1, 2]),   # scattered physical pages
        ("gqa_decode",                 4, 1, 64,  4,  70,  [0, 1]),      # 4 query heads share 1 KV head
    ]
    failures = []
    for (label, qh, kvh, ps, mp, sl, pages) in cases:
        err = run_case(label, qh, kvh, ps, mp, sl, pages)
        if not (err < tol):
            failures.append((label, err))
    if failures:
        for label, err in failures:
            print(f"FAILED: {label} max abs err {err:.5f} >= {tol}")
        raise SystemExit(1)
    print("PASS: all forged-state paged-attention cases match reference; "
          "forged meta tensors survived intact.")


if __name__ == "__main__":
    main()
