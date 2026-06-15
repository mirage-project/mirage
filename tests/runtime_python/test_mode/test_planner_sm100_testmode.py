"""Correctness test for the issue-#627 attention planner + planned consumers.

Builds a 1-layer test_mode kernel using ``planning_paged_attention_layer``,
which emits planned_prefill + planned_decode consumer tasks. The planner is
not a task — it runs inside the scheduler's per-iteration prepare step and
fills the shared plan buffer the consumers read. With a single request whose
``packed_qo > 16`` (prefill iter), the planner routes to the prefill consumer;
``packed_qo ≤ 16`` (decode iter) routes to the decode consumer.

Compared against the same torch causal-SDPA reference used in the
standalone-prefill test.
"""

import os
import sys

import numpy as np
import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel


def build_qkv_interleaved(q, k, v):
    T, Hq, D = q.shape
    Hkv = k.shape[1]
    gqa = Hq // Hkv
    out = torch.empty(T, Hkv, (gqa + 2) * D, dtype=q.dtype, device=q.device)
    out[:, :, 0:gqa * D] = q.view(T, Hkv, gqa, D).reshape(T, Hkv, gqa * D)
    out[:, :, gqa * D:(gqa + 1) * D] = k
    out[:, :, (gqa + 1) * D:(gqa + 2) * D] = v
    return out.reshape(T, Hkv * (gqa + 2) * D)


def torch_attention_ref(q, k_hist, v_hist, k_new, v_new):
    T, Hq, D = q.shape
    Hkv = k_hist.shape[1]
    gqa = Hq // Hkv
    S_hist = k_hist.shape[0]
    seq_len = S_hist + T
    k_full = torch.cat([k_hist, k_new], dim=0)
    v_full = torch.cat([v_hist, v_new], dim=0)
    sm_scale = 1.0 / np.sqrt(D)
    out = torch.zeros(T, Hq, D, dtype=torch.float32, device=q.device)
    for h_kv in range(Hkv):
        k_h = k_full[:, h_kv, :].to(torch.float32)
        v_h = v_full[:, h_kv, :].to(torch.float32)
        for g in range(gqa):
            h_q = h_kv * gqa + g
            q_h = q[:, h_q, :].to(torch.float32)
            scores = q_h @ k_h.transpose(0, 1) * sm_scale
            cols = torch.arange(seq_len, device=q.device)
            abs_pos = torch.arange(S_hist, seq_len, device=q.device)[:, None]
            mask = cols[None, :] <= abs_pos
            scores = scores.masked_fill(~mask, float("-inf"))
            attn = torch.softmax(scores, dim=-1)
            out[:, h_q, :] = attn @ v_h
    return out.to(q.dtype).reshape(T, Hq * D)


def run_one_case(
    num_tokens: int,
    max_num_batched_tokens: int,
    history_pages: int = 2,
    history_len: int = None,
    kv_split_size: int = None,
    use_qk_norm: bool = True,
    num_buckets: int = 128,
    page_size: int = 64,
    max_decode_tokens_per_work: int = 8,
    prefill_threshold: int = 16,
    inspect_plan: bool = False,
):
    torch.manual_seed(0)
    device, dtype = "cuda", torch.bfloat16

    num_q_heads, num_kv_heads, head_dim = 32, 8, 128
    gqa = num_q_heads // num_kv_heads
    max_num_batched_requests = 1

    if history_len is None:
        history_len = history_pages * page_size
    seq_len = history_len + num_tokens
    num_pages_used = (seq_len + page_size - 1) // page_size
    last_pl = seq_len - (num_pages_used - 1) * page_size
    max_seq_length = max(512, seq_len)
    min_pages = 64 if page_size <= 512 else 4
    max_num_pages = max(min_pages, num_pages_used + 1)

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

    q_new = torch.randn(num_tokens, num_q_heads, head_dim, dtype=dtype, device=device)
    k_new = torch.randn(num_tokens, num_kv_heads, head_dim, dtype=dtype, device=device)
    v_new = torch.randn(num_tokens, num_kv_heads, head_dim, dtype=dtype, device=device)
    k_hist = torch.randn(history_len, num_kv_heads, head_dim, dtype=dtype, device=device)
    v_hist = torch.randn(history_len, num_kv_heads, head_dim, dtype=dtype, device=device)

    paged_k = torch.zeros(max_num_pages, page_size, num_kv_heads, head_dim,
                          dtype=dtype, device=device)
    paged_v = torch.zeros(max_num_pages, page_size, num_kv_heads, head_dim,
                          dtype=dtype, device=device)
    paged_k.view(-1, num_kv_heads, head_dim)[:history_len] = k_hist
    paged_v.view(-1, num_kv_heads, head_dim)[:history_len] = v_hist

    qkv = build_qkv_interleaved(q_new, k_new, v_new)
    fused_outdim = (num_q_heads + 2 * num_kv_heads) * head_dim
    qkv_padded = torch.zeros(max_num_batched_tokens, fused_outdim, dtype=dtype, device=device)
    qkv_padded[:num_tokens] = qkv
    output = torch.zeros(max_num_batched_tokens, num_q_heads * head_dim,
                         dtype=dtype, device=device)

    q_norm = torch.ones(head_dim, dtype=dtype, device=device)
    k_norm = torch.ones(head_dim, dtype=dtype, device=device)
    cos_pos = torch.zeros(max_seq_length, head_dim, dtype=dtype, device=device)
    sin_pos = torch.zeros(max_seq_length, head_dim, dtype=dtype, device=device)

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
        "step": step, "tokens": tokens, "input_tokens": input_tokens,
        "output_tokens": output_tokens, "new_token_nums": new_token_nums,
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
    pk.attach_input(cos_pos, name="cos")
    pk.attach_input(sin_pos, name="sin")
    out_dt = pk.attach_input(output, name="out")
    resolved_num_buckets = num_buckets
    if inspect_plan:
        # The planner runs inside the scheduler and fills a shared plan buffer
        # (created internally by planning_paged_attention_layer). Resolve its
        # bucket count so we can parse that buffer after the run.
        resolved_num_buckets, _, _ = pk.resolve_attention_plan_shape(
            num_kv_heads=num_kv_heads,
            gqa_group=gqa,
            num_buckets=num_buckets,
            max_works=1024,
            max_prefill_tokens_per_work=32,
            max_decode_tokens_per_work=max_decode_tokens_per_work,
            consumer="both",
            kv_split_size=kv_split_size,
            prefill_threshold=prefill_threshold,
        )

    # Both consumers active; planner routes per request type.
    pk.planning_paged_attention_layer(
        input=qkv_dt, k_cache=k_cache_dt, v_cache=v_cache_dt,
        q_norm=q_norm_dt if use_qk_norm else None,
        k_norm=k_norm_dt if use_qk_norm else None,
        cos_pos_embed=None, sin_pos_embed=None,
        output=out_dt,
        num_buckets=num_buckets,
        max_works=1024,
        max_prefill_tokens_per_work=32,
        max_decode_tokens_per_work=max_decode_tokens_per_work,
        consumer="both",
        prefill_threshold=prefill_threshold,
        kv_split_size=kv_split_size,
    )

    folder = os.path.dirname(__file__)
    pk.compile(output_dir=folder)
    pk()
    torch.cuda.synchronize()

    ref = torch_attention_ref(q_new, k_hist, v_hist, k_new, v_new)
    actual = output[:num_tokens]
    diff = (actual.to(torch.float32) - ref.to(torch.float32)).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    plan_stats = None
    if inspect_plan:
        plan_storages = pk.get_attention_plan_storage()
        assert len(plan_storages) == 1, (
            f"expected exactly one shared plan buffer, got {len(plan_storages)}")
        plan_cpu = next(iter(plan_storages.values())).cpu()
        decode_indptr_offset = resolved_num_buckets + 1
        decode_work_count = int(
            plan_cpu[decode_indptr_offset + resolved_num_buckets] -
            plan_cpu[decode_indptr_offset]
        )
        prefill_indptr = plan_cpu[:resolved_num_buckets + 1]
        prefill_bucket_counts = [
            int(prefill_indptr[i + 1] - prefill_indptr[i])
            for i in range(resolved_num_buckets)
        ]
        plan_stats = {
            "decode_work_count": decode_work_count,
            "prefill_bucket_counts": prefill_bucket_counts,
        }

    pk.finalize()
    if inspect_plan:
        return max_diff, mean_diff, plan_stats
    return max_diff, mean_diff


def run_mixed_batch_case():
    torch.manual_seed(1)
    device, dtype = "cuda", torch.bfloat16

    num_q_heads, num_kv_heads, head_dim = 32, 8, 128
    page_size = 64
    kv_split_size = 256
    req_specs = [
        (64, 224),   # split prefill; first Q tile uses fewer chunks
        (32, 320),   # split prefill spanning two chunks
        (1, 1024),   # decode request routed through the same planner
    ]
    max_num_batched_requests = 4
    max_num_batched_tokens = 128
    max_seq_length = 1280

    qo_indptr_list = [0]
    paged_kv_indptr_list = [0]
    paged_kv_indices_list = []
    paged_kv_last_page_len_list = []
    req_tensors = []
    page_cursor = 0
    total_tokens = 0

    for num_tokens, history_len in req_specs:
        seq_len = history_len + num_tokens
        num_pages = (seq_len + page_size - 1) // page_size
        last_pl = seq_len - (num_pages - 1) * page_size
        qo_indptr_list.append(qo_indptr_list[-1] + num_tokens)
        paged_kv_indptr_list.append(paged_kv_indptr_list[-1] + num_pages)
        paged_kv_indices_list.extend(range(page_cursor, page_cursor + num_pages))
        paged_kv_last_page_len_list.append(last_pl)
        page_cursor += num_pages
        total_tokens += num_tokens

        q_new = torch.randn(num_tokens, num_q_heads, head_dim,
                            dtype=dtype, device=device)
        k_new = torch.randn(num_tokens, num_kv_heads, head_dim,
                            dtype=dtype, device=device)
        v_new = torch.randn(num_tokens, num_kv_heads, head_dim,
                            dtype=dtype, device=device)
        k_hist = torch.randn(history_len, num_kv_heads, head_dim,
                             dtype=dtype, device=device)
        v_hist = torch.randn(history_len, num_kv_heads, head_dim,
                             dtype=dtype, device=device)
        req_tensors.append((q_new, k_hist, v_hist, k_new, v_new))

    while len(qo_indptr_list) < max_num_batched_requests + 1:
        qo_indptr_list.append(qo_indptr_list[-1])
    while len(paged_kv_indptr_list) < max_num_batched_requests + 1:
        paged_kv_indptr_list.append(paged_kv_indptr_list[-1])
    while len(paged_kv_last_page_len_list) < max_num_batched_requests:
        paged_kv_last_page_len_list.append(0)

    max_num_pages = max(64, page_cursor + 1)
    qo_indptr = torch.tensor(qo_indptr_list, device=device, dtype=torch.int32)
    paged_kv_indptr = torch.tensor(paged_kv_indptr_list, device=device,
                                   dtype=torch.int32)
    paged_kv_indices = torch.tensor(paged_kv_indices_list, device=device,
                                    dtype=torch.int32)
    paged_kv_last_page_len = torch.tensor(paged_kv_last_page_len_list,
                                          device=device, dtype=torch.int32)

    paged_k = torch.zeros(max_num_pages, page_size, num_kv_heads, head_dim,
                          dtype=dtype, device=device)
    paged_v = torch.zeros_like(paged_k)
    for r, (_, history_len) in enumerate(req_specs):
        first_page = paged_kv_indptr_list[r]
        num_pages = paged_kv_indptr_list[r + 1] - first_page
        page_ids = paged_kv_indices_list[first_page:first_page + num_pages]
        _, k_hist, v_hist, _, _ = req_tensors[r]
        assert page_ids == list(range(page_ids[0], page_ids[0] + num_pages))
        page_start = page_ids[0]
        paged_k[page_start:page_start + num_pages].reshape(
            -1, num_kv_heads, head_dim)[:history_len] = k_hist
        paged_v[page_start:page_start + num_pages].reshape(
            -1, num_kv_heads, head_dim)[:history_len] = v_hist

    fused_outdim = (num_q_heads + 2 * num_kv_heads) * head_dim
    qkv_padded = torch.zeros(max_num_batched_tokens, fused_outdim,
                             dtype=dtype, device=device)
    offset = 0
    for q_new, _, _, k_new, v_new in req_tensors:
        num_tokens = q_new.shape[0]
        qkv_padded[offset:offset + num_tokens] = build_qkv_interleaved(
            q_new, k_new, v_new)
        offset += num_tokens

    output = torch.zeros(max_num_batched_tokens, num_q_heads * head_dim,
                         dtype=dtype, device=device)
    q_norm = torch.ones(head_dim, dtype=dtype, device=device)
    k_norm = torch.ones(head_dim, dtype=dtype, device=device)

    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params.update(dict(
        test_mode=True,
        num_workers=num_workers,
        num_local_schedulers=num_schedulers,
        mpi_rank=0,
        world_size=1,
        max_seq_length=max_seq_length,
        max_num_batched_requests=max_num_batched_requests,
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_pages=max_num_pages,
        page_size=page_size,
        meta_tensors={
            "step": torch.zeros(1, dtype=torch.int32, device=device),
            "tokens": torch.zeros(1, max_seq_length, dtype=torch.int64, device=device),
            "input_tokens": torch.zeros(max_num_batched_tokens, dtype=torch.int64, device=device),
            "output_tokens": torch.zeros(max_num_batched_tokens, dtype=torch.int64, device=device),
            "new_token_nums": torch.zeros(max_num_batched_requests, dtype=torch.int32, device=device),
            "prompt_length": torch.tensor([max_seq_length] * max_num_batched_requests,
                                           dtype=torch.int32, device=device),
            "qo_indptr_buffer": qo_indptr,
            "paged_kv_indptr_buffer": paged_kv_indptr,
            "paged_kv_indices_buffer": paged_kv_indices,
            "paged_kv_last_page_len_buffer": paged_kv_last_page_len,
        },
    ))
    pk = PersistentKernel(**params)
    qkv_dt = pk.attach_input(qkv_padded, name="qkv")
    k_cache_dt = pk.attach_input(paged_k, name="paged_k")
    v_cache_dt = pk.attach_input(paged_v, name="paged_v")
    q_norm_dt = pk.attach_input(q_norm, name="q_norm")
    k_norm_dt = pk.attach_input(k_norm, name="k_norm")
    out_dt = pk.attach_input(output, name="out")
    pk.planning_paged_attention_layer(
        input=qkv_dt, k_cache=k_cache_dt, v_cache=v_cache_dt,
        q_norm=q_norm_dt, k_norm=k_norm_dt,
        cos_pos_embed=None, sin_pos_embed=None, output=out_dt,
        num_buckets=None,
        max_works=1024,
        max_prefill_tokens_per_work=32,
        max_decode_tokens_per_work=8,
        consumer="dual",
        kv_split_size=kv_split_size,
    )

    folder = os.path.dirname(__file__)
    pk.compile(output_dir=folder)
    pk()
    torch.cuda.synchronize()

    max_diff = 0.0
    mean_accum = 0.0
    per_request = []
    offset = 0
    for req_idx, (q_new, k_hist, v_hist, k_new, v_new) in enumerate(req_tensors):
        num_tokens = q_new.shape[0]
        ref = torch_attention_ref(q_new, k_hist, v_hist, k_new, v_new)
        actual = output[offset:offset + num_tokens]
        diff = (actual.to(torch.float32) - ref.to(torch.float32)).abs()
        req_max = diff.max().item()
        req_mean = diff.mean().item()
        max_diff = max(max_diff, req_max)
        mean_accum += req_mean
        per_request.append((req_idx, num_tokens, k_hist.shape[0], req_max, req_mean))
        offset += num_tokens

    pk.finalize()
    return max_diff, mean_accum / len(req_tensors), per_request


def main():
    print("\n=== Planner correctness test ===\n")
    for label, num_tokens, max_batched in [
        ("decode (1 token → planner routes to decode consumer)", 1, 8),
        ("prefill (32 tokens → planner routes to prefill consumer)", 32, 32),
        ("prefill (64 tokens → planner emits two Q tiles)", 64, 64),
    ]:
        print(f"{label}")
        max_diff, mean_diff = run_one_case(num_tokens, max_batched)
        print(f"   max_diff = {max_diff:.4f}, mean_diff = {mean_diff:.6f}")
        if max_diff > 0.3 or mean_diff > 0.02:
            print(f"   FAILED")
            sys.exit(1)
    print(f"   PASSED\n")
    print("decode tile sizing (8 tokens → one planned decode tile per KV head)")
    max_diff, mean_diff, plan_stats = run_one_case(
        8,
        8,
        max_decode_tokens_per_work=8,
        prefill_threshold=64,
        inspect_plan=True,
    )
    decode_works = plan_stats["decode_work_count"]
    print(
        f"   max_diff = {max_diff:.4f}, mean_diff = {mean_diff:.6f}, "
        f"decode_works = {decode_works}"
    )
    if max_diff > 0.3 or mean_diff > 0.02 or decode_works != 8:
        print(f"   FAILED")
        sys.exit(1)
    print(f"   PASSED\n")
    print("prefill Q tile scheduling (64 tokens → independent buckets)")
    max_diff, mean_diff, plan_stats = run_one_case(
        64,
        64,
        inspect_plan=True,
    )
    prefill_counts = plan_stats["prefill_bucket_counts"]
    active_prefill_buckets = sum(1 for count in prefill_counts if count > 0)
    max_prefill_bucket_count = max(prefill_counts)
    print(
        f"   max_diff = {max_diff:.4f}, mean_diff = {mean_diff:.6f}, "
        f"active_prefill_buckets = {active_prefill_buckets}, "
        f"max_bucket_count = {max_prefill_bucket_count}"
    )
    if (max_diff > 0.3 or mean_diff > 0.02 or
            active_prefill_buckets != 16 or max_prefill_bucket_count != 1):
        print(f"   FAILED")
        sys.exit(1)
    print(f"   PASSED\n")
    print("prefill split-KV (32 tokens, page_size=64, seq_len spans two chunks)")
    max_diff, mean_diff = run_one_case(
        32, 32, history_pages=5, kv_split_size=256)
    print(f"   max_diff = {max_diff:.4f}, mean_diff = {mean_diff:.6f}")
    if max_diff > 0.3 or mean_diff > 0.02:
        print(f"   FAILED")
        sys.exit(1)
    print(f"   PASSED\n")
    print("prefill split-KV enabled but unsplit (64 tokens, page_size=4096)")
    max_diff, mean_diff = run_one_case(
        64, 64, history_len=0, kv_split_size=256, page_size=4096)
    print(f"   max_diff = {max_diff:.4f}, mean_diff = {mean_diff:.6f}")
    if max_diff > 0.65 or mean_diff > 0.025:
        print(f"   FAILED")
        sys.exit(1)
    print(f"   PASSED\n")
    print("prefill split-KV (64 tokens, page_size=4096, seq_len spans chunks)")
    max_diff, mean_diff = run_one_case(
        64, 64, history_len=512, kv_split_size=256, page_size=4096)
    print(f"   max_diff = {max_diff:.4f}, mean_diff = {mean_diff:.6f}")
    if max_diff > 0.35 or mean_diff > 0.02:
        print(f"   FAILED")
        sys.exit(1)
    print(f"   PASSED\n")
    print("prefill split-KV (64 tokens, first Q tile needs fewer chunks)")
    max_diff, mean_diff = run_one_case(
        64, 64, history_len=224, kv_split_size=256)
    print(f"   max_diff = {max_diff:.4f}, mean_diff = {mean_diff:.6f}")
    if max_diff > 0.35 or mean_diff > 0.02:
        print(f"   FAILED")
        sys.exit(1)
    print(f"   PASSED\n")
    print("mixed split-KV (prefill + decode requests, uneven histories)")
    max_diff, mean_diff, per_request = run_mixed_batch_case()
    print(f"   max_diff = {max_diff:.4f}, mean_diff = {mean_diff:.6f}")
    for req_idx, num_tokens, history_len, req_max, req_mean in per_request:
        print(
            f"   request {req_idx}: tokens={num_tokens}, "
            f"history={history_len}, max_diff={req_max:.4f}, "
            f"mean_diff={req_mean:.6f}"
        )
    if max_diff > 0.4 or mean_diff > 0.02:
        print(f"   FAILED")
        sys.exit(1)
    print(f"   PASSED\n")
    print("ALL PASSED")


if __name__ == "__main__":
    main()
