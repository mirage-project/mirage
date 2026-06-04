"""Verify that planning_paged_attention_layer shares one planner across layers.

Builds a K-layer test_mode kernel where every layer has identical attention
shape (so all layers reuse the same shared plan buffer). Confirms:

  1. Each layer's output matches a torch causal-SDPA reference.
  2. The compiled task graph contains zero planner tasks (task_type 297): the
     planner runs inside the scheduler's per-iteration prepare step, not as a
     graph task.
"""

import json
import os
import sys
import time

import numpy as np
import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel

TASK_ATTENTION_PLANNER_SM100 = 297


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


def run_multilayer_case(num_layers: int = 4, num_tokens: int = 32,
                        time_iters: int = 0):
    torch.manual_seed(0)
    device, dtype = "cuda", torch.bfloat16

    num_q_heads, num_kv_heads, head_dim = 32, 8, 128
    gqa = num_q_heads // num_kv_heads
    page_size = 64
    history_pages = 2
    history_len = history_pages * page_size
    seq_len = history_len + num_tokens
    num_pages_used = (seq_len + page_size - 1) // page_size
    last_pl = seq_len - (num_pages_used - 1) * page_size
    max_seq_length = max(512, seq_len)
    max_num_pages = max(64, num_pages_used + 1)
    max_num_batched_requests = 1
    max_num_batched_tokens = max(32, num_tokens)

    qo_indptr = torch.tensor([0, num_tokens], device=device, dtype=torch.int32)
    paged_kv_indptr = torch.tensor([0, num_pages_used], device=device, dtype=torch.int32)
    paged_kv_indices = torch.arange(num_pages_used, device=device, dtype=torch.int32)
    paged_kv_last_page_len = torch.tensor([last_pl], device=device, dtype=torch.int32)

    layer_data = []
    for _ in range(num_layers):
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
        qkv_padded = torch.zeros(max_num_batched_tokens, fused_outdim,
                                 dtype=dtype, device=device)
        qkv_padded[:num_tokens] = qkv
        output = torch.zeros(max_num_batched_tokens, num_q_heads * head_dim,
                             dtype=dtype, device=device)
        layer_data.append({
            "q_new": q_new, "k_new": k_new, "v_new": v_new,
            "k_hist": k_hist, "v_hist": v_hist,
            "paged_k": paged_k, "paged_v": paged_v,
            "qkv_padded": qkv_padded, "output": output,
        })

    q_norm = torch.ones(head_dim, dtype=dtype, device=device)
    k_norm = torch.ones(head_dim, dtype=dtype, device=device)

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
        "step": torch.zeros(1, dtype=torch.int32, device=device),
        "tokens": torch.zeros(1, max_seq_length, dtype=torch.int64, device=device),
        "input_tokens": torch.zeros(max_num_batched_tokens, dtype=torch.int64, device=device),
        "output_tokens": torch.zeros(max_num_batched_tokens, dtype=torch.int64, device=device),
        "new_token_nums": torch.zeros(max_num_batched_requests, dtype=torch.int32, device=device),
        "prompt_length": torch.tensor([seq_len], dtype=torch.int32, device=device),
        "qo_indptr_buffer": qo_indptr,
        "paged_kv_indptr_buffer": paged_kv_indptr,
        "paged_kv_indices_buffer": paged_kv_indices,
        "paged_kv_last_page_len_buffer": paged_kv_last_page_len,
    }
    pk = PersistentKernel(**params)
    q_norm_dt = pk.attach_input(q_norm, name="q_norm")
    k_norm_dt = pk.attach_input(k_norm, name="k_norm")

    for idx, ld in enumerate(layer_data):
        qkv_dt = pk.attach_input(ld["qkv_padded"], name=f"qkv_{idx}")
        k_cache_dt = pk.attach_input(ld["paged_k"], name=f"paged_k_{idx}")
        v_cache_dt = pk.attach_input(ld["paged_v"], name=f"paged_v_{idx}")
        out_dt = pk.attach_input(ld["output"], name=f"out_{idx}")
        pk.planning_paged_attention_layer(
            input=qkv_dt, k_cache=k_cache_dt, v_cache=v_cache_dt,
            q_norm=q_norm_dt, k_norm=k_norm_dt,
            cos_pos_embed=None, sin_pos_embed=None,
            output=out_dt,
            num_buckets=128,
            max_works=1024,
            max_prefill_tokens_per_work=32,
            max_decode_tokens_per_work=8,
            consumer="dual",
            prefill_threshold=16,
            kv_split_size=None,
        )

    folder = os.path.dirname(__file__)
    pk.compile(output_dir=folder)
    pk()
    torch.cuda.synchronize()

    timings_ms = []
    for _ in range(time_iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        pk()
        torch.cuda.synchronize()
        timings_ms.append((time.perf_counter() - t0) * 1000.0)

    diffs = []
    for ld in layer_data:
        ref = torch_attention_ref(ld["q_new"], ld["k_hist"], ld["v_hist"],
                                  ld["k_new"], ld["v_new"])
        actual = ld["output"][:num_tokens]
        diff = (actual.to(torch.float32) - ref.to(torch.float32)).abs()
        diffs.append((diff.max().item(), diff.mean().item()))

    task_graph_path = os.path.join(folder, "task_graph_rank0.json")
    with open(task_graph_path) as f:
        task_graph = json.load(f)
    planner_count = sum(
        1 for t in task_graph["all_tasks"]
        if t.get("task_type") == TASK_ATTENTION_PLANNER_SM100
    )

    pk.finalize()
    return diffs, planner_count, timings_ms


def main():
    print("\n=== Planner sharing across layers ===\n")
    time_iters = int(os.environ.get("MIRAGE_PLAN_SHARE_TIME_ITERS", "0"))
    for k in (2, 4, 8):
        diffs, planner_count, timings_ms = run_multilayer_case(
            num_layers=k, num_tokens=32, time_iters=time_iters)
        max_diff = max(d[0] for d in diffs)
        mean_diff = max(d[1] for d in diffs)
        timing_str = ""
        if timings_ms:
            timings_ms_sorted = sorted(timings_ms)
            n = len(timings_ms_sorted)
            median = timings_ms_sorted[n // 2]
            p10 = timings_ms_sorted[max(0, n // 10)]
            timing_str = (f", test-mode launch median={median:.3f}ms, "
                          f"p10={p10:.3f}ms over {n} iters")
        print(f"K={k} layers: max_diff={max_diff:.4f}, mean_diff={mean_diff:.6f}, "
              f"planner_tasks_in_graph={planner_count}{timing_str}")
        if max_diff > 0.3 or mean_diff > 0.02:
            print(f"   FAILED (numerical)")
            sys.exit(1)
        # The planner runs inside the scheduler, not as a graph task, so no
        # planner tasks should ever appear in the task graph.
        expected = 0
        if planner_count != expected:
            print(f"   FAILED (expected {expected} planner tasks, got {planner_count})")
            sys.exit(1)
        for i, (mx, mn) in enumerate(diffs):
            print(f"   layer {i}: max_diff={mx:.4f}, mean_diff={mn:.6f}")
    print("\nALL PASSED")


if __name__ == "__main__":
    main()
