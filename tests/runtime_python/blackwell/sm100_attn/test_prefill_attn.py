import torch
import runtime_kernel_blackwell

from torch.nn import functional as F

torch.set_printoptions(sci_mode=False, profile="full")
# torch.set_printoptions(sci_mode=False)

g = torch.Generator(device="cuda").manual_seed(1234)

num_qo_heads = 32
num_kv_heads = 8
head_dim_qk = 128
head_dim_v = 128
max_num_pages = 1024
page_size = 128
causal = True
batch_size = 20
qo_len = 128
kv_len = 2048
p_qo_tile_size = 128
d_qo_tile_size = 16

worker_idx = 0

if __name__ == "__main__":

    print(
        f"\n=== Testing mixed attention SM100 with num_qo_heads={num_qo_heads}, num_kv_heads={num_kv_heads}, head_dim_qk={head_dim_qk}, head_dim_v={head_dim_v}, page_size={page_size}, causal={causal} ==="
    )

    decode_work_indptr = torch.zeros((128 + 1,), dtype=torch.int32, device="cuda")
    prefill_work_indptr = torch.arange(128 + 1, dtype=torch.int32, device="cuda")
    
    worker_batch_indices = torch.randint(0, batch_size, (128,), dtype=torch.int32, device="cuda", generator=g)
    worker_kv_head_indices = torch.randint(0, num_kv_heads, (128,), dtype=torch.int32, device="cuda", generator=g)
    worker_packed_qo_indices = torch.randint(0, qo_len // p_qo_tile_size, (128,), dtype=torch.int32, device="cuda", generator=g)
    worker_kv_start = torch.zeros((128,), dtype=torch.int32, device="cuda")
    worker_kv_end = torch.full((128,), kv_len, dtype=torch.int32, device="cuda")
    
    q_tensor = torch.randn((batch_size * qo_len, num_qo_heads, head_dim_qk), device="cuda", dtype=torch.bfloat16, generator=g)
    output_tensor = torch.randn((batch_size * qo_len, num_qo_heads, head_dim_v), device="cuda", dtype=torch.bfloat16, generator=g)
    paged_k = torch.randn((max_num_pages, page_size, num_kv_heads, head_dim_qk), device="cuda", dtype=torch.bfloat16, generator=g)
    paged_v = torch.randn((max_num_pages, page_size, num_kv_heads, head_dim_v), device="cuda", dtype=torch.bfloat16, generator=g)
    
    paged_kv_indices_buffer = torch.arange(0, (kv_len // page_size) * batch_size, device="cuda", dtype=torch.int32)
    paged_kv_indptr_buffer = torch.arange(0, (kv_len // page_size) * batch_size + 1, step=(kv_len // page_size), device="cuda", dtype=torch.int32)
    paged_kv_last_page_len_buffer = torch.full((batch_size,), page_size, device="cuda", dtype=torch.int32)
    
    # print("decode_work_indptr", decode_work_indptr)
    # print("prefill_work_indptr", prefill_work_indptr)
    # print("worker_batch_indices", worker_batch_indices)
    # print("worker_kv_head_indices", worker_kv_head_indices)
    # print("worker_packed_qo_indices", worker_packed_qo_indices)
    # print("worker_kv_start", worker_kv_start)
    # print("worker_kv_end", worker_kv_end)
    # print("paged_kv_indices_buffer", paged_kv_indices_buffer)
    # print("paged_kv_indptr_buffer", paged_kv_indptr_buffer)
    # print("paged_kv_last_page_len_buffer", paged_kv_last_page_len_buffer)
    # exit(0)
    # mpk impl
    runtime_kernel_blackwell.mixed_attn_sm100(
        q_tensor, 
        paged_k, 
        paged_v,
        output_tensor,
        decode_work_indptr, 
        prefill_work_indptr, 
        worker_batch_indices,
        worker_kv_head_indices,
        worker_packed_qo_indices,
        worker_kv_start,
        worker_kv_end,
        paged_kv_indices_buffer,
        paged_kv_indptr_buffer,
        paged_kv_last_page_len_buffer,
        worker_idx
    )

    # # Warm-up
    # for _ in range(16):
    #     runtime_kernel_blackwell.mixed_attn_sm100(x, residual, torch_topk_weights, output)

    # torch.cuda.synchronize()
    # starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
    #     enable_timing=True
    # )
    # repetitions = 1000
    # starter.record()
    # for rep in range(repetitions):
    #     runtime_kernel_blackwell.mixed_attn_sm100(x, residual, torch_topk_weights, output)
    # ender.record()
    # torch.cuda.synchronize()
    # total_time = starter.elapsed_time(ender)
    # avg_time = total_time / repetitions
    # print(f"Average time over {repetitions} runs: {avg_time:.6f} ms")
