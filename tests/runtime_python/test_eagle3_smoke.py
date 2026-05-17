"""Smoke test for Eagle3Builder: load real Eagle3 weights + build draft graph.

This avoids loading the 30B target model. It mocks the target side by:
- Creating dummy aux buffers (3 × (mbt, H))
- Creating a dummy target_argmax DTensor
- Creating a dummy shared embed weight

Then calls Eagle3Builder.build_draft_loop and triggers task graph generation.
Success means all 4 new task types + the draft loop wiring compile cleanly.
"""

import os
import sys
sys.path.insert(0, '/home/letianr/mirage/python')

import torch
import mirage as mi
from mirage.mpk.models.eagle3.builder import Eagle3Builder, load_eagle3_draft

EAGLE3_PATH = (
    "/raid/catalyst/models/models--lmsys--"
    "SGLang-EAGLE3-Qwen3-30B-A3B-Instruct-2507-SpecForge-Nex/snapshots/"
    "d1ac703a537d2b8a5b748d4f5f8ca7e97efe9214")

print("Loading Eagle3 draft weights from", EAGLE3_PATH)
state_dict, config = load_eagle3_draft(EAGLE3_PATH)
print(f"Config: hidden_size={config['hidden_size']}, "
      f"num_layers={config['num_hidden_layers']}, "
      f"draft_vocab_size={config['draft_vocab_size']}, "
      f"num_q_heads={config['num_attention_heads']}, "
      f"num_kv_heads={config['num_key_value_heads']}, "
      f"head_dim={config['head_dim']}")
print(f"Loaded {len(state_dict)} draft weight tensors")

# Mirror demo_30B_A3B setup
mbt = 5  # K+1 with K=4
num_workers = 96
num_schedulers = 48
max_seq_length = 512
max_num_pages = 16
page_size = 4096
H = config['hidden_size']

step = torch.zeros((1,), dtype=torch.int32, device='cuda')
tokens = torch.zeros((1, max_seq_length), dtype=torch.int64, device='cuda')
input_tokens = torch.zeros((mbt, 1), dtype=torch.int64, device='cuda')
output_tokens = torch.zeros((mbt, 1), dtype=torch.int64, device='cuda')
num_new_tokens = torch.zeros((1,), dtype=torch.int32, device='cuda')
prompt_lengths = torch.zeros((1,), dtype=torch.int32, device='cuda')
qo_indptr = torch.zeros((2,), dtype=torch.int32, device='cuda')
paged_kv_indptr = torch.zeros((2,), dtype=torch.int32, device='cuda')
paged_kv_indices = torch.zeros((max_num_pages,), dtype=torch.int32, device='cuda')
paged_kv_last_page_len = torch.zeros((1,), dtype=torch.int32, device='cuda')

mpk = mi.PersistentKernel(
    mode='offline', world_size=1, mpi_rank=0,
    num_workers=num_workers, num_local_schedulers=num_schedulers, num_remote_schedulers=0,
    max_seq_length=max_seq_length, max_num_batched_requests=1, max_num_batched_tokens=mbt,
    max_num_pages=max_num_pages, page_size=page_size,
    eos_token_id=151645,
    meta_tensors={
        'step': step, 'tokens': tokens,
        'input_tokens': input_tokens, 'output_tokens': output_tokens,
        'num_new_tokens': num_new_tokens, 'prompt_lengths': prompt_lengths,
        'qo_indptr_buffer': qo_indptr,
        'paged_kv_indptr_buffer': paged_kv_indptr,
        'paged_kv_indices_buffer': paged_kv_indices,
        'paged_kv_last_page_len_buffer': paged_kv_last_page_len,
    },
    profiler_tensor=None, trace_name='', spec_decode_config=None,
    use_cutlass_kernel=True,
)
print(f"target_cc = {mpk.target_cc}, num_workers = {mpk.num_workers}")

# Mock target-side DTensors
target_w_embed_buf = torch.zeros((151936, H), dtype=torch.bfloat16, device='cuda')
target_w_embed = mpk.attach_input(torch_tensor=target_w_embed_buf, name='embed_tokens')

cos_buf = torch.zeros((4096, config['head_dim']), dtype=torch.bfloat16, device='cuda')
sin_buf = torch.zeros((4096, config['head_dim']), dtype=torch.bfloat16, device='cuda')
cos_pos = mpk.attach_input(torch_tensor=cos_buf, name='cos_position_embedding')
sin_pos = mpk.attach_input(torch_tensor=sin_buf, name='sin_position_embedding')

aux_h0 = mpk.new_tensor(dims=(mbt, H), dtype=mi.bfloat16, name='eagle3_aux_h0', io_category='cuda_tensor')
aux_h1 = mpk.new_tensor(dims=(mbt, H), dtype=mi.bfloat16, name='eagle3_aux_h1', io_category='cuda_tensor')
aux_h2 = mpk.new_tensor(dims=(mbt, H), dtype=mi.bfloat16, name='eagle3_aux_h2', io_category='cuda_tensor')

target_argmax = mpk.attach_input(torch_tensor=output_tokens, name='output_token')

K = 4
print(f"\nBuilding Eagle3 draft loop (K={K})...")
eagle3 = Eagle3Builder(
    mpk=mpk,
    draft_state_dict=state_dict,
    draft_config=config,
    target_hidden_size=H,
    target_w_embed=target_w_embed,
    cos_pos_embed=cos_pos,
    sin_pos_embed=sin_pos,
    num_draft_steps=K,
    use_aux_norm=False,
)
accepted_count_dummy = mpk.new_tensor(
    dims=(mbt, 1), dtype=mi.int32,
    name="smoke_accepted_count", io_category="cuda_tensor")
all_draft_ids = eagle3.build_draft_loop(
    aux_h0=aux_h0, aux_h1=aux_h1, aux_h2=aux_h2,
    target_argmax_token=target_argmax,
    accepted_count=accepted_count_dummy,
)
print(f"build_draft_loop returned all_draft_ids: dims={[all_draft_ids.dim(i) for i in range(all_draft_ids.num_dims)]}")

print("\nGenerating task graph...")
results = mpk.kn_graph.generate_task_graph(num_gpus=1, my_gpu_id=0)
cu_size = len(results['cuda_code'])
json_size = len(results['json_file'])
print(f"  cuda_code: {cu_size} chars")
print(f"  json_file: {json_size} chars")
print(f"  contains TASK_COPY: {results['cuda_code'].count('TASK_COPY')}")
print(f"  contains TASK_EAGLE3_AUX_CONCAT: {results['cuda_code'].count('TASK_EAGLE3_AUX_CONCAT')}")
print(f"  contains TASK_EAGLE3_INPUT_CONCAT: {results['cuda_code'].count('TASK_EAGLE3_INPUT_CONCAT')}")
print(f"  contains TASK_EAGLE3_D2T_REMAP: {results['cuda_code'].count('TASK_EAGLE3_D2T_REMAP')}")
print(f"  contains TASK_MTP_TOKEN_SCATTER: {results['cuda_code'].count('TASK_MTP_TOKEN_SCATTER')}")

print("\n--- SMOKE TEST PASSED ---")
