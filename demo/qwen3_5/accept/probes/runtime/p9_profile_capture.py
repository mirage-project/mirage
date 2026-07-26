"""P9 probe, step 2 -- profiler-enabled capture at a given batch size.

Same graph as tests/ci-tests/run_batch_perf.py (1-token "." prompt, decode-only,
MODE_OFFLINE, use_cutlass_kernel=True) but with a real profiler_tensor attached
[MG S6.1; persistent_kernel.py profiler_tensor plumbing] so MPK's own runtime
emits per-task begin/end events. This script saves the RAW device buffer to
disk (both of the framework's own export paths crash for this config -- see
the module-level comment below) and a perf-summary JSON; a separate, GPU-free
pair of scripts (p9_decode_tolerant.py, then p9_attribution_analysis.py) turns
the raw buffer into the iteration-boundary attribution.
"""
import argparse
import json
import math
import os
import sys
import time

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "mirage"))
DEMO_DIR = os.path.join(ROOT, "demo", "qwen3")
sys.path.insert(0, DEMO_DIR)

from models.modeling_qwen3 import Qwen3ForCausalLM  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from demo import grid_for_rmsnorm_linear_layer  # noqa: E402

PAGE_SIZE = 4096
PROMPT = "."


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--max-num-batched-tokens", type=int, default=8)
    p.add_argument("--max-num-batched-requests", type=int, required=True)
    p.add_argument("--max-seq-length", type=int, default=512)
    p.add_argument("--ignore-eos", action="store_true", default=True)
    p.add_argument("--profiler-slots", type=int, default=3000 * 128,
                   help="uint64 words in the profiler ring buffer")
    p.add_argument("--out-dir", default="/home/muhengl/mpk-qwen35/probes/runtime_out")
    p.add_argument("--output-dir", default=None, help="mpk.compile() output_dir")
    return p.parse_args()


def main():
    args = parse_args()
    print("Input arguments:", args, flush=True)
    r = args.max_num_batched_requests
    output_dir = args.output_dir or f"/home/muhengl/mpk-qwen35/probes/runtime_out/p9_kernel_cache_r{r}"

    import mirage as mi
    import mirage.mpk.profiler_persistent as profiler_persistent

    # WORKAROUND for two real bugs in the shared profiler plumbing, found
    # while building this probe (neither is in a path M2-I11 owns; noted for
    # the memory inbox / a future profiler-maintenance issue instead of
    # edited here):
    #
    # 1. persistent_kernel.py's __call__ unconditionally runs
    #    export_to_perfetto_trace() BEFORE export_to_csv() whenever
    #    profiler_tensor is set, no try/except around either.
    #    export_to_perfetto_trace's tid_map is pre-populated only for
    #    block_idx in range(header.num_blocks), where the header is written
    #    by whichever kernel's "block 0" gets there first -- observed
    #    KeyError: (80, 0), a legitimate worker block index the header
    #    undercounted.
    # 2. Root cause of (1): workers (num_workers=128) and schedulers
    #    (num_schedulers=80, printed below) are SEPARATE kernel launches,
    #    each computing profiler_write_ptr/stride from its OWN local
    #    blockIdx/gridDim (profiler.h PROFILER_INIT). Worker block b and
    #    scheduler block b therefore both write their FIRST event to the
    #    exact same shared-buffer offset (1+b) -- confirmed by the observed
    #    export_to_csv crash "END without matching BEGIN: block=4 group=0
    #    ... event_no=0" (a worker/scheduler aliasing collision, not a
    #    buffer-overflow or a stale-name-map issue). Every scheduler block
    #    (0..79) aliases the same-indexed worker block repeatedly through
    #    the run since num_workers != num_schedulers (128 vs 80).
    #
    # Neither export function is safe to call as-is for this config. Skip
    # both; we save the raw buffer ourselves right after mpk() returns and
    # decode it tolerantly offline (p9_decode_tolerant.py), discarding only
    # the individual mismatched pairs instead of losing the whole trace.
    profiler_persistent.export_to_perfetto_trace = lambda *a, **kw: None
    profiler_persistent.export_to_csv = lambda *a, **kw: None

    torch.set_default_dtype(torch.bfloat16)
    torch.cuda.set_device(0)

    pages_per_request = math.ceil(args.max_seq_length / PAGE_SIZE)
    max_num_pages = max(16, r * pages_per_request)

    with torch.device("cuda"):
        model = Qwen3ForCausalLM.from_pretrained(
            args.model, 1, max_num_pages=max_num_pages, page_size=PAGE_SIZE
        ).to("cuda")
        tokenizer = AutoTokenizer.from_pretrained(args.model)

    model_inputs = tokenizer([PROMPT], return_tensors="pt").to(model.device)
    prompt_len = model_inputs.input_ids.shape[-1]
    assert prompt_len == 1

    tokens = torch.full((r, args.max_seq_length), 0, dtype=torch.long, device="cuda")
    tokens[:, :prompt_len] = model_inputs.input_ids[0]
    prompt_lengths = torch.full((r,), prompt_len, dtype=torch.int, device="cuda")
    step = torch.full((r,), 0, dtype=torch.int32, device="cuda")
    num_new_tokens = torch.full((r,), 1, dtype=torch.int32, device="cuda")

    positions = torch.arange(32768).unsqueeze(0).to(model.device)
    position_embeddings = model.model.rotary_emb(positions)
    input_tokens = torch.full((args.max_num_batched_tokens, 1), 0, dtype=torch.long, device="cuda")
    output_tokens = torch.full((args.max_num_batched_tokens, 1), 0, dtype=torch.long, device="cuda")

    hidden_size = model.config.hidden_size
    intermediate_size = model.config.intermediate_size
    lm_head_weight = torch.cat(
        (model.lm_head.weight,
         torch.full((153600 - model.config.vocab_size, hidden_size), 0, device="cuda")), 0)
    vocab_size = 153600
    num_q_heads = model.config.num_attention_heads
    num_kv_heads = model.config.num_key_value_heads
    head_dim = model.config.head_dim
    fused_outdim_1 = (num_q_heads + 2 * num_kv_heads) * head_dim
    fused_outdim_2 = 2 * intermediate_size

    num_workers, num_schedulers = mi.get_configurations_from_gpu(0)
    qo_indptr_buffer = torch.empty(r + 1, dtype=torch.int32, device="cuda")
    paged_kv_indptr_buffer = torch.empty(r + 1, dtype=torch.int32, device="cuda")
    paged_kv_indices_buffer = torch.empty(max_num_pages, dtype=torch.int32, device="cuda")
    paged_kv_last_page_len_buffer = torch.empty(r, dtype=torch.int32, device="cuda")

    print(f"[p9] num_workers={num_workers} num_schedulers={num_schedulers}", flush=True)
    profiler_tensor = torch.zeros(args.profiler_slots, dtype=torch.uint64, device="cuda").contiguous()
    os.makedirs(args.out_dir, exist_ok=True)
    trace_name = os.path.join(args.out_dir, f"p9_r{r}")  # __call__ auto-exports to trace_name + ".csv"

    mpk = mi.PersistentKernel(
        mode="offline", world_size=1, mpi_rank=0,
        num_workers=num_workers, num_local_schedulers=num_schedulers,
        num_remote_schedulers=0, max_seq_length=args.max_seq_length,
        max_num_batched_requests=r, max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_pages=max_num_pages, page_size=PAGE_SIZE,
        eos_token_id=model.config.eos_token_id if not args.ignore_eos else -1,
        meta_tensors={
            "step": step, "tokens": tokens, "input_tokens": input_tokens,
            "output_tokens": output_tokens, "num_new_tokens": num_new_tokens,
            "prompt_lengths": prompt_lengths, "qo_indptr_buffer": qo_indptr_buffer,
            "paged_kv_indptr_buffer": paged_kv_indptr_buffer,
            "paged_kv_indices_buffer": paged_kv_indices_buffer,
            "paged_kv_last_page_len_buffer": paged_kv_last_page_len_buffer,
        },
        profiler_tensor=profiler_tensor, trace_name=trace_name, spec_decode_config=None,
        use_cutlass_kernel=True,
    )

    x = mpk.attach_input(torch_tensor=input_tokens, name="input_token")
    cos_pos_embed = mpk.attach_input(torch_tensor=position_embeddings[0][0, :4096, :], name="cos_position_embedding")
    sin_pos_embed = mpk.attach_input(torch_tensor=position_embeddings[1][0, :4096, :], name="sin_position_embedding")

    y = mpk.new_tensor(dims=(args.max_num_batched_tokens, hidden_size), dtype=mi.bfloat16, name="embed_out", io_category="cuda_tensor")
    rmsnorm_out = mpk.new_tensor(dims=(args.max_num_batched_tokens, hidden_size), dtype=mi.bfloat16, name="rmsnorm_out", io_category="cuda_tensor")
    attn_in = mpk.new_tensor(dims=(args.max_num_batched_tokens, fused_outdim_1), dtype=mi.bfloat16, name="attn_in", io_category="cuda_tensor")
    attn_out = mpk.new_tensor(dims=(args.max_num_batched_tokens, num_q_heads * head_dim), dtype=mi.bfloat16, name="attn_out", io_category="cuda_tensor")
    attn_proj_out = mpk.new_tensor(dims=(args.max_num_batched_tokens, hidden_size), dtype=mi.bfloat16, name="attn_proj_out", io_category="cuda_tensor")
    mlp_mid = mpk.new_tensor(dims=(args.max_num_batched_tokens, fused_outdim_2), dtype=mi.bfloat16, name="mlp_mid", io_category="cuda_tensor")
    silu_mul_out = mpk.new_tensor(dims=(args.max_num_batched_tokens, intermediate_size), dtype=mi.bfloat16, name="silu_mul_out", io_category="cuda_tensor")
    mlp_out = mpk.new_tensor(dims=(args.max_num_batched_tokens, hidden_size), dtype=mi.bfloat16, name="mlp_out", io_category="cuda_tensor")
    argmax_in = mpk.new_tensor(dims=(args.max_num_batched_tokens, vocab_size), dtype=mi.bfloat16, name="argmax_in", io_category="cuda_tensor")
    argmax_part_value = mpk.new_tensor(dims=(args.max_num_batched_tokens, mpk.num_workers), dtype=mi.bfloat16, name="argmax_part_value", io_category="cuda_tensor")
    argmax_part_index = mpk.new_tensor(dims=(args.max_num_batched_tokens, mpk.num_workers), dtype=mi.int64, name="argmax_part_index", io_category="cuda_tensor")
    argmax_out = mpk.attach_input(torch_tensor=output_tokens, name="output_token")

    w = mpk.attach_input(torch_tensor=model.model.embed_tokens.weight, name="embed_tokens")
    mpk.embed_layer(input=x, weight=w, output=y, grid_dim=(1, 1, 1), block_dim=(128, 1, 1), input_source=1)
    x = y

    target_cc = torch.cuda.get_device_properties(0).major * 10 + torch.cuda.get_device_properties(0).minor
    use_splitk = (target_cc == 100)

    for i, layer in enumerate(model.model.layers):
        w_norm = mpk.attach_input(torch_tensor=layer.input_layernorm.weight, name=f"layer_{i}_input_layernorm")
        w_q = mpk.attach_input(torch_tensor=layer.self_attn.q_proj.weight, name=f"layer_{i}_q_proj")
        w_k = mpk.attach_input(torch_tensor=layer.self_attn.k_proj.weight, name=f"layer_{i}_k_proj")
        w_v = mpk.attach_input(torch_tensor=layer.self_attn.v_proj.weight, name=f"layer_{i}_v_proj")
        w_qkv = mpk.shuffle_tensors(inputs=[w_q, w_k, w_v], shuffled_dim=0, num_groups=model.config.num_key_value_heads, name=f"layer_{i}_qkv_proj")
        mpk.rmsnorm_layer(input=x, weight=w_norm, output=rmsnorm_out, grid_dim=(mpk.max_num_batched_tokens, 1, 1), block_dim=(128, 1, 1))
        mpk.linear_layer(input=rmsnorm_out, weight=w_qkv, output=attn_in, grid_dim=(grid_for_rmsnorm_linear_layer(w_qkv.dim(0)), 1, 1), block_dim=(128, 1, 1))
        w_q_norm = mpk.attach_input(torch_tensor=layer.self_attn.q_norm.weight, name=f"layer_{i}_q_norm")
        w_k_norm = mpk.attach_input(torch_tensor=layer.self_attn.k_norm.weight, name=f"layer_{i}_k_norm")
        k_cache = mpk.attach_input(torch_tensor=model.model.kv_cache[0][i], name=f"layer_{i}_k_cache")
        v_cache = mpk.attach_input(torch_tensor=model.model.kv_cache[1][i], name=f"layer_{i}_v_cache")
        mpk.paged_attention_layer(input=attn_in, k_cache=k_cache, v_cache=v_cache, q_norm=w_q_norm, k_norm=w_k_norm,
                                   cos_pos_embed=cos_pos_embed, sin_pos_embed=sin_pos_embed, output=attn_out,
                                   grid_dim=(mpk.max_num_batched_requests, num_kv_heads, 1), block_dim=(128, 1, 1))
        w = mpk.attach_input(torch_tensor=layer.self_attn.o_proj.weight, name=f"layer_{i}_o_proj")
        if use_splitk:
            attn_proj_out = x
            mpk.splitk_linear_layer(input=attn_out, weight=w, output=attn_proj_out, grid_dim=(hidden_size // 128, 128 * 128 // hidden_size, 1), block_dim=(256, 1, 1))
        else:
            mpk.linear_with_residual_layer(input=attn_out, weight=w, residual=x, output=attn_proj_out, grid_dim=(hidden_size // 64, 1, 1), block_dim=(128, 1, 1))
        x = attn_proj_out

        w_norm = mpk.attach_input(torch_tensor=layer.post_attention_layernorm.weight, name=f"layer_{i}_post_attn_layernorm")
        w_gate_proj = mpk.attach_input(torch_tensor=layer.mlp.gate_proj.weight, name=f"layer_{i}_gate_proj")
        w_up_proj = mpk.attach_input(torch_tensor=layer.mlp.up_proj.weight, name=f"layer_{i}_up_proj")
        rmsnorm_num_tasks = grid_for_rmsnorm_linear_layer(w_gate_proj.dim(0) + w_up_proj.dim(0))
        w_gatedup = mpk.shuffle_tensors(inputs=[w_gate_proj, w_up_proj], shuffled_dim=0, num_groups=rmsnorm_num_tasks // 2, name=f"layer_{i}_gatedup_proj")
        mpk.rmsnorm_layer(input=x, weight=w_norm, output=rmsnorm_out, grid_dim=(mpk.max_num_batched_tokens, 1, 1), block_dim=(128, 1, 1))
        mpk.linear_layer(input=rmsnorm_out, weight=w_gatedup, output=mlp_mid, grid_dim=(rmsnorm_num_tasks, 1, 1), block_dim=(128, 1, 1))
        mpk.silu_mul_layer(input=mlp_mid, output=silu_mul_out, grid_dim=(rmsnorm_num_tasks // 2, 1, 1), block_dim=(128, 1, 1))
        w = mpk.attach_input(torch_tensor=layer.mlp.down_proj.weight, name=f"layer_{i}_down_proj")
        if use_splitk:
            mlp_out = x
            mpk.splitk_linear_layer(input=silu_mul_out, weight=w, output=mlp_out, grid_dim=(hidden_size // 128, 128 * 128 // hidden_size, 1), block_dim=(256, 1, 1))
        else:
            mpk.linear_with_residual_layer(input=silu_mul_out, weight=w, residual=x, output=mlp_out, grid_dim=(hidden_size // 64, 1, 1), block_dim=(128, 1, 1))
        x = mlp_out

    w_norm = mpk.attach_input(torch_tensor=model.model.norm.weight, name="model_norm_weight")
    w_proj = mpk.attach_input(torch_tensor=lm_head_weight, name="lm_head")
    mpk.rmsnorm_layer(input=x, weight=w_norm, output=rmsnorm_out, grid_dim=(mpk.max_num_batched_tokens, 1, 1), block_dim=(128, 1, 1))
    mpk.linear_layer(input=rmsnorm_out, weight=w_proj, output=argmax_in, grid_dim=(mpk.num_workers, 1, 1), block_dim=(128, 1, 1))
    mpk.argmax_partial_layer(input=argmax_in, output=(argmax_part_value, argmax_part_index), grid_dim=(mpk.num_workers, 1, 1), block_dim=(128, 1, 1))
    mpk.argmax_reduce_layer(input=(argmax_part_value, argmax_part_index), output=argmax_out, grid_dim=(1, 1, 1), block_dim=(128, 1, 1))

    print(f"[p9] compiling r={r} (profiler ON) -> {output_dir}", flush=True)
    t0 = time.time()
    mpk.compile(output_dir=output_dir)
    compile_s = time.time() - t0
    print(f"[p9] compile done in {compile_s:.1f}s", flush=True)

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    starter.record()
    mpk()  # both export_* hooks neutralized above; this just runs the kernel
    ender.record()
    torch.cuda.synchronize()
    run_time_ms = starter.elapsed_time(ender)

    seq_len = step[0].item() + 1
    per_tok_ms = run_time_ms / max(seq_len, 1)

    # Save the RAW device buffer ourselves (decoupled from the two broken
    # export paths) so a decode-time bug can never lose the underlying data.
    # p9_decode_tolerant.py turns this into the same CSV schema
    # export_to_csv would have produced, skipping only individually
    # mismatched (block,group,event_idx) pairs.
    raw_path = trace_name + ".rawbuf.pt"
    torch.save(profiler_tensor.cpu(), raw_path)
    print(f"[p9] saved raw profiler buffer: {raw_path}", flush=True)

    summary = {
        "max_num_batched_requests": r,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "max_seq_length": args.max_seq_length,
        "model": args.model,
        "compile_seconds": compile_s,
        "total_time_ms": run_time_ms,
        "sequence_length": seq_len,
        "latency_ms_per_token": per_tok_ms,
        "raw_profiler_buffer_path": raw_path,
        "profiler_slots": args.profiler_slots,
    }
    summary_path = os.path.join(args.out_dir, f"p9_summary_r{r}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print("==================== P9 CAPTURE (profiler ON) ====================")
    print(json.dumps(summary, indent=2))
    print("====================================================================")

    mpk.finalize()


if __name__ == "__main__":
    main()
