# NOTE: Script made using AI
"""
Demo script showing MPK kernel reuse across sessions.

This script demonstrates the kernel caching feature:
1. First run: Compiles the kernel and saves it to output_dir
2. Runs inference with the compiled kernel
3. Simulates a "new session" by creating a fresh PersistentKernel
4. Loads the pre-compiled kernel instead of recompiling
5. Runs inference again with the loaded kernel

This enables efficient multi-user serving where the first request compiles
the kernel once, and subsequent requests reuse it without recompilation.

Usage:
    python demo_kernel_reuse.py --output-dir ./kernel_cache --max-new-tokens 20
"""

import sys
import os
import time
import shutil
import argparse

import torch
from models.modeling_qwen3 import Qwen3ForCausalLM
from transformers import AutoTokenizer, AutoConfig
from safetensors.torch import load_model


def grid_for_rmsnorm_linear_layer(size: int, use_cutlass_kernel: bool = True):
    if size % 64 == 0 and not use_cutlass_kernel:
        return size // 64
    if size / 96 > 400:
        assert size % 256 == 0, f"FATAL: Linear layer size not support, it's {size}."
        return size // 256
    if size % 96 == 0:
        return 96
    elif size % 64 == 0:
        return 64


def max_factor_leq_n(m: int, n: int) -> int:
    max_factor = 1
    i = 1
    while i * i <= m:
        if m % i == 0:
            if i <= n:
                max_factor = max(max_factor, i)
            if m // i <= n:
                max_factor = max(max_factor, m // i)
        i += 1
    return max_factor


def prepare_prompt(tokenizer, prompt_text, tokens, prompt_lengths, total_num_requests):
    """Tokenize a prompt and fill the tokens tensor."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt_text},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to("cuda")
    
    tokens.zero_()
    for r in range(total_num_requests):
        for i in range(model_inputs.input_ids.shape[-1]):
            tokens[r, i] = model_inputs.input_ids[0, i]
    prompt_lengths.fill_(model_inputs.input_ids.shape[-1])
    
    return model_inputs.input_ids.shape[-1]


def build_mpk_graph(
    model, mpk, args, world_size, rank,
    position_embeddings, lm_head_weight, input_tokens, output_tokens,
):
    """Build the full computation graph for inference - matching the original demo exactly."""
    import mirage as mi
    
    hidden_size = model.config.hidden_size
    intermediate_size = model.config.intermediate_size
    vocab_size = 153600
    num_q_heads = model.config.num_attention_heads
    num_kv_heads = model.config.num_key_value_heads
    num_local_q_heads = num_q_heads // world_size
    num_local_kv_heads = num_kv_heads // world_size
    head_dim = model.config.head_dim
    fused_outdim_1 = (num_q_heads + 2 * num_kv_heads) * head_dim
    fused_outdim_2 = 2 * intermediate_size
    
    # Attach input tensors
    x = mpk.attach_input(torch_tensor=input_tokens, name="input_token")
    cos_pos_embed = mpk.attach_input(
        torch_tensor=position_embeddings[0][0, :4096, :],
        name="cos_position_embedding",
    )
    sin_pos_embed = mpk.attach_input(
        torch_tensor=position_embeddings[1][0, :4096, :],
        name="sin_position_embedding",
    )
    
    # Create intermediate tensors
    y = mpk.new_tensor(
        dims=(args.max_num_batched_tokens, hidden_size),
        dtype=mi.bfloat16,
        name="embed_out",
        io_category="cuda_tensor",
    )
    rmsnorm_out = mpk.new_tensor(
        dims=(args.max_num_batched_tokens, hidden_size),
        dtype=mi.bfloat16,
        name="rmsnorm_out",
        io_category="cuda_tensor",
    )
    attn_in = mpk.new_tensor(
        dims=(args.max_num_batched_tokens, fused_outdim_1 // world_size),
        dtype=mi.bfloat16,
        name="attn_in",
        io_category="cuda_tensor",
    )
    attn_out = mpk.new_tensor(
        dims=(args.max_num_batched_tokens, num_local_q_heads * head_dim),
        dtype=mi.bfloat16,
        name="attn_out",
        io_category="cuda_tensor",
    )
    attn_proj_out = mpk.new_tensor(
        dims=(args.max_num_batched_tokens, hidden_size),
        dtype=mi.bfloat16,
        name="attn_proj_out",
        io_category="nvshmem_tensor" if world_size > 1 else "cuda_tensor",
    )
    allreduce_buf = mpk.new_tensor(
        dims=(world_size, args.max_num_batched_tokens, hidden_size),
        dtype=mi.bfloat16,
        name="all_reduce_buf",
        io_category="nvshmem_tensor" if world_size > 1 else "cuda_tensor",
    )
    attn_allreduce_out = mpk.new_tensor(
        dims=(args.max_num_batched_tokens, hidden_size),
        dtype=mi.bfloat16,
        name="attn_allreduce_out",
        io_category="nvshmem_tensor" if world_size > 1 else "cuda_tensor",
    )
    mlp_mid = mpk.new_tensor(
        dims=(args.max_num_batched_tokens, fused_outdim_2 // world_size),
        dtype=mi.bfloat16,
        name="mlp_mid",
        io_category="cuda_tensor",
    )
    silu_mul_out = mpk.new_tensor(
        dims=(args.max_num_batched_tokens, intermediate_size // world_size),
        dtype=mi.bfloat16,
        name="silu_mul_out",
        io_category="cuda_tensor",
    )
    mlp_out = mpk.new_tensor(
        dims=(args.max_num_batched_tokens, hidden_size),
        dtype=mi.bfloat16,
        name="mlp_out",
        io_category="nvshmem_tensor" if world_size > 1 else "cuda_tensor",
    )
    mlp_final = mpk.new_tensor(
        dims=(args.max_num_batched_tokens, hidden_size),
        dtype=mi.bfloat16,
        name="mlp_final",
        io_category="nvshmem_tensor" if world_size > 1 else "cuda_tensor",
    )
    argmax_in = mpk.new_tensor(
        dims=(args.max_num_batched_tokens, vocab_size),
        dtype=mi.bfloat16,
        name="argmax_in",
        io_category="cuda_tensor",
    )
    argmax_part_value = mpk.new_tensor(
        dims=(args.max_num_batched_tokens, mpk.num_workers),
        dtype=mi.bfloat16,
        name="argmax_part_value",
        io_category="cuda_tensor",
    )
    argmax_part_index = mpk.new_tensor(
        dims=(args.max_num_batched_tokens, mpk.num_workers),
        dtype=mi.int64,
        name="argmax_part_index",
        io_category="cuda_tensor",
    )
    argmax_out = mpk.attach_input(torch_tensor=output_tokens, name="output_token")

    # Embed layer
    w = mpk.attach_input(
        torch_tensor=model.model.embed_tokens.weight, name="embed_tokens"
    )
    mpk.embed_layer(
        input=x, 
        weight=w, 
        output=y, 
        grid_dim=(1, 1, 1), 
        block_dim=(128, 1, 1),
        input_source=1,
    )
    x = y
    
    # Transformer layers
    for i, layer in enumerate(model.model.layers):
        # RMSNorm + Linear for QKV
        w_norm = mpk.attach_input(
            torch_tensor=layer.input_layernorm.weight,
            name=f"layer_{i}_input_layernorm",
        )
        w_q = mpk.attach_input(
            torch_tensor=layer.self_attn.q_proj.weight, name=f"layer_{i}_q_proj"
        )
        w_k = mpk.attach_input(
            torch_tensor=layer.self_attn.k_proj.weight, name=f"layer_{i}_k_proj"
        )
        w_v = mpk.attach_input(
            torch_tensor=layer.self_attn.v_proj.weight, name=f"layer_{i}_v_proj"
        )
        
        # Shuffle QKV weights together
        qkv_num_tasks = grid_for_rmsnorm_linear_layer(w_q.dim(0) + w_k.dim(0) + w_v.dim(0), args.use_cutlass_kernel)
        w_qkv = mpk.shuffle_tensors(
            inputs=[w_q, w_k, w_v],
            shuffled_dim=0,
            num_groups=num_local_kv_heads,
            name=f"layer_{i}_qkv_proj",
        )
        
        mpk.rmsnorm_layer(
            input=x,
            weight=w_norm,
            output=rmsnorm_out,
            grid_dim=(mpk.max_num_batched_tokens, 1, 1),
            block_dim=(128, 1, 1),
        )
        mpk.linear_layer(
            input=rmsnorm_out,
            weight=w_qkv,
            output=attn_in,
            grid_dim=(qkv_num_tasks, 1, 1),
            block_dim=(128, 1, 1),
        )
        
        # Attention
        w_q_norm = mpk.attach_input(
            torch_tensor=layer.self_attn.q_norm.weight, name=f"layer_{i}_q_norm"
        )
        w_k_norm = mpk.attach_input(
            torch_tensor=layer.self_attn.k_norm.weight, name=f"layer_{i}_k_norm"
        )
        k_cache = mpk.attach_input(
            torch_tensor=model.model.kv_cache[0][i], name=f"layer_{i}_k_cache"
        )
        v_cache = mpk.attach_input(
            torch_tensor=model.model.kv_cache[1][i], name=f"layer_{i}_v_cache"
        )
        
        mpk.paged_attention_layer(
            input=attn_in,
            k_cache=k_cache,
            v_cache=v_cache,
            q_norm=w_q_norm,
            k_norm=w_k_norm,
            cos_pos_embed=cos_pos_embed,
            sin_pos_embed=sin_pos_embed,
            output=attn_out,
            grid_dim=(mpk.max_num_batched_requests, num_local_kv_heads, 1),
            block_dim=(128, 1, 1),
        )
        
        # O projection with residual
        w_o = mpk.attach_input(
            torch_tensor=layer.self_attn.o_proj.weight, name=f"layer_{i}_o_proj"
        )
        mpk.linear_with_residual_layer(
            input=attn_out,
            weight=w_o,
            residual=x,
            output=attn_proj_out,
            grid_dim=(hidden_size // 64, 1, 1),
            block_dim=(128, 1, 1),
        )
        x = attn_proj_out
        
        # AllReduce if needed
        if world_size > 1:
            mpk.allreduce_layer(
                input=attn_proj_out,
                buffer=allreduce_buf,
                output=attn_allreduce_out,
                grid_dim=(hidden_size // 64, 1, 1),
                block_dim=(128, 1, 1),
            )
            x = attn_allreduce_out
        
        # MLP: RMSNorm + Gate/Up projection
        w_post_norm = mpk.attach_input(
            torch_tensor=layer.post_attention_layernorm.weight,
            name=f"layer_{i}_post_attn_layernorm",
        )
        w_gate = mpk.attach_input(
            torch_tensor=layer.mlp.gate_proj.weight, name=f"layer_{i}_gate_proj"
        )
        w_up = mpk.attach_input(
            torch_tensor=layer.mlp.up_proj.weight, name=f"layer_{i}_up_proj"
        )
        
        rmsnorm_num_tasks = grid_for_rmsnorm_linear_layer(w_gate.dim(0) + w_up.dim(0), args.use_cutlass_kernel)
        w_gatedup = mpk.shuffle_tensors(
            inputs=[w_gate, w_up],
            shuffled_dim=0,
            num_groups=rmsnorm_num_tasks//2,
            name=f"layer_{i}_gatedup_proj",
        )
        
        mpk.rmsnorm_layer(
            input=x,
            weight=w_post_norm,
            output=rmsnorm_out,
            grid_dim=(mpk.max_num_batched_tokens, 1, 1),
            block_dim=(128, 1, 1),
        )
        mpk.linear_layer(
            input=rmsnorm_out,
            weight=w_gatedup,
            output=mlp_mid,
            grid_dim=(rmsnorm_num_tasks, 1, 1),
            block_dim=(128, 1, 1),
        )
        
        # SiLU + Mul
        mpk.silu_mul_layer(
            input=mlp_mid,
            output=silu_mul_out,
            grid_dim=(rmsnorm_num_tasks//2, 1, 1),
            block_dim=(128, 1, 1),
        )
        
        # Down projection with residual
        w_down = mpk.attach_input(
            torch_tensor=layer.mlp.down_proj.weight, name=f"layer_{i}_down_proj"
        )
        mpk.linear_with_residual_layer(
            input=silu_mul_out,
            weight=w_down,
            residual=x,
            output=mlp_out,
            grid_dim=(hidden_size // 64, 1, 1),
            block_dim=(128, 1, 1),
        )
        x = mlp_out
        
        # AllReduce if needed
        if world_size > 1:
            mpk.allreduce_layer(
                input=mlp_out,
                buffer=allreduce_buf,
                output=mlp_final,
                grid_dim=(hidden_size // 64, 1, 1),
                block_dim=(128, 1, 1),
            )
            x = mlp_final
    
    # Final RMSNorm + LM Head
    w_final_norm = mpk.attach_input(
        torch_tensor=model.model.norm.weight, name="model_norm_weight"
    )
    w_lm_head = mpk.attach_input(torch_tensor=lm_head_weight, name="lm_head")
    
    mpk.rmsnorm_layer(
        input=x,
        weight=w_final_norm,
        output=rmsnorm_out,
        grid_dim=(mpk.max_num_batched_tokens, 1, 1),
        block_dim=(128, 1, 1),
    )
    mpk.linear_layer(
        input=rmsnorm_out,
        weight=w_lm_head,
        output=argmax_in,
        grid_dim=(mpk.num_workers, 1, 1),
        block_dim=(128, 1, 1),
    )
    
    # Argmax
    argmax_partial_grid_dim = (mpk.num_workers, 1, 1)
    argmax_reduce_grid_dim = (1, 1, 1)
    mpk.argmax_partial_layer(
        input=argmax_in,
        output=(argmax_part_value, argmax_part_index),
        grid_dim=argmax_partial_grid_dim,
        block_dim=(128, 1, 1),
    )
    mpk.argmax_reduce_layer(
        input=(argmax_part_value, argmax_part_index),
        output=argmax_out,
        grid_dim=argmax_reduce_grid_dim,
        block_dim=(128, 1, 1),
    )
    
    return mpk


def create_mpk(model, args, world_size, rank, meta_tensors):
    """Create a PersistentKernel instance."""
    import mirage as mi
    
    num_workers, num_schedulers = mi.get_configurations_from_gpu(rank)
    
    mpk = mi.PersistentKernel(
        mode="offline",
        world_size=world_size,
        mpi_rank=rank,
        num_workers=num_workers,
        num_local_schedulers=num_schedulers,
        num_remote_schedulers=0,
        max_seq_length=args.max_seq_length,
        max_num_batched_requests=args.max_num_batched_requests,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_pages=args.max_num_pages,
        page_size=args.page_size,
        eos_token_id=model.config.eos_token_id,
        meta_tensors=meta_tensors,
        profiler_tensor=None,
        trace_name="",
        spec_decode_config=None,
        use_cutlass_kernel=args.use_cutlass_kernel,
    )
    
    return mpk


def run_inference(mpk, tokens, step, tokenizer, prompt_lengths):
    """Run inference and return the response."""
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    
    starter.record()
    mpk()
    ender.record()
    torch.cuda.synchronize()
    run_time = starter.elapsed_time(ender)
    
    generated_ids = tokens[0, : step[0] + 1]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    prompt_len = prompt_lengths[0].item()
    gen_len = step[0].item() + 1 - prompt_len
    per_token_latency = run_time / max(1, step[0].item() + 1)
    
    return response, run_time, per_token_latency, gen_len


def main():
    parser = argparse.ArgumentParser(description="Demo: MPK kernel compile-then-load reuse")
    parser.add_argument("--max-num-batched-tokens", default=8, type=int)
    parser.add_argument("--max-num-batched-requests", default=1, type=int)
    parser.add_argument("--page-size", default=4096, type=int)
    parser.add_argument("--max-num-pages", default=16, type=int)
    parser.add_argument("--output-dir", default="./kernel_cache", help="Directory to store compiled kernels")
    parser.add_argument("--max-seq-length", default=512, type=int)
    parser.add_argument("--model", type=str, default='Qwen/Qwen3-8B')
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--use-cutlass-kernel", action="store_true", default=True)
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--prompt1", type=str, default="What is the capital of France?")
    parser.add_argument("--prompt2", type=str, default="Explain quantum computing in simple terms.")
    args = parser.parse_args()

    world_size = 1
    rank = 0
    
    print("=" * 70)
    print("MPK Kernel Reuse Demo")
    print("=" * 70)
    print(f"Output directory: {args.output_dir}")
    print()

    torch.set_default_dtype(torch.bfloat16)
    torch.cuda.set_device(rank)

    # Load model
    print("Loading model...")
    with torch.device("cuda"):
        if args.model_path:
            config = AutoConfig.from_pretrained(args.model_path)
            model = Qwen3ForCausalLM(config, world_size, args.max_num_pages, args.page_size)
            load_model(model, f"{args.model_path}/model{rank}-mp{world_size}.safetensors")
            tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        else:
            model = Qwen3ForCausalLM.from_pretrained(
                args.model, world_size, 
                max_num_pages=args.max_num_pages, 
                page_size=args.page_size
            ).to("cuda")
            tokenizer = AutoTokenizer.from_pretrained(args.model)
    print("Model loaded.")

    # Prepare tensors
    hidden_size = model.config.hidden_size
    lm_head_weight = torch.cat(
        (model.lm_head.weight,
         torch.full((153600 - model.config.vocab_size, hidden_size), 0, device="cuda")),
        0,
    )
    
    positions = torch.arange(32768).unsqueeze(0).to("cuda")
    position_embeddings = model.model.rotary_emb(positions)
    
    total_num_requests = args.max_num_batched_requests
    tokens = torch.full((total_num_requests, args.max_seq_length), 0, dtype=torch.long, device="cuda")
    input_tokens = torch.full((args.max_num_batched_tokens, 1), 0, dtype=torch.long, device="cuda")
    output_tokens = torch.full((args.max_num_batched_tokens, 1), 0, dtype=torch.long, device="cuda")
    step = torch.full((total_num_requests,), 0, dtype=torch.int32, device="cuda")
    num_new_tokens = torch.full((total_num_requests,), 1, dtype=torch.int32, device="cuda")
    prompt_lengths = torch.full((total_num_requests,), 0, dtype=torch.int32, device="cuda")
    qo_indptr_buffer = torch.empty(args.max_num_batched_requests + 1, dtype=torch.int32, device="cuda")
    paged_kv_indptr_buffer = torch.empty(args.max_num_batched_requests + 1, dtype=torch.int32, device="cuda")
    paged_kv_indices_buffer = torch.empty(args.max_num_pages, dtype=torch.int32, device="cuda")
    paged_kv_last_page_len_buffer = torch.empty(args.max_num_batched_requests, dtype=torch.int32, device="cuda")

    meta_tensors = {
        "step": step,
        "tokens": tokens,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "num_new_tokens": num_new_tokens,
        "prompt_lengths": prompt_lengths,
        "qo_indptr_buffer": qo_indptr_buffer,
        "paged_kv_indptr_buffer": paged_kv_indptr_buffer,
        "paged_kv_indices_buffer": paged_kv_indices_buffer,
        "paged_kv_last_page_len_buffer": paged_kv_last_page_len_buffer,
    }

    # Clean output directory
    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    # ========================================
    # PHASE 1: Compile and run first prompt
    # ========================================
    print("\n" + "=" * 70)
    print("PHASE 1: Compiling kernel and running first prompt")
    print("=" * 70)
    
    prompt_len = prepare_prompt(tokenizer, args.prompt1, tokens, prompt_lengths, total_num_requests)
    step.zero_()
    print(f"Prompt 1: \"{args.prompt1}\"")
    print(f"Prompt length: {prompt_len} tokens")
    print()
    
    compile_start = time.time()
    
    print("Creating PersistentKernel and building computation graph...")
    mpk1 = create_mpk(model, args, world_size, rank, meta_tensors)
    build_mpk_graph(model, mpk1, args, world_size, rank, position_embeddings, lm_head_weight, input_tokens, output_tokens)
    
    print("Compiling kernel...")
    mpk1.compile(output_dir=args.output_dir)
    compile_time = time.time() - compile_start
    print(f"Kernel compiled and saved to: {args.output_dir}")
    print(f"Compilation time: {compile_time:.1f} seconds")
    
    # Run inference
    print("\nRunning inference...")
    response1, time1, latency1, gen_len1 = run_inference(mpk1, tokens, step, tokenizer, prompt_lengths)
    
    print(f"\n--- Response 1 ---")
    print(response1)
    print(f"\nGenerated {gen_len1} tokens, latency: {latency1:.3f} ms/token")
    
    # Finalize first kernel
    mpk1.finalize()
    del mpk1
    
    # Reset KV caches
    model.model.kv_cache[0].zero_()
    model.model.kv_cache[1].zero_()
    
    # ========================================
    # PHASE 2: Load pre-compiled kernel
    # ========================================
    print("\n" + "=" * 70)
    print("PHASE 2: Loading pre-compiled kernel and running second prompt")
    print("=" * 70)
    
    prompt_len = prepare_prompt(tokenizer, args.prompt2, tokens, prompt_lengths, total_num_requests)
    step.zero_()
    print(f"Prompt 2: \"{args.prompt2}\"")
    print(f"Prompt length: {prompt_len} tokens")
    print()
    
    load_start = time.time()
    
    print("Creating new PersistentKernel instance...")
    mpk2 = create_mpk(model, args, world_size, rank, meta_tensors)
    build_mpk_graph(model, mpk2, args, world_size, rank, position_embeddings, lm_head_weight, input_tokens, output_tokens)
    
    print(f"Loading pre-compiled kernel from: {args.output_dir}")
    mpk2.load_mpk_kernel(output_dir=args.output_dir)
    load_time = time.time() - load_start
    print(f"Kernel loaded in: {load_time:.2f} seconds")
    
    # Run inference
    print("\nRunning inference...")
    response2, time2, latency2, gen_len2 = run_inference(mpk2, tokens, step, tokenizer, prompt_lengths)
    
    print(f"\n--- Response 2 ---")
    print(response2)
    print(f"\nGenerated {gen_len2} tokens, latency: {latency2:.3f} ms/token")
    
    mpk2.finalize()
    
    # ========================================
    # PHASE 3: Test Compatibility Validation
    # ========================================
    print("\n" + "=" * 70)
    print("PHASE 3: Testing Compatibility Validation")
    print("=" * 70)
    
    print("Testing that mismatched configuration is detected...")
    
    # Create a PersistentKernel with different max_num_batched_requests
    import copy
    args_mismatch = copy.copy(args)
    args_mismatch.max_num_batched_requests = args.max_num_batched_requests + 1  # Different from compiled kernel
    
    # We need to create new meta tensors with the mismatched size
    meta_tensors_mismatch = {
        "step": torch.full((args_mismatch.max_num_batched_requests,), 0, dtype=torch.int32, device="cuda"),
        "tokens": torch.full((args_mismatch.max_num_batched_requests, args.max_seq_length), 0, dtype=torch.long, device="cuda"),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "num_new_tokens": torch.full((args_mismatch.max_num_batched_requests,), 1, dtype=torch.int32, device="cuda"),
        "prompt_lengths": torch.full((args_mismatch.max_num_batched_requests,), 0, dtype=torch.int32, device="cuda"),
        "qo_indptr_buffer": torch.empty(args_mismatch.max_num_batched_requests + 1, dtype=torch.int32, device="cuda"),
        "paged_kv_indptr_buffer": torch.empty(args_mismatch.max_num_batched_requests + 1, dtype=torch.int32, device="cuda"),
        "paged_kv_indices_buffer": paged_kv_indices_buffer,
        "paged_kv_last_page_len_buffer": torch.empty(args_mismatch.max_num_batched_requests, dtype=torch.int32, device="cuda"),
    }
    
    try:
        mpk3 = create_mpk(model, args_mismatch, world_size, rank, meta_tensors_mismatch)
        build_mpk_graph(model, mpk3, args_mismatch, world_size, rank, position_embeddings, lm_head_weight, input_tokens, output_tokens)
        mpk3.load_mpk_kernel(output_dir=args.output_dir)
        print("ERROR: Expected validation to fail but it passed!")
        mpk3.finalize()
    except ValueError as e:
        print("SUCCESS: Compatibility validation correctly detected mismatch:")
        print(f"  {str(e)[:200]}...")
    except Exception as e:
        print(f"Unexpected error type: {type(e).__name__}: {e}")
    
    # ========================================
    # Summary
    # ========================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Phase 1 - Compile time:  {compile_time:.1f} seconds")
    print(f"Phase 2 - Load time:     {load_time:.2f} seconds")
    print(f"Speedup:                 {compile_time/load_time:.1f}x faster")
    print("=" * 70)


if __name__ == "__main__":
    main()
