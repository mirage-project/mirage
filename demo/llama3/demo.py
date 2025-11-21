import argparse
import os
import torch
import torch.distributed as dist
from transformers import AutoTokenizer, AutoConfig
from safetensors.torch import load_model

from models.modeling_llama3 import Llama3ForCausalLM


# ============================================================================
# Configuration and Setup Functions
# ============================================================================

def parse_arguments():
    parser = argparse.ArgumentParser(description="Llama3 MPK Demo")
    
    parser.add_argument("--use-mirage", action="store_true", 
                       help="Use Mirage kernels")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output files directory")
    parser.add_argument("--profiling", action="store_true", 
                       help="Use Profiler to generate trace")
    parser.add_argument("--trace-name", default="",
                       help="Perfetto trace output name")
    
    parser.add_argument("--model-path", type=str, default=None,
                       help="Path to a local model (necessary for multi-GPU demo)")
    parser.add_argument("--model", type=str, default='meta-llama/Meta-Llama-3-8B-Instruct',
                       help="Model path on hugging face")
    
    parser.add_argument("--max-num-batched-tokens", type=int, default=8,
                       help="Max number of tokens in a batch")
    parser.add_argument("--max-num-batched-requests", default=4, type=int,
                       help="Max number of requests in a batch")
    parser.add_argument("--page-size", default=4096, type=int,
                       help="Page size")
    parser.add_argument("--max-num-pages", default=16, type=int,
                       help="Max num pages")

    parser.add_argument("--spec-decode", default=None,
                       choices=["promptlookup", "lookahead"],
                       help="Enable speculative decoding with 'lookahead' or 'promptlookup' mode.")
    parser.add_argument("--ngram-size", default=3, type=int,
                       help="Ngram size for lookahead spec decode")
    parser.add_argument("--max-seq-length", default=512, type=int,
                       help="Max sequence length for spec decode")
    parser.add_argument("--spec-length", default=3, type=int,
                       help="Spec length for spec decode")
    parser.add_argument(
        "--no-use-cutlass-kernel",
        action="store_false",
        dest="use_cutlass_kernel",
        default=True,
        help="Not use the cutlass version kernel.",
    )
    
    return parser.parse_args()


def setup_env():
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        world_size = comm.Get_size()
        rank = comm.Get_rank()
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
    except ImportError:
        world_size = 1
        rank = 0
    
    if world_size > 1:
        dist.init_process_group(backend="nccl", init_method="env://")

    torch.set_default_dtype(torch.bfloat16)
    torch.cuda.set_device(rank)

    return world_size, rank


def load_model_and_tokenizer(args, world_size, rank):
    model_name = args.model
    
    with torch.device("cuda"):
        if args.model_path is not None:
            # Load model locally (necessary for multi-GPU case)
            print(f"Load model from model path: {args.model_path}")
            config = AutoConfig.from_pretrained(args.model_path)
            model = Llama3ForCausalLM(config, world_size, args.max_num_pages, args.page_size)
            load_model(
                model, f"{args.model_path}/model{rank}-mp{world_size}.safetensors"
            )
            tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        else:
            model = Llama3ForCausalLM.from_pretrained(model_name, world_size, max_num_pages=args.max_num_pages, page_size=args.page_size).to("cuda")

            tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    return model, tokenizer


def process_eos_tokens(model, tokenizer):
    # currently, we only support single eos_token_id in mpk
    eos_token_id_for_mirage = tokenizer.eos_token_id
    eos_token_ids = [eos_token_id_for_mirage]
    return eos_token_id_for_mirage, eos_token_ids


# ============================================================================
# Utility Functions
# ============================================================================

def get_block_dim():
    return 128


def grid_for_rmsnorm_linear_layer(size):
    if size / 96 > 400:
        # TODO: An add-hoc workaround for linear kernel, both MPK ptx and
        # cutlass version will output unexpect result (not same out put for
        # same prompt) if the OUTPUT_SIZE is too big, try to figure it out.
        assert size % 256 == 0, f"FATAL: Linear layer size not support, it's {size}."
        return size // 256
    for grid_size in [96, 80, 64, 48]:
        if size % grid_size == 0 and (size // grid_size >= 128 or size // grid_size == 64):
            return grid_size
    raise ValueError(f"FATAL: RMSNorm/Linear layer size not support, it's {size}.")


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


def prepare_test_prompt():
    prompt = "Give me a short introduction to large language model."
    
    messages = [
        {
            "role": "system",
            "content": "You are llama, a helpful AI assistant.",
        },
        {"role": "user", "content": prompt},
    ]
    
    return messages


# ============================================================================
# Input Preparation Functions
# ============================================================================

def prepare_input_tensors(model, tokenizer, messages, args, use_mirage=True):
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    print("Model input id shape:", model_inputs.input_ids.shape)
    num_requests = args.max_num_batched_requests if use_mirage else 1
    
    # Prepare tokens tensor
    tokens = torch.full((num_requests, args.max_seq_length), 0, dtype=torch.long, device="cuda")
    for r in range(num_requests):
        for i in range(model_inputs.input_ids.shape[-1]):
            tokens[r, i] = model_inputs.input_ids[0, i]
    prompt_lengths = torch.full((num_requests,), model_inputs.input_ids.shape[-1], dtype=torch.int, device="cuda")

    # Prepare position embeddings
    positions = torch.arange(32768).unsqueeze(0).to(model.device)
    position_embeddings = model.model.rotary_emb(positions)
    
    # Prepare control tensors
    input_tokens = torch.full((args.max_num_batched_tokens, 1), 0, dtype=torch.long, device="cuda")
    output_tokens = torch.full((args.max_num_batched_tokens, 1), 0, dtype=torch.long, device="cuda")
    step = torch.full((num_requests, ), 0, dtype=torch.int32, device="cuda")
    num_new_tokens = torch.full((num_requests, ), 1, dtype=torch.int32, device="cuda")

    return {
        'tokens': tokens,
        'prompt_lengths': prompt_lengths,
        'position_embeddings': position_embeddings,
        'input_tokens': input_tokens,
        'output_tokens': output_tokens,
        'step': step,
        'num_new_tokens': num_new_tokens
    }
        

# ============================================================================
# Mirage MPK Setup Functions
# ============================================================================

def setup_mirage_configuration(model, args, world_size, rank):
    hidden_size = model.config.hidden_size
    intermediate_size = model.config.intermediate_size
    # Pad vocab_size to facilitate task graph creation
    padded_vocab_size = ((model.config.vocab_size + 2047) // 2048) * 2048
    lm_head_weight = torch.cat(
            (
                model.lm_head.weight,
                torch.zeros(
                    (padded_vocab_size - model.config.vocab_size, hidden_size), 
                    dtype=torch.bfloat16, 
                    device="cuda"
                ),
            ),
            0,
        )
    assert lm_head_weight.stride()[0] == hidden_size
    # Calculate head dimensions
    num_q_heads = model.config.num_attention_heads
    num_kv_heads = model.config.num_key_value_heads
    num_local_q_heads = num_q_heads // world_size
    num_local_kv_heads = num_kv_heads // world_size
    head_dim = model.config.head_dim
    fused_outdim_1 = (num_q_heads + 2 * num_kv_heads) * head_dim
    fused_outdim_2 = 2 * intermediate_size

    print("\nMirage Model Config:")
    print(f"  hidden_size: {hidden_size}")
    print(f"  intermediate_size: {intermediate_size}")
    print(f"  num_attention_heads: {num_q_heads}")
    print(f"  num_key_value_heads: {num_kv_heads}")
    print(f"  head_dim: {head_dim}")
    print(f"  fused_outdim_1: {fused_outdim_1}")
    print(f"  fused_outdim_2: {fused_outdim_2}")
    print(f"  padded_vocab_size: {padded_vocab_size}")
    print(f"  world_size: {world_size}")
    print(f"  num_local_q_heads: {num_local_q_heads}")
    print(f"  num_local_kv_heads: {num_local_kv_heads}\n\n")

    return {
        'hidden_size': hidden_size,
        'intermediate_size': intermediate_size,
        'padded_vocab_size': padded_vocab_size,
        'lm_head_weight': lm_head_weight,
        'num_q_heads': num_q_heads,
        'num_kv_heads': num_kv_heads,
        'num_local_q_heads': num_local_q_heads,
        'num_local_kv_heads': num_local_kv_heads,
        'head_dim': head_dim,
        'fused_outdim_1': fused_outdim_1,
        'fused_outdim_2': fused_outdim_2,
    }


def create_persistent_kernel(args, world_size, rank, input_data, config, eos_token_id_for_mirage):
    import mirage as mi

    if args.profiling:
        block_dim_for_profiler = get_block_dim()
        profiler_tensor = torch.zeros(
            3000 * block_dim_for_profiler, dtype=torch.uint64, device="cuda"
        ).contiguous()
    else:
        profiler_tensor = None
    
    # Setup speculative decoding configuration
    spec_decode_config = mi.speculative.spec_decode_class(
        args.spec_decode,
        ngram_size=args.ngram_size,
        spec_length=args.spec_length,
    )
    
    # Create auxiliary buffers for paged KV and QO
    qo_indptr_buffer = torch.empty(
            args.max_num_batched_requests + 1, dtype=torch.int32, device="cuda")
    paged_kv_indptr_buffer = torch.empty(
        args.max_num_batched_requests + 1, dtype=torch.int32, device="cuda")
    paged_kv_indices_buffer = torch.empty(
        args.max_num_pages, dtype=torch.int32, device="cuda")
    paged_kv_last_page_len_buffer = torch.empty(
        args.max_num_batched_requests, dtype=torch.int32, device="cuda")

    # Get GPU configurations
    num_workers, num_schedulers = mi.get_configurations_from_gpu(rank)
    
    # Create persistent kernel
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
        eos_token_id=eos_token_id_for_mirage,
        meta_tensors={
                "step": input_data['step'],
                "tokens": input_data['tokens'],
                "input_tokens": input_data['input_tokens'],
                "output_tokens": input_data['output_tokens'],
                "num_new_tokens": input_data['num_new_tokens'],
                "prompt_lengths": input_data['prompt_lengths'],
                "qo_indptr_buffer": qo_indptr_buffer,
                "paged_kv_indptr_buffer": paged_kv_indptr_buffer,
                "paged_kv_indices_buffer": paged_kv_indices_buffer,
                "paged_kv_last_page_len_buffer": paged_kv_last_page_len_buffer,
            },
        profiler_tensor=profiler_tensor,
        trace_name=args.trace_name,
        spec_decode_config=spec_decode_config,

        use_cutlass_kernel=args.use_cutlass_kernel,
    )
    
    return mpk, spec_decode_config


def attach_model_inputs(mpk, model, input_data, config):
    # Attach inputs
    x = mpk.attach_input(torch_tensor=input_data['input_tokens'], name="input_token")
    cos_pos_embed = mpk.attach_input(
        torch_tensor=input_data['position_embeddings'][0][0, :4096, :],
        name="cos_position_embedding",
    )
    sin_pos_embed = mpk.attach_input(
        torch_tensor=input_data['position_embeddings'][1][0, :4096, :],
        name="sin_position_embedding",
    )
    
    # Attach embedding weights
    embed_weight = mpk.attach_input(
        torch_tensor=model.model.embed_tokens.weight, 
        name="embed_tokens"
    )
    
    return {
        'input_token': x,
        'cos_pos_embed': cos_pos_embed,
        'sin_pos_embed': sin_pos_embed,
        'embed_weight': embed_weight,
    }


def create_intermediate_tensors(mpk, config, spec_decode_config, world_size, args):
    import mirage as mi
    
    tensors = {}
    
    # Embedding output
    tensors['embed_out'] = mpk.new_tensor(
        dims=(args.max_num_batched_tokens, config['hidden_size']),
        dtype=mi.bfloat16,
        name="embed_out",
        io_category="cuda_tensor",
    )

    # RMSNorm output tensor
    tensors['rmsnorm_out'] = mpk.new_tensor(
        dims=(args.max_num_batched_tokens, config['hidden_size']),
        dtype=mi.bfloat16,
        name="rmsnorm_out",
        io_category="cuda_tensor",
    )
    
    # Attention tensors
    tensors['attn_in'] = mpk.new_tensor(
        dims=(args.max_num_batched_tokens, config['fused_outdim_1'] // world_size),
        dtype=mi.bfloat16,
        name="attn_in",
        io_category="cuda_tensor",
    )
    
    tensors['attn_out'] = mpk.new_tensor(
        dims=(args.max_num_batched_tokens, config['num_local_q_heads'] * config['head_dim']),
        dtype=mi.bfloat16,
        name="attn_out",
        io_category="cuda_tensor",
    )
    
    # Attention projection output
    tensors['attn_proj_out'] = mpk.new_tensor(
        dims=(args.max_num_batched_tokens, config['hidden_size']),
        dtype=mi.bfloat16,
        name="attn_proj_out",
        io_category="nvshmem_tensor" if world_size > 1 else "cuda_tensor",
    )
    
    # Allreduce tensors for multi-GPU
    tensors['allreduce_buf'] = mpk.new_tensor(
        dims=(world_size, args.max_num_batched_tokens, config['hidden_size']),
        dtype=mi.bfloat16,
        name="all_reduce_buf",
        io_category="nvshmem_tensor" if world_size > 1 else "cuda_tensor",
    )
    
    tensors['attn_allreduce_out'] = mpk.new_tensor(
        dims=(args.max_num_batched_tokens, config['hidden_size']),
        dtype=mi.bfloat16,
        name="attn_allreduce_out",
        io_category="nvshmem_tensor" if world_size > 1 else "cuda_tensor",
    )
    
    # MLP tensors
    tensors['mlp_mid'] = mpk.new_tensor(
        dims=(args.max_num_batched_tokens, config['fused_outdim_2'] // world_size),
        dtype=mi.bfloat16,
        name="mlp_mid",
        io_category="cuda_tensor",
    )

    # MLP SiLU output tensor
    tensors['silu_mul_out'] = mpk.new_tensor(
        dims=(args.max_num_batched_tokens, config['intermediate_size'] // world_size),
        dtype=mi.bfloat16,
        name="silu_mul_out",
        io_category="cuda_tensor",
    )
    
    tensors['mlp_out'] = mpk.new_tensor(
        dims=(args.max_num_batched_tokens, config['hidden_size']),
        dtype=mi.bfloat16,
        name="mlp_out",
        io_category="nvshmem_tensor" if world_size > 1 else "cuda_tensor",
    )
    
    tensors['mlp_final'] = mpk.new_tensor(
        dims=(args.max_num_batched_tokens, config['hidden_size']),
        dtype=mi.bfloat16,
        name="mlp_final",
        io_category="nvshmem_tensor" if world_size > 1 else "cuda_tensor",
    )
    
    # Final layer and argmax tensors
    tensors['argmax_in'] = mpk.new_tensor(
        dims=(args.max_num_batched_tokens, config['padded_vocab_size']),
        dtype=mi.bfloat16,
        name="argmax_in",
        io_category="cuda_tensor",
    )
    
    tensors['argmax_part_value'] = mpk.new_tensor(
        dims=(args.max_num_batched_tokens, mpk.num_workers),
        dtype=mi.bfloat16,
        name="argmax_part_value",
        io_category="cuda_tensor",
    )
    
    tensors['argmax_part_index'] = mpk.new_tensor(
        dims=(args.max_num_batched_tokens, mpk.num_workers),
        dtype=mi.int64,
        name="argmax_part_index",
        io_category="cuda_tensor",
    )

    # Use output_tokens as the argmax output
    tensors['argmax_out'] = mpk.attach_input(torch_tensor=mpk.meta_tensors["output_tokens"], name="output_token")
    
    return tensors


# ============================================================================
# Mirage Layer Construction Functions
# ============================================================================

def add_embedding_layer(mpk, inputs, tensors, config, spec_decode_config):
    # Add embedding layer
    embed_block_dim = get_block_dim()
    mpk.embed_layer(
        input=inputs['input_token'], 
        weight=inputs['embed_weight'], 
        output=tensors['embed_out'], 
        grid_dim=(1,1,1),
        block_dim=(embed_block_dim, 1, 1),
        input_source=1,
    )
    
    return tensors['embed_out']


def add_transformer_layer(mpk, layer_idx, layer, x, inputs, tensors, config, world_size, spec_decode_config, model, args):
    import mirage as mi
    
    # 1. Input layernorm + QKV projection
    w_norm = mpk.attach_input(
        torch_tensor=layer.input_layernorm.weight,
        name=f"layer_{layer_idx}_input_layernorm",
    )
    w_q = mpk.attach_input(
        torch_tensor=layer.self_attn.q_proj.weight, 
        name=f"layer_{layer_idx}_q_proj"
    )
    w_k = mpk.attach_input(
        torch_tensor=layer.self_attn.k_proj.weight, 
        name=f"layer_{layer_idx}_k_proj"
    )
    w_v = mpk.attach_input(
        torch_tensor=layer.self_attn.v_proj.weight, 
        name=f"layer_{layer_idx}_v_proj"
    )

    # Shuffle QKV weights for grouped query attention
    w_qkv = mpk.shuffle_tensors(
        inputs=[w_q, w_k, w_v],
        shuffled_dim=0,
        num_groups=config['num_kv_heads'] // world_size,
        name=f"layer_{layer_idx}_qkv_proj",
    )
    
    # RMSNorm + Linear layers
    rmsnorm_block_dim = get_block_dim()
    mpk.rmsnorm_layer(
        input=x,
        weight=w_norm,
        output=tensors['rmsnorm_out'],
        grid_dim=(args.max_num_batched_tokens, 1, 1),
        block_dim=(rmsnorm_block_dim, 1, 1),
    )

    attn_in_block_dim = get_block_dim()
    mpk.linear_layer(
        input=tensors['rmsnorm_out'],
        weight=w_qkv,
        output=tensors['attn_in'],
        grid_dim=(grid_for_rmsnorm_linear_layer(w_qkv.dim(0)), 1, 1),
        block_dim=(attn_in_block_dim, 1, 1),
    )
    
    # 2. Attention computation (Llama3 doesn't use q_norm/k_norm)
    k_cache = mpk.attach_input(
        torch_tensor=model.model.kv_cache[0][layer_idx], 
        name=f"layer_{layer_idx}_k_cache"
    )
    v_cache = mpk.attach_input(
        torch_tensor=model.model.kv_cache[1][layer_idx], 
        name=f"layer_{layer_idx}_v_cache"
    )
    
    attn_out_block_dim = get_block_dim()
    if spec_decode_config:
        mpk.single_batch_extend_attention_layer(
            input=tensors['attn_in'],
            k_cache=k_cache,
            v_cache=v_cache,
            q_norm=None,
            k_norm=None,
            cos_pos_embed=inputs['cos_pos_embed'],
            sin_pos_embed=inputs['sin_pos_embed'],
            output=tensors['attn_out'],
            grid_dim=(1, config['num_local_kv_heads'], 1),
            block_dim=(attn_out_block_dim, 1, 1),
        )
    else:
        mpk.paged_attention_layer(
            input=tensors['attn_in'],
            k_cache=k_cache,
            v_cache=v_cache,
            q_norm=None,
            k_norm=None,
            cos_pos_embed=inputs['cos_pos_embed'],
            sin_pos_embed=inputs['sin_pos_embed'],
            output=tensors['attn_out'],
            grid_dim=(args.max_num_batched_requests, config['num_local_kv_heads'], 1),
            block_dim=(attn_out_block_dim, 1, 1),
        )
 
    # 3. Attention output projection with residual
    w_o = mpk.attach_input(
        torch_tensor=layer.self_attn.o_proj.weight, 
        name=f"layer_{layer_idx}_o_proj"
    )
    attn_proj_out_dim = config['hidden_size']
    attn_proj_block_dim = get_block_dim()
    mpk.linear_with_residual_layer(
        input=tensors['attn_out'],
        weight=w_o,
        residual=x,
        output=tensors['attn_proj_out'],
        grid_dim=(attn_proj_out_dim // 64, 1, 1),
        block_dim=(attn_proj_block_dim, 1, 1),
    )
    
    # Update x to attention output
    x = tensors['attn_proj_out']
    
    # 4. Allreduce if multi-GPU
    if world_size > 1:
        mpk.allreduce_layer(
            input=tensors['attn_proj_out'],
            buffer=tensors['allreduce_buf'],
            output=tensors['attn_allreduce_out'],
            grid_dim=(config['hidden_size'] // 64, 1, 1),
            block_dim=(128, 1, 1),
        )
        x = tensors['attn_allreduce_out']
    
    # 5. Post-attention layernorm + MLP
    w_norm_mlp = mpk.attach_input(
        torch_tensor=layer.post_attention_layernorm.weight,
        name=f"layer_{layer_idx}_post_attn_layernorm",
    )
    w_gate_proj = mpk.attach_input(
        torch_tensor=layer.mlp.gate_proj.weight, 
        name=f"layer_{layer_idx}_gate_proj"
    )
    w_up_proj = mpk.attach_input(
        torch_tensor=layer.mlp.up_proj.weight, 
        name=f"layer_{layer_idx}_up_proj"
    )

    rmsnorm_num_tasks = grid_for_rmsnorm_linear_layer(w_gate_proj.dim(0) + w_up_proj.dim(0))
    # Shuffle gate and up projections
    w_gatedup = mpk.shuffle_tensors(
        inputs=[w_gate_proj, w_up_proj],
        shuffled_dim=0,
        num_groups=rmsnorm_num_tasks//2,
        name=f"layer_{layer_idx}_gatedup_proj",
    )
     
    # RMSNorm + Linear for MLP
    # MLP RMSNorm
    mlp_rmsnorm_block_dim = get_block_dim()
    mpk.rmsnorm_layer(
        input=x,
        weight=w_norm_mlp,
        output=tensors['rmsnorm_out'],
        grid_dim=(args.max_num_batched_tokens, 1, 1),
        block_dim=(mlp_rmsnorm_block_dim, 1, 1),
    )
    # MLP Linear
    mlp_mid_block_dim = get_block_dim()
    mpk.linear_layer(
        input=tensors['rmsnorm_out'],
        weight=w_gatedup,
        output=tensors['mlp_mid'],
        grid_dim=(rmsnorm_num_tasks, 1, 1),
        block_dim=(mlp_mid_block_dim, 1, 1),
    )
    # SiLU Mul
    silu_mul_block_dim = get_block_dim()
    mpk.silu_mul_layer(
        input=tensors['mlp_mid'],
        output=tensors['silu_mul_out'],
        grid_dim=(rmsnorm_num_tasks//2, 1, 1),
        block_dim=(silu_mul_block_dim, 1, 1),
    )
    
    # 6. MLP computation with residual
    w_down = mpk.attach_input(
        torch_tensor=layer.mlp.down_proj.weight, 
        name=f"layer_{layer_idx}_down_proj"
    )
    mlp_out_dim = config['hidden_size']
    mlp_out_block_dim = get_block_dim()
    mpk.linear_with_residual_layer(
        input=tensors['silu_mul_out'],
        weight=w_down,
        residual=x,
        output=tensors['mlp_out'],
        grid_dim=(mlp_out_dim // 64, 1, 1),
        block_dim=(mlp_out_block_dim, 1, 1),
    )

    # Update x to MLP output
    x = tensors['mlp_out']
    
    # Final allreduce if multi-GPU
    if world_size > 1:
        mpk.allreduce_layer(
            input=tensors['mlp_out'],
            buffer=tensors['allreduce_buf'],
            output=tensors['mlp_final'],
            grid_dim=(config['hidden_size'] // 64, 1, 1),
            block_dim=(128, 1, 1),
        )
        x = tensors['mlp_final']
    
    return x


def add_final_layers(mpk, x, model, config, tensors, spec_decode_config, args):
    # 1. Final RMSNorm + LM head projection
    w_norm = mpk.attach_input(
        torch_tensor=model.model.norm.weight, 
        name="model_norm_weight"
    )
    w_proj = mpk.attach_input(
        torch_tensor=config['lm_head_weight'],
        name="lm_head"
    )

    # RMSNorm + Linear layers
    lm_head_weight_shape0 = config['lm_head_weight'].shape[0]

    # Final RMSNorm
    final_rmsnorm_block_dim = get_block_dim()
    mpk.rmsnorm_layer(
        input=x,
        weight=w_norm,
        output=tensors['rmsnorm_out'],
        grid_dim=(args.max_num_batched_tokens, 1, 1),
        block_dim=(final_rmsnorm_block_dim, 1, 1),
    )

    # Final Linear
    final_linear_block_dim = get_block_dim()
    mpk.linear_layer(
        input=tensors['rmsnorm_out'],
        weight=w_proj,
        output=tensors['argmax_in'],
        grid_dim=(grid_for_rmsnorm_linear_layer(lm_head_weight_shape0), 1, 1),
        block_dim=(final_linear_block_dim, 1, 1),
    )

    # 2. Argmax layers for token selection
    if spec_decode_config and spec_decode_config.method == "promptlookup":
        argmax_partial_grid_dim = (
            max_factor_leq_n(config['padded_vocab_size'], 96 // (spec_decode_config.spec_length + 1)), 
            spec_decode_config.spec_length + 1, 
            1
        )
        argmax_reduce_grid_dim = (1, spec_decode_config.spec_length + 1, 1)
    else:
        argmax_partial_grid_dim = (mpk.num_workers, 1, 1)
        argmax_reduce_grid_dim = (1, 1, 1)
    
    mpk.argmax_partial_layer(
        input=tensors['argmax_in'],
        output=(tensors['argmax_part_value'], tensors['argmax_part_index']),
        grid_dim=argmax_partial_grid_dim,
        block_dim=(128, 1, 1),
    )
    
    mpk.argmax_reduce_layer(
        input=(tensors['argmax_part_value'], tensors['argmax_part_index']),
        output=tensors['argmax_out'],
        grid_dim=argmax_reduce_grid_dim,
        block_dim=(128, 1, 1),
    )
    
    return tensors['argmax_out']


def build_mirage_graph(model, args, world_size, rank, input_data, eos_token_id_for_mirage):
    """Build the complete Mirage computation graph."""
    # Setup configuration
    config = setup_mirage_configuration(model, args, world_size, rank)
    
    # Create persistent kernel
    mpk, spec_decode_config = create_persistent_kernel(
        args, world_size, rank, input_data, config, eos_token_id_for_mirage
    )
    
    # Handle speculative decoding token input if needed
    spec_tokens = None
    if spec_decode_config and spec_decode_config.method == "promptlookup":
        all_tokens = mpk.attach_input(torch_tensor=input_data['tokens'], name="all_tokens")
        spec_tokens = mpk.draft_forward_layer_dispatcher(
            spec_decode_config=spec_decode_config,
            tokens=all_tokens,
            grid_dim=(96, 1, 1),
            block_dim=(128, 1, 1),
        )
        inputs = attach_model_inputs(mpk, model, input_data, config)
        inputs['input_token'] = spec_tokens  # Override input token for spec decode
    else:
        # Attach model inputs (for normal generation or lookahead spec decode)
        inputs = attach_model_inputs(mpk, model, input_data, config)
    
    # Create intermediate tensors
    tensors = create_intermediate_tensors(
        mpk, config, spec_decode_config, world_size, args
    )
    
    # Add embedding layer
    x = add_embedding_layer(mpk, inputs, tensors, config, spec_decode_config)
    
    # Add transformer layers
    for i, layer in enumerate(model.model.layers):
        x = add_transformer_layer(
            mpk, i, layer, x, inputs, tensors, config, world_size, spec_decode_config, model, args
        )
    
    # Add final layers
    output = add_final_layers(mpk, x, model, config, tensors, spec_decode_config, args)
    
    # Add verification layer for speculative decoding
    if spec_decode_config and spec_tokens is not None:
        verify_out = mpk.verify_layer_dispatcher(
            spec_decode_config=spec_decode_config,
            spec_tokens=spec_tokens,
            target_output=output,
            grid_dim=(1, 1, 1),
            block_dim=(128, 1, 1),
        )
    
    # Generate task graph and compile
    results = mpk.kn_graph.generate_task_graph(num_gpus=world_size, my_gpu_id=rank)
    with open(f"task_graph_{rank}.json", "w") as f:
        f.write(results["json_file"])
    with open(f"kernel_{rank}.cu", "w") as f:
        f.write(results["cuda_code"])
    
    mpk.compile(output_dir=args.output_dir)
    
    return mpk


# ============================================================================
# Generation Functions
# ============================================================================

def run_pytorch_generation(model, input_data, eos_token_ids, output_len=512):
    tokens = input_data['tokens']
    prompt_len = input_data['prompt_lengths'][0].item()  # Get the first batch's prompt length
    position_embeddings = input_data['position_embeddings']
    step = input_data['step']
    
    stream = torch.cuda.Stream()
    prev_pos = 0

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    warmup = 0
    
    max_len = tokens.shape[1]
    for cur_pos in range(prompt_len, min(prompt_len + output_len, max_len)):
        step.fill_(cur_pos - 1)
        input_ids = tokens[:, prev_pos:cur_pos]
        cos_embeddings = position_embeddings[0][:, prev_pos:cur_pos]
        sin_embeddings = position_embeddings[1][:, prev_pos:cur_pos]
        
        logits = model.forward(
            input_ids=input_ids,
            position_embeddings=(cos_embeddings, sin_embeddings),
            step=step,
            stream=stream,
        )
        
        next_token = logits.argmax(dim=-1)
        next_token = next_token[0, -1]
        tokens[0, cur_pos] = next_token
        prev_pos = cur_pos
        
        if next_token.item() in eos_token_ids:
            break
            
        if cur_pos == prompt_len + warmup:
            torch.cuda.synchronize()
            starter.record()
    
    ender.record()
    torch.cuda.synchronize()
    run_time = starter.elapsed_time(ender)
    
    return cur_pos, run_time


def run_mirage_generation(mpk):
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    starter.record()
    mpk()
    ender.record()

    torch.cuda.synchronize()
    run_time = starter.elapsed_time(ender)

    return run_time


# ============================================================================
# Main Execution Functions
# ============================================================================

def run_generation_comparison(model, tokenizer, args, world_size, rank):
    # Process EOS tokens
    eos_token_id_for_mirage, eos_token_ids = process_eos_tokens(model, tokenizer)
    
    # Prepare test data
    messages = prepare_test_prompt()
    input_data = prepare_input_tensors(model, tokenizer, messages, args, args.use_mirage)
    
    if args.use_mirage:
        mpk = build_mirage_graph(model, args, world_size, rank, input_data, eos_token_id_for_mirage)
        run_time = run_mirage_generation(mpk)

        print("="*60)
        print("Generation Results (MPK)")
        print("="*60)

        prompt_lengths = input_data['prompt_lengths']
        step = input_data['step']
        tokens = input_data['tokens']
        print("tokens.shape = ", tokens.shape)
        for r in range(args.max_num_batched_requests):
            generated_ids = tokens[r, : step[r] + 1]
            response = tokenizer.decode(generated_ids, skip_special_tokens=True)
            num_generated_tokens = (step[r] + 1 - prompt_lengths[r].item())
            print("-"*40)
            print(f"Request {r}:, generate length = {num_generated_tokens}\n")
            print(response)

        if args.max_num_batched_requests > 1:
            print(f"Output length of each batch is same: {(step.max() == step.min()).item()}")

        print("Prompt length {}, generate length {}, per-token latency (both prefill and decode): {} ms".format(
              prompt_lengths[0], step.max().item() + 1 - prompt_lengths[0], run_time / (step.max().item() + 1)
            )
        )
        
    else:
        end_pos, run_time = run_pytorch_generation(model, input_data, eos_token_ids)
        
        print("="*60)
        print("Generation Results (PyTorch)")
        print("="*60)
        
        tokens = input_data['tokens']
        generated_ids = tokens[:, :end_pos]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(response)
        
        prompt_lengths = input_data['prompt_lengths']
        print(f"\nPrompt length: {prompt_lengths[0].item()}")
        print(f"Generated length: {end_pos - prompt_lengths[0].item()}")
        print(f"Per-token latency: {run_time / (end_pos - prompt_lengths[0].item()):.4f} ms")


def main():
    """Main execution function."""
    global print
    
    # Parse arguments and setup
    args = parse_arguments()
    print("Input arguments:", args)

    world_size, rank = setup_env()
    
    if rank != 0:
        print = lambda *_, **__: None
    
    print(f"world_size({world_size}) rank({rank})")

    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(args, world_size, rank)
    print("Model loaded successfully!")
    
    # Run generation comparison
    run_generation_comparison(model, tokenizer, args, world_size, rank)
    
    # Cleanup
    if world_size > 1:
        dist.destroy_process_group()
    
    print("Demo completed successfully!")


if __name__ == "__main__":
    main()