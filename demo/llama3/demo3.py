import argparse
import os
import torch
import torch.distributed as dist
from transformers import AutoTokenizer, AutoConfig
from safetensors.torch import load_model

from models.modeling_llama import LlamaForCausalLM


# ============================================================================
# Configuration and Setup Functions
# ============================================================================

def parse_arguments():
    """Parse command line arguments for the demo."""
    parser = argparse.ArgumentParser(description="Llama3 MPK Demo")
    
    parser.add_argument("--use-mirage", action="store_true", 
                       help="Use Mirage kernels")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output files directory")
    parser.add_argument("--profiling", action="store_true", 
                       help="Use Profiler to generate trace")
    
    parser.add_argument("--model-path", type=str, default=None,
                       help="Path to a local model (necessary for multi-GPU demo)")
    parser.add_argument("--model", type=str, default='meta-llama/Meta-Llama-3-8B-Instruct',
                       help="Model path on hugging face")
    
    parser.add_argument("--spec-decode", default=None,
                       choices=["promptlookup", "lookahead"],
                       help="Enable speculative decoding with 'lookahead' or 'promptlookup' mode.")
    parser.add_argument("--ngram-size", default=3, type=int,
                       help="Ngram size for lookahead spec decode")
    parser.add_argument("--max-seq-length", default=512, type=int,
                       help="Max sequence length for spec decode")
    parser.add_argument("--spec-length", default=3, type=int,
                       help="Spec length for spec decode")
    
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
            model = LlamaForCausalLM(config, world_size)
            load_model(
                model, f"{args.model_path}/model{rank}-mp{world_size}.safetensors"
            )
            tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        else:
            model = LlamaForCausalLM.from_pretrained(model_name).to("cuda")
            
            tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    return model, tokenizer


def process_eos_tokens(model, tokenizer):
    """Process EOS token IDs to handle both single and multiple EOS tokens."""
    eos_token_id = model.config.eos_token_id
    if isinstance(eos_token_id, (list, tuple)):
        eos_token_ids = eos_token_id  # Keep all EOS tokens for checking
        eos_token_id_for_mirage = eos_token_id[0]  # Use first for Mirage kernel
    else:
        eos_token_ids = [eos_token_id] if eos_token_id is not None else []
        eos_token_id_for_mirage = eos_token_id
    
    if eos_token_id_for_mirage is None:
        eos_token_id_for_mirage = tokenizer.eos_token_id
        eos_token_ids = [eos_token_id_for_mirage]
    
    return eos_token_id_for_mirage, eos_token_ids


# ============================================================================
# Utility Functions
# ============================================================================

def grid_for_rmsnorm_linear_layer(size):
    for grid_size in [96, 64, 32, 16, 8, 4, 2]:
        if size % grid_size == 0:
            return grid_size
    return 1


def max_factor_leq_n(m: int, n: int) -> int:
    """Return the largest factor of m that is less than or equal to n."""
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
    """Prepare a test prompt for generation."""
    
    prompt = "Give me a short introduction to large language model."
    
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        },
        {"role": "user", "content": prompt},
    ]
    
    return messages


# ============================================================================
# Input Preparation Functions
# ============================================================================

def prepare_input_tensors(model, tokenizer, messages):
    """Prepare input tensors and embeddings for generation."""
    # Tokenize the input - handle missing chat template
    text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )    
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    # Prepare tokens tensor
    tokens = torch.full((1, 32768), 0, dtype=torch.long, device="cuda")
    for i in range(model_inputs.input_ids.shape[-1]):
        tokens[0, i] = model_inputs.input_ids[0, i]
    prompt_len = model_inputs.input_ids.shape[-1]
    
    # Prepare position embeddings
    positions = torch.arange(32768).unsqueeze(0).to(model.device)
    position_embeddings = model.model.rotary_emb(positions)
    
    # Prepare control tensors
    input_tokens = torch.full((1, 1), 0, dtype=torch.long, device="cuda")
    step = torch.tensor([0], dtype=torch.int32, device="cuda")
    num_new_tokens = torch.tensor([1], dtype=torch.int32, device="cuda")  # Generate 1 token per MPK call to work around MPK bug
    
    return {
        'tokens': tokens,
        'prompt_len': prompt_len,
        'position_embeddings': position_embeddings,
        'input_tokens': input_tokens,
        'step': step,
        'num_new_tokens': num_new_tokens
    }
        

# ============================================================================
# Mirage MPK Setup Functions
# ============================================================================

def setup_mirage_configuration(model, args, world_size, rank):
    """Setup Mirage configuration parameters."""
    batch_size = 1
    hidden_size = model.config.hidden_size
    intermediate_size = model.config.intermediate_size
    
    # Pad vocab_size to facilitate task graph creation (same as Llama3.2)
    padded_vocab_size = ((model.config.vocab_size + 95) // 96) * 96
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
    
    return {
        'batch_size': batch_size,
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
    """Create and configure the Mirage Persistent Kernel."""
    import mirage as mi
    
    # Setup profiler tensor if needed
    if args.profiling:
        profiler_tensor = torch.empty(
            3000 * 128, dtype=torch.uint64, device="cuda"
        ).contiguous()
    else:
        profiler_tensor = None
    
    # Setup speculative decoding configuration
    spec_decode_config = mi.speculative.spec_decode_class(
        args.spec_decode,
        ngram_size=args.ngram_size,
        spec_length=args.spec_length,
    )
    
    # Get GPU configurations
    num_workers, num_schedulers = mi.get_configurations_from_gpu(rank)
    
    # Create persistent kernel
    mpk = mi.PersistentKernel(
        world_size=world_size,
        mpi_rank=rank,
        num_workers=num_workers,
        num_local_schedulers=num_schedulers,
        num_remote_schedulers=0,
        max_seq_length=args.max_seq_length,
        eos_token_id=eos_token_id_for_mirage,
        meta_tensors=[input_data['step'], input_data['tokens'], input_data['num_new_tokens']],
        profiler_tensor=profiler_tensor,
        spec_decode_config=spec_decode_config,
    )
    
    return mpk, spec_decode_config


def attach_model_inputs(mpk, model, input_data, config):
    """Attach input tensors to the persistent kernel."""
    # Attach basic inputs
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


def create_intermediate_tensors(mpk, config, spec_decode_config, world_size):
    """Create intermediate tensors for the computation graph."""
    import mirage as mi
    
    # Calculate tensor dimensions based on speculative decoding
    if spec_decode_config and spec_decode_config.method == "promptlookup":
        num_tokens_extend = spec_decode_config.spec_length + 1
    else:
        num_tokens_extend = 1
    
    total_tokens_per_iter = config['batch_size'] * num_tokens_extend
    
    # TODO: Make the code run well even if 96 % total_tokens_per_iter != 0
    assert(96 % total_tokens_per_iter == 0), f"96 must be divisible by {total_tokens_per_iter}"
    
    tensors = {}
    
    # Embedding output
    tensors['embed_out'] = mpk.new_tensor(
        dims=(total_tokens_per_iter, config['hidden_size']),
        dtype=mi.bfloat16,
        name="embed_out",
        io_category="cuda_tensor",
    )
    
    # Attention tensors
    tensors['attn_in'] = mpk.new_tensor(
        dims=(total_tokens_per_iter, config['fused_outdim_1'] // world_size),
        dtype=mi.bfloat16,
        name="attn_in",
        io_category="cuda_tensor",
    )
    
    tensors['attn_out'] = mpk.new_tensor(
        dims=(total_tokens_per_iter, config['num_local_q_heads'] * config['head_dim']),
        dtype=mi.bfloat16,
        name="attn_out",
        io_category="cuda_tensor",
    )
    
    # Attention projection output
    tensors['attn_proj_out'] = mpk.new_tensor(
        dims=(total_tokens_per_iter, config['hidden_size']),
        dtype=mi.bfloat16,
        name="attn_proj_out",
        io_category="nvshmem_tensor" if world_size > 1 else "cuda_tensor",
    )
    
    # Allreduce tensors for multi-GPU
    tensors['allreduce_buf'] = mpk.new_tensor(
        dims=(world_size, total_tokens_per_iter, config['hidden_size']),
        dtype=mi.bfloat16,
        name="all_reduce_buf",
        io_category="nvshmem_tensor" if world_size > 1 else "cuda_tensor",
    )
    
    tensors['attn_allreduce_out'] = mpk.new_tensor(
        dims=(total_tokens_per_iter, config['hidden_size']),
        dtype=mi.bfloat16,
        name="attn_allreduce_out",
        io_category="nvshmem_tensor" if world_size > 1 else "cuda_tensor",
    )
    
    # MLP tensors
    tensors['mlp_mid'] = mpk.new_tensor(
        dims=(total_tokens_per_iter, config['fused_outdim_2'] // world_size),
        dtype=mi.bfloat16,
        name="mlp_mid",
        io_category="cuda_tensor",
    )
    
    tensors['mlp_out'] = mpk.new_tensor(
        dims=(total_tokens_per_iter, config['hidden_size']),
        dtype=mi.bfloat16,
        name="mlp_out",
        io_category="nvshmem_tensor" if world_size > 1 else "cuda_tensor",
    )
    
    tensors['mlp_final'] = mpk.new_tensor(
        dims=(total_tokens_per_iter, config['hidden_size']),
        dtype=mi.bfloat16,
        name="mlp_final",
        io_category="nvshmem_tensor" if world_size > 1 else "cuda_tensor",
    )
    
    # Final layer and argmax tensors
    tensors['argmax_in'] = mpk.new_tensor(
        dims=(total_tokens_per_iter, config['padded_vocab_size']),
        dtype=mi.bfloat16,
        name="argmax_in",
        io_category="cuda_tensor",
    )
    
    tensors['argmax_part_value'] = mpk.new_tensor(
        dims=(total_tokens_per_iter, 96 // (config['batch_size'] * num_tokens_extend)),
        dtype=mi.bfloat16,
        name="argmax_part_value",
        io_category="cuda_tensor",
    )
    
    tensors['argmax_part_index'] = mpk.new_tensor(
        dims=(total_tokens_per_iter, 96 // (config['batch_size'] * num_tokens_extend)),
        dtype=mi.int64,
        name="argmax_part_index",
        io_category="cuda_tensor",
    )
    
    tensors['argmax_out'] = mpk.new_tensor(
        dims=(1, total_tokens_per_iter),
        dtype=mi.int64,
        name="argmax_out",
        io_category="cuda_tensor",
    )
    
    return tensors, total_tokens_per_iter


# ============================================================================
# Mirage Layer Construction Functions
# ============================================================================

def add_embedding_layer(mpk, inputs, tensors, config, spec_decode_config):
    """Add embedding layer to the computation graph."""
    # TODO: Add speculative decoding token layer if needed
    
    # Add embedding layer
    total_tokens_per_iter = tensors['embed_out'].dim(0)
    mpk.embed_layer(
        input=inputs['input_token'], 
        weight=inputs['embed_weight'], 
        output=tensors['embed_out'], 
        grid_dim=(max_factor_leq_n(config['hidden_size'], 96 // total_tokens_per_iter), 
                  total_tokens_per_iter, 1), 
        block_dim=(128, 1, 1),
        input_source=(spec_decode_config is not None)
    )
    
    return tensors['embed_out']


def add_transformer_layer(mpk, layer_idx, layer, x, inputs, tensors, config, world_size, spec_decode_config, model):
    """Add a single transformer layer to the computation graph."""
    total_tokens_per_iter = tensors['embed_out'].dim(0)
    
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
    
    # Fuse QKV weights for grouped query attention
    w_qkv = mpk.fuse_tensors(
        inputs=[w_q, w_k, w_v],
        fused_dim=0,
        num_groups=config['num_kv_heads'] // world_size,
        name=f"layer_{layer_idx}_qkv_proj",
    )
    
    mpk.rmsnorm_linear_layer(
        input=x,
        weight_norm=w_norm,
        weight_linear=w_qkv,
        output=tensors['attn_in'],
        grid_dim=(grid_for_rmsnorm_linear_layer(w_qkv.dim(0)), 1, 1),
        # grid_dim=(80, 1, 1),
        block_dim=(128, 1, 1),
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
    
    if spec_decode_config:
        mpk.single_batch_extend_attention_layer(
            input=tensors['attn_in'],
            k_cache=k_cache,
            v_cache=v_cache,
            q_norm=None,  # Llama3 doesn't use q_norm
            k_norm=None,  # Llama3 doesn't use k_norm
            cos_pos_embed=inputs['cos_pos_embed'],
            sin_pos_embed=inputs['sin_pos_embed'],
            output=tensors['attn_out'],
            grid_dim=(1, config['num_local_kv_heads'], 1),
            block_dim=(128, 1, 1),
        )
    else:
        mpk.attention_layer(
            input=tensors['attn_in'],
            k_cache=k_cache,
            v_cache=v_cache,
            q_norm=None,  # Llama3 doesn't use q_norm
            k_norm=None,  # Llama3 doesn't use k_norm
            cos_pos_embed=inputs['cos_pos_embed'],
            sin_pos_embed=inputs['sin_pos_embed'],
            output=tensors['attn_out'],
            grid_dim=(total_tokens_per_iter, config['num_local_kv_heads'], 1),
            block_dim=(128, 1, 1),
        )
    
    # 3. Attention output projection with residual
    w_o = mpk.attach_input(
        torch_tensor=layer.self_attn.o_proj.weight, 
        name=f"layer_{layer_idx}_o_proj"
    )
    mpk.linear_with_residual_layer(
        input=tensors['attn_out'],
        weight=w_o,
        residual=x,
        output=tensors['attn_proj_out'],
        grid_dim=(config['hidden_size'] // 64, 1, 1),
        block_dim=(128, 1, 1),
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
    
    # Fuse gate and up projections
    w_gatedup = mpk.fuse_tensors(
        inputs=[w_gate_proj, w_up_proj],
        fused_dim=0,
        num_groups=1,
        name=f"layer_{layer_idx}_gatedup_proj",
    )
    
    mpk.rmsnorm_linear_layer(
        input=x,
        weight_norm=w_norm_mlp,
        weight_linear=w_gatedup,
        output=tensors['mlp_mid'],
        grid_dim=(grid_for_rmsnorm_linear_layer(w_gatedup.dim(0)), 1, 1),
        block_dim=(128, 1, 1),
    )
    
    # 6. MLP computation with residual
    w_down = mpk.attach_input(
        torch_tensor=layer.mlp.down_proj.weight, 
        name=f"layer_{layer_idx}_down_proj"
    )
    mpk.silu_mul_linear_with_residual_layer(
        input=tensors['mlp_mid'],
        weight=w_down,
        residual=x,
        output=tensors['mlp_out'],
        grid_dim=(config['hidden_size'] // 64, 1, 1),
        block_dim=(128, 1, 1),
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


def add_final_layers(mpk, x, model, config, tensors, spec_decode_config):
    """Add final normalization and language modeling head."""
    total_tokens_per_iter = tensors['embed_out'].dim(0)
    
    # 1. Final RMSNorm + LM head projection
    w_norm = mpk.attach_input(
        torch_tensor=model.model.norm.weight, 
        name="model_norm_weight"
    )
    w_proj = mpk.attach_input(
        torch_tensor=config['lm_head_weight'], 
        name="lm_head"
    )
    
    mpk.rmsnorm_linear_layer(
        input=x,
        weight_norm=w_norm,
        weight_linear=w_proj,
        output=tensors['argmax_in'],
        grid_dim=(grid_for_rmsnorm_linear_layer(w_proj.dim(0)), 1, 1),
        block_dim=(128, 1, 1),
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
        argmax_partial_grid_dim = (96, 1, 1)
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
        # Attach model inputs
        inputs = attach_model_inputs(mpk, model, input_data, config)
    
    # Create intermediate tensors
    tensors, total_tokens_per_iter = create_intermediate_tensors(
        mpk, config, spec_decode_config, world_size
    )
    
    # Add embedding layer
    x = add_embedding_layer(mpk, inputs, tensors, config, spec_decode_config)
    
    # Add transformer layers
    for i, layer in enumerate(model.model.layers):
        x = add_transformer_layer(
            mpk, i, layer, x, inputs, tensors, config, world_size, spec_decode_config, model
        )
    
    # Add final layers
    output = add_final_layers(mpk, x, model, config, tensors, spec_decode_config)
    
    # Add verification layer for speculative decoding
    if spec_decode_config:
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
    """Run generation using PyTorch (baseline)."""
    tokens = input_data['tokens']
    prompt_len = input_data['prompt_len']
    position_embeddings = input_data['position_embeddings']
    step = input_data['step']
    
    stream = torch.cuda.Stream()
    prev_pos = 0
    
    # Timing setup
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    warmup = 0
    
    for cur_pos in range(prompt_len, prompt_len + output_len):
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


def run_mirage_generation(model, mpk, input_data, tokenizer):
    """Run generation using Mirage MPK."""
    tokens = input_data['tokens']
    prompt_len = input_data['prompt_len']
    position_embeddings = input_data['position_embeddings']
    step = input_data['step']
    
    stream = torch.cuda.Stream()
    
    # Timing setup
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    
    # Prefill phase with PyTorch
    step.fill_(prompt_len - 1)
    input_ids = tokens[:, 0:prompt_len]
    cos_embeddings = position_embeddings[0][:, 0:prompt_len]
    sin_embeddings = position_embeddings[1][:, 0:prompt_len]
    
    logits = model.forward(
        input_ids=input_ids,
        position_embeddings=(cos_embeddings, sin_embeddings),
        step=step,
        stream=stream,
    )
    
    next_token = logits.argmax(dim=-1)
    next_token = next_token[0, -1]
    tokens[0, prompt_len] = next_token
    torch.cuda.synchronize()
    
    # Decode phase with Mirage - single call like Qwen3
    starter.record()
    step.fill_(prompt_len)
    mpk()
    ender.record()
    torch.cuda.synchronize()
    
    run_time = starter.elapsed_time(ender)
    
    # Extract generated tokens exactly like working demo.py
    generated_ids = tokens[:, : step[0] + 1]
    final_pos = step[0] + 1

    # Calculate generation length like demo.py: step[0] is total length, subtract prompt_len for new tokens
    generated_length = step[0] - prompt_len
    return final_pos, run_time


# ============================================================================
# Main Execution Functions
# ============================================================================

def run_generation_comparison(model, tokenizer, args, world_size, rank):
    """Run and compare PyTorch vs Mirage generation."""
    # Process EOS tokens
    eos_token_id_for_mirage, eos_token_ids = process_eos_tokens(model, tokenizer)
    
    # Prepare test data
    messages = prepare_test_prompt()
    input_data = prepare_input_tensors(model, tokenizer, messages)
    
    if args.use_mirage:
        mpk = build_mirage_graph(model, args, world_size, rank, input_data, eos_token_id_for_mirage)
        end_pos, run_time = run_mirage_generation(model, mpk, input_data, tokenizer)
        mode = "Mirage"
    else:
        end_pos, run_time = run_pytorch_generation(model, input_data, eos_token_ids)
        mode = "PyTorch"
    
    # Process results
    prompt_len = input_data['prompt_len']
    generated_ids = input_data['tokens'][:, :end_pos]
    
    # Debug: Check the generated token values
    print(f"Debug - Generated token IDs shape: {generated_ids.shape}")
    print(f"Debug - Generated token IDs (first 10): {generated_ids[0, :min(10, generated_ids.shape[1])]}")
    print(f"Debug - Token ID range: min={generated_ids.min()}, max={generated_ids.max()}")
    print(f"Debug - Vocab size: {model.config.vocab_size}")
    
    # Check for invalid token IDs
    invalid_tokens = (generated_ids < 0) | (generated_ids >= model.config.vocab_size)
    if invalid_tokens.any():
        print(f"Warning: Found {invalid_tokens.sum()} invalid token IDs!")
        # Clamp invalid tokens to valid range
        generated_ids = torch.clamp(generated_ids, 0, model.config.vocab_size - 1)
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    # Print results
    print(f"\n{'='*60}")
    print(f"Generation Results ({mode})")
    print(f"{'='*60}")
    print(response)
    print(f"\nPrompt length: {prompt_len}")
    print(f"Generated length: {end_pos - prompt_len}")
    print(f"Per-token latency: {run_time / (end_pos - prompt_len):.4f} ms")


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