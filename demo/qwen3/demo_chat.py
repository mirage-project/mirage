import argparse
import os
import torch
import torch.distributed as dist
from transformers import AutoTokenizer

from models.modeling_qwen3 import Qwen3ForCausalLM

MAX_SEQ_LEN = 32768
MAX_CONTEXT_LEN = 32768
MODEL_NAME = "Qwen/Qwen3-8B"
SYSTEM_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."


def setup_distributed_environment():
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

    if rank != 0:
        # Suppress all print statements on non-root ranks
        __builtins__.print = lambda *args, **kwargs: None

    return world_size, rank


def load_model_and_tokenizer(rank):
    print(f"Loading model: {MODEL_NAME}")
    torch.set_default_dtype(torch.bfloat16)
    torch.cuda.set_device(rank)

    with torch.device("cuda"):
        model = Qwen3ForCausalLM.from_pretrained(MODEL_NAME).to("cuda")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print("Model and tokenizer loaded.")
    return model, tokenizer


def build_mirage_graph(model, world_size, rank, args, tokens_tensor, step_tensor):
    print("Building Mirage execution graph...")
    import mirage as mi

    # --- Model Configuration ---
    batch_size = 1
    hidden_size = model.config.hidden_size
    intermediate_size = model.config.intermediate_size
    lm_head_weight = torch.cat(
        (
            model.lm_head.weight,
            torch.full(
                (153600 - model.config.vocab_size, hidden_size), 0, device="cuda"
            ),
        ),
        0,
    )
    assert lm_head_weight.stride()[0] == hidden_size
    vocab_size = 153600
    num_q_heads = model.config.num_attention_heads
    num_kv_heads = model.config.num_key_value_heads
    num_local_q_heads = num_q_heads // world_size
    num_local_kv_heads = num_kv_heads // world_size
    head_dim = hidden_size // num_q_heads
    fused_outdim_1 = (num_q_heads + 2 * num_kv_heads) * head_dim
    fused_outdim_2 = 2 * intermediate_size

    # --- Profiler Setup ---
    profiler_tensor = (
        torch.empty(3000 * 128, dtype=torch.uint64, device="cuda").contiguous()
        if args.profiling
        else None
    )

    # --- Persistent Kernel Setup ---
    mpk = mi.PersistentKernel(
        world_size=world_size,
        mpi_rank=rank,
        num_workers=96,
        num_local_schedulers=48,
        num_remote_schedulers=0,
        max_seq_length=4096,
        eos_token_id=model.config.eos_token_id,
        meta_tensors=[step_tensor, tokens_tensor],
        profiler_tensor=profiler_tensor,
    )

    # --- Tensor Definitions ---
    input_tokens = torch.full((1, 1), 0, dtype=torch.long, device="cuda")
    positions = torch.arange(MAX_SEQ_LEN).unsqueeze(0).to(model.device)
    position_embeddings = model.model.rotary_emb(positions)

    x = mpk.attach_input(torch_tensor=input_tokens, name="input_token")
    cos_pos_embed = mpk.attach_input(
        torch_tensor=position_embeddings[0][0, :MAX_CONTEXT_LEN, :],
        name="cos_position_embedding",
    )
    sin_pos_embed = mpk.attach_input(
        torch_tensor=position_embeddings[1][0, :MAX_CONTEXT_LEN, :],
        name="sin_position_embedding",
    )

    # Create intermediate tensors for the graph
    embed_out = mpk.new_tensor(dims=(batch_size, hidden_size), dtype=mi.bfloat16, name="embed_out")
    attn_in = mpk.new_tensor(dims=(batch_size, fused_outdim_1 // world_size), dtype=mi.bfloat16, name="attn_in")
    attn_out = mpk.new_tensor(dims=(batch_size, num_local_q_heads * head_dim), dtype=mi.bfloat16, name="attn_out")

    is_nvshmem = "nvshmem_tensor" if world_size > 1 else "cuda_tensor"
    attn_proj_out = mpk.new_tensor(dims=(batch_size, hidden_size), dtype=mi.bfloat16, name="attn_proj_out", io_category=is_nvshmem)
    allreduce_buf = mpk.new_tensor(dims=(world_size, batch_size, hidden_size), dtype=mi.bfloat16, name="all_reduce_buf", io_category=is_nvshmem)
    attn_allreduce_out = mpk.new_tensor(dims=(batch_size, hidden_size), dtype=mi.bfloat16, name="attn_allreduce_out", io_category=is_nvshmem)
    mlp_mid = mpk.new_tensor(dims=(batch_size, fused_outdim_2 // world_size), dtype=mi.bfloat16, name="mlp_mid")
    mlp_out = mpk.new_tensor(dims=(batch_size, hidden_size), dtype=mi.bfloat16, name="mlp_out", io_category=is_nvshmem)
    mlp_final = mpk.new_tensor(dims=(batch_size, hidden_size), dtype=mi.bfloat16, name="mlp_final", io_category=is_nvshmem)
    argmax_in = mpk.new_tensor(dims=(batch_size, vocab_size), dtype=mi.bfloat16, name="argmax_in")
    argmax_part_value = mpk.new_tensor(dims=(batch_size, 96), dtype=mi.bfloat16, name="argmax_part_value")
    argmax_part_index = mpk.new_tensor(dims=(batch_size, 96), dtype=mi.int64, name="argmax_part_index")
    argmax_out = mpk.new_tensor(dims=(batch_size, 1), dtype=mi.int64, name="argmax_out")

    # --- Define the Model Graph ---
    w_embed = mpk.attach_input(torch_tensor=model.model.embed_tokens.weight, name="embed_tokens")
    mpk.embed_layer(input=x, weight=w_embed, output=embed_out, grid_dim=(1, 1, 1), block_dim=(128, 1, 1))
    x = embed_out

    for i, layer in enumerate(model.model.layers):
        # Attention block
        w_norm_attn = mpk.attach_input(torch_tensor=layer.input_layernorm.weight, name=f"layer_{i}_input_layernorm")
        w_q = mpk.attach_input(torch_tensor=layer.self_attn.q_proj.weight, name=f"layer_{i}_q_proj")
        w_k = mpk.attach_input(torch_tensor=layer.self_attn.k_proj.weight, name=f"layer_{i}_k_proj")
        w_v = mpk.attach_input(torch_tensor=layer.self_attn.v_proj.weight, name=f"layer_{i}_v_proj")
        w_qkv = mpk.fuse_tensors(inputs=[w_q, w_k, w_v], fused_dim=0, num_groups=num_local_kv_heads, name=f"layer_{i}_qkv_proj")
        mpk.rmsnorm_linear_layer(input=x, weight_norm=w_norm_attn, weight_linear=w_qkv, output=attn_in, grid_dim=(96, 1, 1), block_dim=(128, 1, 1))

        w_q_norm = mpk.attach_input(torch_tensor=layer.self_attn.q_norm.weight, name=f"layer_{i}_q_norm")
        w_k_norm = mpk.attach_input(torch_tensor=layer.self_attn.k_norm.weight, name=f"layer_{i}_k_norm")
        k_cache = mpk.attach_input(torch_tensor=model.model.kv_cache[0][i], name=f"layer_{i}_k_cache")
        v_cache = mpk.attach_input(torch_tensor=model.model.kv_cache[1][i], name=f"layer_{i}_v_cache")
        mpk.attention_layer(input=attn_in, q_norm=w_q_norm, k_norm=w_k_norm, k_cache=k_cache, v_cache=v_cache, cos_pos_embed=cos_pos_embed, sin_pos_embed=sin_pos_embed, output=attn_out, grid_dim=(batch_size, num_local_kv_heads, 1), block_dim=(128, 1, 1))

        w_o_proj = mpk.attach_input(torch_tensor=layer.self_attn.o_proj.weight, name=f"layer_{i}_o_proj")
        mpk.linear_with_residual_layer(input=attn_out, weight=w_o_proj, residual=x, output=attn_proj_out, grid_dim=(hidden_size // 64, 1, 1), block_dim=(128, 1, 1))
        x = attn_proj_out

        if world_size > 1:
            mpk.allreduce_layer(input=attn_proj_out, buffer=allreduce_buf, output=attn_allreduce_out, grid_dim=(hidden_size // 64, 1, 1), block_dim=(128, 1, 1))
            x = attn_allreduce_out

        # MLP block
        residual_mlp = x
        w_norm_mlp = mpk.attach_input(torch_tensor=layer.post_attention_layernorm.weight, name=f"layer_{i}_post_attn_layernorm")
        w_gate_proj = mpk.attach_input(torch_tensor=layer.mlp.gate_proj.weight, name=f"layer_{i}_gate_proj")
        w_up_proj = mpk.attach_input(torch_tensor=layer.mlp.up_proj.weight, name=f"layer_{i}_up_proj")
        w_gatedup = mpk.fuse_tensors(inputs=[w_gate_proj, w_up_proj], fused_dim=0, num_groups=1, name=f"layer_{i}_gatedup_proj")
        mpk.rmsnorm_linear_layer(input=x, weight_norm=w_norm_mlp, weight_linear=w_gatedup, output=mlp_mid, grid_dim=(96, 1, 1), block_dim=(128, 1, 1))

        w_down_proj = mpk.attach_input(torch_tensor=layer.mlp.down_proj.weight, name=f"layer_{i}_down_proj")
        mpk.silu_mul_linear_with_residual_layer(input=mlp_mid, weight=w_down_proj, residual=residual_mlp, output=mlp_out, grid_dim=(hidden_size // 64, 1, 1), block_dim=(128, 1, 1))
        x = mlp_out

        if world_size > 1:
            mpk.allreduce_layer(input=mlp_out, buffer=allreduce_buf, output=mlp_final, grid_dim=(hidden_size // 64, 1, 1), block_dim=(128, 1, 1))
            x = mlp_final

    # Final layer
    w_final_norm = mpk.attach_input(torch_tensor=model.model.norm.weight, name="model_norm_weight")
    w_lm_head = mpk.attach_input(torch_tensor=lm_head_weight, name="lm_head")
    mpk.rmsnorm_linear_layer(input=x, weight_norm=w_final_norm, weight_linear=w_lm_head, output=argmax_in, grid_dim=(96, 1, 1), block_dim=(128, 1, 1))

    # Argmax
    mpk.argmax_partial_layer(input=argmax_in, output=(argmax_part_value, argmax_part_index), grid_dim=(96, 1, 1), block_dim=(128, 1, 1))
    mpk.argmax_reduce_layer(input=(argmax_part_value, argmax_part_index), output=argmax_out, grid_dim=(1, 1, 1), block_dim=(128, 1, 1))

    mpk.compile()
    print("Mirage graph compiled.")
    return mpk


def run_pytorch_generation(model, tokens, prompt_len, step_tensor, position_embeddings):
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    stream = torch.cuda.Stream()
    output_len = 4096
    end_pos = prompt_len

    prev_pos = 0
    torch.cuda.synchronize()
    starter.record()

    for cur_pos in range(prompt_len, prompt_len + output_len):
        step_tensor.fill_(cur_pos - 1)
        input_ids = tokens[:, prev_pos:cur_pos]
        cos_embeddings = position_embeddings[0][:, prev_pos:cur_pos]
        sin_embeddings = position_embeddings[1][:, prev_pos:cur_pos]

        logits = model.forward(
            input_ids=input_ids,
            position_embeddings=(cos_embeddings, sin_embeddings),
            step=step_tensor,
            stream=stream,
        )

        next_token = logits.argmax(dim=-1)[0, -1]
        tokens[0, cur_pos] = next_token
        prev_pos = cur_pos
        end_pos = cur_pos + 1

        if next_token == model.config.eos_token_id:
            break

    ender.record()
    torch.cuda.synchronize()
    run_time = starter.elapsed_time(ender)
    generated_len = end_pos - prompt_len

    return end_pos, run_time, generated_len


def run_mirage_generation(model, mpk, tokens, prompt_len, step_tensor, position_embeddings):
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    stream = torch.cuda.Stream()

    # Prefill phase
    step_tensor.fill_(prompt_len - 1)
    input_ids = tokens[:, 0:prompt_len]
    cos_embeddings = position_embeddings[0][:, 0:prompt_len]
    sin_embeddings = position_embeddings[1][:, 0:prompt_len]
    logits = model.forward(
        input_ids=input_ids,
        position_embeddings=(cos_embeddings, sin_embeddings),
        step=step_tensor,
        stream=stream,
    )
    next_token = logits.argmax(dim=-1)[0, -1]
    tokens[0, prompt_len] = next_token
    torch.cuda.synchronize()

    # Re-initialize the persistent kernel for the next turn
    meta_tensors_ptr = [tensor.data_ptr() for tensor in mpk.meta_tensors]
    profiler_buffer_ptr = (
        mpk.profiler_tensor.data_ptr() if mpk.profiler_tensor is not None else 0
    )
    mpk.init_func(
        meta_tensors_ptr,
        profiler_buffer_ptr,
        mpk.mpi_rank,
        mpk.num_workers,
        mpk.num_local_schedulers,
        mpk.num_remote_schedulers,
    )

    # Generation phase
    step_tensor.fill_(prompt_len)
    starter.record()
    mpk()
    ender.record()
    torch.cuda.synchronize()
    run_time = starter.elapsed_time(ender)
    end_pos = step_tensor[0].item()
    generated_len = end_pos - prompt_len

    return end_pos, run_time, generated_len


def main():
    parser = argparse.ArgumentParser(description="Interactive chat demo for Qwen3-8B with Mirage.")
    parser.add_argument("--use-mirage", action="store_true", help="Use Mirage kernels for inference.")
    parser.add_argument("--profiling", action="store_true", help="Enable profiler to generate a trace.")
    args = parser.parse_args()

    world_size, rank = setup_distributed_environment()

    print("Input arguments:", args)
    print(f"world_size({world_size}) rank({rank})")

    model, tokenizer = load_model_and_tokenizer(rank)

    tokens = torch.full((1, MAX_SEQ_LEN), 0, dtype=torch.long, device="cuda")
    step_tensor = torch.tensor([0], dtype=torch.int32, device="cuda")

    mpk = None
    if args.use_mirage:
        mpk = build_mirage_graph(model, world_size, rank, args, tokens, step_tensor)

    positions = torch.arange(MAX_SEQ_LEN).unsqueeze(0).to(model.device)
    position_embeddings = model.model.rotary_emb(positions)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if rank == 0:
        print("\n> Welcome to the Mirage Qwen3-8B chat demo. Type 'exit' or 'quit' to end.")

    while True:
        prompt_container = [None]
        if rank == 0:
            try:
                prompt = input("> User: ")
                if not prompt:
                    print("> Prompt cannot be empty.")
                    continue
                prompt_container[0] = prompt
            except EOFError:
                prompt_container[0] = "exit"

        if world_size > 1:
            dist.broadcast_object_list(prompt_container, src=0)

        prompt = prompt_container[0]
        if prompt is None or prompt.lower() in ["exit", "quit"]:
            break

        messages.append({"role": "user", "content": prompt})
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        new_prompt_len = model_inputs.input_ids.shape[-1]
        tokens[0, :new_prompt_len] = model_inputs.input_ids[0]
        if new_prompt_len < tokens.shape[1]:
            tokens[0, new_prompt_len:] = 0
        prompt_len = new_prompt_len

        if args.use_mirage:
            end_pos, run_time, generated_len = run_mirage_generation(
                model, mpk, tokens, prompt_len, step_tensor, position_embeddings
            )
        else:
            end_pos, run_time, generated_len = run_pytorch_generation(
                model, tokens, prompt_len, step_tensor, position_embeddings
            )

        if rank == 0:
            assistant_response_ids = tokens[0, prompt_len:end_pos]
            assistant_response = tokenizer.decode(assistant_response_ids, skip_special_tokens=True)
            print(f"Assistant: {assistant_response}")
            messages.append({"role": "assistant", "content": assistant_response})
            if generated_len > 0 and run_time > 0:
                print(
                    f"[Total generated length {generated_len}, per-token latency {run_time / generated_len:.4f} ms]"
                )
            else:
                print(f"[Total generated length {generated_len}]")

    if world_size > 1:
        dist.destroy_process_group()

    print("Exiting demo.")


if __name__ == "__main__":
    main()
