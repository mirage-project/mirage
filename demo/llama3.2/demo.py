from models.modeling_llama import LlamaForCausalLM
from transformers import AutoTokenizer, AutoConfig
from safetensors.torch import load_model
import torch
import torch.distributed as dist
import argparse
import os

def grid_for_rmsnorm_linear_layer(size):
    if size % 96 == 0:
        return 96
    elif size % 64 == 0:
        return 64

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-mirage", action="store_true", help="Use Mirage kernels")
    parser.add_argument(
        "--profiling", action="store_true", help="Use Profiler to generate trace"
    )
    parser.add_argument(
        "--model", type=str, default='meta-llama/Llama-3.2-3B-Instruct', help="Model path on hugging face"
    )
    args = parser.parse_args()
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
    global print
    if rank != 0:
        print = lambda *_, **__: None

    print("Input arguments:", args)
    print(f"world_size({world_size}) rank({rank})")
    model_name = args.model
    torch.set_default_dtype(torch.bfloat16)

    torch.cuda.set_device(rank)
    with torch.device("cuda"):
        model = LlamaForCausalLM.from_pretrained(model_name).to("cuda")

    tokens = torch.full((1, 32768), 0, dtype=torch.long, device="cuda")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    prompt = "Give me a short introduction to large language model."
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        },
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    for i in range(model_inputs.input_ids.shape[-1]):
        tokens[0, i] = model_inputs.input_ids[0, i]
    prompt_len = model_inputs.input_ids.shape[-1]
    positions = torch.arange(32768).unsqueeze(0).to(model.device)
    position_embeddings = model.model.rotary_emb(positions)

    input_tokens = torch.full((1, 1), 0, dtype=torch.long, device="cuda")
    prev_pos = 0

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    step = torch.tensor([0], dtype=torch.int32, device="cuda")
    if args.use_mirage:
        import mirage as mi

        batch_size = 1
        hidden_size = model.config.hidden_size
        intermediate_size = model.config.intermediate_size
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
        vocab_size = padded_vocab_size
        num_q_heads = model.config.num_attention_heads
        num_kv_heads = model.config.num_key_value_heads
        num_local_q_heads = num_q_heads // world_size
        num_local_kv_heads = num_kv_heads // world_size
        head_dim = model.config.head_dim
        fused_outdim_1 = (num_q_heads + 2 * num_kv_heads) * head_dim
        fused_outdim_2 = 2 * intermediate_size

        eos_token_id = model.config.eos_token_id
        if isinstance(eos_token_id, (list, tuple)):
            eos_token_id = eos_token_id[0]
        if eos_token_id is None:
            eos_token_id = tokenizer.eos_token_id

        if args.profiling:
            profiler_tensor = torch.empty(
                3000 * 128, dtype=torch.uint64, device="cuda"
            ).contiguous()
        else:
            profiler_tensor = None
        num_workers, num_schedulers = mi.get_configurations_from_gpu(rank)
        mpk = mi.PersistentKernel(
            world_size=world_size,
            mpi_rank=rank,
            num_workers=num_workers,
            num_local_schedulers=num_schedulers,
            num_remote_schedulers=0,
            max_seq_length=512,
            eos_token_id=eos_token_id,
            meta_tensors=[step, tokens],
            profiler_tensor=profiler_tensor,
        )
        x = mpk.attach_input(torch_tensor=input_tokens, name="input_token")
        cos_pos_embed = mpk.attach_input(
            torch_tensor=position_embeddings[0][0, :4096, :],
            name="cos_position_embedding",
        )
        sin_pos_embed = mpk.attach_input(
            torch_tensor=position_embeddings[1][0, :4096, :],
            name="sin_position_embedding",
        )
        y = mpk.new_tensor(
            dims=(batch_size, hidden_size),
            dtype=mi.bfloat16,
            name="embed_out",
            io_category="cuda_tensor",
        )
        attn_in = mpk.new_tensor(
            dims=(batch_size, fused_outdim_1 // world_size),
            dtype=mi.bfloat16,
            name="attn_in",
            io_category="cuda_tensor",
        )
        attn_out = mpk.new_tensor(
            dims=(batch_size, num_local_q_heads * head_dim),
            dtype=mi.bfloat16,
            name="attn_out",
            io_category="cuda_tensor",
        )
        attn_proj_out = mpk.new_tensor(
            dims=(batch_size, hidden_size),
            dtype=mi.bfloat16,
            name="attn_proj_out",
            io_category="nvshmem_tensor" if world_size > 1 else "cuda_tensor",
        )
        allreduce_buf = mpk.new_tensor(
            dims=(world_size, batch_size, hidden_size),
            dtype=mi.bfloat16,
            name="all_reduce_buf",
            io_category="nvshmem_tensor" if world_size > 1 else "cuda_tensor",
        )
        attn_allreduce_out = mpk.new_tensor(
            dims=(batch_size, hidden_size),
            dtype=mi.bfloat16,
            name="attn_allreduce_out",
            io_category="nvshmem_tensor" if world_size > 1 else "cuda_tensor",
        )
        mlp_mid = mpk.new_tensor(
            dims=(batch_size, fused_outdim_2 // world_size),
            dtype=mi.bfloat16,
            name="mlp_mid",
            io_category="cuda_tensor",
        )
        mlp_out = mpk.new_tensor(
            dims=(batch_size, hidden_size),
            dtype=mi.bfloat16,
            name="mlp_out",
            io_category="nvshmem_tensor" if world_size > 1 else "cuda_tensor",
        )
        mlp_final = mpk.new_tensor(
            dims=(batch_size, hidden_size),
            dtype=mi.bfloat16,
            name="mlp_final",
            io_category="nvshmem_tensor" if world_size > 1 else "cuda_tensor",
        )
        argmax_in = mpk.new_tensor(
            dims=(batch_size, vocab_size),
            dtype=mi.bfloat16,
            name="argmax_in",
            io_category="cuda_tensor",
        )
        argmax_part_value = mpk.new_tensor(
            dims=(batch_size, 96),
            dtype=mi.bfloat16,
            name="argmax_part_value",
            io_category="cuda_tensor",
        )
        argmax_part_index = mpk.new_tensor(
            dims=(batch_size, 96),
            dtype=mi.int64,
            name="argmax_part_index",
            io_category="cuda_tensor",
        )
        argmax_out = mpk.new_tensor(
            dims=(batch_size, 1),
            dtype=mi.int64,
            name="argmax_out",
            io_category="cuda_tensor",
        )

        w = mpk.attach_input(
            torch_tensor=model.model.embed_tokens.weight, name="embed_tokens"
        )
        mpk.embed_layer(
            input=x, weight=w, output=y, grid_dim=(1, 1, 1), block_dim=(128, 1, 1)
        )
        x = y
        for i, layer in enumerate(model.model.layers):
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
            w_qkv = mpk.fuse_tensors(
                inputs=[w_q, w_k, w_v],
                fused_dim=0,
                num_groups=model.config.num_key_value_heads // world_size,
                name=f"layer_{i}_qkv_proj",
            )
            mpk.rmsnorm_linear_layer(
                input=x,
                weight_norm=w_norm,
                weight_linear=w_qkv,
                output=attn_in,
                grid_dim=(grid_for_rmsnorm_linear_layer(w_qkv.dim(0)), 1, 1),
                block_dim=(128, 1, 1),
            )
            k_cache = mpk.attach_input(
                torch_tensor=model.model.kv_cache[0][i], name=f"layer_{i}_k_cache"
            )
            v_cache = mpk.attach_input(
                torch_tensor=model.model.kv_cache[1][i], name=f"layer_{i}_v_cache"
            )
            mpk.attention_layer(
                input=attn_in,
                q_norm=None,
                k_norm=None,
                k_cache=k_cache,
                v_cache=v_cache,
                cos_pos_embed=cos_pos_embed,
                sin_pos_embed=sin_pos_embed,
                output=attn_out,
                grid_dim=(batch_size, num_local_kv_heads, 1),
                block_dim=(128, 1, 1),
            )
            w = mpk.attach_input(
                torch_tensor=layer.self_attn.o_proj.weight, name=f"layer_{i}_o_proj"
            )
            mpk.linear_with_residual_layer(
                input=attn_out,
                weight=w,
                residual=x,
                output=attn_proj_out,
                grid_dim=(hidden_size // 64, 1, 1),
                block_dim=(128, 1, 1),
            )
            x = attn_proj_out
            if world_size > 1:
                mpk.allreduce_layer(
                    input=attn_proj_out,
                    buffer=allreduce_buf,
                    output=attn_allreduce_out,
                    grid_dim=(hidden_size // 64, 1, 1),
                    block_dim=(128, 1, 1),
                )
                x = attn_allreduce_out
            w_norm = mpk.attach_input(
                torch_tensor=layer.post_attention_layernorm.weight,
                name=f"layer_{i}_post_attn_layernorm",
            )
            w_gate_proj = mpk.attach_input(
                torch_tensor=layer.mlp.gate_proj.weight, name=f"layer_{i}_gate_proj"
            )
            w_up_proj = mpk.attach_input(
                torch_tensor=layer.mlp.up_proj.weight, name=f"layer_{i}_up_proj"
            )
            w_gatedup = mpk.fuse_tensors(
                inputs=[w_gate_proj, w_up_proj],
                fused_dim=0,
                num_groups=1,
                name=f"layer_{i}_gatedup_proj",
            )
            mpk.rmsnorm_linear_layer(
                input=x,
                weight_norm=w_norm,
                weight_linear=w_gatedup,
                output=mlp_mid,
                grid_dim=(grid_for_rmsnorm_linear_layer(w_gatedup.dim(0)), 1, 1),
                block_dim=(128, 1, 1),
            )
            w = mpk.attach_input(
                torch_tensor=layer.mlp.down_proj.weight, name=f"layer_{i}_down_proj"
            )
            mpk.silu_mul_linear_with_residual_layer(
                input=mlp_mid,
                weight=w,
                residual=x,
                output=mlp_out,
                grid_dim=(hidden_size // 64, 1, 1),
                block_dim=(128, 1, 1),
            )
            x = mlp_out
            if world_size > 1:
                mpk.allreduce_layer(
                    input=mlp_out,
                    buffer=allreduce_buf,
                    output=mlp_final,
                    grid_dim=(hidden_size // 64, 1, 1),
                    block_dim=(128, 1, 1),
                )
                x = mlp_final

        w_norm = mpk.attach_input(
            torch_tensor=model.model.norm.weight, name="model_norm_weight"
        )
        w_proj = mpk.attach_input(torch_tensor=lm_head_weight, name="lm_head")
        mpk.rmsnorm_linear_layer(
            input=x,
            weight_norm=w_norm,
            weight_linear=w_proj,
            output=argmax_in,
            grid_dim=(grid_for_rmsnorm_linear_layer(w_proj.dim(0)), 1, 1),
            block_dim=(128, 1, 1),
        )
        mpk.argmax_partial_layer(
            input=argmax_in,
            output=(argmax_part_value, argmax_part_index),
            grid_dim=(96, 1, 1),
            block_dim=(128, 1, 1),
        )
        mpk.argmax_reduce_layer(
            input=(argmax_part_value, argmax_part_index),
            output=argmax_out,
            grid_dim=(1, 1, 1),
            block_dim=(128, 1, 1),
        )

        results = mpk.kn_graph.generate_task_graph(num_gpus=world_size, my_gpu_id=rank)
        with open("task_graph.json", "w") as f:
            f.write(results["json_file"])
        with open("test.cu", "w") as f:
            f.write(results["cuda_code"])

        mpk.compile()

    stream = torch.cuda.Stream()
    warmup = 0
    output_len = 512
    if not args.use_mirage:
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
            if next_token == model.config.eos_token_id:
                break
            if cur_pos == prompt_len + warmup:
                torch.cuda.synchronize()
                starter.record()

        ender.record()
        torch.cuda.synchronize()
        run_time = starter.elapsed_time(ender)

        generated_ids = tokens[:, :prev_pos]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(response)
        print(
            "Prompt length {}, generate length {}, per-token latency {} ms".format(
                prompt_len, cur_pos - prompt_len, run_time / (cur_pos - prompt_len)
            )
        )
    else:
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
        starter.record()

        step.fill_(prompt_len)
        mpk()

        ender.record()
        torch.cuda.synchronize()
        run_time = starter.elapsed_time(ender)

        generated_ids = tokens[:, : step[0]]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(response)

        print(
            "Prompt length {}, generate length {}, per-token latency {} ms".format(
                prompt_len, step[0], run_time / (step[0] - warmup)
            )
        )
    if world_size > 1:
        dist.destroy_process_group()
