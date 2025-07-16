from models.modeling_qwen3 import Qwen3ForCausalLM
from transformers import AutoTokenizer, AutoConfig
from safetensors.torch import load_model
import torch
import torch.distributed as dist
import argparse
import os

# print limitation
# torch.set_printoptions(threshold=2000)

def grid_for_rmsnorm_linear_layer(size):
    # 96 and 64 are enough to cover all Qwen3 model? Please update the method
    # if you meet any incompatibility.
    if size % 96 == 0:
        return 96
    elif size % 64 == 0:
        return 64
    
# Return the largest factor of m that is less than or equal to n
# This is used to determine the grid size
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-mirage", action="store_true", help="Use Mirage kernels")
    parser.add_argument("--output-dir", help="Output files directory")
    parser.add_argument(
        "--profiling", action="store_true", help="Use Profiler to generate trace"
    )
    # lookahead or promptlookup
    parser.add_argument(
        "--spec-decode",
        default=None,
        choices=["promptlookup", "lookahead"],
        help="Enable speculative decoding with 'lookahead' or 'promptlookup' mode.",
    )
    parser.add_argument(
        "--ngram-size",
        default=3,
        type=int,
        help="Ngram size for lookahead spec decode",
    )
    parser.add_argument(
        "--max-seq-length",
        default=512,
        type=int,
        help="Max sequence length for lookahead spec decode",
    )
    parser.add_argument(
        "--spec-length",
        default=3,
        type=int,
        help="Spec length for lookahead spec decode",
    )

    parser.add_argument("--model-path", type=str, default=None, help="Path to a local model (necessary for multi-GPU demo)")
    parser.add_argument(
        "--model", type=str, default='Qwen/Qwen3-8B', help="Model path on hugging face"
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
        if args.model_path is not None:
            # load model locally (necessary for multi-GPU case)
            print(f"Load model from model path: {args.model_path}")
            config = AutoConfig.from_pretrained(args.model_path)
            model = Qwen3ForCausalLM(config, world_size)
            load_model(
                model, f"{args.model_path}/model{rank}-mp{world_size}.safetensors"
            )
            tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        else:
            model = Qwen3ForCausalLM.from_pretrained(model_name).to("cuda")
            tokenizer = AutoTokenizer.from_pretrained(model_name)

    # get all model weight tensors
    tokens = torch.full((1, 32768), 0, dtype=torch.long, device="cuda")

    # prompt = "Give me a short introduction to large language model."
    # This prompt is copied from https://github.com/apoorvumang/prompt-lookup-decoding/blob/main/demo-pld.ipynb
    code_text = """import numpy as np
                import matplotlib.pyplot as plt

                # Calculate the average
                average_throughput = np.mean(tokens_per_sec_arr)
                print(f"Average Throughput: {average_throughput} tokens/sec")

                # Plotting the histogram
                plt.hist(tokens_per_sec_arr, bins=20, color='blue', edgecolor='black', alpha=0.7)
                plt.title('Histogram of Throughput Values')
                plt.xlabel('Tokens per Second')
                plt.ylabel('Frequency')
                plt.axvline(average_throughput, color='red', linestyle='dashed', linewidth=1)
                plt.text(average_throughput*0.9, max(plt.ylim())*0.9, f'Average: {average_throughput:.2f}', color = 'red')
                plt.show()
                """
    question = "Can you please change x axis to start from 0"
    prompt = code_text + "\n" + question
    messages = [
        {
            "role": "system",
            "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
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

    # get all model weight tensors
    input_tokens = torch.full((1, 1), 0, dtype=torch.long, device="cuda")
    prev_pos = 0

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    step = torch.tensor([0], dtype=torch.int32, device="cuda")
    num_new_tokens = torch.tensor([1], dtype=torch.int32, device="cuda")
    if args.use_mirage:
        import mirage as mi

        batch_size = 1
        hidden_size = model.config.hidden_size
        intermediate_size = model.config.intermediate_size
        # pad vocab_size to facilitate task graph creation
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
        head_dim = model.config.head_dim
        fused_outdim_1 = (num_q_heads + 2 * num_kv_heads) * head_dim
        fused_outdim_2 = 2 * intermediate_size

        if args.profiling:
            profiler_tensor = torch.empty(
                3000 * 128, dtype=torch.uint64, device="cuda"
            ).contiguous()
        else:
            profiler_tensor = None
            
        spec_decode_config = mi.speculative.spec_decode_class(
            args.spec_decode,
            ngram_size=args.ngram_size,
            spec_length=args.spec_length,
        )
            
        num_workers, num_schedulers = mi.get_configurations_from_gpu(rank)
        mpk = mi.PersistentKernel(
            world_size=world_size,
            mpi_rank=rank,
            num_workers=num_workers,
            num_local_schedulers=num_schedulers,
            num_remote_schedulers=0,
            max_seq_length=args.max_seq_length,
            eos_token_id=model.config.eos_token_id,
            meta_tensors=[step, tokens, num_new_tokens],
            profiler_tensor=profiler_tensor,
            spec_decode_config=spec_decode_config,
        )
        
        if spec_decode_config and spec_decode_config.method == "promptlookup":
            all_tokens = mpk.attach_input(torch_tensor=tokens, name="all_tokens")
            argmax_batch_size = batch_size * (spec_decode_config.spec_length + 1)
            num_tokens_extend = spec_decode_config.spec_length + 1
        else:
            argmax_batch_size = batch_size
            num_tokens_extend = 1
        total_tokens_per_iter = batch_size * num_tokens_extend
        
        # TODO: Make the code run well even if 96 % total_tokens_per_iter != 0
        assert(96 % total_tokens_per_iter == 0)
        
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
            dims=(total_tokens_per_iter, hidden_size),
            dtype=mi.bfloat16,
            name="embed_out",
            io_category="cuda_tensor",
        )
        attn_in = mpk.new_tensor(
            dims=(total_tokens_per_iter, fused_outdim_1 // world_size), # [6, 6144]
            dtype=mi.bfloat16,
            name="attn_in",
            io_category="cuda_tensor",
        )
        attn_out = mpk.new_tensor(
            dims=(total_tokens_per_iter, num_local_q_heads * head_dim),
            dtype=mi.bfloat16,
            name="attn_out",
            io_category="cuda_tensor",
        )
        attn_proj_out = mpk.new_tensor(
            dims=(total_tokens_per_iter, hidden_size),
            dtype=mi.bfloat16,
            name="attn_proj_out",
            io_category="nvshmem_tensor" if world_size > 1 else "cuda_tensor",
        )
        allreduce_buf = mpk.new_tensor(
            dims=(world_size, total_tokens_per_iter, hidden_size),
            dtype=mi.bfloat16,
            name="all_reduce_buf",
            io_category="nvshmem_tensor" if world_size > 1 else "cuda_tensor",
        )
        attn_allreduce_out = mpk.new_tensor(
            dims=(total_tokens_per_iter, hidden_size),
            dtype=mi.bfloat16,
            name="attn_allreduce_out",
            io_category="nvshmem_tensor" if world_size > 1 else "cuda_tensor",
        )
        mlp_mid = mpk.new_tensor(
            dims=(total_tokens_per_iter, fused_outdim_2 // world_size),
            dtype=mi.bfloat16,
            name="mlp_mid",
            io_category="cuda_tensor",
        )
        mlp_out = mpk.new_tensor(
            dims=(total_tokens_per_iter, hidden_size),
            dtype=mi.bfloat16,
            name="mlp_out",
            io_category="nvshmem_tensor" if world_size > 1 else "cuda_tensor",
        )
        mlp_final = mpk.new_tensor(
            dims=(total_tokens_per_iter, hidden_size),
            dtype=mi.bfloat16,
            name="mlp_final",
            io_category="nvshmem_tensor" if world_size > 1 else "cuda_tensor",
        )

        argmax_in = mpk.new_tensor(
            dims=(total_tokens_per_iter, vocab_size),
            dtype=mi.bfloat16,
            name="argmax_in",
            io_category="cuda_tensor",
        )
        argmax_part_value = mpk.new_tensor(
            dims=(total_tokens_per_iter, 96 // (batch_size * num_tokens_extend)),
            dtype=mi.bfloat16,
            name="argmax_part_value",
            io_category="cuda_tensor",
        )
        argmax_part_index = mpk.new_tensor(
            dims=(total_tokens_per_iter, 96 // (batch_size * num_tokens_extend)),
            dtype=mi.int64,
            name="argmax_part_index",
            io_category="cuda_tensor",
        )
        argmax_out = mpk.new_tensor(
            dims=(1, total_tokens_per_iter),
            dtype=mi.int64,
            name="argmax_out",
            io_category="cuda_tensor",
        )

        # add spec tokens layer
        if spec_decode_config:
            spec_tokens = mpk.draft_forward_layer_dispatcher(
                spec_decode_config = spec_decode_config, 
                tokens = all_tokens,
                grid_dim=(96, 1, 1),
                block_dim=(128, 1, 1),
            )
            x = spec_tokens
        # Add Embed
        w = mpk.attach_input(
            torch_tensor=model.model.embed_tokens.weight, name="embed_tokens"
        )
        
        mpk.embed_layer(
            input=x, 
            weight=w, 
            output=y, 
            grid_dim=(max_factor_leq_n(4096, 96 // total_tokens_per_iter), total_tokens_per_iter, 1), 
            # grid_dim=(1, 1, 1), 
            block_dim=(128, 1, 1),
            input_source=(spec_decode_config is not None) # 0: all_tokens, 1: input_token (spec decoding)
        )
        x = y
        for i, layer in enumerate(model.model.layers):
            # add rmsnorm + linear
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
            # add attention
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
            # TODO: Later attention kernels should be merged as one
            if spec_decode_config:
                mpk.single_batch_extend_attention_layer(
                    input=attn_in,
                    k_cache=k_cache,
                    v_cache=v_cache,
                    q_norm=w_q_norm,
                    k_norm=w_k_norm,
                    cos_pos_embed=cos_pos_embed,
                    sin_pos_embed=sin_pos_embed,
                    output=attn_out,
                    grid_dim=(1, num_local_kv_heads, 1), #TODO: further divide across batch dim
                    block_dim=(128, 1, 1),
                )
            else:
                mpk.attention_layer(
                    input=attn_in,
                    k_cache=k_cache,
                    v_cache=v_cache,
                    q_norm=w_q_norm,
                    k_norm=w_k_norm,
                    cos_pos_embed=cos_pos_embed,
                    sin_pos_embed=sin_pos_embed,
                    output=attn_out,
                    grid_dim=(total_tokens_per_iter, num_local_kv_heads, 1),
                    block_dim=(128, 1, 1),
                )
            # add linear w/ residual
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
            # reset residual input as x
            x = attn_proj_out
            # add allreduce if needed
            if world_size > 1:
                mpk.allreduce_layer(
                    input=attn_proj_out,
                    buffer=allreduce_buf,
                    output=attn_allreduce_out,
                    grid_dim=(hidden_size // 64, 1, 1),
                    block_dim=(128, 1, 1),
                )
                x = attn_allreduce_out
            # add rmsnorm_linear layer
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
            # add silu_mul_linear layer
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
            # reset residual input as x
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

        # add rmsnorm_linear layer
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
        # add argmax layer
        if spec_decode_config and spec_decode_config.method == "promptlookup":
            argmax_partial_grid_dim = (max_factor_leq_n(153600, 96 // (spec_decode_config.spec_length + 1)), 
                                       spec_decode_config.spec_length + 1, 
                                       1)
            argmax_reduce_grid_dim = (1, spec_decode_config.spec_length + 1, 1)
        else:
            argmax_partial_grid_dim = (96, 1, 1)
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
        if spec_decode_config:
            verify_out = mpk.verify_layer_dispatcher(
                spec_decode_config = spec_decode_config,
                spec_tokens = spec_tokens,
                target_output = argmax_out,
                grid_dim = (1, 1, 1),
                block_dim = (128, 1, 1),
            )

        results = mpk.kn_graph.generate_task_graph(num_gpus=world_size, my_gpu_id=rank)
        with open(f"task_graph_{rank}.json", "w") as f:
            f.write(results["json_file"])
        with open(f"kernel_{rank}.cu", "w") as f:
            f.write(results["cuda_code"])

        mpk.compile(output_dir=args.output_dir)

    # g = torch.cuda.CUDAGraph()
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
        # prefill phase
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

        generated_ids = tokens[:, : step[0] + 1]
        # print(f"generated_ids: {generated_ids}")

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(response)

        print(
            "Prompt length {}, generate length {}, per-token latency {} ms".format(
                prompt_len, step[0] - prompt_len, run_time / (step[0] - prompt_len)
            )
        )
    if world_size > 1:
        dist.destroy_process_group()
