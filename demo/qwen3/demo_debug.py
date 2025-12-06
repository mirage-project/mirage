from models.modeling_qwen3 import Qwen3ForCausalLM
from transformers import AutoTokenizer, AutoConfig
from safetensors.torch import load_model
import torch
import torch.distributed as dist
import argparse
import os

# print limitation
torch.set_printoptions(profile="full")

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
    parser.add_argument("--max-num-batched-tokens", default=8, type=int, help="Max number of tokens in a batch")
    parser.add_argument("--max-num-batched-requests", default=1, type=int, help="Max number of requests in a batch")
    parser.add_argument("--page-size", default=4096, type=int, help="Page size")
    parser.add_argument("--max-num-pages", default=16, type=int, help="Max num pages")
    parser.add_argument("--output-dir", help="Output files directory")
    parser.add_argument("--trace-name", default="qwen3", help="Perfetto trace output name")
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
        default=1024,
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
            # model = Qwen3ForCausalLM(config, world_size, args.max_num_pages, args.page_size)
            # load_model(
            #     model, f"{args.model_path}/model{rank}-mp{world_size}.safetensors"
            # )
            model = Qwen3ForCausalLM.from_pretrained(args.model_path, world_size, max_num_pages=args.max_num_pages, page_size=args.page_size).to("cuda")
            tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        else:
            model = Qwen3ForCausalLM.from_pretrained(model_name, world_size=1, max_num_pages=args.max_num_pages, page_size=args.page_size).to("cuda")
            tokenizer = AutoTokenizer.from_pretrained(model_name)

    total_num_requests = 1
    # get all model weight tensors
    tokens = torch.full((total_num_requests, args.max_seq_length), 0, dtype=torch.long, device="cuda")

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
    for r in range(total_num_requests):
        for i in range(model_inputs.input_ids.shape[-1]):
            tokens[r, i] = model_inputs.input_ids[0, i]
    prompt_lengths = torch.full((total_num_requests,), model_inputs.input_ids.shape[-1], dtype=torch.int, device="cuda")
    positions = torch.arange(32768).unsqueeze(0).to(model.device)
    position_embeddings = model.model.rotary_emb(positions)

    # get all model weight tensors
    input_tokens = torch.full((args.max_num_batched_tokens, 1), 0, dtype=torch.long, device="cuda")
    output_tokens = torch.full((args.max_num_batched_tokens, 1), 0, dtype=torch.long, device="cuda")
    prev_pos = 0

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    step = torch.full((total_num_requests, ), 0, dtype=torch.int32, device="cuda")
    num_new_tokens = torch.full((total_num_requests, ), 1, dtype=torch.int32, device="cuda")

    if args.use_mirage:
        import mirage as mi

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
            profiler_tensor = torch.zeros(
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
        print("num_workers: ", num_workers)
        print("num_schedulers: ", num_schedulers)
        qo_indptr_buffer = torch.empty(
            args.max_num_batched_requests + 1, dtype=torch.int32, device="cuda")
        paged_kv_indptr_buffer = torch.empty(
            args.max_num_batched_requests + 1, dtype=torch.int32, device="cuda")
        paged_kv_indices_buffer = torch.empty(
            args.max_num_pages, dtype=torch.int32, device="cuda")
        paged_kv_last_page_len_buffer = torch.empty(
            args.max_num_batched_requests, dtype=torch.int32, device="cuda")
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
            meta_tensors={
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
            },
            profiler_tensor=profiler_tensor,
            trace_name=args.trace_name,
            spec_decode_config=spec_decode_config,
            use_cutlass_kernel=False,
        )
        
        if spec_decode_config and spec_decode_config.method == "promptlookup":
            all_tokens = mpk.attach_input(torch_tensor=tokens, name="all_tokens")
            num_tokens_extend = spec_decode_config.spec_length + 1
        else:
            num_tokens_extend = 1

        num_kv_cache_chunks = args.max_seq_length // 256
        # x = mpk.attach_input(torch_tensor=input_tokens, name="input_token")
        # x_torch = torch.full((8, 4096), 0.1, dtype=torch.bfloat16, device="cuda")
        # x_torch = torch.arange(8 * 4096, dtype=torch.bfloat16, device="cuda").reshape(8, 4096).reshape(8, 4096)
        x_torch = torch.randn((8, 4096), dtype=torch.bfloat16, device="cuda")
        w_qkv_torch = torch.randn((6144, 4096), dtype=torch.bfloat16, device="cuda")

        attn_in_torch = torch.randn((8, 6144), dtype=torch.bfloat16, device="cuda")
        # attn_in_torch = torch.full((8, 6144), 0.1, dtype=torch.bfloat16, device="cuda")
        # w_qkv_torch = 0.1 * torch.arange(6144 * 4096, dtype=torch.bfloat16, device="cuda").reshape(6144, 4096)
        
        w_q_norm_torch = torch.randn((128,), dtype=torch.bfloat16, device="cuda")
        w_k_norm_torch = torch.randn((128,), dtype=torch.bfloat16, device="cuda")

        k_cache_torch = torch.randn((16, 4096, 8, 128), dtype=torch.bfloat16, device="cuda") # (max_num_pages, page_size, num_local_kv_heads, head_dim)
        v_cache_torch = torch.randn((16, 4096, 8, 128), dtype=torch.bfloat16, device="cuda") # (max_num_pages, page_size, num_local_kv_heads, head_dim)
        # k_cache_torch = torch.full((16, 4096, 8, 128), 0.2, dtype=torch.bfloat16, device="cuda")
        # v_cache_torch = torch.full((16, 4096, 8, 128), 0.2, dtype=torch.bfloat16, device="cuda")

        cos_pos_embed_torch = torch.randn((4096, 128), dtype=torch.bfloat16, device="cuda")
        sin_pos_embed_torch = torch.randn((4096, 128), dtype=torch.bfloat16, device="cuda")
        attn_out_torch = torch.zeros(8, 4096, dtype=torch.bfloat16, device="cuda")
        # lse_torch = torch.zeros(8, num_kv_cache_chunks, 32, dtype=torch.float32, device="cuda")
        attn_out_tmp_torch = torch.zeros(8, num_kv_cache_chunks, 32 * 128, dtype=torch.bfloat16, device="cuda")
        # lse = mpk.new_tensor(
        #     dims=(args.max_num_batched_tokens, num_kv_cache_chunks, num_local_q_heads),
        #     dtype=mi.float32,
        #     name="lse",
        #     io_category="cuda_tensor",
        # )
        # attn_out_tmp = mpk.new_tensor(
        #     dims=(args.max_num_batched_tokens, num_kv_cache_chunks, num_local_q_heads * head_dim),
        #     dtype=mi.bfloat16,
        #     name="attn_out_tmp",
        #     io_category="cuda_tensor",
        # )
        attn_out_torch = torch.zeros(8, 32 * 128, dtype=torch.bfloat16, device="cuda")
        attn_out = mpk.attach_input(torch_tensor=attn_out_torch, name="attn_out")

        x = mpk.attach_input(torch_tensor=x_torch, name="input_x")
        attn_in = mpk.attach_input(torch_tensor=attn_in_torch, name="attn_in")
        w_qkv = mpk.attach_input(torch_tensor=w_qkv_torch, name="layer_0_qkv_proj")
        
        w_q_norm = mpk.attach_input(torch_tensor=w_q_norm_torch, name="layer_0_q_norm")
        w_k_norm = mpk.attach_input(torch_tensor=w_k_norm_torch, name="layer_0_k_norm")
        k_cache = mpk.attach_input(torch_tensor=k_cache_torch, name="layer_0_k_cache")
        v_cache = mpk.attach_input(torch_tensor=v_cache_torch, name="layer_0_v_cache")
        cos_pos_embed = mpk.attach_input(
            torch_tensor=cos_pos_embed_torch,
            name="cos_position_embedding",
        )
        sin_pos_embed = mpk.attach_input(
            torch_tensor=sin_pos_embed_torch,
            name="sin_position_embedding",
        )
        # lse = mpk.attach_input(torch_tensor=lse_torch, name="lse")
        lse = mpk.new_tensor(
            dims=(args.max_num_batched_tokens, num_kv_cache_chunks * num_local_q_heads // num_local_kv_heads, num_local_kv_heads),
            strides=(num_kv_cache_chunks * num_local_q_heads, 1, num_kv_cache_chunks * num_local_q_heads // num_local_kv_heads),
            dtype=mi.float32,
            name="lse",
            io_category="cuda_tensor",
        )
        # attn_out_tmp = mpk.attach_input(torch_tensor=attn_out_tmp_torch, name="attn_out_tmp")
        attn_out_tmp = mpk.new_tensor(
            dims=(args.max_num_batched_tokens, num_kv_cache_chunks * num_local_q_heads // num_local_kv_heads * head_dim, num_local_kv_heads),
            strides=(num_kv_cache_chunks * num_local_q_heads, 1, num_kv_cache_chunks * num_local_q_heads // num_local_kv_heads * head_dim),
            dtype=mi.bfloat16,
            name="attn_out_tmp",
            io_category="cuda_tensor",
        )
        # mpk.linear_layer(
        #     input=x,
        #     weight=w_qkv,
        #     output=attn_in,
        #     # grid_dim=(96, 1, 1),
        #     # grid_dim=(128, 1, 1),
        #     grid_dim=(64, 1, 1),
        #     block_dim=(128, 1, 1),
        # )

        # mpk.paged_attention_layer(
        #     input=attn_in,
        #     k_cache=k_cache,
        #     v_cache=v_cache,
        #     q_norm=w_q_norm,
        #     k_norm=w_k_norm,
        #     cos_pos_embed=cos_pos_embed,
        #     sin_pos_embed=sin_pos_embed,
        #     output=attn_out,
        #     grid_dim=(mpk.max_num_batched_requests, num_local_kv_heads, 1),
        #     block_dim=(128, 1, 1),
        # )

        mpk.paged_attention_split_kv_layer(
            input=attn_in,
            k_cache=k_cache,
            v_cache=v_cache,
            q_norm=w_q_norm,
            k_norm=w_k_norm,
            cos_pos_embed=cos_pos_embed,
            sin_pos_embed=sin_pos_embed,
            lse=lse,
            output=attn_out_tmp,
            grid_dim=(mpk.max_num_batched_requests, num_local_kv_heads, num_kv_cache_chunks),
            block_dim=(128, 1, 1),
        )

        mpk.paged_attention_split_kv_merge_layer(
            lse=lse,
            output_tmp=attn_out_tmp,
            output=attn_out,
            grid_dim=(mpk.max_num_batched_requests, num_local_kv_heads, 1),
            block_dim=(128, 1, 1),
        )

            # print("lse data ptr is: ", hex(lse_torch.data_ptr()))
            # print("attn_out_tmp data ptr is: ", hex(attn_out_tmp_torch.data_ptr()))
            # print("attn_out data ptr is: ", hex(attn_out_torch.data_ptr()))


            

        results = mpk.kn_graph.generate_task_graph(num_gpus=world_size, my_gpu_id=rank)
        with open(f"task_graph_{rank}.json", "w") as f:
            f.write(results["json_file"])
        with open(f"kernel_{rank}.cu", "w") as f:
            f.write(results["cuda_code"])

        mpk.compile(output_dir=args.output_dir)




        qo_indptr_buffer_2 = qo_indptr_buffer.clone()
        paged_kv_indptr_buffer_2 = paged_kv_indptr_buffer.clone()
        paged_kv_indices_buffer_2 = paged_kv_indices_buffer.clone()
        paged_kv_last_page_len_buffer_2 = paged_kv_last_page_len_buffer.clone()
        # ref with original paged attention layer
        mpk2 = mi.PersistentKernel(
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
            meta_tensors={
                "step": step.clone(),
                "tokens": tokens.clone(),
                "input_tokens": input_tokens.clone(),
                "output_tokens": output_tokens.clone(),
                "num_new_tokens": num_new_tokens.clone(),
                "prompt_lengths": prompt_lengths.clone(),
                "qo_indptr_buffer": qo_indptr_buffer_2,
                "paged_kv_indptr_buffer": paged_kv_indptr_buffer_2,
                "paged_kv_indices_buffer": paged_kv_indices_buffer_2,
                "paged_kv_last_page_len_buffer": paged_kv_last_page_len_buffer_2,
            },
            profiler_tensor=profiler_tensor,
            trace_name=args.trace_name + "_96",
            spec_decode_config=spec_decode_config,
            use_cutlass_kernel=False
        )

                # x = mpk.attach_input(torch_tensor=input_tokens, name="input_token")
        # x_torch = torch.full((8, 4096), 0.1, dtype=torch.bfloat16, device="cuda")
        x_torch_2 = x_torch.clone()
        # x_torch = torch.randn((8, 4096), dtype=torch.bfloat16, device="cuda")
        w_qkv_torch_2 = w_qkv_torch.clone()
        attn_in_torch_2 = attn_in_torch.clone()
        # w_qkv_torch = 0.1 * torch.arange(6144 * 4096, dtype=torch.bfloat16, device="cuda").reshape(6144, 4096)
        
        w_q_norm_torch_2 = w_q_norm_torch.clone()
        w_k_norm_torch_2 = w_k_norm_torch.clone()
        k_cache_torch_2 = k_cache_torch.clone()
        v_cache_torch_2 = v_cache_torch.clone()
        cos_pos_embed_torch_2 = cos_pos_embed_torch.clone()
        sin_pos_embed_torch_2 = sin_pos_embed_torch.clone()
        attn_out_torch_2 = attn_out_torch.clone()

        x_2 = mpk2.attach_input(torch_tensor=x_torch_2, name="input_x")
        attn_in_2 = mpk2.attach_input(torch_tensor=attn_in_torch_2, name="attn_in")
        w_qkv_2 = mpk2.attach_input(torch_tensor=w_qkv_torch_2, name="layer_0_qkv_proj")
        
        w_q_norm_2 = mpk2.attach_input(torch_tensor=w_q_norm_torch_2, name="layer_0_q_norm")
        w_k_norm_2 = mpk2.attach_input(torch_tensor=w_k_norm_torch_2, name="layer_0_k_norm")
        k_cache_2 = mpk2.attach_input(torch_tensor=k_cache_torch_2, name="layer_0_k_cache")
        v_cache_2 = mpk2.attach_input(torch_tensor=v_cache_torch_2, name="layer_0_v_cache")
        cos_pos_embed_2 = mpk2.attach_input(
            torch_tensor=cos_pos_embed_torch_2,
            name="cos_position_embedding",
        )
        sin_pos_embed_2 = mpk2.attach_input(
            torch_tensor=sin_pos_embed_torch_2,
            name="sin_position_embedding",
        )
        attn_out_2 = mpk2.attach_input(torch_tensor=attn_out_torch_2, name="layer_0_attn_out")

        # mpk2.linear_layer(
        #     input=x_2,
        #     weight=w_qkv_2,
        #     output=attn_in_2,
        #     grid_dim=(96, 1, 1),
        #     # grid_dim=(128, 1, 1),
        #     block_dim=(128, 1, 1),
        # )

        mpk2.paged_attention_layer(
            input=attn_in_2,
            k_cache=k_cache_2,
            v_cache=v_cache_2,
            q_norm=w_q_norm_2,
            k_norm=w_k_norm_2,
            cos_pos_embed=cos_pos_embed_2,
            sin_pos_embed=sin_pos_embed_2,
            output=attn_out_2,
            grid_dim=(mpk2.max_num_batched_requests, num_local_kv_heads, 1),
            block_dim=(128, 1, 1),
        )

        mpk2.compile(output_dir=args.output_dir + "_96")

    starter.record()
    mpk()
    ender.record()
    torch.cuda.synchronize()
    run_time = starter.elapsed_time(ender)

    starter.record()
    mpk2()
    ender.record()
    torch.cuda.synchronize()
    run_time_96 = starter.elapsed_time(ender)

    print("linear layer close?", torch.allclose(attn_in_torch, attn_in_torch_2, rtol=1e-2, atol=1e-2))
    print("paged attention layer close?", torch.allclose(attn_out_torch, attn_out_torch_2, rtol=1e-2, atol=1e-2))

    print("shape of attn_out_torch:")
    print(attn_out_torch.shape)
    # print(out_ref.shape)
    # print(attn_out_torch[0])
    # print(out_ref[0])
    print("first 10 elements of attn_in_torch:")
    print(attn_in_torch[0][:10])
    print("first 10 elements of attn_in_torch_2:")
    print(attn_in_torch_2[0][:10])
    print("first 10 elements of attn_out_torch:")
    print(attn_out_torch[0][:10])
    print("first 10 elements of attn_out_torch_2:")
    print(attn_out_torch_2[0][:10])
    # print("first 10 elements of lse_torch chunk 0:")
    # print(lse_torch[0][0][:10])
    # print("first 10 elements of lse_torch chunk 1:")
    # print(lse_torch[0][1][:10])

    # print("first 10 elements of attn_out_tmp_torch chunk 0:")
    # print(attn_out_tmp_torch[0][0][:10])
    # print("first 10 elements of attn_out_tmp_torch chunk 1:")
    # print(attn_out_tmp_torch[0][1][:10])


    if world_size > 1:
        dist.destroy_process_group()
