from transformers import Qwen3MoeForCausalLM
from transformers import AutoTokenizer, AutoConfig
from safetensors.torch import load_model
import torch
import torch.distributed as dist
import argparse
import os

# ! Caveat
# If you want to run this script, you should do this manual modification:
# Change `include/mirage/persistent_kernel/tasks/blackwell/attention_sm100.cuh` line 43
# from `int MAX_TOKENS = 8` to `int MAX_TOKENS = 1`.
# And this demo currently cannot run with multigpu.

# print limitation
# torch.set_printoptions(threshold=2000)

def grid_for_rmsnorm_linear_layer(size: int):
    # 96 and 64 are enough to cover all Qwen3 model? Please update the method
    # if you meet any incompatibility.
    if size / 96 > 400:
        # TODO: An ad-hoc workaround for linear kernel, both MPK ptx and
        # cutlass version will output unexpected results (not same output for
        # same prompt) if the OUTPUT_SIZE is too big, try to figure it out.
        assert size % 256 == 0, "FATAL: Linear layer size not support, it's {size}."
        return size // 256
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
    parser.add_argument("--max-num-batched-tokens", default=1, type=int, help="Max number of tokens in a batch")
    parser.add_argument("--max-num-batched-requests", default=1, type=int, help="Max number of requests in a batch")
    parser.add_argument("--page-size", default=4096, type=int, help="Page size")
    parser.add_argument("--max-num-pages", default=16, type=int, help="Max num pages")
    parser.add_argument("--output-dir", help="Output files directory")
    parser.add_argument("--trace-name", default="", help="Perfetto trace output name")
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
        "--model", type=str, default='Qwen/Qwen3-30B-A3B', help="Model path on hugging face"
    )
    parser.add_argument(
        "--no-use-cutlass-kernel",
        action="store_false",
        dest="use_cutlass_kernel",
        default=True,
        help="Not use the cutlass version kernel.",
    )
    parser.add_argument("--ignore-eos", action="store_true", help="Ignore eos token during generation")
    parser.add_argument("--splitk-gate", action="store_true", help="Use split-k gating linear")
    parser.add_argument(
        "--eagle3", action="store_true",
        help="Enable Eagle3 speculative decoding (overrides --spec-decode)",
    )
    parser.add_argument(
        "--eagle3-draft-path", type=str,
        default="lmsys/SGLang-EAGLE3-Qwen3-30B-A3B-Instruct-2507-SpecForge-Nex",
        help="HF repo or local path for the Eagle3 draft model",
    )
    parser.add_argument(
        "--num-draft-steps", type=int, default=4,
        help="K: number of Eagle3 draft tokens generated per iteration",
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
            model = Qwen3MoeForCausalLM(config)
            load_model(
                model, f"{args.model_path}/model{rank}-mp{world_size}.safetensors"
            )
            tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        else:
            model = Qwen3MoeForCausalLM.from_pretrained(model_name).to("cuda")
            tokenizer = AutoTokenizer.from_pretrained(model_name)

    total_num_requests = 1 if not args.use_mirage else args.max_num_batched_requests
    # get all model weight tensors
    tokens = torch.full((total_num_requests, args.max_seq_length), 0, dtype=torch.long, device="cuda")

    prompt = "Give me a short introduction to large language model."
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
    #question = "Can you please change x axis to start from 0"
    #prompt = code_text + "\n" + question
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
    prompt_lengths = torch.full((total_num_requests,), model_inputs.input_ids.shape[-1], dtype=torch.int, device=model.device)
    positions = torch.arange(32768).unsqueeze(0).to(model.device)
    dummy_x_for_device = torch.empty(1, dtype=torch.bfloat16, device=model.device)
    position_embeddings = model.model.rotary_emb(dummy_x_for_device, positions)
    
    # kv_cache tensors
    key_cache_torch = torch.empty(
        (
            model.config.num_hidden_layers,
            args.max_num_pages,
            args.page_size,
            model.config.num_key_value_heads // world_size,
            model.config.head_dim,
        ),
        dtype=torch.bfloat16,
        device="cuda",
    )
    value_cache_torch = torch.empty(
        (
            model.config.num_hidden_layers,
            args.max_num_pages,
            args.page_size,
            model.config.num_key_value_heads // world_size,
            model.config.head_dim,
        ),
        dtype=torch.bfloat16,
        device="cuda",
    )

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
        intermediate_size = model.config.moe_intermediate_size
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
        num_experts = model.config.num_experts
        num_experts_per_tok = model.config.num_experts_per_tok

        if args.profiling:
            profiler_tensor = torch.zeros(
                3000 * 128, dtype=torch.uint64, device="cuda"
            ).contiguous()
        else:
            profiler_tensor = None
            
        if args.eagle3:
            # Eagle3: spec_length = K (draft tokens per iter). The main fwd's
            # mbt must equal K+1 so the next iter can verify K draft + 1 main
            # token in one pass. Enforce here.
            assert args.max_num_batched_tokens == args.num_draft_steps + 1, (
                f"Eagle3 requires max_num_batched_tokens == num_draft_steps + 1; "
                f"got mbt={args.max_num_batched_tokens}, K={args.num_draft_steps}")
            assert world_size == 1, "Eagle3 v1 supports only world_size=1"
            spec_decode_config = mi.mpk.spec_decode_class(
                "eagle3",
                spec_length=args.num_draft_steps,
                draft_model_path=args.eagle3_draft_path,
                rejection_sample_method="strict",
            )
        else:
            spec_decode_config = mi.mpk.spec_decode_class(
                args.spec_decode,
                ngram_size=args.ngram_size,
                spec_length=args.spec_length,
            )
            
        num_workers, num_schedulers = mi.get_configurations_from_gpu(rank)
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
            eos_token_id=model.config.eos_token_id if not args.ignore_eos else -1,
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
            use_cutlass_kernel=args.use_cutlass_kernel
        )
        
        if spec_decode_config and spec_decode_config.method == "promptlookup":
            all_tokens = mpk.attach_input(torch_tensor=tokens, name="all_tokens")
            num_tokens_extend = spec_decode_config.spec_length + 1
        else:
            num_tokens_extend = 1
        
        # TODO: Make the code run well even if 96 % max_num_batched_tokens != 0
        # assert(96 % args.max_num_batched_tokens == 0)
        
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
            dims=(args.max_num_batched_tokens, fused_outdim_1 // world_size), # [6, 6144]
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

        # === Eagle3 aux buffers (allocated once; written by copy_layer at
        # capture indices in the for-i loop below) ===
        if args.eagle3:
            num_main_layers = model.config.num_hidden_layers
            eagle3_capture_layer_ids = [
                2, num_main_layers // 2, num_main_layers - 3,
            ]
            print(f"Eagle3 aux capture layer ids = {eagle3_capture_layer_ids} "
                  f"(out of {num_main_layers} layers)")
            eagle3_aux_h0 = mpk.new_tensor(
                dims=(args.max_num_batched_tokens, hidden_size),
                dtype=mi.bfloat16, name="eagle3_aux_h0",
                io_category="cuda_tensor",
            )
            eagle3_aux_h1 = mpk.new_tensor(
                dims=(args.max_num_batched_tokens, hidden_size),
                dtype=mi.bfloat16, name="eagle3_aux_h1",
                io_category="cuda_tensor",
            )
            eagle3_aux_h2 = mpk.new_tensor(
                dims=(args.max_num_batched_tokens, hidden_size),
                dtype=mi.bfloat16, name="eagle3_aux_h2",
                io_category="cuda_tensor",
            )
            eagle3_aux_buffers = [eagle3_aux_h0, eagle3_aux_h1, eagle3_aux_h2]
        else:
            eagle3_capture_layer_ids = []
            eagle3_aux_buffers = []
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
        # TODO(Zhihao): a temporary solution to combine MoE gate_proj and up_proj into one linear 
        # layer on the torch side with extra memory requirements, need to have a shuffle kernel to do this properly
        moe_gate_up_proj_torch_weights = []
        moe_down_proj_torch_weights = []
        for layer in model.model.layers:
            moe_gate_up_proj_torch = []
            moe_down_proj_torch = []
            for expert_id in range(num_experts):
                moe_gate_up_proj_torch.append(torch.concat([layer.mlp.experts[expert_id].gate_proj.weight, layer.mlp.experts[expert_id].up_proj.weight], dim=0))
                moe_down_proj_torch.append(layer.mlp.experts[expert_id].down_proj.weight)
            del layer.mlp.experts
            layer.mlp.experts = {
                "gate_up_proj": torch.stack(moe_gate_up_proj_torch, dim=0),
                "down_proj": torch.stack(moe_down_proj_torch, dim=0),
            }
            torch.cuda.empty_cache()

        moe_gate_out = mpk.new_tensor(
            dims=(args.max_num_batched_tokens, num_experts),
            dtype=mi.bfloat16,
            name="moe_gate_out",
            io_category="cuda_tensor",
        )
        moe_routing_indices = mpk.new_tensor(
            dims=(num_experts, args.max_num_batched_tokens),
            dtype=mi.int32,
            name="moe_routing_indices",
            io_category="cuda_tensor",
        )
        moe_mask = mpk.new_tensor(
            dims=(num_experts + 1,),
            dtype=mi.int32,
            name="moe_mask",
            io_category="cuda_tensor",
        )
        moe_topk_weight = mpk.new_tensor(
            dims=(args.max_num_batched_tokens, num_experts_per_tok),
            dtype=mi.float32,
            name="moe_topk_weight",
            io_category="cuda_tensor",
        )
        mlp_mid = mpk.new_tensor(
            dims=(args.max_num_batched_tokens, num_experts_per_tok, fused_outdim_2 // world_size),
            dtype=mi.bfloat16,
            name="mlp_mid",
            io_category="cuda_tensor",
        )
        silu_mul_out = mpk.new_tensor(
            dims=(args.max_num_batched_tokens, num_experts_per_tok, intermediate_size // world_size),
            dtype=mi.bfloat16,
            name="silu_mul_out",
            io_category="cuda_tensor",
        )
        mlp_out = mpk.new_tensor(
            dims=(args.max_num_batched_tokens, num_experts_per_tok, hidden_size),
            dtype=mi.bfloat16,
            name="mlp_out",
            io_category="cuda_tensor"
        )
        mlp_weighted_sum_out = mpk.new_tensor(
            dims=(args.max_num_batched_tokens, hidden_size),
            dtype=mi.bfloat16,
            name="mlp_weighted_sum_out",
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

        # add spec tokens layer (legacy: only for promptlookup; Eagle3 is wired
        # separately after argmax below — it shares the target's input_tokens
        # path and doesn't need a draft_forward_layer_dispatcher pre-pass).
        if spec_decode_config and spec_decode_config.method == "promptlookup":
            spec_tokens = mpk.draft_forward_layer_dispatcher(
                spec_decode_config = spec_decode_config,
                tokens = all_tokens,
                grid_dim=(96, 1, 1),
                block_dim=(256, 1, 1),
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
            grid_dim=(1, 1, 1), 
            block_dim=(256, 1, 1),
            input_source=1,
        )
        x = y
        for i, layer in enumerate(model.model.layers):
            # Eagle3 aux capture: copy the pre-layer hidden into the
            # corresponding aux buffer before the layer runs. Matches sglang's
            # qwen2.py:362 semantics (capture before layer i, after the
            # previous layer's residual add).
            if args.eagle3 and i in eagle3_capture_layer_ids:
                slot = eagle3_capture_layer_ids.index(i)
                mpk.copy_layer(
                    input=x, output=eagle3_aux_buffers[slot],
                    grid_dim=(1, 1, 1), block_dim=(256, 1, 1),
                )
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
            w_qkv = mpk.shuffle_tensors(
                inputs=[w_q, w_k, w_v],
                shuffled_dim=0,
                num_groups=model.config.num_key_value_heads // world_size,
                name=f"layer_{i}_qkv_proj",
            )
            mpk.rmsnorm_layer(
                input=x,
                weight=w_norm,
                output=rmsnorm_out,
                grid_dim=(mpk.max_num_batched_tokens, 1, 1),
                block_dim=(256, 1, 1),
            )
            mpk.linear_layer(
                input=rmsnorm_out,
                weight=w_qkv,
                output=attn_in,
                grid_dim=(grid_for_rmsnorm_linear_layer(w_qkv.dim(0)), 1, 1), # 5120 split into 64 tasks
                block_dim=(256, 1, 1),
            )
            # add attention
            w_q_norm = mpk.attach_input(
                torch_tensor=layer.self_attn.q_norm.weight, name=f"layer_{i}_q_norm"
            )
            w_k_norm = mpk.attach_input(
                torch_tensor=layer.self_attn.k_norm.weight, name=f"layer_{i}_k_norm"
            )
            k_cache = mpk.attach_input(
                torch_tensor=key_cache_torch[i], name=f"layer_{i}_k_cache"
            )
            v_cache = mpk.attach_input(
                torch_tensor=value_cache_torch[i], name=f"layer_{i}_v_cache"
            )
            # TODO: Later attention kernels should be merged as one
            # Eagle3 uses paged_attention (same as no-spec-decode path); only
            # promptlookup needs the extend variant for its multi-token verify.
            if spec_decode_config and spec_decode_config.method == "promptlookup":
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
                    block_dim=(256, 1, 1),
                )
            else:
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
                    block_dim=(256, 1, 1),
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
                grid_dim=(hidden_size // 128, 1, 1),
                block_dim=(256, 1, 1),
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
                    block_dim=(256, 1, 1),
                )
                x = attn_allreduce_out
            # add rmsnorm_linear layer
            w_norm = mpk.attach_input(
                torch_tensor=layer.post_attention_layernorm.weight,
                name=f"layer_{i}_post_attn_layernorm",
            )
            
            w_moe_gate = mpk.attach_input(
                torch_tensor=layer.mlp.gate.weight, name=f"layer_{i}_moe_gate"
            )
            w_gatedup = mpk.attach_input(
                torch_tensor=layer.mlp.experts["gate_up_proj"], name=f"layer_{i}_gateup_proj"
            )
            w_down_proj = mpk.attach_input(
                torch_tensor=layer.mlp.experts["down_proj"], name=f"layer_{i}_down_proj"
            )
            
            rmsnorm_num_tasks = grid_for_rmsnorm_linear_layer(w_gatedup.dim(1))
            mpk.rmsnorm_layer(
                input=x,
                weight=w_norm,
                output=rmsnorm_out,
                grid_dim=(mpk.max_num_batched_tokens, 1, 1),
                block_dim=(256, 1, 1),
            )
            
            if args.splitk_gate:
                # moe gate with split-k
                mpk.splitk_linear_layer(
                    input=rmsnorm_out,
                    weight=w_moe_gate,
                    output=moe_gate_out,
                    grid_dim=(1, hidden_size // 64, 1),
                    block_dim=(256, 1, 1),
                )
            else:
                # moe gate without split-k
                mpk.linear_layer(
                    input=rmsnorm_out,
                    weight=w_moe_gate,
                    output=moe_gate_out,
                    grid_dim=(1, 1, 1),
                    block_dim=(256, 1, 1),
                )
            # topk+softmax
            mpk.moe_topk_softmax_routing_layer(
                input=moe_gate_out,
                output=(moe_topk_weight, moe_routing_indices, moe_mask),
                grid_dim=(1, 1, 1),
                block_dim=(256, 1, 1),
            )
            # moe w13 linear
            mpk.moe_w13_linear_layer(
                input=rmsnorm_out,
                weight=w_gatedup,
                moe_routing_indices=moe_routing_indices,
                moe_mask=moe_mask,
                output=mlp_mid,
                grid_dim=(10, 12, 1),
                block_dim=(256, 1, 1),
            )
            mpk.moe_silu_mul_layer(
                input=mlp_mid,
                output=silu_mul_out,
                grid_dim=(mpk.max_num_batched_tokens, num_experts_per_tok, 1),
                block_dim=(256, 1, 1),
            )
            mpk.moe_w2_linear_layer(
                input=silu_mul_out,
                weight=w_down_proj,
                moe_routing_indices=moe_routing_indices,
                moe_mask=moe_mask,
                output=mlp_out,
                grid_dim=(8, 16, 1),
                block_dim=(256, 1, 1),
            )
            mpk.moe_mul_sum_add_layer(
                input=mlp_out,
                weight=moe_topk_weight,
                residual=x,
                output=mlp_weighted_sum_out,
                grid_dim=(mpk.max_num_batched_tokens, hidden_size//256, 1),
                block_dim=(256, 1, 1),
            )
            # reset residual input as x
            x = mlp_weighted_sum_out
            if world_size > 1:
                mpk.allreduce_layer(
                    input=mlp_weighted_sum_out,
                    buffer=allreduce_buf,
                    output=mlp_final,
                    grid_dim=(hidden_size // 64, 1, 1),
                    block_dim=(256, 1, 1),
                )
                x = mlp_final

        # add rmsnorm_linear layer
        w_norm = mpk.attach_input(
            torch_tensor=model.model.norm.weight, name="model_norm_weight"
        )
        w_proj = mpk.attach_input(torch_tensor=lm_head_weight, name="lm_head")
        mpk.rmsnorm_layer(
            input=x,
            weight=w_norm,
            output=rmsnorm_out,
            grid_dim=(mpk.max_num_batched_tokens, 1, 1),
            block_dim=(256, 1, 1),
        )
        mpk.linear_layer(
            input=rmsnorm_out,
            weight=w_proj,
            output=argmax_in,
            grid_dim=(mpk.num_workers, 1, 1),
            block_dim=(256, 1, 1),
        )
        # add argmax layer
        if spec_decode_config and spec_decode_config.method == "promptlookup":
            argmax_partial_grid_dim = (max_factor_leq_n(153600, 96 // (spec_decode_config.spec_length + 1)), 
                                       spec_decode_config.spec_length + 1, 
                                       1)
            argmax_reduce_grid_dim = (1, spec_decode_config.spec_length + 1, 1)
        else:
            argmax_partial_grid_dim = (mpk.num_workers, 1, 1)
            argmax_reduce_grid_dim = (1, 1, 1)
        mpk.argmax_partial_layer(
            input=argmax_in,
            output=(argmax_part_value, argmax_part_index),
            grid_dim=argmax_partial_grid_dim,
            block_dim=(256, 1, 1),
        )
        mpk.argmax_reduce_layer(
            input=(argmax_part_value, argmax_part_index),
            output=argmax_out,
            grid_dim=argmax_reduce_grid_dim,
            block_dim=(256, 1, 1),
        )
        if args.eagle3:
            # === Eagle3 draft + verify wiring ===
            #
            # Within one mpk() iteration the scheduler orders task execution
            # by data dependency. Verify reads `all_draft_ids` (written by the
            # PREVIOUS iter's draft loop) and current iter's argmax_out. Then
            # the draft loop overwrites `all_draft_ids` for NEXT iter.
            #
            # At iter 0 the all_draft_ids buffer is zero-initialized → verify
            # accepts 0 tokens, commits only the 1 bonus token. Prefill must
            # therefore be handled by pre-seeding `tokens` with the prompt
            # before the first mpk() call (existing demo pattern).
            from mirage.mpk.models.eagle3.builder import (
                Eagle3Builder, load_eagle3_draft,
            )

            print(f"Loading Eagle3 draft from {args.eagle3_draft_path}...")
            draft_sd, draft_cfg = load_eagle3_draft(args.eagle3_draft_path)
            eagle3 = Eagle3Builder(
                mpk=mpk,
                draft_state_dict=draft_sd,
                draft_config=draft_cfg,
                target_hidden_size=hidden_size,
                target_w_embed=w,  # shared target embed_tokens DTensor
                cos_pos_embed=cos_pos_embed,
                sin_pos_embed=sin_pos_embed,
                num_draft_steps=args.num_draft_steps,
                use_aux_norm=False,
            )

            # Verify (strict): compares last iter's draft against current
            # iter's target argmax. argmax_out (mbt × 1) holds the K+1 target
            # tokens for the K+1 verify positions (since mbt = K + 1).
            mbt_e = args.max_num_batched_tokens
            K = args.num_draft_steps
            accepted_count = mpk.new_tensor(
                dims=(mbt_e, 1), dtype=mi.int32,
                name="eagle3_accepted_count", io_category="cuda_tensor",
            )
            verified_output = mpk.new_tensor(
                dims=(mbt_e, K + 1), dtype=mi.int64,
                name="eagle3_verified_output", io_category="cuda_tensor",
            )
            # Note: `all_draft_ids` is built only after build_draft_loop runs
            # below, but the verify uses the PREVIOUS iter's contents. The
            # DTensor reference for naming is what matters for the task graph;
            # MPK's persistent tensors retain their content across iterations.
            # We pre-allocate it here so the verify task can reference it.
            eagle3_all_draft_ids_prealloc = mpk.new_tensor(
                dims=(mbt_e, K), dtype=mi.int64,
                name="eagle3_all_draft_ids", io_category="cuda_tensor",
            )
            mpk.mtp_verify_strict_layer(
                draft_token_ids=eagle3_all_draft_ids_prealloc,
                target_token_ids=argmax_out,
                accepted_count=accepted_count,
                output_tokens=verified_output,
                grid_dim=(mbt_e, 1, 1),
                block_dim=(128, 1, 1),
                num_draft_tokens=K,
            )

            # Build Eagle3 draft loop (writes all_draft_ids for next iter).
            # NOTE: Eagle3Builder will call mpk.new_tensor(...) for
            # "eagle3_all_draft_ids" via its _new cache; the prealloc above
            # uses the same name so the cache will reuse it.
            eagle3._attach_cache["eagle3_all_draft_ids"] = eagle3_all_draft_ids_prealloc
            eagle3.build_draft_loop(
                aux_h0=eagle3_aux_h0,
                aux_h1=eagle3_aux_h1,
                aux_h2=eagle3_aux_h2,
                target_argmax_token=argmax_out,
            )

            # NOTE (v1, K=1): we intentionally do NOT call mtp_prepare_verify_layer
            # here. That kernel writes main_token + draft tokens into
            # tokens[step+1..step+K+1], which during prefill (step still inside
            # prompt range) corrupts the prompt and produces garbage output. The
            # MTP DeepSeek path likely has a special prefill flow we haven't
            # replicated yet. For v1 we let the runtime use its default token
            # bookkeeping (advance by 1 per iter using output_tokens / argmax_out),
            # which means draft tokens are computed but never actually consumed —
            # output should be strict-identical to baseline (no spec decode),
            # validating the Eagle3 plumbing without paying the prefill-corruption
            # cost. K>1 functional benefit requires fixing prepare_verify (later).
        elif spec_decode_config:
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
        output_len = 3
        prompt_len = prompt_lengths[0].item()
        past_key_values = None

        input_ids = tokens[:, :1].clone()
        attention_mask = torch.ones_like(input_ids)

        with torch.inference_mode():
            out = model(
                input_ids=input_ids, 
                attention_mask=attention_mask,
                use_cache=True,
                past_key_values=None
            )
            past_key_values = out.past_key_values
            next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            print("decode step 1 out", out.logits[:, -1, :5], tokenizer.batch_decode(next_token[0], skip_special_tokens=True)[0])
        
        exit(0)
        
        prompt_len= prompt_lengths[0].item()
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
        starter.record()
        mpk()
        ender.record()
        torch.cuda.synchronize()
        run_time = starter.elapsed_time(ender)

        print("tokens.shape = ", tokens.shape)
        for r in range(total_num_requests):
            generated_ids = tokens[r, : step[r] + 1]
            print(f"{generated_ids=}")
            response = tokenizer.decode(generated_ids, skip_special_tokens=True)
            print(response)
        
        if total_num_requests > 1:
            print(f"Output length of each batch is same: {(step.max() == step.min()).item()}")

        print("Prompt length {}, generate length {}, per-token latency (both prefill and decode): {:.3f} ms".format(
              prompt_lengths[0], step.max().item() + 1 - prompt_lengths[0], run_time / (step.max().item() + 1)
            )
        )
        pass
    if world_size > 1:
        dist.destroy_process_group()
