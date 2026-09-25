import modal

HF_CACHE_DIR = "/hf_cache"
MIRAGE_HOME_DIR = "/mirage"

MIRAGE_BRANCH = "mpk"
MIRAGE_COMMIT_HASH = "a6a00e1"

MAX_SEQ_LEN = 32768
MAX_CONTEXT_LEN = 4096
MAX_OUTPUT_LEN = 512

cuda_version = "12.9.1"
flavor = "devel"
operating_system = "ubuntu24.04"
cudnn_image_tag = f"{cuda_version}-cudnn-{flavor}-{operating_system}"

nvidia_cuda_image = modal.Image.from_registry(
    f"nvidia/cuda:{cudnn_image_tag}", add_python="3.12"
).entrypoint([])

mirage_image = (
    nvidia_cuda_image
    .apt_install("git", "libopenmpi-dev", "libnvshmem3-cuda-12", "libnvshmem3-static-cuda-12", "libnvshmem3-dev-cuda-12")
    .pip_install(
        "huggingface_hub[hf-transfer]==0.33.1",
        "mpi4py==4.1.0",
        "torch==2.7.1",
        "transformers==4.52.4",
        extra_index_url="https://download.pytorch.org/whl/cu128"
    )
    .env(
        {
            "HF_HUB_CACHE" : HF_CACHE_DIR,
            "HF_HUB_ENABLE_HF_TRANSFER" : "1",
            "PMIX_MCA_gds": "hash",
            "MIRAGE_HOME" : MIRAGE_HOME_DIR,
            "OMPI_ALLOW_RUN_AS_ROOT" : "1",
            "OMPI_ALLOW_RUN_AS_ROOT_CONFIRM" : "1",
            "NVSHMEM_LIB_PATH" : "/usr/lib/x86_64-linux-gnu/nvshmem/12",
            "MPI_INC_PATH" : "/usr/lib/x86_64-linux-gnu/openmpi/include",
            "MPI_LIB_PATH" : "/usr/lib/x86_64-linux-gnu/openmpi/lib"
        }
    )
    .run_commands(
        f"git clone --recursive --branch {MIRAGE_BRANCH} https://www.github.com/mirage-project/mirage.git {MIRAGE_HOME_DIR}",
        f"cd {MIRAGE_HOME_DIR} && git checkout {MIRAGE_COMMIT_HASH}",
        f"uv pip install --system -e {MIRAGE_HOME_DIR} -v"
    )
)

app = modal.App("mirage_qwen3_demo", image=mirage_image)

with mirage_image.imports():
    import importlib
    import os
    import sys

    import torch
    import torch.distributed as dist
    from safetensors.torch import load_model
    from transformers import AutoTokenizer, AutoConfig

    sys.path.insert(0, MIRAGE_HOME_DIR)
    module = importlib.import_module("demo.qwen3.models.modeling_qwen3")
    Qwen3ForCausalLM = module.Qwen3ForCausalLM

@app.cls(
    gpu="A100",
    volumes={
        HF_CACHE_DIR: modal.Volume.from_name("huggingface-cache", create_if_missing=True)
    },
    secrets=[
        modal.Secret.from_name("huggingface-secret")
    ],
)
class MirageQwen3Demo:
    model_name: str = modal.parameter(default="Qwen/Qwen3-8B")
    system_prompt: str = modal.parameter(default="You are Qwen, created by Alibaba Cloud. You are a helpful assistant.")
    use_mirage: bool = modal.parameter(default=False)
    profiling: bool = modal.parameter(default=False)

    def _grid_for_rmsnorm_linear_layer(self, size):
        if size % 96 == 0:
            return 96
        elif size % 64 == 0:
            return 64

    def _setup_distributed_environment(self):
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

    def _load_model_and_tokenizer(self, rank):
        print(f"Loading model: {self.model_name}")
        torch.set_default_dtype(torch.bfloat16)
        torch.cuda.set_device(rank)

        with torch.device("cuda"):
            model = Qwen3ForCausalLM.from_pretrained(self.model_name).to("cuda")

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        print("Model and tokenizer loaded.")
        return model, tokenizer

    def _build_mirage_graph(self, model, world_size, rank, tokens_tensor, step_tensor):
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
        head_dim = model.config.head_dim #hidden_size // num_q_heads
        fused_outdim_1 = (num_q_heads + 2 * num_kv_heads) * head_dim
        fused_outdim_2 = 2 * intermediate_size

        # --- Profiler Setup ---
        profiler_tensor = (
            torch.empty(3000 * 128, dtype=torch.uint64, device="cuda").contiguous()
            if self.profiling
            else None
        )

        # --- Persistent Kernel Setup ---
        num_workers, num_schedulers = mi.get_configurations_from_gpu(rank)
        mpk = mi.PersistentKernel(
            world_size=world_size,
            mpi_rank=rank,
            num_workers=num_workers,
            num_local_schedulers=num_schedulers,
            num_remote_schedulers=0,
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
        embed_out = mpk.new_tensor(dims=(batch_size, hidden_size), dtype=mi.bfloat16, name="embed_out", io_category="cuda_tensor")
        attn_in = mpk.new_tensor(dims=(batch_size, fused_outdim_1 // world_size), dtype=mi.bfloat16, name="attn_in", io_category="cuda_tensor")
        attn_out = mpk.new_tensor(dims=(batch_size, num_local_q_heads * head_dim), dtype=mi.bfloat16, name="attn_out", io_category="cuda_tensor")

        is_nvshmem = "nvshmem_tensor" if world_size > 1 else "cuda_tensor"
        attn_proj_out = mpk.new_tensor(dims=(batch_size, hidden_size), dtype=mi.bfloat16, name="attn_proj_out", io_category=is_nvshmem)
        allreduce_buf = mpk.new_tensor(dims=(world_size, batch_size, hidden_size), dtype=mi.bfloat16, name="all_reduce_buf", io_category=is_nvshmem)
        attn_allreduce_out = mpk.new_tensor(dims=(batch_size, hidden_size), dtype=mi.bfloat16, name="attn_allreduce_out", io_category=is_nvshmem)
        mlp_mid = mpk.new_tensor(dims=(batch_size, fused_outdim_2 // world_size), dtype=mi.bfloat16, name="mlp_mid", io_category="cuda_tensor")
        mlp_out = mpk.new_tensor(dims=(batch_size, hidden_size), dtype=mi.bfloat16, name="mlp_out", io_category=is_nvshmem)
        mlp_final = mpk.new_tensor(dims=(batch_size, hidden_size), dtype=mi.bfloat16, name="mlp_final", io_category=is_nvshmem)
        argmax_in = mpk.new_tensor(dims=(batch_size, vocab_size), dtype=mi.bfloat16, name="argmax_in", io_category="cuda_tensor")
        argmax_part_value = mpk.new_tensor(dims=(batch_size, 96), dtype=mi.bfloat16, name="argmax_part_value", io_category="cuda_tensor")
        argmax_part_index = mpk.new_tensor(dims=(batch_size, 96), dtype=mi.int64, name="argmax_part_index", io_category="cuda_tensor")
        argmax_out = mpk.new_tensor(dims=(batch_size, 1), dtype=mi.int64, name="argmax_out", io_category="cuda_tensor")

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
            mpk.rmsnorm_linear_layer(input=x, weight_norm=w_norm_attn, weight_linear=w_qkv, output=attn_in, grid_dim=(self._grid_for_rmsnorm_linear_layer(w_qkv.dim(0)), 1, 1), block_dim=(128, 1, 1))

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
            mpk.rmsnorm_linear_layer(input=x, weight_norm=w_norm_mlp, weight_linear=w_gatedup, output=mlp_mid, grid_dim=(self._grid_for_rmsnorm_linear_layer(w_gatedup.dim(0)), 1, 1), block_dim=(128, 1, 1))

            w_down_proj = mpk.attach_input(torch_tensor=layer.mlp.down_proj.weight, name=f"layer_{i}_down_proj")
            mpk.silu_mul_linear_with_residual_layer(input=mlp_mid, weight=w_down_proj, residual=residual_mlp, output=mlp_out, grid_dim=(hidden_size // 64, 1, 1), block_dim=(128, 1, 1))
            x = mlp_out

            if world_size > 1:
                mpk.allreduce_layer(input=mlp_out, buffer=allreduce_buf, output=mlp_final, grid_dim=(hidden_size // 64, 1, 1), block_dim=(128, 1, 1))
                x = mlp_final

        # Final layer
        w_final_norm = mpk.attach_input(torch_tensor=model.model.norm.weight, name="model_norm_weight")
        w_lm_head = mpk.attach_input(torch_tensor=lm_head_weight, name="lm_head")
        mpk.rmsnorm_linear_layer(input=x, weight_norm=w_final_norm, weight_linear=w_lm_head, output=argmax_in, grid_dim=(self._grid_for_rmsnorm_linear_layer(w_lm_head.dim(0)), 1, 1), block_dim=(128, 1, 1))

        # Argmax
        mpk.argmax_partial_layer(input=argmax_in, output=(argmax_part_value, argmax_part_index), grid_dim=(96, 1, 1), block_dim=(128, 1, 1))
        mpk.argmax_reduce_layer(input=(argmax_part_value, argmax_part_index), output=argmax_out, grid_dim=(1, 1, 1), block_dim=(128, 1, 1))

        mpk.compile()
        return mpk

    def _run_pytorch_generation(self, model, tokens, prompt_len, step_tensor, position_embeddings):
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

    def _run_mirage_generation(self, model, mpk, tokens, prompt_len, step_tensor, position_embeddings):
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
        #meta_tensors_ptr = [tensor.data_ptr() for tensor in mpk.meta_tensors]
        #profiler_buffer_ptr = (
        #    mpk.profiler_tensor.data_ptr() if mpk.profiler_tensor is not None else 0
        #)
        #mpk.init_func(
        #    meta_tensors_ptr,
        #    profiler_buffer_ptr,
        #    mpk.mpi_rank,
        #    mpk.num_workers,
        #    mpk.num_local_schedulers,
        #    mpk.num_remote_schedulers,
        #)

        # Generation phase
        starter.record()
        step_tensor.fill_(prompt_len)
        mpk()
        ender.record()
        torch.cuda.synchronize()
        run_time = starter.elapsed_time(ender)
        end_pos = step_tensor[0].item()
        generated_len = end_pos - prompt_len

        return end_pos, run_time, generated_len

    @modal.enter()
    def enter(self):
        if self.profiling:
            print("Note: Profiling is enabled, which may impact performance.")

        self.world_size, self.rank = self._setup_distributed_environment()
        if self.rank == 0:
            print(f"World size: {self.world_size}, Rank: {self.rank}")

        self.model, self.tokenizer = self._load_model_and_tokenizer(self.rank)

        self.tokens = torch.full((1, MAX_SEQ_LEN), 0, dtype=torch.long, device="cuda")
        self.step_tensor = torch.tensor([0], dtype=torch.int32, device="cuda")

        self.mpk = None
        if self.use_mirage:
            if self.rank == 0:
                print("Using Mirage for generation.")
            self.mpk = self._build_mirage_graph(
                self.model, self.world_size, self.rank, self.tokens, self.step_tensor
            )
        else:
            if self.rank == 0:
                print("Using PyTorch for generation.")

        self.positions = torch.arange(MAX_SEQ_LEN).unsqueeze(0).to(self.model.device)
        self.position_embeddings = self.model.model.rotary_emb(self.positions)

        # Hardcoded prompt for testing
        prompt = "Give me a short introduction to large language model."
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        prompt_len = model_inputs.input_ids.shape[-1]
        self.tokens[0, :prompt_len] = model_inputs.input_ids[0, :]
        self.prompt_len = prompt_len

        if self.rank == 0:
            print(f"Prompt loaded. Length: {self.prompt_len}")

    @modal.method()
    def run(self, prompt: str = ""):
        if self.rank == 0:
            print("\nStarting generation...")

        if self.use_mirage:
            end_pos, run_time, generated_len = self._run_mirage_generation(
                self.model,
                self.mpk,
                self.tokens,
                self.prompt_len,
                self.step_tensor,
                self.position_embeddings,
            )
        else:
            end_pos, run_time, generated_len = self._run_pytorch_generation(
                self.model,
                self.tokens,
                self.prompt_len,
                self.step_tensor,
                self.position_embeddings,
            )

        if self.rank == 0 and generated_len > 0:
            generated_ids = self.tokens[:, :end_pos]
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[
                0
            ]

            prompt_ids = self.tokens[:, :self.prompt_len]
            prompt_text = self.tokenizer.batch_decode(prompt_ids, skip_special_tokens=True)[0]

            print("\n--- RESPONSE ---")
            print(response[len(prompt_text):])
            print("----------------")

            print(
                f"\nPrompt length: {self.prompt_len}, Generated length: {generated_len}, "
                f"Per-token latency: {run_time / generated_len:.2f} ms"
            )

        if self.profiling and self.rank == 0 and self.use_mirage:
            print("Writing profiler trace to `mirage_trace.json`...")
            self.mpk.get_profiler_trace()

@app.local_entrypoint()
def main(
    model_name: str = "Qwen/Qwen3-8B",
    system_prompt: str = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
    use_mirage: bool = False,
    profiling: bool = False,
):
    demo = MirageQwen3Demo(
        model_name=model_name,
        system_prompt=system_prompt,
        use_mirage=use_mirage,
        profiling=profiling,
    )
    demo.run.remote()
