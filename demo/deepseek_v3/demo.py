from transformers import AutoTokenizer, AutoConfig
import torch
import torch.distributed as dist
import argparse
import os
import json

from mirage.mpk.models.deepseek_v3.builder import DeepSeekV3Builder
from mirage.mpk.models.graph_builder import MirageModelConfig


DEFAULT_SAVE_DIR = os.path.join("outputs", "deepseek_v3")
MAX_SAVE_TOKENS = 100

# DeepSeek V3 architecture constants
# MLA: 128 heads, compressed KV dim 512, rope dim 64, total head dim 576
DEEPSEEK_V3_NUM_HEADS = 128
DEEPSEEK_V3_KV_LORA_RANK = 512
DEEPSEEK_V3_QK_ROPE_HEAD_DIM = 64
DEEPSEEK_V3_HEAD_DIM_TOTAL = DEEPSEEK_V3_KV_LORA_RANK + DEEPSEEK_V3_QK_ROPE_HEAD_DIM  # 576


def grid_for_rmsnorm_linear_layer(size: int):
    if size / 96 > 400:
        assert size % 256 == 0, f"FATAL: Linear layer size not supported, it's {size}."
        return size // 256
    if size % 96 == 0:
        return 96
    elif size % 64 == 0:
        return 64


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


def run_correctness_test(args, state_dict, layer_indices, rank, world_size,
                         first_token_id=1):
    """Run PyTorch reference on selected layers and compare against MPK.

    Both use the SAME real weights from the checkpoint.
    Layer indices specify which layers to include (e.g., [0, 3] = 1 dense + 1 MoE).
    The MTP layer (index 61) is included if --mtp is set.
    """
    import torch.nn.functional as F
    import math

    device = f"cuda:{rank}"
    layer_indices = sorted(layer_indices)
    num_layers = len(layer_indices)
    include_mtp = args.mtp

    # DeepSeek V3 constants (after weight absorption)
    KV_LORA_RANK = DEEPSEEK_V3_KV_LORA_RANK  # 512
    QK_ROPE_HEAD_DIM = DEEPSEEK_V3_QK_ROPE_HEAD_DIM  # 64
    QK_HEAD_DIM = DEEPSEEK_V3_HEAD_DIM_TOTAL  # 576
    V_HEAD_DIM = KV_LORA_RANK  # 512
    NUM_Q_HEADS = DEEPSEEK_V3_NUM_HEADS // world_size
    HIDDEN = 7168
    FIRST_MOE = 3
    NUM_EXPERTS = 256
    TOPK = 8

    def rms_norm(x, weight, eps=1e-6):
        orig = x.dtype
        v = x.float().pow(2).mean(-1, keepdim=True)
        return (weight.float() * x.float() * torch.rsqrt(v + eps)).to(orig)

    def sigmoid_topk(logits, bias, k):
        scores = torch.sigmoid(logits.float())
        routing = scores + bias.float().unsqueeze(0)
        _, idx = torch.topk(routing, k, dim=-1)
        w = torch.gather(scores, 1, idx)
        return w / w.sum(dim=-1, keepdim=True), idx

    QK_NOPE = 128   # per-head nope dim (before absorption)
    V_ORIG = 128     # per-head V dim (before absorption)

    def dequant_fp8(weight, scale, block_k=128):
        """Dequantize FP8 weight using block-wise scale_inv.

        Weight: [M, K], Scale: [M/block_k, K/block_k].
        Each scale element covers a block_k × block_k tile.
        """
        w = weight.float()
        s = scale.float()
        if s.dim() == 2 and w.dim() == 2:
            # Expand scale to match weight shape
            s = s.repeat_interleave(block_k, dim=0)[:w.shape[0]]
            s = s.repeat_interleave(block_k, dim=1)[:, :w.shape[1]]
        return (w * s).to(torch.bfloat16)

    def fp8_linear(x, weight_key, sd, block_k=128):
        """Dequant FP8 weight then matmul."""
        w = dequant_fp8(sd[weight_key], sd[f"{weight_key}_scale_inv"], block_k)
        return F.linear(x.float(), w.float()).to(x.dtype)

    def mla_attention(hidden, prefix, sd, kv_cache, seq_pos, num_heads):
        """MLA attention following raw checkpoint flow (no weight absorption)."""
        bs = hidden.shape[0]
        p = prefix + "self_attn."

        # Q path: q_a_proj → norm → q_b_proj → split(q_nope, q_pe)
        q_a = fp8_linear(hidden, f"{p}q_a_proj.weight", sd)
        q_a = rms_norm(q_a, sd[f"{p}q_a_layernorm.weight"])
        q_full = fp8_linear(q_a, f"{p}q_b_proj.weight", sd)
        q_full = q_full.view(bs, num_heads, QK_NOPE + QK_ROPE_HEAD_DIM)  # [bs, H, 192]
        q_nope = q_full[:, :, :QK_NOPE]   # [bs, H, 128]
        q_pe = q_full[:, :, QK_NOPE:]      # [bs, H, 64]

        # KV path: kv_a_proj → split → norm(c_latent)
        kv_full = fp8_linear(hidden, f"{p}kv_a_proj_with_mqa.weight", sd)
        c_lat = kv_full[:, :KV_LORA_RANK]   # [bs, 512]
        k_pe_raw = kv_full[:, KV_LORA_RANK:]  # [bs, 64]
        c_lat = rms_norm(c_lat, sd[f"{p}kv_a_layernorm.weight"])

        # Cache write
        kv_new = torch.cat([c_lat, k_pe_raw], dim=-1)  # [bs, 576]
        for b in range(bs):
            kv_cache[seq_pos + b] = kv_new[b]
        kv_all = kv_cache[:seq_pos + bs]  # [kv_len, 576]

        # Weight absorption via kv_b_proj
        kv_b = dequant_fp8(sd[f"{p}kv_b_proj.weight"],
                           sd[f"{p}kv_b_proj.weight_scale_inv"])
        kv_b = kv_b.view(num_heads, V_ORIG + QK_NOPE, KV_LORA_RANK)  # [H, 256, 512]
        W_UK = kv_b[:, V_ORIG:, :]   # [H, 128, 512] K nope absorption
        W_UV = kv_b[:, :V_ORIG, :]   # [H, 128, 512] V absorption

        # Absorbed Q: q_nope_abs = q_nope @ W_UK → [bs, H, 512]
        q_nope_abs = torch.einsum('bhd,hdk->bhk', q_nope.float(), W_UK.float()).to(hidden.dtype)

        # Attention: Q_abs × c_kv^T + q_pe × k_pe^T
        k_nope = kv_all[:, :KV_LORA_RANK]   # [kv_len, 512]
        k_pe_all = kv_all[:, KV_LORA_RANK:]  # [kv_len, 64]
        s = (torch.einsum('bhd,sd->bhs', q_nope_abs.float(), k_nope.float()) +
             torch.einsum('bhd,sd->bhs', q_pe.float(), k_pe_all.float()))
        s = s / math.sqrt(QK_HEAD_DIM)
        attn_probs = F.softmax(s, dim=-1)

        # V absorption: attn @ c_kv → [bs, H, 512], then × W_UV^T → [bs, H, 128]
        attn_v = torch.einsum('bhs,sd->bhd', attn_probs, kv_all[:, :KV_LORA_RANK].float())
        attn_out = torch.einsum('bhd,hkd->bhk', attn_v, W_UV.float()).to(hidden.dtype)

        # o_proj
        flat = attn_out.reshape(bs, num_heads * V_ORIG)
        return fp8_linear(flat, f"{p}o_proj.weight", sd)

    def dense_mlp(hidden, prefix, sd):
        p = prefix + "mlp."
        gate = F.silu(fp8_linear(hidden, f"{p}gate_proj.weight", sd))
        up = fp8_linear(hidden, f"{p}up_proj.weight", sd)
        return fp8_linear(gate * up, f"{p}down_proj.weight", sd)

    def moe_mlp(hidden, prefix, sd):
        bs = hidden.shape[0]
        p = prefix + "mlp."
        # Router (BF16)
        logits = F.linear(hidden.float(), sd[f"{p}gate.weight"].float()).to(hidden.dtype)
        weights, topk_idx = sigmoid_topk(logits, sd[f"{p}gate.e_score_correction_bias"], TOPK)
        out = torch.zeros(bs, HIDDEN, device=device, dtype=hidden.dtype)
        # Routed experts (per-expert individual weights, FP8)
        for b in range(bs):
            for ki in range(TOPK):
                eid = topk_idx[b, ki].item()
                w = weights[b, ki].item()
                ep = f"{p}experts.{eid}."
                gate = F.silu(fp8_linear(hidden[b:b+1], f"{ep}gate_proj.weight", sd))
                up = fp8_linear(hidden[b:b+1], f"{ep}up_proj.weight", sd)
                down = fp8_linear(gate * up, f"{ep}down_proj.weight", sd)
                out[b] += w * down.squeeze(0)
        # Shared expert (FP8)
        sp = p + "shared_experts."
        sg = F.silu(fp8_linear(hidden, f"{sp}gate_proj.weight", sd))
        su = fp8_linear(hidden, f"{sp}up_proj.weight", sd)
        sd_out = fp8_linear(sg * su, f"{sp}down_proj.weight", sd)
        return out + sd_out

    print(f"\n{'='*60}")
    print(f"Correctness Test: layers={layer_indices}, mtp={include_mtp}")
    print(f"{'='*60}")

    # Run PyTorch reference — use same first token as MPK
    print(f"  Using first_token_id={first_token_id}")
    token_ids = torch.tensor([first_token_id], device=device, dtype=torch.long)
    max_seq = 64
    kv_caches = [torch.zeros(max_seq, QK_HEAD_DIM, device=device, dtype=torch.bfloat16)
                 for _ in range(num_layers + (1 if include_mtp else 0))]

    hidden = F.embedding(token_ids, state_dict["model.embed_tokens.weight"])
    if hidden.dim() == 1:
        hidden = hidden.unsqueeze(0)

    for cache_idx, layer_idx in enumerate(layer_indices):
        prefix = f"model.layers.{layer_idx}."
        normed = rms_norm(hidden, state_dict[f"{prefix}input_layernorm.weight"])
        attn_out = mla_attention(normed, prefix, state_dict, kv_caches[cache_idx], 0, NUM_Q_HEADS)
        hidden = hidden + attn_out
        normed = rms_norm(hidden, state_dict[f"{prefix}post_attention_layernorm.weight"])
        if layer_idx < FIRST_MOE:
            mlp_out = dense_mlp(normed, prefix, state_dict)
        else:
            mlp_out = moe_mlp(normed, prefix, state_dict)
        hidden = hidden + mlp_out

    hidden = rms_norm(hidden, state_dict["model.norm.weight"])
    logits = F.linear(hidden.float(), state_dict["lm_head.weight"].float())
    ref_token = logits.argmax(dim=-1).item()
    print(f"PyTorch reference output token: {ref_token}")
    print(f"PyTorch logits[0,:5]: {logits[0,:5].tolist()}")
    print(f"PyTorch reference completed successfully.")
    return ref_token


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepSeek V3 demo with Mirage megakernel")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to converted DeepSeek V3 weights")
    parser.add_argument("--use-mirage", action="store_true",
                        help="Use Mirage megakernel")
    parser.add_argument("--profiling", action="store_true",
                        help="Enable profiling to generate trace")
    parser.add_argument("--max-num-batched-tokens", default=8, type=int,
                        help="Max number of tokens in a batch")
    parser.add_argument("--max-num-batched-requests", default=1, type=int,
                        help="Max number of requests in a batch")
    parser.add_argument("--page-size", default=128, type=int,
                        help="Page size for KV cache")
    parser.add_argument("--max-num-pages", default=64, type=int,
                        help="Max number of pages")
    parser.add_argument("--max-seq-length", default=4096, type=int,
                        help="Max sequence length")
    parser.add_argument("--prompt", type=str,
                        default="Give me a short introduction to large language model.",
                        help="Input prompt text")
    parser.add_argument("--mtp", action="store_true",
                        help="Enable MTP speculative decoding")
    parser.add_argument("--num-speculative-tokens", default=1, type=int,
                        choices=range(1, 8),
                        help="Number of speculative tokens for MTP (1-7)")
    parser.add_argument("--rejection-sample-method", default="strict", type=str,
                        choices=["strict", "probabilistic", "synthetic"],
                        help="Rejection sampling method for speculative decoding")
    parser.add_argument("--output-dir", help="Output files directory")
    parser.add_argument("--trace-name", default="", help="Perfetto trace output name")
    parser.add_argument("--ignore-eos", action="store_true",
                        help="Ignore eos token during generation")
    parser.add_argument("--max-new-tokens", type=int, default=None,
                        help="Decode cap for CI determinism")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--do-sample", dest="do_sample", action="store_true",
                        help="Enable sampling (default off)")
    parser.add_argument("--save-tokens", nargs="?", const="auto", default=None,
                        help=(
                            "Optionally dump first N generated token_ids, text, and latency to JSON. "
                            "If path omitted, saves to outputs/deepseek_v3/{torch_output.json|mpk_output.json}."
                        ))
    # Developer correctness testing
    parser.add_argument("--correctness", action="store_true",
                        help="Run correctness test: compare MPK output against PyTorch reference")
    parser.add_argument("--layers", type=str, default=None,
                        help="Comma-separated list of layer indices to load (e.g. '0,3,60'). "
                             "Used with --correctness to test a reduced model.")

    args = parser.parse_args()

    # Multi-GPU setup via MPI
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

    if args.save_tokens:
        if args.save_tokens == "auto":
            filename = "mpk_output.json" if args.use_mirage else "torch_output.json"
            save_path = os.path.join(DEFAULT_SAVE_DIR, filename)
        else:
            save_path = args.save_tokens
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    else:
        save_path = None

    if world_size > 1:
        dist.init_process_group(backend="nccl", init_method="env://")
    global print
    if rank != 0:
        print = lambda *_, **__: None

    print("Input arguments:", args)
    print(f"world_size({world_size}) rank({rank})")
    torch.set_default_dtype(torch.bfloat16)
    torch.cuda.set_device(rank)

    # Load model config and tokenizer from converted weights
    print(f"Loading model config from: {args.model_path}")
    config = AutoConfig.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # Extract DeepSeek V3 architecture parameters
    hidden_size = config.hidden_size
    num_layers = config.num_hidden_layers
    vocab_size = config.vocab_size
    # MLA parameters
    kv_lora_rank = getattr(config, "kv_lora_rank", DEEPSEEK_V3_KV_LORA_RANK)
    qk_rope_head_dim = getattr(config, "qk_rope_head_dim", DEEPSEEK_V3_QK_ROPE_HEAD_DIM)
    ckv_kpe_dim = kv_lora_rank + qk_rope_head_dim  # 576 for DeepSeek V3
    num_attention_heads = config.num_attention_heads

    print(f"Model config: hidden_size={hidden_size}, num_layers={num_layers}, "
          f"vocab_size={vocab_size}, num_heads={num_attention_heads}, "
          f"kv_lora_rank={kv_lora_rank}, qk_rope_head_dim={qk_rope_head_dim}, "
          f"ckv_kpe_dim={ckv_kpe_dim}")

    total_num_requests = 1 if not args.use_mirage else args.max_num_batched_requests

    # Allocate token buffers
    tokens = torch.full(
        (total_num_requests, args.max_seq_length), 0, dtype=torch.long, device="cuda"
    )
    input_tokens = torch.full(
        (args.max_num_batched_tokens, 1), 0, dtype=torch.long, device="cuda"
    )
    output_tokens = torch.full(
        (args.max_num_batched_tokens, 1), 0, dtype=torch.long, device="cuda"
    )

    # Tokenize prompt
    messages = [
        {"role": "user", "content": args.prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to("cuda")
    for r in range(total_num_requests):
        for i in range(model_inputs.input_ids.shape[-1]):
            tokens[r, i] = model_inputs.input_ids[0, i]
    prompt_lengths = torch.full(
        (total_num_requests,), model_inputs.input_ids.shape[-1],
        dtype=torch.int, device="cuda"
    )

    step = torch.full((total_num_requests,), 0, dtype=torch.int32, device="cuda")
    num_new_tokens = torch.full((total_num_requests,), 1, dtype=torch.int32, device="cuda")

    starter, ender = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )

    if args.use_mirage:
        import mirage as mi

        # Pad vocab_size for task graph creation
        padded_vocab_size = ((vocab_size + 255) // 256) * 256

        if args.profiling:
            profiler_tensor = torch.zeros(
                3000 * 128, dtype=torch.uint64, device="cuda"
            ).contiguous()
        else:
            profiler_tensor = None

        # MTP speculative decoding config
        if args.mtp:
            spec_decode_config = mi.speculative.spec_decode_class(
                "lookahead",
                ngram_size=3,
                spec_length=args.num_speculative_tokens,
            )
        else:
            spec_decode_config = None

        num_workers, num_schedulers = mi.get_configurations_from_gpu(rank)

        # Meta tensor buffers for paged attention
        qo_indptr_buffer = torch.empty(
            args.max_num_batched_requests + 1, dtype=torch.int32, device="cuda"
        )
        paged_kv_indptr_buffer = torch.empty(
            args.max_num_batched_requests + 1, dtype=torch.int32, device="cuda"
        )
        paged_kv_indices_buffer = torch.empty(
            args.max_num_pages, dtype=torch.int32, device="cuda"
        )
        paged_kv_last_page_len_buffer = torch.empty(
            args.max_num_batched_requests, dtype=torch.int32, device="cuda"
        )

        # MLA uses a single combined ckv_kpe cache per layer
        # Shape: (num_layers, max_num_pages, page_size, ckv_kpe_dim)
        # where ckv_kpe_dim = kv_lora_rank + qk_rope_head_dim = 576
        ckv_kpe_cache = torch.zeros(
            (num_layers, args.max_num_pages, args.page_size, ckv_kpe_dim),
            dtype=torch.bfloat16,
            device="cuda",
        )

        eos_token_id = config.eos_token_id if not args.ignore_eos else -1
        # Handle eos_token_id being a list (common in DeepSeek V3)
        if isinstance(eos_token_id, list):
            eos_token_id = eos_token_id[0]

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
            eos_token_id=eos_token_id,
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
            use_cutlass_kernel=True,
        )

        # Load state dict from converted weights
        print(f"Loading model weights from: {args.model_path}")
        from safetensors.torch import load_file
        from safetensors import safe_open

        # Determine which layers we need
        layer_indices_for_load = None
        if args.correctness and args.layers:
            layer_indices_for_load = [int(x) for x in args.layers.split(',')]
            # Also need MTP layer if --mtp
            if args.mtp:
                layer_indices_for_load.append(num_layers)  # layer 61

        weight_file = os.path.join(
            args.model_path, f"model{rank}-mp{world_size}.safetensors"
        )
        if os.path.exists(weight_file):
            state_dict = load_file(weight_file, device="cuda")
        else:
            # Selective loading: only load needed layers from sharded files
            index_file = os.path.join(args.model_path, "model.safetensors.index.json")
            if os.path.exists(index_file) and layer_indices_for_load is not None:
                # Smart loading: use index to only load relevant shards/keys
                print(f"  Selective loading for layers: {layer_indices_for_load}")
                state_dict = {}
                with open(index_file) as f:
                    index = json.load(f)
                # Build key filter: global keys + selected layer keys
                needed_prefixes = ["model.embed_tokens.", "model.norm.", "lm_head."]
                for li in layer_indices_for_load:
                    needed_prefixes.append(f"model.layers.{li}.")
                # Group keys by shard file
                shard_to_keys = {}
                for key, shard in index["weight_map"].items():
                    if any(key.startswith(p) for p in needed_prefixes):
                        shard_to_keys.setdefault(shard, []).append(key)
                # Load only needed shards and keys
                for shard, keys in sorted(shard_to_keys.items()):
                    shard_path = os.path.join(args.model_path, shard)
                    print(f"  Loading {len(keys)} keys from {shard}")
                    with safe_open(shard_path, framework="pt", device="cuda") as f:
                        for key in keys:
                            state_dict[key] = f.get_tensor(key)
                print(f"  Loaded {len(state_dict)} keys total")
            else:
                # Full loading (no index or no layer filter)
                import glob
                shard_files = sorted(glob.glob(
                    os.path.join(args.model_path, "model-*.safetensors")
                ))
                if shard_files:
                    state_dict = {}
                    for shard_file in shard_files:
                        state_dict.update(load_file(shard_file, device="cuda"))
                else:
                    candidates = [
                        os.path.join(args.model_path, "model.safetensors"),
                    ]
                    state_dict = None
                    for candidate in candidates:
                        if os.path.exists(candidate):
                            state_dict = load_file(candidate, device="cuda")
                            break
                    if state_dict is None:
                        raise FileNotFoundError(
                            f"Could not find model weights at {args.model_path}. "
                            f"Expected {weight_file} or model.safetensors or model-*.safetensors"
                        )

        # Parse layer indices for correctness mode
        layer_indices_arg = None
        if args.correctness and args.layers:
            layer_indices_arg = [int(x) for x in args.layers.split(',')]

        # Correctness test: run PyTorch reference first (on raw weights)
        ref_token = None
        if args.correctness:
            test_layers = layer_indices_arg if layer_indices_arg else list(range(num_layers))
            first_tok = model_inputs.input_ids[0, 0].item()
            ref_token = run_correctness_test(
                args, state_dict, test_layers, rank, world_size,
                first_token_id=first_tok)

            # Convert weights for MPK builder:
            # - Keep FP8 weights + scale_inv as-is (builder uses FP8 GEMM pipeline)
            # - Only dequant kv_b_proj for absorption into q_b_proj
            # - Fuse gate+up for dense layers and per-expert weights
            print("\nConverting weights for MPK builder (in-memory, FP8 preserved)...")
            import sys
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), "models"))
            from convert import (
                dequantize_fp8, absorb_kv_into_q, get_model_params, is_fp8,
                find_scale_for_weight,
            )
            config_dict = AutoConfig.from_pretrained(args.model_path).to_dict()
            mp = get_model_params(config_dict)

            # Absorb kv_b_proj into q_b_proj (requires dequant for matmul)
            # After absorption, q_b_proj becomes BF16 (absorbed), kv_b_proj is deleted
            # Include MTP layer (61) if --mtp is set
            absorb_layers = list(test_layers)
            if args.mtp:
                absorb_layers.append(num_layers)  # layer 61
            for li in absorb_layers:
                attn = f"model.layers.{li}.self_attn."
                q_key = f"{attn}q_b_proj.weight"
                kv_key = f"{attn}kv_b_proj.weight"
                if q_key in state_dict and kv_key in state_dict:
                    # Dequant both for absorption (GPU)
                    q_w = state_dict[q_key]
                    kv_w = state_dict[kv_key]
                    q_s_key = f"{q_key}_scale_inv"
                    kv_s_key = f"{kv_key}_scale_inv"
                    if is_fp8(q_w) and q_s_key in state_dict:
                        q_bf16 = dequantize_fp8(q_w.cuda(), state_dict[q_s_key].cuda()).to(torch.bfloat16)
                    else:
                        q_bf16 = q_w.cuda().to(torch.bfloat16)
                    if is_fp8(kv_w) and kv_s_key in state_dict:
                        kv_bf16 = dequantize_fp8(kv_w.cuda(), state_dict[kv_s_key].cuda()).to(torch.bfloat16)
                    else:
                        kv_bf16 = kv_w.cuda().to(torch.bfloat16)
                    absorbed = absorb_kv_into_q(q_bf16, kv_bf16, mp).to(torch.bfloat16)
                    # Fuse V un-absorption into o_proj:
                    # o_proj_fused[h] = o_proj[h] @ W_UV[h]
                    # where W_UV[h] = kv_b_proj V part: [v_orig, kv_lora_rank]
                    num_heads_loc = mp["num_heads"]  # use full heads, shard later
                    qk_nope = mp["qk_nope_head_dim"]
                    v_dim = mp["v_head_dim"]
                    kv_lora_rank = mp["kv_lora_rank"]
                    kv_head_dim = qk_nope + v_dim
                    kv_b_reshaped = kv_bf16.reshape(num_heads_loc, kv_head_dim, kv_lora_rank)
                    W_UV = kv_b_reshaped[:, :v_dim, :]  # [H, v_dim, kv_lora_rank]

                    # Fuse into o_proj: o_proj [hidden, H*v_dim] → o_proj_fused [hidden, H*kv_lora]
                    o_key = f"{attn}o_proj.weight"
                    if o_key in state_dict:
                        o_w = state_dict[o_key]
                        if is_fp8(o_w):
                            o_s_key = f"{o_key}_scale_inv"
                            if o_s_key in state_dict:
                                o_bf16 = dequantize_fp8(o_w.cuda(), state_dict[o_s_key].cuda()).to(torch.bfloat16)
                                del state_dict[o_s_key]
                            else:
                                o_bf16 = o_w.cuda().to(torch.bfloat16)
                        else:
                            o_bf16 = o_w.cuda().to(torch.bfloat16)
                        # o_bf16: [hidden, H*v_dim] → reshape [hidden, H, v_dim]
                        hidden = o_bf16.shape[0]
                        o_reshaped = o_bf16.reshape(hidden, num_heads_loc, v_dim)
                        # o_fused[h] = o_reshaped[:, h, :] @ W_UV[h] → [hidden, kv_lora_rank]
                        # Batched: o_fused = einsum('dhn,hnk->dhk', o_reshaped, W_UV) → [hidden, H, kv_lora]
                        o_fused = torch.einsum('dhn,hnk->dhk', o_reshaped.float(), W_UV.float())
                        state_dict[o_key] = o_fused.reshape(hidden, num_heads_loc * kv_lora_rank).to(torch.bfloat16)
                        print(f"  Fused o_proj: [{hidden}, {num_heads_loc*v_dim}] → [{hidden}, {num_heads_loc*kv_lora_rank}]")

                    # Replace q_b_proj with absorbed BF16 version, remove scale
                    state_dict[q_key] = absorbed
                    if q_s_key in state_dict:
                        del state_dict[q_s_key]
                    # Remove kv_b_proj (absorbed into q)
                    del state_dict[kv_key]
                    if kv_s_key in state_dict:
                        del state_dict[kv_s_key]

            # Fuse gate_proj + up_proj for dense MLP layers (keep FP8)
            for li in absorb_layers:
                prefix = f"model.layers.{li}.mlp."
                gate_key = f"{prefix}gate_proj.weight"
                up_key = f"{prefix}up_proj.weight"
                if gate_key in state_dict and up_key in state_dict:
                    state_dict[f"{prefix}gate_up_proj.weight"] = torch.cat(
                        [state_dict.pop(gate_key), state_dict.pop(up_key)], dim=0)
                    # Fuse scales if present
                    gs_key = f"{gate_key}_scale_inv"
                    us_key = f"{up_key}_scale_inv"
                    if gs_key in state_dict and us_key in state_dict:
                        state_dict[f"{prefix}gate_up_proj.weight_scale_inv"] = torch.cat(
                            [state_dict.pop(gs_key), state_dict.pop(us_key)], dim=0)

            # Fuse per-expert weights into experts.w13/w2 tensors (keep FP8)
            for li in absorb_layers:
                ep = f"model.layers.{li}.mlp.experts."
                expert_keys = [k for k in list(state_dict.keys())
                               if k.startswith(ep) and ".gate_proj.weight" in k
                               and not k.endswith("_scale_inv")]
                if expert_keys:
                    n_exp = len(expert_keys)
                    w13_list, w2_list = [], []
                    s13_list, s2_list = [], []
                    has_scale = f"{ep}0.gate_proj.weight_scale_inv" in state_dict
                    for e in range(n_exp):
                        g = state_dict.pop(f"{ep}{e}.gate_proj.weight")
                        u = state_dict.pop(f"{ep}{e}.up_proj.weight")
                        d = state_dict.pop(f"{ep}{e}.down_proj.weight")
                        w13_list.append(torch.cat([g, u], dim=0))
                        w2_list.append(d)
                        if has_scale:
                            gs = state_dict.pop(f"{ep}{e}.gate_proj.weight_scale_inv")
                            us = state_dict.pop(f"{ep}{e}.up_proj.weight_scale_inv")
                            ds = state_dict.pop(f"{ep}{e}.down_proj.weight_scale_inv")
                            s13_list.append(torch.cat([gs, us], dim=0))
                            s2_list.append(ds)
                    state_dict[f"{ep}w13.weight"] = torch.stack(w13_list)
                    state_dict[f"{ep}w2.weight"] = torch.stack(w2_list)
                    if has_scale:
                        state_dict[f"{ep}w13.weight_scale_inv"] = torch.stack(s13_list)
                        state_dict[f"{ep}w2.weight_scale_inv"] = torch.stack(s2_list)

            # Ensure all tensors are on GPU and 16B-aligned
            for k in list(state_dict.keys()):
                t = state_dict[k]
                if not t.is_cuda:
                    t = t.cuda()
                if not t.is_contiguous():
                    t = t.contiguous()
                if t.data_ptr() % 16 != 0:
                    aligned = torch.empty_like(t)
                    aligned.copy_(t)
                    t = aligned
                state_dict[k] = t

            print(f"  Converted: {len(state_dict)} keys (FP8 weights preserved)")

        # Build MLA model config for the builder
        model_config = MirageModelConfig(
            hidden_size=hidden_size,
            intermediate_size=getattr(config, "intermediate_size", None) or getattr(config, "moe_intermediate_size", 18432),
            vocab_size=vocab_size,
            local_num_q_heads=num_attention_heads // world_size,
            local_num_kv_heads=1,  # MLA uses single KV head (shared latent)
            head_dim=ckv_kpe_dim,  # 576 for MLA
            num_layers=num_layers,
            k_cache=[ckv_kpe_cache[i] for i in range(num_layers)],
            v_cache=[ckv_kpe_cache[i] for i in range(num_layers)],
            position_embeddings=None,
            state_dict=state_dict,
            with_lm_head=True,
        )

        # Build the computation graph using the DeepSeek V3 builder
        builder = DeepSeekV3Builder(mpk)
        builder.build_from_config(model_config, layer_indices=layer_indices_arg)

        results = mpk.kn_graph.generate_task_graph(
            num_gpus=world_size, my_gpu_id=rank
        )
        with open(f"task_graph_{rank}.json", "w") as f:
            f.write(results["json_file"])
        with open(f"kernel_{rank}.cu", "w") as f:
            f.write(results["cuda_code"])

        mpk.compile(output_dir=args.output_dir)

        # Run inference
        print("Starting inference with Mirage megakernel...")
        starter.record()
        mpk()
        ender.record()
        torch.cuda.synchronize()
        run_time = starter.elapsed_time(ender)

        # Correctness comparison
        if args.correctness and ref_token is not None:
            mpk_token = output_tokens[0, 0].item()
            print(f"\n{'='*60}")
            print(f"Correctness comparison:")
            print(f"  PyTorch reference token: {ref_token}")
            print(f"  MPK output token:        {mpk_token}")
            if ref_token == mpk_token:
                print(f"  PASS: tokens match!")
            else:
                print(f"  FAIL: tokens differ!")
            print(f"{'='*60}\n")

        print("tokens.shape = ", tokens.shape)
        for r in range(total_num_requests):
            generated_ids = tokens[r, : step[r] + 1]
            response = tokenizer.decode(generated_ids, skip_special_tokens=True)
            print(response)

        if total_num_requests > 1:
            print(f"Output length of each batch is same: {(step.max() == step.min()).item()}")

        print("Prompt length {}, generate length {}, per-token latency (both prefill and decode): {:.3f} ms".format(
            prompt_lengths[0], step.max().item() + 1 - prompt_lengths[0],
            run_time / (step.max().item() + 1)
        ))

        # Dump outputs to json
        if save_path and rank == 0:
            end_idx = step[0].item() + 1
            prompt_len = prompt_lengths[0].item()
            tokens_generated = max(0, end_idx - prompt_len)
            per_tok_ms = run_time / max(tokens_generated, 1)
            slice_end = min(end_idx, prompt_len + MAX_SAVE_TOKENS)
            token_ids = tokens[0, prompt_len:slice_end].tolist()
            response_text = tokenizer.decode(
                tokens[0, :end_idx], skip_special_tokens=True
            )
            out = {
                "token_ids": token_ids,
                "text": response_text,
                "latency_ms_per_token": per_tok_ms,
                "prompt_length": prompt_len,
                "generate_length": tokens_generated,
                "mode": "mpk",
            }
            with open(save_path, "w") as f:
                json.dump(out, f, indent=2)
            print(f"Saved tokens to {save_path}")

    else:
        # Native PyTorch path (without Mirage)
        # DeepSeek V3 requires the model implementation for non-Mirage inference
        try:
            from transformers import AutoModelForCausalLM
            print(f"Loading DeepSeek V3 model from: {args.model_path}")
            with torch.device("cuda"):
                model = AutoModelForCausalLM.from_pretrained(
                    args.model_path,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                ).to("cuda")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load DeepSeek V3 model for native inference: {e}. "
                "For native PyTorch inference, ensure transformers supports the model "
                "or use --use-mirage for Mirage megakernel inference."
            )

        prompt_len = prompt_lengths[0].item()
        output_len = (
            args.max_new_tokens
            if args.max_new_tokens is not None
            else (tokens.size(1) - prompt_len)
        )
        output_len = max(0, min(output_len, tokens.size(1) - prompt_len))
        decode_limit = prompt_len + output_len
        prev_pos = 0
        stream = torch.cuda.Stream()

        for cur_pos in range(prompt_len, decode_limit):
            step.fill_(cur_pos - 1)
            input_ids = tokens[:1, prev_pos:cur_pos]
            with torch.no_grad():
                logits = model(input_ids=input_ids).logits
            next_token = logits[:, -1, :].argmax(dim=-1)
            tokens[0, cur_pos] = next_token[0]
            prev_pos = cur_pos
            eos_id = config.eos_token_id
            if isinstance(eos_id, list):
                if next_token[0].item() in eos_id:
                    break
            elif next_token[0].item() == eos_id:
                break
            if cur_pos == prompt_len:
                torch.cuda.synchronize()
                starter.record()

        ender.record()
        torch.cuda.synchronize()
        run_time = starter.elapsed_time(ender)

        end_idx = prev_pos + 1
        generated_ids = tokens[:1, :end_idx]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(response)
        print(
            "Prompt length {}, generate length {}, per-token latency {} ms".format(
                prompt_len, cur_pos - prompt_len,
                run_time / max(cur_pos - prompt_len, 1)
            )
        )

        # Dump outputs to json
        if save_path and rank == 0:
            tokens_generated = max(0, end_idx - prompt_len)
            per_tok_ms = run_time / max(tokens_generated, 1)
            slice_end = min(end_idx, prompt_len + MAX_SAVE_TOKENS)
            token_ids = tokens[0, prompt_len:slice_end].tolist()
            out = {
                "token_ids": token_ids,
                "text": tokenizer.decode(
                    tokens[0, :end_idx], skip_special_tokens=True
                ),
                "latency_ms_per_token": per_tok_ms,
                "prompt_length": prompt_len,
                "generate_length": tokens_generated,
                "mode": "torch",
            }
            with open(save_path, "w") as f:
                json.dump(out, f, indent=2)
            print(f"Saved tokens to {save_path}")

    if world_size > 1:
        dist.destroy_process_group()
