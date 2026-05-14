from transformers import AutoTokenizer, AutoConfig
import torch
import torch.distributed as dist
import argparse
import os
import sys
import json
import socket
import re
import time
import hashlib

from mirage.mpk.models.deepseek_v3.builder import DeepSeekV3Builder
from mirage.mpk.models.graph_builder import MirageModelConfig


DEFAULT_SAVE_DIR = os.path.join("outputs", "deepseek_v3")
MAX_SAVE_TOKENS = 100
DEFAULT_PROFILER_BUFFER_ENTRIES = 6000 * 128

# DeepSeek V3 architecture constants
# MLA: 128 heads, compressed KV dim 512, rope dim 64, total head dim 576
DEEPSEEK_V3_NUM_HEADS = 128
DEEPSEEK_V3_KV_LORA_RANK = 512
DEEPSEEK_V3_QK_ROPE_HEAD_DIM = 64
DEEPSEEK_V3_HEAD_DIM_TOTAL = DEEPSEEK_V3_KV_LORA_RANK + DEEPSEEK_V3_QK_ROPE_HEAD_DIM  # 576


def get_profiler_buffer_entries() -> int:
    raw_value = os.environ.get("MPK_PROFILER_BUFFER_ENTRIES")
    if raw_value is None:
        return DEFAULT_PROFILER_BUFFER_ENTRIES
    entries = int(raw_value)
    if entries <= 0:
        raise ValueError("MPK_PROFILER_BUFFER_ENTRIES must be positive")
    return entries


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


def safe_tokenizer_decode(tokenizer, token_ids, *, context: str) -> str:
    if torch.is_tensor(token_ids):
        ids = [int(x) for x in token_ids.detach().cpu().reshape(-1).tolist()]
    else:
        ids = [int(x) for x in token_ids]

    try:
        vocab_size = len(tokenizer)
    except TypeError:
        vocab_size = getattr(tokenizer, "vocab_size", None)

    if vocab_size is not None:
        bad_ids = [tok for tok in ids if tok < 0 or tok >= vocab_size]
        if bad_ids:
            first = bad_ids[0]
            return (
                f"[decode skipped for {context}: {len(bad_ids)} token id(s) "
                f"outside tokenizer range [0, {vocab_size}); first={first}]"
            )

    try:
        return tokenizer.decode(ids, skip_special_tokens=True)
    except (OverflowError, ValueError) as exc:
        return f"[decode skipped for {context}: {type(exc).__name__}: {exc}]"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepSeek V3 demo with Mirage megakernel")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to converted DeepSeek V3 weights")
    parser.add_argument("--use-mirage", action="store_true",
                        help="Use Mirage megakernel")
    parser.add_argument("--profiling", action="store_true",
                        help="Enable profiling to generate trace")
    parser.add_argument("--profile-start-step", type=int, default=None,
                        help="Profiling-only: initialize the runtime step "
                             "after MPK compile so a single profiled graph "
                             "starts from an existing context position.")
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
    parser.add_argument("--prompts-json", type=str, default=None,
                        help="JSON array of prompts for batched-request testing. "
                             "When set with --use-mirage, its length must equal "
                             "--max-num-batched-requests.")
    parser.add_argument("--prompt-length", type=int, default=0,
                        help="If >0, override the --prompt with a synthetic "
                             "prompt of exactly this many tokens. Useful for "
                             "stress-testing prefill throughput independent of "
                             "actual prompt text. The generated tokens are a "
                             "deterministic cycle over a small vocab subset so "
                             "numerical behavior is reproducible across runs.")
    parser.add_argument("--mtp", type=int, default=0, choices=[0, 1, 2, 3],
                        help="MTP speculative decoding. 0=disabled, 1-3=number of "
                             "speculative tokens drafted per step.")
    parser.add_argument("--ep-size", type=int, default=1,
                        help="Expert-parallel group count for routed MoE experts. "
                             "Non-MoE layers and shared experts keep TP=world_size; "
                             "routed experts use TP=world_size/ep_size.")
    parser.add_argument("--disable-vocab-parallel-lm-head", action="store_true",
                        help="Disable the TP vocab-parallel LM head fast path. "
                             "By default it is enabled for TP>1 when MTP is disabled.")
    parser.add_argument("--rejection-sample-method", default="strict", type=str,
                        choices=["strict", "probabilistic", "synthetic"],
                        help="Rejection sampling method for speculative decoding")
    parser.add_argument("--output-dir", help="Output files directory")
    parser.add_argument("--trace-name", default="", help="Perfetto trace output name")
    parser.add_argument("--dump-task-graph", action="store_true",
                        help="Dump Mirage task graph JSON and generated CUDA code")
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
    parser.add_argument("--dump-hidden-dir", type=str, default=None,
                        help=(
                            "Diagnostic: dump per-layer residual stream after each "
                            "decoder layer's MLP/MoE residual fusion to "
                            "<dir>/layer_NN_residual.pt. Saved on rank 0 only. "
                            "Used to localize the layer where MPK first diverges "
                            "from the PyTorch reference."
                        ))
    parser.add_argument("--weight-cache-dir", type=str,
                        default=os.environ.get("MPK_DEEPSEEK_WEIGHT_CACHE_DIR"),
                        help=(
                            "Optional directory for MPK-ready per-rank weight cache. "
                            "When set, converted/fused/sharded tensors are saved after "
                            "the first run and loaded directly by later runs with the "
                            "same model/layers/TP/EP/vocab-parallel settings."
                        ))
    parser.add_argument("--layers", type=str, default=None,
                        help="Comma-separated list of layer indices to load (e.g. '0,3,60') "
                             "or a range '0-39'.")

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
        master_port = os.environ.get("MASTER_PORT")
        if master_port is None:
            if rank == 0:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    sock.bind(("127.0.0.1", 0))
                    master_port = str(sock.getsockname()[1])
            master_port = comm.bcast(master_port, root=0)
        os.environ["MASTER_PORT"] = master_port
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

    if args.use_mirage and args.mtp > 0:
        min_decode_tokens = args.mtp + 1
        supported_decode_token_caps = (1, 2, 4, 8, 16)
        required_decode_tokens = next(
            cap for cap in supported_decode_token_caps
            if cap >= min_decode_tokens
        )
        if args.max_num_batched_tokens < required_decode_tokens:
            print(
                "[mtp] Raising max_num_batched_tokens from "
                f"{args.max_num_batched_tokens} to {required_decode_tokens} "
                "so one decode/verify iteration can hold the main token plus "
                "MTP draft tokens on a supported FP8 runtime batch shape."
            )
            args.max_num_batched_tokens = required_decode_tokens

    print("Input arguments:", args)
    print(f"world_size({world_size}) rank({rank})")
    if args.ep_size < 1 or world_size % args.ep_size != 0:
        raise ValueError(
            f"--ep-size must divide world_size: ep_size={args.ep_size}, "
            f"world_size={world_size}"
        )
    if 256 % args.ep_size != 0:
        raise ValueError(f"--ep-size must divide 256 routed experts: {args.ep_size}")
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
    expected_config = {
        "hidden_size": 7168,
        "num_hidden_layers": 61,
        "num_attention_heads": DEEPSEEK_V3_NUM_HEADS,
        "kv_lora_rank": DEEPSEEK_V3_KV_LORA_RANK,
        "qk_rope_head_dim": DEEPSEEK_V3_QK_ROPE_HEAD_DIM,
    }
    for field_name, expected_value in expected_config.items():
        actual_value = getattr(config, field_name, None)
        if actual_value != expected_value:
            raise ValueError(
                f"DeepSeek V3 builder constant mismatch: config.{field_name}="
                f"{actual_value}, expected {expected_value}."
            )

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

    # Tokenize prompt (or synthesize a fixed-length prompt for stress tests)
    if args.prompt_length > 0:
        # Reserve at least 1 slot for the next decoded token. For prefill-only
        # diagnostics (max_seq_length == prompt_length), this collapses to a
        # one-iter run that terminates right after the prefill iter.
        pl = min(args.prompt_length, args.max_seq_length - 1)
        # Deterministic synthetic prompt: cycle over a small subset of the
        # vocab. Excludes special IDs (pad, bos, eos, etc.) by staying in
        # [1024, 1024 + 4096) which is safely inside any tokenizer's main
        # text vocabulary for DeepSeek V3 (vocab_size = 129280).
        synth = torch.arange(pl, dtype=torch.long, device="cuda") % 4096 + 1024
        for r in range(total_num_requests):
            tokens[r, :pl] = synth
        prompt_lengths = torch.full(
            (total_num_requests,), pl,
            dtype=torch.int, device="cuda"
        )
        print(f"[stress] Using synthetic prompt of length {pl} "
              f"(max_seq_length={args.max_seq_length}).")
    elif args.prompts_json:
        prompt_list = json.loads(args.prompts_json)
        if not isinstance(prompt_list, list) or not prompt_list:
            raise ValueError("--prompts-json must be a non-empty JSON array")
        if len(prompt_list) != total_num_requests:
            raise ValueError(
                f"--prompts-json length ({len(prompt_list)}) must equal "
                f"total_num_requests ({total_num_requests})"
            )
        messages_batch = [
            [{"role": "user", "content": prompt}]
            for prompt in prompt_list
        ]
        texts = [
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            for messages in messages_batch
        ]
        prompt_lens = []
        for r, text in enumerate(texts):
            model_inputs = tokenizer([text], return_tensors="pt").to("cuda")
            prompt_width = model_inputs.input_ids.shape[-1]
            if prompt_width > args.max_seq_length:
                raise ValueError(
                    f"Prompt width {prompt_width} exceeds max_seq_length "
                    f"{args.max_seq_length}"
                )
            tokens[r, :prompt_width] = model_inputs.input_ids[0, :prompt_width]
            prompt_lens.append(prompt_width)
        prompt_lengths = torch.tensor(
            prompt_lens, dtype=torch.int, device="cuda"
        )
        print(f"[batch] Using {len(prompt_list)} distinct prompts; "
              f"prompt_lengths={prompt_lengths.tolist()}")
    else:
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
        vocab_parallel_lm_head = (
            world_size > 1
            and args.mtp == 0
            and not args.disable_vocab_parallel_lm_head
        )
        lm_head_local_vocab_padded = None
        lm_head_vocab_offset = 0
        lm_head_valid_vocab_size = padded_vocab_size
        if vocab_parallel_lm_head:
            local_nominal = (padded_vocab_size + world_size - 1) // world_size
            lm_head_local_vocab_padded = ((local_nominal + 255) // 256) * 256
            lm_head_vocab_offset = rank * lm_head_local_vocab_padded
            lm_head_valid_vocab_size = max(
                0,
                min(lm_head_local_vocab_padded,
                    padded_vocab_size - lm_head_vocab_offset),
            )

        if args.profiling:
            profiler_tensor = torch.zeros(
                get_profiler_buffer_entries(), dtype=torch.uint64, device="cuda"
            ).contiguous()
        else:
            profiler_tensor = None

        # MTP speculative decoding config
        if args.mtp > 0:
            spec_decode_config = mi.spec_decode_class(
                "lookahead",
                ngram_size=3,
                spec_length=args.mtp,
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
            trace_name=(
                f"{args.trace_name}_rank{rank}" if args.trace_name else ""
            ),
            spec_decode_config=spec_decode_config,
            use_cutlass_kernel=True,
        )
        mpk.ep_size = args.ep_size
        mpk.deepseek_vocab_parallel_lm_head = vocab_parallel_lm_head
        mpk.deepseek_lm_head_vocab_offset = lm_head_vocab_offset
        mpk.deepseek_lm_head_valid_vocab_size = lm_head_valid_vocab_size
        mpk.deepseek_lm_head_local_vocab_padded = (
            lm_head_local_vocab_padded or padded_vocab_size
        )

        # Load state dict from converted weights
        print(f"Loading model weights from: {args.model_path}")
        load_start_time = time.perf_counter()
        from safetensors.torch import load_file, save_file
        from safetensors import safe_open

        def _parse_layers(spec):
            result = []
            for part in spec.split(','):
                if '-' in part:
                    lo, hi = part.split('-', 1)
                    result.extend(range(int(lo), int(hi) + 1))
                else:
                    result.append(int(part))
            return result

        # `layer_indices_arg` is what the builder consumes; `_for_load` is
        # the same set plus the MTP layer (we still have to load its weights).
        layer_indices_arg = _parse_layers(args.layers) if args.layers else None
        layer_indices_for_load = list(layer_indices_arg) if layer_indices_arg else None
        if layer_indices_for_load is not None and args.mtp > 0:
            layer_indices_for_load.append(num_layers)

        num_routed_experts_for_load = getattr(
            config,
            "n_routed_experts",
            getattr(config, "num_experts", 256),
        )
        assert world_size % args.ep_size == 0
        assert num_routed_experts_for_load % args.ep_size == 0
        routed_tp_size_for_load = world_size // args.ep_size
        ep_rank_for_load = rank // routed_tp_size_for_load
        local_num_experts_for_load = num_routed_experts_for_load // args.ep_size
        local_expert_start_for_load = ep_rank_for_load * local_num_experts_for_load
        local_expert_end_for_load = (
            local_expert_start_for_load + local_num_experts_for_load
        )
        local_expert_ids_for_load = (
            set(range(local_expert_start_for_load, local_expert_end_for_load))
            if args.ep_size > 1
            else None
        )
        expert_key_re = re.compile(r"^model\.layers\.\d+\.mlp\.experts\.(\d+)\.")

        cache_file = None
        state_dict_from_cache = False
        if args.weight_cache_dir:
            cache_payload = {
                "format": "deepseek_v3_mpk_fp8_runtime_cache_v1",
                "model_path": os.path.realpath(args.model_path),
                "layers": layer_indices_arg,
                "layers_for_load": layer_indices_for_load,
                "world_size": world_size,
                "rank": rank,
                "mtp": args.mtp,
                "ep_size": args.ep_size,
                "vocab_parallel_lm_head": vocab_parallel_lm_head,
                "lm_head_local_vocab_padded": lm_head_local_vocab_padded,
                "lm_head_vocab_offset": lm_head_vocab_offset,
                "lm_head_valid_vocab_size": lm_head_valid_vocab_size,
                "hidden_size": hidden_size,
                "vocab_size": vocab_size,
                "num_layers": num_layers,
            }
            cache_key = hashlib.sha256(
                json.dumps(cache_payload, sort_keys=True).encode("utf-8")
            ).hexdigest()[:24]
            cache_dir = os.path.join(args.weight_cache_dir, cache_key)
            cache_file = os.path.join(cache_dir, f"rank{rank}.safetensors")
            if os.path.exists(cache_file):
                cache_load_device = f"cuda:{rank}"
                print(f"  Loading MPK weight cache: {cache_file}")
                cache_load_start = time.perf_counter()
                state_dict = load_file(cache_file, device=cache_load_device)
                state_dict_from_cache = True
                print(
                    f"  Weight cache load time: "
                    f"{time.perf_counter() - cache_load_start:.3f}s"
                )

        weight_file = os.path.join(
            args.model_path, f"model{rank}-mp{world_size}.safetensors"
        )
        if state_dict_from_cache:
            pass
        elif os.path.exists(weight_file):
            state_dict = load_file(weight_file, device=f"cuda:{rank}")
        else:
            # Selective loading: only load needed layers from sharded files
            # * selective loading is what we want
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
                        expert_match = expert_key_re.match(key)
                        if (
                            local_expert_ids_for_load is not None
                            and expert_match is not None
                            and int(expert_match.group(1)) not in local_expert_ids_for_load
                        ):
                            continue
                        shard_to_keys.setdefault(shard, []).append(key)
                # Load only needed shards and keys
                # Load to CPU first, then move to GPU. This avoids GPU OOM when
                # multiple TP ranks each load full (un-sharded) weights — the
                # builder will shard them later during weight conversion.
                _load_device = "cpu" if world_size > 1 else f"cuda:{rank}"
                sliced_lm_head = False
                for shard, keys in sorted(shard_to_keys.items()):
                    shard_path = os.path.join(args.model_path, shard)
                    print(f"  Loading {len(keys)} keys from {shard}")
                    # safetensors loads only the requested keys from each shard.
                    # The granularity is still whole tensor per key; TP/EP
                    # sharding happens after absorption because q_b/o_proj
                    # fusion needs full per-layer matrices.
                    with safe_open(shard_path, framework="pt", device=_load_device) as f:
                        for key in keys:
                            if key == "lm_head.weight" and vocab_parallel_lm_head:
                                lm_head_slice = f.get_slice(key)
                                full_shape = lm_head_slice.get_shape()
                                tensor = lm_head_slice[
                                    lm_head_vocab_offset:
                                    lm_head_vocab_offset + lm_head_valid_vocab_size,
                                    :,
                                ]
                                if lm_head_valid_vocab_size < lm_head_local_vocab_padded:
                                    pad = torch.zeros(
                                        lm_head_local_vocab_padded
                                        - lm_head_valid_vocab_size,
                                        full_shape[1],
                                        dtype=tensor.dtype,
                                        device=tensor.device,
                                    )
                                    tensor = torch.cat([tensor, pad], dim=0)
                                state_dict[key] = tensor.contiguous()
                                sliced_lm_head = True
                            else:
                                state_dict[key] = f.get_tensor(key)
                print(f"  Loaded {len(state_dict)} keys total (device={_load_device})")
                if sliced_lm_head:
                    print(
                        "  Sliced lm_head.weight during load: "
                        f"rows {lm_head_vocab_offset}:"
                        f"{lm_head_vocab_offset + lm_head_valid_vocab_size} "
                        f"-> local padded {lm_head_local_vocab_padded}"
                    )
            else:
                raise RuntimeError("No valid weight files found for selective loading.")
        print(f"  Weight read time: {time.perf_counter() - load_start_time:.3f}s")

        # Weight conversion (absorption + gate/up + expert fusion) runs only
        # for selective loading. The pre-converted single-file path
        # (`model{rank}-mp{world_size}.safetensors`) is already absorbed/fused.
        if args.layers and not state_dict_from_cache:
            # Absorb kv_b_proj into q_b_proj and fuse V un-absorption into
            # o_proj (matches vLLM/SGLang runtime). The attached weights stay
            # in the same FP8+scale format that the multi-GPU graph expects.
            print("\nConverting weights for MPK builder (FP8 preserved)...")
            convert_start_time = time.perf_counter()
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), "models"))
            from convert import (
                dequantize_fp8, absorb_kv_into_q, get_model_params, is_fp8,
            )
            config_dict = AutoConfig.from_pretrained(args.model_path).to_dict()
            mp = get_model_params(config_dict)
            absorb_layers = list(layer_indices_arg) if layer_indices_arg else list(range(num_layers))
            if args.mtp > 0:
                absorb_layers.append(num_layers)

            def _quantize_f32_to_checkpoint_fp8(w_f32, block_size=128):
                """Quantize float32 weight to FP8 + per-2D-block scale_inv."""
                M, K = w_f32.shape
                bM = (M + block_size - 1) // block_size
                bK = (K + block_size - 1) // block_size
                w_pad = torch.zeros(bM * block_size, bK * block_size,
                                    dtype=torch.float32, device=w_f32.device)
                w_pad[:M, :K] = w_f32
                blocks = w_pad.reshape(bM, block_size, bK, block_size)
                amax = blocks.abs().amax(dim=(1, 3)).clamp(min=1e-12)
                scale_inv = amax / 448.0
                scale_exp = scale_inv.unsqueeze(1).unsqueeze(3).expand_as(blocks)
                w_q = (blocks / scale_exp).clamp(-448, 448)
                w_fp8 = w_q.reshape(bM * block_size, bK * block_size)[:M, :K].to(torch.float8_e4m3fn)
                return w_fp8, scale_inv

            for li in absorb_layers:
                attn = f"model.layers.{li}.self_attn."
                q_key = f"{attn}q_b_proj.weight"
                kv_key = f"{attn}kv_b_proj.weight"
                if q_key in state_dict and kv_key in state_dict:
                    q_w = state_dict[q_key]
                    kv_w = state_dict[kv_key]
                    q_s_key = f"{q_key}_scale_inv"
                    kv_s_key = f"{kv_key}_scale_inv"
                    if is_fp8(q_w) and q_s_key in state_dict:
                        q_f32 = dequantize_fp8(q_w.cuda(), state_dict[q_s_key].cuda())
                    else:
                        q_f32 = q_w.cuda().float()
                    if is_fp8(kv_w) and kv_s_key in state_dict:
                        kv_f32 = dequantize_fp8(kv_w.cuda(), state_dict[kv_s_key].cuda())
                    else:
                        kv_f32 = kv_w.cuda().float()
                    absorbed_f32 = absorb_kv_into_q(q_f32, kv_f32, mp)
                    q_fp8, q_scale = _quantize_f32_to_checkpoint_fp8(absorbed_f32)
                    state_dict[q_key] = q_fp8
                    state_dict[q_s_key] = q_scale
                    num_heads_loc = mp["num_heads"]
                    qk_nope = mp["qk_nope_head_dim"]
                    qk_rope = mp["qk_rope_head_dim"]
                    v_dim = mp["v_head_dim"]
                    kv_lora_rank = mp["kv_lora_rank"]
                    H_ = num_heads_loc

                    # Prefill follows vLLM's compute-friendly MLA path:
                    # original q_b_proj produces per-head q_nope/q_pe, while
                    # kv_b_proj decompresses cached c_latent into per-head
                    # K_nope/V. Decode keeps the absorbed q_b/o_proj path.
                    q_orig = q_f32.reshape(H_, qk_nope + qk_rope, -1)
                    q_b_nope_f32 = q_orig[:, :qk_nope, :].contiguous().reshape(
                        H_ * qk_nope, -1)
                    q_b_pe_f32 = q_orig[:, qk_nope:, :].contiguous().reshape(
                        H_ * qk_rope, -1)
                    q_b_nope_fp8, q_b_nope_scale = _quantize_f32_to_checkpoint_fp8(
                        q_b_nope_f32)
                    q_b_pe_fp8, q_b_pe_scale = _quantize_f32_to_checkpoint_fp8(
                        q_b_pe_f32)
                    state_dict[f"{attn}q_b_nope.weight"] = q_b_nope_fp8
                    state_dict[f"{attn}q_b_nope.weight_scale_inv"] = q_b_nope_scale
                    state_dict[f"{attn}q_b_pe.weight"] = q_b_pe_fp8
                    state_dict[f"{attn}q_b_pe.weight_scale_inv"] = q_b_pe_scale
                    # 2026-05-12 (user #2 FuseTensor v2 — row-swap layout):
                    # Build a single unabsorbed q_b weight so the prefill
                    # path can do ONE FP8 GEMM (instead of q_b_nope + q_b_pe).
                    # The per-rank layout is [rank_heads_nope_concat;
                    # rank_heads_pe_concat] so the 128-row FP8 quantization
                    # block boundaries align cleanly with the nope/pe split.
                    # The FULL weight is constructed as concatenation of
                    # per-rank slices so that the default dim=0 TP-sharding
                    # gives each rank its [nope; pe] block.
                    #
                    # Layout per rank (TP=4 → H_local=32):
                    #   rows [0,         H_local*128)  = nope of rank's heads
                    #   rows [H_local*128, H_local*192) = pe   of rank's heads
                    # Full weight = concat([rank_0_slice, rank_1_slice, ...]).
                    # Kernel reads Qn at base, Qp at base + H_local*128.
                    H_local_w = H_ // world_size
                    if H_ % world_size != 0:
                        raise ValueError(
                            f"num_heads {H_} not divisible by world_size "
                            f"{world_size} — row-swap fused q_b can't shard")
                    rank_parts = []
                    for r in range(world_size):
                        rank_heads = q_orig[r * H_local_w : (r + 1) * H_local_w]
                        # (H_local, 192, K)
                        rank_nope = rank_heads[:, :qk_nope, :].reshape(
                            H_local_w * qk_nope, -1)        # (H_local*128, K)
                        rank_pe = rank_heads[:, qk_nope:, :].reshape(
                            H_local_w * qk_rope, -1)        # (H_local*64, K)
                        rank_parts.append(rank_nope)
                        rank_parts.append(rank_pe)
                    q_b_unabs_f32 = torch.cat(rank_parts, dim=0).contiguous()
                    q_b_unabs_fp8, q_b_unabs_scale = (
                        _quantize_f32_to_checkpoint_fp8(q_b_unabs_f32))
                    state_dict[f"{attn}q_b_proj_unabsorbed.weight"] = q_b_unabs_fp8
                    state_dict[f"{attn}q_b_proj_unabsorbed.weight_scale_inv"] = q_b_unabs_scale

                    kv_head_dim = qk_nope + v_dim
                    kv_b_reshaped = kv_f32.reshape(
                        num_heads_loc, kv_head_dim, kv_lora_rank)
                    W_UK = kv_b_reshaped[:, :qk_nope, :]
                    W_UV = kv_b_reshaped[:, qk_nope:, :]
                    kv_b_k_f32 = W_UK.contiguous().reshape(H_ * qk_nope, kv_lora_rank)
                    kv_b_v_f32 = W_UV.contiguous().reshape(H_ * v_dim, kv_lora_rank)
                    kv_b_k_fp8, kv_b_k_scale = _quantize_f32_to_checkpoint_fp8(
                        kv_b_k_f32)
                    kv_b_v_fp8, kv_b_v_scale = _quantize_f32_to_checkpoint_fp8(
                        kv_b_v_f32)
                    state_dict[f"{attn}kv_b_k.weight"] = kv_b_k_fp8
                    state_dict[f"{attn}kv_b_k.weight_scale_inv"] = kv_b_k_scale
                    state_dict[f"{attn}kv_b_v.weight"] = kv_b_v_fp8
                    state_dict[f"{attn}kv_b_v.weight_scale_inv"] = kv_b_v_scale

                    # BMM repack of kv_b_k for MPK_DSV3_BMM=1 decode path:
                    # transpose W_UK (H, 128, 512) → (H, 512, 128) and
                    # per-row quantize. BMM kernel expects weight layout
                    # (H, D_out=512, D_in=128) FP8 + (H, 512, 1) uint32
                    # UE8M0-packed scale (1 packed scale per row since
                    # D_in=128 == one BLOCK_K). Row-major contiguous.
                    W_UK_bmm = W_UK.transpose(-2, -1).contiguous()  # (H, 512, 128)
                    bmm_amax = W_UK_bmm.abs().amax(dim=-1, keepdim=True).clamp(
                        min=1e-12)
                    bmm_scale_inv_f32 = (bmm_amax / 448.0).squeeze(-1)  # (H, 512)
                    bmm_w_q = (W_UK_bmm / bmm_amax).clamp(-448, 448).to(
                        torch.float8_e4m3fn).contiguous()  # (H, 512, 128) FP8
                    # UE8M0 encode (CEIL log2 to match kernel-side encode_ue8m0).
                    # Direct exponent math (bias = 127) avoids the
                    # float-as-int-bits trick which silently broke when
                    # torch.pow returned float64 (last dim halved on view
                    # to int32).
                    pos = torch.where(
                        bmm_scale_inv_f32 > 0, bmm_scale_inv_f32,
                        torch.full_like(bmm_scale_inv_f32, 1e-30))
                    log2_val = torch.ceil(torch.log2(pos))
                    ue = (log2_val + 127.0).clamp(0, 255).to(torch.uint8)
                    ue = torch.where(bmm_scale_inv_f32 > 0, ue,
                                     torch.zeros_like(ue))
                    # Pack into uint32 (1 packed scale per row at K=128 ≤ 512).
                    bmm_scale_packed = ue.to(torch.uint32).unsqueeze(-1).contiguous()
                    # ^ shape (H, 512, 1) uint32, low byte holds UE8M0 exponent.
                    state_dict[f"{attn}kv_b_k_bmm.weight"] = bmm_w_q
                    state_dict[f"{attn}kv_b_k_bmm.weight_scale_ue8m0"] = (
                        bmm_scale_packed)
                    # DEBUG 2026-05-10: also store bf16 versions of the
                    # split kv_b weights for the BF16 ablation in
                    # _fp8_dense_kv_b_proj. Used to verify whether the FP8
                    # GEMM path is the source of the layer-0 attn_proj
                    # orthogonality. Toggle via builder._kv_b_use_bf16.
                    state_dict[f"{attn}kv_b_k_bf16.weight"] = (
                        kv_b_k_f32.bfloat16().contiguous())
                    state_dict[f"{attn}kv_b_v_bf16.weight"] = (
                        kv_b_v_f32.bfloat16().contiguous())

                    # Preserve original W_O for PR674 prefill output. The
                    # existing o_proj key is replaced by the decode-only fused
                    # W_UV→W_O path below.
                    o_key = f"{attn}o_proj.weight"
                    if o_key in state_dict:
                        o_w = state_dict[o_key]
                        o_s_key = f"{o_key}_scale_inv"
                        state_dict[f"{attn}o_proj_original.weight"] = o_w
                        if o_s_key in state_dict:
                            state_dict[f"{attn}o_proj_original.weight_scale_inv"] = (
                                state_dict[o_s_key])
                        if is_fp8(o_w) and o_s_key in state_dict:
                            o_bf16 = dequantize_fp8(o_w.cuda(), state_dict[o_s_key].cuda()).to(torch.bfloat16)
                            del state_dict[o_s_key]
                        else:
                            o_bf16 = o_w.cuda().to(torch.bfloat16)
                        hidden_dim = o_bf16.shape[0]
                        o_reshaped = o_bf16.reshape(hidden_dim, num_heads_loc, v_dim)
                        o_fused_f32 = torch.einsum('dhn,hnk->dhk', o_reshaped.float(), W_UV.float())
                        o_flat = o_fused_f32.reshape(hidden_dim, num_heads_loc * kv_lora_rank)
                        o_fp8, o_scale = _quantize_f32_to_checkpoint_fp8(o_flat)
                        state_dict[o_key] = o_fp8
                        state_dict[o_s_key] = o_scale
                        print(
                            f"  L{li}: absorbed q_b {q_fp8.shape}, "
                            f"prefill q/nope {q_b_nope_fp8.shape}/"
                            f"{q_b_pe_fp8.shape}, kv_b K/V "
                            f"{kv_b_k_fp8.shape}/{kv_b_v_fp8.shape}, "
                            f"fused decode o_proj [{hidden_dim}, "
                            f"{num_heads_loc * kv_lora_rank}]")
                    del state_dict[kv_key]
                    if kv_s_key in state_dict:
                        del state_dict[kv_s_key]

            # Fuse gate_proj + up_proj for dense MLP layers (keep FP8).
            # IMPORTANT: silu_mul kernel reads `gate` from the first half of
            # each block and `up` from the second half (per-block layout). The
            # FP8 linear partitions the output dim by `grid_for_rmsnorm_linear_layer`
            # into N tasks, so silu_mul uses N/2 tasks. We must INTERLEAVE
            # gate/up at granularity = N/2 chunks per tensor (NOT torch.cat).
            from mirage.mpk.models.utils import shuffle_tensors as _shuffle_tensors
            from mirage.mpk.models.utils import grid_for_rmsnorm_linear_layer as _grid_fn
            for li in absorb_layers:
                prefix = f"model.layers.{li}.mlp."
                gate_key = f"{prefix}gate_proj.weight"
                up_key = f"{prefix}up_proj.weight"
                if gate_key in state_dict and up_key in state_dict:
                    g = state_dict.pop(gate_key)
                    u = state_dict.pop(up_key)
                    out_dim_total = g.shape[0] + u.shape[0]
                    split = _grid_fn(out_dim_total) // 2
                    state_dict[f"{prefix}gate_up_proj.weight"] = _shuffle_tensors(
                        [g, u], split=split, dim=0)
                    # Fuse scales with the SAME interleave (each scale row covers
                    # 128 weight rows, so the chunk-row count = chunk_weight_rows/128).
                    gs_key = f"{gate_key}_scale_inv"
                    us_key = f"{up_key}_scale_inv"
                    if gs_key in state_dict and us_key in state_dict:
                        gs = state_dict.pop(gs_key)
                        us = state_dict.pop(us_key)
                        state_dict[f"{prefix}gate_up_proj.weight_scale_inv"] = (
                            _shuffle_tensors([gs, us], split=split, dim=0))

            # 2026-05-12 (user #2 part-a) — Fuse q_a_proj + kv_a_proj_with_mqa
            # into a single qkv_a_proj.weight of shape (2176, hidden). Layout:
            #   rows [0     : 1536) = q_a_proj.weight              (12 fp8 blocks)
            #   rows [1536  : 2048) = kv_a_proj_with_mqa[:512]     (4  blocks, c_latent)
            #   rows [2048  : 2112) = kv_a_proj_with_mqa[512:576]  (k_pe real, 64 rows)
            #   rows [2112  : 2176) = zero-padded                  (k_pe pad block tail)
            # FP8 block boundaries fall cleanly at 128-row borders → no nope/pe
            # mixing within blocks. Scales just concat.
            #
            # Both q_a_proj and kv_a_proj_with_mqa are REPLICATED (no TP shard),
            # so the fused weight is also replicated. The fused 2176-wide
            # qkv_a_proj is the only path the MPK builder supports — the
            # 3-GEMM unfused path was removed (was +14.9% slower e2e).
            FUSED_ROWS = 2176  # 1536 q_a + 512 c_latent + 64 k_pe + 64 pad
            for li in absorb_layers:
                attn = f"model.layers.{li}.self_attn."
                q_a_key = f"{attn}q_a_proj.weight"
                kv_a_key = f"{attn}kv_a_proj_with_mqa.weight"
                if q_a_key not in state_dict or kv_a_key not in state_dict:
                    continue
                q_a_w = state_dict[q_a_key]
                kv_a_w = state_dict[kv_a_key]
                q_a_s_key = f"{q_a_key}_scale_inv"
                kv_a_s_key = f"{kv_a_key}_scale_inv"
                q_a_s = state_dict.get(q_a_s_key)
                kv_a_s = state_dict.get(kv_a_s_key)
                if q_a_s is None or kv_a_s is None:
                    raise RuntimeError(
                        f"layer {li}: q_a_proj/kv_a_proj_with_mqa FP8 scales "
                        f"missing — MPK requires FP8 DeepSeek-V3 weights.")
                assert q_a_w.shape[0] == 1536 and kv_a_w.shape[0] == 576, (
                    f"layer {li}: unexpected q_a/kv_a shape "
                    f"({q_a_w.shape}, {kv_a_w.shape})")
                K = q_a_w.shape[1]
                fused_w = torch.zeros(
                    (FUSED_ROWS, K), dtype=q_a_w.dtype, device=q_a_w.device)
                fused_w[0:1536] = q_a_w
                fused_w[1536:2048] = kv_a_w[:512]
                fused_w[2048:2112] = kv_a_w[512:576]
                # rows [2112:2176) stay zero (pad to 128-row MMA tile)
                scale_rows = FUSED_ROWS // 128
                fused_s = torch.zeros(
                    (scale_rows, q_a_s.shape[1]),
                    dtype=q_a_s.dtype, device=q_a_s.device)
                fused_s[0:12] = q_a_s
                fused_s[12:17] = kv_a_s
                state_dict[f"{attn}qkv_a_proj.weight"] = fused_w
                state_dict[f"{attn}qkv_a_proj.weight_scale_inv"] = fused_s.contiguous()

            # Fuse per-expert weights into experts.w13/w2 tensors (keep FP8)
            for li in absorb_layers:
                ep = f"model.layers.{li}.mlp.experts."
                expert_id_re = re.compile(
                    rf"^{re.escape(ep)}(\d+)\.gate_proj\.weight$"
                )
                expert_ids = []
                for k in list(state_dict.keys()):
                    m = expert_id_re.match(k)
                    if m is not None:
                        expert_ids.append(int(m.group(1)))
                expert_ids.sort()
                if expert_ids:
                    print(f"  Fusing {len(expert_ids)} experts for layer {li}")
                    w13_list, w2_list = [], []
                    s13_list, s2_list = [], []
                    has_scale = (
                        f"{ep}{expert_ids[0]}.gate_proj.weight_scale_inv"
                        in state_dict
                    )
                    for e in expert_ids:
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

            # Contiguous + 16B-align. For TP>1 keep tensors on CPU here —
            # the TP sharding step below halves them, and the post-shard
            # block (after line ~750) moves the small sharded copies to GPU.
            # Pushing full un-sharded weights to GPU here OOMs once the
            # layer count × world_size exceeds per-GPU capacity.
            for k in list(state_dict.keys()):
                t = state_dict[k]
                if world_size == 1 and not t.is_cuda:
                    t = t.cuda()
                if not t.is_contiguous():
                    t = t.contiguous()
                if t.data_ptr() % 16 != 0:
                    aligned = torch.empty_like(t)
                    aligned.copy_(t)
                    t = aligned
                state_dict[k] = t

            print(f"  Converted: {len(state_dict)} keys (FP8 weights preserved)")
            print(
                f"  Weight conversion time: "
                f"{time.perf_counter() - convert_start_time:.3f}s"
            )

            # TP/EP weight sharding happens after conversion. For TP>1, tensors
            # stay on CPU until this step to avoid GPU OOM from full
            # unsharded weights. A future shard-aware converter could reduce
            # CPU peak memory further, but the current path is deterministic
            # and keeps absorption/fusion math identical across ranks.
            if world_size > 1:
                from convert import shard_tensor
                import re
                num_routed_experts = getattr(
                    config,
                    "n_routed_experts",
                    getattr(config, "num_experts", 256),
                )
                assert num_routed_experts % args.ep_size == 0
                routed_tp_size = world_size // args.ep_size
                routed_tp_rank = rank % routed_tp_size
                ep_rank = rank // routed_tp_size
                local_num_experts = num_routed_experts // args.ep_size
                local_expert_start = ep_rank * local_num_experts
                local_expert_end = local_expert_start + local_num_experts
                # Sharding rules for POST-CONVERSION keys (absorbed, fused).
                # dim=0: row-parallel (shard output), dim=1: col-parallel (shard input),
                # None: replicate. For 3D expert tensors, None (replicated).
                _TP_SHARD_RULES = [
                    (r"^model\.embed_tokens\.weight$",                       None),  # replicate: embedding lookup needs full vocab
                    (r"^model\.norm\.weight$",                               None),
                    (r"^lm_head\.weight$",                                   None),  # replicate: needs full vocab for argmax
                    (r"self_attn\.q_a_proj\.weight",                         None),  # ReplicatedLinear (vLLM): hidden→q_lora_rank, output feeds full-width q_b_proj
                    (r"self_attn\.q_a_layernorm\.weight",                    None),
                    (r"self_attn\.q_b_proj\.weight",                         0),     # ColumnParallelLinear: shard output heads
                    # Prefill path uses original q_b split and kv_b
                    # decompression, both column-parallel on local heads.
                    (r"self_attn\.q_b_nope\.weight",                         0),
                    (r"self_attn\.q_b_pe\.weight",                           0),
                    # FuseTensor (user #2, 2026-05-12): unabsorbed q_b_proj
                    # used by the fused prefill path (MPK_DSV3_QB_FUSED=1).
                    # Same column-parallel sharding as q_b_proj.
                    (r"self_attn\.q_b_proj_unabsorbed\.weight",              0),
                    (r"self_attn\.kv_b_k\.weight",                           0),
                    (r"self_attn\.kv_b_v\.weight",                           0),
                    # BMM repack of kv_b_k: per-head (H, 512, 128); shard
                    # head dim (dim=0). The packed UE8M0 scale (H, 512, 1)
                    # shards the same way.
                    (r"self_attn\.kv_b_k_bmm\.weight",                       0),
                    (r"self_attn\.kv_b_k_bmm\.weight_scale_ue8m0",           0),
                    (r"self_attn\.kv_b_k_bf16\.weight",                      0),
                    (r"self_attn\.kv_b_v_bf16\.weight",                      0),
                    (r"self_attn\.kv_a_proj_with_mqa\.weight",               None),
                    (r"self_attn\.kv_a_layernorm\.weight",                   None),
                    # FuseTensor part-a (user #2 part-a, landed 2026-05-13):
                    # fused q_a_proj + kv_a_proj_with_mqa weight, replicated
                    # like its constituents.
                    (r"self_attn\.qkv_a_proj\.weight",                       None),
                    (r"self_attn\.o_proj_original\.weight",                  1),
                    (r"self_attn\.o_proj\.weight",                           1),
                    (r"input_layernorm\.weight$",                            None),
                    (r"post_attention_layernorm\.weight$",                   None),
                    (r"mlp\.gate_up_proj\.weight",                           0),
                    (r"mlp\.down_proj\.weight",                              1),
                    (r"mlp\.gate\.weight$",                                  None),
                    (r"mlp\.gate\.e_score_correction_bias$",                 None),
                    # MoE experts w13/w2: routed experts can be EP-sharded.
                    # Each EP group keeps a disjoint expert range and shards the
                    # intermediate dimension across routed_tp_size=world_size/EP.
                    # w13 [E, 2*inter, hidden]: column-parallel on intermediate (dim=1)
                    # w2  [E, hidden, inter]:   row-parallel on intermediate    (dim=2)
                    # AllReduce after moe_mul_sum_add sums partial contributions.
                    (r"mlp\.experts\.w13\.weight",                           1),
                    (r"mlp\.experts\.w2\.weight",                            2),
                    # Shared expert: gate_proj/up_proj are separate keys (not fused
                    # as gate_up_proj). Builder does its own interleave at build time.
                    (r"mlp\.shared_experts\.gate_proj\.weight",              0),
                    (r"mlp\.shared_experts\.up_proj\.weight",                0),
                    (r"mlp\.shared_experts\.down_proj\.weight",              1),
                    # MTP layers
                    (r"enorm\.weight$",                                      None),
                    (r"hnorm\.weight$",                                      None),
                    (r"eh_proj\.",                                            None),
                    (r"shared_head\.norm\.",                                   None),
                    (r"shared_head\.head\.",                                   None),  # replicate: MTP needs full vocab for draft token selection
                ]
                _compiled = [(re.compile(p), d) for p, d in _TP_SHARD_RULES]

                def _get_tp_shard_dim(name):
                    for regex, dim in _compiled:
                        if regex.search(name):
                            return dim
                    return None  # default: replicate

                print(
                    f"\n  TP/EP sharding (world_size={world_size}, rank={rank}, "
                    f"ep_size={args.ep_size}, routed_tp_size={routed_tp_size}, "
                    f"local_experts=[{local_expert_start},{local_expert_end}))..."
                )
                for k in list(state_dict.keys()):
                    base_key = k.replace("_scale_inv", "")
                    shard_dim = _get_tp_shard_dim(base_key)
                    if base_key == "lm_head.weight" and vocab_parallel_lm_head:
                        old_shape = tuple(state_dict[k].shape)
                        if state_dict[k].shape[0] == lm_head_local_vocab_padded:
                            shard = state_dict[k].contiguous()
                        else:
                            start = lm_head_vocab_offset
                            valid = lm_head_valid_vocab_size
                            if valid > 0:
                                shard = state_dict[k].narrow(0, start, valid).contiguous()
                            else:
                                shard = state_dict[k].new_empty(
                                    (0, state_dict[k].shape[1])
                                )
                            if valid < lm_head_local_vocab_padded:
                                pad = torch.zeros(
                                    lm_head_local_vocab_padded - valid,
                                    state_dict[k].shape[1],
                                    dtype=state_dict[k].dtype,
                                    device=state_dict[k].device,
                                )
                                shard = torch.cat([shard, pad], dim=0).contiguous()
                        state_dict[k] = shard
                        if rank == 0:
                            print(
                                f"    {k}: {old_shape} → {tuple(state_dict[k].shape)} "
                                f"(vocab-parallel lm_head, local padded rows="
                                f"{lm_head_local_vocab_padded})"
                            )
                        continue
                    is_routed_expert = (
                        "mlp.experts.w13.weight" in base_key
                        or "mlp.experts.w2.weight" in base_key
                    )
                    tp_rank_for_key = routed_tp_rank if is_routed_expert else rank
                    tp_size_for_key = routed_tp_size if is_routed_expert else world_size
                    if is_routed_expert and args.ep_size > 1:
                        old_shape = tuple(state_dict[k].shape)
                        if state_dict[k].shape[0] == num_routed_experts:
                            state_dict[k] = state_dict[k].narrow(
                                0, local_expert_start, local_num_experts
                            ).contiguous()
                            if rank == 0:
                                print(
                                    f"    {k}: {old_shape} → {tuple(state_dict[k].shape)} "
                                    f"(EP experts {local_expert_start}:{local_expert_end})"
                                )
                        elif state_dict[k].shape[0] == local_num_experts:
                            if rank == 0:
                                print(
                                    f"    {k}: {old_shape} "
                                    f"(already EP-filtered experts "
                                    f"{local_expert_start}:{local_expert_end})"
                                )
                        else:
                            raise ValueError(
                                f"{k}: unexpected expert dimension "
                                f"{state_dict[k].shape[0]}, expected "
                                f"{num_routed_experts} or {local_num_experts}"
                            )
                    if shard_dim is not None and state_dict[k].dim() >= 2:
                        # Special handling for experts.w13: it's cat([gate, up], dim=1)
                        # so naive dim=1 shard takes all-gate or all-up per rank.
                        # Fix: split into gate/up halves, shard each, then re-cat.
                        if "experts.w13.weight" in base_key and shard_dim == 1:
                            w = state_dict[k]
                            half = w.shape[shard_dim] // 2
                            gate_half = w.narrow(shard_dim, 0, half)
                            up_half = w.narrow(shard_dim, half, half)
                            gate_shard = shard_tensor(
                                gate_half, shard_dim, tp_rank_for_key, tp_size_for_key)
                            up_shard = shard_tensor(
                                up_half, shard_dim, tp_rank_for_key, tp_size_for_key)
                            old_shape = tuple(w.shape)
                            state_dict[k] = torch.cat([gate_shard, up_shard], dim=shard_dim).contiguous()
                            if rank == 0:
                                print(
                                    f"    {k}: {old_shape} → {tuple(state_dict[k].shape)} "
                                    f"(w13 split-shard dim={shard_dim}, tp={tp_size_for_key})"
                                )
                        else:
                            old_shape = tuple(state_dict[k].shape)
                            state_dict[k] = shard_tensor(
                                state_dict[k], shard_dim, tp_rank_for_key, tp_size_for_key)
                            if rank == 0 and old_shape != tuple(state_dict[k].shape):
                                print(
                                    f"    {k}: {old_shape} → {tuple(state_dict[k].shape)} "
                                    f"(dim={shard_dim}, tp={tp_size_for_key})"
                                )
                    elif shard_dim is not None and state_dict[k].dim() < 2:
                        if rank == 0 and "shared_experts" in k:
                            print(f"    [WARN] {k}: dim={state_dict[k].dim()} < 2, shard_dim={shard_dim} → SKIPPED (1D tensor)")
                    else:
                        if rank == 0 and "shared_experts" in k:
                            print(f"    [INFO] {k}: shard_dim={shard_dim}, shape={tuple(state_dict[k].shape)} → {'REPLICATED' if shard_dim is None else 'BUG'}")

            if cache_file is not None:
                os.makedirs(os.path.dirname(cache_file), exist_ok=True)
                cache_save_start = time.perf_counter()
                cache_state_dict = {}
                for k, t in state_dict.items():
                    cache_state_dict[k] = t.detach().cpu().contiguous()
                tmp_cache_file = f"{cache_file}.tmp.{os.getpid()}"
                save_file(
                    cache_state_dict,
                    tmp_cache_file,
                    metadata={"cache_payload": json.dumps(cache_payload, sort_keys=True)},
                )
                os.replace(tmp_cache_file, cache_file)
                print(
                    f"  Saved MPK weight cache: {cache_file} "
                    f"({time.perf_counter() - cache_save_start:.3f}s)"
                )

        # Move state_dict to GPU after conversion + TP sharding.
        # Loading to CPU first (when TP>1) avoids single-GPU OOM from
        # holding full un-sharded weights before the sharding step above.
        if world_size > 1 and not state_dict_from_cache:
            print(f"  Moving {len(state_dict)} tensors to cuda:{rank}...")
            for k in state_dict:
                if state_dict[k].device.type == "cpu":
                    state_dict[k] = state_dict[k].to(f"cuda:{rank}")

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
            # DeepSeek builder precomputes RoPE cos/sin internally because MLA
            # uses the compressed KV cache and separate c_latent/k_pe paths.
            position_embeddings=None,
            rope_theta=getattr(config, "rope_theta", 10000.0),
            rope_parameters=getattr(config, "rope_parameters", None)
            or getattr(config, "rope_scaling", None),
            max_position_embeddings=getattr(config, "max_position_embeddings", None),
            state_dict=state_dict,
            with_lm_head=True,
        )

        # Optional per-layer residual dump (diagnostic). Allocate one
        # PyTorch tensor per built layer so the builder can wire a copy
        # task at the end of each layer iteration. Saved on rank 0
        # after mpk() returns.
        if args.dump_hidden_dir is not None:
            num_dump_layers = (
                len(layer_indices_arg) if layer_indices_arg
                else model_config.num_layers
            )
            mpk.dump_hidden_tensors = [
                torch.zeros(
                    (args.max_num_batched_tokens, model_config.hidden_size),
                    dtype=torch.bfloat16, device="cuda",
                )
                for _ in range(num_dump_layers)
            ]
            # Also dump self.y (embed_layer output) so we can isolate
            # whether MPK's hidden-state divergence from reference starts
            # at the embedding step or later in attn/MLP.
            mpk.dump_embed_tensor = torch.zeros(
                (args.max_num_batched_tokens, model_config.hidden_size),
                dtype=torch.bfloat16, device="cuda",
            )
            # Layer-0 intra-layer dumps: 4 slots, all bf16
            # 0=input_norm    (mbt, hidden)
            # 1=qkv_a_out     (mbt, 2176) — full fused QKV-a GEMM output;
            #                  Python comparator slices to q_a/c_latent/k_pe.
            # 2=attn_out      (mbt, hidden) — o_proj+residual
            # 3=dense_mlp_out (mbt, hidden) — full layer 0 residual
            slot1_cols = 2176  # = QKV_A_FUSED_N in builder.py
            mpk.dump_layer0_intra_tensors = [
                torch.zeros(
                    (args.max_num_batched_tokens, model_config.hidden_size),
                    dtype=torch.bfloat16, device="cuda",
                ),
                torch.zeros(
                    (args.max_num_batched_tokens, slot1_cols),
                    dtype=torch.bfloat16, device="cuda",
                ),
                torch.zeros(
                    (args.max_num_batched_tokens, model_config.hidden_size),
                    dtype=torch.bfloat16, device="cuda",
                ),
                torch.zeros(
                    (args.max_num_batched_tokens, model_config.hidden_size),
                    dtype=torch.bfloat16, device="cuda",
                ),
            ]
        else:
            mpk.dump_hidden_tensors = None
            mpk.dump_embed_tensor = None
            mpk.dump_layer0_intra_tensors = None


        # Build the computation graph using the DeepSeek V3 builder
        builder = DeepSeekV3Builder(mpk)
        builder.build_from_config(model_config, layer_indices=layer_indices_arg)

        if args.dump_task_graph:
            results = mpk.kn_graph.generate_task_graph(
                num_gpus=world_size, my_gpu_id=rank
            )
            with open(f"task_graph_{rank}.json", "w") as f:
                f.write(results["json_file"])
            with open(f"kernel_{rank}.cu", "w") as f:
                f.write(results["cuda_code"])

        mpk.compile(output_dir=args.output_dir)
        if args.profile_start_step is not None:
            if not args.profiling:
                raise ValueError("--profile-start-step requires --profiling")
            if args.profile_start_step < 0 or args.profile_start_step >= args.max_seq_length:
                raise ValueError(
                    f"--profile-start-step must be in [0, {args.max_seq_length}); "
                    f"got {args.profile_start_step}"
                )
            step.fill_(args.profile_start_step)
            torch.cuda.synchronize()
            if rank == 0:
                print(
                    f"[profiling] Starting first task graph at "
                    f"step={args.profile_start_step}; prompt_lengths="
                    f"{prompt_lengths.detach().cpu().tolist()}"
                )

        # Run inference
        print("Starting inference with Mirage megakernel...")
        starter.record()
        mpk()
        ender.record()
        torch.cuda.synchronize()
        run_time = starter.elapsed_time(ender)

        # Optional debug dump: per-row argmax outputs from the lm_head.
        # During prefill, output_tokens[i] is the argmax of position i's logits.
        # Comparing across rows tells us whether degenerate logits are
        # specific to the last prefill position (off-by-one on hidden state)
        # or affect every prefill row (lm_head/final-norm bug).
        if os.environ.get("MPK_DEBUG_OUTPUT_TOKENS") == "1" and rank == 0:
            ot = output_tokens.flatten().cpu().tolist()
            ot_str = ", ".join(str(v) for v in ot[:min(110, len(ot))])
            print(f"[debug] output_tokens[0:{min(110, len(ot))}] = [{ot_str}]")

        if args.dump_hidden_dir is not None and rank == 0 and getattr(mpk, "dump_hidden_tensors", None) is not None:
            os.makedirs(args.dump_hidden_dir, exist_ok=True)
            ordered_layer_idx = layer_indices_arg or list(range(model_config.num_layers))
            for slot, t in enumerate(mpk.dump_hidden_tensors):
                idx = ordered_layer_idx[slot] if slot < len(ordered_layer_idx) else slot
                path = os.path.join(args.dump_hidden_dir, f"layer_{idx:02d}_residual.pt")
                torch.save(t.detach().cpu(), path)
            if getattr(mpk, "dump_embed_tensor", None) is not None:
                torch.save(mpk.dump_embed_tensor.detach().cpu(),
                           os.path.join(args.dump_hidden_dir, "embed.pt"))
                print(f"Saved embed.pt to {args.dump_hidden_dir}")
            if getattr(mpk, "dump_layer0_intra_tensors", None) is not None:
                names = ["layer0_input_norm.pt",
                         "layer0_q_a_out.pt",
                         "layer0_attn_out.pt",
                         "layer0_dense_mlp_out.pt"]
                for k, t in enumerate(mpk.dump_layer0_intra_tensors):
                    torch.save(t.detach().cpu(),
                               os.path.join(args.dump_hidden_dir, names[k]))
                print(f"Saved layer-0 intra-layer dumps to {args.dump_hidden_dir}")
            if getattr(mpk, "dump_attn_unabsorbed_tensor", None) is not None:
                torch.save(
                    mpk.dump_attn_unabsorbed_tensor.detach().cpu(),
                    os.path.join(args.dump_hidden_dir, "attn_unabsorbed.pt"),
                )
                print(f"Saved attn_unabsorbed.pt to {args.dump_hidden_dir}")
            # Debug-only: when MPK_DSV3_FP8_BUF_ATTACH=1, builder attaches the
            # shared FP8 input/scale buffers as torch tensors so we can inspect
            # their post-megakernel state here.
            if getattr(mpk, "_fp8_input_torch", None) is not None:
                for rsize, tensor in mpk._fp8_input_torch.items():
                    name = f"fp8_input_v2_{rsize}.pt"
                    torch.save(tensor.detach().view(torch.uint8).cpu(),
                               os.path.join(args.dump_hidden_dir, name))
                    print(f"Saved {name} to {args.dump_hidden_dir}")
                for rsize, tensor in mpk._fp8_scale_torch.items():
                    name = f"fp8_scale_v2_{rsize}.pt"
                    torch.save(tensor.detach().cpu(),
                               os.path.join(args.dump_hidden_dir, name))
                    print(f"Saved {name} to {args.dump_hidden_dir}")
            if getattr(mpk, "_rmsnorm_out_torch", None) is not None:
                torch.save(mpk._rmsnorm_out_torch.detach().cpu(),
                           os.path.join(args.dump_hidden_dir, "rmsnorm_out_attached.pt"))
                print(f"Saved rmsnorm_out_attached.pt to {args.dump_hidden_dir}")

            for _attr, _fname in [
                ("dump_ckv_sep_tensor", "ckv_sep.pt"),
                ("dump_kpe_sep_tensor", "kpe_sep.pt"),
                ("dump_prefill_k_nope_tensor", "prefill_k_nope.pt"),
                ("dump_prefill_v_tensor", "prefill_v.pt"),
            ]:
                _t = getattr(mpk, _attr, None)
                if _t is not None:
                    torch.save(_t.detach().cpu(),
                               os.path.join(args.dump_hidden_dir, _fname))
                    print(f"Saved {_fname} to {args.dump_hidden_dir}")
            print(f"Saved {len(mpk.dump_hidden_tensors)} per-layer residual dumps to {args.dump_hidden_dir}")

        print("tokens.shape = ", tokens.shape)
        for r in range(total_num_requests):
            generated_ids = tokens[r, : step[r] + 1]
            response = safe_tokenizer_decode(
                tokenizer, generated_ids, context=f"request {r}"
            )
            if total_num_requests > 1:
                print(f"[request {r}]")
            print(response)

        if total_num_requests > 1:
            print(f"Output length of each batch is same: {(step.max() == step.min()).item()}")

        print("Prompt length {}, generate length {}, per-token latency (both prefill and decode): {:.3f} ms".format(
            prompt_lengths[0], step.max().item() + 1 - prompt_lengths[0],
            run_time / (step.max().item() + 1)
        ))

        # Dump outputs to json
        if save_path and rank == 0:
            out = []
            for r in range(total_num_requests):
                end_idx = step[r].item() + 1
                prompt_len = prompt_lengths[r].item()
                tokens_generated = max(0, end_idx - prompt_len)
                per_tok_ms = run_time / max(tokens_generated, 1)
                slice_end = min(end_idx, prompt_len + MAX_SAVE_TOKENS)
                token_ids = tokens[r, prompt_len:slice_end].tolist()
                response_text = safe_tokenizer_decode(
                    tokenizer, tokens[r, :end_idx], context=f"request {r}"
                )
                out.append({
                    "request_id": r,
                    "token_ids": token_ids,
                    "text": response_text,
                    "latency_ms_per_token": per_tok_ms,
                    "prompt_length": prompt_len,
                    "generate_length": tokens_generated,
                    "mode": "mpk",
                })
            if total_num_requests == 1:
                out = out[0]
            with open(save_path, "w") as f:
                json.dump(out, f, indent=2)
            print(f"Saved tokens to {save_path}")

    else:
        raise RuntimeError("Pytorch ref is not allowed for now, which may cause OOM.")
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
        response = safe_tokenizer_decode(
            tokenizer, generated_ids, context="torch request 0"
        )
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
                "text": safe_tokenizer_decode(
                    tokenizer, tokens[0, :end_idx], context="torch request 0"
                ),
                "latency_ms_per_token": per_tok_ms,
                "prompt_length": prompt_len,
                "generate_length": tokens_generated,
                "mode": "torch",
            }
            with open(save_path, "w") as f:
                json.dump(out, f, indent=2)
            print(f"Saved tokens to {save_path}")

    if "mpk" in locals() and hasattr(mpk, "finalize") and not getattr(
        mpk, "__finalized__", True
    ):
        mpk.finalize()
    if world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()
