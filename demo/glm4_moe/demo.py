"""GLM-4.6 (zai-org/GLM-4.6, Glm4MoeForCausalLM) demo with the Mirage
persistent megakernel.

Graph construction is fully delegated to
mirage.mpk.models.glm4_moe.Glm4MoeBuilder; this script only handles the
tokenizer, meta-tensor setup, the decode loop, and (optionally) a
HuggingFace reference run for correctness comparison.

Examples
--------
# Mirage megakernel, full checkpoint (needs a B200/H100-class GPU and enough
# HBM for the layers you load):
python demo/glm4_moe/demo.py --model-path /path/to/GLM-4.6 --use-mirage \
    --prompt "Give me a short introduction to large language models."

# Smoke test on a subset of layers (the builder honors whatever layer
# indices are present in the loaded state dict):
python demo/glm4_moe/demo.py --model-path /path/to/GLM-4.6 --use-mirage \
    --layers 0-4 --max-new-tokens 8

# HuggingFace reference (no Mirage), for parity checks:
python demo/glm4_moe/demo.py --model-path /path/to/GLM-4.6 \
    --max-new-tokens 16

GLM-4.6 is a 355B-parameter MoE model; loading the whole thing takes many
GPUs / a lot of HBM. Use --layers to load a slice for functional smoke
tests. v1 of the Mirage builder is decode-only, world_size 1, BF16, and
skips the MTP nextn layer.
"""

import argparse
import glob
import json
import os

import torch
from transformers import AutoConfig, AutoTokenizer

DEFAULT_SAVE_DIR = os.path.join("outputs", "glm4_moe")
MAX_SAVE_TOKENS = 100


def parse_layers(spec: str):
    """'0,3,5' or '0-4' (inclusive) -> sorted list of layer indices."""
    result = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            result.update(range(int(lo), int(hi) + 1))
        else:
            result.add(int(part))
    return sorted(result)


def load_state_dict(model_path: str, layers, num_layers: int):
    """Load a GLM-4.6 checkpoint. When `layers` is given, only the shards
    holding those layers (plus the global embed/norm/lm_head tensors) are
    read, so you can smoke-test a subset without materializing 355B params.
    """
    from safetensors import safe_open
    from safetensors.torch import load_file

    index_file = os.path.join(model_path, "model.safetensors.index.json")
    if layers is not None and os.path.exists(index_file):
        needed_prefixes = ["model.embed_tokens.", "model.norm.", "lm_head."]
        for li in layers:
            needed_prefixes.append(f"model.layers.{li}.")
        with open(index_file) as f:
            weight_map = json.load(f)["weight_map"]
        shard_to_keys = {}
        for key, shard in weight_map.items():
            if any(key.startswith(p) for p in needed_prefixes):
                shard_to_keys.setdefault(shard, []).append(key)
        state_dict = {}
        for shard, keys in sorted(shard_to_keys.items()):
            path = os.path.join(model_path, shard)
            print(f"  loading {len(keys)} keys from {shard}")
            with safe_open(path, framework="pt", device="cuda") as f:
                for key in keys:
                    state_dict[key] = f.get_tensor(key)
        print(f"  loaded {len(state_dict)} keys for layers {layers}")
        return state_dict

    # full load
    shard_files = sorted(glob.glob(os.path.join(model_path, "model-*.safetensors")))
    if not shard_files:
        single = os.path.join(model_path, "model.safetensors")
        if not os.path.exists(single):
            raise FileNotFoundError(
                f"No safetensors found under {model_path} "
                "(expected model-*.safetensors or model.safetensors)")
        return load_file(single, device="cuda")
    state_dict = {}
    for shard_file in shard_files:
        print(f"  loading {os.path.basename(shard_file)}")
        state_dict.update(load_file(shard_file, device="cuda"))
    return state_dict


def main():
    parser = argparse.ArgumentParser(
        description="GLM-4.6 demo with the Mirage megakernel")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to a GLM-4.6 HuggingFace checkpoint dir")
    parser.add_argument("--use-mirage", action="store_true",
                        help="Run the Mirage megakernel (else HF reference)")
    parser.add_argument("--prompt", type=str,
                        default="Give me a short introduction to large language models.")
    parser.add_argument("--layers", type=str, default=None,
                        help="Subset of layer indices to load, e.g. '0-4' or "
                             "'0,3,5'. Omit to load the full model.")
    parser.add_argument("--max-num-batched-tokens", type=int, default=1)
    parser.add_argument("--max-num-batched-requests", type=int, default=1)
    parser.add_argument("--page-size", type=int, default=64,
                        help="KV-cache page size (must be a multiple of 64)")
    parser.add_argument("--max-num-pages", type=int, default=64)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--profiling", action="store_true")
    parser.add_argument("--trace-name", type=str, default="")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--save-tokens", nargs="?", const="auto", default=None,
                        help="Dump first N generated tokens + text + latency "
                             "to JSON (auto path under outputs/glm4_moe/).")
    args = parser.parse_args()
    print("Input arguments:", args)

    assert args.page_size % 64 == 0, "GLM paged attention needs page_size % 64 == 0"

    torch.set_default_dtype(torch.bfloat16)
    torch.cuda.set_device(0)

    config = AutoConfig.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    layers = parse_layers(args.layers) if args.layers else None
    num_layers = config.num_hidden_layers

    eos_token_id = config.eos_token_id
    if isinstance(eos_token_id, list):
        eos_token_id = eos_token_id[0]

    # tokenize prompt
    messages = [{"role": "user", "content": args.prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer([text], return_tensors="pt").input_ids[0].to("cuda")
    prompt_len = int(input_ids.shape[0])
    assert prompt_len < args.max_seq_length, "prompt longer than max_seq_length"

    save_path = None
    if args.save_tokens:
        save_path = (os.path.join(DEFAULT_SAVE_DIR, "mpk_output.json"
                     if args.use_mirage else "torch_output.json")
                     if args.save_tokens == "auto" else args.save_tokens)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)

    if not args.use_mirage:
        # ---- HuggingFace reference path ----
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16).to("cuda").eval()
        gen = model.generate(
            input_ids.unsqueeze(0),
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            eos_token_id=None if args.ignore_eos else eos_token_id,
        )
        out_ids = gen[0]
        text_out = tokenizer.decode(out_ids[prompt_len:], skip_special_tokens=True)
        print("\n=== HF reference output ===\n" + text_out)
        if save_path:
            json.dump({
                "token_ids": out_ids[prompt_len:prompt_len + MAX_SAVE_TOKENS].tolist(),
                "text": text_out, "prompt_length": prompt_len, "mode": "torch",
            }, open(save_path, "w"), indent=2)
            print(f"Saved tokens to {save_path}")
        return

    # ---- Mirage megakernel path ----
    import mirage as mi
    from mirage.mpk.models.glm4_moe import Glm4MoeBuilder

    total_num_requests = args.max_num_batched_requests
    tokens = torch.zeros((total_num_requests, args.max_seq_length),
                         dtype=torch.long, device="cuda")
    for r in range(total_num_requests):
        tokens[r, :prompt_len] = input_ids
    input_tokens = torch.zeros((args.max_num_batched_tokens, 1),
                               dtype=torch.long, device="cuda")
    output_tokens = torch.zeros((args.max_num_batched_tokens, 1),
                                dtype=torch.long, device="cuda")
    step = torch.zeros((total_num_requests,), dtype=torch.int32, device="cuda")
    num_new_tokens = torch.ones((total_num_requests,), dtype=torch.int32, device="cuda")
    prompt_lengths = torch.full((total_num_requests,), prompt_len,
                                dtype=torch.int32, device="cuda")
    qo_indptr_buffer = torch.empty(args.max_num_batched_requests + 1,
                                   dtype=torch.int32, device="cuda")
    paged_kv_indptr_buffer = torch.empty(args.max_num_batched_requests + 1,
                                         dtype=torch.int32, device="cuda")
    paged_kv_indices_buffer = torch.empty(args.max_num_pages,
                                          dtype=torch.int32, device="cuda")
    paged_kv_last_page_len_buffer = torch.empty(args.max_num_batched_requests,
                                                dtype=torch.int32, device="cuda")

    profiler_tensor = (torch.zeros(3000 * 128, dtype=torch.uint64, device="cuda")
                       if args.profiling else None)
    num_workers, num_schedulers = mi.get_configurations_from_gpu(0)

    mpk = mi.PersistentKernel(
        mode="offline",
        world_size=1,
        mpi_rank=0,
        num_workers=num_workers,
        num_local_schedulers=num_schedulers,
        num_remote_schedulers=0,
        max_seq_length=args.max_seq_length,
        max_num_batched_requests=args.max_num_batched_requests,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_pages=args.max_num_pages,
        page_size=args.page_size,
        eos_token_id=-1 if args.ignore_eos else eos_token_id,
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
        trace_name=(f"{args.trace_name}_rank0" if args.trace_name else ""),
    )

    print(f"Loading GLM-4.6 weights from: {args.model_path}")
    state_dict = load_state_dict(args.model_path, layers, num_layers)

    builder = Glm4MoeBuilder(mpk)
    builder.build_from_dict(state_dict, with_lm_head=True)

    print("Compiling megakernel...")
    mpk.compile(output_dir=args.output_dir)

    print("Running megakernel...")
    starter.record()
    mpk()
    ender.record()
    torch.cuda.synchronize()
    run_time = starter.elapsed_time(ender)

    for r in range(total_num_requests):
        end_idx = int(step[r].item()) + 1
        response = tokenizer.decode(tokens[r, :end_idx], skip_special_tokens=True)
        print(f"\n=== request {r} output ===\n{response}")

    tokens_generated = int(step.max().item()) + 1 - prompt_len
    per_tok_ms = run_time / max(prompt_len + tokens_generated, 1)
    print(f"\nPrompt length {prompt_len}, generated {tokens_generated}, "
          f"per-token latency {per_tok_ms:.3f} ms")

    if save_path:
        end_idx = int(step[0].item()) + 1
        slice_end = min(end_idx, prompt_len + MAX_SAVE_TOKENS)
        json.dump({
            "token_ids": tokens[0, prompt_len:slice_end].tolist(),
            "text": tokenizer.decode(tokens[0, :end_idx], skip_special_tokens=True),
            "latency_ms_per_token": per_tok_ms,
            "prompt_length": prompt_len,
            "generate_length": tokens_generated,
            "mode": "mpk",
        }, open(save_path, "w"), indent=2)
        print(f"Saved tokens to {save_path}")


if __name__ == "__main__":
    main()
