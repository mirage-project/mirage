"""DeepSeek V3 driver using the new mirage.mpk.layers catalog.

Companion to ``demo/deepseek_v3/demo.py``: same end-to-end behavior on
``--max-num-batched-requests 1`` but built against the PyTorch-module
catalog (Phase-3 refactor). The two demos are kept separate so the
existing one can serve as a correctness oracle until this path is fully
proven.

Usage::

    python demo/deepseek_v3/demo_new.py \\
        --model-path /raid/catalyst/models/DeepSeek-V3 \\
        --max-num-batched-requests 1 \\
        --output-dir ./output/output_dsv3_new \\
        --layers 0-3

Scope (v1):
  * Single-GPU only (world_size=1; TP deferred).
  * BF16 weights only — HF FP8 weights are dequantized to BF16 at load.
  * Decode-only (max_num_batched_tokens <= 8). No prefill chunking.
  * Greedy decode (no spec-decode, no sampling, no MTP).
  * KV absorption + W_UV→o_proj fusion done inline during streaming load.

Weight loading: uses ``DeepseekV3ForCausalLM.load_weights`` which streams
from the sharded HF safetensors, dequantizes FP8 → BF16, absorbs
``kv_b_proj`` into ``q_b_proj``, fuses ``W_UV`` into ``o_proj``, stacks
per-expert weights, and calls ``process_weights()`` at the end.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import List

import torch
from transformers import AutoConfig, AutoTokenizer

import mirage as mi
from mirage.mpk.models.deepseek_v3.modeling import DeepseekV3ForCausalLM
from mirage.mpk.weight_loader import find_safetensors_files, safetensors_weights_iterator


DEFAULT_SAVE_DIR = os.path.join("outputs", "deepseek_v3")
MAX_SAVE_TOKENS = 100


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DeepSeek V3 demo (new catalog-based path)"
    )
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to a HuggingFace DeepSeek V3 checkpoint")
    parser.add_argument("--max-num-batched-tokens", default=8, type=int)
    parser.add_argument("--max-num-batched-requests", default=1, type=int)
    parser.add_argument("--page-size", default=128, type=int)
    parser.add_argument("--max-num-pages", default=16, type=int)
    parser.add_argument("--max-seq-length", default=128, type=int,
                        help="Keep small in v1 (decode-only, single split)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Where to write the compiled .cu/.so")
    parser.add_argument("--trace-name", default="", help="Perfetto trace name")
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--prompt", type=str,
                        default="Give me a short introduction to large language model.")
    parser.add_argument("--layers", type=str, default="0-3",
                        help="Comma-separated layer indices, or a range "
                             "'lo-hi'. e.g. '0,1,2,3' or '0-3'.")
    parser.add_argument("--skip-weight-load", action="store_true",
                        help="Skip HF weight loading (use random-init). "
                             "Use for smoke-testing the compile() path.")
    parser.add_argument("--save-tokens", nargs="?", const="auto", default=None,
                        help="Dump first N generated token IDs + decoded text "
                             "to JSON.")
    return parser.parse_args()


def _parse_layers(spec: str) -> List[int]:
    out: List[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = _parse_args()
    torch.set_default_dtype(torch.bfloat16)
    torch.cuda.set_device(0)

    # ---- 1. Config + tokenizer + layer subset --------------------------
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path,
                                              trust_remote_code=True)
    layer_indices = _parse_layers(args.layers)
    # Override num_hidden_layers so the modeling sees the reduced layer
    # count when building the layer stack (only the selected layers
    # actually have weights loaded; the rest get random init).
    if layer_indices:
        max_layer = max(layer_indices)
        if max_layer + 1 < config.num_hidden_layers:
            print(
                f"[v1] Reducing config.num_hidden_layers from "
                f"{config.num_hidden_layers} to {max_layer + 1} to match "
                f"--layers={args.layers}"
            )
            config.num_hidden_layers = max_layer + 1

    print(
        f"Model config: hidden={config.hidden_size}, "
        f"num_heads={config.num_attention_heads}, "
        f"kv_lora_rank={config.kv_lora_rank}, "
        f"qk_rope_head_dim={config.qk_rope_head_dim}, "
        f"qk_nope_head_dim={config.qk_nope_head_dim}, "
        f"q_lora_rank={config.q_lora_rank}, "
        f"v_head_dim={config.v_head_dim}, "
        f"num_layers={config.num_hidden_layers}, "
        f"first_k_dense_replace={getattr(config, 'first_k_dense_replace', 3)}, "
        f"n_routed_experts={getattr(config, 'n_routed_experts', 256)}"
    )

    # ---- 2. Tokenize prompt -------------------------------------------
    messages = [{"role": "user", "content": args.prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to("cuda")
    prompt_len = model_inputs.input_ids.shape[-1]

    # ---- 3. Meta tensors ----------------------------------------------
    total_num_requests = args.max_num_batched_requests
    tokens = torch.zeros(
        (total_num_requests, args.max_seq_length),
        dtype=torch.long, device="cuda",
    )
    prompt_len_clipped = min(prompt_len, args.max_seq_length - 1)
    tokens[0, :prompt_len_clipped] = model_inputs.input_ids[0, :prompt_len_clipped]
    prompt_lengths = torch.full(
        (total_num_requests,), prompt_len_clipped,
        dtype=torch.int32, device="cuda",
    )
    step = torch.zeros((total_num_requests,), dtype=torch.int32, device="cuda")
    num_new_tokens = torch.full(
        (total_num_requests,), 1, dtype=torch.int32, device="cuda"
    )
    input_tokens = torch.zeros(
        (args.max_num_batched_tokens, 1), dtype=torch.long, device="cuda"
    )
    output_tokens = torch.zeros(
        (args.max_num_batched_tokens, 1), dtype=torch.long, device="cuda"
    )
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

    # ---- 4. KV cache pool (MLA: single combined ckv_kpe per layer) ----
    # Shape: (num_layers, max_num_pages, page_size, kv_lora_rank +
    # qk_rope_head_dim). For v1 BF16, this is the same layout the existing
    # demo uses on line 426.
    ckv_kpe_dim = config.kv_lora_rank + config.qk_rope_head_dim
    ckv_kpe_cache = torch.zeros(
        (
            config.num_hidden_layers,
            args.max_num_pages,
            args.page_size,
            ckv_kpe_dim,
        ),
        dtype=torch.bfloat16, device="cuda",
    )

    # ---- 5. PersistentKernel ------------------------------------------
    num_workers, num_schedulers = mi.get_configurations_from_gpu(0)
    spec_decode_config = mi.mpk.spec_decode_class(None, 3, 5)
    eos = config.eos_token_id if not args.ignore_eos else -1
    if isinstance(eos, list):
        eos = eos[0]

    # The KV-cache hook expects a dict; MLA uses a SINGLE combined pool, so
    # we register the same tensor under both "k_cache" and "v_cache" — the
    # MLA gather path reads ``pk.get_kv_cache(layer_idx)[0]`` (the k slice)
    # and ignores the second element. This matches the modeling's
    # MLAKVGather.compile() call site.
    pk = mi.PersistentKernel(
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
        eos_token_id=eos,
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
        profiler_tensor=None,
        trace_name=args.trace_name,
        spec_decode_config=spec_decode_config,
        use_cutlass_kernel=True,
        kv_cache={"k_cache": ckv_kpe_cache, "v_cache": ckv_kpe_cache},
    )

    # ---- 6. Instantiate model -----------------------------------------
    with torch.device("cuda"):
        model = DeepseekV3ForCausalLM(config).to("cuda", dtype=torch.bfloat16)

    if not args.skip_weight_load:
        print("Loading HF weights (streaming: FP8 dequant + MLA absorption)...")
        files = find_safetensors_files(args.model_path)
        model.load_weights(safetensors_weights_iterator(files))
    else:
        print("[v1] --skip-weight-load set; using random-initialized weights.")

    # ---- 7. Pre-pad lm_head to a 256-multiple vocab -------------------
    raw_vocab = config.vocab_size
    padded_vocab = ((raw_vocab + 255) // 256) * 256
    hidden = config.hidden_size
    padded_weight = torch.zeros(
        padded_vocab, hidden, dtype=torch.bfloat16, device="cuda"
    )
    padded_weight[:raw_vocab] = model.lm_head.weight.data
    model.lm_head.weight = torch.nn.Parameter(padded_weight)
    model.lm_head.out_features = padded_vocab
    model.argmax_partial.vocab_size = padded_vocab
    model.argmax_reduce.num_partial_tasks = num_workers
    model.argmax_partial.num_partial_tasks = num_workers

    # ---- 8. Compile the graph -----------------------------------------
    input_tokens_dt = pk.attach_input(input_tokens, name="input_token")
    with pk.compile_scope():
        model.compile(
            input_tokens_dt,
            output_tokens=output_tokens,
            lm_head_padded_vocab=padded_vocab,
        )

    # Optional: dump the task graph + cuda code for inspection.
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        results = pk.kn_graph.generate_task_graph(num_gpus=1, my_gpu_id=0)
        with open(os.path.join(args.output_dir, "task_graph_0.json"), "w") as f:
            f.write(results["json_file"])
        with open(os.path.join(args.output_dir, "kernel_0.cu"), "w") as f:
            f.write(results["cuda_code"])

    # nvcc compile.
    pk.compile(output_dir=args.output_dir)
    print(f"Compiled into {args.output_dir or '<scratch>'}")

    # ---- 9. Launch + decode -------------------------------------------
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    starter.record()
    pk()
    ender.record()
    torch.cuda.synchronize()
    run_time_ms = starter.elapsed_time(ender)

    end_idx = step[0].item() + 1
    generated_ids = tokens[0, :end_idx]
    try:
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    except Exception as e:
        response = f"[decode failed: {e}]"
    print(response)
    per_tok = run_time_ms / max(end_idx, 1)
    print(f"Prompt length {prompt_len_clipped}, end_idx {end_idx}, "
          f"per-token latency: {per_tok:.3f} ms")

    # ---- 10. Save tokens for diff -------------------------------------
    if args.save_tokens:
        if args.save_tokens == "auto":
            filename = "mpk_output_new.json"
            save_path = os.path.join(DEFAULT_SAVE_DIR, filename)
        else:
            save_path = args.save_tokens
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        slice_end = min(end_idx, prompt_len_clipped + MAX_SAVE_TOKENS)
        token_ids = tokens[0, prompt_len_clipped:slice_end].tolist()
        with open(save_path, "w") as f:
            json.dump({
                "token_ids": token_ids,
                "text": response,
                "latency_ms_per_token": per_tok,
                "prompt_length": prompt_len_clipped,
                "generate_length": max(0, end_idx - prompt_len_clipped),
                "mode": "mpk_dsv3_new",
            }, f, indent=2)
        print(f"Saved tokens to {save_path}")


if __name__ == "__main__":
    main()
