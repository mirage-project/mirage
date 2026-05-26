"""Qwen3 driver using the new mirage.mpk.layers catalog.

Companion to ``demo/qwen3/demo.py``: same end-to-end behavior on
``--max-num-batched-requests 1`` (the canonical smoke command in
CLAUDE.md), but built against the PyTorch-module catalog from the
Phase 1-3 refactor. The two demos are intentionally kept separate so
the existing one can serve as a correctness oracle (token-for-token
diff) until the new path is fully proven.

Usage::

    python demo/qwen3/demo_new.py --model /raid/catalyst/models/Qwen3-8B/ \\
        --max-num-batched-requests 1 --output-dir ./output/output_new

Scope (Phase 3):
  * Single-GPU only (world_size=1; TP is deferred per plan decision #14).
  * Greedy decode only (no spec-decode, no sampling).
  * Paged-attention path only (no --split-kv-cache).

KV cache allocation: this driver allocates one
(num_layers, max_num_pages, page_size, num_kv_heads, head_dim) tensor
for k and one for v, registers them with the PK via the new
``kv_cache=`` kwarg, and each ``PagedAttention`` leaf fetches its slice
via ``current_pk().get_kv_cache(layer_idx)`` at compile time
(Option-III in the design).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict

import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoTokenizer

import mirage as mi
from mirage.mpk.models.qwen3.modeling import Qwen3ForCausalLM


DEFAULT_SAVE_DIR = os.path.join("outputs", "qwen3")
MAX_SAVE_TOKENS = 100


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B",
                        help="Model path on HuggingFace or local dir")
    parser.add_argument("--max-num-batched-tokens", default=8, type=int)
    parser.add_argument("--max-num-batched-requests", default=1, type=int)
    parser.add_argument("--page-size", default=4096, type=int)
    parser.add_argument("--max-num-pages", default=16, type=int)
    parser.add_argument("--max-seq-length", default=512, type=int)
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Where to write the compiled .cu/.so")
    parser.add_argument("--trace-name", default="", help="Perfetto trace name")
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--prompt", type=str,
                        default="Give me a short introduction to large language model.")
    parser.add_argument("--save-tokens", nargs="?", const="auto", default=None,
                        help="Dump first N generated token IDs + decoded text "
                             "to JSON for diff against demo.py output.")
    return parser.parse_args()


def _load_hf_weights(model_path: str) -> Dict[str, torch.Tensor]:
    # Load all *.safetensors shards from model_path. Returns a single
    # state_dict on CPU; the model's load_state_dict moves things to
    # CUDA via the Parameter device. Keeps memory bounded for the
    # 8B model (~16 GB bf16); fine on the 80 GB H100/B200 boxes.
    files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
    if not files:
        raise FileNotFoundError(
            f"No .safetensors found under {model_path}. The new driver "
            "expects a local path; use --model /local/path."
        )
    state_dict: Dict[str, torch.Tensor] = {}
    for f in files:
        state_dict.update(load_file(f, device="cpu"))
    return state_dict


def main() -> None:
    args = _parse_args()
    torch.set_default_dtype(torch.bfloat16)
    torch.cuda.set_device(0)

    # ---- 1. Config + tokenizer -------------------------------------------
    config = AutoConfig.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    print(f"Model: {config.architectures}, "
          f"layers={config.num_hidden_layers}, hidden={config.hidden_size}, "
          f"heads={config.num_attention_heads}/{config.num_key_value_heads} "
          f"head_dim={config.head_dim}")

    # ---- 2. Tokenize the prompt ------------------------------------------
    messages = [
        {"role": "system",
         "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": args.prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to("cuda")
    prompt_len = model_inputs.input_ids.shape[-1]

    # ---- 3. Runtime state tensors ----------------------------------------
    total_num_requests = args.max_num_batched_requests
    tokens = torch.zeros((total_num_requests, args.max_seq_length),
                         dtype=torch.long, device="cuda")
    tokens[0, :prompt_len] = model_inputs.input_ids[0]
    prompt_lengths = torch.full((total_num_requests,), prompt_len,
                                dtype=torch.int32, device="cuda")
    step = torch.zeros((total_num_requests,), dtype=torch.int32, device="cuda")
    num_new_tokens = torch.full((total_num_requests,), 1,
                                dtype=torch.int32, device="cuda")
    input_tokens = torch.zeros((args.max_num_batched_tokens, 1),
                               dtype=torch.long, device="cuda")
    output_tokens = torch.zeros((args.max_num_batched_tokens, 1),
                                dtype=torch.long, device="cuda")
    qo_indptr_buffer = torch.empty(
        args.max_num_batched_requests + 1, dtype=torch.int32, device="cuda")
    paged_kv_indptr_buffer = torch.empty(
        args.max_num_batched_requests + 1, dtype=torch.int32, device="cuda")
    paged_kv_indices_buffer = torch.empty(
        args.max_num_pages, dtype=torch.int32, device="cuda")
    paged_kv_last_page_len_buffer = torch.empty(
        args.max_num_batched_requests, dtype=torch.int32, device="cuda")

    # ---- 4. KV cache pool ------------------------------------------------
    # Shape matches what PagedAttention asserts: per-layer slice is
    # (max_num_pages, page_size, num_kv_heads, head_dim).
    kv_cache_shape = (
        config.num_hidden_layers,
        args.max_num_pages,
        args.page_size,
        config.num_key_value_heads,
        config.head_dim,
    )
    k_cache_pool = torch.zeros(kv_cache_shape, dtype=torch.bfloat16, device="cuda")
    v_cache_pool = torch.zeros(kv_cache_shape, dtype=torch.bfloat16, device="cuda")

    # ---- 5. PersistentKernel ---------------------------------------------
    num_workers, num_schedulers = mi.get_configurations_from_gpu(0)
    # Spec-decode is out of scope for the new driver; pass None + dummy
    # ngram/spec lengths (the helper requires all three positional args).
    spec_decode_config = mi.mpk.spec_decode_class(None, 3, 5)
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
        eos_token_id=config.eos_token_id if not args.ignore_eos else -1,
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
        kv_cache={"k_cache": k_cache_pool, "v_cache": v_cache_pool},
    )

    # ---- 6. Instantiate model + load weights -----------------------------
    with torch.device("cuda"):
        model = Qwen3ForCausalLM(config).to("cuda", dtype=torch.bfloat16)
    state_dict = _load_hf_weights(args.model)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[warn] {len(missing)} missing keys (first 5): {missing[:5]}")
    if unexpected:
        print(f"[warn] {len(unexpected)} unexpected keys (first 5): {unexpected[:5]}")

    # ---- 7. Pre-pad lm_head to multi-of-grid vocab -----------------------
    # The argmax-partial grid covers num_workers slabs of CHUNK_SIZE each.
    # We pad vocab to 153600 to match the existing demo's layout.
    padded_vocab = 153600
    assert padded_vocab >= config.vocab_size
    hidden = config.hidden_size
    padded_weight = torch.zeros(padded_vocab, hidden,
                                dtype=torch.bfloat16, device="cuda")
    padded_weight[:config.vocab_size] = model.lm_head.weight.data
    # Reshape the lm_head's Parameter to padded size by re-creating it.
    model.lm_head.weight = torch.nn.Parameter(padded_weight)
    model.lm_head.out_features = padded_vocab
    model.argmax_partial.vocab_size = padded_vocab
    model.argmax_reduce.num_partial_tasks = num_workers
    model.argmax_partial.num_partial_tasks = num_workers

    # ---- 8. Compile the graph --------------------------------------------
    input_tokens_dt = pk.attach_input(input_tokens, name="input_token")
    with pk.compile_scope():
        model.compile(input_tokens_dt, output_tokens=output_tokens,
                      lm_head_padded_vocab=padded_vocab)

    # Optional: dump the task graph + cuda code for inspection.
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        results = pk.kn_graph.generate_task_graph(num_gpus=1, my_gpu_id=0)
        with open(os.path.join(args.output_dir, "task_graph_0.json"), "w") as f:
            f.write(results["json_file"])
        with open(os.path.join(args.output_dir, "kernel_0.cu"), "w") as f:
            f.write(results["cuda_code"])

    # nvcc compile the generated kernel.
    pk.compile(output_dir=args.output_dir)

    # ---- 9. Launch + decode ---------------------------------------------
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)

    starter.record()
    pk()
    ender.record()
    torch.cuda.synchronize()
    run_time = starter.elapsed_time(ender)

    generated_ids = tokens[0, : step[0].item() + 1]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    print(response)

    generate_len = step.max().item() + 1 - prompt_lengths[0].item()
    per_tok_ms = run_time / max(step.max().item() + 1, 1)
    print(f"Prompt length {prompt_len}, generate length {generate_len}, "
          f"per-token latency: {per_tok_ms:.3f} ms")

    # ---- 10. Save tokens for diff against demo.py output ----------------
    if args.save_tokens:
        if args.save_tokens == "auto":
            filename = "mpk_output_new.json"
            save_path = os.path.join(DEFAULT_SAVE_DIR, filename)
        else:
            save_path = args.save_tokens
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        end_idx = step[0].item() + 1
        slice_end = min(end_idx, prompt_len + MAX_SAVE_TOKENS)
        token_ids = tokens[0, prompt_len:slice_end].tolist()
        out = {
            "token_ids": token_ids,
            "text": tokenizer.decode(tokens[0, :end_idx], skip_special_tokens=True),
            "latency_ms_per_token": per_tok_ms,
            "prompt_length": prompt_len,
            "generate_length": max(0, end_idx - prompt_len),
            "mode": "mpk_new",
        }
        with open(save_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Saved tokens to {save_path}")


if __name__ == "__main__":
    main()
