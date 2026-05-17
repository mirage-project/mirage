"""Qwen3 driver using the new mirage.mpk.layers catalog (TP-aware, vLLM-style load).

Single-GPU and multi-GPU (TP) end-to-end driver:

  # tp_size=1 single GPU
  python demo/qwen3/demo_new.py --model /raid/catalyst/models/Qwen3-8B/ \\
      --max-num-batched-requests 1 --output-dir ./output/output_new

  # tp_size=2 across 2 GPUs (mpirun)
  mpirun -n 2 python demo/qwen3/demo_new.py --tp-size 2 \\
      --model /raid/catalyst/models/Qwen3-8B/ \\
      --max-num-batched-requests 1 --output-dir ./output/output_new_tp2

Weights are streamed from safetensors via ``safetensors_weights_iterator``
and dispatched through ``model.load_weights(...)``. Each TP-aware leaf
narrows the unsharded source to its local slice during load — no full
checkpoint ever materializes in CPU RAM.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict

import torch
from transformers import AutoConfig, AutoTokenizer

import mirage as mi
from mirage.mpk.models.qwen3.modeling import Qwen3ForCausalLM
from mirage.mpk.parallel import ParallelConfig
from mirage.mpk.weight_loader import (
    find_safetensors_files,
    safetensors_weights_iterator,
)


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
    parser.add_argument("--tp-size", type=int, default=1,
                        help="Tensor-parallel size; must equal mpirun world size "
                             "when >1.")
    parser.add_argument("--save-tokens", nargs="?", const="auto", default=None,
                        help="Dump first N generated token IDs + decoded text "
                             "to JSON for diff against demo.py output.")
    return parser.parse_args()


def _bootstrap_distributed(tp_size: int):
    """Return (rank, world_size). Pulls rank from MPI when mpirun-launched;
    otherwise falls back to single-process rank 0. Initializes
    ``torch.distributed`` (NCCL) for tp>1 — required by NVSHMEM bootstrap.
    """
    if tp_size <= 1:
        return 0, 1
    try:
        from mpi4py import MPI  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "tp_size > 1 requires mpi4py to read the per-process rank. "
            "Install mpi4py and launch with mpirun."
        ) from e
    comm = MPI.COMM_WORLD
    world_size = comm.Get_size()
    rank = comm.Get_rank()
    if world_size != tp_size:
        raise RuntimeError(
            f"--tp-size ({tp_size}) must match mpirun world size ({world_size})."
        )
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12355")
    # NCCL init mirrors the legacy demo's bootstrap (needed by NVSHMEM).
    import torch.distributed as dist  # local import keeps cold startup fast
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
    return rank, world_size


def main() -> None:
    args = _parse_args()
    rank, world_size = _bootstrap_distributed(args.tp_size)

    torch.set_default_dtype(torch.bfloat16)
    torch.cuda.set_device(rank if world_size > 1 else 0)

    global print
    if rank != 0:
        print = lambda *_, **__: None

    # ---- 1. Config + tokenizer -------------------------------------------
    config = AutoConfig.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    print(f"Model: {config.architectures}, "
          f"layers={config.num_hidden_layers}, hidden={config.hidden_size}, "
          f"heads={config.num_attention_heads}/{config.num_key_value_heads} "
          f"head_dim={config.head_dim}, tp_size={world_size} rank={rank}")

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

    # ---- 4. Per-rank KV cache pool ---------------------------------------
    # Each rank holds only its own KV-head slice (num_kv_heads // tp_size).
    num_kv_heads_per_rank = config.num_key_value_heads // world_size
    kv_cache_shape = (
        config.num_hidden_layers,
        args.max_num_pages,
        args.page_size,
        num_kv_heads_per_rank,
        config.head_dim,
    )
    k_cache_pool = torch.zeros(kv_cache_shape, dtype=torch.bfloat16, device="cuda")
    v_cache_pool = torch.zeros(kv_cache_shape, dtype=torch.bfloat16, device="cuda")

    # ---- 5. PersistentKernel ---------------------------------------------
    num_workers, num_schedulers = mi.get_configurations_from_gpu(0)
    spec_decode_config = mi.mpk.spec_decode_class(None, 3, 5)
    parallel_config = ParallelConfig(
        world_size=world_size, rank=rank, tp_size=world_size, ep_size=1,
    )
    pk = mi.PersistentKernel(
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
        parallel_config=parallel_config,
    )

    # ---- 6. Instantiate model (sharded shape) inside compile_scope -------
    # TP-aware leaves read current_pk().parallel_config in __init__, so
    # model construction MUST happen inside compile_scope.
    with pk.compile_scope():
        with torch.device("cuda"):
            model = Qwen3ForCausalLM(config).to("cuda", dtype=torch.bfloat16)

        # ---- 7. Stream-load weights from safetensors --------------------
        safetensors_files = find_safetensors_files(args.model)
        print(f"Loading {len(safetensors_files)} safetensors file(s) from {args.model}")
        consumed = model.load_weights(safetensors_weights_iterator(safetensors_files))
        print(f"Loaded {len(consumed)} parameter slices")

        # ---- 8. Post-load processing ------------------------------------
        model.process_weights()

        # ---- 9. Pre-pad lm_head to multi-of-grid vocab ------------------
        padded_vocab = 153600
        assert padded_vocab >= config.vocab_size
        hidden = config.hidden_size
        padded_weight = torch.zeros(padded_vocab, hidden,
                                    dtype=torch.bfloat16, device="cuda")
        padded_weight[:config.vocab_size] = model.lm_head.weight.data
        model.lm_head.weight = torch.nn.Parameter(padded_weight)
        model.lm_head.out_features = padded_vocab
        model.argmax_partial.vocab_size = padded_vocab
        model.argmax_reduce.num_partial_tasks = num_workers
        model.argmax_partial.num_partial_tasks = num_workers

        # ---- 10. Compile the graph --------------------------------------
        input_tokens_dt = pk.attach_input(input_tokens, name="input_token")
        model.compile(input_tokens_dt, output_tokens=output_tokens,
                      lm_head_padded_vocab=padded_vocab)

    # ---- 11. Optional task graph dump + nvcc compile ---------------------
    if args.output_dir:
        out_dir = (args.output_dir if world_size == 1
                   else os.path.join(args.output_dir, f"rank{rank}"))
        os.makedirs(out_dir, exist_ok=True)
        results = pk.kn_graph.generate_task_graph(num_gpus=world_size, my_gpu_id=rank)
        with open(os.path.join(out_dir, f"task_graph_{rank}.json"), "w") as f:
            f.write(results["json_file"])
        with open(os.path.join(out_dir, f"kernel_{rank}.cu"), "w") as f:
            f.write(results["cuda_code"])
        pk.compile(output_dir=out_dir)
    else:
        pk.compile(output_dir=None)

    # ---- 12. Launch + decode --------------------------------------------
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

    # ---- 13. Save tokens for diff vs legacy demo ------------------------
    if args.save_tokens and rank == 0:
        if args.save_tokens == "auto":
            filename = f"mpk_output_new_tp{world_size}.json"
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
            "tp_size": world_size,
            "mode": "mpk_new",
        }
        with open(save_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Saved tokens to {save_path}")


if __name__ == "__main__":
    main()
