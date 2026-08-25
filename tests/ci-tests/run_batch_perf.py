"""MPK batch-size performance test.

Runs the Qwen3 MPK demo pipeline with a 1-token prompt (".") and decodes to
the full sequence length (default 512 total tokens, --ignore-eos recommended),
reporting per-token latency and aggregate throughput. Sweep
--max-num-batched-tokens / --max-num-batched-requests externally to compare
batch configurations, e.g.:

    for r in 1 2 4 8; do
        python tests/ci-tests/run_batch_perf.py \
            --max-num-batched-requests $r --ignore-eos
    done
"""

import argparse
import json
import math
import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEMO_DIR = os.path.join(ROOT, "demo", "qwen3")
sys.path.insert(0, DEMO_DIR)

from models.modeling_qwen3 import Qwen3ForCausalLM  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from demo import grid_for_rmsnorm_linear_layer  # noqa: E402

DEFAULT_SAVE_DIR = os.path.join("outputs", "qwen3")
PAGE_SIZE = 4096
PROMPT = "."


def _aligned_lm_head_workers(out_width, max_workers, align=8):
    """Choose the largest worker count with aligned columns per task."""
    for workers in range(max_workers, 0, -1):
        if out_width % workers == 0 and (out_width // workers) % align == 0:
            return workers
    return 1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-num-batched-tokens", default=8, type=int,
                        help="Max number of tokens in a batch")
    parser.add_argument("--max-num-batched-requests", default=1, type=int,
                        help="Max number of requests in a batch")
    parser.add_argument("--output-dir", help="Output files directory")
    parser.add_argument("--max-seq-length", default=512, type=int,
                        help="Total sequence length (prompt + generation)")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B",
                        help="Model path on hugging face")
    parser.add_argument("--ignore-eos", action="store_true",
                        help="Ignore eos token during generation")
    return parser.parse_args()


def main():
    args = parse_args()
    print("Input arguments:", args)

    import mirage as mi

    torch.set_default_dtype(torch.bfloat16)
    torch.cuda.set_device(0)

    # One page per request is enough for max_seq_length <= PAGE_SIZE.
    pages_per_request = math.ceil(args.max_seq_length / PAGE_SIZE)
    max_num_pages = max(16, args.max_num_batched_requests * pages_per_request)

    with torch.device("cuda"):
        model = Qwen3ForCausalLM.from_pretrained(
            args.model, 1, max_num_pages=max_num_pages, page_size=PAGE_SIZE
        ).to("cuda")
        tokenizer = AutoTokenizer.from_pretrained(args.model)

    total_num_requests = args.max_num_batched_requests

    # 1-token prompt: tokenize "." directly (no chat template).
    model_inputs = tokenizer([PROMPT], return_tensors="pt").to(model.device)
    prompt_len = model_inputs.input_ids.shape[-1]
    assert prompt_len == 1, f"Expected 1-token prompt, got {prompt_len} tokens"

    tokens = torch.full((total_num_requests, args.max_seq_length), 0,
                        dtype=torch.long, device="cuda")
    tokens[:, :prompt_len] = model_inputs.input_ids[0]
    prompt_lengths = torch.full((total_num_requests,), prompt_len,
                                dtype=torch.int, device="cuda")

    positions = torch.arange(32768).unsqueeze(0).to(model.device)
    position_embeddings = model.model.rotary_emb(positions)

    input_tokens = torch.full((args.max_num_batched_tokens, 1), 0,
                              dtype=torch.long, device="cuda")
    output_tokens = torch.full((args.max_num_batched_tokens, 1), 0,
                               dtype=torch.long, device="cuda")
    step = torch.full((total_num_requests,), 0, dtype=torch.int32, device="cuda")
    num_new_tokens = torch.full((total_num_requests,), 1, dtype=torch.int32,
                                device="cuda")

    hidden_size = model.config.hidden_size
    intermediate_size = model.config.intermediate_size
    # pad vocab_size to facilitate task graph creation
    lm_head_weight = torch.cat(
        (
            model.lm_head.weight,
            torch.full((153600 - model.config.vocab_size, hidden_size), 0,
                       device="cuda"),
        ),
        0,
    )
    assert lm_head_weight.stride()[0] == hidden_size
    vocab_size = 153600
    num_q_heads = model.config.num_attention_heads
    num_kv_heads = model.config.num_key_value_heads
    head_dim = model.config.head_dim
    fused_outdim_1 = (num_q_heads + 2 * num_kv_heads) * head_dim
    fused_outdim_2 = 2 * intermediate_size

    num_workers, num_schedulers = mi.get_configurations_from_gpu(0)
    lm_head_workers = _aligned_lm_head_workers(vocab_size, num_workers)
    qo_indptr_buffer = torch.empty(
        args.max_num_batched_requests + 1, dtype=torch.int32, device="cuda")
    paged_kv_indptr_buffer = torch.empty(
        args.max_num_batched_requests + 1, dtype=torch.int32, device="cuda")
    paged_kv_indices_buffer = torch.empty(
        max_num_pages, dtype=torch.int32, device="cuda")
    paged_kv_last_page_len_buffer = torch.empty(
        args.max_num_batched_requests, dtype=torch.int32, device="cuda")

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
        max_num_pages=max_num_pages,
        page_size=PAGE_SIZE,
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
        profiler_tensor=None,
        trace_name="",
        spec_decode_config=None,
        use_cutlass_kernel=True,
    )

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
        dtype=mi.bfloat16, name="embed_out", io_category="cuda_tensor",
    )
    rmsnorm_out = mpk.new_tensor(
        dims=(args.max_num_batched_tokens, hidden_size),
        dtype=mi.bfloat16, name="rmsnorm_out", io_category="cuda_tensor",
    )
    attn_in = mpk.new_tensor(
        dims=(args.max_num_batched_tokens, fused_outdim_1),
        dtype=mi.bfloat16, name="attn_in", io_category="cuda_tensor",
    )
    attn_out = mpk.new_tensor(
        dims=(args.max_num_batched_tokens, num_q_heads * head_dim),
        dtype=mi.bfloat16, name="attn_out", io_category="cuda_tensor",
    )
    attn_proj_out = mpk.new_tensor(
        dims=(args.max_num_batched_tokens, hidden_size),
        dtype=mi.bfloat16, name="attn_proj_out", io_category="cuda_tensor",
    )
    mlp_mid = mpk.new_tensor(
        dims=(args.max_num_batched_tokens, fused_outdim_2),
        dtype=mi.bfloat16, name="mlp_mid", io_category="cuda_tensor",
    )
    silu_mul_out = mpk.new_tensor(
        dims=(args.max_num_batched_tokens, intermediate_size),
        dtype=mi.bfloat16, name="silu_mul_out", io_category="cuda_tensor",
    )
    mlp_out = mpk.new_tensor(
        dims=(args.max_num_batched_tokens, hidden_size),
        dtype=mi.bfloat16, name="mlp_out", io_category="cuda_tensor",
    )
    argmax_in = mpk.new_tensor(
        dims=(args.max_num_batched_tokens, vocab_size),
        dtype=mi.bfloat16, name="argmax_in", io_category="cuda_tensor",
    )
    argmax_part_value = mpk.new_tensor(
        dims=(args.max_num_batched_tokens, lm_head_workers),
        dtype=mi.bfloat16, name="argmax_part_value", io_category="cuda_tensor",
    )
    argmax_part_index = mpk.new_tensor(
        dims=(args.max_num_batched_tokens, lm_head_workers),
        dtype=mi.int64, name="argmax_part_index", io_category="cuda_tensor",
    )
    argmax_out = mpk.attach_input(torch_tensor=output_tokens, name="output_token")

    # Add Embed
    w = mpk.attach_input(
        torch_tensor=model.model.embed_tokens.weight, name="embed_tokens"
    )
    mpk.embed_layer(
        input=x, weight=w, output=y,
        grid_dim=(1, 1, 1), block_dim=(128, 1, 1), input_source=1,
    )
    x = y

    target_cc = (torch.cuda.get_device_properties(0).major * 10
                 + torch.cuda.get_device_properties(0).minor)
    # A current workaround to use splitk for only B200 GPUs
    use_splitk = (target_cc == 100)

    for i, layer in enumerate(model.model.layers):
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
            num_groups=model.config.num_key_value_heads,
            name=f"layer_{i}_qkv_proj",
        )
        mpk.rmsnorm_layer(
            input=x, weight=w_norm, output=rmsnorm_out,
            grid_dim=(mpk.max_num_batched_tokens, 1, 1), block_dim=(128, 1, 1),
        )
        mpk.linear_layer(
            input=rmsnorm_out, weight=w_qkv, output=attn_in,
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
        mpk.paged_attention_layer(
            input=attn_in,
            k_cache=k_cache,
            v_cache=v_cache,
            q_norm=w_q_norm,
            k_norm=w_k_norm,
            cos_pos_embed=cos_pos_embed,
            sin_pos_embed=sin_pos_embed,
            output=attn_out,
            grid_dim=(mpk.max_num_batched_requests, num_kv_heads, 1),
            block_dim=(128, 1, 1),
        )
        # add linear w/ residual
        w = mpk.attach_input(
            torch_tensor=layer.self_attn.o_proj.weight, name=f"layer_{i}_o_proj"
        )
        if use_splitk:
            attn_proj_out = x
            mpk.splitk_linear_layer(
                input=attn_out, weight=w, output=attn_proj_out,
                grid_dim=(hidden_size // 128, 128 * 128 // hidden_size, 1),
                block_dim=(256, 1, 1),
                accumulate=True,
            )
        else:
            mpk.linear_with_residual_layer(
                input=attn_out, weight=w, residual=x, output=attn_proj_out,
                grid_dim=(hidden_size // 64, 1, 1), block_dim=(128, 1, 1),
            )
        x = attn_proj_out

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
        rmsnorm_num_tasks = grid_for_rmsnorm_linear_layer(
            w_gate_proj.dim(0) + w_up_proj.dim(0))
        w_gatedup = mpk.shuffle_tensors(
            inputs=[w_gate_proj, w_up_proj],
            shuffled_dim=0,
            num_groups=rmsnorm_num_tasks // 2,
            name=f"layer_{i}_gatedup_proj",
        )
        mpk.rmsnorm_layer(
            input=x, weight=w_norm, output=rmsnorm_out,
            grid_dim=(mpk.max_num_batched_tokens, 1, 1), block_dim=(128, 1, 1),
        )
        mpk.linear_layer(
            input=rmsnorm_out, weight=w_gatedup, output=mlp_mid,
            grid_dim=(rmsnorm_num_tasks, 1, 1), block_dim=(128, 1, 1),
        )
        mpk.silu_mul_layer(
            input=mlp_mid, output=silu_mul_out,
            grid_dim=(rmsnorm_num_tasks // 2, 1, 1), block_dim=(128, 1, 1),
        )
        # add silu_mul_linear layer
        w = mpk.attach_input(
            torch_tensor=layer.mlp.down_proj.weight, name=f"layer_{i}_down_proj"
        )
        if use_splitk:
            mlp_out = x
            mpk.splitk_linear_layer(
                input=silu_mul_out, weight=w, output=mlp_out,
                grid_dim=(hidden_size // 128, 128 * 128 // hidden_size, 1),
                block_dim=(256, 1, 1),
                accumulate=True,
            )
        else:
            mpk.linear_with_residual_layer(
                input=silu_mul_out, weight=w, residual=x, output=mlp_out,
                grid_dim=(hidden_size // 64, 1, 1), block_dim=(128, 1, 1),
            )
        x = mlp_out

    # add final rmsnorm + lm_head + argmax
    w_norm = mpk.attach_input(
        torch_tensor=model.model.norm.weight, name="model_norm_weight"
    )
    w_proj = mpk.attach_input(torch_tensor=lm_head_weight, name="lm_head")
    mpk.rmsnorm_layer(
        input=x, weight=w_norm, output=rmsnorm_out,
        grid_dim=(mpk.max_num_batched_tokens, 1, 1), block_dim=(128, 1, 1),
    )
    mpk.linear_layer(
        input=rmsnorm_out, weight=w_proj, output=argmax_in,
        grid_dim=(lm_head_workers, 1, 1), block_dim=(128, 1, 1),
    )
    mpk.argmax_partial_layer(
        input=argmax_in,
        output=(argmax_part_value, argmax_part_index),
        grid_dim=(lm_head_workers, 1, 1),
        block_dim=(128, 1, 1),
    )
    mpk.argmax_reduce_layer(
        input=(argmax_part_value, argmax_part_index),
        output=argmax_out,
        grid_dim=(1, 1, 1),
        block_dim=(128, 1, 1),
    )

    mpk.compile(output_dir=args.output_dir)

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)

    starter.record()
    mpk()
    ender.record()
    torch.cuda.synchronize()
    run_time_ms = starter.elapsed_time(ender)

    # -------- Performance summary ----------
    per_request_generated = [
        max(0, step[r].item() + 1 - prompt_len) for r in range(total_num_requests)
    ]
    total_generated = sum(per_request_generated)
    seq_len = step[0].item() + 1
    per_tok_ms = run_time_ms / max(seq_len, 1)
    throughput = total_generated / (run_time_ms / 1000.0) if run_time_ms > 0 else 0.0

    sample = tokenizer.decode(
        tokens[0, : min(seq_len, 32)], skip_special_tokens=True
    )
    print(f"Sample output (first 32 tokens of request 0): {sample!r}")

    print("")
    print("==================== MPK Batch Perf ====================")
    print(f"  max_num_batched_tokens:   {args.max_num_batched_tokens}")
    print(f"  max_num_batched_requests: {args.max_num_batched_requests}")
    print(f"  prompt length:            {prompt_len}")
    print(f"  sequence length:          {seq_len} / {args.max_seq_length}")
    print(f"  generated (per request):  {per_request_generated}")
    print(f"  generated (total):        {total_generated}")
    print(f"  total time:               {run_time_ms:.1f} ms")
    print(f"  per-token latency:        {per_tok_ms:.3f} ms")
    print(f"  throughput:               {throughput:.1f} tokens/s")
    print("========================================================")

    if args.ignore_eos and seq_len < args.max_seq_length:
        print(f"WARNING: expected to decode to {args.max_seq_length} tokens "
              f"with --ignore-eos, but stopped at {seq_len}")

    # -------- Dump result JSON ----------
    os.makedirs(DEFAULT_SAVE_DIR, exist_ok=True)
    result_path = os.path.join(
        DEFAULT_SAVE_DIR,
        f"batch_perf_t{args.max_num_batched_tokens}"
        f"_r{args.max_num_batched_requests}.json",
    )
    result = {
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "max_num_batched_requests": args.max_num_batched_requests,
        "max_seq_length": args.max_seq_length,
        "model": args.model,
        "prompt_length": prompt_len,
        "sequence_length": seq_len,
        "generate_length_per_request": per_request_generated,
        "generate_length_total": total_generated,
        "total_time_ms": run_time_ms,
        "latency_ms_per_token": per_tok_ms,
        "throughput_tokens_per_s": throughput,
    }
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved perf result to {result_path}")


if __name__ == "__main__":
    main()
