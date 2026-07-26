"""P8 probe -- prefill-iteration cost vs decode-iteration cost.

Adaptation of tests/ci-tests/run_batch_perf.py [MG S6.1]: same MODE_OFFLINE
Qwen3-8B graph (single request, use_cutlass_kernel=True on B200), but the
1-token "." prompt is replaced by an exact-length synthetic prompt so we can
sweep --input-len. Construction (v1-architecture.md S14 P8):

    t_pf  = [T(512,128) - T(32,128)] / ((512-32)/mbt)
    t_dec = latency_ms_per_token of the T(32,128) run (mostly-decode; "the
            same run" per the probe spec -- run_batch_perf.py already reports
            this field, we just reuse it instead of re-deriving decode cost)
    r     = t_pf / t_dec

Emits a raw per-run JSON plus the final {r, band, workload_pin_stands}
verdict (probes/runtime/p8_verdict.json is the copy that ships in-repo).
"""
import argparse
import json
import math
import os
import sys
import time

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "mirage"))
DEMO_DIR = os.path.join(ROOT, "demo", "qwen3")
sys.path.insert(0, DEMO_DIR)

from models.modeling_qwen3 import Qwen3ForCausalLM  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from demo import grid_for_rmsnorm_linear_layer  # noqa: E402

PAGE_SIZE = 4096


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--mbt", type=int, default=8, help="max_num_batched_tokens")
    p.add_argument("--input-len", type=int, action="append", required=True,
                   help="repeatable; needs exactly 2 values, low then high")
    p.add_argument("--output-len", type=int, default=128)
    p.add_argument("--requests", type=int, default=1)
    p.add_argument("--ignore-eos", action="store_true")
    p.add_argument("--output-dir-base", default="/home/muhengl/mpk-qwen35/probes/runtime_out/p8_kernel_cache")
    p.add_argument("--result-json", default="/home/muhengl/mpk-qwen35/probes/runtime_out/p8_raw_result.json")
    p.add_argument("--verdict-json", default="/home/muhengl/mpk-qwen35/probes/runtime_out/p8_verdict.json")
    return p.parse_args()


def build_prompt_ids(tokenizer, n):
    text = "The quick brown fox jumps over the lazy dog. " * (n // 4 + 20)
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    while len(ids) < n:
        ids = ids + ids
    return ids[:n]


def run_one_config(model, tokenizer, args, input_len):
    assert input_len % args.mbt == 0, "input_len must be a multiple of --mbt"
    max_seq_length = input_len + args.output_len
    pages_per_request = math.ceil(max_seq_length / PAGE_SIZE)
    max_num_pages = max(16, args.requests * pages_per_request)

    tokens = torch.full((args.requests, max_seq_length), 0, dtype=torch.long, device="cuda")
    prompt_ids = build_prompt_ids(tokenizer, input_len)
    tokens[:, :input_len] = torch.tensor(prompt_ids, dtype=torch.long, device="cuda")
    prompt_lengths = torch.full((args.requests,), input_len, dtype=torch.int, device="cuda")
    step = torch.full((args.requests,), 0, dtype=torch.int32, device="cuda")
    num_new_tokens = torch.full((args.requests,), 1, dtype=torch.int32, device="cuda")

    positions = torch.arange(32768).unsqueeze(0).to(model.device)
    position_embeddings = model.model.rotary_emb(positions)
    input_tokens = torch.full((args.mbt, 1), 0, dtype=torch.long, device="cuda")
    output_tokens = torch.full((args.mbt, 1), 0, dtype=torch.long, device="cuda")

    hidden_size = model.config.hidden_size
    intermediate_size = model.config.intermediate_size
    lm_head_weight = torch.cat(
        (model.lm_head.weight,
         torch.full((153600 - model.config.vocab_size, hidden_size), 0, device="cuda")), 0)
    vocab_size = 153600
    num_q_heads = model.config.num_attention_heads
    num_kv_heads = model.config.num_key_value_heads
    head_dim = model.config.head_dim
    fused_outdim_1 = (num_q_heads + 2 * num_kv_heads) * head_dim
    fused_outdim_2 = 2 * intermediate_size

    import mirage as mi
    num_workers, num_schedulers = mi.get_configurations_from_gpu(0)
    qo_indptr_buffer = torch.empty(args.requests + 1, dtype=torch.int32, device="cuda")
    paged_kv_indptr_buffer = torch.empty(args.requests + 1, dtype=torch.int32, device="cuda")
    paged_kv_indices_buffer = torch.empty(max_num_pages, dtype=torch.int32, device="cuda")
    paged_kv_last_page_len_buffer = torch.empty(args.requests, dtype=torch.int32, device="cuda")

    mpk = mi.PersistentKernel(
        mode="offline", world_size=1, mpi_rank=0,
        num_workers=num_workers, num_local_schedulers=num_schedulers,
        num_remote_schedulers=0, max_seq_length=max_seq_length,
        max_num_batched_requests=args.requests, max_num_batched_tokens=args.mbt,
        max_num_pages=max_num_pages, page_size=PAGE_SIZE,
        eos_token_id=model.config.eos_token_id if not args.ignore_eos else -1,
        meta_tensors={
            "step": step, "tokens": tokens, "input_tokens": input_tokens,
            "output_tokens": output_tokens, "num_new_tokens": num_new_tokens,
            "prompt_lengths": prompt_lengths, "qo_indptr_buffer": qo_indptr_buffer,
            "paged_kv_indptr_buffer": paged_kv_indptr_buffer,
            "paged_kv_indices_buffer": paged_kv_indices_buffer,
            "paged_kv_last_page_len_buffer": paged_kv_last_page_len_buffer,
        },
        profiler_tensor=None, trace_name="", spec_decode_config=None,
        use_cutlass_kernel=True,
    )

    x = mpk.attach_input(torch_tensor=input_tokens, name="input_token")
    cos_pos_embed = mpk.attach_input(torch_tensor=position_embeddings[0][0, :4096, :], name="cos_position_embedding")
    sin_pos_embed = mpk.attach_input(torch_tensor=position_embeddings[1][0, :4096, :], name="sin_position_embedding")

    y = mpk.new_tensor(dims=(args.mbt, hidden_size), dtype=mi.bfloat16, name="embed_out", io_category="cuda_tensor")
    rmsnorm_out = mpk.new_tensor(dims=(args.mbt, hidden_size), dtype=mi.bfloat16, name="rmsnorm_out", io_category="cuda_tensor")
    attn_in = mpk.new_tensor(dims=(args.mbt, fused_outdim_1), dtype=mi.bfloat16, name="attn_in", io_category="cuda_tensor")
    attn_out = mpk.new_tensor(dims=(args.mbt, num_q_heads * head_dim), dtype=mi.bfloat16, name="attn_out", io_category="cuda_tensor")
    attn_proj_out = mpk.new_tensor(dims=(args.mbt, hidden_size), dtype=mi.bfloat16, name="attn_proj_out", io_category="cuda_tensor")
    mlp_mid = mpk.new_tensor(dims=(args.mbt, fused_outdim_2), dtype=mi.bfloat16, name="mlp_mid", io_category="cuda_tensor")
    silu_mul_out = mpk.new_tensor(dims=(args.mbt, intermediate_size), dtype=mi.bfloat16, name="silu_mul_out", io_category="cuda_tensor")
    mlp_out = mpk.new_tensor(dims=(args.mbt, hidden_size), dtype=mi.bfloat16, name="mlp_out", io_category="cuda_tensor")
    argmax_in = mpk.new_tensor(dims=(args.mbt, vocab_size), dtype=mi.bfloat16, name="argmax_in", io_category="cuda_tensor")
    argmax_part_value = mpk.new_tensor(dims=(args.mbt, mpk.num_workers), dtype=mi.bfloat16, name="argmax_part_value", io_category="cuda_tensor")
    argmax_part_index = mpk.new_tensor(dims=(args.mbt, mpk.num_workers), dtype=mi.int64, name="argmax_part_index", io_category="cuda_tensor")
    argmax_out = mpk.attach_input(torch_tensor=output_tokens, name="output_token")

    w = mpk.attach_input(torch_tensor=model.model.embed_tokens.weight, name="embed_tokens")
    mpk.embed_layer(input=x, weight=w, output=y, grid_dim=(1, 1, 1), block_dim=(128, 1, 1), input_source=1)
    x = y

    target_cc = torch.cuda.get_device_properties(0).major * 10 + torch.cuda.get_device_properties(0).minor
    use_splitk = (target_cc == 100)

    for i, layer in enumerate(model.model.layers):
        w_norm = mpk.attach_input(torch_tensor=layer.input_layernorm.weight, name=f"layer_{i}_input_layernorm")
        w_q = mpk.attach_input(torch_tensor=layer.self_attn.q_proj.weight, name=f"layer_{i}_q_proj")
        w_k = mpk.attach_input(torch_tensor=layer.self_attn.k_proj.weight, name=f"layer_{i}_k_proj")
        w_v = mpk.attach_input(torch_tensor=layer.self_attn.v_proj.weight, name=f"layer_{i}_v_proj")
        w_qkv = mpk.shuffle_tensors(inputs=[w_q, w_k, w_v], shuffled_dim=0, num_groups=model.config.num_key_value_heads, name=f"layer_{i}_qkv_proj")
        mpk.rmsnorm_layer(input=x, weight=w_norm, output=rmsnorm_out, grid_dim=(mpk.max_num_batched_tokens, 1, 1), block_dim=(128, 1, 1))
        mpk.linear_layer(input=rmsnorm_out, weight=w_qkv, output=attn_in, grid_dim=(grid_for_rmsnorm_linear_layer(w_qkv.dim(0)), 1, 1), block_dim=(128, 1, 1))
        w_q_norm = mpk.attach_input(torch_tensor=layer.self_attn.q_norm.weight, name=f"layer_{i}_q_norm")
        w_k_norm = mpk.attach_input(torch_tensor=layer.self_attn.k_norm.weight, name=f"layer_{i}_k_norm")
        k_cache = mpk.attach_input(torch_tensor=model.model.kv_cache[0][i], name=f"layer_{i}_k_cache")
        v_cache = mpk.attach_input(torch_tensor=model.model.kv_cache[1][i], name=f"layer_{i}_v_cache")
        mpk.paged_attention_layer(input=attn_in, k_cache=k_cache, v_cache=v_cache, q_norm=w_q_norm, k_norm=w_k_norm,
                                   cos_pos_embed=cos_pos_embed, sin_pos_embed=sin_pos_embed, output=attn_out,
                                   grid_dim=(mpk.max_num_batched_requests, num_kv_heads, 1), block_dim=(128, 1, 1))
        w = mpk.attach_input(torch_tensor=layer.self_attn.o_proj.weight, name=f"layer_{i}_o_proj")
        if use_splitk:
            attn_proj_out = x
            mpk.splitk_linear_layer(input=attn_out, weight=w, output=attn_proj_out, grid_dim=(hidden_size // 128, 128 * 128 // hidden_size, 1), block_dim=(256, 1, 1))
        else:
            mpk.linear_with_residual_layer(input=attn_out, weight=w, residual=x, output=attn_proj_out, grid_dim=(hidden_size // 64, 1, 1), block_dim=(128, 1, 1))
        x = attn_proj_out

        w_norm = mpk.attach_input(torch_tensor=layer.post_attention_layernorm.weight, name=f"layer_{i}_post_attn_layernorm")
        w_gate_proj = mpk.attach_input(torch_tensor=layer.mlp.gate_proj.weight, name=f"layer_{i}_gate_proj")
        w_up_proj = mpk.attach_input(torch_tensor=layer.mlp.up_proj.weight, name=f"layer_{i}_up_proj")
        rmsnorm_num_tasks = grid_for_rmsnorm_linear_layer(w_gate_proj.dim(0) + w_up_proj.dim(0))
        w_gatedup = mpk.shuffle_tensors(inputs=[w_gate_proj, w_up_proj], shuffled_dim=0, num_groups=rmsnorm_num_tasks // 2, name=f"layer_{i}_gatedup_proj")
        mpk.rmsnorm_layer(input=x, weight=w_norm, output=rmsnorm_out, grid_dim=(mpk.max_num_batched_tokens, 1, 1), block_dim=(128, 1, 1))
        mpk.linear_layer(input=rmsnorm_out, weight=w_gatedup, output=mlp_mid, grid_dim=(rmsnorm_num_tasks, 1, 1), block_dim=(128, 1, 1))
        mpk.silu_mul_layer(input=mlp_mid, output=silu_mul_out, grid_dim=(rmsnorm_num_tasks // 2, 1, 1), block_dim=(128, 1, 1))
        w = mpk.attach_input(torch_tensor=layer.mlp.down_proj.weight, name=f"layer_{i}_down_proj")
        if use_splitk:
            mlp_out = x
            mpk.splitk_linear_layer(input=silu_mul_out, weight=w, output=mlp_out, grid_dim=(hidden_size // 128, 128 * 128 // hidden_size, 1), block_dim=(256, 1, 1))
        else:
            mpk.linear_with_residual_layer(input=silu_mul_out, weight=w, residual=x, output=mlp_out, grid_dim=(hidden_size // 64, 1, 1), block_dim=(128, 1, 1))
        x = mlp_out

    w_norm = mpk.attach_input(torch_tensor=model.model.norm.weight, name="model_norm_weight")
    w_proj = mpk.attach_input(torch_tensor=lm_head_weight, name="lm_head")
    mpk.rmsnorm_layer(input=x, weight=w_norm, output=rmsnorm_out, grid_dim=(mpk.max_num_batched_tokens, 1, 1), block_dim=(128, 1, 1))
    mpk.linear_layer(input=rmsnorm_out, weight=w_proj, output=argmax_in, grid_dim=(mpk.num_workers, 1, 1), block_dim=(128, 1, 1))
    mpk.argmax_partial_layer(input=argmax_in, output=(argmax_part_value, argmax_part_index), grid_dim=(mpk.num_workers, 1, 1), block_dim=(128, 1, 1))
    mpk.argmax_reduce_layer(input=(argmax_part_value, argmax_part_index), output=argmax_out, grid_dim=(1, 1, 1), block_dim=(128, 1, 1))

    output_dir = f"{args.output_dir_base}_il{input_len}"
    print(f"[p8] compiling for input_len={input_len} (max_seq_length={max_seq_length}) -> {output_dir}", flush=True)
    t0 = time.time()
    mpk.compile(output_dir=output_dir)
    compile_s = time.time() - t0
    print(f"[p8] compile done in {compile_s:.1f}s", flush=True)

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    starter.record()
    mpk()
    ender.record()
    torch.cuda.synchronize()
    run_time_ms = starter.elapsed_time(ender)

    seq_len = step[0].item() + 1
    per_tok_ms = run_time_ms / max(seq_len, 1)
    n_pf = input_len // args.mbt

    mpk.finalize()
    del mpk
    torch.cuda.empty_cache()

    return {
        "input_len": input_len, "output_len": args.output_len, "mbt": args.mbt,
        "requests": args.requests, "max_seq_length": max_seq_length,
        "n_pf_iters": n_pf, "compile_seconds": compile_s,
        "total_time_ms": run_time_ms, "sequence_length_achieved": seq_len,
        "latency_ms_per_token_blended": per_tok_ms,
        "expected_seq_len": max_seq_length,
        "seq_len_matches_expected": seq_len == max_seq_length,
    }


def main():
    args = parse_args()
    assert len(args.input_len) == 2, "need exactly --input-len low --input-len high"
    lo, hi = sorted(args.input_len)
    print("Input arguments:", args, flush=True)

    torch.set_default_dtype(torch.bfloat16)
    torch.cuda.set_device(0)

    with torch.device("cuda"):
        model = Qwen3ForCausalLM.from_pretrained(args.model, 1, max_num_pages=16, page_size=PAGE_SIZE).to("cuda")
        tokenizer = AutoTokenizer.from_pretrained(args.model)

    runs = {}
    for input_len in (lo, hi):
        model.model.kv_cache[0].zero_()
        model.model.kv_cache[1].zero_()
        runs[input_len] = run_one_config(model, tokenizer, args, input_len)
        print(f"[p8] run(input_len={input_len}) -> {runs[input_len]}", flush=True)

    r_lo, r_hi = runs[lo], runs[hi]
    delta_n_pf = r_hi["n_pf_iters"] - r_lo["n_pf_iters"]
    assert delta_n_pf > 0
    t_pf = (r_hi["total_time_ms"] - r_lo["total_time_ms"]) / delta_n_pf
    t_dec_primary = r_lo["latency_ms_per_token_blended"]  # "the same run" = the low (mostly-decode) run, per spec
    # cross-check: purified decode estimate backing out the low run's own prefill cost
    t_dec_purified = (r_lo["total_time_ms"] - r_lo["n_pf_iters"] * t_pf) / args.output_len

    r_primary = t_pf / t_dec_primary
    r_secondary = t_pf / t_dec_purified if t_dec_purified > 0 else float("nan")

    def band_of(r):
        if r <= 1.0:
            return "r<=1.0"
        elif r <= 2.25:
            return "1.0<r<=2.25"
        else:
            return "r>2.25"

    band = band_of(r_primary)
    workload_pin_stands = band in ("r<=1.0", "1.0<r<=2.25")

    raw = {
        "probe": "P8", "model": args.model, "runs": runs,
        "t_pf_ms_per_prefill_iter": t_pf,
        "t_dec_ms_primary_source": f"latency_ms_per_token_blended of input_len={lo} run",
        "t_dec_ms_primary": t_dec_primary,
        "t_dec_ms_purified_crosscheck": t_dec_purified,
        "r_primary": r_primary, "r_secondary_crosscheck": r_secondary,
        "r_primary_vs_secondary_rel_diff": (abs(r_primary - r_secondary) / r_primary) if r_primary else None,
    }
    verdict = {
        "r": r_primary, "band": band, "workload_pin_stands": workload_pin_stands,
        "r_crosscheck": r_secondary, "t_pf_ms": t_pf, "t_dec_ms": t_dec_primary,
        "point_prediction_1.5x_held": r_primary <= 1.5,
        "evidence": raw,
    }

    os.makedirs(os.path.dirname(args.result_json), exist_ok=True)
    with open(args.result_json, "w") as f:
        json.dump(raw, f, indent=2)
    with open(args.verdict_json, "w") as f:
        json.dump(verdict, f, indent=2)

    print("==================== P8 VERDICT ====================")
    print(json.dumps(verdict, indent=2))
    print("======================================================")


if __name__ == "__main__":
    main()
