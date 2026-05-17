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
  * KV absorption + W_UV→o_proj fusion done in Python at load time.

Weight loading: ``_load_hf_weights_with_absorption`` loads only the
requested layer indices from the sharded HF safetensors, dequantizes
FP8 → BF16 via the shared :func:`demo.deepseek_v3.models.convert.dequantize_fp8`
helper, absorbs ``kv_b_proj`` into ``q_b_proj``, fuses ``W_UV`` into
``o_proj``, and (for MoE layers) stacks per-expert weights into
``experts.w13.weight`` / ``experts.w2.weight``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

import torch
from transformers import AutoConfig, AutoTokenizer

import mirage as mi
from mirage.mpk.models.deepseek_v3.modeling import DeepseekV3ForCausalLM


# Make the existing demo's convert.py importable for FP8 dequant + absorb.
_DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_DEMO_DIR, "models"))
from convert import dequantize_fp8, absorb_kv_into_q, get_model_params, is_fp8  # noqa: E402


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
# Weight loading
# ---------------------------------------------------------------------------


def _selectively_load_layers(
    model_path: str,
    layer_indices: List[int],
) -> Dict[str, torch.Tensor]:
    """Load only the requested layers + global (embed/norm/lm_head) tensors.

    Returns a state_dict keyed by the original HF names, on CPU.
    """
    from safetensors import safe_open

    index_path = os.path.join(model_path, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"No model.safetensors.index.json under {model_path}. The new "
            "driver expects a sharded HF DeepSeek V3 checkpoint."
        )
    with open(index_path) as f:
        index = json.load(f)

    needed_prefixes = [
        "model.embed_tokens.",
        "model.norm.",
        "lm_head.",
    ]
    for li in layer_indices:
        needed_prefixes.append(f"model.layers.{li}.")

    shard_to_keys: Dict[str, List[str]] = {}
    for key, shard in index["weight_map"].items():
        if any(key.startswith(p) for p in needed_prefixes):
            shard_to_keys.setdefault(shard, []).append(key)

    state_dict: Dict[str, torch.Tensor] = {}
    for shard, keys in sorted(shard_to_keys.items()):
        shard_path = os.path.join(model_path, shard)
        print(f"  Loading {len(keys)} keys from {shard}")
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for key in keys:
                state_dict[key] = f.get_tensor(key)
    print(f"  Loaded {len(state_dict)} keys total (CPU).")
    return state_dict


def _maybe_dequant(name: str, w: torch.Tensor,
                   state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    """If w is FP8 and a ``<name>_scale_inv`` companion is present,
    dequantize to BF16. Otherwise return w unchanged."""
    if not is_fp8(w):
        return w
    s_key = name + "_scale_inv"
    if s_key in state_dict:
        return dequantize_fp8(w, state_dict[s_key], target_dtype=torch.bfloat16)
    # No scale found — best-effort cast.
    return w.to(torch.bfloat16)


def _load_hf_weights_with_absorption(
    model_path: str,
    config,
    layer_indices: List[int],
) -> Dict[str, torch.Tensor]:
    """Load + convert HF DeepSeek V3 weights into the modeling.py's
    parameter names.

    Conversions performed:
      1. FP8 → BF16 (via demo.deepseek_v3.models.convert.dequantize_fp8).
      2. ``kv_b_proj`` absorbed into ``q_b_proj`` so the resulting
         ``q_b_proj`` has shape ``(H*(kv_lora_rank+qk_rope_head_dim),
         q_lora_rank)``.
      3. ``W_UV`` (the V half of kv_b_proj) fused into ``o_proj`` so the
         resulting ``o_proj`` has shape ``(hidden, H*kv_lora_rank)``.
      4. Per-MoE-layer: stack all ``experts.{e}.{gate,up,down}_proj.weight``
         into single ``experts.w13.weight`` and ``experts.w2.weight``
         tensors with the MoEW13 / MoEW2 layouts.
      5. Build modeling.py-shaped keys
         (``model.layers.{i}.self_attn.q_a_proj_weight`` etc.).
    """
    state_dict = _selectively_load_layers(model_path, layer_indices)

    config_dict = config.to_dict()
    mp = get_model_params(config_dict)
    num_heads = mp["num_heads"]
    qk_nope = mp["qk_nope_head_dim"]
    qk_rope = mp["qk_rope_head_dim"]
    kv_lora_rank = mp["kv_lora_rank"]
    v_dim = mp["v_head_dim"]
    first_moe = mp["first_moe_layer"]
    num_experts = mp["num_experts"]

    out: Dict[str, torch.Tensor] = {}

    # ---- 1. Global tensors: embed, final norm, lm_head ----
    for key in ("model.embed_tokens.weight", "model.norm.weight",
                "lm_head.weight"):
        if key in state_dict:
            out[key] = _maybe_dequant(key, state_dict[key], state_dict).to(
                torch.bfloat16
            ).contiguous()

    # ---- 2. Per-layer conversions ----
    for li in layer_indices:
        layer_prefix = f"model.layers.{li}."
        attn = f"{layer_prefix}self_attn."

        # ---- Layernorms ----
        for hf_name, modeling_name in [
            (f"{layer_prefix}input_layernorm.weight",
             f"{layer_prefix}input_layernorm.weight"),
            (f"{layer_prefix}post_attention_layernorm.weight",
             f"{layer_prefix}post_attention_layernorm.weight"),
            (f"{attn}q_a_layernorm.weight",
             f"{attn}q_a_layernorm.weight"),
            (f"{attn}kv_a_layernorm.weight",
             f"{attn}kv_a_layernorm.weight"),
        ]:
            if hf_name in state_dict:
                out[modeling_name] = _maybe_dequant(
                    hf_name, state_dict[hf_name], state_dict
                ).to(torch.bfloat16).contiguous()

        # ---- MLA absorption: q_b absorbs W_UK ----
        q_key = f"{attn}q_b_proj.weight"
        kv_key = f"{attn}kv_b_proj.weight"
        o_key = f"{attn}o_proj.weight"
        if q_key in state_dict and kv_key in state_dict:
            q_w = _maybe_dequant(
                q_key, state_dict[q_key], state_dict
            ).float()
            kv_w = _maybe_dequant(
                kv_key, state_dict[kv_key], state_dict
            ).float()

            # Absorb kv_b into q_b: result shape
            # (H * (kv_lora_rank + qk_rope_head_dim), q_lora_rank)
            absorbed = absorb_kv_into_q(q_w, kv_w, mp).to(torch.bfloat16)
            out[q_key] = absorbed.contiguous()

            # Fuse W_UV (the v half of kv_b_proj) into o_proj. Final
            # o_proj shape: (hidden, H * kv_lora_rank).
            kv_b_reshaped = kv_w.reshape(num_heads, qk_nope + v_dim, kv_lora_rank)
            W_UV = kv_b_reshaped[:, qk_nope:, :]  # (H, v_dim, kv_lora_rank)
            if o_key in state_dict:
                o_w_bf16 = _maybe_dequant(
                    o_key, state_dict[o_key], state_dict
                ).to(torch.bfloat16)
                hidden_dim = o_w_bf16.shape[0]
                # Original o_proj: (hidden, H * v_dim) — reshape to per-head.
                o_reshaped = o_w_bf16.reshape(
                    hidden_dim, num_heads, v_dim
                ).float()
                # Fused o_proj: (hidden, H, kv_lora_rank)
                o_fused = torch.einsum("dhn,hnk->dhk", o_reshaped, W_UV.float())
                o_flat = o_fused.reshape(
                    hidden_dim, num_heads * kv_lora_rank
                ).to(torch.bfloat16)
                out[o_key] = o_flat.contiguous()

        # ---- q_a_proj, kv_a_proj_with_mqa ----
        for hf_name in [
            f"{attn}q_a_proj.weight",
            f"{attn}kv_a_proj_with_mqa.weight",
        ]:
            if hf_name in state_dict:
                out[hf_name] = _maybe_dequant(
                    hf_name, state_dict[hf_name], state_dict
                ).to(torch.bfloat16).contiguous()

        # ---- MLP: dense vs MoE ----
        if li < first_moe:
            # Dense MLP: gate / up / down (BF16, no Python-level fusion;
            # pk.shuffle_tensors fuses gate+up at compile time).
            for hf_name in [
                f"{layer_prefix}mlp.gate_proj.weight",
                f"{layer_prefix}mlp.up_proj.weight",
                f"{layer_prefix}mlp.down_proj.weight",
            ]:
                if hf_name in state_dict:
                    out[hf_name] = _maybe_dequant(
                        hf_name, state_dict[hf_name], state_dict
                    ).to(torch.bfloat16).contiguous()
        else:
            # MoE layer.
            # Router gate.weight + e_score_correction_bias.
            for hf_name in [
                f"{layer_prefix}mlp.gate.weight",
                f"{layer_prefix}mlp.gate.e_score_correction_bias",
            ]:
                if hf_name in state_dict:
                    raw = _maybe_dequant(
                        hf_name, state_dict[hf_name], state_dict
                    )
                    # Router gate matrix in BF16; bias kept in FP32
                    # (MoETopkRouting.bias dtype).
                    if hf_name.endswith("e_score_correction_bias"):
                        out[hf_name] = raw.to(torch.float32).contiguous()
                    else:
                        out[hf_name] = raw.to(torch.bfloat16).contiguous()

            # Shared experts.
            for hf_name in [
                f"{layer_prefix}mlp.shared_experts.gate_proj.weight",
                f"{layer_prefix}mlp.shared_experts.up_proj.weight",
                f"{layer_prefix}mlp.shared_experts.down_proj.weight",
            ]:
                if hf_name in state_dict:
                    out[hf_name] = _maybe_dequant(
                        hf_name, state_dict[hf_name], state_dict
                    ).to(torch.bfloat16).contiguous()

            # Routed experts: stack into experts.w13 / experts.w2.
            inter = config.moe_intermediate_size
            hidden = config.hidden_size
            w13_stack = torch.empty(
                num_experts, 2 * inter, hidden, dtype=torch.bfloat16
            )
            w2_stack = torch.empty(
                num_experts, hidden, inter, dtype=torch.bfloat16
            )
            for e in range(num_experts):
                ep = f"{layer_prefix}mlp.experts.{e}."
                g_key = f"{ep}gate_proj.weight"
                u_key = f"{ep}up_proj.weight"
                d_key = f"{ep}down_proj.weight"
                if g_key in state_dict:
                    g = _maybe_dequant(
                        g_key, state_dict[g_key], state_dict
                    ).to(torch.bfloat16)
                    u = _maybe_dequant(
                        u_key, state_dict[u_key], state_dict
                    ).to(torch.bfloat16)
                    d = _maybe_dequant(
                        d_key, state_dict[d_key], state_dict
                    ).to(torch.bfloat16)
                    # W13 layout: [gate | up] concatenated along dim 0.
                    w13_stack[e, :inter] = g
                    w13_stack[e, inter:] = u
                    w2_stack[e] = d
            out[f"{layer_prefix}mlp.experts.w13.weight"] = (
                w13_stack.contiguous()
            )
            out[f"{layer_prefix}mlp.experts.w2.weight"] = (
                w2_stack.contiguous()
            )

    # Move to CUDA for the load_state_dict step (Parameter device).
    cuda_out = {k: v.to("cuda") for k, v in out.items()}
    return cuda_out


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
        print("Loading HF weights with KV absorption + W_UV fusion...")
        state_dict = _load_hf_weights_with_absorption(
            args.model_path, config, layer_indices
        )
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[warn] {len(missing)} missing keys (first 10): "
                  f"{missing[:10]}")
        if unexpected:
            print(f"[warn] {len(unexpected)} unexpected keys (first 10): "
                  f"{unexpected[:10]}")
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
