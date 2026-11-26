"""Two-process, same-device BF16 token gate for Qwen3.8 and Qwen3.6 offline inference.

Freeze the HF oracle before building MPK so two model copies are never resident
on the same GPU. Use tokenwise eager prefill on both sides. An exact-token
mismatch is a failed validation, not a performance result.
"""

import argparse
import hashlib
import json
from pathlib import Path


def prompts(path, vocab):
    rows = json.loads(path.read_text())
    if (not isinstance(rows, list) or len(rows) not in (1, 8)
            or not all(isinstance(row, list) and row for row in rows)
            or len({len(row) for row in rows}) != 1
            or any(type(token) is not int or not 0 <= token < vocab
                   for row in rows for token in row)):
        raise ValueError("Expected one or eight equal-length in-vocabulary token rows")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("freeze", "verify"), required=True)
    parser.add_argument("--architecture", choices=("qwen38", "qwen36"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prompts-json", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--capture-layers", action="store_true",
                        help="Compare every decoder layer at the final generated token")
    parser.add_argument("--capture-attention-layer", type=int,
                        help="With layer capture, inspect one full-attention layer's projections and probabilities")
    parser.add_argument("--capture-gdn-layer", type=int,
                        help="With layer capture, inspect one GDN layer's projections and recurrent output")
    parser.add_argument("--compiled-dir", type=Path)
    parser.add_argument("--reuse-compiled", action="store_true",
                        help="Load a previously compiled matching kernel for repeated diagnostics")
    parser.add_argument("--final-head-snapshot", type=Path,
                        help="Save final hidden, norm and logits tensors for numerical diagnosis")
    parser.add_argument("--gdn-snapshot", type=Path,
                        help="Save captured HF and MPK GDN tensors for a one-step probe")
    parser.add_argument("--moe-snapshot-layer", type=int,
                        help="Save HF and MPK MoE intermediates for this decoder layer")
    parser.add_argument("--moe-snapshot", type=Path,
                        help="Output path for the MoE intermediate snapshot")
    args = parser.parse_args()

    import torch
    import transformers
    from mirage.mpk.models.qwen3_5.weight_loader import Qwen35WeightLoader, Qwen36WeightLoader

    if torch.cuda.device_count() != 1 or torch.cuda.get_device_capability() != (10, 0):
        raise RuntimeError("Expose exactly one B200 GPU for this B1/B8 acceptance gate")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    loader = (Qwen36WeightLoader if args.architecture == "qwen36" else Qwen35WeightLoader)(args.model)
    loader.inspect()  # Validate the complete indexed text checkpoint first.
    rows = prompts(args.prompts_json, loader.config.vocab_size)
    if args.max_new_tokens < 1 or len(rows[0]) + args.max_new_tokens > 1024:
        raise ValueError("Prompt plus generated tokens must fit one 1024-token page")
    if args.capture_layers and args.max_new_tokens < 1:
        raise ValueError("Layer capture requires a generated token")
    if args.capture_attention_layer is not None and (not args.capture_layers or
            args.capture_attention_layer not in loader.config.attention_layers):
        raise ValueError("Attention capture requires a valid full-attention layer")
    if args.capture_gdn_layer is not None and (not args.capture_layers or
            args.capture_gdn_layer not in loader.config.gdn_layers):
        raise ValueError("GDN capture requires a valid linear-attention layer")
    if args.moe_snapshot_layer is not None and (
            args.architecture != "qwen36" or not args.capture_layers or
            args.moe_snapshot is None or
            not 0 <= args.moe_snapshot_layer < loader.config.num_hidden_layers):
        raise ValueError("MoE snapshot requires Qwen3.6 layer capture and an output path")
    config_hash = hashlib.sha256((args.model / "config.json").read_bytes()).hexdigest()
    index_hash = hashlib.sha256((args.model / "model.safetensors.index.json").read_bytes()).hexdigest()
    prompt_hash = hashlib.sha256(args.prompts_json.read_bytes()).hexdigest()
    gpu_uuid = str(torch.cuda.get_device_properties(0).uuid)

    if args.phase == "freeze":
        if args.architecture == "qwen36":
            from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
                Qwen3_5MoeForConditionalGeneration as HFModel,
            )
        else:
            from transformers import Qwen3_5ForConditionalGeneration as HFModel
        model_options = ({"experts_implementation": "grouped_mm"}
                         if args.architecture == "qwen36" else {})
        model = HFModel.from_pretrained(
            args.model, dtype=torch.bfloat16, attn_implementation="eager",
            local_files_only=True, **model_options).to("cuda").eval()
        captured = {}
        captured_norm = {}
        captured_route = {}
        captured_router_logits = {}
        captured_router_scores = {}
        captured_experts = {}
        captured_shared = {}
        captured_shared_gate = {}
        captured_mlp = {}
        captured_attention = {}
        captured_gdn = {}
        captured_value_history = []
        capture_enabled = [False]
        handles = []
        if args.capture_layers:
            for index, layer in enumerate(model.model.language_model.layers):
                def capture(_module, _inputs, output, index=index):
                    if capture_enabled[0]:
                        value = output[0] if isinstance(output, tuple) else output
                        captured[index] = value[:, -1].detach().cpu().clone()
                handles.append(layer.register_forward_hook(capture))
                if index == args.capture_attention_layer:
                    attn = layer.self_attn

                    def capture_projection(name):
                        def hook(_module, inputs, output):
                            if name == "v":
                                captured_value_history.append(output[:, -1].detach().cpu().clone())
                            if capture_enabled[0]:
                                captured_attention[name] = output[:, -1].detach().cpu().clone()
                                if name == "output":
                                    captured_attention["gated"] = inputs[0][:, -1].detach().cpu().clone()
                        return hook

                    def capture_attn(_module, _inputs, output):
                        if capture_enabled[0]:
                            captured_attention["probabilities"] = output[1][:, :, -1].detach().cpu().clone()

                    def capture_input_norm(_module, _inputs, output):
                        if capture_enabled[0]:
                            captured_attention["normalized"] = output[:, -1].detach().cpu().clone()

                    for projection, name in (("q_proj", "qg"), ("k_proj", "k"),
                                             ("v_proj", "v"), ("o_proj", "output")):
                        handles.append(getattr(attn, projection).register_forward_hook(
                            capture_projection(name)))
                    handles.append(attn.register_forward_hook(capture_attn))
                    handles.append(layer.input_layernorm.register_forward_hook(capture_input_norm))
                if index == args.capture_gdn_layer:
                    gdn = layer.linear_attn

                    def capture_gdn_projection(name):
                        def hook(_module, inputs, output):
                            if capture_enabled[0]:
                                captured_gdn[name] = output.reshape(len(rows), -1).detach().cpu().clone()
                                if name == "output":
                                    captured_gdn["gated"] = inputs[0].reshape(
                                        len(rows), -1).detach().cpu().clone()
                        return hook

                    def capture_gdn_norm(_module, inputs, output):
                        if capture_enabled[0]:
                            captured_gdn["raw"] = inputs[0].reshape(
                                len(rows), -1).detach().cpu().clone()
                            captured_gdn["norm_gated"] = output.reshape(
                                len(rows), -1).detach().cpu().clone()

                    def capture_gdn_input_norm(_module, _inputs, output):
                        if capture_enabled[0]:
                            captured_gdn["normalized"] = output[:, -1].detach().cpu().clone()

                    for name in ("qkv", "z", "a", "b"):
                        handles.append(getattr(gdn, "in_proj_" + name).register_forward_hook(
                            capture_gdn_projection(name)))
                    handles.append(gdn.norm.register_forward_hook(capture_gdn_norm))
                    handles.append(gdn.out_proj.register_forward_hook(
                        capture_gdn_projection("output")))
                    handles.append(layer.input_layernorm.register_forward_hook(capture_gdn_input_norm))
                if args.architecture == "qwen36":
                    def capture_norm(_module, _inputs, output, index=index):
                        if capture_enabled[0]:
                            captured_norm[index] = output[:, -1].detach().cpu().clone()
                    def capture_route(_module, _inputs, output, index=index):
                        if capture_enabled[0]:
                            captured_router_logits[index] = output[0].detach().cpu().clone()
                            captured_router_scores[index] = output[1].detach().cpu().clone()
                            captured_route[index] = output[2].detach().cpu().clone()
                    handles.append(layer.post_attention_layernorm.register_forward_hook(capture_norm))
                    handles.append(layer.mlp.gate.register_forward_hook(capture_route))
                    def capture_experts(_module, _inputs, output, index=index):
                        if capture_enabled[0]:
                            captured_experts[index] = output.detach().cpu().clone()
                    def capture_shared(_module, _inputs, output, index=index):
                        if capture_enabled[0]:
                            captured_shared[index] = output.detach().cpu().clone()
                    def capture_shared_gate(_module, _inputs, output, index=index):
                        if capture_enabled[0]:
                            captured_shared_gate[index] = output.detach().cpu().clone()
                    def capture_mlp(_module, _inputs, output, index=index):
                        if capture_enabled[0]:
                            captured_mlp[index] = output[:, -1].detach().cpu().clone()
                    handles.append(layer.mlp.experts.register_forward_hook(capture_experts))
                    handles.append(layer.mlp.shared_expert.register_forward_hook(capture_shared))
                    handles.append(layer.mlp.shared_expert_gate.register_forward_hook(capture_shared_gate))
                    handles.append(layer.mlp.register_forward_hook(capture_mlp))
        with torch.inference_mode():
            cache = None
            for position, column in enumerate(zip(*rows)):
                capture_enabled[0] = (args.capture_layers and args.max_new_tokens == 1
                                      and position == len(rows[0]) - 1)
                ids = torch.tensor(column, dtype=torch.int64, device="cuda")[:, None]
                result = model(input_ids=ids, attention_mask=torch.ones(len(rows), position + 1,
                               dtype=torch.int64, device="cuda"), use_cache=True,
                               logits_to_keep=1, past_key_values=cache)
                cache = result.past_key_values
                capture_enabled[0] = False
            next_ids = result.logits[:, -1].argmax(-1)
            generated = [[] for _ in rows]
            for step in range(args.max_new_tokens):
                for row, token in enumerate(next_ids.tolist()):
                    generated[row].append(token)
                if step + 1 < args.max_new_tokens:
                    position = len(rows[0]) + step
                    capture_enabled[0] = args.capture_layers and step + 2 == args.max_new_tokens
                    result = model(input_ids=next_ids[:, None], attention_mask=torch.ones(
                                   len(rows), position + 1, dtype=torch.int64, device="cuda"),
                                   use_cache=True, logits_to_keep=1, past_key_values=cache)
                    cache = result.past_key_values
                    next_ids = result.logits[:, -1].argmax(-1)
                    capture_enabled[0] = False
        if any(len(row) != args.max_new_tokens for row in generated):
            raise RuntimeError("HF did not generate the requested token count")
        for handle in handles:
            handle.remove()
        args.oracle.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "config_sha256": config_hash, "prompts_sha256": prompt_hash,
            "index_sha256": index_hash, "model_path": str(args.model.resolve()),
            "architecture": args.architecture,
            "device": torch.cuda.get_device_name(0), "torch": torch.__version__,
            "gpu_uuid": gpu_uuid,
            "transformers": transformers.__version__,
            "experts_implementation": "grouped_mm" if args.architecture == "qwen36" else None,
            "batch_size": len(rows), "new_tokens": args.max_new_tokens,
            "generated_ids": generated,
        }
        if args.capture_layers:
            if len(captured) != loader.config.num_hidden_layers:
                raise RuntimeError("HF did not capture every decoder layer")
            layer_path = args.oracle.with_name(args.oracle.stem + "-layers.pt")
            layers = {"hidden": torch.stack([captured[i] for i in range(len(captured))])}
            if args.architecture == "qwen36":
                if any(len(part) != len(captured) for part in
                       (captured_norm, captured_route, captured_router_logits,
                        captured_router_scores,
                        captured_experts, captured_shared, captured_shared_gate,
                        captured_mlp)):
                    raise RuntimeError("HF did not capture every MoE component")
                layers.update({
                    "postnorm": torch.stack([captured_norm[i] for i in range(len(captured))]),
                    "router_logits": torch.stack([captured_router_logits[i] for i in range(len(captured))]),
                    "router_scores": torch.stack([captured_router_scores[i] for i in range(len(captured))]),
                    "route": torch.stack([captured_route[i] for i in range(len(captured))]),
                    "experts": torch.stack([captured_experts[i] for i in range(len(captured))]),
                    "shared": torch.stack([captured_shared[i] for i in range(len(captured))]),
                    "shared_gate": torch.stack([captured_shared_gate[i] for i in range(len(captured))]),
                    "mlp": torch.stack([captured_mlp[i] for i in range(len(captured))]),
                })
            if args.capture_attention_layer is not None:
                required = {"normalized", "qg", "k", "v", "gated", "output", "probabilities"}
                if set(captured_attention) != required:
                    raise RuntimeError("HF did not capture every attention component")
                if len(captured_value_history) != len(rows[0]) + args.max_new_tokens - 1:
                    raise RuntimeError("HF did not capture every attention V cache entry")
                layers["attention_probe_layer"] = torch.tensor(args.capture_attention_layer)
                layers.update({"attention_" + name: value
                               for name, value in captured_attention.items()})
                layers["attention_value_history"] = torch.stack(captured_value_history, dim=1)
                keys = cache.layers[args.capture_attention_layer].keys
                layers["attention_key_history"] = keys.transpose(1, 2).reshape(
                    len(rows), keys.shape[2], -1).detach().cpu().clone()
            if args.capture_gdn_layer is not None:
                required = {"normalized", "qkv", "z", "a", "b", "raw", "norm_gated",
                            "gated", "output"}
                if set(captured_gdn) != required:
                    raise RuntimeError("HF did not capture every GDN component")
                state = cache.layers[args.capture_gdn_layer]
                captured_gdn["state"] = state.recurrent_states[0].detach().cpu().clone()
                captured_gdn["conv_state"] = state.conv_states[0].detach().cpu().clone()
                layers["gdn_probe_layer"] = torch.tensor(args.capture_gdn_layer)
                layers.update({"gdn_" + name: value for name, value in captured_gdn.items()})
            torch.save(layers, layer_path)
            record["layer_oracle_sha256"] = hashlib.sha256(layer_path.read_bytes()).hexdigest()
        args.oracle.write_text(json.dumps(record, indent=2) + "\n")
        return

    if args.compiled_dir is None:
        parser.error("--compiled-dir is required for verify")
    expected = json.loads(args.oracle.read_text())
    for key, value in (("config_sha256", config_hash), ("prompts_sha256", prompt_hash),
                       ("index_sha256", index_hash), ("model_path", str(args.model.resolve())),
                       ("architecture", args.architecture),
                       ("batch_size", len(rows)), ("new_tokens", args.max_new_tokens),
                       ("device", torch.cuda.get_device_name(0)),
                       ("gpu_uuid", gpu_uuid),
                       ("torch", torch.__version__),
                       ("experts_implementation", "grouped_mm" if args.architecture == "qwen36" else None),
                       ("transformers", transformers.__version__)):
        if expected.get(key) != value:
            raise ValueError(f"Oracle {key} does not match this verification run")
    layer_expected = None
    if args.capture_layers:
        layer_path = args.oracle.with_name(args.oracle.stem + "-layers.pt")
        if hashlib.sha256(layer_path.read_bytes()).hexdigest() != expected.get("layer_oracle_sha256"):
            raise ValueError("HF layer oracle hash mismatch")
        layer_expected = torch.load(layer_path, map_location="cpu", weights_only=True)
    from mirage.mpk.models.qwen3_5.builder import Qwen38Builder
    from mirage.mpk.models.qwen3_5.moe_builder import Qwen36MoeBuilder
    from mirage.mpk.models.qwen3_5.offline import create_offline_runtime, offline_generated_ids

    mpk, plan, metadata = create_offline_runtime(loader.config, rows, args.max_new_tokens,
                                                 ignore_eos=True)
    try:
        builder = (Qwen36MoeBuilder if args.architecture == "qwen36" else Qwen38Builder)(
            mpk, kv_plan=plan)
        builder.build_from_model(model_path=args.model)
        if args.reuse_compiled:
            mpk.load_mpk_kernel(output_dir=str(args.compiled_dir))
        else:
            mpk.compile(output_dir=str(args.compiled_dir))
        mpk()
        torch.cuda.synchronize()
        actual = offline_generated_ids(metadata, len(rows[0]), args.max_new_tokens)
        if args.capture_layers:
            layer_actual = {"hidden": torch.stack([
                layer.tensors["decoder_output"].detach().cpu().clone()
                for layer in builder.model.layers])}
            if args.architecture == "qwen36":
                from mirage.mpk.models.qwen3_5.moe_reference import sum_expert_slots

                final_hidden = layer_expected["hidden"][-1].cuda().float()
                norm_weight = builder.model.tensors["norm_weight"].float()
                expected_norm = ((final_hidden * torch.rsqrt(
                    final_hidden.square().mean(-1, keepdim=True) + 1e-6)) *
                    (1.0 + norm_weight)).bfloat16()
                expected_logits = torch.nn.functional.linear(
                    expected_norm, builder.model.tensors["lm_head"])
                layer_expected["final_norm"] = expected_norm.cpu()
                layer_expected["logits"] = expected_logits.cpu()
                layer_actual["final_norm"] = builder.model.tensors["normalized"].detach().cpu().clone()
                layer_actual["logits"] = builder.model.tensors["logits"].detach().cpu().clone()
                if args.final_head_snapshot:
                    args.final_head_snapshot.parent.mkdir(parents=True, exist_ok=True)
                    torch.save({"hf_hidden": layer_expected["hidden"][-1],
                                "mpk_hidden": layer_actual["hidden"][-1],
                                "hf_norm": layer_expected["final_norm"],
                                "mpk_norm": layer_actual["final_norm"],
                                "hf_logits_recomputed": layer_expected["logits"],
                                "mpk_logits": layer_actual["logits"],
                                "norm_weight": norm_weight.cpu(),
                                "steps": metadata["step"].cpu()}, args.final_head_snapshot)
                layer_actual["postnorm"] = torch.stack([
                    layer.tensors["post_normalized"].detach().cpu().clone()
                    for layer in builder.model.layers])
                layer_actual["router_scores"] = torch.stack([
                    layer.tensors["router_scores"].detach().cpu().clone()
                    for layer in builder.model.layers])
                layer_actual["shared"] = torch.stack([
                    layer.tensors["shared_output"].detach().cpu().clone()
                    for layer in builder.model.layers])
                layer_actual["shared_gate"] = torch.stack([
                    layer.tensors["shared_gate"][:, :1].detach().cpu().clone()
                    for layer in builder.model.layers])
                layer_actual["mlp"] = torch.stack([
                    layer.tensors["mlp_output"].detach().cpu().clone()
                    for layer in builder.model.layers])
                routes = []
                experts = []
                for layer in builder.model.layers:
                    routing_gpu = layer.tensors["routing_indices"]
                    scores_gpu = layer.tensors["router_scores"]
                    outputs_gpu = layer.tensors["expert_out"]
                    experts.append(sum_expert_slots(
                        outputs_gpu, scores_gpu, routing_gpu,
                        torch.zeros((len(rows), loader.config.hidden_size),
                                    dtype=torch.bfloat16, device="cuda")).detach().cpu().clone())
                    routing = routing_gpu.detach().cpu()
                    route = torch.full((len(rows), 8), -1, dtype=torch.int64)
                    for expert in range(routing.shape[0]):
                        for row in range(len(rows)):
                            slot = int(routing[expert, row]) - 1
                            if slot >= 0:
                                route[row, slot] = expert
                    routes.append(route)
                layer_actual["route"] = torch.stack(routes)
                layer_actual["experts"] = torch.stack(experts)
                if args.moe_snapshot_layer is not None:
                    index = args.moe_snapshot_layer
                    tensors = builder.model.layers[index].tensors
                    names = ("post_normalized", "router_scores", "routing_indices",
                             "expert_mid", "expert_activated", "expert_out",
                             "shared_output", "mlp_output")
                    args.moe_snapshot.parent.mkdir(parents=True, exist_ok=True)
                    torch.save({
                        "hf": {name: layer_expected[name][index].clone()
                               for name in ("postnorm", "router_scores", "route",
                                            "experts", "shared", "mlp")},
                        "mpk": {name: tensors[name].detach().cpu().clone()
                                for name in names},
                    }, args.moe_snapshot)
            if args.capture_attention_layer is not None:
                mixer = builder.model.layers[args.capture_attention_layer].mixer
                attention_query = mixer.tensors["query"].detach().cpu().clone().reshape(
                    len(rows), loader.config.num_attention_heads, loader.config.head_dim)
                attention_actual = {name: mixer.tensors[name].detach().cpu().clone().reshape(len(rows), -1)
                                    for name in ("normalized", "qg", "k", "v", "gated", "output")}
                attention_actual["probabilities"] = (
                    mixer.tensors["attention_weights"].reshape(
                        len(rows), loader.config.num_attention_heads, -1)[:, :,
                        :len(rows[0]) + args.max_new_tokens - 1].detach().cpu().clone())
                slot = plan.layer_info(args.capture_attention_layer)[1]
                cache = builder.kv_pool[slot].view(torch.bfloat16)
                component = 1024 * loader.config.num_key_value_heads * loader.config.head_dim
                last_v = mixer.tensors["v"].detach().cpu()
                history = []
                key_history = []
                for row in range(len(rows)):
                    matches = [page for page in range(len(rows))
                               if torch.equal(cache[page, component:2 * component].view(
                                   1024, loader.config.num_key_value_heads * loader.config.head_dim
                               )[len(rows[0]) + args.max_new_tokens - 2].detach().cpu(), last_v[row])]
                    if len(matches) != 1:
                        raise RuntimeError("Last V projection did not identify one physical KV page")
                    page = matches[0]
                    history.append(cache[page, component:2 * component].view(
                        1024, loader.config.num_key_value_heads * loader.config.head_dim
                    )[:len(rows[0]) + args.max_new_tokens - 1].detach().cpu().clone())
                    key_history.append(cache[page, :component].view(
                        1024, loader.config.num_key_value_heads * loader.config.head_dim
                    )[:len(rows[0]) + args.max_new_tokens - 1].detach().cpu().clone())
                attention_actual["value_history"] = torch.stack(history)
                attention_actual["key_history"] = torch.stack(key_history)
                torch.save({"query": attention_query,
                            "key_history": attention_actual["key_history"],
                            "probabilities": attention_actual["probabilities"]},
                           args.compiled_dir / "attention-operands.pt")
            if args.capture_gdn_layer is not None:
                mixer = builder.model.layers[args.capture_gdn_layer].mixer
                gdn_actual = {name: mixer.tensors[name].detach().cpu().clone().reshape(len(rows), -1)
                              for name in ("normalized", "qkv", "z", "raw", "gated", "output")}
                ab = mixer.tensors["ab"].detach().cpu().clone()
                heads = loader.config.linear_num_value_heads
                gdn_actual["a"] = ab[:, :heads]
                gdn_actual["b"] = ab[:, heads:2 * heads]
                gdn_actual["norm_gated"] = gdn_actual["gated"]
                gdn_actual["state"] = mixer.tensors["state"].detach().cpu().clone()
                gdn_actual["conv_state"] = mixer.tensors["conv_state"].detach().cpu().clone()
                if args.gdn_snapshot:
                    args.gdn_snapshot.parent.mkdir(parents=True, exist_ok=True)
                    torch.save({"hf": {name: layer_expected["gdn_" + name]
                                       for name in gdn_actual},
                                "mpk": gdn_actual}, args.gdn_snapshot)
    finally:
        mpk.finalize()
    mismatches = [(r, t, expected["generated_ids"][r][t], actual[r][t])
                  for r in range(len(rows)) for t in range(args.max_new_tokens)
                  if expected["generated_ids"][r][t] != actual[r][t]]
    report = {"matches": len(rows) * args.max_new_tokens - len(mismatches),
              "total": len(rows) * args.max_new_tokens,
              "expected_ids": expected["generated_ids"], "actual_ids": actual,
              "mismatches": mismatches}
    if args.capture_layers:
        if layer_actual["hidden"].shape != layer_expected["hidden"].shape:
            raise RuntimeError("HF and MPK layer output shapes differ")
        for name in ("hidden", "postnorm", "router_scores",
                     "experts", "shared",
                     "shared_gate", "mlp"):
            if name not in layer_actual:
                continue
            errors = []
            for layer in range(len(layer_actual[name])):
                difference = layer_actual[name][layer].float() - layer_expected[name][layer].float()
                reference = layer_expected[name][layer].float()
                rel_rms = (difference.square().mean(dim=1).sqrt() /
                           reference.square().mean(dim=1).sqrt().clamp_min(1e-12))
                errors.append({"layer": layer, "per_row_rel_rms": rel_rms.tolist(),
                               "per_row_max_abs": difference.abs().amax(dim=1).tolist()})
            report[name + "_errors"] = errors
        if args.architecture == "qwen36":
            report["final_head_errors"] = {}
            for name in ("final_norm", "logits"):
                actual_head = layer_actual[name].float()
                expected_head = layer_expected[name].float()
                difference = actual_head - expected_head
                report["final_head_errors"][name] = {
                    "per_row_max_abs": difference.abs().amax(dim=1).tolist(),
                    "per_row_mismatch_count": (difference != 0).sum(dim=1).tolist(),
                    "reference_top2": expected_head.topk(2, dim=1).indices.tolist()
                    if name == "logits" else None,
                    "mpk_top2": actual_head.topk(2, dim=1).indices.tolist()
                    if name == "logits" else None,
                }
        if "route" in layer_actual:
            report["route_mismatches"] = [
                {"layer": layer, "row": row,
                 "hf": layer_expected["route"][layer, row].tolist(),
                 "mpk": layer_actual["route"][layer, row].tolist()}
                for layer in range(len(layer_actual["route"]))
                for row in range(len(rows))
                if not torch.equal(layer_expected["route"][layer, row],
                                   layer_actual["route"][layer, row])]
        if args.capture_attention_layer is not None:
            if int(layer_expected["attention_probe_layer"]) != args.capture_attention_layer:
                raise ValueError("HF attention oracle belongs to another layer")
            report["attention_probe"] = {"layer": args.capture_attention_layer}
            for name, actual_tensor in attention_actual.items():
                expected_tensor = layer_expected["attention_" + name]
                if actual_tensor.shape != expected_tensor.shape:
                    raise RuntimeError(f"Attention {name} shapes differ")
                actual_float = actual_tensor.float().reshape(len(rows), -1)
                expected_float = expected_tensor.float().reshape(len(rows), -1)
                difference = actual_float - expected_float
                report["attention_probe"][name] = {
                    "per_row_rel_rms": (difference.square().mean(dim=1).sqrt() /
                                        expected_float.square().mean(dim=1).sqrt().clamp_min(1e-12)).tolist(),
                    "per_row_exact_fraction": (actual_float == expected_float).float().mean(dim=1).tolist(),
                    "first_differences": [
                        {"row": int(row), "flat_index": int(column),
                         "hf": float(expected_float[row, column]),
                         "mpk": float(actual_float[row, column])}
                        for row, column in torch.nonzero(actual_float != expected_float)[:8]],
                }
        if args.capture_gdn_layer is not None:
            if int(layer_expected["gdn_probe_layer"]) != args.capture_gdn_layer:
                raise ValueError("HF GDN oracle belongs to another layer")
            report["gdn_probe"] = {"layer": args.capture_gdn_layer}
            for name, actual_tensor in gdn_actual.items():
                expected_tensor = layer_expected["gdn_" + name]
                if name == "conv_state":
                    expected_tensor = expected_tensor[..., -3:]
                if actual_tensor.numel() != expected_tensor.numel():
                    raise RuntimeError(f"GDN {name} shape differs")
                actual_float = actual_tensor.float().reshape(len(rows), -1)
                expected_float = expected_tensor.float().reshape(len(rows), -1)
                difference = actual_float - expected_float
                report["gdn_probe"][name] = {
                    "per_row_rel_rms": (difference.square().mean(dim=1).sqrt() /
                                        expected_float.square().mean(dim=1).sqrt().clamp_min(1e-12)).tolist(),
                    "per_row_exact_fraction": (actual_float == expected_float).float().mean(dim=1).tolist(),
                    "first_differences": [
                        {"row": int(row), "flat_index": int(column),
                         "hf": float(expected_float[row, column]),
                         "mpk": float(actual_float[row, column])}
                        for row, column in torch.nonzero(actual_float != expected_float)[:8]],
                }
    args.compiled_dir.mkdir(parents=True, exist_ok=True)
    (args.compiled_dir / "acceptance.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"matches": report["matches"], "total": report["total"],
                      "first_mismatches": mismatches[:16]}, indent=2))
    if mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
