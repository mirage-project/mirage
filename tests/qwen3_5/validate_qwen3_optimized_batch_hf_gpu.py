"""Independent HF/MPK correctness gate for Qwen3.8 dense and Qwen3.6 MoE B8/B16.

Run ``freeze`` and ``verify`` in separate processes so each owns the B200.
The verifier forces the frozen HF token history, checks every greedy token,
and compares selected full-model checkpoints plus all final GDN states.
"""

import argparse
import hashlib
import inspect
import json
from pathlib import Path


SCENARIOS = ((3, 4), (65, 8), (128, 64), (1020, 4))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metric(actual, expected):
    import torch

    a, b = actual.float(), expected.float()
    if a.shape != b.shape:
        raise ValueError(f"Tensor shapes differ: {tuple(a.shape)} versus {tuple(b.shape)}")
    difference = a - b
    reference_rms = b.square().mean().sqrt().clamp_min(1e-12)
    relative_rms = float(difference.square().mean().sqrt() / reference_rms)
    max_abs = float(difference.abs().max())
    bound = .05 * float(b.abs().max()) + 1e-5
    return {"exact": bool(torch.equal(a, b)), "relative_rms": relative_rms,
            "max_abs": max_abs, "max_abs_bound": bound,
            "accepted": bool(torch.isfinite(a).all()) and relative_rms <= .02 and max_abs <= bound}


def checkpoints(new_tokens):
    return sorted({0, 1, 7, 31, new_tokens - 1} & set(range(new_tokens)))


def prompt_rows(tokenizer, length, batch):
    if length == 3:
        return [[101 + row, 231 + row, 355 + row] for row in range(batch)]
    from benchmark_qwen38_gpu import prompts
    return prompts(tokenizer, length, batch)


def metadata(model_path, architecture, torch, transformers, hf_module):
    result = {"model_path": str(model_path.resolve()),
            "config_sha256": sha256(model_path / "config.json"),
            "index_sha256": sha256(model_path / "model.safetensors.index.json"),
            "torch": torch.__version__, "transformers": transformers.__version__,
            "hf_source_sha256": sha256(inspect.getfile(hf_module)),
            "gpu_uuid": str(torch.cuda.get_device_properties(0).uuid)}
    if architecture == "qwen36":
        result["architecture"] = architecture
        result["prefill_mode"] = "tokenwise_hf_and_mpk"
    return result


def freeze(args, torch, transformers, hf_module):
    from transformers import AutoTokenizer
    if args.architecture == "qwen36":
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
            Qwen3_5MoeForConditionalGeneration as HFModel)
        model_options = {"experts_implementation": "grouped_mm"}
    else:
        from transformers import Qwen3_5ForConditionalGeneration as HFModel
        model_options = {}

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = HFModel.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="eager",
        local_files_only=True, device_map="cuda", **model_options).eval()
    captured_layers, captured_raw, captured_gated, captured_probe = {}, {}, {}, {}
    capture = {"layers": False, "gdn": False, "probe": False}
    handles = []
    language_layers = model.model.language_model.layers
    layer_types = model.config.text_config.layer_types
    if args.probe_gdn_layer is not None:
        mixer = language_layers[args.probe_gdn_layer].linear_attn
        causal_conv_update = mixer.causal_conv1d_update
        def save_conv(*call_args, **call_kwargs):
            output = causal_conv_update(*call_args, **call_kwargs)
            if capture["probe"]:
                captured_probe["conv"] = output.detach().cpu().clone()
            return output
        mixer.causal_conv1d_update = save_conv
    for index, layer in enumerate(language_layers):
        def save_layer(_module, _inputs, output, index=index):
            if capture["layers"]:
                value = output[0] if isinstance(output, tuple) else output
                captured_layers[index] = value[:, -1].detach().cpu().clone()
        handles.append(layer.register_forward_hook(save_layer))
        if index == args.probe_gdn_layer:
            for name, module in (("normalized", layer.input_layernorm),
                                 ("qkv", layer.linear_attn.in_proj_qkv),
                                 ("z", layer.linear_attn.in_proj_z),
                                 ("a", layer.linear_attn.in_proj_a),
                                 ("b", layer.linear_attn.in_proj_b)):
                def save_probe(_module, _inputs, output, name=name):
                    if capture["probe"]:
                        captured_probe[name] = output[:, -1].detach().cpu().clone()
                handles.append(module.register_forward_hook(save_probe))
            def save_router(_module, _inputs, output):
                if capture["probe"]:
                    logits, scores, indices = output
                    captured_probe["hf_router_logits"] = logits.detach().cpu().clone().reshape(batch, -1)
                    captured_probe["hf_router_scores"] = scores.detach().cpu().clone().reshape(batch, -1)
                    captured_probe["hf_router_indices"] = indices.detach().cpu().clone().reshape(batch, -1)
            handles.append(layer.mlp.gate.register_forward_hook(save_router))
            for name, module in (("hf_mlp_output", layer.mlp),
                                 ("hf_experts_output", layer.mlp.experts),
                                 ("hf_shared_output", layer.mlp.shared_expert),
                                 ("hf_shared_gate", layer.mlp.shared_expert_gate)):
                def save_moe(_module, _inputs, output, name=name):
                    if capture["probe"]:
                        captured_probe[name] = output.detach().cpu().clone().reshape(batch, -1)
                handles.append(module.register_forward_hook(save_moe))
        if layer_types[index] == "linear_attention":
            def save_gdn(_module, inputs, output, index=index):
                if capture["gdn"]:
                    width = language_layers[index].linear_attn.value_dim
                    captured_raw[index] = inputs[0].reshape(batch, -1, width)[:, -1].detach().cpu().clone()
                    captured_gated[index] = output.reshape(batch, -1, width)[:, -1].detach().cpu().clone()
            handles.append(layer.linear_attn.norm.register_forward_hook(save_gdn))

    oracle = {"schema": "qwen3_optimized_batch_hf_oracle_v1",
              "metadata": metadata(args.model, args.architecture, torch, transformers, hf_module),
              "cases": []}
    for batch in args.batch_sizes:
        for length, new_tokens in args.scenarios:
            rows = prompt_rows(tokenizer, length, batch)
            model.model.rope_deltas = None
            cache = None
            history = [[] for _ in rows]
            points = list(range(new_tokens)) if args.all_checkpoints else checkpoints(new_tokens)
            snapshots = {}
            probe_state_history = []
            prefill_hidden, prefill_probe = {}, {}
            capture["layers"] = False
            capture["gdn"] = False
            capture["probe"] = False
            if args.architecture == "qwen36":
                # The MoE adapter has tokenwise MPK prefill. HF must enter the
                # same recurrent path before comparing decode positions.
                for position in range(length - 1):
                    capture["layers"] = (args.probe_prefill_position is not None and
                                         position in (args.probe_prefill_position - 1,
                                                      args.probe_prefill_position))
                    capture["probe"] = position == args.probe_prefill_position
                    capture["gdn"] = capture["probe"]
                    if capture["layers"]:
                        captured_layers.clear()
                    if capture["probe"]:
                        captured_probe.clear()
                        captured_raw.clear()
                        captured_gated.clear()
                    prefix = torch.tensor([[row[position]] for row in rows],
                                          dtype=torch.long, device="cuda")
                    output = model(input_ids=prefix,
                                   attention_mask=torch.ones(batch, position + 1,
                                                             dtype=torch.long, device="cuda"),
                                   past_key_values=cache, use_cache=True, logits_to_keep=1)
                    cache = output.past_key_values
                    if capture["layers"]:
                        prefill_hidden[position] = torch.stack(
                            [captured_layers[i] for i in range(len(language_layers))])
                    if capture["probe"]:
                        prefill_probe = {**captured_probe,
                                         "raw": captured_raw[args.probe_gdn_layer],
                                         "gated": captured_gated[args.probe_gdn_layer]}
                    if args.probe_gdn_layer is not None:
                        probe_state_history.append(
                            cache.layers[args.probe_gdn_layer].recurrent_states[0].cpu().clone())
                input_ids = torch.tensor([[row[-1]] for row in rows],
                                         dtype=torch.long, device="cuda")
            else:
                input_ids = torch.tensor(rows, dtype=torch.long, device="cuda")
            for step in range(new_tokens):
                capture["layers"] = step in points
                capture["gdn"] = step == new_tokens - 1
                capture["probe"] = step == 0 and args.probe_gdn_layer is not None
                if capture["probe"]:
                    captured_probe.clear()
                    probe_pre_state = cache.layers[args.probe_gdn_layer].recurrent_states[0].cpu().clone()
                captured_layers.clear()
                if capture["gdn"]:
                    captured_raw.clear()
                    captured_gated.clear()
                output = model(input_ids=input_ids,
                               attention_mask=torch.ones(batch, length + step,
                                                         dtype=torch.long, device="cuda"),
                               past_key_values=cache, use_cache=True, logits_to_keep=1)
                cache = output.past_key_values
                logits = output.logits[:, -1]
                next_ids = logits.argmax(-1)
                for row, token in zip(history, next_ids.tolist()):
                    row.append(int(token))
                if step in points:
                    if len(captured_layers) != len(language_layers):
                        raise RuntimeError("HF did not capture every decoder layer")
                    snapshots[step] = {"logits": logits.cpu().clone(),
                                       "hidden": torch.stack([captured_layers[i]
                                                              for i in range(len(language_layers))])}
                    if capture["probe"]:
                        snapshots[step]["probe"] = captured_probe.copy()
                        snapshots[step]["probe_pre_state"] = probe_pre_state
                input_ids = next_ids[:, None]
            gdn_layers = [i for i, kind in enumerate(layer_types) if kind == "linear_attention"]
            if set(captured_raw) != set(gdn_layers) or set(captured_gated) != set(gdn_layers):
                raise RuntimeError("HF did not capture every GDN layer")
            final_gdn = {i: {"raw": captured_raw[i], "gated": captured_gated[i],
                             "state": cache.layers[i].recurrent_states[0].cpu().clone(),
                             "conv_state": cache.layers[i].conv_states[0][..., -3:].cpu().clone()}
                         for i in gdn_layers}
            oracle["cases"].append({"batch": batch, "length": length,
                                     "new_tokens": new_tokens, "rows": rows,
                                     "history": history, "snapshots": snapshots,
                                     "probe_state_history": probe_state_history,
                                     "prefill_hidden": prefill_hidden,
                                     "prefill_probe": prefill_probe,
                                     "final_gdn": final_gdn})
            print(f"FROZEN B{batch} L{length} G{new_tokens}: {len(snapshots)} full-layer checkpoints",
                  flush=True)
            del cache, output
    for handle in handles:
        handle.remove()
    args.oracle.parent.mkdir(parents=True, exist_ok=True)
    torch.save(oracle, args.oracle)
    print(f"HF oracle saved: {args.oracle}", flush=True)


def verify(args, torch, transformers, hf_module):
    from mirage.mpk.models.qwen3_5.model import Qwen35Model, Qwen36MoeModel

    # torch.__version__ is a TorchVersion object in this environment. The
    # oracle is produced by this script on the same host; allow that one type.
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        oracle = torch.load(args.oracle, map_location="cpu", weights_only=True)
    actual_metadata = metadata(args.model, args.architecture, torch, transformers, hf_module)
    # A checkpoint may be presented through hard links rather than HF cache
    # symlinks. Its content hashes, source, software, and GPU must still match.
    if any(actual_metadata.get(key) != value for key, value in oracle["metadata"].items()
           if key != "model_path"):
        raise ValueError("HF oracle checkpoint, environment, or GPU differs")
    report = {"schema": "qwen3_optimized_batch_hf_validation_v1",
              "architecture": args.architecture,
              "oracle_sha256": sha256(args.oracle), "metadata": oracle["metadata"],
              "loaded_model_path": str(args.model.resolve()),
              "thresholds": {"relative_rms": .02, "reference_peak_fraction": .05,
                             "absolute_floor": 1e-5}, "cases": [], "passed": False}
    model = None
    for batch in args.batch_sizes:
        if args.architecture == "qwen36":
            model = Qwen36MoeModel.from_pretrained(args.model, weight_policy="hf_bf16",
                                                   batch_size=batch)
        else:
            model = Qwen35Model.from_pretrained(args.model, weight_policy="hf_bf16",
                                                 attn_implementation="eager", batch_size=batch)
        compiled = (args.compiled_b16 if batch == 16 and args.compiled_b16 is not None
                    else args.compiled_root / f"b{batch}" / "kernel")
        if args.compile:
            compiled.parent.mkdir(parents=True, exist_ok=True)
            model.compile(compiled)
        else:
            model.pk.load_mpk_kernel(str(compiled))
            model._compiled = True
        try:
            for case in (c for c in oracle["cases"] if c["batch"] == batch and
                         (c["length"], c["new_tokens"]) in args.scenarios):
                model.reset()
                probed = args.probe_gdn_layer is not None and "probe" in case["snapshots"].get(0, {})
                if probed:
                    state_history = case.get("probe_state_history", [])
                    prefill_checks = []
                    if state_history:
                        model.prefill([[row[0]] for row in case["rows"]], mode="stream")
                        for position, reference_state in enumerate(state_history):
                            if position:
                                model.step([row[position] for row in case["rows"]])
                            actual_state = model.layers[args.probe_gdn_layer].mixer.tensors["state"].cpu()
                            prefill_checks.append({"step": position, "tensor": "probe_prefill_state",
                                                   "layer": args.probe_gdn_layer,
                                                   **metric(actual_state.reshape(reference_state.shape),
                                                            reference_state)})
                            if position in case.get("prefill_hidden", {}):
                                hidden = case["prefill_hidden"][position]
                                for index, layer in enumerate(model.layers):
                                    prefill_checks.append({"step": position, "tensor": "probe_prefill_hidden",
                                                           "layer": index,
                                                           **metric(layer.tensors["decoder_output"].cpu(),
                                                                    hidden[index])})
                            if position == args.probe_prefill_position:
                                probe = case["prefill_probe"]
                                tensors = model.layers[args.probe_gdn_layer].mixer.tensors
                                for name in ("normalized", "qkv", "z", "conv", "raw", "gated"):
                                    reference = probe[name]
                                    prefill_checks.append({"step": position,
                                                           "tensor": "probe_prefill_" + name,
                                                           "layer": args.probe_gdn_layer,
                                                           **metric(tensors[name].cpu().reshape(reference.shape),
                                                                    reference)})
                                reference = torch.cat((probe["a"], probe["b"]), dim=-1)
                                actual = tensors["ab"].cpu().reshape(batch, -1)[:, :reference.shape[-1]]
                                prefill_checks.append({"step": position, "tensor": "probe_prefill_ab",
                                                       "layer": args.probe_gdn_layer,
                                                       **metric(actual, reference)})
                                import torch.nn.functional as F
                                expected_output = F.linear(
                                    probe["gated"].cuda().reshape(batch, 1, -1),
                                    tensors["out_weight"]).reshape(batch, -1).cpu()
                                prefill_checks.append({"step": position,
                                                       "tensor": "probe_prefill_mixer_output",
                                                       "layer": args.probe_gdn_layer,
                                                       **metric(tensors["output"].cpu(), expected_output)})
                                previous = case["prefill_hidden"][position][args.probe_gdn_layer - 1]
                                expected_residual = (previous.cuda() + expected_output.cuda()).cpu()
                                prefill_checks.append({"step": position,
                                                       "tensor": "probe_prefill_residual",
                                                       "layer": args.probe_gdn_layer,
                                                       **metric(model.layers[args.probe_gdn_layer].tensors["residual"].cpu(),
                                                                expected_residual)})
                                from mirage.mpk.models.qwen3_5.reference import rms_norm
                                layer_tensors = model.layers[args.probe_gdn_layer].tensors
                                expected_post = rms_norm(expected_residual.cuda(),
                                                         layer_tensors["post_norm_weight"],
                                                         model.config.rms_norm_eps).cpu()
                                prefill_checks.append({"step": position,
                                                       "tensor": "probe_prefill_post_normalized",
                                                       "layer": args.probe_gdn_layer,
                                                       **metric(layer_tensors["post_normalized"].cpu(),
                                                                expected_post)})
                                expected_router = F.linear(expected_post.cuda().reshape(batch, 1, -1),
                                                           layer_tensors["router_weight"]).reshape(batch, -1).cpu()
                                prefill_checks.append({"step": position,
                                                       "tensor": "probe_prefill_recomputed_router_logits",
                                                       "layer": args.probe_gdn_layer,
                                                       **metric(expected_router,
                                                                probe["hf_router_logits"])})
                                if "hf_router_indices" in probe:
                                    routed = layer_tensors["routing_indices"].cpu()
                                    selected = torch.full_like(probe["hf_router_indices"], -1)
                                    for expert in range(routed.shape[0]):
                                        for row in range(batch):
                                            slot = int(routed[expert, row]) - 1
                                            if 0 <= slot < selected.shape[1]:
                                                selected[row, slot] = expert
                                    routing_differences = (selected != probe["hf_router_indices"]).nonzero()
                                    prefill_checks.append({"step": position,
                                                           "tensor": "probe_prefill_router_indices",
                                                           "layer": args.probe_gdn_layer,
                                                           "different": len(routing_differences),
                                                           "mismatch_slots": [
                                                               {"row": int(row), "slot": int(slot),
                                                                "mpk": int(selected[row, slot]),
                                                                "hf": int(probe["hf_router_indices"][row, slot])}
                                                               for row, slot in routing_differences[:32]],
                                                           **metric(selected, probe["hf_router_indices"])})
                                    actual_scores = layer_tensors["router_scores"].cpu()
                                    prefill_checks.append({"step": position,
                                                           "tensor": "probe_prefill_router_scores",
                                                           "layer": args.probe_gdn_layer,
                                                           "different": int((actual_scores != probe["hf_router_scores"]).sum()),
                                                           **metric(actual_scores, probe["hf_router_scores"])})
                                    expected_expert = probe["hf_experts_output"]
                                    actual_expert = (layer_tensors["expert_out"].cpu() *
                                                     actual_scores.to(torch.bfloat16).unsqueeze(-1)).sum(1)
                                    prefill_checks.append({"step": position,
                                                           "tensor": "probe_prefill_hf_experts_output",
                                                           "layer": args.probe_gdn_layer,
                                                           **metric(actual_expert, expected_expert)})
                                    for name, tensor_name in (("hf_mlp_output", "mlp_output"),
                                                              ("hf_shared_output", "shared_output"),
                                                              ("hf_shared_gate", "shared_gate")):
                                        reference = probe[name]
                                        actual = layer_tensors[tensor_name].cpu().reshape(batch, -1)
                                        actual = actual[:, :reference.shape[-1]]
                                        prefill_checks.append({"step": position,
                                                               "tensor": "probe_prefill_" + name,
                                                               "layer": args.probe_gdn_layer,
                                                               **metric(actual, reference)})
                    else:
                        model.prefill([row[:-1] for row in case["rows"]],
                                      mode="stream" if args.architecture == "qwen36" else "block")
                    reference = case["snapshots"][0]["probe_pre_state"]
                    actual = model.layers[args.probe_gdn_layer].mixer.tensors["state"].cpu()
                    pre_state_check = {"step": 0, "tensor": "probe_pre_state",
                                       "layer": args.probe_gdn_layer,
                                       **metric(actual.reshape(reference.shape), reference)}
                    model.step([row[-1] for row in case["rows"]])
                else:
                    if args.architecture == "qwen36" and case["length"] >= 512:
                        for position, column in enumerate(zip(*case["rows"])):
                            model.step(column)
                            if position % 64 == 63 or position >= 128:
                                print(f"MPK prefill position {position}", flush=True)
                    else:
                        model.prefill(case["rows"],
                                      mode="stream" if args.architecture == "qwen36" else "block")
                length, new_tokens = case["length"], case["new_tokens"]
                checks = []
                if probed:
                    checks.append(pre_state_check)
                    checks.extend(prefill_checks)
                mismatches = []
                for step in range(new_tokens):
                    if step:
                        frozen_previous = [row[step - 1] for row in case["history"]]
                        model.step(frozen_previous)
                    actual_tokens = [int(t) for t in model.tensors["next_token"].tolist()]
                    expected_tokens = [row[step] for row in case["history"]]
                    if actual_tokens != expected_tokens:
                        mismatches.append({"step": step, "rows": [i for i, (a, b) in
                                                       enumerate(zip(actual_tokens, expected_tokens)) if a != b]})
                    if step in case["snapshots"]:
                        expected = case["snapshots"][step]
                        if "probe" in expected:
                            probe = expected["probe"]
                            tensors = model.layers[args.probe_gdn_layer].mixer.tensors
                            for name in ("normalized", "qkv", "z", "conv"):
                                reference = probe[name]
                                checks.append({"step": step, "tensor": "probe_" + name,
                                               "layer": args.probe_gdn_layer,
                                               **metric(tensors[name].cpu().reshape(reference.shape),
                                                        reference)})
                            reference = torch.cat((probe["a"], probe["b"]), dim=-1)
                            actual = tensors["ab"].cpu().reshape(batch, -1)[:, :reference.shape[-1]]
                            checks.append({"step": step, "tensor": "probe_ab",
                                           "layer": args.probe_gdn_layer,
                                           **metric(actual, reference)})
                        checks.append({"step": step, "tensor": "logits",
                                       **metric(model.tensors["logits"].cpu(), expected["logits"])})
                        for index, layer in enumerate(model.layers):
                            checks.append({"step": step, "tensor": "hidden", "layer": index,
                                           **metric(layer.tensors["decoder_output"].cpu(),
                                                    expected["hidden"][index])})
                for index, expected in case["final_gdn"].items():
                    tensors = model.layers[index].mixer.tensors
                    for name, reference in expected.items():
                        # Dense block prefill uses Torch and does not populate
                        # the MPK decode-only raw/gated scratch buffers.
                        if args.architecture == "qwen38" and new_tokens == 1 and name in ("raw", "gated"):
                            continue
                        checks.append({"step": new_tokens - 1, "tensor": name, "layer": index,
                                       **metric(tensors[name].cpu().reshape(reference.shape), reference)})
                passed = not mismatches and all(c["accepted"] for c in checks)
                result = {"batch": batch, "length": length, "new_tokens": new_tokens,
                          "checkpoint_steps": sorted(case["snapshots"]),
                          "checks": checks, "token_mismatches": mismatches, "passed": passed}
                report["cases"].append(result)
                args.report.parent.mkdir(parents=True, exist_ok=True)
                args.report.write_text(json.dumps(report, indent=2) + "\n")
                print(f"B{batch} L{length} G{new_tokens}: {sum(c['accepted'] for c in checks)}/"
                      f"{len(checks)} numerical checks, {len(mismatches)} token mismatch steps",
                      flush=True)
                if not passed and args.stop_on_failure:
                    raise RuntimeError("HF numerical or token validation failed")
        finally:
            model.close()
            model = None
            import gc
            gc.collect()
            torch.cuda.empty_cache()
    report["passed"] = (all(c["passed"] for c in report["cases"]) and
                        len(report["cases"]) == len(args.batch_sizes) * len(args.scenarios))
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise RuntimeError("HF numerical or token validation failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("freeze", "verify"), required=True)
    parser.add_argument("--architecture", choices=("qwen38", "qwen36"), default="qwen38")
    parser.add_argument("--batch-sizes", type=int, nargs="+", choices=(8, 16), default=(8, 16))
    parser.add_argument("--scenarios", nargs="+", metavar="PROMPT:GENERATED",
                        help="Select focused cases; default is 3:4 65:8 128:64 1020:4")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--compiled-root", type=Path)
    parser.add_argument("--compiled-b16", type=Path,
                        help="Override B16 compiled graph, for baseline attribution")
    parser.add_argument("--compile", action="store_true",
                        help="Compile each graph in --compiled-root before validation")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--stop-on-failure", action="store_true")
    parser.add_argument("--all-checkpoints", action="store_true",
                        help="Compare decoder outputs and logits at every generated step")
    parser.add_argument("--probe-gdn-layer", type=int,
                        help="Capture one GDN layer's projected inputs at the first decode step")
    parser.add_argument("--probe-prefill-position", type=int,
                        help="Capture the selected GDN layer and hidden outputs at a tokenwise prefill position")
    args = parser.parse_args()
    if args.scenarios is None:
        args.scenarios = SCENARIOS
    else:
        try:
            args.scenarios = tuple(tuple(map(int, item.split(":"))) for item in args.scenarios)
        except ValueError:
            parser.error("Each scenario must have the form PROMPT:GENERATED")
        if any(len(item) != 2 or item[0] < 1 or item[1] < 1 or
               item[0] + item[1] - 1 > 1024 for item in args.scenarios):
            parser.error("Scenarios require positive lengths within the 1024-token capacity")
    if args.phase == "verify" and (args.compiled_root is None or args.report is None):
        parser.error("verify requires --compiled-root and --report")
    if args.probe_prefill_position is not None and (
            args.architecture != "qwen36" or args.probe_gdn_layer is None or
            not 0 <= args.probe_prefill_position < min(length for length, _ in args.scenarios) - 1):
        parser.error("prefill probe requires MoE, a GDN layer, and a valid prefix position")

    import torch
    import transformers
    if args.architecture == "qwen36":
        import transformers.models.qwen3_5_moe.modeling_qwen3_5_moe as hf_module
    else:
        import transformers.models.qwen3_5.modeling_qwen3_5 as hf_module

    if torch.cuda.device_count() != 1 or torch.cuda.get_device_capability() != (10, 0):
        raise RuntimeError("Expose exactly one B200 GPU")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(4)
    with torch.inference_mode():
        if args.phase == "freeze":
            freeze(args, torch, transformers, hf_module)
        else:
            verify(args, torch, transformers, hf_module)


if __name__ == "__main__":
    main()
