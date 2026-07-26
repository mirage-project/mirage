#!/usr/bin/env python3
"""P1 -- dense-bf16 token equivalence probe (M2-I2, v1-architecture.md SS14).

DIAGNOSTIC ONLY after the SS6.2 amendment (bf16-dense is a debugging scaffold, never an M2
acceptance endpoint) -- this probe attributes future MPK integration mismatches to
dense-vs-new-kernel causes, it is NOT a GO/NO-GO for any shipped path.

Loads Qwen/Qwen3.5-35B-A3B-FP8, replaces every FP8Linear module EXCEPT the routed experts
(`mlp.experts`, an FP8Experts instance -- NOT an FP8Linear subclass, so a plain
`isinstance(mod, FP8Linear)` check already excludes it; we additionally assert by name for
defense-in-depth and full transparency) with a plain bf16 nn.Linear carrying the exact
block-dequantized weight. Runs greedy decode (64 new tokens) on the SAME input_ids that
produced the pinned reference_outputs.json (byte-identical starting point, not a
re-tokenization), diffs output token ids, and reports this run's own top1/top2 margin at the
first divergence of any mismatching prompt (reference_outputs.json only stores the scalar
top-1 logit per step, not a full vector, so a true reference-side margin isn't recoverable
without a second full model load; this run's own margin is the correct, honestly-computable
diagnostic for "was this a close call").
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR", "1")

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402


def log(msg: str) -> None:
    print(f"[p1_dense_bf16] {msg}", flush=True)


def load_model_and_tokenizer(model_id: str, revision: str):
    """Same loading dance as accept/reference/generate_reference.py (kept in sync
    deliberately -- both need the CausalLM-first / VLM-wrapper-fallback dance)."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    load_notes = {"attempts": []}
    model = None

    def try_causal_lm():
        from transformers import AutoModelForCausalLM
        return AutoModelForCausalLM.from_pretrained(model_id, revision=revision, dtype="auto", device_map="cuda:0")

    def try_image_text_to_text():
        from transformers import AutoModelForImageTextToText
        return AutoModelForImageTextToText.from_pretrained(model_id, revision=revision, dtype="auto", device_map="cuda:0")

    def try_direct_class():
        from transformers import Qwen3_5MoeForConditionalGeneration
        return Qwen3_5MoeForConditionalGeneration.from_pretrained(model_id, revision=revision, dtype="auto", device_map="cuda:0")

    for name, fn in [("AutoModelForCausalLM", try_causal_lm),
                      ("AutoModelForImageTextToText", try_image_text_to_text),
                      ("Qwen3_5MoeForConditionalGeneration (direct)", try_direct_class)]:
        try:
            log(f"attempting load via {name} ...")
            t0 = time.time()
            model = fn()
            load_notes["attempts"].append({"class": name, "ok": True, "seconds": round(time.time() - t0, 1)})
            load_notes["loaded_via"] = name
            break
        except Exception as e:  # noqa: BLE001
            log(f"  failed via {name}: {type(e).__name__}: {e}")
            load_notes["attempts"].append({"class": name, "ok": False, "error": f"{type(e).__name__}: {e}"})

    if model is None:
        raise RuntimeError("could not load the FP8 checkpoint via any known auto class")
    model.eval()
    return model, tokenizer, load_notes


def dequantize_fp8_linear_weight(mod):
    """W_real ~= W_fp8 * weight_scale_inv, block-expanded (same semantics as p10's
    dequant_bf16 / vllm-graph.md SS3.4)."""
    w = mod.weight.data
    s = mod.weight_scale_inv.data.float()
    n, k = w.shape
    block_size = getattr(mod, "block_size", None) or (n, k)
    bn, bk = block_size
    s_exp = s.repeat_interleave(bn, dim=0)[:n].repeat_interleave(bk, dim=1)[:, :k]
    return (w.float() * s_exp).to(torch.bfloat16)


def patch_dense_fp8_to_bf16(model):
    """Replace every FP8Linear NOT under `.experts` with a bf16 nn.Linear carrying the
    dequantized weight. Returns (patched_names, skipped_expert_related_names, class_inventory)."""
    from transformers.integrations.finegrained_fp8 import FP8Linear

    named = dict(model.named_modules())
    patched, skipped_experts, inventory = [], [], []

    targets = []
    for name, mod in named.items():
        is_fp8_linear = isinstance(mod, FP8Linear)
        name_says_experts = ".experts" in name or name.endswith("experts")
        if is_fp8_linear or name_says_experts:
            inventory.append({"name": name, "class": type(mod).__name__,
                               "is_fp8_linear": is_fp8_linear, "name_says_experts": name_says_experts})
        if is_fp8_linear and name_says_experts:
            skipped_experts.append(name)  # would be redundant (isinstance already false in
            continue                       # practice for FP8Experts) but logged for transparency
        if is_fp8_linear and not name_says_experts:
            targets.append(name)
        elif (not is_fp8_linear) and name_says_experts:
            skipped_experts.append(name)  # the actual routed-expert module(s), left native fp8

    for name in targets:
        mod = named[name]
        if not (hasattr(mod, "weight_scale_inv") and mod.weight.dtype == torch.float8_e4m3fn):
            continue  # defensive: only patch modules that are actually fp8-quantized
        new_w = dequantize_fp8_linear_weight(mod)
        out_f, in_f = new_w.shape
        new_lin = nn.Linear(in_f, out_f, bias=(mod.bias is not None), dtype=torch.bfloat16, device=new_w.device)
        new_lin.weight.data = new_w
        if mod.bias is not None:
            new_lin.bias.data = mod.bias.data.to(torch.bfloat16)

        parent_path, _, attr = name.rpartition(".")
        parent = named[parent_path] if parent_path else model
        setattr(parent, attr, new_lin)
        patched.append(name)

    torch.cuda.empty_cache()
    return patched, skipped_experts, inventory


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen3.5-35B-A3B-FP8")
    ap.add_argument("--revision", default="9d1823d2dee688a6b25e77009dc727688c44936e")
    ap.add_argument("--reference-json", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--out", default=os.path.expanduser("~/mpk-qwen35/probes/fp8_out/p1_dense_bf16_result.json"))
    ap.add_argument("--limit-prompts", type=int, default=None, help="debug: only run the first N prompts")
    args = ap.parse_args()

    import transformers
    log(f"torch {torch.__version__}, transformers {transformers.__version__}")
    assert torch.cuda.is_available()
    log(f"GPU: {torch.cuda.get_device_name(0)}")

    with open(args.reference_json) as f:
        ref = json.load(f)
    ref_results = ref["results"]
    pids = list(ref_results.keys())
    if args.limit_prompts:
        pids = pids[: args.limit_prompts]
    log(f"loaded reference with {len(ref_results)} prompts from {args.reference_json}; running {len(pids)}")

    model, tokenizer, load_notes = load_model_and_tokenizer(args.model, args.revision)
    log(f"model loaded via {load_notes.get('loaded_via')}")

    patched, skipped_experts, inventory = patch_dense_fp8_to_bf16(model)
    log(f"patched {len(patched)} dense FP8Linear modules to bf16; left {len(skipped_experts)} "
        f"expert-related modules native fp8")
    for entry in inventory[:6]:
        log(f"  inventory sample: {entry}")

    eos_ids = model.generation_config.eos_token_id
    if not isinstance(eos_ids, list):
        eos_ids = [eos_ids]

    results = {}
    total_positions = 0
    total_matches = 0
    n_prompts_full_match = 0
    t_start = time.time()
    for pid in pids:
        row = ref_results[pid]
        input_ids = torch.tensor([row["input_ids"]], device=model.device)
        attention_mask = torch.ones_like(input_ids)

        t0 = time.time()
        with torch.no_grad():
            gen = model.generate(
                input_ids=input_ids, attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens, do_sample=False,
                temperature=None, top_p=None, top_k=None, num_beams=1,
                output_logits=True, return_dict_in_generate=True, use_cache=True,
            )
        elapsed = time.time() - t0

        prompt_len = input_ids.shape[1]
        my_output_ids = gen.sequences[0].tolist()[prompt_len:]
        ref_output_ids = row["output_ids"]

        cmp_len = min(len(my_output_ids), len(ref_output_ids))
        n_match = sum(1 for i in range(cmp_len) if my_output_ids[i] == ref_output_ids[i])
        full_match = (my_output_ids == ref_output_ids)
        total_positions += cmp_len
        total_matches += n_match
        n_prompts_full_match += int(full_match)

        divergence = None
        if not full_match:
            first_div = next(i for i in range(cmp_len) if my_output_ids[i] != ref_output_ids[i])
            step_logits = gen.logits[first_div][0].float()
            top2 = torch.topk(step_logits, k=2)
            my_top1_id, my_top2_id = int(top2.indices[0]), int(top2.indices[1])
            my_top1_logit, my_top2_logit = float(top2.values[0]), float(top2.values[1])
            divergence = {
                "position": first_div,
                "ref_token_id": ref_output_ids[first_div],
                "my_token_id": my_output_ids[first_div],
                "my_top1_id": my_top1_id, "my_top1_logit": my_top1_logit,
                "my_top2_id": my_top2_id, "my_top2_logit": my_top2_logit,
                "my_own_margin_top1_minus_top2": my_top1_logit - my_top2_logit,
                "ref_output_matches_my_top2": (ref_output_ids[first_div] == my_top2_id),
                "reference_top1_logit_at_step": row["top1_logit_per_step"][first_div]
                    if first_div < len(row["top1_logit_per_step"]) else None,
            }

        results[pid] = {
            "prompt_len": prompt_len, "num_generated_mine": len(my_output_ids),
            "num_generated_ref": len(ref_output_ids), "n_match": n_match, "cmp_len": cmp_len,
            "full_match": full_match, "divergence": divergence, "elapsed_seconds": round(elapsed, 3),
        }
        log(f"{pid}: full_match={full_match} n_match={n_match}/{cmp_len} ({elapsed:.1f}s)"
            + (f"  DIVERGE@{divergence['position']}: ref={divergence['ref_token_id']} mine={divergence['my_token_id']} "
               f"my_margin={divergence['my_own_margin_top1_minus_top2']:.4f}" if divergence else ""))

    total_elapsed = time.time() - t_start
    summary = {
        "verdict_informational": "640/640-style bf16-dense v1 GO (all prompts full_match)" if n_prompts_full_match == len(pids)
                                  else "MISMATCH -- diagnostic only per SS6.2 amendment, not a GO/NO-GO gate",
        "n_prompts": len(pids),
        "n_prompts_full_match": n_prompts_full_match,
        "total_positions_compared": total_positions,
        "total_positions_matched": total_matches,
        "n_dense_modules_patched": len(patched),
        "n_expert_related_modules_left_native_fp8": len(skipped_experts),
        "patched_module_names": patched,
        "expert_related_module_names": skipped_experts,
        "load_notes": load_notes,
        "model": args.model, "revision": args.revision,
        "reference_json": args.reference_json,
        "transformers_disable_deepgemm_linear": os.environ.get("TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR"),
        "versions": {"python": sys.version, "torch": torch.__version__, "transformers": transformers.__version__},
        "gpu": torch.cuda.get_device_name(0),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "total_elapsed_seconds": round(total_elapsed, 1),
    }
    log(f"SUMMARY: {summary['n_prompts_full_match']}/{summary['n_prompts']} prompts full-match, "
        f"{total_matches}/{total_positions} positions match -- {summary['verdict_informational']}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "module_inventory": inventory, "results": results}, f, indent=2)
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
