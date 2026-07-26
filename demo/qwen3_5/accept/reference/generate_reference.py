#!/usr/bin/env python3
"""Self-contained HF `transformers` reference generator for Qwen/Qwen3.5-35B-A3B-FP8.

Produces the AC-3 correctness reference for the mpk-qwen3.5 project: greedy
(do_sample=False) decode of 64 new tokens per pinned prompt, using the model's
own chat template, running the FP8 checkpoint as-shipped (no re-quantization).

Regeneration:
    python generate_reference.py \
        --model-id Qwen/Qwen3.5-35B-A3B-FP8 \
        --revision 9d1823d2dee688a6b25e77009dc727688c44936e \
        --prompts-file <repo>/.pm/eval/prompts.jsonl \
        --output-dir <repo>/workspace/demo/qwen3_5/accept/reference \
        --max-new-tokens 64

Requires a single free CUDA GPU (pin with CUDA_VISIBLE_DEVICES before running)
and HF_HOME pointed at a cache that already has (or can download) the
checkpoint. See README.md in this directory for exact versions and the exact
command used to produce reference_outputs.json.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

# On this project's B200 (SM100), transformers' preferred DeepGEMM fp8 GEMM backend
# crashes on the GDN in_proj_qkv linear ("Assertion error ... sfb_dtype == torch::kFloat
# or sfb_dtype == torch::kInt"). transformers' own source documents a known, still-
# unexplained DeepGEMM-vs-Triton interaction on B200 and ships this exact env var to
# force the Triton finegrained-fp8 backend instead (still genuine native FP8 execution,
# just the other of the two backends - see README.md "FP8 execution path"). Set as a
# default (not a hard override) so this script reproduces out of the box on B200;
# export it to "0" yourself first if you want to test DeepGEMM on other hardware.
os.environ.setdefault("TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR", "1")

import torch


def log(msg: str) -> None:
    print(f"[generate_reference] {msg}", flush=True)


def load_prompts(path: Path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_model_and_tokenizer(model_id: str, revision: str):
    """Load tokenizer + model, trying the CausalLM auto class first and
    falling back to the conditional-generation (VLM-wrapper) auto class.
    Returns (model, tokenizer, load_notes: dict) — load_notes documents which
    path succeeded, for the README.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

    load_notes = {"attempts": []}
    model = None

    # Qwen3.5-35B-A3B-FP8 ships as a *ForConditionalGeneration (VLM wrapper)
    # architecture even though we only exercise the text-only path (AC-2).
    # Try the plain CausalLM auto class first (some wrapper archs dual
    # register); fall back to the ImageTextToText / direct class import.
    def try_causal_lm():
        from transformers import AutoModelForCausalLM

        return AutoModelForCausalLM.from_pretrained(
            model_id, revision=revision, dtype="auto", device_map="cuda:0"
        )

    def try_image_text_to_text():
        from transformers import AutoModelForImageTextToText

        return AutoModelForImageTextToText.from_pretrained(
            model_id, revision=revision, dtype="auto", device_map="cuda:0"
        )

    def try_direct_class():
        from transformers import Qwen3_5MoeForConditionalGeneration

        return Qwen3_5MoeForConditionalGeneration.from_pretrained(
            model_id, revision=revision, dtype="auto", device_map="cuda:0"
        )

    for name, fn in [
        ("AutoModelForCausalLM", try_causal_lm),
        ("AutoModelForImageTextToText", try_image_text_to_text),
        ("Qwen3_5MoeForConditionalGeneration (direct)", try_direct_class),
    ]:
        try:
            log(f"attempting load via {name} ...")
            t0 = time.time()
            model = fn()
            load_notes["attempts"].append(
                {"class": name, "ok": True, "seconds": round(time.time() - t0, 1)}
            )
            load_notes["loaded_via"] = name
            break
        except Exception as e:  # noqa: BLE001 - we want to record and try the next path
            log(f"  failed via {name}: {type(e).__name__}: {e}")
            load_notes["attempts"].append(
                {"class": name, "ok": False, "error": f"{type(e).__name__}: {e}"}
            )

    if model is None:
        raise RuntimeError(
            "transformers could not load the FP8 checkpoint via any known auto "
            "class - see load_notes['attempts'] for the per-class errors. This "
            "is the AC-3-at-risk STOP condition (M1-I5 step 7)."
        )

    model.eval()
    return model, tokenizer, load_notes


def introspect_fp8_execution(model) -> dict:
    """Best-effort, defensive introspection of whether FP8 weights execute via
    native fp8 GEMM kernels or a dequant-to-bf16 fallback. Never raises -
    records whatever it can find.
    """
    info = {"quantizer_class": None, "quantization_config": None, "sampled_modules": []}
    try:
        hq = getattr(model, "hf_quantizer", None)
        if hq is not None:
            info["quantizer_class"] = type(hq).__name__
            qc = getattr(hq, "quantization_config", None)
            if qc is not None:
                try:
                    info["quantization_config"] = qc.to_dict()
                except Exception:
                    info["quantization_config"] = str(qc)
    except Exception as e:  # noqa: BLE001
        info["quantizer_introspection_error"] = str(e)

    # Sample a handful of named modules that SHOULD be fp8-quantized per
    # modules_to_not_convert (any full-attention q_proj / MoE expert proj)
    # and record the concrete class + dtype(s) actually holding the weights.
    wanted_substrings = [
        "layers.3.self_attn.q_proj",
        "layers.3.mlp.experts",
        "layers.0.mlp.shared_expert",
        "layers.3.self_attn.o_proj",
    ]
    try:
        named = dict(model.named_modules())
        for substr in wanted_substrings:
            hit = None
            for name, mod in named.items():
                if substr in name:
                    hit = (name, mod)
                    break
            if hit is None:
                continue
            name, mod = hit
            entry = {"module": name, "class": type(mod).__name__, "params": []}
            for pname, p in mod.named_parameters(recurse=False):
                entry["params"].append(
                    {"param": pname, "dtype": str(p.dtype), "shape": list(p.shape)}
                )
            # Common fp8 quantizer attribute names across HF quantizer impls -
            # record whichever exist rather than assuming one.
            for attr in ["weight_scale", "weight_scale_inv", "input_scale", "scale"]:
                if hasattr(mod, attr):
                    val = getattr(mod, attr)
                    shape = list(val.shape) if hasattr(val, "shape") else None
                    entry.setdefault("scale_attrs", {})[attr] = {
                        "dtype": str(getattr(val, "dtype", type(val))),
                        "shape": shape,
                    }
            info["sampled_modules"].append(entry)
    except Exception as e:  # noqa: BLE001
        info["sampling_error"] = str(e)

    # Direct, cheap signal: does ANY parameter still live in a float8 dtype at
    # rest (native fp8 storage), vs everything already upcast to bf16/fp16
    # (eager dequant-on-load fallback)?
    try:
        float8_dtype_names = {
            f"torch.{n}" for n in ("float8_e4m3fn", "float8_e5m2", "float8_e4m3fnuz")
            if hasattr(torch, n)
        }
        seen_dtypes = {}
        for _, p in model.named_parameters():
            seen_dtypes[str(p.dtype)] = seen_dtypes.get(str(p.dtype), 0) + 1
        info["parameter_dtype_histogram"] = seen_dtypes
        info["any_native_float8_storage"] = any(k in float8_dtype_names for k in seen_dtypes)
    except Exception as e:  # noqa: BLE001
        info["dtype_histogram_error"] = str(e)

    return info


def _default_prompts_file() -> str:
    """Canonical repo-relative location: <repo>/.pm/eval/prompts.jsonl, five
    parents up from workspace/demo/qwen3_5/accept/reference/. Falls back to a
    sibling file when this script has been copied elsewhere (e.g. scp'd
    standalone to a remote box for execution) rather than crashing - callers
    running off-repo should pass --prompts-file explicitly regardless.
    """
    here = Path(__file__).resolve()
    try:
        return str(here.parents[5] / ".pm" / "eval" / "prompts.jsonl")
    except IndexError:
        return str(here.parent / "prompts.jsonl")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-id", default="Qwen/Qwen3.5-35B-A3B-FP8")
    ap.add_argument(
        "--revision",
        default="9d1823d2dee688a6b25e77009dc727688c44936e",
        help="Pinned checkpoint revision sha (do not use a moving 'main').",
    )
    ap.add_argument(
        "--prompts-file",
        default=_default_prompts_file(),
        help="Path to the pinned prompt set (read-only; never modified).",
    )
    ap.add_argument("--output-dir", default=str(Path(__file__).resolve().parent))
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument(
        "--topk-logits",
        type=int,
        default=4,
        help=(
            "Per-step top-k token ids/logits to persist (M2-I3 addendum: the AC-3 harness's "
            "reference_loader.py needs at least top-2 for margin/tie-flip evidence; k=4 default "
            "gives headroom beyond that minimum). Must be >= 2."
        ),
    )
    args = ap.parse_args()
    if args.topk_logits < 2:
        raise SystemExit("--topk-logits must be >= 2 (the AC-3 harness needs at least top-2)")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import transformers
    import accelerate

    log(f"torch {torch.__version__}, transformers {transformers.__version__}, accelerate {accelerate.__version__}")
    log(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}, HF_HOME={os.environ.get('HF_HOME')!r}")
    assert torch.cuda.is_available(), "This script requires a visible CUDA GPU."
    log(f"visible GPU: {torch.cuda.get_device_name(0)}")

    prompts = load_prompts(Path(args.prompts_file))
    log(f"loaded {len(prompts)} prompts from {args.prompts_file}")

    model, tokenizer, load_notes = load_model_and_tokenizer(args.model_id, args.revision)
    log(f"model loaded via {load_notes.get('loaded_via')}")

    fp8_exec_info = introspect_fp8_execution(model)
    log(f"fp8 execution introspection: {json.dumps(fp8_exec_info, indent=2, default=str)}")

    eos_ids = model.generation_config.eos_token_id
    if not isinstance(eos_ids, list):
        eos_ids = [eos_ids]
    eos_strs = {i: tokenizer.decode([i]) for i in eos_ids}
    log(f"generation_config.eos_token_id={eos_ids} -> {eos_strs}")
    log(f"generation_config.pad_token_id={model.generation_config.pad_token_id}")

    results = {}
    t_start_all = time.time()
    for row in prompts:
        pid = row["id"]
        messages = row["messages"]
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        # Some transformers versions return a bare Tensor here, others a
        # BatchEncoding (dict-like, with .input_ids / .attention_mask) -
        # handle both rather than assuming one.
        if hasattr(encoded, "input_ids"):
            input_ids = encoded.input_ids.to(model.device)
            am = encoded.get("attention_mask") if hasattr(encoded, "get") else None
            attention_mask = am.to(model.device) if am is not None else torch.ones_like(input_ids)
        else:
            input_ids = encoded.to(model.device)
            attention_mask = torch.ones_like(input_ids)

        t0 = time.time()
        with torch.no_grad():
            gen = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                num_beams=1,
                output_logits=True,
                return_dict_in_generate=True,
                use_cache=True,
            )
        elapsed = time.time() - t0

        prompt_len = input_ids.shape[1]
        full_seq = gen.sequences[0].tolist()
        output_ids = full_seq[prompt_len:]
        num_generated = len(output_ids)
        hit_eos = num_generated < args.max_new_tokens
        eos_step = num_generated - 1 if hit_eos else None

        # M2-I3 addendum: persist per-step top-k (not just top-1) so the AC-3 harness's
        # reference_loader.py can compute a real margin (logit[top1] - logit[top2]) and the
        # tie-flip waiver path (goal.md AC-3's second sentence) stops being structurally
        # unsubstantiable. `k = args.topk_logits` (default 4: harness needs >=2, this gives
        # headroom for a future 3-/4-way-tie analysis without a second regeneration).
        #
        # top1 extraction is DELIBERATELY UNCHANGED (still torch.max, not torch.topk[...][0]):
        # a first attempt used torch.topk's own index-0 for top1 and hit a real assertion
        # failure on p06-poem — torch.max and torch.topk are different kernels and can break an
        # EXACT top-logit tie differently even on identical input (the same tie-breaking-is-not-
        # guaranteed risk class M2-I3's oracle work independently found for the MoE router's
        # torch.topk). Keeping top1 on the original torch.max call preserves this script's
        # existing identity-critical behavior byte-for-byte; the separately-computed top-k is
        # purely additive.
        top1_ids = []
        top1_logits = []
        topk_ids_per_step = []
        topk_logits_per_step = []
        for step_logits in gen.logits:  # tuple length == num_generated
            logits_f32 = step_logits[0].float()
            val, idx = torch.max(logits_f32, dim=-1)
            top1_ids.append(int(idx.item()))
            top1_logits.append(float(val.item()))

            k = min(args.topk_logits, logits_f32.shape[-1])
            topk_vals, topk_idxs = torch.topk(logits_f32, k, dim=-1)
            ids_row = [int(x) for x in topk_idxs.tolist()]
            logits_row = [float(x) for x in topk_vals.tolist()]
            top1_id = int(idx.item())
            if ids_row[0] != top1_id:
                # Only reachable on an EXACT top-logit tie (torch.topk resolved it to a
                # different index than torch.max). FIX (caught in review: an earlier version
                # blindly OVERWROTE slot 0 with top1_id, which silently DUPLICATED it when
                # top1_id already occupied a later slot -- e.g. p06-poem step 56 produced
                # [288, 288, 75635, 91491], dropping the genuinely distinct tied alternative).
                # Correct handling: the verified top1 (torch.max's pick) shares the max logit,
                # so it must already be SOMEWHERE in topk's own window -- find it and SWAP it
                # into slot 0 (both id and logit) rather than overwrite, so the displaced id
                # survives at its new position and becomes the real, distinct top-2.
                try:
                    max_pos = ids_row.index(top1_id)
                except ValueError:
                    raise AssertionError(
                        f"[{pid}] torch.max's argmax id {top1_id} (logit {float(val.item())}) "
                        f"is not present anywhere in the top-{k} window {ids_row} -- the tie "
                        f"group is larger than k; increase --topk-logits"
                    )
                assert abs(logits_row[max_pos] - float(val.item())) < 1e-6, (
                    f"[{pid}] found id {top1_id} at topk slot {max_pos} but its logit "
                    f"{logits_row[max_pos]} != torch.max's {float(val.item())} - not a tie, "
                    f"a real bug"
                )
                if max_pos != 0:
                    ids_row[0], ids_row[max_pos] = ids_row[max_pos], ids_row[0]
                    logits_row[0], logits_row[max_pos] = logits_row[max_pos], logits_row[0]
            assert len(set(ids_row)) == k, (
                f"[{pid}] topk ids not distinct after tie reconciliation: {ids_row}"
            )
            topk_ids_per_step.append(ids_row)
            topk_logits_per_step.append(logits_row)
        # Sanity: greedy argmax of returned logits must equal the actually
        # emitted token at every step (verifies do_sample=False was honored
        # and no logits warper altered the pick).
        assert top1_ids == output_ids, (
            f"[{pid}] greedy argmax(logits) != generated token ids - "
            f"do_sample=False was not purely greedy: {top1_ids} vs {output_ids}"
        )
        # top2_{id,logit}_per_step: the exact optional keys reference_loader.py checks for
        # (accept/harness/reference_loader.py `_OPTIONAL_TOP2_ID_KEYS`/`_OPTIONAL_TOP2_LOGIT_KEYS`).
        # `None` per position only if a caller ever set --topk-logits below 2 at the vocab-size
        # floor (can't happen given the >=2 CLI validation above plus vocab_size >> 2).
        top2_id_per_step = [row[1] if len(row) > 1 else None for row in topk_ids_per_step]
        top2_logit_per_step = [row[1] if len(row) > 1 else None for row in topk_logits_per_step]

        results[pid] = {
            "input_ids": input_ids[0].tolist(),
            "output_ids": output_ids,
            "num_generated": num_generated,
            "hit_eos": hit_eos,
            "eos_step": eos_step,
            "top1_logit_per_step": top1_logits,
            "top2_id_per_step": top2_id_per_step,
            "top2_logit_per_step": top2_logit_per_step,
            "topk_ids_per_step": topk_ids_per_step,
            "topk_logits_per_step": topk_logits_per_step,
            "decoded_output": tokenizer.decode(output_ids, skip_special_tokens=False),
            "elapsed_seconds": round(elapsed, 3),
        }
        log(f"{pid}: prompt_len={prompt_len} generated={num_generated} hit_eos={hit_eos} ({elapsed:.1f}s)")

    total_elapsed = time.time() - t_start_all

    meta = {
        "model_id": args.model_id,
        "revision": args.revision,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "greedy": True,
        "eos_token_id": eos_ids,
        "eos_token_strs": eos_strs,
        "pad_token_id": model.generation_config.pad_token_id,
        "chat_template_source": "tokenizer.apply_chat_template (model-shipped template)",
        "load_notes": load_notes,
        "fp8_execution_introspection": fp8_exec_info,
        "transformers_disable_deepgemm_linear": os.environ.get("TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR"),
        "topk_logits_k": args.topk_logits,
        "versions": {
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "accelerate": accelerate.__version__,
        },
        "gpu": torch.cuda.get_device_name(0),
        "total_elapsed_seconds": round(total_elapsed, 1),
        "prompts_file": str(args.prompts_file),
        "num_prompts": len(prompts),
    }

    out_path = out_dir / "reference_outputs.json"
    with open(out_path, "w") as f:
        json.dump({"meta": meta, "results": results}, f, indent=2)
    log(f"wrote {out_path}")


if __name__ == "__main__":
    main()
