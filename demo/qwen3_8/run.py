"""One-call MPK offline inference for Qwen3.8-27B dense BF16 text."""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from mirage.mpk.models.qwen3_5.builder import Qwen38Builder
from mirage.mpk.models.qwen3_5.offline import create_offline_runtime, offline_generated_ids
from mirage.mpk.models.qwen3_5.weight_loader import Qwen35WeightLoader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--batch-size", type=int, choices=(1, 8), default=1,
                        help="Number of equal-length offline requests; B8 requires MPK_QWEN_EXPERIMENTAL_B8=1")
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--tokens-out", type=Path,
                        help="Write exact prompt and generated token IDs for oracle comparison")
    parser.add_argument("--logits-out", type=Path,
                        help="Save final BF16 logits for independent numerical comparison")
    parser.add_argument("--shard-root", type=Path, action="append", default=[])
    args = parser.parse_args()

    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    loader = Qwen35WeightLoader(args.model, shard_roots=args.shard_root)
    config = loader.config
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    rendered = tokenizer.apply_chat_template(
        [dict(role="user", content=args.prompt)], tokenize=False,
        add_generation_prompt=True, enable_thinking=False)
    prompt_ids = tokenizer(rendered, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    prompt_length = prompt_ids.numel()
    if prompt_length < 1:
        parser.error("A nonempty prompt is required")
    try:
        mpk, plan, metadata = create_offline_runtime(
            config, [prompt_ids.tolist()] * args.batch_size, args.max_new_tokens,
            ignore_eos=args.ignore_eos or args.batch_size == 8)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        builder = Qwen38Builder(mpk, kv_plan=plan)
        builder.build_from_model(model_path=args.model, shard_roots=args.shard_root)
        mpk.compile(output_dir=str(args.output_dir))
        mpk()  # The offline scheduler consumes the prompt and decodes in this call.
        torch.cuda.synchronize()
        generated = offline_generated_ids(metadata, prompt_length, args.max_new_tokens,
                                          require_full=args.ignore_eos or args.batch_size == 8)
        if args.tokens_out is not None:
            args.tokens_out.parent.mkdir(parents=True, exist_ok=True)
            args.tokens_out.write_text(json.dumps({
                "prompt_ids": prompt_ids.tolist() if args.batch_size == 1 else
                              [prompt_ids.tolist()] * args.batch_size,
                "generated_ids": generated[0] if args.batch_size == 1 else generated,
                "final_step": int(metadata["step"][0].item()) if args.batch_size == 1 else
                              metadata["step"].tolist(),
                "max_new_tokens": args.max_new_tokens,
            }, indent=2) + "\n")
        if args.logits_out is not None:
            args.logits_out.parent.mkdir(parents=True, exist_ok=True)
            torch.save(builder.model.tensors["logits"].detach().cpu().clone(), args.logits_out)
        for row, ids in enumerate(generated):
            decoded = tokenizer.decode(ids, skip_special_tokens=True)
            print(decoded if args.batch_size == 1 else f"[{row}] {decoded}")
    finally:
        mpk.finalize()


if __name__ == "__main__":
    main()
