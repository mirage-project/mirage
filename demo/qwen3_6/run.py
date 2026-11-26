"""One-call BF16 text generation with MPK's Qwen3.6-35B-A3B offline graph."""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from mirage.mpk.models.qwen3_5.moe_builder import Qwen36MoeBuilder
from mirage.mpk.models.qwen3_5.offline import create_offline_runtime, offline_generated_ids
from mirage.mpk.models.qwen3_5.weight_loader import Qwen36WeightLoader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--batch-size", type=int, choices=(1, 8), default=1,
                        help="B8 requires MPK_QWEN_EXPERIMENTAL_B8=1 and is not HF-exact")
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--tokens-out", type=Path)
    parser.add_argument("--shard-root", type=Path, action="append", default=[])
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")

    loader = Qwen36WeightLoader(args.model, shard_roots=args.shard_root)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    rendered = tokenizer.apply_chat_template(
        [dict(role="user", content=args.prompt)], tokenize=False,
        add_generation_prompt=True, enable_thinking=False)
    prompt_ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
    if not prompt_ids:
        parser.error("A nonempty prompt is required")
    ignore_eos = args.ignore_eos or args.batch_size == 8
    mpk, plan, metadata = create_offline_runtime(
        loader.config, [prompt_ids] * args.batch_size, args.max_new_tokens,
        ignore_eos=ignore_eos)
    try:
        builder = Qwen36MoeBuilder(mpk, kv_plan=plan)
        builder.build_from_model(model_path=args.model, shard_roots=args.shard_root)
        mpk.compile(output_dir=str(args.output_dir))
        mpk()
        torch.cuda.synchronize()
        generated = offline_generated_ids(metadata, len(prompt_ids), args.max_new_tokens,
                                          require_full=ignore_eos)
        if args.tokens_out:
            args.tokens_out.parent.mkdir(parents=True, exist_ok=True)
            args.tokens_out.write_text(json.dumps({
                "prompt_ids": prompt_ids if args.batch_size == 1 else [prompt_ids] * args.batch_size,
                "generated_ids": generated[0] if args.batch_size == 1 else generated,
                "final_step": metadata["step"].tolist(),
            }, indent=2) + "\n")
        for row, ids in enumerate(generated):
            decoded = tokenizer.decode(ids, skip_special_tokens=True)
            print(decoded if args.batch_size == 1 else f"[{row}] {decoded}")
    finally:
        mpk.finalize()


if __name__ == "__main__":
    main()
