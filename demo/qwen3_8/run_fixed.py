"""Fixed-step 27B numerical probe; launch MPK once per token."""
import argparse
from pathlib import Path

from transformers import AutoTokenizer, GenerationConfig
from mirage.mpk.models.qwen3_5.model import Qwen35Model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--prompt', required=True)
    parser.add_argument('--max-new-tokens', type=int, default=16)
    parser.add_argument('--attn-implementation', choices=('eager', 'sdpa'), default='sdpa')
    parser.add_argument('--prefill-mode', choices=('mpk', 'block', 'stream'), default='mpk')
    parser.add_argument('--shard-root', type=Path, action='append', default=[],
                        help='Explicitly trust a storage root containing symlinked checkpoint shards')
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    generation = GenerationConfig.from_pretrained(args.model, local_files_only=True)
    rendered = tokenizer.apply_chat_template([dict(role='user', content=args.prompt)],
                    tokenize=False, add_generation_prompt=True, enable_thinking=False)
    prompt_ids = tokenizer(rendered, add_special_tokens=False, return_tensors='pt')['input_ids'][0].tolist()
    model = Qwen35Model.from_pretrained(args.model, attn_implementation=args.attn_implementation,
                                       shard_roots=args.shard_root)
    try:
        model.compile(args.output_dir, prefill_chunk_size=64 if args.prefill_mode == 'mpk' else None)
        generated = model.generate(prompt_ids, args.max_new_tokens,
                                   eos_token_ids=generation.eos_token_id,
                                   prefill_mode=args.prefill_mode)
        print(tokenizer.decode(generated, skip_special_tokens=True))
    finally:
        model.close()


if __name__ == '__main__':
    main()
