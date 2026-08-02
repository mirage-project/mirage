"""GPT-OSS-20B on MPK. Pass --use-mirage for the megakernel, omit it for the
HuggingFace reference; --save-tokens auto writes both to outputs/gpt_oss/ for
a diff.

The checkpoint must be plain bf16. The released one is MXFP4, which MPK has no
kernels for; convert it once with Mxfp4Config(dequantize=True).
"""

import argparse
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import mirage as mi
from mirage.mpk.models.gpt_oss.builder import GptOssBuilder

DEFAULT_SAVE_DIR = os.path.join("outputs", "gpt_oss")
MAX_SAVE_TOKENS = 100
DEFAULT_PROMPT = "Give me a short introduction to large language models."


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--use-mirage", action="store_true", help="Use Mirage kernels")
    p.add_argument("--model", type=str,
                   default="/raid/catalyst/models/gpt-oss-20b-bf16",
                   help="local bf16 checkpoint directory")
    p.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    p.add_argument("--max-seq-length", type=int, default=512)
    p.add_argument("--max-new-tokens", type=int, default=None,
                   help="defaults to filling max_seq_length")
    p.add_argument("--max-num-batched-tokens", type=int, default=8)
    p.add_argument("--max-num-batched-requests", type=int, default=1)
    p.add_argument("--page-size", type=int, default=4096)
    p.add_argument("--max-num-pages", type=int, default=16)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--save-tokens", type=str, default=None,
                   help="dump token ids, text and latency to JSON; 'auto' "
                        "picks outputs/gpt_oss/{mpk,torch}_output.json")
    p.add_argument("--raw-prompt", action="store_true",
                   help="skip the chat template")
    p.add_argument("--ignore-eos", action="store_true",
                   help="keep generating past EOS")
    return p.parse_args()


def dump(path, mode, token_ids, text, per_tok_ms, prompt_len, generated):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump({
            "token_ids": token_ids[:MAX_SAVE_TOKENS],
            "text": text,
            "latency_ms_per_token": per_tok_ms,
            "prompt_length": prompt_len,
            "generate_length": generated,
            "mode": mode,
        }, f, indent=2)
    print(f"Saved tokens to {path}")


def run_hf(args, tokenizer, ids, num_new):
    print("Loading the HuggingFace reference...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda").eval()
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    starter.record()
    inp = ids.unsqueeze(0).cuda()
    with torch.no_grad():
        out = model.generate(
            inp,
            # pad_token_id == eos_token_id here, so generate() cannot infer
            # the mask; pass it rather than take the fallback
            attention_mask=torch.ones_like(inp),
            max_new_tokens=num_new, do_sample=False,
            temperature=None, top_p=None, top_k=None,
            eos_token_id=None if args.ignore_eos else model.config.eos_token_id,
            pad_token_id=tokenizer.pad_token_id or 199999)
    ender.record()
    torch.cuda.synchronize()
    run_time = starter.elapsed_time(ender)
    generated = out[0, ids.shape[-1]:].cpu().tolist()
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    return generated, text, run_time


def run_mpk(args, tokenizer, ids, num_new):
    device = "cuda"
    prompt_len = ids.shape[-1]
    # the megakernel stops at max_seq_length, not at a token budget
    seq_len = min(args.max_seq_length, prompt_len + num_new)
    tokens = torch.zeros(1, seq_len, dtype=torch.long, device=device)
    tokens[0, :prompt_len] = ids.to(device)
    mbt = args.max_num_batched_tokens
    meta = {
        "step": torch.zeros(1, dtype=torch.int32, device=device),
        "tokens": tokens,
        "input_tokens": torch.zeros(mbt, 1, dtype=torch.long, device=device),
        "output_tokens": torch.zeros(mbt, 1, dtype=torch.long, device=device),
        "num_new_tokens": torch.ones(1, dtype=torch.int32, device=device),
        "prompt_lengths": torch.full((1,), prompt_len, dtype=torch.int32,
                                     device=device),
        "qo_indptr_buffer": torch.empty(args.max_num_batched_requests + 1,
                                        dtype=torch.int32, device=device),
        "paged_kv_indptr_buffer": torch.empty(args.max_num_batched_requests + 1,
                                              dtype=torch.int32, device=device),
        "paged_kv_indices_buffer": torch.empty(args.max_num_pages,
                                               dtype=torch.int32, device=device),
        "paged_kv_indices_snapshot": torch.empty(args.max_num_pages,
                                                 dtype=torch.int32, device=device),
        "paged_kv_last_page_len_buffer": torch.empty(
            args.max_num_batched_requests, dtype=torch.int32, device=device),
    }
    num_workers, num_schedulers = mi.get_configurations_from_gpu(0)
    mpk = mi.PersistentKernel(
        mode="offline", world_size=1, mpi_rank=0,
        num_workers=num_workers, num_local_schedulers=num_schedulers,
        num_remote_schedulers=0,
        max_seq_length=seq_len,
        max_num_batched_requests=args.max_num_batched_requests,
        max_num_batched_tokens=mbt,
        max_num_pages=args.max_num_pages, page_size=args.page_size,
        eos_token_id=-1 if args.ignore_eos else 200002,
        meta_tensors=meta, profiler_tensor=None, trace_name="",
        spec_decode_config=None, use_cutlass_kernel=False,
    )
    print("Building the task graph...")
    GptOssBuilder(mpk).build_from_model(model_name=args.model,
                                        model_path=args.model)
    print("Compiling the megakernel...")
    mpk.compile(output_dir=args.output_dir)

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    starter.record()
    mpk()
    ender.record()
    torch.cuda.synchronize()
    run_time = starter.elapsed_time(ender)

    end = min(meta["step"][0].item() + 1, prompt_len + num_new)
    generated = tokens[0, prompt_len:end].cpu().tolist()
    text = tokenizer.decode(tokens[0, :end], skip_special_tokens=True)
    mpk.finalize()
    return generated, text, run_time


def main():
    args = parse_args()
    torch.set_default_dtype(torch.bfloat16)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if args.raw_prompt:
        text = args.prompt
    else:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": args.prompt}],
            tokenize=False, add_generation_prompt=True)
    ids = tokenizer([text], return_tensors="pt").input_ids[0]
    prompt_len = ids.shape[-1]
    assert prompt_len < args.max_seq_length
    num_new = (args.max_new_tokens if args.max_new_tokens is not None
               else args.max_seq_length - prompt_len)

    mode = "mpk" if args.use_mirage else "torch"
    runner = run_mpk if args.use_mirage else run_hf
    generated, out_text, run_time = runner(args, tokenizer, ids, num_new)

    per_tok_ms = run_time / max(prompt_len + len(generated), 1)
    print(out_text)
    print(f"Prompt length {prompt_len}, generate length {len(generated)}, "
          f"per-token latency: {per_tok_ms:.3f} ms")

    if args.save_tokens:
        path = args.save_tokens
        if path == "auto":
            path = os.path.join(DEFAULT_SAVE_DIR, f"{mode}_output.json")
        dump(path, mode, generated, out_text, per_tok_ms, prompt_len,
             len(generated))


if __name__ == "__main__":
    main()
