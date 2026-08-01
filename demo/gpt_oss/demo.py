"""GPT-OSS-20B on MPK.

The checkpoint must be plain bf16. The released one is MXFP4, which MPK has no
kernels for; dequantise it once with

    AutoModelForCausalLM.from_pretrained(
        <mxfp4 dir>, quantization_config=Mxfp4Config(dequantize=True),
        dtype=torch.bfloat16).save_pretrained(<bf16 dir>)

Only the expert tensors are quantised, so that is an expansion, not a
re-quantisation.

    python demo/gpt_oss/demo.py --model /raid/catalyst/models/gpt-oss-20b-bf16
"""

import argparse

import torch
from transformers import AutoTokenizer

import mirage as mi
from mirage.mpk.models.gpt_oss.builder import GptOssBuilder

DEFAULT_PROMPT = "Give me a short introduction to large language models."


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str,
                   default="/raid/catalyst/models/gpt-oss-20b-bf16",
                   help="local bf16 checkpoint directory")
    p.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    p.add_argument("--max-seq-length", type=int, default=512)
    p.add_argument("--max-num-batched-tokens", type=int, default=8)
    p.add_argument("--max-num-batched-requests", type=int, default=1)
    p.add_argument("--page-size", type=int, default=4096)
    p.add_argument("--max-num-pages", type=int, default=16)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--raw-prompt", action="store_true",
                   help="skip the harmony chat template; GPT-OSS is trained "
                        "with it, so raw prompts are off-distribution")
    return p.parse_args()


def main():
    args = parse_args()
    torch.set_default_dtype(torch.bfloat16)
    device = "cuda"

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

    total_num_requests = 1
    tokens = torch.zeros(total_num_requests, args.max_seq_length,
                         dtype=torch.long, device=device)
    tokens[0, :prompt_len] = ids.to(device)
    step = torch.zeros(total_num_requests, dtype=torch.int32, device=device)
    prompt_lengths = torch.full((total_num_requests,), prompt_len,
                                dtype=torch.int32, device=device)
    num_new_tokens = torch.ones(total_num_requests, dtype=torch.int32,
                                device=device)
    input_tokens = torch.zeros(args.max_num_batched_tokens, 1,
                               dtype=torch.long, device=device)
    output_tokens = torch.zeros(args.max_num_batched_tokens, 1,
                                dtype=torch.long, device=device)
    qo_indptr = torch.empty(args.max_num_batched_requests + 1,
                            dtype=torch.int32, device=device)
    kv_indptr = torch.empty(args.max_num_batched_requests + 1,
                            dtype=torch.int32, device=device)
    kv_indices = torch.empty(args.max_num_pages, dtype=torch.int32,
                             device=device)
    kv_indices_snapshot = torch.empty(args.max_num_pages, dtype=torch.int32,
                                      device=device)
    kv_last_page_len = torch.empty(args.max_num_batched_requests,
                                   dtype=torch.int32, device=device)

    num_workers, num_schedulers = mi.get_configurations_from_gpu(0)
    mpk = mi.PersistentKernel(
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
        eos_token_id=200002,
        meta_tensors={
            "step": step,
            "tokens": tokens,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "num_new_tokens": num_new_tokens,
            "prompt_lengths": prompt_lengths,
            "qo_indptr_buffer": qo_indptr,
            "paged_kv_indptr_buffer": kv_indptr,
            "paged_kv_indices_buffer": kv_indices,
            "paged_kv_indices_snapshot": kv_indices_snapshot,
            "paged_kv_last_page_len_buffer": kv_last_page_len,
        },
        profiler_tensor=None,
        trace_name="",
        spec_decode_config=None,
        use_cutlass_kernel=False,
    )

    print("Building the task graph...")
    builder = GptOssBuilder(mpk)
    builder.build_from_model(model_name=args.model, model_path=args.model)

    print("Compiling the megakernel...")
    mpk.compile(output_dir=args.output_dir)

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    starter.record()
    mpk()
    ender.record()
    torch.cuda.synchronize()
    run_time = starter.elapsed_time(ender)

    end = step[0].item() + 1
    print(tokenizer.decode(tokens[0, :end], skip_special_tokens=True))
    generated = max(0, end - prompt_len)
    print(f"Prompt length {prompt_len}, generated {generated}, "
          f"per-token latency: {run_time / max(end, 1):.3f} ms")
    mpk.finalize()


if __name__ == "__main__":
    main()
