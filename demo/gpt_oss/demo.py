"""GPT-OSS-20B on MPK. Pass --use-mirage for the megakernel, omit it for the
HuggingFace reference.

The checkpoint must be plain bf16. The released one is MXFP4, which MPK has no
kernels for; convert it once with Mxfp4Config(dequantize=True).
TODO: native MXFP4 support.
"""

import argparse
import os

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import mirage as mi
from mirage.mpk.kv_group import resolve_pool_size
from mirage.mpk.models.gpt_oss.builder import GptOssBuilder, plan_kv_cache

DEFAULT_PROMPT = "Give me a short introduction to large language models."


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-mirage", action="store_true",
                        help="Use Mirage kernels")
    parser.add_argument("--model", type=str,
                        default="/raid/catalyst/models/gpt-oss-20b-bf16",
                        help="Local bf16 checkpoint directory")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--raw-prompt", action="store_true",
                        help="Skip the chat template")
    parser.add_argument("--max-seq-length", default=512, type=int,
                        help="Max sequence length")
    parser.add_argument("--max-new-tokens", type=int, default=None,
                        help="Decode cap; defaults to filling max_seq_length")
    parser.add_argument("--max-num-batched-tokens", default=8, type=int,
                        help="Max number of tokens in a batch")
    parser.add_argument("--max-num-batched-requests", default=1, type=int,
                        help="Max number of requests in a batch")
    parser.add_argument("--page-size", default=4096, type=int, help="Page size")
    parser.add_argument("--kv-budget", type=str, default=None,
                        help="Memory for the KV pool: an absolute size "
                             "('24GiB') or a fraction of the device's TOTAL "
                             "memory ('0.6'). Exclusive with --max-num-pages")
    parser.add_argument("--max-num-pages", default=None, type=int,
                        help="Size the pool by page count instead; a page id "
                             "costs slots x page_bytes, which changes with "
                             "--page-size")
    parser.add_argument("--output-dir", help="Output files directory")
    parser.add_argument("--ignore-eos", action="store_true",
                        help="Ignore eos token during generation")
    args = parser.parse_args()

    print("Input arguments:", args)
    torch.set_default_dtype(torch.bfloat16)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if args.raw_prompt:
        text = args.prompt
    else:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": args.prompt}],
            tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer([text], return_tensors="pt").input_ids[0]
    prompt_len = input_ids.shape[-1]
    assert prompt_len < args.max_seq_length
    output_len = (args.max_new_tokens if args.max_new_tokens is not None
                  else args.max_seq_length - prompt_len)

    starter, ender = (torch.cuda.Event(enable_timing=True),
                      torch.cuda.Event(enable_timing=True))

    if args.use_mirage:
        # The megakernel stops at max_seq_length, so shorten it to make
        # --max-new-tokens the binding limit.
        seq_len = min(args.max_seq_length, prompt_len + output_len)
        tokens = torch.zeros(1, seq_len, dtype=torch.long, device="cuda")
        tokens[0, :prompt_len] = input_ids.to("cuda")
        mbt = args.max_num_batched_tokens
        n_req = tokens.shape[0]
        # KV 2.0: the plan is the source of truth for the cache layout
        config = AutoConfig.from_pretrained(args.model)
        kv_plan = plan_kv_cache(config, args.page_size)
        try:
            max_num_pages = resolve_pool_size(
                kv_plan, kv_budget=args.kv_budget,
                max_num_pages=args.max_num_pages, max_seq_length=seq_len,
                max_num_batched_requests=args.max_num_batched_requests,
                max_num_batched_tokens=mbt)
        except ValueError as e:
            raise SystemExit(str(e))
        meta_tensors = {
            "step": torch.zeros(n_req, dtype=torch.int32, device="cuda"),
            "tokens": tokens,
            "input_tokens": torch.zeros(mbt, 1, dtype=torch.long, device="cuda"),
            "output_tokens": torch.zeros(mbt, 1, dtype=torch.long, device="cuda"),
            "num_new_tokens": torch.ones(n_req, dtype=torch.int32, device="cuda"),
            "prompt_lengths": torch.full((n_req,), prompt_len,
                                         dtype=torch.int32, device="cuda"),
            "qo_indptr_buffer": torch.empty(args.max_num_batched_requests + 1,
                                            dtype=torch.int32, device="cuda"),
            **kv_plan.build_meta_tensors(
                max_num_batched_requests=args.max_num_batched_requests,
                max_seq_length=seq_len),
        }

        num_workers, num_schedulers = mi.get_configurations_from_gpu(0)
        mpk = mi.PersistentKernel(
            mode="offline", world_size=1, mpi_rank=0,
            num_workers=num_workers, num_local_schedulers=num_schedulers,
            num_remote_schedulers=0,
            max_seq_length=seq_len,
            max_num_batched_requests=args.max_num_batched_requests,
            max_num_batched_tokens=mbt,
            max_num_pages=max_num_pages,
            kv_groups=kv_plan.group_specs(),
            eos_token_id=-1 if args.ignore_eos else 200002,
            meta_tensors=meta_tensors, profiler_tensor=None, trace_name="",
            spec_decode_config=None, use_cutlass_kernel=False,
        )

        print("Building the task graph...")
        GptOssBuilder(mpk, kv_plan=kv_plan).build_from_model(
            model_name=args.model, model_path=args.model)
        print("Compiling the megakernel...")
        mpk.compile(output_dir=args.output_dir)

        starter.record()
        mpk()
        ender.record()
        torch.cuda.synchronize()
        run_time = starter.elapsed_time(ender)

        end_idx = min(meta_tensors["step"][0].item() + 1,
                      prompt_len + output_len)
        token_ids = tokens[0, prompt_len:end_idx].cpu().tolist()
        response = tokenizer.decode(tokens[0, :end_idx],
                                    skip_special_tokens=True)
        mpk.finalize()
    else:
        print("Loading the HuggingFace reference...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, device_map="cuda").eval()

        starter.record()
        inp = input_ids.unsqueeze(0).cuda()
        with torch.no_grad():
            out = model.generate(
                inp,
                # pad_token_id == eos_token_id here, so generate() cannot infer
                # the mask; pass it explicitly to avoid the fallback
                attention_mask=torch.ones_like(inp),
                max_new_tokens=output_len, do_sample=False,
                temperature=None, top_p=None, top_k=None,
                eos_token_id=(None if args.ignore_eos
                              else model.config.eos_token_id),
                pad_token_id=tokenizer.pad_token_id or 199999)
        ender.record()
        torch.cuda.synchronize()
        run_time = starter.elapsed_time(ender)

        token_ids = out[0, prompt_len:].cpu().tolist()
        response = tokenizer.decode(out[0], skip_special_tokens=True)

    tokens_generated = len(token_ids)
    per_tok_ms = run_time / max(prompt_len + tokens_generated, 1)
    print(response)
    print("Prompt length {}, generate length {}, per-token latency {:.3f} ms"
          .format(prompt_len, tokens_generated, per_tok_ms))
