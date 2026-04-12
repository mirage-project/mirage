import torch
import argparse

import mirage as mi
from mirage.mpk.mpk import MPK, MPKMetadata, MirageModelConfig

# Return the largest factor of m that is less than or equal to n
# This is used to determine the grid size
def max_factor_leq_n(m: int, n: int) -> int:
    max_factor = 1
    i = 1
    while i * i <= m:
        if m % i == 0:
            if i <= n:
                max_factor = max(max_factor, i)
            if m // i <= n:
                max_factor = max(max_factor, m // i)
        i += 1
    return max_factor

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-num-batched-tokens", default=8, type=int, help="Max number of tokens in a batch")
    parser.add_argument("--max-num-batched-requests", default=4, type=int, help="Max number of requests in a batch")
    parser.add_argument("--page-size", default=4096, type=int, help="Page size")
    parser.add_argument("--max-num-pages", default=16, type=int, help="Max num pages")
    parser.add_argument("--output-dir", help="Output files directory")
    parser.add_argument("--trace-name", default="", help="Perfetto trace output name")
    parser.add_argument(
        "--profiling", action="store_true", help="Use Profiler to generate trace"
    )
    # lookahead or promptlookup
    parser.add_argument(
        "--spec-decode",
        default=None,
        choices=["promptlookup", "lookahead"],
        help="Enable speculative decoding with 'lookahead' or 'promptlookup' mode.",
    )
    parser.add_argument(
        "--ngram-size",
        default=3,
        type=int,
        help="Ngram size for lookahead spec decode",
    )
    parser.add_argument(
        "--max-seq-length",
        default=512,
        type=int,
        help="Max sequence length for lookahead spec decode",
    )
    parser.add_argument(
        "--spec-length",
        default=3,
        type=int,
        help="Spec length for lookahead spec decode",
    )

    parser.add_argument("--model-path", type=str, default=None, help="Path to a local model (necessary for multi-GPU demo)")
    parser.add_argument(
        "--model", type=str, default='Qwen/Qwen3-8B', help="Model path on hugging face"
    )
    parser.add_argument(
        "--no-use-cutlass-kernel",
        action="store_false",
        dest="use_cutlass_kernel",
        default=True,
        help="Not use the cutlass version kernel.",
    )
    parser.add_argument("--ignore-eos", action="store_true", help="Ignore eos token during generation")
    args = parser.parse_args()

    print("Input arguments:", args)
    model_name = args.model

    total_num_requests = args.max_num_batched_requests
    # get all model weight tensors
    tokens = torch.full((total_num_requests, args.max_seq_length), 0, dtype=torch.long, device="cuda")

    #prompt = "Give me a short introduction to large language model."
    prompt = "How to implement GEMM kernel at nvidia blackwell gpu, please explain in detail."
    prompt_lengths = torch.full((total_num_requests,), 0, dtype=torch.int, device="cuda")
    
    # get all model weight tensors
    input_tokens = torch.full((args.max_num_batched_tokens, 1), 0, dtype=torch.long, device="cuda")
    output_tokens = torch.full((args.max_num_batched_tokens, 1), 0, dtype=torch.long, device="cuda")
    prev_pos = 0

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    step = torch.full((total_num_requests, ), 0, dtype=torch.int32, device="cuda")
    num_new_tokens = torch.full((total_num_requests, ), 1, dtype=torch.int32, device="cuda")
    

    if args.profiling:
        profiler_tensor = torch.zeros(
            3000 * 128, dtype=torch.uint64, device="cuda"
        ).contiguous()
    else:
        profiler_tensor = None
        
    spec_decode_config = mi.mpk.speculative.spec_decode_class(
        args.spec_decode,
        ngram_size=args.ngram_size,
        spec_length=args.spec_length,
    )
        
    # num_workers, num_schedulers = mi.get_configurations_from_gpu(self.rank)
    qo_indptr_buffer = torch.zeros(
        args.max_num_batched_requests + 1, dtype=torch.int32, device="cuda")
    paged_kv_indptr_buffer = torch.zeros(
        args.max_num_batched_requests + 1, dtype=torch.int32, device="cuda")
    paged_kv_indices_buffer = torch.zeros(
        args.max_num_pages, dtype=torch.int32, device="cuda")
    paged_kv_last_page_len_buffer = torch.zeros(
        args.max_num_batched_requests, dtype=torch.int32, device="cuda")
    
    mirage_model_config = MirageModelConfig(with_lm_head=True)
    
    mpk_metadata = MPKMetadata(
        mode="offline",
        total_num_requests=total_num_requests,
        num_remote_schedulers=0,
        max_seq_length=args.max_seq_length,
        max_num_batched_requests=args.max_num_batched_requests,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_pages=args.max_num_pages,
        page_size=args.page_size, #
        # model
        weight_from_model=True,
        model_name=args.model,
        # meta tensors
        step=step,
        tokens=tokens,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        num_new_tokens=num_new_tokens,
        prompt_lengths=prompt_lengths,
        qo_indptr_buffer=qo_indptr_buffer,
        paged_kv_indptr_buffer=paged_kv_indptr_buffer,
        paged_kv_indices_buffer=paged_kv_indices_buffer,
        paged_kv_last_page_len_buffer=paged_kv_last_page_len_buffer,
        # model config
        model_config=mirage_model_config,
        # meta tensors end
        profiling=args.profiling,
        profiler_tensor=profiler_tensor,
        trace_name=args.trace_name,
        spec_decode=args.spec_decode,
        spec_decode_config=spec_decode_config,
        use_cutlass_kernel=args.use_cutlass_kernel,
    )
    mpk = MPK(mpk_metadata)
        
    # Building graph logics
    mpk.build()
    mpk.compile(output_dir=args.output_dir)
    
    mpk.load_new_request(prompt)

    stream = torch.cuda.Stream()
    warmup = 0
    output_len = 512

    starter.record()
    mpk()
    ender.record()
    torch.cuda.synchronize()
    run_time = starter.elapsed_time(ender)

    for r in range(total_num_requests):
        generated_ids = tokens[r, : step[r] + 1]
        response = mpk.decode(generated_ids)
        print(response)

    print("Prompt length {}, generate length {}, per-token latency (both prefill and decode): {:.3f} ms".format(
            prompt_lengths[0], step.max().item() + 1 - prompt_lengths[0], run_time / (step.max().item() + 1)
        )
    )
