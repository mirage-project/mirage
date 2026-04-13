from mirage.engine import *
import argparse

"""
Usage (single GPU)::

    config = RunnerConfig(model="Qwen/Qwen3-8B")
    runner = ModelRunner(config)
    engine  = LLMEngine(runner)

TODO: Usage (multiple GPU)
"""

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
    parser.add_argument(
        "--use-nsys", action="store_true", help="Use nsys for profiling"
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

    runnerConfig = RunnerConfig(
        model=args.model,
        max_num_batched_requests=args.max_num_batched_requests,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_seq_length=args.max_seq_length,
        max_num_pages=args.max_num_pages,
        page_size=args.page_size,
        output_dir=args.output_dir
    )
    runner = ModelRunner(runnerConfig)
    llm = LLMEngine(runner)

    # (prompt, delay_in_microseconds) — delays are relative to the start of generate_incremental
    arrivals = [
        ("Introduce yourself",               0),
        ("How to implement GEMM kernel at nvidia blackwell gpu, please explain in detail.",            1000),
        ("Explain the difference between lpl and lck",   23),
        ("what is buggy in CMU? CMU means Carnegie Mellon University", 300),
        ("Lebron James and Steven Curry, who is the goat?",    75),
        ("Do you think Attack on Titan really have a good end?", 5)
    ]
    if args.use_nsys:
        import ctypes
        _cudart = ctypes.CDLL("libcudart.so")

        _cudart.cudaProfilerStart()
        outputs = llm.generate_incremental(arrivals)
        _cudart.cudaProfilerStop()
    else:    
        outputs = llm.generate_incremental(arrivals,timeout=60)

    # for (prompt, _), output in zip(arrivals, outputs):
    #     print(f"\nPrompt: {prompt!r}")
    #     print(f"Completion: {output['text']!r}")

