from mirage.engine import *
import argparse

"""
Usage (single GPU)::

    config = RunnerConfig(model="Qwen/Qwen3-8B")
    runner = ModelRunner(config)
    engine  = LLMEngine(manager, runner, tokenizer)

Usage (multi-GPU via mpirun)::
    # this functionality needs further debug, now deprecated
    # Launch: mpirun -n 2 python script.py
    config = RunnerConfig(model="Qwen/Qwen3-8B", tensor_parallel_size=2)
    runner = ModelRunner(config)
    ...
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
        ("introduce yourself",               0),
        ("What is buggy in CMU?",            0),
        ("How to use ncu for profilling?",   200),
        ("list all prime numbers within 100", 200),
        ("what is the capital of France?",    300),
        ("Tell me the difference between lpl and lck", 100)
    ]

    outputs = llm.generate_incremental(arrivals)

    # for (prompt, _), output in zip(arrivals, outputs):
    #     print(f"\nPrompt: {prompt!r}")
    #     print(f"Completion: {output['text']!r}")

