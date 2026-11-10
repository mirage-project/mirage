"""GLM-5.3 decode on SGLang, one engine per run, as demo/glm53_flash/sglang_dsa/sgl_spec_bench.py measures GLM-5.3-Flash (plain
decode): --bs prompts of each --prompt-tokens length, --tokens new tokens each (greedy, EOS ignored), CUDA graphs.
Per prompt length (median over --reps): the wall time of the bs requests' --tokens new tokens (the time of 1 new token
subtracted: the prefill), us per output token per request, tokens/s over the batch, and the first --show requests' outputs.
--profile: then --profile-steps forward steps under the CUDA profiler API (run under nsys --capture-range=cudaProfilerApi
--cuda-graph-trace=graph; demo/glm53_flash/sglang_dsa/nsys_e2e.py --one <trace>.sqlite: the period of the decode graph's
replays). The static megakernel decode replaces SGLang's when the environment selects the plugin (sgl_chain.sh):
  PYTHONPATH=<repo>:<repo>/demo/glm53_model SGLANG_EXTERNAL_MODEL_PACKAGE=sglang_glm53 [MIRAGE_STATIC_ALLREDUCE_IN_KERNEL=1]
All runs take the same engine flags.

python sgl_bench.py --model <GLM-5.3 snapshot> [--tp 8] [--bs 8] [--prompt-tokens 256 4000] [--tokens 256] [--reps 3]
    [--profile [--profile-steps 64] [--profile-tokens 256]]
"""
import argparse
import statistics
import time

import sglang as sgl

TASK = " Write a long story about the history of computing, chapter {i}."


def prompts_of(tok, n_tokens: int, bs: int) -> list:
    """bs prompts of n_tokens tokens each (request i: filler words cut to the count, then the task)."""
    out = []
    for i in range(bs):
        head, tail = f"Request {i}:", TASK.format(i=i)
        lo, hi = 0, max(8, n_tokens)
        while lo < hi:   # the filler word count whose prompt has <= n_tokens tokens
            mid = (lo + hi + 1) // 2
            filler = " ".join(f"item{j}" for j in range(mid))
            if len(tok.encode(f"{head} {filler}{tail}", add_special_tokens=False)) <= n_tokens:
                lo = mid
            else:
                hi = mid - 1
        filler = " ".join(f"item{j}" for j in range(lo))
        out.append(f"{head} {filler}{tail}" if lo else head + tail)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tp", type=int, default=8)
    ap.add_argument("--bs", type=int, default=8, help="at most 8 (the kernels' 8 slots)")
    ap.add_argument("--prompt-tokens", type=int, nargs="+", default=[256, 4000])
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--ctx", type=int, default=8448, help="context_length (at most 16384, the kernels' plans)")
    ap.add_argument("--mem-fraction", type=float, default=0.8)
    ap.add_argument("--show", type=int, default=2, help="print the outputs of the first N requests")
    ap.add_argument("--temperature", type=float, default=0.0, help="sampling temperature (0: greedy). A TP shard alone "
                    "decodes the same token for every request under greedy, so its MoE routes all rows alike: sample to "
                    "keep the rows apart")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--profile-steps", type=int, default=64)
    ap.add_argument("--profile-tokens", type=int, default=256)
    args = ap.parse_args()
    llm = sgl.Engine(model_path=args.model, tp_size=args.tp, max_running_requests=args.bs, cuda_graph_max_bs_decode=args.bs,
                     context_length=args.ctx, kv_cache_dtype="bfloat16", moe_runner_backend="flashinfer_trtllm",
                     disable_radix_cache=True, cuda_graph_backend_prefill="disabled", mem_fraction_static=args.mem_fraction,
                     log_level="warning")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    sp = {"max_new_tokens": args.tokens, "ignore_eos": True, "temperature": args.temperature}
    for n in args.prompt_tokens:
        prompts = prompts_of(tok, n, args.bs)
        out = llm.generate(prompts, sp)   # warm-up (the CUDA graphs of this batch size)
        times = []
        for _ in range(args.reps):
            t0 = time.perf_counter()
            llm.generate(prompts, {**sp, "max_new_tokens": 1})
            t1 = time.perf_counter()
            out = llm.generate(prompts, sp)
            t2 = time.perf_counter()
            times.append((t2 - t1) - (t1 - t0))
        wall = statistics.median(times)
        got = out[0]["meta_info"]["prompt_tokens"]
        print(f"bs {args.bs} prompt {got} tokens ({n} asked), {args.tokens} new: wall {wall * 1e6:.1f} us (reps "
              f"{', '.join(f'{t * 1e6:.1f}' for t in times)}), {wall / (args.tokens - 1) * 1e6:.1f} us per output token per "
              f"request, {args.bs * (args.tokens - 1) / wall:.1f} tok/s", flush=True)
        for i in range(min(args.show, len(out))):
            print(f"output {i} (prompt {n}):", repr(out[i]["text"][:160]), flush=True)
    if args.profile:
        prompts = prompts_of(tok, args.prompt_tokens[-1], args.bs)
        llm.start_profile(activities=["CUDA_PROFILER"], num_steps=args.profile_steps)
        llm.generate(prompts, {**sp, "max_new_tokens": args.profile_tokens})
        try:
            llm.stop_profile()
        except Exception as e:
            print("stop_profile:", e)
    llm.shutdown()


if __name__ == "__main__":
    main()
