"""GLM-5.3-Flash decode on SGLang 0.5.22, TP4, one engine per run (a batch size): bs prompts of each --prompt-tokens
length, --tokens new tokens each, CUDA graphs, plain decode (one token per request per step).
Per prompt length (median over --reps): the wall time of the bs requests' --tokens new tokens (the time of 1 new token
subtracted: the prefill), us per output token per request (one decode step), tokens/s over the batch. The prompts are cut
to the token count with the model's tokenizer (filler words before the task).
With the static layers (DSA_STATIC_LAYERS, ...: boot/sitecustomize.py) the hooks replace the layers as the chain's other runs.

python sgl_spec_bench.py --bs 1 [--prompt-tokens 4000] [--tokens 256] [--reps 3] [--ctx 8448]
    [--kv bfloat16] [--max-total-tokens N] [--profile [--profile-steps 24] [--profile-tokens 96]]
(end-to-end comparisons: nsys step periods, --profile under nsys (sgl_chain.py prof_* runs, nsys_e2e.py), one run per
prompt length, 256 and 4000 tokens)
"""
import argparse
import os
import statistics
import time

import sglang as sgl

MODEL = os.environ.get("GLM53F_MODEL", "/raid/catalyst/models/GLM-5.3-Flash")   # the checkpoint directory
# GLM53F_NCCL_PORT: SGLang's torch.distributed port (nccl_port; default: a random free one), one per chain when two chains
# start engines at the same time on two halves of a box
NCCL = {"nccl_port": int(os.environ["GLM53F_NCCL_PORT"])} if os.environ.get("GLM53F_NCCL_PORT") else {}
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
    ap.add_argument("--bs", type=int, required=True)
    ap.add_argument("--prompt-tokens", type=int, nargs="+", default=[4000])
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--ctx", type=int, default=8448, help="context_length (8448: 2048 index pools, as 8192)")
    ap.add_argument("--kv", default="bfloat16")
    ap.add_argument("--max-total-tokens", type=int, default=None)
    ap.add_argument("--profile", action="store_true", help="then --profile-steps forward steps under the CUDA profiler API "
                    "(nsys), the last prompt length's prompts with --profile-tokens new tokens")
    ap.add_argument("--profile-steps", type=int, default=24)
    ap.add_argument("--profile-tokens", type=int, default=96)
    args = ap.parse_args()
    extra = {} if args.max_total_tokens is None else {"max_total_tokens": args.max_total_tokens}
    llm = sgl.Engine(model_path=MODEL, tp_size=4, max_running_requests=args.bs, cuda_graph_max_bs_decode=args.bs,
                     context_length=args.ctx, mem_fraction_static=0.6, dsa_prefill_backend="trtllm",
                     dsa_decode_backend="trtllm", kv_cache_dtype=args.kv, moe_runner_backend="flashinfer_trtllm",
                     disable_radix_cache=True, log_level="warning", **NCCL, **extra)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    sp = {"max_new_tokens": args.tokens, "ignore_eos": True, "temperature": 0.0}
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
        line = (f"bs {args.bs} prompt {got} tokens ({n} asked), {args.tokens} new: wall {wall * 1e6:.1f} us "
                f"(reps {', '.join(f'{t * 1e6:.1f}' for t in times)}), {wall / (args.tokens - 1) * 1e6:.1f} us per output token "
                f"per request, {args.bs * (args.tokens - 1) / wall:.1f} tok/s")
        print(line, flush=True)
        print("output 0:", repr(out[0]["text"][:120]), flush=True)
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
