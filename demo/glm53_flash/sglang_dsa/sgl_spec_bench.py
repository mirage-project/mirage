"""GLM-5.3-Flash decode on SGLang 0.5.22, TP4, one engine per run (a batch size and a speculative configuration): bs prompts
of each --prompt-tokens length, --tokens new tokens each, CUDA graphs:
  --spec off      plain decode
  --spec 5/1/6    MTP through SGLang's EAGLE path: --speculative-num-steps 5, --speculative-eagle-topk 1,
                  --speculative-num-draft-tokens 6 (the cookbook's low-latency configuration); 3/1/4 likewise
Per prompt length (median over --reps): the wall time of the bs requests' --tokens new tokens (the time of 1 new token
subtracted: the prefill), us per output token per request (the decode round: one verify + its draft chain with MTP), tokens/s over
the batch, and with MTP the mean accept length (meta_info spec_accept_length: tokens per verify step) and the verify steps per
request (spec_verify_ct). The prompts are cut to the token count with the model's tokenizer (filler words before the task).
With the static layers (DSA_STATIC_LAYERS, ...: boot/sitecustomize.py) the hooks replace the layers as the chain's other runs.

python sgl_spec_bench.py --bs 1 --spec 5/1/6 [--prompt-tokens 4000] [--tokens 256] [--reps 3] [--ctx 8448]
    [--kv bfloat16] [--max-total-tokens N] [--profile [--profile-steps 24] [--profile-tokens 96]]
(end-to-end comparisons: nsys step periods, --profile under nsys (sgl_chain.py prof_* runs, nsys_e2e.py), one run per
prompt length, 256 and 4000 tokens)
"""
import argparse
import statistics
import time

import sglang as sgl

MODEL = "/raid/catalyst/models/GLM-5.3-Flash"
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
    ap.add_argument("--spec", default="off", help="off, or steps/topk/draft tokens, e.g. 5/1/6")
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
    if args.spec != "off":
        steps, topk, draft = (int(x) for x in args.spec.split("/"))
        extra.update(speculative_algorithm="EAGLE", speculative_num_steps=steps, speculative_eagle_topk=topk,
                     speculative_num_draft_tokens=draft, enable_linear_replayssm_spec=True)   # KDA: the ReplaySSM rings
                                                                                              # (the static layer's contract)
    llm = sgl.Engine(model_path=MODEL, tp_size=4, max_running_requests=args.bs, cuda_graph_max_bs_decode=args.bs,
                     context_length=args.ctx, mem_fraction_static=0.6, dsa_prefill_backend="trtllm",
                     dsa_decode_backend="trtllm", kv_cache_dtype=args.kv, moe_runner_backend="flashinfer_trtllm",
                     disable_radix_cache=True, log_level="warning", **extra)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    sp = {"max_new_tokens": args.tokens, "ignore_eos": True, "temperature": 0.0}
    for n in args.prompt_tokens:
        prompts = prompts_of(tok, n, args.bs)
        out = llm.generate(prompts, sp)   # warm-up (the CUDA graphs of this batch size)
        times, acc, vct = [], [], []
        for _ in range(args.reps):
            t0 = time.perf_counter()
            llm.generate(prompts, {**sp, "max_new_tokens": 1})
            t1 = time.perf_counter()
            out = llm.generate(prompts, sp)
            t2 = time.perf_counter()
            times.append((t2 - t1) - (t1 - t0))
            for o in out:
                m = o["meta_info"]
                if "spec_accept_length" in m:
                    acc.append(float(m["spec_accept_length"]))
                if "spec_verify_ct" in m:
                    vct.append(float(m["spec_verify_ct"]))
        wall = statistics.median(times)
        got = out[0]["meta_info"]["prompt_tokens"]
        line = (f"bs {args.bs} spec {args.spec} prompt {got} tokens ({n} asked), {args.tokens} new: wall {wall * 1e6:.1f} us "
                f"(reps {', '.join(f'{t * 1e6:.1f}' for t in times)}), {wall / (args.tokens - 1) * 1e6:.1f} us per output token "
                f"per request, {args.bs * (args.tokens - 1) / wall:.1f} tok/s")
        if acc:
            line += (f" | accept length mean {statistics.mean(acc):.2f} (min {min(acc):.2f} max {max(acc):.2f}), verify steps "
                     f"per request {statistics.mean(vct) if vct else float('nan'):.1f}, "
                     f"{wall / statistics.mean(vct) * 1e6 if vct else float('nan'):.1f} us per verify step")
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
