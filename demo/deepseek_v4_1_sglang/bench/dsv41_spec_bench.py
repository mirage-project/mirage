"""DeepSeek-V4.1-Flash decode on SGLang, TP4, one engine per run (a batch size): bs prompts of each --prompt-tokens length,
--tokens new tokens each, CUDA graphs, plain decode (--spec off; this model has no MTP).
Per prompt length (median over --reps): the wall time of the bs requests' --tokens new tokens (the time of 1 new token
subtracted: the prefill), us per output token per request and tokens/s over the batch. The prompts are cut to the token
count with the model's tokenizer (filler words before the task). With the static layers (DSV41_STATIC_*:
../boot/sitecustomize.py) the hook replaces SGLang's layers.

python dsv41_spec_bench.py --bs 8 --spec off [--prompt-tokens 4000] [--tokens 256] [--reps 3] [--ctx 8448]
    [--max-total-tokens N] [--profile [--profile-steps 24] [--profile-tokens 96]]
(the comparison against SGLang under nsys: README_bs8.md)
"""
import argparse
import os
import statistics
import time

import sglang as sgl

# (PR #819's demo/glm53_flash/sglang_dsa/sgl_spec_bench.py in upstream mpk, for DSv4.1-Flash: the model path, the engine
# arguments, plain decode only, a hash of every row's output after each generate call, and --span-file)
MODEL = os.environ["DSV41_MODEL"]   # the 40-layer checkpoint (SGLang's model path)
TASK = " Write a long story about the history of computing, chapter {i}."


def _row_hash(o: dict) -> str:
    """8 hex digits of the md5 of a request's whole output (its token ids when SGLang gives them, else its text)."""
    import hashlib
    ids = o.get("output_ids")
    data = (",".join(map(str, ids)) if ids else o.get("text", "")).encode()
    return hashlib.md5(data).hexdigest()[:8]


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
    ap.add_argument("--max-total-tokens", type=int, default=None)
    ap.add_argument("--profile", action="store_true", help="then --profile-steps forward steps under the CUDA profiler API "
                    "(nsys), the last prompt length's prompts with --profile-tokens new tokens")
    ap.add_argument("--profile-steps", type=int, default=24)
    ap.add_argument("--profile-tokens", type=int, default=96)
    ap.add_argument("--span-file", default="", help="DSv4.1 only, after everything else: per --span-steps value n one generate "
                    "call of n new tokens, then the label 'n<n>' written to this file: the static layers (DSV41_STATIC_PROF=the "
                    "same file) write their kernels' stamps of that call's last decode step at their next forward")
    ap.add_argument("--span-steps", default="9,13,17,25,33,41,49,57,63,128,256")
    args = ap.parse_args()
    extra = {} if args.max_total_tokens is None else {"max_total_tokens": args.max_total_tokens}
    assert args.spec == "off", "DSv4.1: plain decode only (no MTP in this model)"
    # DSv4.1 (the only changes against #819's script): no GLM backend arguments (dsa_prefill_backend, dsa_decode_backend,
    # kv_cache_dtype, moe_runner_backend: SGLang's own DSv4.1 defaults), trust_remote_code; DSV41_TP / DSV41_OVERRIDE /
    # DSV41_NO_SWA_REPLAY only for the dry run on the local 4-layer checkpoint
    if os.environ.get("DSV41_OVERRIDE"):
        extra["json_model_override_args"] = os.environ["DSV41_OVERRIDE"]
    if os.environ.get("DSV41_NO_SWA_REPLAY"):
        extra["enable_decoder_swa_bounded_replay"] = False
    # (DSV41_PAD_SMALL, the static layers: a batch below 8 rows replays the 8-row graph, padded: its slots and graph buffers)
    gbs = max(args.bs, 8) if os.environ.get("DSV41_PAD_SMALL", "0") != "0" else args.bs
    llm = sgl.Engine(model_path=MODEL, tp_size=int(os.environ.get("DSV41_TP", "4")), max_running_requests=gbs,
                     cuda_graph_max_bs_decode=gbs, context_length=args.ctx, mem_fraction_static=0.6,
                     trust_remote_code=True, disable_radix_cache=True, log_level="warning", **extra)
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
        if os.environ.get("DSV41_PRINT_ALL"):   # (DSv4.1 only: every request's first characters, for a look at all rows)
            for i, o in enumerate(out[1:], 1):
                print(f"output {i}:", repr(o["text"][:60]), flush=True)
            print("hashes:", " ".join(_row_hash(o) for o in out), flush=True)   # (every row's whole output: arms compare)
    if args.profile:
        prompts = prompts_of(tok, args.prompt_tokens[-1], args.bs)
        llm.start_profile(activities=["CUDA_PROFILER"], num_steps=args.profile_steps)
        pout = llm.generate(prompts, {**sp, "max_new_tokens": args.profile_tokens})
        # the profiled steps' outputs: two runs whose hashes are equal made the same tokens, so their traced decode steps
        # compare one by one (local/e2e_ab.sh, pair_ab.py)
        print("profiled hashes:", " ".join(_row_hash(o) for o in pout), flush=True)
        try:
            llm.stop_profile()
        except Exception as e:
            print("stop_profile:", e)
    if args.span_file:   # (after the measured part: nothing above changes)
        prompts = prompts_of(tok, args.prompt_tokens[-1], args.bs)
        for n in [int(v) for v in args.span_steps.split(",")]:
            llm.generate(prompts, {**sp, "max_new_tokens": n})
            with open(args.span_file, "w") as f:
                f.write(f"n{n}\n")
            time.sleep(0.05)
            llm.generate(prompts, {**sp, "max_new_tokens": 1})   # its first forward takes the sample (no decode step in it)
        print(f"span samples asked: {args.span_steps} -> {args.span_file}.r<rank>.jsonl", flush=True)
    llm.shutdown()


if __name__ == "__main__":
    main()
