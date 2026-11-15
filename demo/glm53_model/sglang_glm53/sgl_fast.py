"""GLM-5.3 decode timing and GSM8K for several batch sizes in ONE SGLang engine (sgl_bench.py / sgl_gsm8k.py start an
engine per batch size; a GLM-5.3 TP8 start loads 756 GB and captures the CUDA graphs, 15-20 min, so a bs sweep was mostly
engine starts). The engine runs at max_running_requests = cuda_graph_max_bs_decode = max(--bs); per bs in --bs order:
one warm-up generate of bs prompts, then --profile-steps forward steps under the CUDA profiler API (run under nsys
--capture-range=cudaProfilerApi --capture-range-end=repeat:<number of bs> --cuda-graph-trace=graph: one report per bs,
in --bs order; demo/glm53_flash/sglang_dsa/nsys_e2e.py --one <report>.sqlite gives the decode graph's period). A
"profile window: bs <b>, utc_ns <start> <end>" line per bs lets the caller check each report against its window.
Then, with --gsm8k-data, GSM8K (sglang.test.few_shot_gsm8k's prompts and scoring, 5 shots, --gsm8k-questions, greedy)
with all questions submitted at once, so at most max(--bs) run at a time and the batch shrinks to 1 as they finish.
Only decode is compared: prompts of --prompt-tokens tokens, no wall-clock timing.
The static megakernel decode replaces SGLang's when the environment selects the plugin (as sgl_chain.sh's PLUGIN):
  PYTHONPATH=<repo>/python:<repo>:<repo>/demo/glm53_model SGLANG_EXTERNAL_MODEL_PACKAGE=sglang_glm53 STATIC_TOKENS=<T>
A decode step of more than T rows runs SGLang's layers, so an ours engine takes only bs <= T (e.g. T 8: --bs 1 2 4 8).

python sgl_fast.py --model <GLM-5.3 snapshot> --bs 1 2 4 8 [--tp 8] [--prompt-tokens 256] [--profile-steps 64]
    [--ctx 8448] [--mem-fraction 0.8] [--port P] [--gsm8k-data test.jsonl [--gsm8k-questions 200]]
"""
import argparse
import os
import sys
import time

import sglang as sgl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sgl_bench import prompts_of  # noqa: E402

INVALID = -9999999


def gsm8k(llm, path: str, n: int, shots_n: int, label: str):
    from sglang.test.few_shot_gsm8k import get_answer_value, get_few_shot_examples, get_one_example
    from sglang.utils import read_jsonl
    lines = list(read_jsonl(path))
    shots = get_few_shot_examples(lines, shots_n)
    questions = [shots + get_one_example(lines, i, False) for i in range(n)]
    labels = [get_answer_value(lines[i]["answer"]) for i in range(n)]
    sp = {"max_new_tokens": 512, "temperature": 0.0, "stop": ["Question", "Assistant:", "<|separator|>"]}
    t0 = time.perf_counter()
    outs = llm.generate(questions, sp)
    dt = time.perf_counter() - t0
    preds = [get_answer_value(o["text"]) for o in outs]
    correct = sum(p == l for p, l in zip(preds, labels))
    invalid = sum(p == INVALID for p in preds)
    print(f"gsm8k {label}: accuracy {correct / n:.3f} ({correct}/{n}, invalid {invalid}), {dt:.1f} s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tp", type=int, default=8)
    ap.add_argument("--bs", type=int, nargs="+", required=True)
    ap.add_argument("--prompt-tokens", type=int, default=256)
    ap.add_argument("--profile-steps", type=int, default=64)
    ap.add_argument("--profile-tokens", type=int, default=256)
    ap.add_argument("--ctx", type=int, default=8448)
    ap.add_argument("--mem-fraction", type=float, default=0.8)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--gsm8k-data", default=None)
    ap.add_argument("--gsm8k-questions", type=int, default=200)
    ap.add_argument("--gsm8k-shots", type=int, default=5)
    ap.add_argument("--no-profile", action="store_true", help="GSM8K only")
    args = ap.parse_args()
    mx = max(args.bs)
    ports = {} if args.port is None else {"port": args.port, "nccl_port": args.port + 1}
    llm = sgl.Engine(model_path=args.model, tp_size=args.tp, max_running_requests=mx, cuda_graph_max_bs_decode=mx,
                     context_length=args.ctx, kv_cache_dtype="bfloat16", moe_runner_backend="flashinfer_trtllm",
                     disable_radix_cache=True, cuda_graph_backend_prefill="disabled", mem_fraction_static=args.mem_fraction,
                     log_level="warning", **ports)
    print(f"engine ready: max bs {mx}, STATIC_TOKENS {os.environ.get('STATIC_TOKENS', '-')}, "
          f"plugin {os.environ.get('SGLANG_EXTERNAL_MODEL_PACKAGE', 'off')}", flush=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    sp = {"max_new_tokens": args.profile_tokens, "ignore_eos": True, "temperature": 0.0}
    for bs in ([] if args.no_profile else args.bs):
        prompts = prompts_of(tok, args.prompt_tokens, bs)
        out = llm.generate(prompts, {**sp, "max_new_tokens": 16})   # warm-up at this batch size
        print(f"bs {bs}: output 0 {out[0]['text'][:120]!r}", flush=True)
        print(f"profile: bs {bs}, prompt {args.prompt_tokens} tokens, {args.profile_steps} steps", flush=True)
        t0 = time.time_ns()
        llm.start_profile(activities=["CUDA_PROFILER"], num_steps=args.profile_steps)
        llm.generate(prompts, sp)
        try:
            llm.stop_profile()
        except Exception as e:   # the profiler stops itself after num_steps
            print("stop_profile:", e, flush=True)
        print(f"profile window: bs {bs}, utc_ns {t0} {time.time_ns()}", flush=True)
    if args.gsm8k_data:
        gsm8k(llm, args.gsm8k_data, args.gsm8k_questions, args.gsm8k_shots,
              f"max bs {mx} T {os.environ.get('STATIC_TOKENS', '-')} plugin "
              f"{os.environ.get('SGLANG_EXTERNAL_MODEL_PACKAGE', 'off')}")
    llm.shutdown()


if __name__ == "__main__":
    main()
