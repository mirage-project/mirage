"""GSM8K (few-shot, greedy) on GLM-5.3-Flash through an SGLang Engine, TP4, as sglang.test.few_shot_gsm8k scores it (its prompt
builder and answer parsing; 5 shots, 200 questions): the accuracy of SGLang alone vs the static layers (the hooks through
boot/sitecustomize.py), with or without MTP (--spec as sgl_spec_bench.py). The questions run --bs at a time (the batch size
only sets the concurrency; the answers are greedy). --data-path: GSM8K's test.jsonl (the openai/grade-school-math file
few_shot_gsm8k downloads; no download here).

python sgl_gsm8k.py --data-path gsm8k_test.jsonl [--spec 5/1/6] [--bs 4] [--num-questions 200]
    [--max-total-tokens N]
"""
import argparse
import statistics
import time

import sglang as sgl
from sglang.test.few_shot_gsm8k import get_answer_value, get_few_shot_examples, get_one_example
from sglang.utils import read_jsonl

MODEL = "/raid/catalyst/models/GLM-5.3-Flash"
INVALID = -9999999


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-path", required=True)
    ap.add_argument("--spec", default="off")
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--num-questions", type=int, default=200)
    ap.add_argument("--num-shots", type=int, default=5)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--ctx", type=int, default=8448)
    ap.add_argument("--kv", default="bfloat16")
    ap.add_argument("--max-total-tokens", type=int, default=None)
    args = ap.parse_args()
    extra = {} if args.max_total_tokens is None else {"max_total_tokens": args.max_total_tokens}
    if args.spec != "off":
        steps, topk, draft = (int(x) for x in args.spec.split("/"))
        extra.update(speculative_algorithm="EAGLE", speculative_num_steps=steps, speculative_eagle_topk=topk,
                     speculative_num_draft_tokens=draft, enable_linear_replayssm_spec=True)   # KDA: the ReplaySSM rings
                                                                                              # (the static layer's contract)
    lines = list(read_jsonl(args.data_path))
    shots = get_few_shot_examples(lines, args.num_shots)
    questions = [shots + get_one_example(lines, i, False) for i in range(args.num_questions)]
    labels = [get_answer_value(lines[i]["answer"]) for i in range(args.num_questions)]
    assert all(l != INVALID for l in labels)
    llm = sgl.Engine(model_path=MODEL, tp_size=4, max_running_requests=args.bs, cuda_graph_max_bs_decode=args.bs,
                     context_length=args.ctx, mem_fraction_static=0.6, dsa_prefill_backend="trtllm",
                     dsa_decode_backend="trtllm", kv_cache_dtype=args.kv, moe_runner_backend="flashinfer_trtllm",
                     disable_radix_cache=True, log_level="warning", **extra)
    sp = {"max_new_tokens": args.max_new_tokens, "temperature": 0.0, "stop": ["Question", "Assistant:", "<|separator|>"]}
    t0 = time.perf_counter()
    outs = llm.generate(questions, sp)
    dt = time.perf_counter() - t0
    preds = [get_answer_value(o["text"]) for o in outs]
    correct = sum(p == l for p, l in zip(preds, labels))
    invalid = sum(p == INVALID for p in preds)
    acc = [float(o["meta_info"]["spec_accept_length"]) for o in outs if "spec_accept_length" in o["meta_info"]]
    toks = sum(o["meta_info"]["completion_tokens"] for o in outs)
    print(f"gsm8k spec {args.spec} bs {args.bs}: accuracy {correct / args.num_questions:.3f} ({correct}/{args.num_questions}, "
          f"invalid {invalid}), {toks} output tokens in {dt:.1f} s ({toks / dt:.1f} tok/s)"
          + (f", accept length mean {statistics.mean(acc):.2f}" if acc else ""), flush=True)
    llm.shutdown()


if __name__ == "__main__":
    main()
