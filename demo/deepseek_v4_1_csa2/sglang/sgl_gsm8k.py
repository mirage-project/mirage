"""GSM8K (few-shot, greedy) on DeepSeek-V4.1-Flash through an SGLang Engine, TP4, as sglang.test.few_shot_gsm8k scores it (its prompt
builder and answer parsing; 5 shots, 200 questions): SGLang alone, or with the static layers (the hooks, through boot/sitecustomize.py).
The questions run --bs at a time; a decode step of 8 rows (with CUDA graphs: of 5 to 8, the graph of 8) runs the static layers.

python sgl_gsm8k.py --model DIR --data-path test.jsonl [--bs 8] [--num-questions 200] [--out FILE]
"""
import argparse
import json
import time

import sglang as sgl
from sglang.test.few_shot_gsm8k import get_answer_value, get_few_shot_examples, get_one_example
from sglang.utils import read_jsonl

INVALID = -9999999


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data-path", required=True)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--num-questions", type=int, default=200)
    ap.add_argument("--num-shots", type=int, default=5)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--max-total-tokens", type=int, default=65536)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    lines = list(read_jsonl(a.data_path))
    shots = get_few_shot_examples(lines, a.num_shots)
    questions = [shots + get_one_example(lines, i, False) for i in range(a.num_questions)]
    labels = [get_answer_value(lines[i]["answer"]) for i in range(a.num_questions)]
    assert all(l != INVALID for l in labels)
    llm = sgl.Engine(model_path=a.model, tokenizer_path=a.model, trust_remote_code=True, tp_size=4, max_running_requests=a.bs,
                     cuda_graph_max_bs_decode=a.bs, context_length=a.ctx, max_total_tokens=a.max_total_tokens,
                     mem_fraction_static=0.7, disable_radix_cache=True, disable_flashinfer_autotune=True, log_level="warning")
    sp = {"max_new_tokens": a.max_new_tokens, "temperature": 0.0, "stop": ["Question", "Assistant:", "<|separator|>"]}
    t0 = time.perf_counter()
    outs = llm.generate(questions, sp)
    dt = time.perf_counter() - t0
    llm.shutdown()
    preds = [get_answer_value(o["text"]) for o in outs]
    correct = sum(p == l for p, l in zip(preds, labels))
    invalid = sum(p == INVALID for p in preds)
    toks = sum(o["meta_info"]["completion_tokens"] for o in outs)
    print(f"gsm8k bs {a.bs}: accuracy {correct / a.num_questions:.3f} ({correct}/{a.num_questions}, invalid {invalid}), {toks} output "
          f"tokens in {dt:.1f} s", flush=True)
    if a.out:
        json.dump({"correct": correct, "n": a.num_questions, "preds": preds, "labels": labels, "texts": [o["text"] for o in outs]},
                  open(a.out, "w"), ensure_ascii=False)


if __name__ == "__main__":
    main()
