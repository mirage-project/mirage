"""8 real prompts through SGLang, TP4: different questions of different lengths, so the 8 rows of a step route to different
experts and the requests end when they end (with --same-length-copies: the first prompt 8 times, ignore_eos, as a layer that keeps
its own caches needs). A decode step of 8 rows runs the static layers; once a request has ended the others go on on SGLang's
layers, on the same pools. Without DSV41_STATIC_*: SGLang alone, the reference answers. Greedy.

python sgl_generate.py --model DIR [--prompt "..."] [--tokens 96] [--graph] [--override JSON] [--out FILE]
"""
import argparse
import json
import os
import time

import sglang as sgl
from transformers import AutoTokenizer


PROMPTS = ["please explain multihead attention",
           "What is the capital of Australia, and why was it chosen over Sydney and Melbourne?",
           "Write a Python function that returns the n-th Fibonacci number.",
           "A train leaves at 3 pm at 60 km/h and another follows at 4 pm at 80 km/h. When does the second catch up?",
           "Name three causes of the French Revolution.",
           "Explain how a hash table handles collisions.",
           "Translate 'good morning, how are you?' into Spanish, French and German.",
           "Why is the sky blue? Answer in two sentences."]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--template", default=None, help="a chat_template.jinja (the checkpoint's if it has one)")
    ap.add_argument("--same-length-copies", action="store_true", help="the first prompt 8 times, ignore_eos")
    ap.add_argument("--tokens", type=int, default=96)
    ap.add_argument("--ignore-eos", action="store_true", help="every request runs all --tokens steps (every decode step has 8 rows)")
    ap.add_argument("--repeat", type=int, default=1, help="the prompt text this many times (a longer context)")
    ap.add_argument("--graph", action="store_true", help="CUDA graphs (default: eager)")
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--max-total-tokens", type=int, default=65536)
    ap.add_argument("--mem-fraction", type=float, default=0.7)
    ap.add_argument("--override", default=None, help="--json-model-override-args")
    ap.add_argument("--load-format", default=None)
    ap.add_argument("--no-bounded-replay", action="store_true", help="a truncated model: --no-enable-decoder-swa-bounded-replay")
    ap.add_argument("--no-autotune", action="store_true", help="--disable-flashinfer-autotune: about 10 min less at startup")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.tokenizer or a.model, trust_remote_code=True)
    if a.template:
        tok.chat_template = open(a.template).read()
    prompts = [PROMPTS[0]] * 8 if a.same_length_copies else PROMPTS
    texts = [tok.apply_chat_template([{"role": "user", "content": " ".join([q] * a.repeat)}], tokenize=False,
                                     add_generation_prompt=True, enable_thinking=False) for q in prompts]
    ids = [tok.encode(t, add_special_tokens=False) for t in texts]
    print(f"prompts {[len(i) for i in ids]} tokens", flush=True)
    extra = {}
    if a.override:
        extra["json_model_override_args"] = a.override
    if a.load_format:
        extra["load_format"] = a.load_format
    if a.no_autotune:
        extra["disable_flashinfer_autotune"] = True
    if a.no_bounded_replay:
        extra["enable_decoder_swa_bounded_replay"] = False
    llm = sgl.Engine(model_path=a.model, tokenizer_path=a.tokenizer or a.model, trust_remote_code=True, tp_size=int(os.environ.get("SGL_TP", "4")),
                     max_running_requests=8, cuda_graph_max_bs_decode=8, context_length=a.ctx, max_total_tokens=a.max_total_tokens,
                     mem_fraction_static=a.mem_fraction, disable_radix_cache=True, disable_cuda_graph=not a.graph,
                     log_level="warning", **extra)
    t0 = time.perf_counter()
    out = llm.generate(input_ids=ids, sampling_params={"max_new_tokens": a.tokens, "ignore_eos": a.same_length_copies or a.ignore_eos,
                                                       "temperature": 0.0})
    dt = time.perf_counter() - t0
    llm.shutdown()
    answers = []
    for o in out:
        s = o["text"]
        cut = s.find(tok.eos_token) if tok.eos_token else -1
        answers.append(s[:cut] if cut >= 0 and not a.ignore_eos else s)
    print(f"up to {a.tokens} new tokens in {dt:.1f} s; distinct answers among the 8: {len(set(answers))}", flush=True)
    for q, s in zip(prompts, answers):
        print("ANSWER:", repr(q[:40]), "->", repr(s[:200]), flush=True)
    if a.out:
        json.dump({"prompts": prompts, "ids": ids, "answers": answers, "static": {k: v for k, v in os.environ.items()
                                                                                    if k.startswith("DSV41_STATIC")}},
                  open(a.out, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
