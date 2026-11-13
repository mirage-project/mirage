"""GSM8K (few-shot, greedy) on GPT-OSS: this demo's static kernels against SGLang and vLLM, scored as the GLM-5.3-Flash
demo scores its static layers (glm53_flash/sglang_dsa/sgl_gsm8k.py): sglang.test.few_shot_gsm8k's prompts (the first
--num-shots questions with their answers, then each of the first --num-questions questions) and answer parsing (the last
number of the output). Every engine gets the same token ids (the checkpoint's tokenizer; --format below), decodes
greedily --max-new-tokens tokens, and its output is cut here the same way: at the first end token (generation_config's
eos_token_id), then at the first of few_shot_gsm8k's stop strings (greedy tokens before a stop do not depend on it, so this
is the engines' own stop). --json: per question the prompt, the generated ids, the text and the parsed answer, so that two
engines' outputs can be compared token by token.

--format raw: few_shot_gsm8k's text as is (as the GLM demo; GPT-OSS, a chat model, mostly degenerates on it: about 40 % on
both the static kernels and SGLang, their outputs apart after a few tokens). --format chat (default): the same text as the
user message of the checkpoint's chat template (reasoning effort --effort), no stop strings, the answer the last number of
the final channel (none: invalid).

  static   the static kernels (gpt_oss_static.py), --bs requests a step; the prompts of a step have different lengths
           (sm90_next_token: a request generates from its own length), --max-seq covers the longest prompt + new tokens
  sglang   sglang.Engine, one GPU, at most --bs running requests (radix cache off)
  vllm     vllm.LLM, one GPU, at most --bs sequences (prefix caching off)

python gsm8k.py CKPT --engine static|sglang|vllm --data-path test.jsonl [--bs 32] [--num-questions 200] [--json OUT]
--data-path: GSM8K's test.jsonl (openai/grade-school-math grade_school_math/data/test.jsonl, as few_shot_gsm8k downloads).
"""
import argparse, ast, json, os, re, time

INVALID = -9999999
STOP = ["Question", "Assistant:", "<|separator|>"]   # few_shot_gsm8k's


def one_example(lines, i, include_answer):            # few_shot_gsm8k.get_one_example
    return "Question: " + lines[i]["question"] + "\nAnswer:" + (" " + lines[i]["answer"] if include_answer else "")


def answer_value(text):                               # few_shot_gsm8k.get_answer_value
    numbers = re.findall(r"\d+", text.replace(",", ""))
    if not numbers:
        return INVALID
    try:
        return ast.literal_eval(numbers[-1])
    except SyntaxError:
        return INVALID


FINAL = "<|channel|>final<|message|>"                # chat: the answer's channel


def cut(tok, ids, eos, chat):
    """The output text: the ids to the first end token, decoded, then raw: to the first stop string; chat: the final
    channel's text ("" if the output has none)."""
    ids = next((ids[:i] for i, t in enumerate(ids) if t in eos), ids)
    text = tok.decode(ids)
    if chat:
        return text.rsplit(FINAL, 1)[1] if FINAL in text else ""
    return min((text[:text.index(s)] for s in STOP if s in text), key=len, default=text)


def run_static(args, prompts, new):
    """The prompts --bs at a time through the static kernels; the generated ids of each."""
    import torch
    import gpt_oss_static as g
    R = args.bs
    g.T = next(b for b in g.BUCKETS if b >= R)
    P = max(len(p) for p in prompts)
    assert P + new <= args.max_seq, f"--max-seq {args.max_seq}: the longest prompt ({P}) + {new} new tokens"
    m = g.Model(args.ckpt, args.max_seq, R)
    prompt_t = torch.full((R, P), -1, dtype=torch.long, device="cuda")   # past a prompt's length: generate (-1)
    gen = torch.zeros(R, new, dtype=torch.long, device="cuda")
    sliding = [i for i in range(m.L) if m.types[i] == "sliding_attention"]
    full = [i for i in range(m.L) if m.types[i] != "sliding_attention"]
    kinds = {"sliding": (lambda mpk, t: g.build_layer(m, mpk, t, m.window), m.layer_tensors(sliding[0])),
             "full": (lambda mpk, t: g.build_layer(m, mpk, t, 0), m.layer_tensors(full[0])),
             "head": (lambda mpk, t: g.build_head(m, mpk, t), m.head_tensors(prompt_t, gen))}
    plans = args.plans or None
    k_sl, k_fu, k_hd = (g.compile_kernel(k, b, t, args.out, plans and g.plan_path(m, plans, k))
                        for k, (b, t) in kinds.items())
    kernel_of, inst = {}, {}
    for i in range(m.L):
        kernel_of[i] = k_sl if i in sliding else k_fu
        inst[i] = kernel_of[i].add_instance(m.layer_tensors(i))
    k_hd.add_instance(m.head_tensors(prompt_t, gen))
    out = []
    for b in range(0, len(prompts), R):
        batch = prompts[b:b + R]
        n = len(batch)
        batch = batch + [batch[-1]] * (R - n)          # a short last batch: its last prompt again (dropped)
        prompt_t.fill_(-1)
        for r, p in enumerate(batch):
            prompt_t[r, :len(p)] = torch.tensor(p, device="cuda")
        gen.zero_()
        for layer in m.layers:
            layer["k_cache"].zero_()
            layer["v_cache"].zero_()
        m.set_position(0)
        m.x_embed[:R].copy_(m.embed[prompt_t[:, 0]])
        steps = max(len(p) for p in batch) + new - 1   # every request's new tokens (the shortest prompt: more)
        t0 = time.perf_counter()
        for _ in range(steps):
            for i in range(m.L):
                kernel_of[i].launch_instances(inst[i], 1)
            k_hd.launch_instances(0, 1)
        torch.cuda.synchronize()
        toks = torch.cat([prompt_t, gen], 1).tolist()
        out += [toks[r][len(batch[r]):len(batch[r]) + new] for r in range(n)]
        print(f"static: questions {b} .. {b + n - 1}: {steps} steps in {time.perf_counter() - t0:.1f} s", flush=True)
    for k in (k_sl, k_fu, k_hd):
        k.finalize()
    return out


def run_sglang(args, prompts, new):
    import sglang as sgl
    eng = sgl.Engine(model_path=args.ckpt, context_length=args.max_seq, mem_fraction_static=args.mem,
                     max_running_requests=args.bs, disable_radix_cache=True, log_level="error")
    outs = eng.generate(input_ids=prompts, sampling_params={"max_new_tokens": new, "temperature": 0, "ignore_eos": True})
    eng.shutdown()
    return [o["output_ids"] for o in outs]


def run_vllm(args, prompts, new):
    from vllm import LLM, SamplingParams
    llm = LLM(args.ckpt, max_model_len=args.max_seq, gpu_memory_utilization=args.mem, max_num_seqs=args.bs,
              enable_prefix_caching=False)
    outs = llm.generate([{"prompt_token_ids": p} for p in prompts],
                        SamplingParams(max_tokens=new, temperature=0, ignore_eos=True), use_tqdm=False)
    return [list(o.outputs[0].token_ids) for o in outs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--engine", choices=("static", "sglang", "vllm"), required=True)
    ap.add_argument("--data-path", required=True)
    ap.add_argument("--num-questions", type=int, default=200)
    ap.add_argument("--num-shots", type=int, default=5)
    ap.add_argument("--format", choices=("chat", "raw"), default="chat")
    ap.add_argument("--effort", default="low", help="chat: the reasoning effort")
    ap.add_argument("--max-new-tokens", type=int, default=None, help="default chat 512, raw 256")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--max-seq", type=int, default=2048)
    ap.add_argument("--mem", type=float, default=0.85, help="sglang / vllm: the GPU memory fraction")
    ap.add_argument("--plans", default=None, help="static: the plan files' directory (default: round robin)")
    ap.add_argument("--out", default="gsm8k_static", help="static: the build directory")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    from transformers import AutoTokenizer, GenerationConfig
    tok = AutoTokenizer.from_pretrained(args.ckpt)
    eos = GenerationConfig.from_pretrained(args.ckpt).eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    lines = [json.loads(l) for l in open(args.data_path)]
    shots = "".join(one_example(lines, i, True) + "\n\n" for i in range(args.num_shots))
    questions = [shots + one_example(lines, i, False) for i in range(args.num_questions)]
    labels = [answer_value(lines[i]["answer"]) for i in range(args.num_questions)]
    assert INVALID not in labels
    chat = args.format == "chat"
    if args.max_new_tokens is None:
        args.max_new_tokens = 512 if chat else 256
    if chat:   # (the template stamps today's date into the system message: --json keeps the ids)
        prompts = [tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True,
                                           reasoning_effort=args.effort) for q in questions]
        prompts = [list(p["input_ids"] if hasattr(p, "keys") else p) for p in prompts]   # (a BatchEncoding in some versions)
    else:
        prompts = [tok(q).input_ids for q in questions]
    t0 = time.perf_counter()
    gen = {"static": run_static, "sglang": run_sglang, "vllm": run_vllm}[args.engine](args, prompts, args.max_new_tokens)
    dt = time.perf_counter() - t0
    texts = [cut(tok, ids, eos, chat) for ids in gen]
    preds = [answer_value(t) for t in texts]
    correct = sum(p == l for p, l in zip(preds, labels))
    invalid = sum(p == INVALID for p in preds)
    name = os.path.basename(os.path.normpath(args.ckpt))
    print(f"gsm8k {args.format} {args.engine} {name} bs {args.bs}: accuracy {correct / len(labels):.3f} ({correct}/{len(labels)}, "
          f"invalid {invalid}) in {dt:.0f} s", flush=True)
    if args.json:
        json.dump({"engine": args.engine, "format": args.format, "ckpt": name, "accuracy": correct / len(labels),
                   "questions": [{"prompt_ids": p, "gen": o, "text": t, "pred": pr, "label": l}
                                 for p, o, t, pr, l in zip(prompts, gen, texts, preds, labels)]},
                  open(args.json, "w"))


if __name__ == "__main__":
    main()
