"""Standalone greedy PyTorch reference decode for Qwen3-30B-A3B.

Produces the ground-truth token stream that the EAGLE3 speculative-decoding
megakernel output is compared against. EAGLE3 strict rejection sampling accepts a
draft token only when it equals the target model's greedy argmax, and the final
committed token of every iteration is the target's argmax; therefore the accepted
stream equals this plain greedy decode regardless of the number of draft steps.

The prompt, system message, chat template, tokenizer, dtype, and EOS policy here
MUST match demo/qwen3/demo_30B_A3B_eagle3.py exactly, or the comparison will
differ for reasons unrelated to the kernel.

Decoding is a manual argmax loop (NOT model.generate()) so there is no sampling
path and full determinism control: model.eval(), inference_mode, bf16, argmax.
"""

import argparse
import json
import os

import torch
from transformers import AutoTokenizer, Qwen3MoeForCausalLM

# Mirrors demo_30B_A3B_eagle3.py exactly.
SYSTEM_MESSAGE = (
    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
)
USER_PROMPT = "Give me a short introduction to large language model."
DEFAULT_MODEL = "Qwen/Qwen3-30B-A3B"
DEFAULT_OUT = os.path.join("outputs", "qwen3_30b_a3b", "torch_reference.json")
DEFAULT_NUM_TOKENS = 50


def build_prompt_ids(tokenizer, device):
    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": USER_PROMPT},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    enc = tokenizer([text], return_tensors="pt").to(device)
    return text, enc.input_ids


def eos_ids_from_config(config):
    """Return the set of EOS token ids, mirroring the demo's eos handling.

    The demo passes model.config.eos_token_id straight through to the kernel.
    Qwen3 configs may carry a single int or a list; honor either.
    """
    eos = config.eos_token_id
    if eos is None:
        return set()
    if isinstance(eos, (list, tuple)):
        return {int(e) for e in eos}
    return {int(eos)}


@torch.inference_mode()
def greedy_decode(model, input_ids, num_tokens, eos_ids):
    """Manual greedy (argmax) autoregressive decode with KV cache.

    Returns (generated_token_ids, top2_gaps) where top2_gaps[i] is the logit
    margin between the top-1 and top-2 candidates at generated position i. A near
    zero gap means a bf16 argmax tie at that position: the megakernel may pick the
    other co-equal token there (a different reduction order breaks the tie
    differently), which is NOT an error. The correctness test uses this margin to
    accept tie-driven single-token differences while still rejecting real
    divergences (where the gap is large).

    Stops early when an EOS id is produced, including that EOS token so the
    comparison length matches the kernel's committed stream.
    """
    generated = []
    top2_gaps = []
    top2_tokens = []
    past = None
    cur = input_ids
    for _ in range(num_tokens):
        out = model(input_ids=cur, use_cache=True, past_key_values=past)
        past = out.past_key_values
        logits = out.logits[:, -1, :].float()  # fp32 view of the bf16 logits
        top2 = torch.topk(logits[0], 2)
        gap = (top2.values[0] - top2.values[1]).item()
        next_token = logits.argmax(dim=-1)  # pure argmax, no sampling
        tok = int(next_token[0].item())
        generated.append(tok)
        top2_gaps.append(gap)
        # The runner-up token id, so the test can confirm a tie-driven mismatch
        # is the megakernel picking THIS co-equal alternate (not an unrelated
        # wrong token at a coincidentally-close position).
        top2_tokens.append([int(top2.indices[0].item()), int(top2.indices[1].item())])
        if tok in eos_ids:
            break
        cur = next_token.unsqueeze(-1)
    return generated, top2_gaps, top2_tokens


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--num-tokens", type=int, default=DEFAULT_NUM_TOKENS)
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()

    torch.set_default_dtype(torch.bfloat16)
    device = "cuda"

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = Qwen3MoeForCausalLM.from_pretrained(args.model).to(device)
    model.eval()

    prompt_text, prompt_ids = build_prompt_ids(tokenizer, device)
    eos_ids = eos_ids_from_config(model.config)

    generated, top2_gaps, top2_tokens = greedy_decode(
        model, prompt_ids, args.num_tokens, eos_ids)

    prompt_token_ids = prompt_ids[0].tolist()
    payload = {
        "token_ids": generated,
        "top2_logit_gaps": top2_gaps,
        "top2_token_ids": top2_tokens,
        "prompt_token_ids": prompt_token_ids,
        "prompt_text": prompt_text,
        "prompt_length": len(prompt_token_ids),
        "generate_length": len(generated),
        "num_tokens": args.num_tokens,
        "model": args.model,
        "dtype": "bfloat16",
        "eos_token_ids": sorted(eos_ids),
        "transformers": __import__("transformers").__version__,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "tokenizer": args.model,
        "mode": "torch_greedy",
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {len(generated)} generated tokens to {args.out}")
    print(f"token_ids[:10] = {generated[:10]}")
    decoded = tokenizer.decode(generated, skip_special_tokens=True)
    print(f"decoded: {decoded!r}")


if __name__ == "__main__":
    main()
