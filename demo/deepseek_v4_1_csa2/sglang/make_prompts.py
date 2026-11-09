"""The 8 text prompts of the whole-step comparison (python -m sglang.benchmark.one_batch --prompt-filename): request i is "Request i:", filler
words up to the token count, then question i of sgl_generate.PROMPTS.

python make_prompts.py --model DIR --tokens 4000 --out prompts.txt
"""
import argparse
import os
import sys

from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sgl_generate import PROMPTS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokens", type=int, default=4000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    out = []
    for i, task in enumerate(PROMPTS):
        make = lambda n: f"Request {i}: " + " ".join(f"item{j}" for j in range(n)) + " " + task
        lo, hi = 0, a.tokens
        while lo < hi:   # the filler word count whose prompt has at most tokens - 1 tokens
            mid = (lo + hi + 1) // 2
            if len(tok.encode(make(mid), add_special_tokens=False)) <= a.tokens - 1:
                lo = mid
            else:
                hi = mid - 1
        out.append(make(lo))
    with open(a.out, "w") as f:
        f.write("\n".join(out) + "\n")
    print("prompts, tokens:", [len(tok.encode(p)) for p in out])


if __name__ == "__main__":
    main()
