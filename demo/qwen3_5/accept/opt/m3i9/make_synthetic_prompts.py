#!/usr/bin/env python3
"""M3-I9 -- materialise the pinned benchmark prompts as token ids for MPK.

`bench_vllm.py:build_synthetic_prompts` is the pinned source of the vLLM
baseline's prompts (bench-protocol.md 2): `batch_size` distinct sequences of
exactly `input_len` ids sampled uniformly from `[0, tokenizer.vocab_size)` with
`seed = seed_base + batch_size*1000 + rep_index`. This reproduces that sampler
EXACTLY -- same `random.Random`, same call order -- and writes the ids in the
`{"id", "input_ids"}` form `mpk_engine_run.py --prompts-file` consumes.

Reproducing rather than importing is deliberate: `build_synthetic_prompts`
imports `vllm.TokensPrompt`, and MPK's venv is not vLLM's. The sampler itself is
five lines of stdlib `random`, so the reproduction is checkable by eye and is
asserted against the committed baseline artifact with `--verify`.

Only *length* carries fairness-relevant signal (bench-protocol.md 2), but using
the identical ids removes even the appearance of a difference, and it removes
the tokenizer from between the two engines entirely.
"""
from __future__ import annotations

import argparse
import json
import random
import sys

VOCAB_N = 248044          # tokenizer.vocab_size for this checkpoint (bench-protocol.md 2)
SEED_BASE = 20260726      # confirmed from every committed baseline rep's `seed`
                          # field: seed = SEED_BASE + batch_size*1000 + rep_index
                          # (bs1 20261726/7/8 ... bs16 20276726/7/8)


def build(batch_size: int, input_len: int, seed: int, vocab_n: int = VOCAB_N):
    rng = random.Random(seed)
    return [[rng.randrange(0, vocab_n) for _ in range(input_len)]
            for _ in range(batch_size)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--input-len", type=int, default=256)
    ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--seed-base", type=int, default=SEED_BASE)
    ap.add_argument("--vocab-n", type=int, default=VOCAB_N)
    ap.add_argument("--out", required=True)
    ap.add_argument("--verify", default=None,
                    help="a committed vLLM baseline bs<N>.json: cross-check the "
                         "seed formula against the per-rep `seed` it records. "
                         "The baseline does NOT persist prompt ids, so this "
                         "checks the seed, not the ids -- stated plainly rather "
                         "than implied.")
    a = ap.parse_args(argv)
    seed = a.seed_base + a.batch_size * 1000 + a.rep
    ids = build(a.batch_size, a.input_len, seed, a.vocab_n)
    if a.verify:
        base = json.load(open(a.verify))
        want = [r["seed"] for r in base.get("reps", []) if "seed" in r]
        if not want:
            print(f"cannot verify: {a.verify} records no per-rep seed", file=sys.stderr)
            return 2
        got = [a.seed_base + base["batch_size"] * 1000 + i for i in range(len(want))]
        if got != want:
            print(f"MISMATCH: seed formula gives {got}, baseline recorded {want}. The "
                  "prompts would not be the ones vLLM was measured on -- fix the "
                  "seed base, do not proceed.", file=sys.stderr)
            return 1
        if base["input_len"] != a.input_len:
            print(f"MISMATCH: baseline input_len={base['input_len']}, asked for "
                  f"{a.input_len}", file=sys.stderr)
            return 1
        print(f"verified against {a.verify}: seeds {want} and input_len "
              f"{a.input_len} match. NOTE the baseline does not persist prompt "
              f"ids, so this establishes the sampler INPUTS, not the ids.")
    with open(a.out, "w") as f:
        for i, row in enumerate(ids):
            f.write(json.dumps({"id": f"syn{a.input_len}-{i:02d}",
                                "input_ids": row}) + "\n")
    print(f"wrote {len(ids)} prompts x {a.input_len} ids (seed={seed}) -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
