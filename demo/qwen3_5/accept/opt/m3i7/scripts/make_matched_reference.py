#!/usr/bin/env python3
"""M3-I7 -- materialise the PINNED 256/1024 benchmark prompts in the shape
``mpk_engine_run.py --reference`` consumes.

Why this exists (a real defect found at the milestone gate):
``mpk_engine_run.py --prompts-file`` is NOT a prompt source -- it is only read
under ``--verify-chat-template`` (mpk_engine_run.py:678). M3-I9's stage 7
passed ``--prompts-file synthetic256.jsonl`` alongside ``--reference
reference_outputs.json`` and therefore measured the **AC-3 reference prompts
(24-68 tokens)** at msl=1280, not the 256-token benchmark prompts. The
committed evidence shows it directly: ``opt/m3i9/results/window2/out/
s7_base_bs1_rep1/timings_bs1.json`` records ``prompt_ids: ['p06-poem']`` and
``max_decode_steps: 1255`` (= 1280 - 24 - 1), while ``analyze_m3i9.matched``
divided by a hardcoded 1024. So the "matched 256/1024" row in the ledger is
neither matched in prompt length nor consistent in its token count.

The only prompt source ``mpk_engine_run.py`` honours is ``--reference``
(``load_reference_requests``: ``{"results": {pid: {"input_ids": [...]}}}``),
so this writes exactly that, with ids drawn by the pinned baseline sampler.

Sampler: identical to ``opt/m3i9/make_synthetic_prompts.build`` -- which is
itself the checked reproduction of ``bench_vllm.py:build_synthetic_prompts``,
the pinned source of the vLLM baseline's prompts (bench-protocol.md 2):
``batch_size`` distinct sequences of exactly ``input_len`` ids drawn from
``random.Random(seed_base + batch_size*1000 + rep)`` over
``[0, tokenizer.vocab_size)``. We import that module and assert equality rather
than restate the five lines, so the two can never drift apart.

Usage:
    python3 make_matched_reference.py --batch-size 8 --rep 0 \
        --out /path/synthref_bs8_rep0.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# .../accept/opt/m3i7/scripts -> .../accept/opt/m3i9/make_synthetic_prompts.py
I9 = HERE.parent.parent / "m3i9" / "make_synthetic_prompts.py"


def _load_i9():
    spec = importlib.util.spec_from_file_location("m3i9_make_synth", I9)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(argv=None) -> int:
    i9 = _load_i9()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch-size", type=int, required=True)
    ap.add_argument("--input-len", type=int, default=256)
    ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--seed-base", type=int, default=i9.SEED_BASE)
    ap.add_argument("--vocab-n", type=int, default=i9.VOCAB_N)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    seed = a.seed_base + a.batch_size * 1000 + a.rep
    ids = i9.build(a.batch_size, a.input_len, seed, a.vocab_n)
    assert len(ids) == a.batch_size and all(len(r) == a.input_len for r in ids)

    doc = {
        "provenance": {
            "generator": "opt/m3i7/scripts/make_matched_reference.py",
            "sampler": "opt/m3i9/make_synthetic_prompts.build "
                       "(== bench_vllm.build_synthetic_prompts, bench-protocol.md 2)",
            "seed": seed, "seed_base": a.seed_base, "vocab_n": a.vocab_n,
            "batch_size": a.batch_size, "input_len": a.input_len, "rep": a.rep,
            "note": "synthetic ids -- there is no HF reference completion for "
                    "these; this file is a PROMPT SOURCE for perf runs only and "
                    "must never be handed to run_ac3.py.",
        },
        "results": {f"syn{a.input_len}-{i:02d}": {"input_ids": row}
                    for i, row in enumerate(ids)},
    }
    Path(a.out).write_text(json.dumps(doc) + "\n")
    print(f"wrote {a.batch_size} x {a.input_len} ids (seed={seed}) -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
