#!/usr/bin/env python3
"""AC-3 non-regression spot check for the M3-I10 re-measure.

Only arm B (msl=132, the actual AC-3 prompt set) has a ground truth to check
against -- arm A's prompts are synthetic random token ids (matched-geometry,
performance only; bench_vllm.py's build_synthetic_prompts docstring: "content
carries no signal, only length does"), so there is no reference continuation
to compare arm A's output to. This checks that CURRENT HEAD (MOE_GATE_PADDING_
ROWS=True, post-M3-I2b) still reproduces the committed reference byte-for-byte
at the geometry the reference was generated at (msl=132, 64 new tokens) --
i.e. that the performance numbers in this re-measure come from a
correct engine, not a silently-regressed one.

Usage: python3 ac3_spotcheck.py --tokens-dir prof_B --reference reference_outputs.json --out out.json
"""
import argparse
import glob
import json
import re
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens-dir", required=True,
                    help="dir with tokens_bs<N>_rep<R>.json (arm B)")
    ap.add_argument("--reference", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ref = json.load(open(args.reference))["results"]
    files = sorted(glob.glob(str(Path(args.tokens_dir) / "tokens_bs*_rep*.json")))
    if not files:
        raise SystemExit(f"no tokens_bs*_rep*.json under {args.tokens_dir}")

    rows = []
    n_checked = 0
    n_match = 0
    for fp in files:
        m = re.search(r"tokens_bs(\d+)_rep(\d+)\.json", fp)
        bs, rep = int(m.group(1)), int(m.group(2))
        got = json.load(open(fp))
        for pid, ids in got.items():
            if pid not in ref:
                rows.append(dict(bs=bs, rep=rep, prompt_id=pid,
                                 status="NO_REFERENCE", first_divergence=None))
                continue
            exp = ref[pid]["output_ids"]
            n_checked += 1
            n = min(len(ids), len(exp))
            first_div = next((k for k in range(n) if ids[k] != exp[k]), None)
            full_match = (first_div is None) and (len(ids) == len(exp))
            if full_match:
                n_match += 1
            rows.append(dict(
                bs=bs, rep=rep, prompt_id=pid,
                len_got=len(ids), len_reference=len(exp),
                overlap_positions=n,
                first_divergence=first_div,
                full_match=full_match,
                status="MATCH" if full_match else (
                    "DIVERGED" if first_div is not None else "LENGTH_MISMATCH_NO_DIVERGENCE"),
            ))

    out = dict(
        n_files=len(files), n_checked=n_checked, n_full_match=n_match,
        all_match=(n_checked > 0 and n_match == n_checked),
        rows=rows,
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"AC-3 spot check: {n_match}/{n_checked} full byte-matches "
          f"({len(files)} files). all_match={out['all_match']}")
    for r in rows:
        if r["status"] != "MATCH":
            print(f"  NON-MATCH bs={r['bs']} rep={r['rep']} {r['prompt_id']}: {r['status']} "
                  f"first_divergence={r.get('first_divergence')}")


if __name__ == "__main__":
    main()
