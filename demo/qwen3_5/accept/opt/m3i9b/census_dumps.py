#!/usr/bin/env python3
"""M3-I9b: census every committed engine token dump for divergence signatures.

The M3-I9 window's AC-3 report for the cap arm was built from ONE bs4 dump
(`s4_cap4/bs4.json`).  This walks every `bs<N>.json` under a results tree,
records each dump's divergence set against the reference, md5-groups the dumps,
and cross-tabulates base-vs-cap and rep-vs-rep so a "the cap changed the tokens"
claim can be separated from "one run differed from the other nine".

    python census_dumps.py --root .../results/window2/out --out census.json
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import re
from pathlib import Path

STD = ["p06-poem", 60, 40581]     # the pre-existing MPK-vs-HF signature at the
                                  # reference's own exact tie (margin 0.0)


def divergences(dump: dict, ref: dict):
    out = []
    for pid, v in sorted(dump.items()):
        r = ref[pid]["output_ids"]
        g = v["token_ids"]
        d = next((i for i, (a, b) in enumerate(zip(r, g)) if a != b), None)
        if d is not None:
            out.append([pid, d, g[d]])
    return out


def first_div(a: dict, b: dict):
    best = None
    for k in a:
        if k not in b:
            continue
        x, y = a[k]["token_ids"], b[k]["token_ids"]
        d = next((i for i, (u, v) in enumerate(zip(x, y)) if u != v), None)
        if d is not None and (best is None or d < best[1]):
            best = [k, d]
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ref = json.load(open(a.reference))["results"]
    root = Path(a.root)

    dumps, rep = {}, {"root": str(root), "dumps": {}, "anomalies": [],
                      "md5_groups": {}, "rep_matrix": {}}
    for p in sorted(root.glob("*/bs*.json")):
        if "timings" in p.name:
            continue
        try:
            d = json.load(open(p))
        except Exception:
            continue
        if not isinstance(d, dict) or not d:
            continue
        if not isinstance(next(iter(d.values())), dict) or \
                "token_ids" not in next(iter(d.values())):
            continue
        tag = str(p.relative_to(root))
        dumps[tag] = d
        md5 = hashlib.md5(p.read_bytes()).hexdigest()
        div = divergences(d, ref)
        extra = [x for x in div if x != STD]
        rep["dumps"][tag] = {"md5": md5, "n_prompts": len(d),
                             "gen_len": len(next(iter(d.values()))["token_ids"]),
                             "divergences": div, "extra": extra}
        rep["md5_groups"].setdefault(md5, []).append(tag)
        if extra:
            rep["anomalies"].append({"dump": tag, "extra": extra})

    # rep-vs-rep and base-vs-cap, for any `<stage>_<arm>_bs<N>_rep<R>` layout
    pat = re.compile(r"^(?P<stage>[^/]*?)_(?P<arm>base|cap)_bs(?P<bs>\d+)_rep(?P<rep>\d+)/")
    cells = {}
    for tag in dumps:
        m = pat.match(tag)
        if m:
            cells.setdefault((m["stage"], m["arm"], m["bs"]), {})[m["rep"]] = tag
    for (stage, arm, bs), reps in sorted(cells.items()):
        key = f"{stage}_{arm}_bs{bs}"
        rep["rep_matrix"][key] = {
            f"{i}v{j}": first_div(dumps[reps[i]], dumps[reps[j]])
            for i, j in itertools.combinations(sorted(reps), 2)}
    for (stage, _, bs) in {(s, None, b) for (s, _, b) in cells}:
        b = cells.get((stage, "base", bs))
        c = cells.get((stage, "cap", bs))
        if b and c:
            rep["rep_matrix"][f"{stage}_bs{bs}_base_vs_cap"] = {
                f"rep{r}": first_div(dumps[b[r]], dumps[c[r]])
                for r in sorted(set(b) & set(c))}

    rep["summary"] = {
        "n_dumps": len(dumps),
        "n_anomalous": len(rep["anomalies"]),
        "n_md5_groups": len(rep["md5_groups"]),
        "largest_md5_group": max((len(v) for v in rep["md5_groups"].values()),
                                 default=0),
    }
    Path(a.out).write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep["summary"], indent=1))
    for x in rep["anomalies"]:
        print("ANOMALY", x["dump"], x["extra"])
    for k, v in rep["rep_matrix"].items():
        if any(x is not None for x in v.values()):
            print("NONDET", k, v)
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
