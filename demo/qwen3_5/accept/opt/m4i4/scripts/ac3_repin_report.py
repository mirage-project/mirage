#!/usr/bin/env python3
"""M4-I4 -- report the RE-PINNED AC-3 (goal.md, re-pinned 2026-07-29) for a dump tree.

The re-pinned gate is three conjunctive parts. This script reports what can be
scored from token dumps plus the committed reference and baseline, and says
explicitly which part it did not evaluate rather than implying a verdict:

  (a) COHERENCE
      * degenerate repetition -- no n-gram with n>=4 repeated more than 3x in a
        continuation. Token-level, scored here.
      * non-language byte soup -- scored here when a tokenizer is importable
        (decode the 64 ids and measure the replacement-char / non-printable
        fraction against the reference continuation's own).
      * perplexity within 1.5x of the reference continuation's -- needs the HF
        reference MODEL, which this script does not load. Where a case is
        BYTE-IDENTICAL to the adjudicated baseline the perplexity is identical by
        construction, so the part transfers by identity and is reported as such;
        any case that is not byte-identical is reported as NOT-EVALUATED so a
        real measurement is owed rather than assumed.
  (b) AGREEMENT FLOOR -- >=90% of positions equal to the HF reference top-1, per
      (prompt, bs), scored here from reference_outputs.json.
  (c) NO SILENT DEGRADATION -- bit-exactness against the committed baseline
      (results/dumps_final) reported for every case, every rep.

Differing positions are CLASSIFIED, never waived here. Only the FIRST divergence
in a (prompt, bs) is independent evidence -- greedy decode conditions on its own
output, so everything after it is a different conditioning sequence
(`ac3_types.TieVerdict.POST_DIVERGENCE`). The committed reference carries only
top1 logits (`ac3_types.ReferenceStep`), so a reference-side margin is
unavailable; what is reported per flip is the position, the baseline token, the
engine token, whether the same flip is in the committed adjudication, and whether
it reproduces across reps. An unexplained, reproducing first divergence is a stop.

SCORER PROVENANCE. The canonical re-pinned AC-3 scorers belong to M4-I1's
`accept/final/ac3_criteria.py`. This script IMPORTS them when they are present, so
there is one implementation; the local fallbacks exist only so the report still
runs before that file lands, and they implement the same rules -- including the
reference-relative repetition bar, which M4-I1 derived first and for the same
measured reason (the goal's literal absolute bound of 3 is exceeded by the HF
reference's own p07-format continuation, a markdown list prefix).
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ACC = HERE.parents[2]

AGREEMENT_FLOOR = 0.90      # goal.md AC-3(b)
REPETITION_N = 4            # goal.md AC-3(a): n-gram, n>=4
REPETITION_MAX = 3          # ... repeated more than 3x is degenerate
NEW_TOKENS = 64


# ------------------------------------------------------------- primitives ---
def worst_ngram_repeat(ids, n=REPETITION_N):
    """Highest repeat count of any n-gram in `ids` (1 = every n-gram unique)."""
    if len(ids) < n:
        return 0, None
    c = collections.Counter(tuple(ids[i:i + n]) for i in range(len(ids) - n + 1))
    gram, cnt = c.most_common(1)[0]
    return cnt, list(gram)


def _local_repetition_verdict(ids, reference_ids=None):
    """Degeneracy check with the goal's absolute bound RAISED, per prompt, to the
    pinned reference continuation's own worst repetition -- see the module
    docstring's scorer-provenance note. With reference_ids=None the literal goal
    bound applies unchanged."""
    worst, gram = worst_ngram_repeat(ids)
    bar = REPETITION_MAX
    ref_worst = None
    if reference_ids is not None:
        ref_worst, _ = worst_ngram_repeat(reference_ids)
        bar = max(REPETITION_MAX, ref_worst)
    return dict(passed=worst <= bar, worst_ngram_repeats=worst,
                ngram_n=REPETITION_N, limit=bar, goal_limit=REPETITION_MAX,
                reference_worst=ref_worst, worst_ngram=gram,
                scorer="opt/m4i4 local")


def _load_canonical():
    """M4-I1's scorers if they are in the tree; otherwise the local fallbacks."""
    sys.path.insert(0, str(ACC / "final"))
    try:
        import ac3_criteria as canon            # noqa: E402
    except Exception:
        return None
    if not all(hasattr(canon, n) for n in ("repetition_verdict",)):
        return None
    return canon


NONPRINT = re.compile(r"[^\S\n\t]")   # unused placeholder; see nonlanguage()


def nonlanguage(text: str) -> dict:
    """Cheap byte-soup detector: replacement chars and control bytes."""
    if text is None:
        return dict(available=False)
    n = max(1, len(text))
    repl = text.count("�")
    ctrl = sum(1 for ch in text if ord(ch) < 32 and ch not in "\n\t\r")
    return dict(available=True, chars=len(text), replacement_chars=repl,
                replacement_frac=repl / n, control_chars=ctrl,
                control_frac=ctrl / n)


def agreement(engine_ids, ref_ids) -> dict:
    n = min(len(engine_ids), len(ref_ids))
    diffs = [i for i in range(n) if engine_ids[i] != ref_ids[i]]
    matched = n - len(diffs)
    frac = matched / n if n else None
    return dict(n_positions=n, n_match=matched, agreement=frac,
                floor=AGREEMENT_FLOOR,
                passed=(frac is not None and frac >= AGREEMENT_FLOOR),
                differing_positions=diffs,
                length_engine=len(engine_ids), length_reference=len(ref_ids))


# ------------------------------------------------------------------ inputs ---
def load_reference(path: Path) -> dict:
    d = json.load(open(path))["results"]
    return {pid: v["output_ids"][:NEW_TOKENS] for pid, v in d.items()}


def load_tree(root: Path, batch_sizes) -> dict:
    """{bs: {pid: ids}} from a tree with bs<N>.json at its top level."""
    out = {}
    for bs in batch_sizes:
        p = root / f"bs{bs}.json"
        if not p.is_file():
            continue
        out[bs] = {pid: v["token_ids"] for pid, v in json.load(open(p)).items()}
    return out


def get_tokenizer(model_path):
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(model_path)
    except Exception as exc:
        print(f"[note] tokenizer unavailable ({type(exc).__name__}: {exc}); "
              f"the byte-soup part of AC-3(a) is reported NOT-EVALUATED",
              file=sys.stderr)
        return None


def committed_adjudicated_flips(report_path: Path) -> set:
    """(prompt_id, bs, position) triples the committed M2 report already knows."""
    if not report_path.is_file():
        return set()
    rep = json.load(open(report_path))
    out = set()
    for w in rep.get("waiver_requests", []) or []:
        out.add((w.get("prompt_id"), w.get("batch_size"), w.get("position")))
    for r in rep.get("prompt_results", []) or []:
        p = r.get("first_divergent_position")
        if p is not None:
            out.add((r.get("prompt_id"), r.get("batch_size"), p))
    return out


# ------------------------------------------------------------------- main ----
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tree", action="append", required=True, metavar="LABEL=DIR",
                    help="a dump tree to score; repeatable (e.g. auto_rep0=/path)")
    ap.add_argument("--baseline", default=str(ACC / "results" / "dumps_final"))
    ap.add_argument("--reference",
                    default=str(ACC / "reference" / "reference_outputs.json"))
    ap.add_argument("--committed-report",
                    default=str(ACC / "results" / "run_report_all_bs.json"))
    ap.add_argument("--batch-sizes", default="1,2,4,8,16")
    ap.add_argument("--model-path", default=os.environ.get("MPK_MODEL_PATH"))
    ap.add_argument("--out-json", default=None)
    a = ap.parse_args(argv)

    bss = [int(x) for x in a.batch_sizes.split(",") if x.strip()]
    canon = _load_canonical()
    if canon is not None:
        def repetition_verdict(ids, reference_ids=None):
            v = dict(canon.repetition_verdict(ids, reference_ids))
            # normalise the canonical scorer's key names onto this report's
            v["passed"] = v.get("pass")
            v["worst_ngram_repeats"] = v.get("worst_count")
            v["limit"] = v.get("max_allowed_count")
            v["goal_limit"] = v.get("goal_absolute_bound")
            v["reference_worst"] = v.get("reference_worst_count")
            v["worst_ngram"] = v.get("worst_gram")
            v["scorer"] = "final/ac3_criteria.py"
            return v
        scorer = "final/ac3_criteria.py (M4-I1 canonical)"
    else:
        repetition_verdict = _local_repetition_verdict
        scorer = "opt/m4i4 local fallback (final/ac3_criteria.py not in tree)"
    print(f"[scorer] repetition/coherence primitives: {scorer}", file=sys.stderr)

    ref = load_reference(Path(a.reference))
    base = load_tree(Path(a.baseline), bss)
    adjudicated = committed_adjudicated_flips(Path(a.committed_report))
    tok = get_tokenizer(a.model_path) if a.model_path else None
    ref_text = ({pid: tok.decode(ids) for pid, ids in ref.items()} if tok else {})

    trees = {}
    for spec in a.tree:
        label, _, d = spec.partition("=")
        trees[label] = Path(d)

    doc = dict(schema="m4i4/ac3_repin/v1", agreement_floor=AGREEMENT_FLOOR,
               repetition=dict(n=REPETITION_N, max_repeats=REPETITION_MAX),
               scorer=scorer, baseline=str(a.baseline), reference=str(a.reference),
               tokenizer_available=bool(tok), trees={})
    print("=" * 108)
    print("RE-PINNED AC-3 REPORT (goal.md AC-3 as re-pinned 2026-07-29)")
    print(f"  (a) coherence: repetition + byte-soup scored here; perplexity "
          f"transfers by identity on byte-identical cases")
    print(f"  (b) agreement floor: >= {AGREEMENT_FLOOR:.0%} of positions == HF "
          f"reference top-1, per (prompt, bs)")
    print(f"  (c) no silent degradation: bit-exactness vs {a.baseline} reported, "
          f"not required")
    print("=" * 108)

    all_flips = collections.Counter()
    for label, root in trees.items():
        got = load_tree(root, bss)
        rec = dict(dir=str(root), per_bs={})
        print(f"\n### tree {label}  ({root})")
        for bs in bss:
            if bs not in got:
                print(f"  bs{bs:<2} MISSING")
                rec["per_bs"][bs] = dict(present=False)
                continue
            cases = {}
            n_exact = n_agree = 0
            worst_agree = (1.1, None)
            for pid in sorted(ref):
                eng = got[bs].get(pid)
                if eng is None:
                    cases[pid] = dict(present=False)
                    continue
                eng = eng[:NEW_TOKENS]
                bexact = (base.get(bs, {}).get(pid, [])[:NEW_TOKENS] == eng)
                agr = agreement(eng, ref[pid])
                rep = repetition_verdict(eng, ref[pid])
                soup = (nonlanguage(tok.decode(eng)) if tok else dict(available=False))
                soup_ref = (nonlanguage(ref_text.get(pid)) if tok
                            else dict(available=False))
                flips = []
                first = (agr["differing_positions"][0]
                         if agr["differing_positions"] else None)
                for pos in agr["differing_positions"]:
                    bt = base.get(bs, {}).get(pid, [])
                    independent = (pos == first)
                    flips.append(dict(
                        position=pos, engine=eng[pos], reference=ref[pid][pos],
                        baseline=(bt[pos] if pos < len(bt) else None),
                        # only the first divergence is independent evidence:
                        # greedy decode conditions on its own output afterwards
                        independent=independent,
                        classification=("first-divergence" if independent
                                        else "post-divergence-not-independent"),
                        in_committed_adjudication=((pid, bs, pos) in adjudicated),
                        differs_from_baseline=(pos < len(bt) and bt[pos] != eng[pos])))
                    if independent:
                        all_flips[(pid, bs, pos)] += 1
                cases[pid] = dict(
                    present=True, bit_exact_vs_baseline=bexact, agreement=agr,
                    repetition=rep, nonlanguage=soup,
                    nonlanguage_reference=soup_ref, flips=flips,
                    perplexity_part=("transfers-by-identity (byte-identical to the "
                                     "adjudicated baseline)" if bexact
                                     else "NOT-EVALUATED (needs the HF reference model)"))
                n_exact += bool(bexact)
                n_agree += bool(agr["passed"])
                if agr["agreement"] is not None and agr["agreement"] < worst_agree[0]:
                    worst_agree = (agr["agreement"], pid)
            n = sum(1 for c in cases.values() if c.get("present"))
            rep_ok = all(c["repetition"]["passed"] for c in cases.values()
                         if c.get("present"))
            soup_ok = None
            if tok:
                soup_ok = all(
                    c["nonlanguage"]["replacement_frac"]
                    <= max(0.01, c["nonlanguage_reference"]["replacement_frac"])
                    for c in cases.values() if c.get("present"))
            rec["per_bs"][bs] = dict(present=True, n_cases=n, n_bit_exact=n_exact,
                                     n_agreement_pass=n_agree,
                                     repetition_all_pass=rep_ok,
                                     bytesoup_all_pass=soup_ok,
                                     worst_agreement=worst_agree[0],
                                     worst_agreement_prompt=worst_agree[1],
                                     cases=cases)
            print(f"  bs{bs:<2} cases={n:2d}  bit-exact={n_exact}/{n}  "
                  f"agreement>={AGREEMENT_FLOOR:.0%}: {n_agree}/{n}  "
                  f"worst={worst_agree[0]:.4f} ({worst_agree[1]})  "
                  f"repetition={'ok' if rep_ok else 'FAIL'}  "
                  f"byte-soup={'ok' if soup_ok else ('FAIL' if soup_ok is False else 'n/e')}")
            for pid, c in cases.items():
                for f in c.get("flips", []):
                    if not f["independent"]:
                        continue           # post-divergence: not evidence
                    mark = ("known-adjudicated" if f["in_committed_adjudication"]
                            else "NEW-vs-adjudication")
                    tail = sum(1 for g in c["flips"] if not g["independent"])
                    print(f"        first divergence {pid} bs{bs} "
                          f"pos{f['position']}: engine={f['engine']} "
                          f"ref={f['reference']} baseline={f['baseline']} "
                          f"{'differs-from-baseline' if f['differs_from_baseline'] else 'same-as-baseline'}"
                          f" [{mark}]  (+{tail} post-divergence positions)")
        doc["trees"][label] = rec

    print("\n### flip reproduction across trees (a flip in ONE tree only is an "
          "anomaly candidate, not a finding -- determinism protocol)")
    if not all_flips:
        print("  none")
    for (pid, bs, pos), k in sorted(all_flips.items()):
        print(f"  {pid} bs{bs} pos{pos}: present in {k}/{len(trees)} trees "
              f"[{'known-adjudicated' if (pid, bs, pos) in adjudicated else 'NEW'}]")
    doc["flip_reproduction"] = {f"{pid}|bs{bs}|pos{pos}": dict(
        trees_with_flip=k, n_trees=len(trees),
        in_committed_adjudication=((pid, bs, pos) in adjudicated))
        for (pid, bs, pos), k in all_flips.items()}

    if a.out_json:
        Path(a.out_json).write_text(json.dumps(doc, indent=1) + "\n")
        print(f"\nwrote {a.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
