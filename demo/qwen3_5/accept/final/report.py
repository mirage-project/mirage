#!/usr/bin/env python3
"""Assemble the final gate report and decide the EXIT CODE.  Pure stdlib.

EXIT CODES (the contract ``.pm/accept.sh`` expects)
---------------------------------------------------
  0  every criterion in scope PASSED.
  1  a criterion FAILED (which one, with numbers), or an integrity violation.
  3  NOT-APPLICABLE: a prerequisite genuinely could not run, so a criterion
     could not be evaluated -- and NOTHING already failed.

PRECEDENCE IS FAIL-FIRST, deliberately: ``FAIL`` outranks ``NOT_EVALUABLE``, so
a criterion cannot be turned from red into "not applicable" by breaking its
measurement.  Exit 3 is only reachable when there is nothing red at all.

The human summary prints, for every criterion, the numbers behind the verdict
and the ONE claim the run is entitled to make (``honest_claim``).  Anything the
gate tolerates but must not hide -- the fingerprint divergence rate, an
exactness degradation, an engine-vs-reference near-tie contradiction -- is
printed under DIAGNOSTICS whether the gate passed or failed.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

SCHEMA = "final/report/v1"


def _load(p):
    if not p:
        return None
    path = Path(p)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-meta", required=True)
    ap.add_argument("--integrity", default=None)
    ap.add_argument("--ac3", default=None)
    ap.add_argument("--perf", default=None)
    ap.add_argument("--stages", required=True,
                    help="comma list of the stages this invocation was asked to run")
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--output-summary", required=True)
    ap.add_argument("--non-binding", action="store_true",
                    help="set by --rescore; the report says so and the caller "
                         "must not treat the verdict as an AC-6 result")
    a = ap.parse_args(argv)

    stages = [s for s in a.stages.split(",") if s.strip()]
    rep = {"schema": SCHEMA,
           "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "run": _load(a.run_meta), "stages_requested": stages,
           "binding": not a.non_binding,
           "integrity": _load(a.integrity), "ac3": _load(a.ac3),
           "perf": _load(a.perf),
           "criteria": {}, "failures": [], "not_evaluable": [], "verdict": None}

    integ, ac3, perf = rep["integrity"], rep["ac3"], rep["perf"]
    if integ is not None and not integ.get("binding", True):
        rep["binding"] = False

    # ---- integrity -------------------------------------------------------
    if "integrity" in stages:
        if integ is None:
            rep["failures"].append("integrity stage produced no report")
            rep["criteria"]["integrity"] = "FAIL"
        else:
            rep["criteria"]["integrity"] = integ.get("verdict")
            rep["failures"] += [f"integrity: {v}" for v in integ.get("violations", [])]

    # ---- AC-3 ------------------------------------------------------------
    if "ac3" in stages:
        if ac3 is None:
            rep["not_evaluable"].append("AC-3 stage produced no report")
            rep["criteria"]["AC-3"] = "NOT_EVALUABLE"
        else:
            rep["criteria"]["AC-3"] = ac3.get("verdict")
            rep["failures"] += [f"AC-3: {f}" for f in ac3.get("failures", [])]
            rep["not_evaluable"] += [f"AC-3: {x}" for x in ac3.get("not_evaluable", [])]

    # ---- AC-4 / AC-5 -----------------------------------------------------
    if "perf" in stages:
        if perf is None:
            rep["not_evaluable"].append("perf stage produced no report")
            rep["criteria"]["AC-4"] = rep["criteria"]["AC-5"] = "NOT_EVALUABLE"
        else:
            rep["criteria"]["AC-4"] = perf.get("ac4", {}).get("verdict")
            rep["criteria"]["AC-5"] = perf.get("ac5", {}).get("verdict")
            rep["failures"] += [f"perf: {f}" for f in perf.get("failures", [])]
            rep["not_evaluable"] += [f"perf: {x}" for x in perf.get("not_evaluable", [])]

    # stages that were NOT requested are recorded as such -- never as passes
    for name, st in (("AC-3", "ac3"), ("AC-4", "perf"), ("AC-5", "perf"),
                     ("integrity", "integrity")):
        rep["criteria"].setdefault(name, f"NOT_RUN ({st} stage not requested)")
    if any(str(v).startswith("NOT_RUN") for v in rep["criteria"].values()):
        rep["not_evaluable"].append(
            "not every criterion was in scope for this invocation: "
            + json.dumps({k: v for k, v in rep["criteria"].items()
                          if str(v).startswith("NOT_RUN")}))

    rep["verdict"] = ("FAIL" if rep["failures"]
                      else "NOT_APPLICABLE" if rep["not_evaluable"]
                      else "PASS")
    rc = {"FAIL": 1, "NOT_APPLICABLE": 3, "PASS": 0}[rep["verdict"]]

    # ---- the one claim this run is entitled to make ---------------------
    claim = []
    if rep["verdict"] == "PASS":
        claim.append(
            "AC-3/AC-4/AC-5 all PASS at bs {1,2,4,8,16} on the pinned 256/1024 "
            "workload, from a clean tree, against a same-window fresh vLLM "
            "comparator that agrees with the pinned baseline inside the "
            "protocol's own two-boot statistic.")
    else:
        claim.append(f"The gate is {rep['verdict']}: "
                     + "; ".join(rep["failures"][:4] or rep["not_evaluable"][:4]))
    if ac3:
        ex = (ac3.get("diagnostics") or {}).get("exactness") or {}
        st = (ac3.get("diagnostics") or {}).get("stability") or {}
        claim.append(
            f"AC-3(c) diagnostic: {ex.get('cases_exact')}/{ex.get('cases_compared')} "
            f"cases byte-identical to results/dumps_final, "
            f"{ex.get('cases_degraded')} degraded "
            f"({ex.get('degradations_unexplained')} unexplained); cold-rep "
            f"fingerprint divergence "
            f"{(st.get('totals') or {}).get('fingerprint_divergence_rate')} "
            f"of scored reps. No claim is made that a single cold run is "
            f"reproducible.")
    if not rep["binding"]:
        claim.insert(0, "NON-BINDING RUN -- not an AC-6 result.")
    rep["honest_claim"] = claim

    out = Path(a.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2))

    # ------------------------------------------------------------ summary --
    L = []
    run = rep["run"] or {}
    L.append("=" * 78)
    L.append(f"MPK Qwen3.5-35B-A3B-FP8 FINAL GATE -- {rep['verdict']} (exit {rc})")
    L.append(f"  utc={rep['generated_utc']}  sha={run.get('git_sha')}  "
             f"host={run.get('host')}  mode={run.get('run_mode')}")
    L.append(f"  stages={','.join(stages)}  binding={rep['binding']}")
    L.append("=" * 78)
    for k in ("integrity", "AC-3", "AC-4", "AC-5"):
        L.append(f"  {k:<10} {rep['criteria'].get(k)}")
    if perf:
        L.append("")
        L.append("  AC-4/AC-5 per batch size (mpk vs FRESH vLLM, pinned 256/1024):")
        L.append(f"    {'bs':>3} {'mpk tok/s':>10} {'vllm tok/s':>11} {'ratio':>7} "
                 f"{'AC-4':>13} {'mpk e2e s':>10} {'vllm e2e s':>11} {'x':>6} "
                 f"{'AC-5':>13}")
        for bs in sorted(perf.get("per_bs", {}), key=int):
            r = perf["per_bs"][bs]
            a4, a5 = r["ac4"], r["ac5"]

            def f(x, w, p=1):
                return " " * w if x is None else f"{x:{w}.{p}f}"
            L.append(f"    {bs:>3} {f(a4['mpk_decode_tok_s'], 10)} "
                     f"{f(a4['vllm_decode_tok_s'], 11)} "
                     f"{f(a4['ratio_mpk_over_vllm'], 7, 3)} {a4['verdict']:>13} "
                     f"{f(a5['mpk_e2e_s'], 10, 2)} {f(a5['vllm_e2e_s'], 11, 2)} "
                     f"{f(a5['ratio_mpk_over_vllm'], 6, 2)} {a5['verdict']:>13}")
            for arm in ("full", "pre"):
                w = r["mpk"][arm]
                L.append(f"         mpk {arm:<4} per-rep wall ms "
                         f"{[round(x['wall_ms'], 1) for x in w['reps']]} "
                         f"D={w['distinct_D']} n={w['n']} "
                         f"range={'' if w['range_pct'] is None else format(w['range_pct'], '.2f')}%"
                         + (f" excluded={len(w['excluded'])}" if w["excluded"] else ""))
            vf = (r["vllm"].get("fresh") or {})
            L.append(f"         vllm fresh per-rep tok/s "
                     f"{[round(x, 1) for x in (vf.get('decode') or [])]} "
                     f"e2e {[round(x, 3) for x in (vf.get('e2e') or [])]}")
            dr = (r["vllm"].get("drift") or {}).get("decode") or {}
            L.append(f"         drift(decode): merged median "
                     f"{dr.get('merged_median')} n={dr.get('n_total')} "
                     f"IQR/med={dr.get('merged_iqr_over_median_pct')} "
                     f"boot-dev={dr.get('boot_median_max_deviation_pct')} "
                     f"valid={dr.get('valid')}")
    if ac3:
        L.append("")
        L.append("  AC-3 (re-pinned: coherence + >=90% agreement floor + "
                 "no silent degradation):")
        for bs in sorted(ac3.get("per_bs", {}), key=int):
            b = ac3["per_bs"][bs]
            L.append(f"    bs={bs:<3} {b['verdict']:<14} reps scored "
                     f"{b['reps_scored']} errors {b['reps_error']}")
            for r in b["reps"]:
                if not r.get("scored"):
                    L.append(f"         {r['tag']:<12} RUN ERROR {r.get('note')}")
                    continue
                worst = min((c["checks"]["agreement_floor"]["agreement"]
                             for c in r["cases"]
                             if "agreement_floor" in c.get("checks", {})), default=None)
                ppls = [c["checks"]["perplexity"]["ratio"] for c in r["cases"]
                        if c.get("checks", {}).get("perplexity", {}).get("ratio")]
                L.append(f"         {r['tag']:<12} exact {r['n_exact']}/{r['n_cases']}"
                         f"  min agreement "
                         f"{'n/a' if worst is None else format(worst, '.4f')}"
                         f"  max ppl ratio "
                         f"{'n/a' if not ppls else format(max(ppls), '.3f')}"
                         f"  {'PASS' if r['pass'] else 'NOT PASS'}")
        ex = (ac3.get("diagnostics") or {}).get("exactness") or {}
        L.append(f"    AC-3(c): {ex.get('cases_exact')}/{ex.get('cases_compared')} "
                 f"cases byte-identical to dumps_final; degraded "
                 f"{ex.get('cases_degraded')}, unexplained "
                 f"{ex.get('degradations_unexplained')}")
        if ex.get("degradations_unexplained"):
            L.append("    ** UNEXPLAINED DEGRADATION from bit-exact to "
                     "merely-passing -- AC-3(c) requires this to be explained in "
                     "the run report, not absorbed **")
            for d in ex.get("degradations", [])[:10]:
                if not d.get("explained_by_mechanism"):
                    L.append(f"       {d['rep']} bs{d['batch_size']} "
                             f"{d['prompt_id']} first diff @"
                             f"{d['first_divergent_position']} "
                             f"(still passes re-pinned AC-3: "
                             f"{d['still_passes_repinned_ac3']})")
        st = (ac3.get("diagnostics") or {}).get("stability") or {}
        L.append(f"    stability diagnostic: {st.get('verdict')} "
                 f"fingerprint divergence "
                 f"{(st.get('totals') or {}).get('fingerprint_divergence_rate')} "
                 f"token divergence "
                 f"{(st.get('totals') or {}).get('token_divergence_rate')}")
    # ---- PER-SIZE RESULT LINES ------------------------------------------
    # The pinned gate's own contract: "It must save raw run artifacts + configs
    # and print per-size RESULT lines."  One grep-able line per batch size,
    # carrying every number the three criteria rest on.
    L.append("")
    bss = sorted({*(perf or {}).get("per_bs", {}), *(ac3 or {}).get("per_bs", {})},
                 key=int)
    for bs in bss:
        pr = ((perf or {}).get("per_bs") or {}).get(bs) or {}
        a4 = pr.get("ac4") or {}
        a5 = pr.get("ac5") or {}
        a3 = (((ac3 or {}).get("per_bs") or {}).get(bs) or {}).get("verdict", "NOT_RUN")

        def n(x, p=1):
            return "n/a" if x is None else format(x, f".{p}f")
        L.append(
            f"RESULT bs={bs} AC3={a3} "
            f"AC4={a4.get('verdict', 'NOT_RUN')} "
            f"mpk_decode_tok_s={n(a4.get('mpk_decode_tok_s'))} "
            f"vllm_decode_tok_s={n(a4.get('vllm_decode_tok_s'))} "
            f"decode_ratio={n(a4.get('ratio_mpk_over_vllm'), 3)} "
            f"AC5={a5.get('verdict', 'NOT_RUN')} "
            f"mpk_e2e_s={n(a5.get('mpk_e2e_s'), 3)} "
            f"vllm_e2e_s={n(a5.get('vllm_e2e_s'), 3)} "
            f"e2e_ratio={n(a5.get('ratio_mpk_over_vllm'), 3)} "
            f"e2e_ratio_max={n(a5.get('max_ratio'), 2)}")

    if rep["failures"]:
        L.append("")
        L.append("  FAILURES")
        for f_ in rep["failures"]:
            L.append(f"    - {f_}")
    if rep["not_evaluable"]:
        L.append("")
        L.append("  NOT EVALUABLE (exit 3 only if nothing failed)")
        for x in rep["not_evaluable"]:
            L.append(f"    - {x}")
    L.append("")
    L.append("  HONEST CLAIM")
    for c in rep["honest_claim"]:
        L.append(f"    {c}")
    L.append("")
    L.append(f"  machine-readable report: {out}")
    L.append("=" * 78)
    text = "\n".join(L) + "\n"
    Path(a.output_summary).write_text(text)
    print(text)
    return rc


if __name__ == "__main__":
    sys.exit(main())
