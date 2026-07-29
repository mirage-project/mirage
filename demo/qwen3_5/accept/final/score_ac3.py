#!/usr/bin/env python3
"""Score the RE-PINNED AC-3 over a tree of cold reps.  PURE: reads committed
artifacts + a collected rep tree, writes one JSON verdict, needs no GPU, no
torch and no tokenizer (``tests/test_score_ac3.py`` drives it on fixtures).

WHAT IT ENFORCES  (``.pm/goal.md`` AC-3, re-pinned 2026-07-29)
-------------------------------------------------------------
Per (rep, batch size, prompt), all three parts, conjunctively:

  (a) COHERENCE     no n-gram n>=4 repeated >3x; no more non-language characters
                    than the pinned reference continuation for that prompt; HF
                    reference-model perplexity within 1.5x of the reference
                    continuation's own on the same prompt.
  (b) FLOOR         >= 90% of the 64 positions match the HF reference top-1, and
                    EVERY differing position is accounted for -- reference
                    near-tie, engine-side <=3 bf16 ULPs, or a documented
                    mechanism entry.
  (c) DIAGNOSTIC    bit-exactness against the committed ``results/dumps_final``
                    is measured and reported for every rep and every case, and
                    any drop from exact to merely-passing is listed as a
                    degradation (explained iff a mechanism entry covers it).

EVERY rep is scored, not only the fingerprint-consensus ones
-----------------------------------------------------------
``docs/qwen35/bench-protocol.md`` ("M4 gate policy") argues the strict form:
scoping the correctness assertion to the reps that survive fingerprint
quarantine is indistinguishable from hiding a real bug, and "a gate that can be
quieted by re-running is not a gate".  That argument was made when the pass
condition was byte-identity, where it cost a ~2% false-FAIL rate.  Under the
re-pinned AC-3 the strict form is FREE: the engine's measured cold-rep token
divergence (2.08%, M4-I0) moves a handful of positions, which leaves agreement
far above the 90% floor and coherence untouched.  So this scorer keeps the
strict form -- every rep that produced a dump must satisfy (a) and (b) -- and
the quarantine machinery is used only for the reported divergence RATE.

NOT-EVALUABLE is not a pass
---------------------------
A missing perplexity number, a missing reference margin, fewer than the required
number of scored reps, a wrong-length sequence: each is recorded and makes the
batch size NOT_EVALUABLE, never PASS.  The caller (``report.py``) maps
NOT_EVALUABLE to exit 3 only when nothing else already failed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from ac3_criteria import (  # noqa: E402
    ACCOUNTED_CLASSES, AGREEMENT_FLOOR, CLASS_UNEXPLAINED, agreement_verdict,
    classify_difference, nonlanguage_verdict, perplexity_verdict,
    repetition_verdict,
)

SCHEMA = "final/ac3_score/v1"


# ---------------------------------------------------------------- inputs ---
def load_reference(path: Path, expect_prompts: int, expect_tokens: int) -> dict:
    doc = json.loads(path.read_text())
    res = doc["results"]
    if len(res) != expect_prompts:
        raise SystemExit(f"INTEGRITY: reference has {len(res)} prompts, "
                         f"expected {expect_prompts}: {path}")
    out = {}
    for pid, e in res.items():
        ids = e["output_ids"]
        if len(ids) != expect_tokens or e.get("num_generated") != expect_tokens:
            raise SystemExit(f"INTEGRITY: reference {pid} has {len(ids)} generated "
                             f"tokens, expected {expect_tokens}")
        t1 = e.get("top1_logit_per_step") or [None] * len(ids)
        t2i = e.get("top2_id_per_step") or [None] * len(ids)
        t2l = e.get("top2_logit_per_step") or [None] * len(ids)
        margins = [None if (a is None or b is None) else (a - b)
                   for a, b in zip(t1, t2l)]
        out[pid] = {"output_ids": ids, "top1_logit": t1, "top2_id": t2i,
                    "top2_logit": t2l, "margin": margins}
    return out


def load_reps(reps_root: Path, batch_sizes) -> dict:
    """{bs: [ {tag, rep_index, status, dump, meta} ]} straight off the rep tree
    ``harness/gate_ac3_stable.sh`` produced.  A rep whose artifacts are missing
    or unreadable is kept as a run error -- never dropped, because dropping it
    would shrink the denominator of everything reported here."""
    per_bs = {bs: [] for bs in batch_sizes}
    for d in sorted(reps_root.glob("bs*_r*")):
        try:
            bs = int(d.name.split("_r", 1)[0][2:])
            ri = int(d.name.rsplit("_r", 1)[1])
        except (IndexError, ValueError):
            continue
        if bs not in per_bs:
            continue
        rec = {"tag": d.name, "rep_index": ri, "dir": str(d),
               "status": "error", "dump": None, "meta": {}, "note": None}
        metas = sorted(d.glob("meta_*.json"))
        if metas:
            try:
                rec["meta"] = json.loads(metas[0].read_text())
                rec["status"] = rec["meta"].get("status", "error")
            except Exception as e:                      # noqa: BLE001
                rec["note"] = f"unreadable {metas[0].name}: {type(e).__name__}: {e}"
        else:
            rec["note"] = "no meta_*.json emitted"
        dump = d / f"bs{bs}.json"
        if dump.exists():
            try:
                rec["dump"] = json.loads(dump.read_text())
                rec["dump_sha256"] = hashlib.sha256(dump.read_bytes()).hexdigest()
            except Exception as e:                      # noqa: BLE001
                rec["status"] = "error"
                rec["note"] = f"unreadable {dump.name}: {type(e).__name__}: {e}"
        elif rec["status"] == "ok":
            rec["status"] = "error"
            rec["note"] = f"meta says ok but bs{bs}.json is absent"
        per_bs[bs].append(rec)
    for bs in per_bs:
        per_bs[bs].sort(key=lambda r: r["rep_index"])
    return per_bs


def _coh_index(coherence: dict) -> tuple:
    """(reference-by-prompt, engine-by-(bs,prompt,token-sha)).

    Engine records are keyed by the CONTENT hash of the continuation, not by rep
    tag: identical continuations are scored by the HF model once (an exact
    dedup, not an approximation), and every rep that produced those bytes gets
    the same number.
    """
    ref = {pid: r for pid, r in (coherence.get("reference") or {}).items()}
    eng = {}
    for r in coherence.get("engine") or []:
        eng[(int(r["batch_size"]), r["prompt_id"], r["token_ids_sha256"])] = r
    return ref, eng


def _sha_ids(ids) -> str:
    return hashlib.sha256(json.dumps(list(ids)).encode()).hexdigest()


# ----------------------------------------------------------------- score ---
def score_case(*, prompt_id, batch_size, engine_ids, ref, coh_ref, coh_eng,
               mechanisms, engine_margins, expect_tokens) -> dict:
    """One (rep, bs, prompt) case: (a) + (b) + the (c) inputs."""
    ref_ids = ref["output_ids"]
    rec = {"prompt_id": prompt_id, "batch_size": batch_size,
           "n_engine_tokens": len(engine_ids), "n_reference_tokens": len(ref_ids),
           "checks": {}, "not_evaluable": [], "failures": []}

    # Length is part of the run protocol, not of AC-3's content: a sequence that
    # is not exactly the pinned horizon means the run was configured wrong, and
    # scoring content against a mis-configured run would be scoring the wrong
    # thing.  It is a hard failure with its own name (harness/README.md's
    # ENGINE_TOO_LONG/SHORT argument, kept).
    if len(engine_ids) != expect_tokens:
        rec["checks"]["length"] = {
            "criterion": f"engine produced exactly {expect_tokens} new tokens "
                         f"(run protocol, --correct-new-tokens)",
            "pass": False}
        rec["failures"].append("wrong_generated_length")
        rec["pass"] = False
        return rec
    rec["checks"]["length"] = {"criterion": f"exactly {expect_tokens} new tokens",
                              "pass": True}

    # ---- (b) floor + per-position accounting ----
    n = expect_tokens
    matches = sum(1 for i in range(n) if engine_ids[i] == ref_ids[i])
    agree = agreement_verdict(n, matches)
    rec["checks"]["agreement_floor"] = agree
    if not agree["pass"]:
        rec["failures"].append("agreement_below_floor")

    # Two passes, because the cascade class is only available once the FIRST
    # divergence has been accounted for on its own evidence (ac3_criteria's
    # CLASS_POST_DIVERGENCE): pass 1 classifies the first divergence, pass 2 the
    # rest.  An unaccounted first divergence therefore excuses nothing.
    positions = [i for i in range(n) if engine_ids[i] != ref_ids[i]]
    first_div = positions[0] if positions else None

    def _classify(i, first_ok):
        em = (engine_margins or {}).get(f"{prompt_id}|{batch_size}|{i}") or {}
        d = classify_difference(
            prompt_id=prompt_id, batch_size=batch_size, position=i,
            ref_top1_id=ref_ids[i], ref_top2_id=ref["top2_id"][i],
            ref_margin=ref["margin"][i], engine_id=engine_ids[i],
            engine_margin=em.get("margin"),
            engine_margin_ref_logit=em.get("logit_basis"),
            mechanisms=mechanisms,
            is_first_divergence=(i == first_div),
            first_divergence_accounted=first_ok)
        d["conditioning_shared_with_reference"] = (i == first_div)
        return d

    diffs = []
    first_ok = False
    if first_div is not None:
        d0 = _classify(first_div, False)
        first_ok = d0["accounted"]
        diffs.append(d0)
        for i in positions[1:]:
            diffs.append(_classify(i, first_ok))
    rec["differing_positions"] = diffs
    rec["first_divergent_position"] = first_div
    unexplained = [d for d in diffs if not d["accounted"]]
    rec["checks"]["all_differences_accounted"] = {
        "criterion": "every differing position is a reference near-tie, an "
                     "engine-side <=3-bf16-ULP near-tie, covered by a documented "
                     "mechanism entry, or downstream of an already-accounted "
                     "first divergence (goal AC-3(b); see ac3_criteria's "
                     "CLASS_POST_DIVERGENCE for the guard on that last class)",
        "first_divergence_accounted": first_ok,
        "n_differing": len(diffs), "n_unexplained": len(unexplained),
        "classes": {c: sum(1 for d in diffs if d["classification"] == c)
                    for c in list(ACCOUNTED_CLASSES) + [CLASS_UNEXPLAINED]},
        "pass": not unexplained}
    if unexplained:
        rec["failures"].append("unexplained_differing_positions")

    # ---- (a) coherence ----
    rep_v = repetition_verdict(engine_ids, ref_ids)
    rec["checks"]["repetition"] = rep_v
    if not rep_v["pass"]:
        rec["failures"].append("degenerate_repetition")

    key = (batch_size, prompt_id, _sha_ids(engine_ids))
    eng_coh = coh_eng.get(key)
    ref_coh = coh_ref.get(prompt_id)
    if eng_coh is None or ref_coh is None:
        rec["checks"]["nonlanguage"] = {"pass": False, "available": False,
                                        "reason": "no HF coherence record for this "
                                                  "continuation"}
        rec["checks"]["perplexity"] = perplexity_verdict(None, None)
        rec["not_evaluable"].append("coherence_inputs_missing")
    else:
        nl = nonlanguage_verdict(eng_coh["text"], ref_coh["text"])
        rec["checks"]["nonlanguage"] = nl
        if not nl["pass"]:
            rec["failures"].append("non_language_byte_soup")
        ppl = perplexity_verdict(eng_coh.get("ppl"), ref_coh.get("ppl"))
        rec["checks"]["perplexity"] = ppl
        if not ppl["available"]:
            rec["not_evaluable"].append("perplexity_unavailable")
        elif not ppl["pass"]:
            rec["failures"].append("perplexity_above_1.5x")

    rec["pass"] = not rec["failures"] and not rec["not_evaluable"]
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--reps-root", required=True)
    ap.add_argument("--baseline", required=True,
                    help="results/dumps_final -- the (c) exactness diagnostic baseline")
    ap.add_argument("--coherence", default=None,
                    help="coherence_inputs.json from hf_score.py")
    ap.add_argument("--mechanisms", default=str(HERE / "mechanisms.json"))
    ap.add_argument("--engine-margins", default=None,
                    help="optional engine-side logit margins "
                         '{"<pid>|<bs>|<pos>": {"margin": f, "logit_basis": f}}')
    ap.add_argument("--gate-report", default=None,
                    help="gate_ac3_stable.json -- fingerprint divergence diagnostics")
    ap.add_argument("--batch-sizes", default="1,2,4,8,16")
    ap.add_argument("--expect-new-tokens", type=int, default=64)
    ap.add_argument("--expect-num-prompts", type=int, default=10)
    ap.add_argument("--reps-required", type=int, default=3)
    ap.add_argument("--output-json", required=True)
    a = ap.parse_args(argv)

    batch_sizes = [int(x) for x in a.batch_sizes.split(",") if x.strip()]
    ref = load_reference(Path(a.reference), a.expect_num_prompts, a.expect_new_tokens)
    reps = load_reps(Path(a.reps_root), batch_sizes)
    coherence = (json.loads(Path(a.coherence).read_text())
                 if a.coherence and Path(a.coherence).exists() else {})
    coh_ref, coh_eng = _coh_index(coherence)
    mech_doc = (json.loads(Path(a.mechanisms).read_text())
                if Path(a.mechanisms).exists() else {})
    mechanisms = mech_doc.get("mechanisms") or []
    engine_margins = (json.loads(Path(a.engine_margins).read_text())
                      if a.engine_margins and Path(a.engine_margins).exists() else {})
    gate_report = (json.loads(Path(a.gate_report).read_text())
                   if a.gate_report and Path(a.gate_report).exists() else {})
    baseline_dir = Path(a.baseline)

    report = {
        "schema": SCHEMA,
        "criteria": {
            "source": ".pm/goal.md AC-3 (re-pinned 2026-07-29)",
            "agreement_floor": AGREEMENT_FLOOR,
            "expect_new_tokens": a.expect_new_tokens,
            "reps_required_per_bs": a.reps_required,
            "scored_reps": "EVERY rep that produced a dump (see module docstring)",
        },
        "inputs": {
            "reference": str(a.reference), "reps_root": str(a.reps_root),
            "baseline": str(a.baseline), "coherence": a.coherence,
            "mechanisms": a.mechanisms,
            "n_mechanism_entries": len(mechanisms),
            "engine_margins": a.engine_margins,
            "coherence_meta": {k: coherence.get(k) for k in
                               ("model_id", "revision", "git_sha", "utc",
                                "device", "tokenizer_vocab_len")},
        },
        "per_bs": {}, "verdict": None, "failures": [], "not_evaluable": [],
        "diagnostics": {},
    }

    all_fail, all_ne = [], []
    exact_total = exact_ok = 0
    degradations = []

    for bs in batch_sizes:
        bl_path = baseline_dir / f"bs{bs}.json"
        baseline = json.loads(bl_path.read_text()) if bl_path.exists() else None
        bs_rec = {"reps": [], "reps_scored": 0, "reps_error": 0,
                  "verdict": None, "failures": [], "not_evaluable": []}
        if baseline is None:
            bs_rec["not_evaluable"].append(f"exactness baseline missing: {bl_path}")
        for r in reps.get(bs, []):
            if r["status"] != "ok" or r["dump"] is None:
                bs_rec["reps_error"] += 1
                bs_rec["reps"].append({"tag": r["tag"], "status": r["status"],
                                       "note": r["note"], "scored": False})
                continue
            cases, rep_fail, rep_ne = [], [], []
            exact_cases = {}
            for pid in sorted(ref):
                got = (r["dump"].get(pid) or {}).get("token_ids")
                if got is None:
                    rep_fail.append(f"{pid}: absent from the rep's dump")
                    cases.append({"prompt_id": pid, "batch_size": bs,
                                  "pass": False, "failures": ["prompt_absent"]})
                    continue
                c = score_case(prompt_id=pid, batch_size=bs, engine_ids=got,
                               ref=ref[pid], coh_ref=coh_ref, coh_eng=coh_eng,
                               mechanisms=mechanisms, engine_margins=engine_margins,
                               expect_tokens=a.expect_new_tokens)
                cases.append(c)
                rep_fail += [f"{pid}: {f}" for f in c["failures"]]
                rep_ne += [f"{pid}: {x}" for x in c["not_evaluable"]]
                if baseline is not None:
                    exact = ((baseline.get(pid) or {}).get("token_ids") == got)
                    exact_cases[pid] = exact
                    exact_total += 1
                    exact_ok += 1 if exact else 0
                    if not exact:
                        degradations.append(
                            {"rep": r["tag"], "batch_size": bs, "prompt_id": pid,
                             "first_divergent_position": c.get("first_divergent_position"),
                             "still_passes_repinned_ac3": c["pass"],
                             "explained_by_mechanism": bool(
                                 c.get("differing_positions") and
                                 all(d["classification"] == "mechanism_documented"
                                     for d in c["differing_positions"]))})
            rep_rec = {"tag": r["tag"], "status": "ok", "scored": True,
                       "device": (r["meta"].get("device") or {}).get("phys_index"),
                       "gpu_before_mib": (r["meta"].get("gpu_before") or {}).get(
                           "memory_used_mib"),
                       "dump_md5": r["meta"].get("dump_md5"),
                       "dump_sha256": r.get("dump_sha256"),
                       "secs": r["meta"].get("secs"),
                       "exact_vs_dumps_final": exact_cases,
                       "n_exact": sum(1 for v in exact_cases.values() if v),
                       "n_cases": len(exact_cases),
                       "cases": cases,
                       "failures": rep_fail, "not_evaluable": rep_ne,
                       "pass": not rep_fail and not rep_ne}
            bs_rec["reps"].append(rep_rec)
            bs_rec["reps_scored"] += 1
            bs_rec["failures"] += [f"{r['tag']}: {f}" for f in rep_fail]
            bs_rec["not_evaluable"] += [f"{r['tag']}: {x}" for x in rep_ne]

        if bs_rec["reps_scored"] < a.reps_required:
            bs_rec["not_evaluable"].append(
                f"only {bs_rec['reps_scored']} scored rep(s), "
                f"{a.reps_required} required")
        bs_rec["verdict"] = ("FAIL" if bs_rec["failures"]
                             else "NOT_EVALUABLE" if bs_rec["not_evaluable"]
                             else "PASS")
        all_fail += [f"bs{bs}: {f}" for f in bs_rec["failures"]]
        all_ne += [f"bs{bs}: {x}" for x in bs_rec["not_evaluable"]]
        report["per_bs"][str(bs)] = bs_rec

    # ---- (c) no silent degradation -------------------------------------
    # Exactness is a DIAGNOSTIC under the re-pinned AC-3, so a degradation does
    # not fail the gate -- but the gate fails if the diagnostic could not be
    # computed at all, because AC-3(c) requires it to be reported every run.
    unexplained_deg = [d for d in degradations if not d["explained_by_mechanism"]]
    report["diagnostics"]["exactness"] = {
        "criterion": "goal AC-3(c): bit-exactness vs results/dumps_final is "
                     "measured and reported every run; it is a diagnostic, not a "
                     "pass condition",
        "cases_compared": exact_total, "cases_exact": exact_ok,
        "cases_degraded": len(degradations),
        "degradations": degradations,
        "degradations_unexplained": len(unexplained_deg),
        "computed": exact_total > 0,
    }
    if exact_total == 0:
        all_fail.append("AC-3(c): the bit-exactness diagnostic could not be "
                        "computed (no baseline/case comparisons) -- AC-3(c) "
                        "requires it to be reported every run")
    report["diagnostics"]["stability"] = {
        "note": "fingerprint divergence is an in-band phenomenon the re-pinned "
                "AC-3 tolerates and reports (goal Plan Evolution Log 2026-07-29)",
        "verdict": gate_report.get("verdict"),
        "totals": gate_report.get("totals"),
        "per_bs_divergence_rate": {k: v.get("divergence_rate")
                                   for k, v in (gate_report.get("per_bs") or {}).items()},
        "consensus_state_signature": {k: v.get("consensus_state_signature")
                                      for k, v in (gate_report.get("per_bs") or {}).items()},
    }

    report["failures"] = all_fail
    report["not_evaluable"] = all_ne
    report["verdict"] = ("FAIL" if all_fail else
                         "NOT_EVALUABLE" if all_ne else "PASS")

    out = Path(a.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    print(f"=== AC-3 (re-pinned) {report['verdict']} ===")
    for bs in batch_sizes:
        b = report["per_bs"][str(bs)]
        print(f"  bs={bs:<3} {b['verdict']:<14} reps scored {b['reps_scored']}"
              f" errors {b['reps_error']}"
              f"  exact {sum(r.get('n_exact', 0) for r in b['reps'] if r.get('scored'))}"
              f"/{sum(r.get('n_cases', 0) for r in b['reps'] if r.get('scored'))}")
        for f in b["failures"][:6]:
            print(f"      FAIL {f}")
        for x in b["not_evaluable"][:6]:
            print(f"      N/E  {x}")
    ex = report["diagnostics"]["exactness"]
    print(f"  AC-3(c) exactness diagnostic: {ex['cases_exact']}/{ex['cases_compared']} "
          f"cases byte-identical to dumps_final; degraded {ex['cases_degraded']} "
          f"({ex['degradations_unexplained']} unexplained)")
    print(f"  report -> {out}")
    return 0 if report["verdict"] == "PASS" else (1 if all_fail else 3)


if __name__ == "__main__":
    sys.exit(main())
