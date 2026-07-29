#!/usr/bin/env python3
"""Score AC-4 (decode throughput) and AC-5 (e2e latency) from collected
artifacts.  PURE: stdlib only, no GPU, no torch, no vLLM
(``tests/test_score_perf.py`` drives it on fixtures).

MEASUREMENT DEFINITIONS -- all inherited, none invented here
------------------------------------------------------------
* MPK decode tok/s is the PREFILL-SUBTRACTED SLOPE
  ``bs*(D_full - D_pre)/(wall_full - wall_pre)`` over two arms of the same
  prompts (full: msl 1280 / 1024 new tokens; pre: msl 259 / 2 new tokens).
  Binding per ``docs/qwen35/bench-protocol.md`` ("Decode-throughput
  measurement (M3-I7, binding)").  ``bs*1024/wave_wall`` is explicitly NOT the
  quantity: it bills the 256-token prefill to decode.
* MPK e2e is the full arm's wave wall -- prefill + decode of the whole batch,
  the same bracket vLLM's e2e uses (bench-protocol.md 5.2).
* vLLM decode tok/s / e2e come from ``bench_vllm.py``'s own artifacts, i.e.
  vLLM's own decode-window definition (bench-protocol.md 5.1).
* Dispersion bound 5% (full range / median) and the two-boot merge rule
  (merged IQR/median <= 5%, every boot median within 3% of the merged median,
  >= 6 total reps) are the pinned protocol's, imported from ``bench_vllm.py``
  so this scorer can never run on a looser bound than the benchmark tool.

THE AC-4 COMPARATOR AND THE DRIFT RULE  (design decision, see final/README.md)
-----------------------------------------------------------------------------
A FRESH vLLM sweep at the binding config, captured in the same window as the MPK
runs, is the PRIMARY comparator; the committed pinned baseline table is the
CROSS-CHECK.  The drift rule is the pinned protocol's own two-boot agreement
statistic applied to (pinned boots + this fresh boot): if they disagree beyond
it, the gate FAILS with both numbers rather than choosing one.  There is no code
path that picks the more favourable of the two.

VALIDITY ASYMMETRY (why this cannot be gamed by degrading the measurement)
-------------------------------------------------------------------------
A criterion can only be PASSED on a measurement that satisfies the protocol's
dispersion bound.  If the point estimate says PASS but validity is missing, the
result is NOT_EVALUABLE, never PASS.  If the point estimate says FAIL, it FAILS
regardless of dispersion -- a losing candidate needs no measurement defence.
So making the measurement noisier can never turn a FAIL into a PASS.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import statistics as st
import sys
from pathlib import Path

SCHEMA = "final/perf_score/v1"

#: Slack over the session's quietest observed pre-run sample on the pinned
#: device before a rep counts as "started on a non-clean device".  Same value
#: and same reasoning as ``harness/gate_ac3_stable.py``'s co-tenancy audit
#: (``mib_above_foreign_floor > 1024``), which is where it comes from.
DIRTY_SLACK_MIB = 1024


def load_bench_constants(bench_vllm_py: Path) -> dict:
    """Import the pinned bounds from ``bench_vllm.py`` (stdlib-only at module
    scope, so importing it does not need vllm/torch).  Fails closed: without the
    pinned constants this scorer refuses to score rather than fall back to
    numbers typed in here."""
    spec = importlib.util.spec_from_file_location("bench_vllm_consts", bench_vllm_py)
    if spec is None or spec.loader is None:
        raise SystemExit(f"INTEGRITY: cannot import pinned bounds from {bench_vllm_py}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    need = ("BINDING_INPUT_LEN", "BINDING_OUTPUT_LEN", "BINDING_BATCH_SIZES",
            "BINDING_MIN_REPS", "BINDING_MIN_WARMUP", "BINDING_MAX_DISPERSION_PCT",
            "BINDING_BOOT_MEDIAN_AGREE_PCT", "MODEL_ID_DEFAULT", "REVISION_DEFAULT")
    missing = [n for n in need if not hasattr(mod, n)]
    if missing:
        raise SystemExit(f"INTEGRITY: {bench_vllm_py} is missing pinned constants: {missing}")
    return {n: getattr(mod, n) for n in need}


# ------------------------------------------------------------------ stats ---
def _range_pct(vals):
    if not vals:
        return None
    m = st.median(vals)
    return None if m == 0 else (max(vals) - min(vals)) / m * 100.0


def _iqr_pct(vals):
    if len(vals) < 2:
        return None
    m = st.median(vals)
    q = st.quantiles(sorted(vals), n=4, method="inclusive")
    return None if m == 0 else (q[2] - q[0]) / m * 100.0


def merge_boots(boots, max_iqr_pct, max_boot_dev_pct, min_total=6) -> dict:
    """``bench_vllm.py --mode merge`` as a pure function: identical statistic,
    identical bounds.  ``boots`` = [{"name", "values"}]."""
    allv = [v for b in boots for v in b["values"]]
    out = {"rule": "bench-protocol.md §6 two-boot merge (IQR-based)",
           "boots": [{"name": b["name"], "n": len(b["values"]),
                      "median": (st.median(b["values"]) if b["values"] else None)}
                     for b in boots],
           "n_total": len(allv), "bounds": {"merged_iqr_over_median_pct": max_iqr_pct,
                                            "boot_median_max_deviation_pct": max_boot_dev_pct,
                                            "min_total_reps": min_total}}
    if len(allv) < min_total or any(not b["values"] for b in boots):
        out.update({"merged_median": (st.median(allv) if allv else None),
                    "valid": False,
                    "why": f"{len(allv)} total reps across "
                           f"{len(boots)} boot(s), need >= {min_total} and every boot non-empty"})
        return out
    med = st.median(allv)
    iqr = _iqr_pct(allv)
    dev = max(abs(st.median(b["values"]) - med) / med * 100.0 for b in boots)
    out.update({"merged_median": med, "merged_iqr_over_median_pct": iqr,
                "merged_fullrange_over_median_pct": _range_pct(allv),
                "boot_median_max_deviation_pct": dev,
                "valid": bool(iqr is not None and iqr <= max_iqr_pct
                              and dev <= max_boot_dev_pct)})
    if not out["valid"]:
        out["why"] = (f"merged IQR/median {iqr:.2f}% (bound {max_iqr_pct}%) / "
                      f"max boot-median deviation {dev:.2f}% "
                      f"(bound {max_boot_dev_pct}%)")
    return out


# -------------------------------------------------------------------- MPK ---
def _mpk_arm(root: Path, arm: str, bs: int, expect_msl: int, expect_mbt: int,
             floor_mib, audit: dict, reps_required: int,
             expect_cap=None, policy_snapshot=None) -> dict:
    """One arm (full|pre) at one batch size: per-rep walls + decode steps, with
    every geometry assertion the slope depends on checked explicitly."""
    rec = {"arm": arm, "expect_max_seq_length": expect_msl, "reps": [],
           "walls_ms": [], "decode_steps": [], "excluded": [], "problems": []}
    for p in sorted((root / arm).glob(f"timings_bs{bs}_rep*.json")):
        # collect_perf.sh names its per-cell audit records "<arm>_bs<N>_rep<R>";
        # the timings file is named by mpk_engine_run.py's --dump-name, which has
        # no arm in it.  Joining on the wrong key silently loses the device audit
        # (which is how M3-I7 charged a co-tenant's memory to the wrong device),
        # so the key is reconstructed explicitly here.
        tag = f"{arm}_" + p.stem.replace("timings_", "")
        try:
            d = json.loads(p.read_text())
        except Exception as e:                          # noqa: BLE001
            rec["problems"].append(f"{p.name}: unreadable ({type(e).__name__})")
            continue
        waves = d.get("waves") or []
        r = {"tag": tag, "file": p.name, "n_waves": len(waves)}
        if len(waves) != 1:
            r["excluded_why"] = (f"{len(waves)} waves -- the slope is defined on ONE "
                                 f"decode window; a multi-wave run's wall is not it")
            rec["excluded"].append(r)
            rec["problems"].append(f"{tag}: {r['excluded_why']}")
            continue
        w = waves[0]
        # The admission cap is a COMPILE-TIME define, so the value that actually
        # ran is the only one worth checking -- and the run records it itself
        # (mpk_engine_run.py's timings `admission_policy` block).  Checking the
        # collector's self-report instead would miss exactly the failure M3-I7
        # hit: a kernel dir reused across cap values, where the CPU-side replay
        # claims a difference the binary does not have.
        ap = d.get("admission_policy") or {}
        r["cap_compiled"] = ap.get("compiled_cap")
        r["cap_requested"] = ap.get("requested")
        r["cap_authority"] = ap.get("authority")
        if expect_cap is not None and r["cap_compiled"] != expect_cap:
            r["excluded_why"] = (
                f"compiled admission cap {r['cap_compiled']!r} != the shipped "
                f"policy's {expect_cap!r} for bs{bs} "
                f"(authority: {ap.get('authority') or 'accept/admission_policy.py'})")
            rec["excluded"].append(r)
            rec["problems"].append(f"{tag}: {r['excluded_why']}")
            continue
        if policy_snapshot and ap and any(
                ap.get(k) != policy_snapshot.get(k)
                for k in ("cap_mode", "cap_min_batch_size")):
            rec["problems"].append(
                f"{tag}: the run's admission policy "
                f"{ {k: ap.get(k) for k in ('cap_mode', 'cap_min_batch_size')} } "
                f"differs from the policy this scorer resolved against "
                f"{ {k: policy_snapshot.get(k) for k in ('cap_mode', 'cap_min_batch_size')} }")
        r.update({"wall_ms": w.get("wall_ms"), "D": w.get("max_decode_steps"),
                  "max_seq_length": w.get("max_seq_length"),
                  "mbt": w.get("max_num_batched_tokens"),
                  "num_requests": w.get("num_requests"),
                  "num_distinct_prompts": w.get("num_distinct_prompts"),
                  "prompt_ids": w.get("prompt_ids")})
        geo = []
        if r["max_seq_length"] != expect_msl:
            geo.append(f"max_seq_length {r['max_seq_length']} != {expect_msl}")
        if r["mbt"] != expect_mbt:
            geo.append(f"mbt {r['mbt']} != {expect_mbt}")
        if r["num_requests"] != bs or r["num_distinct_prompts"] != bs:
            geo.append(f"wave holds {r['num_requests']} slots / "
                       f"{r['num_distinct_prompts']} distinct prompts, expected {bs}/{bs}"
                       " -- padding by repetition would make this not a bs-wide window")
        if geo:
            r["excluded_why"] = "; ".join(geo)
            rec["excluded"].append(r)
            rec["problems"].append(f"{tag}: {r['excluded_why']}")
            continue
        au = audit.get(tag) or {}
        r["gpu_before_mib"] = au.get("gpu_before_mib")
        r["gpu_index"] = au.get("gpu_index")
        if (floor_mib is not None and r["gpu_before_mib"] is not None
                and r["gpu_before_mib"] - floor_mib > DIRTY_SLACK_MIB):
            r["excluded_why"] = (f"started on a non-clean device: "
                                 f"{r['gpu_before_mib']} MiB resident, "
                                 f"{r['gpu_before_mib'] - floor_mib} MiB above the "
                                 f"session's foreign floor {floor_mib} MiB")
            rec["excluded"].append(r)
            continue
        if r["wall_ms"] is None or r["D"] is None:
            r["excluded_why"] = "no wall_ms / max_decode_steps in the wave record"
            rec["excluded"].append(r)
            rec["problems"].append(f"{tag}: {r['excluded_why']}")
            continue
        rec["reps"].append(r)
        rec["walls_ms"].append(r["wall_ms"])
        rec["decode_steps"].append(r["D"])
    rec["n"] = len(rec["reps"])
    rec["median_wall_ms"] = st.median(rec["walls_ms"]) if rec["walls_ms"] else None
    rec["median_D"] = st.median(rec["decode_steps"]) if rec["decode_steps"] else None
    rec["distinct_D"] = sorted(set(rec["decode_steps"]))
    rec["range_pct"] = _range_pct(rec["walls_ms"])
    rec["reps_required"] = reps_required
    if rec["n"] < reps_required:
        rec["problems"].append(f"n={rec['n']} < {reps_required} required reps "
                               f"(bench-protocol.md 6)")
    return rec


def score_mpk(root: Path, bs: int, policy, consts: dict,
              reps_required: int, msl_full: int, msl_pre: int,
              expect_mbt: int) -> dict:
    audit_path = root / "audit.json"
    audit_doc = json.loads(audit_path.read_text()) if audit_path.exists() else {}
    audit = audit_doc.get("cells") or {}
    floor = audit_doc.get("foreign_floor_mib")
    # DERIVED, never restated: the expected compile-time cap comes from
    # accept/admission_policy.py, which owns the policy.
    expect_cap = policy.resolve_int("policy", expect_mbt, bs)
    snap = policy.summary()
    full = _mpk_arm(root, "full", bs, msl_full, expect_mbt, floor, audit,
                    reps_required, expect_cap, snap)
    pre = _mpk_arm(root, "pre", bs, msl_pre, expect_mbt, floor, audit,
                   reps_required, expect_cap, snap)
    rec = {"batch_size": bs, "full": full, "pre": pre,
           "device": audit_doc.get("device"), "foreign_floor_mib": floor,
           "admission_policy_expected": {"compiled_cap": expect_cap, **snap},
           "admission_policy_recorded_by_collector": audit_doc.get("admission_policy"),
           "problems": list(full["problems"]) + list(pre["problems"])}
    dec = e2e = None
    if (full["median_wall_ms"] and pre["median_wall_ms"] and full["median_D"]
            is not None and pre["median_D"] is not None):
        dtok = bs * (full["median_D"] - pre["median_D"])
        dsec = (full["median_wall_ms"] - pre["median_wall_ms"]) / 1000.0
        if dsec > 0 and dtok > 0:
            dec = dtok / dsec
        else:
            rec["problems"].append(
                f"degenerate slope: D_full-D_pre={full['median_D'] - pre['median_D']}, "
                f"wall_full-wall_pre={dsec:.4f}s")
        e2e = full["median_wall_ms"] / 1000.0
    rec["decode_tokens_per_second"] = dec
    rec["e2e_seconds"] = e2e
    rec["decode_slope_formula"] = "bs*(D_full - D_pre)/(wall_full - wall_pre)"
    # Dispersion: the protocol's own bound (5%), applied SYMMETRICALLY to the MPK
    # arms because the two sides are being compared on the same quantity, with the
    # protocol's own escalation statistic once an arm has enough reps to use it:
    # single-run sets are bounded on FULL RANGE / median (bench-protocol.md 6) and
    # sets of >= 6 reps on IQR / median, because range is monotone in rep count and
    # one outlier would otherwise dominate a grown set forever (protocol 6,
    # corrected 2026-07-25).  The documented remedy for an arm over the bound is
    # therefore to ADD reps, never to drop one.
    bound = consts["BINDING_MAX_DISPERSION_PCT"]
    disp = {"bound_pct": bound, "arms": {}}
    ok = True
    for name, arm in (("full", full), ("pre", pre)):
        vals = arm["walls_ms"]
        stat = "iqr_over_median_pct" if len(vals) >= 6 else "fullrange_over_median_pct"
        val = _iqr_pct(vals) if len(vals) >= 6 else arm["range_pct"]
        disp["arms"][name] = {"n": len(vals), "statistic": stat, "value_pct": val,
                             "full_range_pct": arm["range_pct"],
                             "iqr_pct": _iqr_pct(vals),
                             "ok": bool(val is not None and val <= bound)}
        ok = ok and disp["arms"][name]["ok"]
    disp["ok"] = ok
    rec["dispersion"] = disp
    rec["measurement_valid"] = bool(
        dec is not None and e2e is not None and not rec["problems"]
        and rec["dispersion"]["ok"]
        and full["n"] >= reps_required and pre["n"] >= reps_required)
    return rec


# ------------------------------------------------------------------ vLLM ---
def _vllm_reps(path: Path) -> dict:
    d = json.loads(path.read_text())
    reps = d.get("reps") or []
    return {"file": str(path), "n": len(reps),
            "decode": [r["decode_tokens_per_second"] for r in reps],
            "e2e": [r["e2e_wall_seconds"] for r in reps],
            "summary": d.get("summary"), "shared_meta": d.get("shared_meta"),
            "discarded_reps": len(d.get("discarded_reps") or [])}


def score_vllm(fresh_dir: Path, pinned_dir: Path, bs: int, consts: dict) -> dict:
    """Fresh boot = primary; pinned capture = cross-check; drift = the pinned
    protocol's own two-boot agreement statistic over both."""
    rec = {"batch_size": bs, "problems": []}
    fp = fresh_dir / f"bs{bs}.json"
    if not fp.exists():
        rec["problems"].append(f"fresh vLLM artifact missing: {fp}")
        rec["measurement_valid"] = False
        return rec
    fresh = _vllm_reps(fp)
    rec["fresh"] = {k: fresh[k] for k in ("file", "n", "decode", "e2e",
                                          "discarded_reps")}
    sm = fresh.get("shared_meta") or {}
    cli = sm.get("cli_args") or {}
    ea = sm.get("engine_assertions") or {}
    rec["fresh_identity"] = {
        "vllm_version": (sm.get("versions") or {}).get("vllm"),
        "model_id": sm.get("model_id"), "revision": sm.get("revision"),
        "input_len": cli.get("input_len"), "output_len": cli.get("output_len"),
        "reps": cli.get("reps"), "warmup_reps": cli.get("warmup_reps"),
        "language_model_only": cli.get("language_model_only"),
        "max_dispersion_pct": cli.get("max_dispersion_pct"),
        "dispersion_ok": (fresh.get("summary") or {}).get("dispersion_ok"),
    }
    # ---- pinned capture: every boot directory that holds this bs -------
    boots = []
    for sub in sorted(p for p in pinned_dir.iterdir() if p.is_dir()):
        f = sub / f"bs{bs}.json"
        if f.exists():
            b = _vllm_reps(f)
            boots.append({"name": f"pinned/{sub.name}", "decode": b["decode"],
                          "e2e": b["e2e"], "n": b["n"]})
    rec["pinned_boots"] = [{"name": b["name"], "n": b["n"],
                            "decode_median": st.median(b["decode"]) if b["decode"] else None,
                            "e2e_median": st.median(b["e2e"]) if b["e2e"] else None}
                           for b in boots]
    merged_path = pinned_dir / f"bs{bs}.merged.json"
    if merged_path.exists():
        m = json.loads(merged_path.read_text())
        rec["pinned_binding"] = {
            "source": merged_path.name,
            "decode": m.get("decode_tokens_per_second_median"),
            "e2e": m.get("e2e_wall_seconds_median"),
            "binding_valid": m.get("binding_valid")}
        if not m.get("binding_valid"):
            rec["problems"].append(f"{merged_path.name}: pinned capture is not "
                                   f"binding-valid")
    else:
        pf = pinned_dir / "full" / f"bs{bs}.json"
        if pf.exists():
            b = _vllm_reps(pf)
            rec["pinned_binding"] = {
                "source": "full/bs%d.json" % bs,
                "decode": st.median(b["decode"]) if b["decode"] else None,
                "e2e": st.median(b["e2e"]) if b["e2e"] else None,
                "binding_valid": (b.get("summary") or {}).get("dispersion_ok")}
        else:
            rec["problems"].append(f"pinned baseline for bs{bs} not found under {pinned_dir}")
            rec["pinned_binding"] = None

    # ---- the drift rule --------------------------------------------------
    all_boots_dec = ([{"name": b["name"], "values": b["decode"]} for b in boots]
                     + [{"name": "fresh", "values": fresh["decode"]}])
    all_boots_e2e = ([{"name": b["name"], "values": b["e2e"]} for b in boots]
                     + [{"name": "fresh", "values": fresh["e2e"]}])
    max_iqr = consts["BINDING_MAX_DISPERSION_PCT"]
    max_dev = consts["BINDING_BOOT_MEDIAN_AGREE_PCT"]
    rec["drift"] = {
        "primary": "fresh",
        "policy": "fresh sweep is the binding comparator; the pinned capture is "
                  "the cross-check; disagreement beyond the protocol's own "
                  "two-boot statistic FAILS the gate (it never selects the "
                  "favourable number)",
        "decode": merge_boots(all_boots_dec, max_iqr, max_dev),
        "e2e": merge_boots(all_boots_e2e, max_iqr, max_dev),
    }
    # A numeric disagreement between the fresh boot and the pinned capture is a
    # FAILURE, not a missing measurement: the gate must not choose between two
    # comparators that disagree.  The message carries the word DRIFT, which is
    # what routes it into the report's failures (see main()).
    for series in ("decode", "e2e"):
        m = rec["drift"][series]
        if not m["valid"]:
            rec["problems"].append(
                "COMPARATOR DRIFT (%s): the fresh vLLM boot and the pinned capture "
                "disagree beyond the protocol's own two-boot statistic -- %s. Boot "
                "medians: %s. The gate refuses to pick a number; re-capture and "
                "re-pin the baseline (a human decision) or root-cause the box."
                % (series, m.get("why", "not valid"),
                   json.dumps({b["name"]: b["median"] for b in m["boots"]})))

    # Identity drift: a different vLLM build/config is a different comparator,
    # not a noisier measurement of the same one.
    pinned_meta_path = pinned_dir / "full" / "summary.json"
    if pinned_meta_path.exists():
        psm = (json.loads(pinned_meta_path.read_text()).get("shared_meta") or {})
        pcli = psm.get("cli_args") or {}
        want = {"vllm_version": (psm.get("versions") or {}).get("vllm"),
                "model_id": psm.get("model_id"), "revision": psm.get("revision"),
                "input_len": pcli.get("input_len"), "output_len": pcli.get("output_len"),
                "language_model_only": pcli.get("language_model_only")}
        got = {k: rec["fresh_identity"].get(k) for k in want}
        rec["identity_cross_check"] = {"pinned": want, "fresh": got,
                                       "match": want == got}
        if want != got:
            diffs = {k: (want[k], got[k]) for k in want if want[k] != got[k]}
            rec["problems"].append(
                "BASELINE IDENTITY DRIFT: the fresh vLLM run is not the same "
                "comparator as the pinned capture "
                + json.dumps({k: {"pinned": v[0], "fresh": v[1]} for k, v in diffs.items()})
                + " -- re-capture and re-pin the baseline table (a human decision), "
                  "do not compare across identities")
    else:
        rec["problems"].append(f"pinned baseline summary missing: {pinned_meta_path}")

    # ---- pinned-contract checks on the fresh run ------------------------
    fi = rec["fresh_identity"]
    if fi["input_len"] != consts["BINDING_INPUT_LEN"] or \
            fi["output_len"] != consts["BINDING_OUTPUT_LEN"]:
        rec["problems"].append(
            f"fresh workload {fi['input_len']}/{fi['output_len']} != pinned "
            f"{consts['BINDING_INPUT_LEN']}/{consts['BINDING_OUTPUT_LEN']}")
    if (fi["reps"] or 0) < consts["BINDING_MIN_REPS"]:
        rec["problems"].append(f"fresh reps {fi['reps']} < {consts['BINDING_MIN_REPS']}")
    if (fi["warmup_reps"] or 0) < consts["BINDING_MIN_WARMUP"]:
        rec["problems"].append(f"fresh warmup_reps {fi['warmup_reps']} < "
                               f"{consts['BINDING_MIN_WARMUP']}")
    if fi["model_id"] != consts["MODEL_ID_DEFAULT"] or \
            fi["revision"] != consts["REVISION_DEFAULT"]:
        rec["problems"].append(f"fresh model identity {fi['model_id']}@{fi['revision']} "
                               f"!= pinned {consts['MODEL_ID_DEFAULT']}@"
                               f"{consts['REVISION_DEFAULT']}")
    missing_assert = [k for k, v in (ea or {}).items()
                      if k.startswith("assert_") and v is not True]
    if missing_assert:
        rec["problems"].append(f"fresh run's fp8/fairness assertions not all true: "
                               f"{missing_assert}")
    if fi["dispersion_ok"] is False:
        rec["problems"].append("fresh run's own dispersion check failed "
                               "(bench-protocol.md 6)")

    rec["decode_tokens_per_second"] = (st.median(fresh["decode"])
                                       if fresh["decode"] else None)
    rec["e2e_seconds"] = st.median(fresh["e2e"]) if fresh["e2e"] else None
    rec["measurement_valid"] = bool(
        rec["decode_tokens_per_second"] is not None
        and rec["e2e_seconds"] is not None
        and not rec["problems"]
        and rec["drift"]["decode"]["valid"] and rec["drift"]["e2e"]["valid"])
    return rec


# ----------------------------------------------------------------- verdict --
def _criterion(name, passes, valid, detail) -> dict:
    """PASS needs both the point estimate AND a valid measurement; a failing
    point estimate fails regardless (see module docstring)."""
    if not passes:
        v = "FAIL"
    elif not valid:
        v = "NOT_EVALUABLE"
    else:
        v = "PASS"
    return dict(criterion=name, verdict=v, point_estimate_passes=passes,
                measurement_valid=valid, **detail)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mpk-root", required=True)
    ap.add_argument("--vllm-fresh", required=True)
    ap.add_argument("--vllm-pinned", required=True)
    ap.add_argument("--bench-vllm", required=True)
    ap.add_argument("--batch-sizes", default="1,2,4,8,16")
    ap.add_argument("--e2e-factor-max", type=float, default=1.25)
    ap.add_argument("--reps-required", type=int, default=3)
    ap.add_argument("--msl-full", type=int, default=1280)
    ap.add_argument("--msl-pre", type=int, default=259)
    ap.add_argument("--mbt", type=int, default=16)
    ap.add_argument("--output-json", required=True)
    a = ap.parse_args(argv)

    consts = load_bench_constants(Path(a.bench_vllm))
    batch_sizes = [int(x) for x in a.batch_sizes.split(",") if x.strip()]
    bad = [b for b in batch_sizes if b not in consts["BINDING_BATCH_SIZES"]]
    if bad:
        raise SystemExit(f"INTEGRITY: batch sizes {bad} are not in the pinned set "
                         f"{sorted(consts['BINDING_BATCH_SIZES'])}")
    # THE admission-cap authority (accept/admission_policy.py).  Imported, not
    # restated: a policy written down twice is a divergence bug.
    sys.path.insert(0, str(Path(a.bench_vllm).resolve().parent))
    try:
        import admission_policy as policy                # noqa: E402
    except Exception as e:                               # noqa: BLE001
        raise SystemExit(f"INTEGRITY: cannot import the admission-cap authority "
                         f"accept/admission_policy.py: {type(e).__name__}: {e}")

    report = {"schema": SCHEMA,
              "definitions": {
                  "mpk_decode": "bs*(D_full - D_pre)/(wall_full - wall_pre) "
                                "[bench-protocol.md, M3-I7 binding]",
                  "mpk_e2e": "full arm wave wall (prefill+decode of the batch)",
                  "vllm": "bench_vllm.py artifacts [bench-protocol.md 5.1/5.2]",
                  "workload": f"{consts['BINDING_INPUT_LEN']}/"
                              f"{consts['BINDING_OUTPUT_LEN']}, identical at every bs",
                  "ac5_factor_max": a.e2e_factor_max,
                  "pinned_bounds": {k: consts[k] for k in
                                    ("BINDING_MAX_DISPERSION_PCT",
                                     "BINDING_BOOT_MEDIAN_AGREE_PCT",
                                     "BINDING_MIN_REPS", "BINDING_MIN_WARMUP")},
                  "admission_policy": policy.summary()},
              "per_bs": {}, "ac4": {}, "ac5": {},
              "failures": [], "not_evaluable": [], "verdict": None}

    ac4_verdicts, ac5_verdicts = [], []
    for bs in batch_sizes:
        mpk = score_mpk(Path(a.mpk_root), bs, policy, consts, a.reps_required,
                        a.msl_full, a.msl_pre, a.mbt)
        vllm = score_vllm(Path(a.vllm_fresh), Path(a.vllm_pinned), bs, consts)
        md, vd = mpk["decode_tokens_per_second"], vllm["decode_tokens_per_second"]
        me, ve = mpk["e2e_seconds"], vllm["e2e_seconds"]
        valid = mpk["measurement_valid"] and vllm["measurement_valid"]

        ac4 = _criterion(
            f"AC-4 bs{bs}: mpk decode tok/s STRICTLY GREATER than vLLM",
            bool(md is not None and vd is not None and md > vd),
            bool(valid and md is not None and vd is not None),
            {"mpk_decode_tok_s": md, "vllm_decode_tok_s": vd,
             "ratio_mpk_over_vllm": (md / vd) if (md and vd) else None,
             "vllm_pinned_cross_check": (vllm.get("pinned_binding") or {}).get("decode"),
             "drift_valid": (vllm.get("drift") or {}).get("decode", {}).get("valid")})
        ac5 = _criterion(
            f"AC-5 bs{bs}: mpk e2e <= {a.e2e_factor_max}x vLLM e2e",
            bool(me is not None and ve not in (None, 0)
                 and (me / ve) <= a.e2e_factor_max),
            bool(valid and me is not None and ve not in (None, 0)),
            {"mpk_e2e_s": me, "vllm_e2e_s": ve,
             "ratio_mpk_over_vllm": (me / ve) if (me and ve) else None,
             "max_ratio": a.e2e_factor_max,
             "vllm_pinned_cross_check": (vllm.get("pinned_binding") or {}).get("e2e"),
             "drift_valid": (vllm.get("drift") or {}).get("e2e", {}).get("valid")})
        report["per_bs"][str(bs)] = {"mpk": mpk, "vllm": vllm,
                                     "ac4": ac4, "ac5": ac5}
        ac4_verdicts.append((bs, ac4))
        ac5_verdicts.append((bs, ac5))
        for src, problems in (("mpk", mpk["problems"]), ("vllm", vllm["problems"])):
            for p in problems:
                (report["failures"] if "DRIFT" in p else report["not_evaluable"]
                 ).append(f"bs{bs} {src}: {p}")

    def roll(name, items):
        vs = [v for _, v in items]
        verdict = ("FAIL" if any(v["verdict"] == "FAIL" for v in vs)
                   else "NOT_EVALUABLE" if any(v["verdict"] == "NOT_EVALUABLE" for v in vs)
                   else "PASS" if vs else "NOT_EVALUABLE")
        return {"verdict": verdict,
                "per_bs": {str(bs): v["verdict"] for bs, v in items},
                "criterion": name}

    report["ac4"] = roll("mpk decode tok/s > vLLM at EVERY batch size (goal AC-4)",
                         ac4_verdicts)
    report["ac5"] = roll(f"mpk e2e <= {a.e2e_factor_max}x vLLM at EVERY batch size "
                         f"(goal AC-5)", ac5_verdicts)
    for bs, v in ac4_verdicts + ac5_verdicts:
        if v["verdict"] == "FAIL":
            numbers = {k: v[k] for k in v
                       if k.startswith(("mpk_", "vllm_", "ratio", "max_ratio"))}
            report["failures"].append(f"{v['criterion']}: " + json.dumps(numbers))
        elif v["verdict"] == "NOT_EVALUABLE":
            report["not_evaluable"].append(f"{v['criterion']}: measurement not valid")

    report["verdict"] = ("FAIL" if report["ac4"]["verdict"] == "FAIL"
                         or report["ac5"]["verdict"] == "FAIL"
                         or any("DRIFT" in f for f in report["failures"])
                         else "NOT_EVALUABLE"
                         if "NOT_EVALUABLE" in (report["ac4"]["verdict"],
                                                report["ac5"]["verdict"])
                         else "PASS")

    out = Path(a.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    print(f"=== AC-4 {report['ac4']['verdict']} | AC-5 {report['ac5']['verdict']} ===")
    print(f"{'bs':>3} {'mpk tok/s':>10} {'vllm tok/s':>11} {'ratio':>7} {'AC-4':>13} "
          f"{'mpk e2e':>8} {'vllm e2e':>9} {'x':>6} {'AC-5':>13}")
    for bs in batch_sizes:
        r = report["per_bs"][str(bs)]
        a4, a5 = r["ac4"], r["ac5"]
        def f(x, w, p=1):
            return " " * w if x is None else f"{x:{w}.{p}f}"
        print(f"{bs:>3} {f(a4['mpk_decode_tok_s'],10)} {f(a4['vllm_decode_tok_s'],11)} "
              f"{f(a4['ratio_mpk_over_vllm'],7,3)} {a4['verdict']:>13} "
              f"{f(a5['mpk_e2e_s'],8,2)} {f(a5['vllm_e2e_s'],9,2)} "
              f"{f(a5['ratio_mpk_over_vllm'],6,2)} {a5['verdict']:>13}")
    for f_ in report["failures"]:
        print(f"  FAIL {f_}")
    for x in report["not_evaluable"][:20]:
        print(f"  N/E  {x}")
    print(f"  report -> {out}")
    return 0 if report["verdict"] == "PASS" else (1 if report["verdict"] == "FAIL" else 3)


if __name__ == "__main__":
    sys.exit(main())
