#!/usr/bin/env python3
"""M4-I0: pool every gate window into one AC-3-geometry cold-rep census table.

Each window is one gate run on one device; the gate VERDICT is per window. The
pooled table is only for the divergence-RATE statistic, which needs a bigger
denominator than any single window provides.

usage: pool_m4i0.py <out.json> <label>=<report.json> ...
"""
from __future__ import annotations
import json, sys
from pathlib import Path


def main() -> int:
    out = Path(sys.argv[1])
    rows, pooled = [], {"scored": 0, "accepted": 0, "quarantined": 0,
                        "errors": 0, "launched": 0, "token_reaching": 0,
                        "state_only": 0, "token_mismatch_reps": 0,
                        "device_grew": 0}
    per_bs_sig = {}
    events = []
    for arg in sys.argv[2:]:
        label, _, path = arg.partition("=")
        d = json.loads(Path(path).read_text())
        t = d["totals"]
        gpus = t["physical_gpus_used"]
        row = {"window": label, "gpus": gpus, "verdict": d["verdict"],
               "launched": t["reps_launched"], "scored": t["reps_scored"],
               "accepted": t["accepted"], "quarantined": t["quarantined"],
               "errors": t["run_errors"],
               "token_mismatch_reps": t["token_mismatch_reps"],
               "fp_div_rate": t["fingerprint_divergence_rate"],
               "token_reaching": t.get("quarantined_token_reaching", 0),
               "state_only": t.get("quarantined_state_only", 0),
               "device_grew": t.get("reps_starting_on_a_non_clean_device", 0),
               "batch_sizes": sorted(d["per_bs"], key=int),
               "reps_needed": {bs: d["per_bs"][bs]["reps_needed_to_reach_verdict"]
                               for bs in sorted(d["per_bs"], key=int)}}
        rows.append(row)
        for k in ("scored", "accepted", "quarantined", "errors", "launched",
                  "token_reaching", "state_only", "token_mismatch_reps",
                  "device_grew"):
            pooled[k] += row[k] if k != "launched" else row["launched"]
        for bs, b in d["per_bs"].items():
            per_bs_sig.setdefault(bs, {}).setdefault(
                b["consensus_state_signature"], []).append(label)
            for r in b["reps"]:
                if r["classification"] == "quarantined":
                    fd = r.get("fingerprint_delta_vs_consensus", {})
                    tv = r.get("tokens", {})
                    mm = tv.get("mismatched") or []
                    events.append({
                        "window": label, "gpu": (r.get("device") or {}).get("phys_index"),
                        "bs": int(bs), "rep": r["tag"],
                        "waves_touched": fd.get("waves_touched"),
                        "n_keys": fd.get("n_keys"),
                        "reached_tokens": bool(r.get("divergence_reached_tokens")),
                        "dump_md5": r["dump_md5"],
                        "token_mismatch": [
                            {"prompt": m["prompt_id"],
                             "first_divergent_position": m["first_divergent_position"]}
                            for m in mm],
                        "fp_keys": fd.get("keys"),
                    })

    s = pooled["scored"]
    pooled["fingerprint_divergence_rate"] = round(pooled["quarantined"] / s, 4) if s else None
    pooled["token_divergence_rate"] = round(pooled["token_mismatch_reps"] / s, 4) if s else None
    pooled["fraction_of_divergences_reaching_tokens"] = (
        round(pooled["token_reaching"] / pooled["quarantined"], 4)
        if pooled["quarantined"] else None)
    # one-sided 95% upper bound on a binomial rate given k events in n trials,
    # by the exact (Clopper-Pearson) rule; reported so a null window is readable
    if s:
        from math import isclose
        def cp_upper(k, n, alpha=0.05):
            lo, hi = 0.0, 1.0
            for _ in range(200):
                mid = (lo + hi) / 2
                # P(X <= k | p=mid) = alpha  ->  solve for mid
                from math import comb
                p = sum(comb(n, i) * mid**i * (1 - mid)**(n - i) for i in range(k + 1))
                if p > alpha:
                    lo = mid
                else:
                    hi = mid
            return (lo + hi) / 2
        pooled["fp_rate_95pct_upper_bound"] = round(
            cp_upper(pooled["quarantined"], s), 4)
        pooled["token_rate_95pct_upper_bound"] = round(
            cp_upper(pooled["token_mismatch_reps"], s), 4)

    consensus_stable = {bs: (len(sigs) == 1) for bs, sigs in per_bs_sig.items()}
    report = {"windows": rows, "pooled": pooled,
              "per_bs_consensus_signatures": per_bs_sig,
              "consensus_signature_identical_across_windows": consensus_stable,
              "divergence_events": events}
    out.write_text(json.dumps(report, indent=2))

    print(f"{'window':10} {'gpu':>4} {'verdict':9} {'launch':>6} {'scored':>6} "
          f"{'acc':>4} {'quar':>4} {'err':>4} {'tokmm':>5} {'rate':>7}")
    for r in rows:
        rate = "n/a" if r["fp_div_rate"] is None else f"{r['fp_div_rate']:.1%}"
        print(f"{r['window']:10} {str(r['gpus']):>4} {r['verdict']:9} "
              f"{r['launched']:>6} {r['scored']:>6} {r['accepted']:>4} "
              f"{r['quarantined']:>4} {r['errors']:>4} "
              f"{r['token_mismatch_reps']:>5} {rate:>7}")
    p = pooled
    print(f"\nPOOLED  launched {p['launched']}  scored {p['scored']}  "
          f"accepted {p['accepted']}  quarantined {p['quarantined']}  "
          f"run-errors {p['errors']}")
    print(f"  fingerprint divergence rate  {p['fingerprint_divergence_rate']:.2%}"
          f"   (95% upper bound {p.get('fp_rate_95pct_upper_bound', float('nan')):.2%})")
    print(f"  token   divergence rate      {p['token_divergence_rate']:.2%}"
          f"   (95% upper bound {p.get('token_rate_95pct_upper_bound', float('nan')):.2%})")
    frt = p["fraction_of_divergences_reaching_tokens"]
    print(f"  of {p['quarantined']} divergences: {p['token_reaching']} reached the "
          f"tokens, {p['state_only']} stayed sub-argmax"
          + ("" if frt is None else f"  ({frt:.0%} supra-argmax)"))
    print(f"  reps starting on a non-clean device: {p['device_grew']}")
    print(f"\nper-bs consensus fingerprint identical across ALL windows: "
          f"{consensus_stable}")
    for bs, sigs in sorted(per_bs_sig.items(), key=lambda kv: int(kv[0])):
        for sig, labels in sigs.items():
            print(f"  bs={bs:<3} {sig}  <- {' '.join(labels)}")
    print(f"\n{len(events)} divergence event(s):")
    for e in events:
        print(f"  {e['window']}/{e['rep']} GPU{e['gpu']} bs={e['bs']} "
              f"waves={e['waves_touched']} keys={e['n_keys']} "
              f"reached_tokens={e['reached_tokens']} md5={e['dump_md5']}")
        for m in e["token_mismatch"]:
            print(f"      token divergence: {m['prompt']} @ position "
                  f"{m['first_divergent_position']}")
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
