#!/usr/bin/env python3
"""M4-I4 -- assemble the admission-cap A/B tables from the campaign's raw runs.

Two geometries, two arms, >=3 reps, arms interleaved inside ONE GPU claim:

  geomA  AC-3 geometry (msl=132, 64 new tokens, the 10 pinned reference prompts).
         The metric is the SUM of the wave walls, because bs<16 runs
         ceil(10/bs) waves and only the sum is the same amount of work at every
         batch size.
  geomM  pinned 256/1024 benchmark geometry. e2e = the full run's wave wall;
         decode tok/s = bs*(D_full - D_pre)/(wall_full - wall_pre), the
         prefill-subtracted slope that matches vLLM's own
         tokens-over-decode-window definition (bench-protocol.md).

The vLLM comparison numbers are READ from the committed baseline artifacts, never
hardcoded here: `baselines/<v>/full/summary.json` for bs1/2/4 and the two-boot
merges `bs8.merged.json` / `bs16.merged.json` for bs8/bs16 (bench-protocol.md 6).

Also reports, per arm, the CPU-side admission replay's `predicted_iterations`
from each run's own timings artifact, so the measured win can be split into the
two mechanisms it comes from: fewer iterations, and a wider task graph per
iteration.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
from pathlib import Path

ARMS = ("none", "auto")
BSS = (1, 2, 4, 8, 16)
AC5_BOUND = 1.25            # .pm/goal.md AC-5: mpk e2e <= 1.25x vLLM e2e

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))   # .../accept
import admission_policy as _policy   # noqa: E402  -- THE policy authority


def admission_policy_on(batch_size: int) -> bool:
    """Is the SHIPPED policy capping this batch size? Read from the authority."""
    return _policy.policy_cap(batch_size) is not None


# --------------------------------------------------------------- baseline ---
def load_vllm(accept: Path, version_dir: str | None = None) -> dict:
    base = accept / "baselines"
    cands = sorted(p for p in base.iterdir() if p.is_dir()) if base.is_dir() else []
    if version_dir:
        d = base / version_dir
    elif len(cands) == 1:
        d = cands[0]
    else:
        raise SystemExit(f"pass --vllm-dir; candidates: {[c.name for c in cands]}")
    table = json.load(open(d / "full" / "summary.json"))["table"]
    out = {}
    for bs in BSS:
        row = table[str(bs)]
        out[bs] = {
            "decode_tok_s": row["decode_tokens_per_second"]["median"],
            "e2e_s": row["e2e_latency_seconds"]["median"],
            "source": f"{d.name}/full/summary.json",
            "n": row["decode_tokens_per_second"]["n"],
        }
    for bs in (8, 16):
        p = d / f"bs{bs}.merged.json"
        if p.is_file():
            m = json.load(open(p))
            out[bs] = {
                "decode_tok_s": m["decode_tokens_per_second_median"],
                "e2e_s": m["e2e_wall_seconds_median"],
                "source": f"{d.name}/bs{bs}.merged.json (two-boot merge)",
                "n": m["n_total"],
            }
    return out


# ------------------------------------------------------------------ stats ---
def med_range(vals):
    v = [x for x in vals if x is not None]
    if not v:
        return dict(median=None, lo=None, hi=None, n=0, range_pct=None, reps=[])
    m = st.median(v)
    return dict(median=m, lo=min(v), hi=max(v), n=len(v),
                range_pct=(100.0 * (max(v) - min(v)) / m if m else None),
                reps=v)


def _waves(path: Path):
    try:
        return json.load(open(path))
    except Exception:
        return None


# ------------------------------------------------------------------ geomA ---
def geomA(root: Path, reps: int) -> dict:
    """AC-3 geometry: total wave wall per (arm, bs), per rep."""
    out = {}
    for arm in ARMS:
        for bs in BSS:
            walls, iters, caps = [], [], []
            for r in range(reps):
                d = _waves(root / "geomA" / arm / f"rep{r}" / f"timings_bs{bs}.json")
                if not d:
                    continue
                walls.append(sum(w["wall_ms"] for w in d["waves"]))
                iters.append(sum((w.get("compaction") or {}).get(
                    "predicted_iterations") or 0 for w in d["waves"]))
                caps.append((d.get("admission_policy") or {}).get("compiled_cap"))
            out[f"bs{bs}_{arm}"] = dict(
                bs=bs, arm=arm, wall_ms=med_range(walls),
                predicted_iterations=med_range(iters),
                compiled_cap=(caps[0] if caps else None),
                compiled_cap_consistent=(len(set(map(str, caps))) <= 1),
                n_waves=(len(_waves(root / "geomA" / arm / f"rep0" /
                                    f"timings_bs{bs}.json")["waves"])
                         if _waves(root / "geomA" / arm / "rep0" /
                                   f"timings_bs{bs}.json") else None))
    return out


# ------------------------------------------------------------------ geomM ---
def geomM(root: Path, reps: int) -> dict:
    out = {}
    for arm in ARMS:
        for bs in BSS:
            rec = dict(bs=bs, arm=arm)
            for cfg in ("full", "pre"):
                walls, steps, iters, caps = [], [], [], []
                for r in range(reps):
                    d = _waves(root / "geomM" / arm / cfg /
                               f"timings_bs{bs}_rep{r}.json")
                    if not d or not d.get("waves"):
                        continue
                    w = d["waves"][0]
                    walls.append(w["wall_ms"])
                    steps.append(w["max_decode_steps"])
                    iters.append((w.get("compaction") or {}).get(
                        "predicted_iterations"))
                    caps.append((d.get("admission_policy") or {}).get("compiled_cap"))
                rec[cfg] = dict(wall_ms=med_range(walls),
                                decode_steps=(st.median(steps) if steps else None),
                                predicted_iterations=med_range(iters),
                                compiled_cap=(caps[0] if caps else None),
                                compiled_cap_consistent=(
                                    len(set(map(str, caps))) <= 1))
            f, p = rec["full"], rec["pre"]
            if (f["wall_ms"]["median"] and p["wall_ms"]["median"]
                    and f["decode_steps"] is not None and p["decode_steps"] is not None):
                dw = (f["wall_ms"]["median"] - p["wall_ms"]["median"]) / 1000.0
                rec["decode_tok_s"] = (bs * (f["decode_steps"] - p["decode_steps"]) / dw
                                       if dw > 0 else None)
                rec["e2e_s"] = f["wall_ms"]["median"] / 1000.0
                rec["prefill_s"] = p["wall_ms"]["median"] / 1000.0
                rec["prefill_frac_of_e2e"] = (rec["prefill_s"] / rec["e2e_s"]
                                              if rec["e2e_s"] else None)
            else:
                rec["decode_tok_s"] = rec["e2e_s"] = rec["prefill_s"] = None
                rec["prefill_frac_of_e2e"] = None
            out[f"bs{bs}_{arm}"] = rec
    return out


# ------------------------------------------------------------------ print ---
def fmt(x, w=9, p=1):
    return " " * w if x is None else f"{x:{w}.{p}f}"


def ratio(a, b):
    return None if not (a and b) else a / b


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.path.expanduser("~/mpk-qwen35/m4i4"))
    ap.add_argument("--accept", default=str(Path(__file__).resolve().parents[3]))
    ap.add_argument("--vllm-dir", default=None)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-csv-prefix", default=None)
    a = ap.parse_args(argv)

    root = Path(a.root)
    vllm = load_vllm(Path(a.accept), a.vllm_dir)
    A = geomA(root, a.reps)
    M = geomM(root, a.reps)

    print("=" * 100)
    print("1. AC-3 GEOMETRY (msl=132, 64 new tokens, 10 pinned prompts) -- total wave wall")
    print("=" * 100)
    print(f"{'bs':>3} {'cap':>4} | {'uncapped ms':>12} {'r%':>5} {'n':>2} | "
          f"{'capped ms':>10} {'r%':>5} {'n':>2} | {'speedup':>8} | "
          f"{'replay iters u/c':>18}")
    for bs in BSS:
        u, c = A[f"bs{bs}_none"], A[f"bs{bs}_auto"]
        sp = ratio(u["wall_ms"]["median"], c["wall_ms"]["median"])
        iu = u["predicted_iterations"]["median"]
        ic = c["predicted_iterations"]["median"]
        ir = f"{iu:.0f}/{ic:.0f} = {iu / ic:.3f}x" if iu and ic else ""
        print(f"{bs:3d} {str(c['compiled_cap']):>4} |{fmt(u['wall_ms']['median'], 12)} "
              f"{fmt(u['wall_ms']['range_pct'], 5, 2)} {u['wall_ms']['n']:2d} |"
              f"{fmt(c['wall_ms']['median'], 10)} {fmt(c['wall_ms']['range_pct'], 5, 2)} "
              f"{c['wall_ms']['n']:2d} |{fmt(sp, 8, 3)} | {ir:>18}")

    print()
    print("=" * 118)
    print("2. PINNED 256/1024 GEOMETRY -- per-arm compiled kernels, arms interleaved in one claim")
    print("=" * 118)
    print(f"{'bs':>3} {'arm':>5} {'cap':>4} | {'e2e s':>7} {'r%':>5} {'n':>2} | "
          f"{'prefill s':>9} {'r%':>5} {'n':>2} | {'pre/e2e':>7} | "
          f"{'decode tok/s':>12} | {'replay it':>9}")
    for bs in BSS:
        for arm in ARMS:
            r = M[f"bs{bs}_{arm}"]
            f_, p_ = r["full"]["wall_ms"], r["pre"]["wall_ms"]
            print(f"{bs:3d} {arm:>5} {str(r['full']['compiled_cap']):>4} |"
                  f"{fmt(r['e2e_s'], 7, 3)} {fmt(f_['range_pct'], 5, 2)} {f_['n']:2d} |"
                  f"{fmt(r['prefill_s'], 9, 3)} {fmt(p_['range_pct'], 5, 2)} {p_['n']:2d} |"
                  f"{fmt(100 * r['prefill_frac_of_e2e'] if r['prefill_frac_of_e2e'] else None, 6, 1)}%"
                  f" |{fmt(r['decode_tok_s'], 12)} |"
                  f"{fmt(r['full']['predicted_iterations']['median'], 9, 0)}")

    print()
    print("3. CAP EFFECT at 256/1024 (capped vs uncapped, same window)")
    print(f"{'bs':>3} | {'prefill':>9} | {'decode':>8} | {'e2e':>8}")
    for bs in BSS:
        u, c = M[f"bs{bs}_none"], M[f"bs{bs}_auto"]
        pf = ratio(u["prefill_s"], c["prefill_s"])
        dd = ratio(c["decode_tok_s"], u["decode_tok_s"])
        ee = ratio(u["e2e_s"], c["e2e_s"])
        print(f"{bs:3d} | {('%.3fx' % pf) if pf else '':>9} | "
              f"{('%+.1f%%' % (100 * (dd - 1))) if dd else '':>8} | "
              f"{('%+.1f%%' % (100 * (ee - 1))) if ee else '':>8}")

    print()
    print("=" * 100)
    print(f"4. AC-5 (mpk e2e / vLLM e2e, bound {AC5_BOUND}x) and AC-4 position (decode)")
    print("=" * 100)
    print(f"{'bs':>3} {'arm':>5} | {'mpk e2e s':>9} {'vllm e2e s':>10} "
          f"{'AC-5 ratio':>10} {'verdict':>8} | {'mpk dec':>8} {'vllm dec':>8} "
          f"{'AC-4 gap':>8}")
    ac5 = {}
    for bs in BSS:
        for arm in ARMS:
            r = M[f"bs{bs}_{arm}"]
            v = vllm[bs]
            g = ratio(r["e2e_s"], v["e2e_s"])
            gap = ratio(v["decode_tok_s"], r["decode_tok_s"])
            verdict = "" if g is None else ("PASS" if g <= AC5_BOUND else "FAIL")
            ac5[f"bs{bs}_{arm}"] = dict(ratio=g, verdict=verdict,
                                        vllm_e2e_s=v["e2e_s"],
                                        vllm_decode=v["decode_tok_s"],
                                        vllm_source=v["source"])
            print(f"{bs:3d} {arm:>5} |{fmt(r['e2e_s'], 9, 3)} {fmt(v['e2e_s'], 10, 3)} "
                  f"{fmt(g, 10, 3)} {verdict:>8} |{fmt(r['decode_tok_s'], 8)} "
                  f"{fmt(v['decode_tok_s'], 8)} {fmt(gap, 8, 2)}")
    print("\nvLLM sources: " + "; ".join(
        f"bs{bs}={vllm[bs]['source']} (n={vllm[bs]['n']})" for bs in BSS))

    # AC-5 is an END-TO-END bound, so our prefill deficit is a TAX ON THE DECODE
    # MARGIN AC-4 has to earn. Turn the bound into the decode throughput it
    # implies at the landed prefill cost, so "prefill sanity" is a number the
    # decode track can plan against instead of a vibe.
    print()
    print("=" * 112)
    print(f"5. AC-5 AS A DECODE REQUIREMENT (landed policy arm). Given prefill P "
          f"and the {AC5_BOUND}x bound on vLLM e2e V,")
    print("   the decode window may be at most 1.25V - P; the implied decode "
          "throughput is what AC-4 must reach for AC-5 to hold.")
    print("=" * 112)
    print(f"{'bs':>3} | {'P (mpk pre s)':>13} {'V (vllm e2e s)':>14} "
          f"{'vllm pre s*':>11} | {'max dec win s':>13} {'req dec tok/s':>13} "
          f"{'/ vllm dec':>10} | {'now':>8} {'need x':>7} | {'P_max@par':>9} "
          f"{'pre must':>8}")
    ac5req = {}
    for bs in BSS:
        # the landed policy arm for this bs, read from the policy authority
        landed = "auto" if admission_policy_on(bs) else "none"
        r = M[f"bs{bs}_{landed}"]
        v = vllm[bs]
        if not (r["prefill_s"] and r["e2e_s"]):
            continue
        D = r["full"]["decode_steps"] - r["pre"]["decode_steps"]
        vpre = v["e2e_s"] - (bs * D / v["decode_tok_s"]) if v["decode_tok_s"] else None
        budget = AC5_BOUND * v["e2e_s"] - r["prefill_s"]
        req = (bs * D / budget) if budget and budget > 0 else None
        ac5req[bs] = dict(landed_arm=landed, prefill_s=r["prefill_s"],
                          vllm_e2e_s=v["e2e_s"], vllm_prefill_s_implied=vpre,
                          decode_tokens=D, max_decode_window_s=budget,
                          required_decode_tok_s=req,
                          required_over_vllm=(req / v["decode_tok_s"] if req else None),
                          current_decode_tok_s=r["decode_tok_s"],
                          required_over_current=(req / r["decode_tok_s"]
                                                 if req and r["decode_tok_s"] else None))
        # the other side of the same joint constraint: with decode at exactly
        # vLLM's throughput, how large may prefill be, and by how much must it
        # fall from where it is now?
        par_win = bs * D / v["decode_tok_s"] if v["decode_tok_s"] else None
        p_max = (AC5_BOUND * v["e2e_s"] - par_win) if par_win else None
        must = (r["prefill_s"] / p_max) if p_max and p_max > 0 else None
        ac5req[bs]["prefill_budget_at_decode_parity_s"] = p_max
        ac5req[bs]["prefill_must_fall_by_x"] = must
        print(f"{bs:3d} |{fmt(r['prefill_s'], 13, 3)} {fmt(v['e2e_s'], 14, 3)} "
              f"{fmt(vpre, 11, 3)} |{fmt(budget, 13, 3)} {fmt(req, 13, 1)} "
              f"{fmt(req / v['decode_tok_s'] if req else None, 10, 3)} |"
              f"{fmt(r['decode_tok_s'], 8)} "
              f"{fmt(req / r['decode_tok_s'] if req and r['decode_tok_s'] else None, 7, 2)}"
              f" |{fmt(p_max, 9, 3)} "
              f"{('%.2fx' % must) if must and must > 1 else 'ok':>8}")
    print("  * vllm pre s is IMPLIED (its e2e minus its own decode window at its "
          "measured throughput), not separately measured.")
    print("  'req dec tok/s / vllm dec' > 1 means AC-5 cannot be met at decode "
          "PARITY: our prefill deficit alone")
    print("  spends more than the 25% slack, so AC-4 has to beat vLLM by that "
          "factor for AC-5 to pass.")

    print()
    print("5. EVERY REP (ms), in rep order -- the >=3-rep rule checkable inline")
    for bs in BSS:
        for arm in ARMS:
            u = A[f"bs{bs}_{arm}"]["wall_ms"]
            print(f"  A  bs{bs:<2} {arm:<4} " +
                  " ".join(f"{x:.1f}" for x in u["reps"]) +
                  f"   median={fmt(u['median'], 0, 1).strip()} n={u['n']}")
    for bs in BSS:
        for arm in ARMS:
            for cfg in ("full", "pre"):
                w = M[f"bs{bs}_{arm}"][cfg]["wall_ms"]
                print(f"  M{cfg[0].upper()} bs{bs:<2} {arm:<4} " +
                      " ".join(f"{x:.1f}" for x in w["reps"]) +
                      f"   median={fmt(w['median'], 0, 1).strip()} n={w['n']}")

    bad = [k for k, v in list(A.items())
           if not v["compiled_cap_consistent"]] + \
          [f"{k}.{c}" for k, v in M.items() for c in ("full", "pre")
           if not v[c]["compiled_cap_consistent"]]
    print()
    print("6. INTEGRITY -- compiled cap value constant within every arm: "
          + ("OK" if not bad else f"VIOLATED in {bad}"))

    doc = dict(schema="m4i4/tables/v1", reps_requested=a.reps,
               ac5_bound=AC5_BOUND, admission_policy=_policy.summary(),
               vllm=vllm, geomA=A, geomM=M, ac5=ac5, ac5_requirement=ac5req,
               integrity_cap_consistent=(not bad), integrity_violations=bad)
    if a.out_json:
        Path(a.out_json).write_text(json.dumps(doc, indent=1) + "\n")
        print(f"\nwrote {a.out_json}")
    if a.out_csv_prefix:
        def reps_cell(vals):
            return '"' + " ".join("%.1f" % x for x in vals) + '"'

        rows = ["bs,arm,compiled_cap,n_waves,total_wall_ms_median,range_pct,n,"
                "reps_ms,replay_iterations"]
        for v in A.values():
            w = v["wall_ms"]
            rows.append(f"{v['bs']},{v['arm']},{v['compiled_cap']},{v['n_waves']},"
                        f"{w['median']},{w['range_pct']},{w['n']},"
                        f"{reps_cell(w['reps'])},"
                        f"{v['predicted_iterations']['median']}")
        pa = Path(a.out_csv_prefix + "geomA.csv")
        pa.write_text("\n".join(rows) + "\n")

        rows = ["bs,arm,compiled_cap,e2e_s,e2e_range_pct,e2e_n,prefill_s,"
                "prefill_range_pct,prefill_n,decode_tok_s,vllm_e2e_s,ac5_ratio,"
                "vllm_decode,full_reps_ms,pre_reps_ms"]
        for key, v in M.items():
            fw, pw = v["full"]["wall_ms"], v["pre"]["wall_ms"]
            rows.append(f"{v['bs']},{v['arm']},{v['full']['compiled_cap']},"
                        f"{v['e2e_s']},{fw['range_pct']},{fw['n']},"
                        f"{v['prefill_s']},{pw['range_pct']},{pw['n']},"
                        f"{v['decode_tok_s']},{vllm[v['bs']]['e2e_s']},"
                        f"{ac5[key]['ratio']},{vllm[v['bs']]['decode_tok_s']},"
                        f"{reps_cell(fw['reps'])},{reps_cell(pw['reps'])}")
        pm = Path(a.out_csv_prefix + "geomM.csv")
        pm.write_text("\n".join(rows) + "\n")
        print(f"wrote {pa} {pm}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
