#!/usr/bin/env python3
"""M3-I7 -- every arm's PER-REP values, and the corrected 256/1024 table.

Two jobs, both in service of the same rule.

1. The pinned statistical rule is warmup + >=3 reps, median and dispersion. The
   first pass met it everywhere except one place: the drain gate discarded rep0
   of the prefill-only arm at bs1/bs4/bs16, and since the prefill median is a
   term in the decode slope, three of the five binding rows rested on n=2.
   Phase perfM2 re-captured those three batch sizes -- BOTH arms, interleaved,
   in one clean window (foreign floor 5 MiB) so the subtraction stays a
   same-window control and the already-clean full arm doubles as a cross-window
   check. This prints the corrected table beside the superseded one.

2. It prints every rep of every arm, so the >=3 rule can be verified arm by arm
   without opening a raw artifact.

Window selection is explicit, never implicit: `M2` wins at the batch sizes it
covers, `M` supplies the rest, and each row says which window it came from.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
from pathlib import Path

BSL = [1, 2, 4, 8, 16]
M2_BS = {1, 4, 16}
VLLM_DECODE = {1: 285.5, 2: 529.8, 4: 934.4, 8: 1692.5, 16: 3018.1}
VLLM_E2E_S = {1: 3.60, 2: 3.89, 4: 4.45, 8: 4.953, 16: 5.568}


def tag_gpu(root):
    """tag -> the device that phase actually pinned.

    The audit file records ALL eight devices per run, so picking a device by
    hand (or taking the last matching line) can silently report a co-tenant's
    memory instead of the pinned one -- which is the difference between "this
    rep started dirty" and "some other device did". Each phase log names its own
    device once (`GATE gpu=N`) and contains every tag that phase ran, so the
    mapping is recoverable exactly.
    """
    import re
    out = {}
    for log in sorted((Path(root) / "logs").glob("run_*.log")):
        txt = log.read_text(errors="replace")
        m = re.search(r"GATE gpu=(\d+)", txt)
        if not m:
            continue
        g = int(m.group(1))
        for t in set(re.findall(r"^\s*\[(\S+)\] rc=", txt, re.M)):
            out[t] = g
    return out


def audit(root, gpus, limit):
    """tag -> MiB resident on the PINNED device just before that run started."""
    import re
    tg = tag_gpu(root)
    out = {}
    p = Path(root) / "audit" / "gpu_before.txt"
    if not p.exists():
        return out
    for line in p.read_text().splitlines():
        m = re.match(r"^(\S+)\s+(\d+),\s*(\d+)\s*MiB,\s*(\d+)\s*%", line.strip())
        if not m:
            continue
        tag, dev, mib = m.group(1), int(m.group(2)), int(m.group(3))
        if tag in tg and dev == tg[tag]:
            out[tag] = mib
    return out


def arm(root, sub, tag, bs, aud, limit, reps=3):
    """Per-rep wall_ms for one arm, each rep labelled clean or dirty."""
    rows = []
    for r in range(reps):
        p = f"{root}/{sub}/timings_bs{bs}_rep{r}.json"
        try:
            w = json.load(open(p))["waves"][0]
        except Exception:
            continue
        mib = aud.get(f"{tag}_bs{bs}_rep{r}")
        rows.append(dict(rep=r, wall_ms=w["wall_ms"], D=w["max_decode_steps"],
                         gpu_before_mib=mib,
                         dirty=(mib is not None and mib > limit)))
    return rows


def stats(rows):
    v = [r["wall_ms"] for r in rows if not r["dirty"]]
    if not v:
        return None
    m = st.median(v)
    return dict(median=m, n=len(v), lo=min(v), hi=max(v),
                range_pct=100.0 * (max(v) - min(v)) / m,
                D=st.median([r["D"] for r in rows if not r["dirty"]]))


def slope(full, pre, bs):
    if not (full and pre):
        return None
    return bs * (full["D"] - pre["D"]) / ((full["median"] - pre["median"]) / 1000.0)


def reps_str(rows):
    return " ".join(f"{r['wall_ms']:.1f}" + ("*" if r["dirty"] else "") for r in rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.path.expanduser("~/mpk-qwen35/i7"))
    ap.add_argument("--gpus", default="2,5,6,7")
    ap.add_argument("--limit", type=int, default=1032,
                    help="dirty threshold used by the first window's audit")
    ap.add_argument("--limit-m2", type=int, default=405,
                    help="dirty threshold for the perfM2 window (floor 5 MiB + 400)")
    ap.add_argument("--out")
    ap.add_argument("--csv-dir", help="also emit the canonical geometry-M CSVs here")
    a = ap.parse_args(argv)
    R = a.root
    gpus = {int(x) for x in a.gpus.split(",")}
    aud = audit(R, gpus, a.limit)

    rec = {}
    print("=" * 116)
    print("GEOMETRY M -- 256/1024, PER-REP wall_ms (a trailing * = discarded, device dirty "
          "at run start)")
    print("=" * 116)
    print(f"{'bs':>3} {'window':>6} {'arm':>5} | {'per-rep wall_ms':<34} | "
          f"{'median':>9} {'n':>2} {'range%':>7} | {'gpu_before MiB':>26}")
    for bs in BSL:
        for win, sub_f, sub_p, tf, tp, lim in (
                ("M", "perf/M/full", "perf/M/pre", "pM_full", "pM_pre", a.limit),
                ("M2", "perf/M2/full", "perf/M2/pre", "p2_full", "p2_pre", a.limit_m2)):
            for label, sub, tg in (("full", sub_f, tf), ("pre", sub_p, tp)):
                rows = arm(R, sub, tg, bs, aud, lim)
                if not rows:
                    continue
                s = stats(rows)
                mibs = ",".join(str(r["gpu_before_mib"]) for r in rows)
                print(f"{bs:3d} {win:>6} {label:>5} | {reps_str(rows):<34} | "
                      f"{s['median']:9.1f} {s['n']:2d} {s['range_pct']:7.2f} | {mibs:>26}")
                rec.setdefault(f"bs{bs}", {}).setdefault(win, {})[label] = dict(
                    per_rep=rows, **s)

    print()
    print("=" * 116)
    print("CORRECTED 256/1024 TABLE  (bs1/bs4/bs16 from the perfM2 re-capture, n=3 both "
          "arms; bs2/bs8 unchanged from perfM)")
    print("=" * 116)
    print(f"{'bs':>3} {'win':>4} | {'full ms':>9} {'n':>2} {'r%':>6} | {'pre ms':>9} {'n':>2} "
          f"{'r%':>6} | {'decode tok/s':>12} {'vLLM':>8} {'gap':>6} | {'e2e s':>7} {'gap':>6}")
    final = {}
    for bs in BSL:
        win = "M2" if bs in M2_BS and rec.get(f"bs{bs}", {}).get("M2") else "M"
        d = rec[f"bs{bs}"][win]
        f_, p_ = d["full"], d["pre"]
        dec = slope(f_, p_, bs)
        e2e = f_["median"] / 1000.0
        print(f"{bs:3d} {win:>4} | {f_['median']:9.1f} {f_['n']:2d} {f_['range_pct']:6.2f} | "
              f"{p_['median']:9.1f} {p_['n']:2d} {p_['range_pct']:6.2f} | {dec:12.1f} "
              f"{VLLM_DECODE[bs]:8.1f} {VLLM_DECODE[bs]/dec:6.2f} | {e2e:7.2f} "
              f"{e2e/VLLM_E2E_S[bs]:6.2f}")
        final[bs] = dict(window=win, full_ms=f_["median"], full_n=f_["n"],
                         full_range_pct=f_["range_pct"], pre_ms=p_["median"],
                         pre_n=p_["n"], pre_range_pct=p_["range_pct"],
                         decode_tok_s=dec, vllm_decode_tok_s=VLLM_DECODE[bs],
                         gap_decode_x=VLLM_DECODE[bs] / dec, e2e_s=e2e,
                         vllm_e2e_s=VLLM_E2E_S[bs], gap_e2e_x=e2e / VLLM_E2E_S[bs])

    print()
    print("SUPERSEDED (kept for audit): the n=2-prefill values the first pass reported")
    print(f"{'bs':>3} | {'pre ms n=2':>11} -> {'pre ms n=3':>11} | "
          f"{'decode tok/s':>12} -> {'decode tok/s':>12} | {'gap':>6} -> {'gap':>6} | "
          f"{'shift':>7}")
    superseded = {}
    for bs in sorted(M2_BS):
        old = rec.get(f"bs{bs}", {}).get("M")
        new = rec.get(f"bs{bs}", {}).get("M2")
        if not (old and new):
            continue
        od, nd = slope(old["full"], old["pre"], bs), slope(new["full"], new["pre"], bs)
        print(f"{bs:3d} | {old['pre']['median']:11.1f} -> {new['pre']['median']:11.1f} | "
              f"{od:12.1f} -> {nd:12.1f} | {VLLM_DECODE[bs]/od:6.2f} -> "
              f"{VLLM_DECODE[bs]/nd:6.2f} | {100*(nd/od-1):+6.2f}%")
        superseded[bs] = dict(old_pre_ms=old["pre"]["median"], old_pre_n=old["pre"]["n"],
                              old_decode_tok_s=od, old_gap_x=VLLM_DECODE[bs] / od,
                              new_decode_tok_s=nd, shift_pct=100 * (nd / od - 1))

    print()
    print("CROSS-WINDOW CONTROL: the full-run arm was already clean at n=3 in both windows, "
          "so its\nagreement is independent evidence the two windows are comparable.")
    for bs in sorted(M2_BS):
        old, new = rec[f"bs{bs}"]["M"]["full"], rec[f"bs{bs}"]["M2"]["full"]
        print(f"  bs{bs:<3d} full: {old['median']:9.1f} (M) vs {new['median']:9.1f} (M2) = "
              f"{100*(new['median']/old['median']-1):+.2f}%")

    if a.csv_dir:
        import csv as _csv
        cd = Path(a.csv_dir); cd.mkdir(parents=True, exist_ok=True)
        with open(cd / "geomM_matched_256_1024.csv", "w", newline="") as fh:
            w = _csv.DictWriter(fh, fieldnames=["bs"] + list(next(iter(final.values())).keys()))
            w.writeheader()
            for bs, v in sorted(final.items()):
                w.writerow(dict(bs=bs, **v))
        with open(cd / "geomM_per_rep.csv", "w", newline="") as fh:
            w = _csv.writer(fh)
            w.writerow(["bs", "window", "arm", "rep", "wall_ms", "decode_steps",
                        "gpu_before_mib_pinned_device", "discarded"])
            for k, wins in sorted(rec.items(), key=lambda x: int(x[0][2:])):
                for win, arms in wins.items():
                    for label, d in arms.items():
                        for r in d["per_rep"]:
                            w.writerow([k[2:], win, label, r["rep"], f"{r['wall_ms']:.2f}",
                                        r["D"], r["gpu_before_mib"], r["dirty"]])
        print(f"wrote {cd / 'geomM_matched_256_1024.csv'} and {cd / 'geomM_per_rep.csv'}")
    if a.out:
        Path(a.out).write_text(json.dumps(
            dict(per_rep=rec, final=final, superseded=superseded,
                 dirty_limit_M=a.limit, dirty_limit_M2=a.limit_m2), indent=1,
            default=str) + "\n")
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
