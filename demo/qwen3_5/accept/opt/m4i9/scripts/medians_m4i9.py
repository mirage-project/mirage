#!/usr/bin/env python3
"""M4-I9 -- per-rep e2e table + per-bs medians for the fusion A/B.

Reads the sweep's own meta JSONs (one per (arm, bs, rep)) and prints every rep's
value, not just the median, because the per-rep RANGE is what says whether a
median difference means anything: M4-I8's arm S moved medians by 0.0-1.0 ms
against per-rep ranges of 1.1-75.0 ms and was correctly called an exact null.

Also emits the same-window control: each rep's own `gpu_before` audit line, so a
cell whose device was not clean is visible rather than averaged in.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def load(root: Path, arm: str, bs: int, rep: int):
    f = root / f"out_{arm}" / f"meta_bs{bs}_rep{rep}_{arm}.json"
    if not f.exists():
        return None
    d = json.load(open(f))
    return d


def val(d):
    """Wall ms of the single decode wave (the harness's own number)."""
    w = d.get("waves") or []
    if w and isinstance(w[0], dict) and isinstance(w[0].get("wall_ms"), (int, float)):
        return float(w[0]["wall_ms"])
    return None


def audit(d):
    """Per-rep device audit derived from THIS run's OWN log line.

    `cuda_visible_devices` names the pinned device and `gpu_before` /
    `gpu_after` are that run's own nvidia-smi snapshots, so the check is
    per-cell rather than a single claim for the whole sweep: the PINNED device
    must have been below the observed idle floor before the run.
    """
    dev = str(d.get("cuda_visible_devices", "")).strip()
    def used(rows, idx):
        for line in rows or []:
            p = [x.strip() for x in str(line).split(",")]
            if p and p[0] == idx:
                try:
                    return int(p[1].split()[0]), int(p[2].split()[0])
                except (ValueError, IndexError):
                    return None, None
        return None, None
    ub, tb = used(d.get("gpu_before"), dev)
    ua, ta = used(d.get("gpu_after"), dev)
    return dict(device=dev, before_mib=ub, before_util=tb,
                after_mib=ua, after_util=ta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--arms", default="A,F")
    ap.add_argument("--bs", default="1,2,4,8,16")
    ap.add_argument("--reps", default="0,1,2")
    ap.add_argument("--out")
    ap.add_argument("--steps", default=96,
                    help="decode steps per run, to convert ms -> us/step")
    ap.add_argument("--floor-mib", default=120,
                    help="observed pinned-device idle floor for this box")
    a = ap.parse_args()
    root = Path(a.root)
    arms = a.arms.split(",")
    bss = [int(x) for x in a.bs.split(",")]
    reps = [int(x) for x in a.reps.split(",")]

    rows = []
    missing = []
    audits = []
    for bs in bss:
        row = {"bs": bs}
        for arm in arms:
            vs = []
            for rep in reps:
                d = load(root, arm, bs, rep)
                if d is None:
                    missing.append(f"{arm}_bs{bs}_rep{rep}")
                    continue
                v = val(d)
                if v is not None:
                    vs.append(v)
                au = audit(d)
                au.update(arm=arm, bs=bs, rep=rep,
                          tokens_sha256=d.get("tokens_sha256"))
                audits.append(au)
            row[arm] = vs
            row[arm + "_med"] = statistics.median(vs) if vs else None
        rows.append(row)

    base = arms[0]
    print(f"{'bs':>3s} | {base} per-rep (ms)                | med      | "
          + " | ".join(f"{x} per-rep (ms)                | med      | {x}/{base}"
                       for x in arms[1:]))
    for r in rows:
        cells = [f"{r['bs']:3d}",
                 " / ".join(f"{v:.1f}" for v in r[base]).ljust(28),
                 f"{r[base + '_med']:.1f}".ljust(8) if r[base + "_med"] else "-".ljust(8)]
        for x in arms[1:]:
            cells.append(" / ".join(f"{v:.1f}" for v in r[x]).ljust(28))
            cells.append(f"{r[x + '_med']:.1f}".ljust(8) if r[x + "_med"] else "-".ljust(8))
            if r[base + "_med"] and r[x + "_med"]:
                cells.append(f"{r[base + '_med'] / r[x + '_med']:.4f}x")
            else:
                cells.append("-")
        print(" | ".join(cells))

    if missing:
        print(f"\nMISSING cells ({len(missing)}): {', '.join(missing[:20])}")

    # ---- PAIRED per-rep deltas ------------------------------------------
    # The arms are interleaved per (bs, rep) with the same seed inside one GPU
    # claim, so the rep is a BLOCK and the paired difference is the right
    # statistic: it cancels the drift that makes the unpaired per-rep range
    # (up to 112 ms at bs4) swamp a ~20 ms effect.
    paired = {}
    print(f"\n{'bs':>3s} | paired delta (base - arm), ms, per rep      | mean  "
          f"| us/step | sign")
    for r in rows:
        bs = r["bs"]
        for x in arms[1:]:
            b, o = r[base], r[x]
            if len(b) != len(o) or not b:
                continue
            ds = [bb - oo for bb, oo in zip(b, o)]
            mean = sum(ds) / len(ds)
            nwin = sum(1 for d in ds if d > 0)
            paired[(bs, x)] = dict(deltas=ds, mean_ms=mean, arm_wins=nwin,
                                   n=len(ds))
            per_step = mean * 1000.0 / int(a.steps)
            print(f"{bs:3d} | " + " / ".join(f"{d:+7.1f}" for d in ds).ljust(42)
                  + f" | {mean:+6.1f}| {per_step:+7.1f} | "
                  f"{nwin}/{len(ds)} reps favour {x}")
    tw = sum(v["arm_wins"] for v in paired.values())
    tn = sum(v["n"] for v in paired.values())
    print(f"    paired reps favouring the fused arm: {tw}/{tn}")

    # ---- per-rep device audit, derived from each run's OWN log line -------
    FLOOR = int(a.floor_mib)
    devs = sorted({x["device"] for x in audits})
    dirty = [x for x in audits
             if x["before_mib"] is None or x["before_mib"] > FLOOR
             or (x["before_util"] or 0) > 5]
    print(f"\naudit: {len(audits)} runs, devices used {devs}, "
          f"pinned-device floor {FLOOR} MiB")
    print(f"  dirty (before_mib > floor or before_util > 5%): {len(dirty)}")
    for x in dirty[:12]:
        print(f"    {x['arm']}_bs{x['bs']}_rep{x['rep']}: dev{x['device']} "
              f"before={x['before_mib']}MiB/{x['before_util']}%")
    unaud = [x for x in audits if x["before_mib"] is None]
    print(f"  unauditable (no own log line for the pinned device): {len(unaud)}")

    # ---- token identity, arm vs base, at the same (bs, rep) ---------------
    by = {(x["arm"], x["bs"], x["rep"]): x["tokens_sha256"] for x in audits}
    same = diff = 0
    for bs in bss:
        for rep in reps:
            b = by.get((base, bs, rep))
            for x in arms[1:]:
                o = by.get((x, bs, rep))
                if b and o:
                    if b == o:
                        same += 1
                    else:
                        diff += 1
                        print(f"  TOKENS DIFFER {x} vs {base} bs{bs} rep{rep}")
    print(f"  tokens_sha256 identical: {same} pairs, differing: {diff}")

    if a.out:
        Path(a.out).write_text(json.dumps(
            dict(root=str(root), arms=arms, bs=bss, reps=reps, rows=rows,
                 missing=missing, audits=audits,
                 n_dirty=len(dirty), n_unauditable=len(unaud),
                 tokens_identical_pairs=same,
                 tokens_differing_pairs=diff,
                 paired={f"bs{k[0]}_{k[1]}": v for k, v in paired.items()}),
            indent=1))


if __name__ == "__main__":
    main()
